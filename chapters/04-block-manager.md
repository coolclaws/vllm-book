# 第 4 章：BlockManager 实现

> "Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away." —— Antoine de Saint-Exupery。Block Manager 的设计正体现了这种极简之美——用最少的抽象，管理最复杂的内存生命周期。

## 4.1 Block Manager 的角色定位

在上一章中，我们理解了 PagedAttention 的设计理念。本章将深入它的工程实现——Block Manager。Block Manager 是 vLLM 内存管理子系统的核心组件，负责物理块的分配、回收、共享和换入换出。它是连接调度器（Scheduler）与执行层（Worker）的关键纽带：调度器决定"做什么"，Block Manager 决定"用哪块内存来做"。

在 vLLM v1 架构中，这一职责主要由以下文件承担：

- `vllm/v1/core/kv_cache_manager.py`：KV Cache 管理器的主逻辑
- `vllm/v1/core/block_pool.py`：物理块池，管理块的分配与释放
- `vllm/v1/core/single_type_kv_cache_manager.py`：单类型 KV Cache 的专用管理器
- `vllm/v1/core/kv_cache_coordinator.py`：多层 KV Cache 的协调器

## 4.2 BlockPool：物理块的生命周期

`block_pool.py` 中的 `BlockPool` 类是最底层的内存管理单元。它维护着一个物理块的池子，每个块由一个整数 ID 标识，对应 GPU 显存中一段固定大小的区域。

### 块的状态流转

物理块在其生命周期中经历以下状态：

```
Free（空闲）→ Allocated（已分配）→ Shared（共享/CoW）→ Free（释放回池）
```

当调度器决定接纳一个新请求时，`BlockPool` 从空闲列表中取出所需数量的物理块。每个块维护一个**引用计数（ref_count）**：初始为 1，被多个序列共享时递增，某个序列释放时递减，引用计数归零时块回到空闲列表。

下图展示物理块的完整生命周期状态机：

```text
// 物理块生命周期状态机

                  系统初始化
                      │
                      ▼
               ┌────────────┐
               │    Free     │◀──────────────────────────┐
               │   (空闲)    │                            │
               └─────┬──────┘                            │
          allocate() │ ref_count=1                       │ free()
                     ▼                                   │ ref_count归零
               ┌────────────┐         free()             │
               │ Allocated  │────────────────────────────┘
               │ (已分配)   │  ref_count归零
               └─────┬──────┘
            fork()   │ ref_count+1
                     ▼
               ┌────────────┐
               │   Shared   │──▶ free() ref_count-1 ──▶ 回到 Allocated
               │   (共享)   │──▶ fork() ref_count+1 ──▶ 保持 Shared
               └────────────┘──▶ free() ref_count归零 ─▶ 回到 Free
```

### 核心接口

`BlockPool` 提供的关键操作包括：

- **`allocate()`**：从空闲池中获取一个物理块，引用计数设为 1
- **`free()`**：递减块的引用计数，归零时回收
- **`fork()`**：为一个已有块增加引用计数，实现共享（用于 beam search 的序列分裂）
- **`get_num_free_blocks()`**：查询剩余可用块数量，供调度器决策

### Watermark 机制

为了防止 GPU 显存耗尽导致 OOM（Out of Memory），`BlockPool` 实现了**水位线（Watermark）机制**。系统会预留一定比例（如 1%~4%）的物理块作为安全缓冲区。当空闲块数量低于水位线时，调度器将停止接纳新请求，转而等待现有请求完成释放资源。这一机制确保了系统在高负载下的稳定性。

## 4.3 KVCacheManager：逻辑层的编排

`kv_cache_manager.py` 中的 `KVCacheManager` 工作在更高的抽象层次。它不直接操作物理块，而是管理序列与块之间的映射关系。

### 序列的块管理

每个活跃序列在 `KVCacheManager` 中都有一条记录，包含：

- **逻辑块到物理块的映射**（block table 的逻辑视图）
- **当前已使用的 token 数量**
- **最后一个块的填充状态**

当序列增长时（decode 阶段每步生成一个 token），`KVCacheManager` 检查最后一个块是否还有空间。如果有，直接在块内追加；如果满了，向 `BlockPool` 申请新的物理块并追加到 block table 中。

### 资源预估

调度器在决定是否接纳新请求之前，需要知道该请求将消耗多少物理块。`KVCacheManager` 提供了资源预估能力：给定一个 prompt 长度和预期最大输出长度，计算所需的物理块数量，并与当前空闲块数量比较。这一信息对调度器的准入控制至关重要。

## 4.4 Prefix Caching：内容寻址的块共享

Prefix Caching 是 vLLM 的一项重要优化。在实际服务场景中，许多请求共享相同的系统提示词（system prompt）。如果每个请求都独立计算并存储 system prompt 的 KV Cache，将造成大量重复计算和内存浪费。

### 原理

Prefix Caching 的核心思想是：**如果两个块存储的 token 序列完全相同，它们的 KV Cache 也必然相同**。因此，可以用 token 序列的哈希值作为块的标识符，实现内容寻址（content-addressable）的块共享。

具体流程如下：

1. 当一个块被填满时，计算该块中 token 序列的哈希值
2. 在全局的哈希表中查找是否已存在相同哈希值的物理块
3. 如果存在，直接复用该物理块（增加引用计数），无需重新计算
4. 如果不存在，将当前块注册到哈希表中，供后续请求复用

```text
// Prefix Cache 命中 vs 未命中 处理路径

         新块填满，计算哈希值
                  │
                  ▼
         ┌──────────────────┐
         │  哈希表中是否存在？│
         └───┬──────────┬───┘
          命中│          │未命中
             ▼          ▼
   ┌──────────────┐  ┌──────────────────┐
   │ 复用已有物理块│  │ 分配新物理块      │
   │ 引用计数 +1  │  │ 计算 KV Cache     │
   └──────┬───────┘  └────────┬─────────┘
          ▼                   ▼
   ┌──────────────┐  ┌──────────────────┐
   │ 跳过 KV 计算 │  │ 将 哈希→块ID     │
   │ 直接使用缓存 │  │ 注册到哈希表      │
   └──────────────┘  └──────────────────┘
```

### 实现细节

在 v1 架构中，prefix caching 的逻辑集成在 `kv_cache_manager.py` 和 `block_pool.py` 中。`BlockPool` 维护了一个哈希到物理块 ID 的映射表。当块被释放时，如果开启了 prefix caching，块不会立即回到空闲列表，而是进入一个"缓存态"——保留内容但标记为可驱逐（evictable）。当空闲块不足时，系统按 LRU（Least Recently Used）策略驱逐缓存态的块。

这种设计使得高频出现的 prompt 前缀（如 ChatGPT 的 system prompt）的 KV Cache 几乎永远驻留在 GPU 显存中，大幅降低了首次 token 的延迟（Time to First Token, TTFT）。

## 4.5 Swap 机制：GPU-CPU 块迁移

当 GPU 显存压力过大时，vLLM 可以将部分序列的 KV Cache 从 GPU 换出（swap out）到 CPU 内存，等资源充裕时再换入（swap in）。Block Manager 在这一过程中负责：

1. **选择换出对象**：通常由调度器选择优先级最低或等待时间最长的序列
2. **生成迁移列表**：记录需要从 GPU 物理块到 CPU 物理块的映射关系
3. **更新 block table**：将相关序列的 block table 从 GPU 块 ID 替换为 CPU 块 ID
4. **触发异步拷贝**：Worker 层根据迁移列表执行 GPU-CPU 之间的数据传输

Swap 机制确保了系统在显存紧张时不会拒绝服务，而是通过"牺牲延迟换取吞吐"的策略优雅降级。

## 4.6 Block Table 的物理表示

在 worker 端（`vllm/v1/worker/block_table.py`），block table 被表示为一个二维 `torch.Tensor`，形状为 `[max_num_seqs, max_num_blocks_per_seq]`。这个张量直接传递给 CUDA attention 内核。内核通过序列 ID 和逻辑块索引查表，获取物理块 ID，再据此计算出 KV Cache 在 GPU 显存中的实际地址。

这种紧凑的张量表示避免了在 CUDA 内核中访问复杂的 Python 数据结构，确保了内存访问的高效性。

## 本章常见问题

**Q：BlockPool 和 KVCacheManager 的职责边界是什么？**

BlockPool 是底层的物理块池管理器，只关心块的分配、释放和引用计数，不关心块属于哪个序列。KVCacheManager 是更高层的抽象，负责维护"序列→逻辑块→物理块"的映射关系。类比操作系统：BlockPool 像物理内存管理器，KVCacheManager 像进程的虚拟内存管理器。

---

**Q：Prefix Caching 的哈希冲突如何处理？**

vLLM 使用 token 序列内容计算哈希值，并在命中时会验证 token 内容完全匹配。实践中哈希空间足够大，冲突概率极低。即使发生冲突，验证步骤也能保证不会使用错误的缓存块。

---

**Q：Watermark 水位线设太高或太低会怎样？**

水位线太高（预留太多块）会限制并发请求数，降低吞吐量；水位线太低则在突发负载时可能触发 OOM 或大量抢占。默认的 1%~4% 是在稳定性和利用率之间的折衷值，实际可根据负载模式微调。

---

**Q：Prefix Caching 中"缓存态"的块什么时候会被驱逐？**

当空闲块不足以满足新请求的分配需求时，系统会按 LRU（最近最少使用）策略驱逐缓存态的块。高频使用的前缀（如系统提示词）会因不断被访问而保持在缓存中，低频前缀则优先被驱逐。

---

## 本章小结

Block Manager 是 vLLM 内存管理的工程实现核心。`BlockPool`（`vllm/v1/core/block_pool.py`）管理物理块的分配、引用计数和回收；`KVCacheManager`（`vllm/v1/core/kv_cache_manager.py`）维护序列到物理块的映射关系。Prefix Caching 通过哈希寻址实现跨请求的块共享，大幅降低重复 prompt 的计算开销。Watermark 机制和 Swap 机制分别从预防和应对两个角度保障了系统在高负载下的稳定性。block table 最终以紧凑的张量形式传递给 CUDA 内核，完成从调度决策到硬件执行的最后一公里。
