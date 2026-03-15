# 第 3 章：KV Cache 传统问题与 PagedAttention 设计

> "Those who cannot remember the past are condemned to repeat it." —— George Santayana。对 Transformer 而言，KV Cache 就是它的"记忆"——而如何高效管理这份记忆，决定了系统的上限。

## 3.1 什么是 KV Cache

在 Transformer 的自回归解码过程中，每生成一个新 token，模型都需要对所有之前的 token 执行 attention 计算。具体来说，Self-Attention 层会为每个 token 计算出三个向量：Query（Q）、Key（K）和 Value（V）。新 token 的 Q 需要与所有历史 token 的 K 做点积来计算注意力权重，再以这些权重对所有历史 token 的 V 做加权求和。

如果每次生成都重新计算所有历史 token 的 K 和 V，计算量将随序列长度呈二次方增长。因此，工程实践中的标准做法是缓存已计算过的 K 和 V 向量，这就是 **KV Cache**。第 $t$ 步解码时，只需计算新 token 的 Q、K、V，然后将新的 K、V 追加到缓存中，再用完整的缓存参与 attention 计算。

KV Cache 的内存占用十分可观。以一个 13B 参数的模型为例，40 层 Transformer、hidden size 为 5120、attention head 数为 40，使用 FP16 存储，单个 token 的 KV Cache 大小约为：

$$2 \times 40 \times 5120 \times 2 = 800 \text{ KB}$$

对于一个最大序列长度为 2048 的请求，KV Cache 最多需要约 1.6 GB。当系统同时服务数十个请求时，KV Cache 轻易占满整块 GPU 显存。

## 3.2 传统 KV Cache 管理的三大问题

### 预分配浪费

传统框架（包括早期的 HuggingFace Transformers 和 FasterTransformer）的做法是：当一个请求到来时，立即按照 `max_seq_len` 为其分配一块连续的 KV Cache 内存。然而，大多数请求的实际输出长度远小于最大值。如果最大长度为 2048 而实际只生成了 200 个 token，则有 **90% 的预分配内存被白白浪费**。

### 内部碎片（Internal Fragmentation）

即使在请求生命周期内，已分配的连续内存块中也存在大量空洞。在解码的中间阶段，序列从 prompt 长度逐渐增长到最终长度，预分配的后半段空间始终处于"已占用但未使用"的状态。

### 外部碎片（External Fragmentation）

当多个请求先后到达并离开系统时，释放的内存块大小不一、位置分散。虽然总的空闲内存可能足够服务新请求，但由于缺少足够大的**连续**空闲块而无法分配。这种碎片问题在长时间运行的服务中尤为严重。

### 无法共享

在 beam search 等生成策略中，多个候选序列共享相同的 prompt 前缀。传统方案下，每个候选序列都独立持有一份完整的 KV Cache 副本，prompt 部分被重复存储多次，造成巨大浪费。

## 3.3 PagedAttention 的设计哲学

vLLM 的核心创新在于将操作系统虚拟内存管理的思想引入 KV Cache。这一类比极为精妙：

| 操作系统概念 | PagedAttention 对应 |
|------------|-------------------|
| 虚拟页（Virtual Page） | 逻辑块（Logical Block） |
| 物理页框（Physical Frame） | 物理块（Physical Block） |
| 页表（Page Table） | Block Table |
| 按需分页（Demand Paging） | 按需分配物理块 |
| 写时复制（Copy-on-Write） | KV Cache 共享与 CoW |

### 逻辑块与物理块

每个序列的 KV Cache 在逻辑上被划分为一系列固定大小的 **逻辑块（Logical Block）**。每个块可以存储 `block_size` 个 token 的 KV 向量，`block_size` 通常为 16。逻辑块的编号从 0 开始递增，代表序列中 token 的位置分组。

**物理块（Physical Block）** 则是 GPU 显存中实际分配的内存单元。关键在于：物理块不需要连续排列。逻辑块 0、1、2 对应的物理块可能分别位于 GPU 显存的完全不同的位置。

### Block Table

**Block Table** 是连接逻辑世界和物理世界的桥梁。它为每个序列维护一个映射数组，将逻辑块编号映射到物理块编号。当 attention 内核需要读取某个 token 的 KV Cache 时，先根据 token 位置计算出逻辑块编号和块内偏移，再通过 block table 查到物理块的实际地址。

在 vLLM v1 的代码中，block table 的管理分布在多个层次：`vllm/v1/core/kv_cache_manager.py` 负责逻辑层面的块分配决策，`vllm/v1/worker/block_table.py` 在 worker 端维护供 CUDA 内核使用的物理 block table 张量。

### 按需分配

与传统方案一次性预分配不同，PagedAttention 仅在当前块被填满时才分配新的物理块。一个正在解码的序列，初始只需为 prompt 分配 $\lceil \text{prompt\_len} / \text{block\_size} \rceil$ 个物理块，之后每当最后一个块写满，再追加一个新块。这将内存浪费压缩到**不超过一个块**（即最后一个块的未用部分）。

### Copy-on-Write 共享

在 beam search 场景下，多个候选序列的 prompt 部分完全相同。PagedAttention 让这些序列的 block table 指向相同的物理块，通过引用计数跟踪共享关系。当某个序列需要修改共享块的内容时（例如 beam 分裂后不同路径产生不同的 token），才执行 copy-on-write：复制一份物理块给该序列独享。这一机制在 `vllm/v1/core/block_pool.py` 的引用计数逻辑中实现。

## 3.4 量化收益

PagedAttention 带来的内存效率提升可以量化。假设系统同时服务 N 个请求，传统方案的内存占用为 $N \times \text{max\_seq\_len} \times \text{per\_token\_kv\_size}$，而 PagedAttention 的实际占用约为 $\sum_{i=1}^{N} \lceil \text{actual\_len}_i / \text{block\_size} \rceil \times \text{block\_size} \times \text{per\_token\_kv\_size}$。论文数据显示，实际浪费率降低到 4% 以内，相比传统方案 60%~80% 的浪费率，这意味着同等显存下可以服务 **2~4 倍** 的并发请求。

## 本章小结

KV Cache 是 LLM 推理中内存消耗的核心瓶颈。传统的连续内存预分配方案存在预分配浪费、内部碎片、外部碎片和无法共享四大问题。vLLM 的 PagedAttention 借鉴操作系统虚拟内存的分页机制，通过逻辑块、物理块和 block table 的三层抽象，实现了按需分配和 copy-on-write 共享，将内存浪费率从 60%~80% 降至 4% 以内。这一设计是 vLLM 高吞吐量的根本保障，其实现细节分布在 `vllm/v1/core/` 下的 KV Cache 管理模块和 `vllm/v1/worker/block_table.py` 中。
