# 第 8 章：Preemption 与 Swap

> "资源有限是工程的永恒主题。当 GPU 显存不足以容纳所有请求时，系统必须做出艰难的选择——暂时牺牲一些，才能让整体继续运转。"

## 8.1 为什么需要抢占

LLM 推理的核心资源瓶颈是 GPU 显存。每个正在运行的请求都需要维护 KV Cache，随着生成的 token 越来越多，KV Cache 占用的显存也持续增长。当多个请求同时运行且生成长度各不相同时，系统很容易遇到显存耗尽的情况。

这时有两个选择：要么拒绝新请求（降低吞吐量），要么暂停一些已有请求（释放资源给其他请求）。vLLM 选择了后者——**抢占（Preemption）**机制，允许系统在资源紧张时暂停低优先级请求，待资源充裕时再恢复执行。

## 8.2 抢占的触发条件

抢占发生在 Scheduler 的 `schedule()` 方法中（`vllm/v1/core/sched/scheduler.py`）。具体时机是：当 Scheduler 试图为一个 RUNNING 状态的请求分配新的 KV Cache 块时，`KVCacheManager`（`vllm/v1/core/kv_cache_manager.py`）的 `allocate_slots()` 返回 `None`，表示 `BlockPool`（`vllm/v1/core/block_pool.py`）中已经没有足够的空闲物理块。

此时 Scheduler 进入抢占循环：

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(request, ...)
    if new_blocks is not None:
        break  # 分配成功，退出循环
    # 选择一个受害者进行抢占
    victim = self._select_victim()
    self._preempt_request(victim, timestamp)
```

Scheduler 会反复抢占受害者，直到为当前请求成功分配到资源为止。

## 8.3 受害者选择策略

选择哪个请求被抢占直接影响系统的公平性和效率。vLLM 根据调度策略采用不同的选择逻辑：

**FCFS（先来先服务）策略**：抢占 running 列表中最后的请求——即最晚被调度的请求。这保证了先到的请求能够持续执行，符合先来先服务的公平原则。

**Priority（优先级）策略**：抢占优先级最低的请求。具体实现为：

```python
victim = max(self.running,
             key=lambda r: (r.priority, r.arrival_time))
```

在优先级相同时，后到的请求优先被抢占。这确保了高优先级请求的执行不会被低优先级请求阻塞。

## 8.4 抢占的执行过程

`_preempt_request()` 方法执行抢占的具体操作：

```python
def _preempt_request(self, request: Request, timestamp: float) -> None:
    assert request.status == RequestStatus.RUNNING
    self.kv_cache_manager.free(request)
    self.encoder_cache_manager.free(request)
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0
```

关键步骤包括：

1. **释放 KV Cache**：调用 `kv_cache_manager.free()` 释放该请求占用的所有物理块，最终回到 `BlockPool` 的空闲列表
2. **释放 Encoder Cache**：如果有 encoder 缓存（如多模态输入的编码结果），通过 `EncoderCacheManager`（`vllm/v1/core/encoder_cache_manager.py`）同时释放
3. **重置状态**：将请求状态设为 `RequestStatus.PREEMPTED`（定义在 `vllm/v1/request.py`）
4. **清零计算进度**：将 `num_computed_tokens` 重置为 0
5. **回到等待队列**：请求被放回 waiting 队列，等待重新调度

下图展示了从抢占触发到恢复执行的完整流程：

```text
// 触发抢占到恢复执行的完整流程

  allocate_slots() 返回 None ──▶ 显存不足！
          │
          ▼
  选择受害者（FCFS 或 Priority）
          │
          ▼
  ┌───────────────────────────┐
  │ _preempt_request()        │
  │  ├─ 释放 KV Cache         │
  │  ├─ 重置 num_computed = 0 │
  │  ├─ 状态 → PREEMPTED      │
  │  └─ 放回 waiting 队列     │
  └─────────────┬─────────────┘
                │ ...后续轮次重新调度...
                ▼
       Prefix Cache 命中？
       ┌────┴────┐
       │是       │否
       ▼         ▼
  跳过已缓存   从头重算
  仅重算剩余   全部 KV Cache
```

## 8.5 Recompute vs Swap：两种策略的权衡

在 LLM 推理系统中，处理被抢占请求有两种经典策略：

**Recompute（重计算）**：丢弃 KV Cache，下次调度时从头计算。这是 vLLM v1 当前采用的策略。优点是实现简单，不需要 CPU 内存作为中转；缺点是恢复时需要重新计算整个 prompt 的 KV Cache，对于长 prompt 来说代价较高。

**Swap（换出）**：将 KV Cache 从 GPU 显存复制到 CPU 内存，恢复时再从 CPU 复制回 GPU。优点是恢复速度快（尤其对长序列），缺点是需要额外的 CPU 内存，且数据传输有开销。

下面是 Recompute 与 Swap 的策略选择决策示意：

```text
// Recompute vs Swap 策略对比

┌─── Recompute（v1 采用）──────┐  ┌─── Swap ────────────────────┐
│                              │  │                              │
│  抢占时：丢弃 KV Cache       │  │  抢占时：KV Cache → CPU 内存 │
│  恢复时：从头计算 prompt KV  │  │  恢复时：CPU 内存 → GPU 显存 │
│                              │  │                              │
│  ✓ 实现简单，无需 CPU 内存   │  │  ✓ 恢复快（尤其长序列）      │
│  ✓ 配合 Prefix Cache 高效   │  │  ✗ 需要额外 CPU 内存         │
│  ✗ 长 prompt 重计算代价高    │  │  ✗ GPU↔CPU 数据传输有开销    │
└──────────────────────────────┘  └──────────────────────────────┘
```

v1 架构选择 Recompute 策略有几个原因：

1. **前缀缓存的补偿**：开启 Prefix Caching 后，重计算时可以命中缓存，大幅减少实际需要重新计算的 token 数量
2. **架构简洁性**：不需要管理 GPU-CPU 之间的块映射，也不需要维护 swapped 队列
3. **内存效率**：不需要预留 CPU 内存来存储换出的 KV Cache

### Swap 机制的原理

尽管 v1 目前未采用 Swap 策略，理解其原理仍有价值。Swap 机制的核心是在 GPU 和 CPU 之间建立一个块映射表：

- **swap_out()**：当请求被抢占时，将其 KV Cache 的物理块从 GPU 显存异步复制到 CPU 内存。使用 CUDA 的异步内存拷贝（`cudaMemcpyAsync`）来避免阻塞 GPU 计算
- **swap_in()**：当有足够 GPU 显存时，将之前换出的块从 CPU 复制回 GPU

这一机制要求系统维护 CPU 端的物理块池和 GPU-CPU 块 ID 映射关系。Worker 层通过 `CacheEngine` 执行实际的内存拷贝操作。

## 8.6 抢占对系统行为的影响

抢占是一个代价高昂的操作。每次抢占都意味着：

- **浪费已有计算**：被抢占请求之前生成的 KV Cache 全部丢失
- **增加延迟**：该请求重新被调度后需要重新计算 prompt 的 KV Cache
- **降低有效吞吐量**：系统花时间在"重复劳动"上

因此，减少抢占频率是优化 vLLM 性能的重要方向。关键的配置参数包括：

**`gpu_memory_utilization`**：控制可用于 KV Cache 的 GPU 显存比例。默认值通常为 0.9（90%），增大此值可以容纳更多并发请求，但留给其他 CUDA 操作的余量也更少。

**`max_num_running_reqs`**（对应 `max_num_seqs`）：限制同时运行的最大请求数。减小此值可以降低抢占概率，但也会限制吞吐量。

**`max_num_scheduled_tokens`**（对应 `max_num_batched_tokens`）：限制单步处理的 token 总数。这间接影响了系统的内存压力。

## 8.7 抢占与调度的联动

抢占机制与调度逻辑紧密耦合。一个关键设计是：**当任何 running 请求发生了抢占时，Scheduler 不会在同一步中接纳新的 waiting 请求**。

```python
if not preempted_reqs:
    # 只有没有抢占发生时，才调度新请求
    while self.waiting and token_budget > 0:
        ...
```

这一设计的原因是：抢占说明当前系统资源已经紧张，此时再引入新请求只会加剧资源竞争，很可能导致更多的抢占，形成恶性循环。等到下一步时，被抢占请求释放的资源已经到位，系统可以更稳定地做出调度决策。

## 8.8 监控与调优

vLLM 通过 metrics 系统追踪抢占事件。每次抢占发生时，Scheduler 会递增内部计数器，这些信息通过 `StatLoggerManager` 上报到 Prometheus 等监控系统。

运维人员可以通过以下指标判断系统是否需要调优：

- **抢占频率**：高频抢占说明显存配置过于激进
- **请求排队时间**：过长的排队时间可能需要增加 GPU 资源
- **有效吞吐量 vs 原始吞吐量**：差距过大说明重计算浪费严重

## 本章常见问题

**Q：抢占频率过高说明什么问题？如何优化？**

高频抢占通常意味着 GPU 显存配置过于激进——同时运行的请求太多，或者 `gpu_memory_utilization` 设得太高。优化方向包括：降低 `max_num_running_reqs`、增大 GPU 显存（换更大的 GPU）、开启 Prefix Caching 以减少重计算开销，或使用量化降低单个 token 的 KV Cache 占用。

---

**Q：被抢占的请求 num_computed_tokens 重置为 0，是否意味着所有计算都白费了？**

在不开启 Prefix Caching 的情况下确实如此，之前的 KV Cache 计算全部丢失。但开启 Prefix Caching 后，已经缓存的前缀块仍保留在 GPU 显存中（除非被 LRU 驱逐）。重新调度时，这些块可以被直接复用，实际需要重新计算的只是未被缓存的部分。

---

**Q：为什么 v1 放弃了 Swap 策略而选择 Recompute？**

三个原因：一是 Prefix Caching 大幅降低了重计算的实际开销；二是 Recompute 不需要维护 GPU-CPU 块映射和 swapped 队列，架构更简洁；三是不需要预留 CPU 内存，在多 GPU 部署时减少了 CPU 侧的内存压力。

---

**Q：Swap 策略在什么场景下仍然有优势？**

当请求的 prompt 非常长且无法命中 Prefix Cache 时，Swap 的恢复成本（GPU-CPU 数据传输）远低于 Recompute（重新计算整个 prompt 的 KV Cache）。如果系统有充足的 CPU 内存且 PCIe 带宽不是瓶颈，Swap 可能是更好的选择。

---

## 本章小结

本章详细分析了 vLLM 的抢占机制。当 GPU 显存不足时，Scheduler 通过 `_preempt_request()` 方法释放低优先级请求的 KV Cache 资源。v1 架构采用 Recompute 策略，被抢占的请求回到 waiting 队列，下次调度时从头计算。与 Swap 策略相比，Recompute 更简洁，配合前缀缓存可以有效减少重计算开销。通过合理配置 `gpu_memory_utilization` 和 `max_num_running_reqs` 等参数，可以在吞吐量和抢占频率之间找到最佳平衡点。
