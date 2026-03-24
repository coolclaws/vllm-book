# 第 6 章：Scheduler 架构

> "The art of scheduling is the art of choosing wisely under constraints." —— 在推理系统中，Scheduler 就是那个在有限资源下做出关键取舍的决策者。每一次迭代，它都在回答同一个问题：哪些请求值得现在被处理？

## 6.1 Scheduler 的角色

如果把 vLLM 比作一个繁忙的餐厅，那 Scheduler 就是前台经理。它决定哪些客人（请求）可以入座（获得 GPU 资源），哪些需要在等候区排队，哪些已经用餐完毕需要让出座位。Scheduler 是整个推理引擎的"大脑"，在每一个推理步骤中统筹全局。

在 vLLM v1 架构中，Scheduler 的核心实现位于 `vllm/v1/core/sched/scheduler.py`，它实现了 `SchedulerInterface`（定义在 `vllm/v1/core/sched/interface.py`）。辅助模块包括：

- `vllm/v1/core/sched/output.py`：定义 `SchedulerOutput` 数据结构
- `vllm/v1/core/sched/request_queue.py`：请求优先队列的实现
- `vllm/v1/core/sched/utils.py`：调度工具函数

## 6.2 两个核心队列

Scheduler 维护着两个关键队列来管理请求的生命周期：

**Waiting 队列**：所有新到达的请求首先进入 `self.waiting`，这是一个优先队列（Priority Queue）。请求在此排队等待被调度执行。优先级可以基于到达时间（FCFS，先来先服务）或显式优先级字段。

**Running 列表**：当请求被选中执行后，它们进入 `self.running` 列表。这些请求正在占用 GPU 上的 KV Cache 资源，每一步都会生成新的 token。

此外还有一个 `self.skipped_waiting` 队列，用于暂存那些因为外部依赖（如等待远程 KV Cache 传输、结构化输出的 FSM 编译等）而暂时无法调度的请求。

与早期版本不同，v1 架构中不再维护独立的 swapped 队列。被抢占的请求会重置其已计算的 token 数，回到 waiting 队列重新排队——这意味着 v1 采用了**重计算（Recompute）策略**而非换出到 CPU 内存。

请求在调度器中的状态流转如下图所示：

```text
// 请求状态流转图

                 新请求到达
                     │
                     ▼
              ┌────────────┐    重置计算进度
              │  Waiting   │◀───────────────┐
              │  (等待中)  │                │
              └─────┬──────┘                │
       调度器选中   │                       │
       分配KV Cache │                ┌──────┴─────┐
                    ▼                │ Preempted  │
              ┌────────────┐        │ (被抢占)   │
              │  Running   │────────┘            │
              │  (运行中)  │  显存不足，释放KV   │
              └─────┬──────┘                     │
                    │ 生成 EOS 或达到最大长度
                    ▼
              ┌────────────┐
              │ Finished   │
              │  (完成)    │
              └────────────┘
```

## 6.3 调度循环：schedule() 方法

`schedule()` 方法是 Scheduler 的心脏，每个推理步骤调用一次。它的执行流程分为两个主要阶段：

### 第一阶段：处理 Running 请求

调度器首先遍历 `self.running` 中的所有请求，尝试为每个请求分配新的 token 预算。核心逻辑如下：

```python
token_budget = self.max_num_scheduled_tokens
req_index = 0
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
    num_new_tokens = request.num_tokens - request.num_computed_tokens
    new_blocks = self.kv_cache_manager.allocate_slots(request, ...)
    if new_blocks is not None:
        # 分配成功，扣减预算
        token_budget -= num_new_tokens
    else:
        # 分配失败，需要抢占
        self._preempt_request(victim, ...)
```

当 `kv_cache_manager.allocate_slots()` 返回 `None` 时，说明 GPU 显存不足。此时 Scheduler 必须**抢占（preempt）**优先级最低的请求来释放资源。

### 第二阶段：调度 Waiting 请求

只有在没有发生抢占的情况下，Scheduler 才会从 waiting 队列中取出新请求。这一设计保证了已经在运行的请求不会被新请求"饿死"：

```python
if not preempted_reqs:
    while self.waiting and token_budget > 0:
        if len(self.running) == self.max_num_running_reqs:
            break
        request = self.waiting.popleft()
        # 尝试为新请求分配 KV Cache 块
        ...
```

两个硬约束贯穿始终：
- **Token 预算**（`max_num_scheduled_tokens`）：单步最多处理的 token 总数
- **请求数上限**（`max_num_running_reqs`）：同时运行的最大请求数

下图展示 `schedule()` 方法的完整决策流程：

```text
// schedule() 两阶段决策流程

┌─────────────────────────────────────────┐
│  Phase 1：处理 running 请求              │
│                                         │
│  遍历 running 请求                      │
│    ├─ 能分配 KV Cache？                 │
│    │   ├─ 是 → 扣减 token 预算          │
│    │   └─ 否 → 抢占最低优先级请求       │
│    │           释放资源后重试分配        │
│    └─ 继续直到遍历完毕或预算用尽        │
└───────────────────┬─────────────────────┘
                    │
                    ▼
            发生过抢占？
           ┌────┴────┐
           │是       │否
           ▼         ▼
    ┌──────────┐  ┌──────────────────────────────┐
    │ 跳过      │  │ Phase 2：调度 waiting 请求    │
    │ waiting   │  │                              │
    │ 本轮结束  │  │  从 waiting 取请求            │
    └──────────┘  │  分配资源，加入 running        │
                  │  直到预算用尽或达到请求数上限  │
                  └──────────────────────────────┘
```

## 6.4 SchedulerOutput：调度决策的表达

`schedule()` 方法的返回值是一个 `SchedulerOutput` 对象（定义在 `vllm/v1/core/sched/output.py`），它完整描述了本步的调度决策：

- `scheduled_new_reqs`：首次调度的新请求列表
- `scheduled_cached_reqs`：继续执行的已有请求数据
- `num_scheduled_tokens`：每个请求本步分配的 token 数
- `total_num_scheduled_tokens`：所有请求的 token 总数
- `finished_req_ids`：本步完成的请求 ID 集合
- `preempted_req_ids`：被抢占的请求 ID 集合
- `new_block_ids_to_zero`：需要初始化的新 GPU 内存块

这个数据结构是 Scheduler 与下游 Worker 之间的"契约"——Worker 根据它准备模型输入并执行前向计算。

## 6.5 调度策略

Scheduler 支持两种调度策略：

**FCFS（先来先服务）**：默认策略，按请求到达时间排序。当需要抢占时，最后到达的请求最先被牺牲。这种策略简单公平，适合大多数在线服务场景。

**Priority（优先级调度）**：根据请求的 `priority` 字段排序。抢占时选择优先级数值最大（优先级最低）的请求。适用于需要区分请求重要性的场景。

在优先级策略下，抢占受害者的选择逻辑为：

```python
if self.policy == SchedulingPolicy.PRIORITY:
    victim = max(self.running,
                 key=lambda r: (r.priority, r.arrival_time))
```

## 6.6 与 KVCacheManager 的协作

Scheduler 不直接管理内存，而是通过 `KVCacheManager`（`vllm/v1/core/kv_cache_manager.py`）来感知和操作 GPU 内存状态。两者的协作模式如下：

1. **查询可用块**：Scheduler 通过 `allocate_slots()` 尝试为请求分配 KV Cache 块
2. **触发抢占**：分配失败时，Scheduler 调用 `kv_cache_manager.free()` 释放被抢占请求的块
3. **前缀缓存**：通过 `get_computed_blocks()` 复用已缓存的前缀块，减少重复计算
4. **块初始化**：新分配的块通过 `take_new_block_ids()` 记录，由 Worker 在 GPU 上初始化为零

这种分离设计使得调度策略和内存管理可以独立演进，也便于支持不同的 KV Cache 架构（如分布式 KV Cache）。

## 6.7 SchedulerConfig：关键配置

影响调度行为的核心配置参数包括：

| 参数 | 含义 |
|------|------|
| `max_num_running_reqs` | 最大并发运行请求数 |
| `max_num_scheduled_tokens` | 单步最大调度 token 数 |
| `max_model_len` | 模型支持的最大序列长度 |
| `gpu_memory_utilization` | GPU 显存利用率上限 |

这些参数共同决定了系统的吞吐量和延迟特征。`max_num_scheduled_tokens` 较大时有利于批量处理提高吞吐量，较小时有利于降低单请求延迟。

## 6.8 异步调度与暂停机制

在 v1 架构中，Scheduler 还支持异步调度模式（`vllm/v1/core/sched/async_scheduler.py`），允许调度与执行在不同的线程或进程中并行运行，进一步降低调度开销。

此外，Scheduler 还实现了**暂停机制**（pause/resume）。当系统需要暂停接收新请求时（例如进行模型更新或内存整理），可以将 Scheduler 置于暂停状态。暂停模式支持三种策略处理正在运行的请求：中止所有请求（abort）、等待当前请求完成（wait）、或保持运行但不接纳新请求（keep）。这一机制在生产环境中的灰度发布和热更新场景中非常实用。

## 6.9 结构化输出与调度的协作

当请求需要结构化输出（如 JSON Schema 约束）时，Scheduler 需要与 `structured_output_manager` 协作。请求首先进入 `WAITING_FOR_FSM` 状态，等待有限状态机编译完成后才能被正式调度。这避免了在 FSM 未就绪时浪费 GPU 计算资源，也体现了 Scheduler 作为全局协调者需要感知各种子系统状态的特点。

## 本章常见问题

**Q：为什么发生抢占后不再接纳新的 waiting 请求？**

抢占意味着当前系统资源已经紧张到无法维持现有请求。如果此时再接纳新请求，很可能立即触发更多抢占，形成恶性循环。等到下一个调度周期，被抢占请求释放的资源已经稳定到位，调度器能做出更合理的决策。

---

**Q：token 预算和请求数上限哪个更容易成为瓶颈？**

取决于负载特征。如果请求以短 prompt + 长生成为主（如对话场景），请求数上限通常先触顶；如果请求以长 prompt 为主（如文档摘要），token 预算更容易成为瓶颈。实际部署中两者需要配合调优。

---

**Q：v1 为什么去掉了独立的 swapped 队列？**

v1 采用 Recompute 策略替代 Swap，被抢占的请求直接回到 waiting 队列。这简化了状态管理（只需维护 waiting 和 running 两个队列），配合 Prefix Caching 可以复用已缓存的前缀块，使重计算的实际开销大大降低。

---

**Q：SchedulerOutput 为什么要区分 scheduled_new_reqs 和 scheduled_cached_reqs？**

新请求需要 Worker 执行完整的 prefill（加载 prompt token、初始化 KV Cache），而已有请求只需执行增量 decode（处理 1 个新 token）。两者的数据准备和计算逻辑完全不同，区分开来能让 Worker 更高效地构建模型输入。

---

## 本章常见问题

**Q：为什么发生抢占后当轮不再接纳新请求？**

抢占说明系统资源已经紧张。如果此时再引入新请求，很可能导致更多抢占，形成"抢占→新请求加剧压力→再次抢占"的恶性循环。跳过一轮让系统回到稳态后，下一步可以更稳定地做出调度决策。

---

**Q：token 预算和请求数上限为什么需要同时约束？**

两者控制不同维度的资源。token 预算限制的是 GPU 计算量（单步处理的 token 总数），请求数上限限制的是内存管理开销和 attention 计算中 batch 维度的大小。即使 token 预算充裕，过多的并发序列会带来过高的内存管理开销；反之，少量超长序列即使不多也可能耗尽计算预算。

---

**Q：v1 架构去掉了 swapped 队列，被抢占的请求怎么办？**

v1 采用 Recompute 策略。被抢占的请求释放所有 KV Cache，`num_computed_tokens` 重置为 0，回到 waiting 队列重新排队。下次被调度时从头计算 prompt 的 KV Cache。配合 Prefix Caching，重计算时可以命中缓存，大幅降低实际重计算量。

---

**Q：FCFS 和 Priority 策略在生产中如何选择？**

大多数在线服务使用 FCFS，它简单公平，避免低优先级请求被长期饿死。当业务需要区分请求重要性时（如 VIP 用户优先、实时请求优先于离线任务），使用 Priority 策略，通过 `priority` 字段控制调度顺序和抢占选择。

---

## 本章小结

本章剖析了 vLLM Scheduler 的架构设计。Scheduler 作为推理引擎的调度核心，通过 waiting 和 running 两个队列管理请求生命周期，在每一步的 `schedule()` 调用中做出资源分配决策。它先保障已运行请求的资源需求，再接纳新请求，必要时通过抢占机制释放资源。`SchedulerOutput` 将调度决策传达给下游的 Worker 执行。这一设计在吞吐量与延迟之间取得了良好的平衡，也为后续章节中将要介绍的序列管理、抢占机制和引擎架构奠定了基础。
