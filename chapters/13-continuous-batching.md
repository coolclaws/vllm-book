# 第 13 章：Continuous Batching 原理

> "Efficiency is doing things right; effectiveness is doing the right things." —— Peter Drucker。Continuous Batching 的本质，是让 GPU 在每一个推理迭代中都做"正确的事"——不等待、不空闲、不浪费。

## 13.1 静态批处理的困境

在传统的 LLM 推理系统中，批处理（batching）的方式是**静态的**：将一组请求打包为一个 batch，一起送入模型执行，直到 batch 中所有序列都完成生成才释放整个 batch。这种方式的问题显而易见。

假设一个 batch 包含 4 个请求，生成长度分别为 10、50、200、500 个 token。在静态批处理下，前三个请求分别在第 10、50、200 步完成，但它们必须等到第 500 步才能释放——因为第四个请求还没有结束。这意味着：

- 请求 1 在第 10 步后的 490 个迭代中白白占据 GPU 算力和显存
- 系统在此期间无法接纳新请求，等待队列不断增长
- 平均响应延迟远高于必要水平
- GPU 利用率随着 batch 中请求逐渐完成而持续下降

对于在线服务场景，这种"木桶效应"是不可接受的。

## 13.2 Continuous Batching 的核心思想

Continuous Batching（也称为 iteration-level scheduling）的核心思想出奇地简单：**让序列可以在任意迭代边界加入或离开 batch**。

具体来说，调度器在每一次迭代（即每一步 token 生成）之前重新评估 batch 的组成：

1. 检查当前 batch 中是否有序列已经完成生成（遇到 EOS token 或达到最大长度）
2. 将完成的序列立即移出 batch，释放其占用的 KV Cache 和计算资源
3. 从等待队列中取出新的请求填充空出的位置
4. 将更新后的 batch 送入模型执行

这种方式下，前面的例子中请求 1 在第 10 步完成后立刻释放，其空出的"槽位"马上被等待队列中的下一个请求占据。GPU 始终在处理有意义的工作，吞吐量接近理论上限。

这一概念最早在 Orca 系统中被提出，vLLM 在其基础上与 PagedAttention 深度结合，实现了更高效的变体。

## 13.3 vLLM 中的实现

vLLM 的 continuous batching 逻辑主要实现在调度器（Scheduler）中。在 v1 架构下，核心代码位于 `vllm/v1/core/sched/scheduler.py`。每次引擎调用 `step()` 执行一轮推理时，调度器的 `schedule()` 方法会被调用，动态构建本轮的 batch。

调度流程可以概括为以下步骤：

**第一步：处理运行中的序列。** 调度器首先检查所有正在 decode 阶段的序列。对于已完成的序列（生成了 EOS 或达到 `max_tokens` 上限），将其从运行集合中移除，释放 KV Cache 块。对于继续运行的序列，为其分配新的 KV Cache 块用于存储即将生成的 token。

**第二步：接纳新请求。** 调度器从等待队列（waiting queue）中按优先级取出请求，判断是否有足够的资源（KV Cache 块和计算预算）来容纳它们。满足条件的请求被加入当前 batch，开始 prefill 阶段。

**第三步：构建执行输入。** 将所有参与本轮推理的序列——包括继续 decode 的老序列和新加入的 prefill 序列——打包为 `SchedulerOutput`，传递给 Model Runner 执行前向传播。

## 13.4 令牌预算与序列预算

Continuous batching 并不意味着无限制地接纳请求。两个关键参数控制着每轮迭代的规模：

**`max_num_batched_tokens`** 限制单次迭代中处理的总 token 数。这个参数直接影响 GPU 计算量。Prefill 阶段的 token 数等于 prompt 长度，decode 阶段每个序列贡献 1 个 token。例如，若 `max_num_batched_tokens=2048`，当前有 100 个 decode 序列（贡献 100 个 token），则还有 1948 个 token 的预算可以用于新请求的 prefill。

**`max_num_seqs`** 限制同时运行的序列数量。即使 token 预算充裕，过多的并发序列也会带来问题：每个序列的 KV Cache 都需要独立的内存管理开销，attention 计算中的 batch 维度过大也会影响效率。

调度器在 `schedule()` 中同时检查这两个预算，确保不超出任何一个限制。这种双重约束使得系统在高吞吐量和低延迟之间取得平衡。

## 13.5 Prefill 与 Decode 的混合

Continuous batching 带来了一个独特的挑战：同一个 batch 中可能同时包含 prefill 序列和 decode 序列。这两种模式的计算特性截然不同：

- **Prefill** 是计算密集型（compute-bound）：需要处理大量 prompt token，矩阵乘法的 FLOPs 占主导
- **Decode** 是内存带宽密集型（memory-bound）：每次只处理 1 个 token，瓶颈在于读取 KV Cache

将两者混合在同一 batch 中反而可以取长补短：prefill 的高计算需求和 decode 的高带宽需求可以在 GPU 上实现一定程度的重叠。vLLM 的 attention kernel 能够在同一次调用中处理不同长度的序列，通过 attention metadata 区分每个序列的状态。

## 13.6 与 Orca 方案的比较

Continuous batching 的概念源自 2022 年的 Orca 系统。vLLM 在此基础上有几个关键改进：

首先，vLLM 的 PagedAttention 使得序列加入和离开 batch 时的内存管理更加高效——新序列只需按需分配 KV Cache 块，离开的序列直接释放块到公共池，不存在内存碎片问题。

其次，vLLM 的调度器支持更复杂的策略，包括后续章节将讨论的 chunked prefill（第 14 章）和 speculative decoding（第 15 章），这些高级特性都建立在 continuous batching 的基础之上。

## 13.7 吞吐量收益

Continuous batching 的吞吐量提升效果在实际部署中非常显著。根据公开的基准测试数据，与静态批处理相比：

- 在中等并发场景下，吞吐量提升 **2~4 倍**
- 在高并发、请求长度差异大的场景下，提升可达 **10 倍以上**
- 平均首 token 延迟（Time to First Token, TTFT）显著降低，因为新请求不再需要等待整个 batch 完成

提升的本质来源于 **GPU 利用率的最大化**——在静态批处理中，GPU 在 batch 末期几乎空转；在 continuous batching 中，空出的算力立刻被新请求填满。这就像一条流水线，产品源源不断地进出，机器永不停歇。

## 本章小结

Continuous Batching 是 vLLM 高吞吐量的核心调度机制。它打破了静态批处理中"所有序列必须一起等待"的限制，允许序列在任意迭代边界加入或离开 batch。vLLM 通过调度器的 `schedule()` 方法在每轮迭代前动态构建 batch，利用令牌预算和序列预算双重约束控制规模，并与 PagedAttention 的动态内存管理深度协同。这一机制使 GPU 利用率接近理论上限，为在线 LLM 服务带来了数倍的吞吐量提升。
