# 第 14 章：Chunked Prefill

> "The art of progress is to preserve order amid change and to preserve change amid order." —— Alfred North Whitehead。Chunked Prefill 正是在长 prompt 处理与低延迟响应之间寻找秩序的艺术。

## 14.1 长 Prefill 的延迟问题

在第 13 章中，我们了解了 continuous batching 如何通过动态调度提升吞吐量。但这一机制存在一个隐含的问题：**当一个超长 prompt 进入 prefill 阶段时，它会独占大量的计算预算，阻塞其他正在 decode 的序列。**

设想这样一个场景：系统正在为 50 个用户生成文本（decode 阶段），每个用户每轮只需要处理 1 个 token，共 50 个 token。此时一个新请求到来，prompt 长度为 32000 个 token。在不做分块的情况下，调度器会在下一轮迭代中将这 32000 个 token 全部送入 prefill，此时 GPU 需要处理 32050 个 token。

这次"巨型"前向传播可能需要数百毫秒甚至数秒才能完成。在此期间，那 50 个 decode 用户不得不等待——他们本来每 20~50 毫秒就能收到一个新 token，现在突然出现了数百毫秒的延迟空洞。对于流式生成（streaming）场景，用户会明显感受到"卡顿"。

这就是 **prefill 抢占 decode** 问题，也是 chunked prefill 要解决的核心痛点。

## 14.2 Chunked Prefill 的解决方案

Chunked Prefill 的思路直观而优雅：**不要一次性处理整个长 prompt，而是将其切分为多个小块（chunk），每轮迭代只处理一个 chunk，与 decode token 交错执行。**

以上面的场景为例，假设 `max_num_batched_tokens=2048`。长 prompt 被切分为 16 个 chunk（每个 2000 token），每轮迭代处理一个 chunk 外加 48 个 decode token。16 轮迭代后，长 prompt 的 prefill 全部完成，而 decode 用户在这期间始终以正常节奏收到新 token。

vLLM 通过 `enable_chunked_prefill` 配置项启用此功能。在 v1 架构中，chunked prefill 是默认行为，由 `vllm/v1/core/sched/scheduler.py` 中的调度逻辑自动处理。

## 14.3 调度器中的分块逻辑

Chunked Prefill 的核心实现位于调度器的 prefill 调度阶段。当一个新请求进入系统时，调度器不再假设必须一次性完成所有 prompt token 的处理。具体流程如下：

**计算预算分配**：调度器首先为当前运行中的 decode 序列预留 token 预算（每个序列 1 个 token）。剩余的 token 预算分配给等待 prefill 的请求。

**分块处理**：对于等待 prefill 的请求，调度器检查剩余的 token 预算。如果请求的未处理 token 数超过预算，只调度预算允许的部分——这就是一个 chunk。请求的 `num_computed_tokens` 字段记录已经处理了多少 prompt token，下一轮迭代将从这个偏移量继续。

**多轮完成**：一个长 prompt 可能需要跨越多轮迭代才能完成 prefill。每轮处理一个 chunk，直到 `num_computed_tokens` 等于 prompt 总长度。此时请求进入 decode 阶段，开始逐 token 生成。

**优先级控制**：调度器可以决定优先完成已经开始 prefill 的请求（减少首 token 延迟），还是将预算分散给多个请求（提高公平性）。这一策略通过调度器的配置来调节。

## 14.4 Attention 计算的挑战

Chunked prefill 给 attention 计算带来了额外的复杂性。在同一个 batch 中，不同序列处于不同的状态：

- 一些序列在做 decode，query 长度为 1，但 KV Cache 可能很长
- 一些序列在做 prefill 的某个 chunk，query 长度为 chunk 大小，KV Cache 包含之前 chunk 已经计算的部分
- 同一序列的 prefill 部分和已缓存部分需要正确拼接

vLLM 的 attention metadata 需要精确记录每个序列的状态：哪些 token 是本轮新增的 prefill token，哪些 token 的 KV 已经缓存在 PagedAttention 的块中。FlashAttention 和 FlashInfer 等 attention kernel 需要根据这些信息正确处理变长的 query 和 key-value 序列。

在 `vllm/v1/worker/gpu_model_runner.py` 中，Model Runner 负责构建这些 attention metadata。它根据调度器输出的信息，为每个序列设置正确的 query 起始位置、KV Cache 的块表映射和序列长度，确保 attention 计算的正确性。

## 14.5 与 PagedAttention 的协同

Chunked prefill 与 PagedAttention 的块分配机制有着天然的契合。每处理一个 chunk，系统需要为新计算的 KV 分配物理块。由于 PagedAttention 采用按需分配的策略，每个 chunk 只需分配其覆盖的 token 所需的块数。

例如，block size 为 16 时，一个 2048 token 的 chunk 需要 128 个新块。`vllm/v1/core/kv_cache_manager.py` 中的 KV Cache Manager 在每轮调度时为 prefill chunk 分配所需的块。如果系统的空闲块不足以容纳完整的 chunk，调度器会进一步缩小 chunk 大小以适应可用资源。

这种弹性分配机制使得 chunked prefill 不仅是计算层面的优化，也是内存管理层面的优化——系统始终按实际需要分配块，不会因为一个超长 prompt 而一次性耗尽全部可用块。

## 14.6 性能权衡与调优

Chunked prefill 引入了一个关键的调优参数：**chunk 大小**（由 `max_num_batched_tokens` 间接控制）。

**较大的 chunk 大小**（如 4096 或 8192）：
- prefill 的总迭代次数更少，首 token 延迟（TTFT）更低
- 每轮 prefill 占用更多计算预算，decode 用户的 inter-token 延迟波动更大
- GPU 计算效率更高，因为矩阵乘法在较大维度上有更好的硬件利用率

**较小的 chunk 大小**（如 512 或 1024）：
- decode 用户的 inter-token 延迟更加平稳，流式体验更好
- prefill 需要更多轮次完成，首 token 延迟增加
- 频繁的 chunk 切换带来更多调度开销

实际部署中的最优值取决于业务场景。对于交互式对话服务，较小的 chunk 大小（1024~2048）能提供更平滑的用户体验；对于批量处理场景，较大的 chunk 大小能最大化整体吞吐量。vLLM 在 v1 架构中默认启用 chunked prefill，并根据模型规模和 GPU 型号自动选择合理的默认值。

## 14.7 超长上下文的支持

Chunked prefill 的另一个重要价值在于**支持超长上下文窗口**。当模型支持 128K 甚至 1M token 的上下文时，一次性 prefill 整个 prompt 所需的显存和计算量可能超出单次前向传播的能力。Chunked prefill 将这个问题化解为多次可管理的小规模计算，使得 vLLM 能够在标准 GPU 配置下服务超长上下文请求，而不需要为峰值负载预留巨大的计算和内存资源。

## 本章小结

Chunked Prefill 解决了 continuous batching 中长 prompt prefill 阻塞 decode 的延迟问题。通过将长 prompt 分块处理并与 decode token 交错执行，它在首 token 延迟和 inter-token 延迟之间取得平衡。调度器通过 `num_computed_tokens` 追踪 prefill 进度，Model Runner 通过精细的 attention metadata 区分同一 batch 中的不同序列状态，KV Cache Manager 按 chunk 粒度进行弹性块分配。chunk 大小是核心调优参数，需要根据业务场景在吞吐量和延迟平滑度之间权衡。这一机制也为超长上下文窗口的实际部署提供了工程基础。
