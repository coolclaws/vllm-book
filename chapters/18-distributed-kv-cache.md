# 第 18 章：分布式 KV Cache 与 P/D 分离

> "工业革命的秘密不是蒸汽机本身，而是将不同工序交给不同的专业工人。P/D 分离架构对大模型推理做了同样的事——让擅长计算的去 prefill，让擅长访存的去 decode。"

## 18.1 为什么要分离 Prefill 和 Decode

大模型推理包含两个性质截然不同的阶段。**Prefill**（预填充）阶段需要一次性处理整个 prompt 序列的所有 token，涉及大规模的矩阵乘法运算，属于**计算密集型**任务，能够充分利用 GPU 的算力。**Decode**（解码）阶段则是逐步生成 token，每步只处理一个新 token，但需要读取之前所有 token 累积的 KV Cache，其瓶颈在于从显存中读取庞大的缓存数据，属于**访存密集型**任务。

当这两个阶段混合在同一组 GPU 上交替执行时，它们会产生严重的互相干扰。具体表现为：一个长 prompt 的 prefill 任务可能需要数百毫秒甚至数秒的计算时间，在此期间所有正在进行中的 decode 请求都不得不等待，导致这些请求的生成延迟（即每个 token 的输出间隔）出现明显的抖动和飙升。反过来，decode 阶段每步只处理一个 token，对 GPU 算力的利用率往往不足百分之十，大量的计算资源被浪费。

Prefill/Decode 分离（P/D Disaggregation）的核心思想应运而生：将这两个阶段**分配到不同的 GPU 集群**上独立运行，各自根据负载独立扩缩容。这样 Prefill 集群可以专注于快速处理长 prompt 以优化 TTFT（Time To First Token，首 token 延迟），而 Decode 集群则专注于高效的逐 token 生成以优化 TPOT（Time Per Output Token，每 token 延迟）和整体吞吐量。

## 18.2 架构总览

vLLM 的 P/D 分离架构包含三个核心角色和相应的数据流转路径：

1. **Prefill Worker**：接收用户的原始 prompt，执行完整的 prefill 前向传播，计算出该 prompt 所有 token 位置的 Key 和 Value 缓存张量（即 KV Cache），同时生成第一个输出 token。
2. **Decode Worker**：接收从 Prefill Worker 传输过来的 KV Cache 数据，将其填入自己的 Block Table 中，然后接管该请求的后续自回归 decode 过程，逐步生成剩余的输出 token。
3. **KV Connector**：作为 Prefill 和 Decode Worker 之间的数据传输层，负责高效地搬运 KV Cache 张量。

相关代码集中在 `vllm/distributed/kv_transfer/` 目录下。该目录的核心文件包括 `kv_transfer_state.py`（管理传输过程中的状态和同步逻辑）以及 `kv_connector/` 子目录（实现具体的传输后端）。整体设计采用了精心分层的三层抽象架构：最底层是 FIFO 管道（Pipe），负责在两个 Worker 进程之间传输原始张量数据；中间层是 Lookup Buffer，以 token 序列的哈希值为索引来缓存和检索 KV Cache 数据块；顶层是 Connector，将底层的传输能力与 vLLM 的调度器和 Worker 执行逻辑无缝集成。

## 18.3 KV Cache 传输的数据量挑战

KV Cache 的数据量非常可观。以 Llama-3-70B 为例，该模型有 80 层 Transformer、每层 8 个 KV head、每个 head 维度为 128。一个长度为 2048 个 token 的序列，其完整 KV Cache 在 FP16 精度下的大小约为：

```
2 (K+V) × 80 层 × 8 head × 128 dim × 2048 tokens × 2 bytes ≈ 6.4 GB
```

如此大的数据量对传输效率提出了极高的要求——如果传输延迟过大，就会抵消 P/D 分离带来的性能收益。vLLM 因此支持多种高性能传输后端：

- **NCCL 传输**：利用 NVIDIA 的集合通信库在 GPU 之间直接传输张量数据。NCCL 针对 NVLink 和 NVSwitch 互联进行了深度优化，非常适合同一台机器内或通过高速交换机连接的 GPU 之间的数据搬运。
- **RDMA 传输**：基于 InfiniBand 的远程直接内存访问技术，适合跨节点的长距离传输场景。RDMA 可以绕过 CPU 和操作系统内核，直接在两台机器的 GPU 显存之间进行数据拷贝，实现极低的传输延迟和极高的带宽利用率。

`kv_connector/` 子目录中定义了统一的 `KVConnectorBase` 抽象接口，所有传输后端都实现这个接口。这种可插拔的设计意味着用户和第三方开发者可以根据自己的硬件环境和网络拓扑，实现定制化的传输后端并无缝集成到 vLLM 中。

## 18.4 分布式 TP 下的 KV Cache 管理

在实际部署中，Prefill Worker 和 Decode Worker 通常各自内部使用 Tensor Parallelism 来进一步加速计算。这为 KV Cache 的传输增加了一层复杂度：在 TP=4 的配置下，每块 GPU 只持有全部 KV head 中的四分之一。传输时不能简单地将所有数据发到一个地方，而是需要 Prefill 侧的第 `i` 个 TP rank 将自己管理的 KV Cache 分片精确地发送给 Decode 侧对应编号的第 `i` 个 TP rank。

这意味着 KV Cache 传输不是一条单一的数据链路，而是 **TP rank 之间的一对一并行传输**。如果双方各有 4 块 GPU（TP=4），那么就有 4 条并行的传输链路同时工作，每条链路搬运约四分之一的 KV Cache 数据。`kv_transfer_state.py` 负责协调和同步这一并行传输过程，确保所有 TP rank 的数据都成功到位后，Decode Worker 才开始接手该请求的后续 decode 计算。如果某一条链路出现延迟或错误，整个传输过程会等待或重试，避免数据不一致的问题。

## 18.5 Prefix Caching 的分布式挑战

前面章节中介绍过的 Prefix Caching 机制，在 P/D 分离架构中需要特殊的处理。考虑一个常见的多轮对话场景：用户发送了第二轮对话，其中包含大量与第一轮相同的 system prompt 和历史消息前缀。如果 Decode Worker 的 Block Table 中已经缓存了第一轮对话的 KV Cache（因为这些 block 的哈希值命中了缓存），那么 Prefill Worker 就不需要重新传输这些已有的数据块，只需传输新增部分的 KV Cache 即可。

但实现这一优化需要 Prefill Worker 和 Decode Worker 之间**共享 prefix 的缓存状态信息**。vLLM 通过在 Connector 层维护双方的 prefix 元数据（主要是 block 的 content hash 信息）来解决这个问题：每次传输之前，先快速比对 Prefill 侧已计算的 block 哈希与 Decode 侧已缓存的 block 哈希，仅传输 Decode 侧缺失的那些 block。这种增量传输策略在多轮对话、带有固定 system prompt 的批量请求等高 prefix 复用率场景下效果显著，可以将传输量减少到完整 KV Cache 的十分之一甚至更少。

## 18.6 收益分析与工程权衡

P/D 分离架构带来的核心收益可以从三个维度衡量：

- **独立扩缩容的灵活性**：运维团队可以根据实际的 prefill 和 decode 负载分别调整 GPU 数量。如果业务场景以长 prompt 短生成为主（如文档摘要、RAG 检索增强生成），可以增配更多的 Prefill Worker；如果以长文本生成为主（如代码生成、故事续写），则增加 Decode Worker。这种弹性调度能力在云原生部署中尤为重要。
- **显著降低的首 token 延迟**：Prefill Worker 不受任何 decode 请求的干扰，可以集中全部算力处理 prompt 的预填充计算，从而大幅缩短 TTFT。
- **更高的系统整体吞吐**：Decode Worker 的显存可以完全用于 KV Cache 存储和 batch 管理，不需要为 prefill 的大矩阵运算预留显存空间，因此能支持更大的并发请求数。

当然，P/D 分离也带来了不可忽视的工程代价：KV Cache 传输本身消耗额外的时间和网络带宽；双集群架构增加了部署和运维的复杂度；请求在两个集群间的调度和负载均衡需要精心设计。此外，P/D 分离目前仍是 vLLM 中活跃演进的特性方向，社区在持续优化传输效率、降低调度延迟、以及增强容错能力。

## 本章小结

P/D 分离架构是 vLLM 面向大规模生产部署的重要架构创新。通过将计算密集的 prefill 和访存密集的 decode 分配到各自专属的 GPU 集群，配合 `vllm/distributed/kv_transfer/` 中实现的高效 KV Cache 传输机制，系统可以同时优化首 token 延迟和生成吞吐两个关键指标。`KVConnectorBase` 的可插拔接口设计确保了传输后端的灵活适配，TP 感知的分片并行传输解决了分布式环境下的数据路由问题，而基于 content hash 的增量传输策略则通过 prefix 缓存去重大幅减少了冗余的数据搬运。理解 P/D 分离的架构思想和实现细节，对于设计面向生产环境的大模型推理服务架构具有重要的参考价值。
