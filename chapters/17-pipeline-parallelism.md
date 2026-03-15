# 第 17 章：Pipeline Parallelism

> "流水线的发明让福特 T 型车的装配时间从 12 小时缩短到 93 分钟。同样的思想，也让大模型推理跨越了单机的边界。"

## 17.1 从 TP 到 PP：另一种切分维度

上一章介绍的 Tensor Parallelism 在层**内部**切分权重矩阵，需要频繁的 AllReduce 通信，因此最适合 NVLink 等高带宽互联环境。然而当模型规模继续增长，单台机器的所有 GPU 加起来仍然无法容纳全部权重时，我们就需要跨节点部署模型。此时，AllReduce 的高通信量就会成为跨节点低带宽互联的瓶颈。Pipeline Parallelism（流水线并行，简称 PP）应运而生。

PP 的思路截然不同：它不在层内切分，而是将模型按**层的边界**划分为若干阶段（stage），每个阶段部署在不同的 GPU 或 GPU 组上。数据像流水线上的产品一样，从第一个阶段依次流向最后一个阶段。相邻阶段之间只需要传递一组激活值（hidden states），通信模式为简单的点对点 Send/Recv 操作，通信量远小于 AllReduce，因此对网络带宽的要求也低得多。

## 17.2 层的分配策略

假设模型有 `N` 个 Transformer 层，`pipeline_parallel_size` 为 `PP`，则最直接的均匀分配方式为：

- Stage 0：层 `0` 到 `N/PP - 1`
- Stage 1：层 `N/PP` 到 `2N/PP - 1`
- ...
- Stage PP-1：层 `(PP-1)*N/PP` 到 `N - 1`

但 Transformer 模型不仅仅有中间的 Decoder 层。Stage 0 通常还额外负责 Embedding 层和位置编码的计算——这些是模型的"入口"，只有第一个阶段需要处理原始的 token ID 输入。而 Stage PP-1 则额外包含最终的 LayerNorm 和 LM Head——这些是模型的"出口"，负责将最后一层的 hidden states 转换为词表空间上的 logits 概率分布并完成 token 采样。

这一分配逻辑在各模型的具体实现文件（如 `modeling_llama.py`）中完成。vLLM 的 Worker 通过 `vllm/worker/worker.py` 启动时，根据自身的 PP rank 信息决定只加载和初始化属于自己阶段的层权重，其余层则完全跳过，不占用任何显存。这种按需加载机制使得每个 Worker 的显存占用近似为 `总模型大小 / PP`。

## 17.3 流水线调度与 Micro-batch

PP 的最大挑战是**流水线气泡（pipeline bubble）**问题。在最朴素的执行方式中，当 Stage 0 正在处理一个 batch 的输入时，Stage 1 到 Stage PP-1 都处于空闲等待状态。直到 Stage 0 完成并将激活值发送给 Stage 1 后，Stage 1 才能开始工作——而此时 Stage 0 又变成了空闲状态。这种"某一时刻只有一个 stage 在工作"的情况导致 GPU 利用率极低。

经典的缓解方案是 **Micro-batching**：将一个大 batch 拆分为多个小的 micro-batch，让多个 micro-batch 在流水线的不同 stage 中交错执行。当 Stage 0 完成第一个 micro-batch 并发送给 Stage 1 后，它不必等待 Stage 1 完成，而是立即开始处理第二个 micro-batch。这样，多个 stage 可以同时忙碌，显著减少了气泡。

对于推理场景，vLLM 的 PP 调度策略有其特殊性。推理只有前向传播没有反向传播，因此不能直接套用训练中经典的 1F1B（one-forward-one-backward）调度。在 prefill 阶段，完整的 prompt 序列从 Stage 0 逐级流经所有 stage，生成完整的 KV Cache；在 decode 阶段，每个生成步骤产生的单个 token 同样需要经过完整的流水线路径。推理 PP 调度更关注**延迟最小化**——每个生成步骤的延迟等于所有 stage 的计算时间加上 stage 间的通信时间再加上气泡等待时间。

## 17.4 阶段间通信

相邻 stage 之间的通信通过 `torch.distributed` 的点对点原语 `send()` 和 `recv()` 实现。`vllm/distributed/parallel_state.py` 提供了完整的 PP 分组管理接口：

- `get_pipeline_model_parallel_world_size()`：返回 PP 组的 stage 总数。
- `get_pipeline_model_parallel_rank()`：返回当前 Worker 在 PP 流水线中的位置编号。
- `is_pipeline_first_stage()`：判断当前 Worker 是否是流水线的第一个阶段。该阶段负责接收 tokenized 输入并执行 Embedding 嵌入计算。
- `is_pipeline_last_stage()`：判断当前 Worker 是否是最后一个阶段。该阶段负责计算输出 logits 并执行采样得到生成的 token。
- `get_pipeline_model_parallel_prev_rank()` 和 `get_pipeline_model_parallel_next_rank()`：获取流水线中前驱和后继 stage 的 rank，用于建立点对点通信链路。

第一个 stage 接收外部输入并完成 Embedding 计算，将 hidden states 发送给第二个 stage。中间的 stage 接收上游的 hidden states，经过自己负责的若干 Transformer 层处理后，将结果转发给下游。最后一个 stage 完成最终的 LayerNorm、LM Head 计算和 token 采样，然后将采样结果返回给调度器。

传输的数据量等于 `batch_size × seq_len × hidden_size` 个浮点数。对于一个 hidden_size=8192 的模型，一个长度为 2048 的序列在 FP16 精度下的激活值大小约为 32MB。这在 200Gbps 的 InfiniBand 网络上的传输时间约为 1.3 毫秒——远低于一个 stage 的计算时间，因此通信开销通常不是瓶颈所在。

## 17.5 TP 与 PP 的联合使用

在实际的大规模生产部署中，TP 和 PP 往往需要联合使用。总的 GPU 数量等于 `TP × PP`。最佳实践的配置策略是：

- **同一节点内**使用 TP，充分利用 NVLink/NVSwitch 的高带宽（单链路 600GB/s）完成频繁的 AllReduce 通信。
- **跨节点**使用 PP，只需通过 InfiniBand 传递点对点的激活值，对带宽要求较低。

举一个具体例子：在 4 台 8×A100（共 32 块 GPU）机器上部署 Llama-3-405B 模型，可以设置 `TP=8, PP=4`。每台机器内 8 块 GPU 组成一个 TP 组，利用机内 NVLink 互联完成层内的 AllReduce；4 台机器按顺序组成一条 4 级流水线，通过跨机器的 InfiniBand 网络传递激活值。`parallel_state.py` 会自动将 32 块 GPU 正确地划分为 4 个独立的 TP 组和若干 PP 组，确保不同类型的通信操作在各自的进程组内独立执行、互不干扰。

vLLM 通过 `EngineArgs` 中的 `tensor_parallel_size` 和 `pipeline_parallel_size` 两个参数暴露这一配置：

```bash
# 启动 TP=8, PP=4 的混合并行部署
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-405B \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 4
```

## 17.6 何时选择 PP

PP 的核心优势在于低通信量和跨节点友好，但它不可避免地引入了流水线气泡导致的延迟增加。以下是选择 PP 的实用决策指南：

- **必须使用 PP 的场景**：模型规模超出单节点的总显存容量，纯 TP 已经无法容纳全部权重。此时 PP 是唯一的选择。
- **优先使用 PP 的场景**：需要跨节点部署时，应优先使用 PP 而非跨节点 TP。跨节点 AllReduce 的延迟在低带宽互联下会成为严重瓶颈，而 PP 的点对点通信量小得多。
- **不建议使用 PP 的场景**：如果模型可以在单节点内通过 TP 完全容纳（例如在 8 块 80GB A100 上部署 70B 模型），那么纯 TP 没有气泡开销，延迟更低，是更优的选择。
- **混合使用的经验法则**：PP 的 stage 数量不宜过多，通常 2~8 级为宜。过多的 stage 会导致气泡占比升高，且每个 stage 分到的层数过少反而降低了计算效率。

## 本章小结

Pipeline Parallelism 通过将模型按层边界切分为多个 stage，以轻量级的点对点通信替代开销较大的 AllReduce 操作，使大模型推理得以优雅地跨越单机边界。vLLM 在 `vllm/worker/worker.py` 中完成每个 stage 的层分配和按需权重加载，在 `vllm/distributed/parallel_state.py` 中管理 PP 分组、rank 信息以及阶段判断接口。PP 与 TP 正交互补——TP 适合节点内高带宽互联场景，PP 适合跨节点低通信量场景——二者的组合使用是当前大模型分布式推理部署的标准实践。理解这两种并行策略的分工、权衡与协作方式，是设计高效分布式推理方案的关键能力。
