# 第 16 章：Tensor Parallelism

> "If you want to go fast, go alone. If you want to go far, go together." —— 非洲谚语。Tensor Parallelism 的精髓正在于此：让多块 GPU 携手拆解同一层计算，共同驾驭单卡无法容纳的巨型模型。

## 16.1 什么是 Tensor Parallelism

当模型规模超出单块 GPU 的显存上限时，最直觉的做法是把模型的**同一层**拆分到多块 GPU 上并行计算。这就是 Tensor Parallelism（张量并行，简称 TP）的核心思想——不是把不同的层放到不同的卡上（那是后文将介绍的 Pipeline Parallelism），而是把**一个层内部的权重矩阵**切分到多块 GPU 上，让每块 GPU 只负责一部分矩阵乘法，最后再通过集合通信汇总结果。

vLLM 的 TP 实现深受 Megatron-LM 论文的影响，核心代码分布在 `vllm/distributed/` 和 `vllm/model_executor/layers/` 两个目录下。用户只需在启动引擎时设置 `tensor_parallel_size` 参数，即可透明地开启张量并行——所有的权重切分、通信同步、以及结果汇总都由框架自动完成，模型开发者几乎无需关心底层的并行细节。

要理解 TP 的工作原理，关键在于理解 Transformer 层中线性变换的矩阵切分方式。一个标准的 Transformer 层由 Self-Attention 和 Feed-Forward Network（MLP）两个子层组成，每个子层内部包含若干线性变换。TP 的核心就是决定如何将这些线性变换的权重矩阵沿不同维度切分到多块 GPU 上。

## 16.2 Megatron 风格的矩阵切分

Transformer 中的核心运算是线性变换 `Y = XW`。对权重矩阵 `W` 的切分方式不同，就产生了两种基本的并行模式，二者在 Transformer 层中交替配合使用。

### Column Parallel Linear

`ColumnParallelLinear`（定义在 `vllm/model_executor/layers/linear.py`）沿**列方向**切分权重矩阵。假设有 `tp_size` 块 GPU，第 `i` 块 GPU 持有 `W[:, i*k:(i+1)*k]`，其中 `k = hidden_size / tp_size`。每块 GPU 用完整的输入 `X` 乘以自己持有的权重切片，独立计算 `Y_i = X @ W_i`，得到输出张量沿特征维度的一个切片。这种方式天然适合 Transformer 中 MLP 的第一层线性变换（将 hidden_size 映射到 intermediate_size）和 QKV 投影——因为 Multi-Head Attention 的多个 head 可以自然地按 head 维度分配给不同的 GPU，每块 GPU 计算若干个 head 的 Q、K、V 即可。

列切分的一个重要优势是：**前向传播不需要任何通信**。每块 GPU 各自独立计算，输出是完整结果的一个切片，可以直接传给下游的行切分层。

### Row Parallel Linear

`RowParallelLinear` 则沿**行方向**切分权重。第 `i` 块 GPU 持有 `W[i*k:(i+1)*k, :]`，对应地，输入 `X` 也需要是按列切分的（通常正好是上游 Column Parallel 层输出的切片）。每块 GPU 计算 `Y_i = X_i @ W_i`，得到的 `Y_i` 具有与最终输出相同的形状，但只是**部分和**——因为每块 GPU 只看到了输入的一部分特征。要得到最终结果，必须通过 AllReduce 操作对所有 GPU 的 `Y_i` 求和。这种方式通常用于 MLP 的第二层线性变换（将 intermediate_size 映射回 hidden_size）以及 Attention 的输出投影（将多个 head 的输出合并）。

在 `linear.py` 中，这两个基础类还有多个对应的 Merged 变体。`MergedColumnParallelLinear` 将多个并行的列切分线性层合并为一次大矩阵乘法；`QKVParallelLinear` 则专门针对 Q、K、V 三个投影矩阵的合并做了优化，支持 GQA（Grouped Query Attention）场景下 Q 和 KV 具有不同 head 数量的情况。这些合并变体通过减少 kernel 启动次数来提升计算效率。

## 16.3 AllReduce 通信

TP 的通信开销集中在 `RowParallelLinear` 之后的 AllReduce 操作。在一个标准 Transformer 层中，这种通信恰好出现两次：一次在 Attention 输出投影（Row Parallel）之后，一次在 MLP 第二层线性变换（Row Parallel）之后。AllReduce 将所有 GPU 上的部分和汇总为全局和，并将结果广播回每块 GPU，使得每块 GPU 都拥有完整的输出，用于后续的 LayerNorm 和残差连接。

vLLM 默认使用 NCCL 作为 GPU 间通信后端，但在 `vllm/distributed/device_communicators/` 目录下还实现了自定义的 AllReduce 内核。对于小消息量场景（例如 decode 阶段单个 token 的 hidden states），自定义内核通过共享内存（NVLink 直连）或 P2P 拷贝避免 NCCL 的启动开销和同步延迟，能获得显著的延迟优势。通信后端的选择在运行时根据消息大小自动决定，开发者无需手动干预。

值得注意的是，AllReduce 的通信量与序列长度和 hidden_size 成正比，但与模型层数无关——因为每层的 AllReduce 是独立的、流水线化的。在 NVLink 互联的单机环境下（如 DGX A100 的 600GB/s 双向带宽），AllReduce 的延迟通常在微秒到毫秒级别，对总推理延迟的影响有限。

## 16.4 Vocabulary Parallel Embedding

除了线性层，Embedding 层同样需要并行化处理。`VocabParallelEmbedding`（定义在 `vllm/model_executor/layers/vocab_parallel_embedding.py`）将词表沿词汇维度均匀切分，每块 GPU 只存储 `vocab_size / tp_size` 个 token 的嵌入向量。

查表过程并不复杂：每块 GPU 接收到完整的 token ID 序列后，判断每个 token 是否落在自己负责的词汇范围内。对于自己范围内的 token，正常查表取得嵌入向量；对于范围外的 token，返回全零向量。最后通过 AllReduce 将所有 GPU 的结果求和——由于每个 token 只在一块 GPU 上有非零值，求和后即得到完整的 Embedding 输出。

同样的思路也适用于输出层的 LM Head。当 LM Head 与 Embedding 共享权重时，输出 logits 的计算也自然地被并行化：每块 GPU 计算自己负责的词汇范围内的 logits，最后通过 AllGather 拼接为完整的 logits 向量供采样使用。

## 16.5 并行组管理

TP 的分组逻辑由 `vllm/distributed/parallel_state.py` 统一管理。该模块是整个分布式推理的中枢，维护了全局的进程组信息。与 TP 相关的核心接口包括：

- `get_tensor_model_parallel_world_size()`：返回 TP 组的 GPU 数量，即 `tensor_parallel_size`。
- `get_tensor_model_parallel_rank()`：返回当前 GPU 在 TP 组中的 rank 编号（从 0 开始）。
- `get_tensor_model_parallel_group()`：返回底层的 `torch.distributed` 通信组对象，供 AllReduce、AllGather 等集合操作使用。

当 TP 与 PP 联合使用时，`parallel_state.py` 会根据 TP size 和 PP size 的组合，将全局的 GPU 集合正确地划分为多个独立的 TP 组和 PP 组，确保 AllReduce 等通信操作只在同一个 TP 组内执行，不会跨组串扰。这种分组管理对上层模型代码完全透明。

## 16.6 显存节省与通信开销分析

TP 的显存收益非常直观：每块 GPU 只需存储约 `1/tp_size` 的模型权重参数。一个 70B 参数的模型在 FP16 下需要约 140GB 显存用于权重存储，使用 TP=4 后每块 GPU 只需约 35GB——恰好可以装入一块 80GB 的 A100 GPU，还有充裕的空间留给 KV Cache 和运行时的激活值。

但通信开销不可忽视。每个 Transformer 层需要两次 AllReduce，以 Llama-3-70B 的 80 层为例，一次完整的前向传播需要 160 次 AllReduce 操作。在 NVLink 互联的单机 8 卡 A100 环境下，AllReduce 延迟通常在数十微秒级别，160 次累计约几毫秒，对端到端延迟影响不大。但一旦 TP 跨越节点边界（如通过 InfiniBand 或以太网互联），每次 AllReduce 的延迟会增加一个数量级，累计起来会成为严重的性能瓶颈。

因此，**TP 最适合在同一台机器内的高速互联 GPU 之间使用**。如果模型大到单机无法容纳，应优先考虑与 Pipeline Parallelism 结合——这正是下一章将要深入探讨的话题。

## 本章小结

Tensor Parallelism 是 vLLM 支撑大模型推理的基石能力。通过 Megatron 风格的 Column/Row Parallel 矩阵切分方案，配合高效的 AllReduce 集合通信，vLLM 能够将单个 Transformer 层的计算负载和显存占用均匀分摊到多块 GPU 上。`vllm/model_executor/layers/linear.py` 中的 `ColumnParallelLinear` 和 `RowParallelLinear` 是这一机制的核心抽象，`vocab_parallel_embedding.py` 负责词表的分布式嵌入，而 `vllm/distributed/parallel_state.py` 则统一管理全局的并行分组状态。理解 TP 的矩阵切分逻辑和通信代价模型，是掌握 vLLM 分布式推理架构的第一步，也是后续理解更复杂的混合并行策略的基础。
