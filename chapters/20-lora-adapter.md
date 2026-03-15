# 第 20 章：LoRA 与多 Adapter 服务

> "不要为每位顾客建一座工厂，而是在同一条产线上快速切换模具。" LoRA 让 vLLM 用一份基座模型权重，同时为数十个微调版本提供服务。

## 20.1 LoRA 原理回顾

LoRA（Low-Rank Adaptation）的核心思想极为简洁而优雅：冻结预训练模型的原始权重矩阵 `W`，在其旁边注入两个低秩矩阵 `A`（降维矩阵）和 `B`（升维矩阵），使得前向传播的计算变为：

```
Y = X @ W + X @ A @ B
```

其中 `A` 的形状是 `(d, r)`，`B` 的形状是 `(r, d)`，`r` 是 LoRA 的秩（rank），它远小于原始维度 `d`（典型值为 8 到 64）。这意味着一个 LoRA adapter 的可训练参数量仅为原始权重的千分之一到百分之一——但实验表明，这些少量参数却能够有效地捕获特定任务的知识和风格偏好，在各种下游任务上达到接近全量微调的效果。

对于推理服务而言，LoRA 的真正价值在于经济性：一台机器只需在显存中加载一份完整的基座模型权重（往往占据数十 GB 显存），就能通过动态加载和切换极其轻量的 LoRA adapter 来为数十个不同的微调版本提供服务。相比为每个微调版本各部署一个完整的模型实例，这种方案在显存占用和硬件成本上节省了数十倍之多。

## 20.2 vLLM 的 LoRA 实现架构

vLLM 的 LoRA 功能实现代码位于 `vllm/lora/` 目录下，架构设计清晰地分为若干层次，每个核心模块承担明确的职责：

- **`model_manager.py`**：这是 LoRA 系统的中枢管理器，负责 adapter 的完整生命周期管理——包括从磁盘或远程仓库加载 adapter 权重、在 GPU 显存和 CPU 内存之间调度 adapter 的驻留策略、以及根据实时的请求需求决定哪些 adapter 应该驻留在 GPU 显存中以备即时使用、哪些可以暂时驱逐到 CPU 内存中等待后续按需重新加载。它内部维护了一个 adapter 资源池，使用类似 LRU 缓存的策略来管理有限的 GPU 显存空间。
- **`lora_model.py`**：LoRA 模型的封装层，负责将基座模型的各个目标线性层（通常是 Attention 的 Q、K、V、O 投影和 MLP 的 gate、up、down 投影）与对应的 LoRA 低秩权重矩阵关联起来，形成完整的"基座模型 + LoRA 增强"的前向传播路径。
- **`lora_weights.py`**：处理 LoRA 权重的加载、格式转换和数据类型适配。它支持从 HuggingFace PEFT 格式（目前最流行的 LoRA 权重发布格式）直接读取 adapter 文件，并将其转换为 vLLM 内部计算所需的张量格式和数据精度。
- **`layers/`**：LoRA 化的神经网络层实现目录，包含对 `ColumnParallelLinear`、`RowParallelLinear` 以及各种合并变体层的 LoRA 扩展。这些 LoRA 层在基座层的 `forward()` 方法中注入了额外的低秩矩阵乘法计算。
- **`worker_manager.py`**：在分布式多 Worker 部署场景下，管理每个 Worker 进程上的 LoRA adapter 加载状态，确保所有 TP rank 上的 adapter 保持同步一致。
- **`peft_helper.py`**：与 HuggingFace PEFT 库的兼容适配层，负责解析 PEFT 格式的 adapter 配置文件（`adapter_config.json`），提取 LoRA 的目标模块列表、秩大小、alpha 缩放因子等关键参数。

## 20.3 LoRA 化的线性层

LoRA 的计算核心在于线性层的 `forward()` 方法中需要注入额外的低秩矩阵乘法。`vllm/lora/layers/` 目录中实现了多种 LoRA 增强的层变体，每种都需要仔细处理与 Tensor Parallelism 的交互。

对于 `ColumnParallelLinear` 的 LoRA 版本，在原始的列切分矩阵乘法 `X @ W_col` 之外，还需额外计算 LoRA 增量 `X @ A @ B` 并累加到输出上。由于列切分要求输出按特征维度分片，LoRA 的 `B` 矩阵（升维矩阵）也必须相应地按列切分——第 `i` 块 GPU 只持有 `B` 的第 `i` 个列分片，确保 LoRA 的输出分片与基座权重的输出分片对齐。而 `A` 矩阵（降维矩阵）接收完整的输入 `X`，在所有 GPU 上保持相同的副本。

对于 `RowParallelLinear` 的 LoRA 版本，情况恰好相反：`A` 矩阵需要按行切分（与输入的 TP 分片对齐），每块 GPU 用自己的输入切片 `X_i` 乘以自己持有的 `A` 的行分片；而 `B` 矩阵则在所有 GPU 上保持完整副本。LoRA 计算的部分结果需要与基座权重的部分结果一同参与后续的 AllReduce 求和操作。

QKV 合并投影层（`QKVParallelLinear`）的 LoRA 化尤为复杂：Q、K、V 三个投影各自有独立的一组 LoRA 权重（各自的 A 和 B 矩阵），但它们的基座权重被合并为一个大矩阵以提升计算效率。LoRA 层需要在合并后的输出张量中精确定位 Q、K、V 各自对应的输出区间，将三组 LoRA 的增量分别累加到正确的位置上。vLLM 通过 `punica_wrapper/` 子目录中的高效 CUDA kernel 来加速这些涉及多组 LoRA 权重的批量矩阵运算。

## 20.4 多 Adapter 批处理

vLLM 在 LoRA 领域最具竞争力的能力是 **Multi-Adapter Batching**（多 adapter 混合批处理）：在同一个推理 batch 中，不同的请求可以指定使用不同的 LoRA adapter，引擎会在单次前向传播中同时处理所有这些请求。这意味着一个推理服务实例可以同时处理"用户 A 要求使用客服风格 adapter 的请求"和"用户 B 要求使用代码生成 adapter 的请求"，无需将不同 adapter 的请求分开排队、等待清空当前 adapter 的所有请求后再切换到下一个 adapter。

实现多 adapter 混合批处理的关键技术在于计算路径的分离与合并。前向传播时，基座模型的矩阵乘法 `X @ W` 对整个 batch 的所有请求统一执行——这一步与 LoRA 无关，所有请求共享同一个基座权重。而 LoRA 增量部分 `X @ A @ B` 则需要根据每个请求关联的 adapter 索引，分别查表取出对应的 A 和 B 矩阵进行计算。`vllm/lora/punica_wrapper/` 中的 Punica CUDA kernel 通过**分组矩阵乘法（grouped GEMM）**技术高效地完成这一操作——它将使用同一个 adapter 的请求聚合在一起进行批量矩阵乘法，然后按请求顺序将各组的计算结果拼接回完整的输出张量。这种方法避免了逐个请求串行执行 LoRA 计算的巨大开销。

用户在 API 请求中指定所需 adapter 的方式非常自然——直接将 adapter 的名称作为模型名传入：

```python
# 使用 OpenAI 兼容 API 指定 LoRA adapter
response = client.chat.completions.create(
    model="my-customer-service-lora",  # adapter 名称作为 model 参数
    messages=[{"role": "user", "content": "请帮我查询订单状态"}]
)
```

在服务启动时，通过 `--lora-modules` 参数注册可用的 adapter 名称及其对应的权重路径。

## 20.5 Adapter 调度与显存管理

GPU 显存容量是有限的，当系统注册了数十甚至上百个 LoRA adapter 时，不可能将它们全部同时加载到 GPU 显存中。`model_manager.py` 实现了一套精细的 adapter 调度和显存管理策略。其核心配置参数包括：

- **`max_loras`**：允许同时驻留在 GPU 显存中的最大 adapter 数量。这直接决定了单次批处理中能混合使用的不同 adapter 上限。
- **`max_cpu_loras`**：允许缓存在 CPU 主存中的最大 adapter 数量。CPU 缓存的 adapter 可以在需要时快速加载到 GPU，延迟远低于从磁盘重新读取。
- **`max_lora_rank`**：系统支持的最大 LoRA 秩。这个参数决定了 GPU 上为 LoRA 权重预分配的缓冲区大小——更大的秩需要更多的显存。

当某个请求指定的 adapter 当前不在 GPU 显存中时，调度器会触发 adapter 切换流程：首先根据 LRU（最近最少使用）策略选择一个当前最不活跃的 adapter 从 GPU 显存中驱逐，释放其占用的权重缓冲区；然后从 CPU 缓存或磁盘中加载目标 adapter 的权重到空出的缓冲区中。这一过程对 API 调用方完全透明，只会带来首次使用某个不常用 adapter 时的轻微额外延迟。

`LoRAConfig` 作为统一的配置入口对象，还包含其他重要参数：`lora_extra_vocab_size` 指定 LoRA adapter 可能引入的额外词汇量（某些 adapter 在微调时会扩展词表）；`lora_dtype` 控制 LoRA 权重的计算精度。

## 20.6 显存开销的定量分析

LoRA 的显存开销之低，是其能够支撑多 adapter 并发服务的根本原因。我们可以做一个具体的定量估算：以 LoRA rank=16、应用于一个 hidden_size=4096 的模型的所有线性层为例，每个 LoRA adapter 在每个线性层上需要存储 A 和 B 两个矩阵的参数：

```
单层每个 adapter：(d × r + r × d) × 2 bytes = (4096 × 16 + 16 × 4096) × 2 ≈ 256 KB
```

一个 32 层的 Transformer 模型中，如果对每层的 Q、K、V、O 四个投影加上 MLP 的 gate、up、down 三个投影（共 7 个线性层）都应用 LoRA，则单个 adapter 的总参数量约为：

```
256 KB × 7 层 × 32 = 约 56 MB
```

这 56 MB 相比基座模型动辄几十 GB 的权重，真正是沧海一粟。因此，即使在 GPU 显存中同时保持 20 个 adapter，也仅增加约 1.1 GB 的显存占用——对于一块 80 GB 的 A100 来说几乎可以忽略不计。这一特性使得一个部署了 70B 基座模型的推理服务，可以在几乎零额外成本的条件下同时为几十个不同的微调版本提供高效服务，真正实现了"一模多用"的经济效益最大化。

## 本章小结

LoRA 让 vLLM 以极低的显存开销实现了多 adapter 并发服务的能力。在架构层面，`vllm/lora/model_manager.py` 统一管理 adapter 的生命周期和 GPU/CPU 多级缓存调度，`vllm/lora/layers/` 目录中的各种 LoRA 化线性层将低秩矩阵计算无缝融入 Tensor Parallelism 并行的前向传播路径，而 `punica_wrapper/` 中的 Punica kernel 通过高效的 grouped GEMM 技术支撑起同一 batch 内多个不同 adapter 的混合并行计算。`lora_weights.py` 和 `peft_helper.py` 则确保了与 HuggingFace PEFT 生态的兼容互通。从工程视角来看，LoRA 多 adapter 服务的核心挑战不在于 LoRA 计算本身的复杂度，而在于 adapter 的动态调度策略和多级显存管理——这正是 `model_manager.py` 和 `worker_manager.py` 所解决的关键问题。
