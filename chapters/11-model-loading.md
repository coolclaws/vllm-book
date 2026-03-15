# 第 11 章：模型加载与权重

> "Programs must be written for people to read, and only incidentally for machines to execute." —— Harold Abelson。模型加载看似只是"读文件"，但在 vLLM 中，它是连接 HuggingFace 生态与高性能推理引擎的关键桥梁。

## 11.1 模型加载的全景视图

当用户通过 `LLM("meta-llama/Llama-3-8B")` 启动一个推理服务时，vLLM 需要完成一系列复杂的准备工作：识别模型架构、下载或定位权重文件、将权重映射到 vLLM 的内部模型结构、处理量化格式、在多 GPU 场景下正确分片（shard）参数，最终将模型放置到 GPU 显存中。这整个流程的核心代码集中在 `vllm/model_executor/model_loader/` 目录下。

模型加载的入口位于 `vllm/model_executor/model_loader/__init__.py`，其中定义了顶层的加载函数，根据配置选择合适的 loader 实现。每种 loader 对应一种权重来源或格式，体现了策略模式（Strategy Pattern）的经典应用。

## 11.2 模型注册表：架构到实现的映射

vLLM 支持数十种模型架构，从 LLaMA、Mistral、Qwen 到 ChatGLM、Baichuan 等。这些实现分布在 `vllm/model_executor/models/` 目录下，每个文件对应一种或一系列模型架构。例如 `chatglm.py` 实现了 ChatGLM 系列，`baichuan.py` 实现了 Baichuan 系列。

`vllm/model_executor/models/__init__.py` 扮演着注册表（registry）的角色，维护了从 HuggingFace 模型的 `architectures` 字段到 vLLM 模型类的映射关系。当加载模型时，vLLM 读取 `config.json` 中的 `architectures`（如 `"LlamaForCausalLM"`），然后在注册表中查找对应的实现类。这一设计使得添加新模型支持变得规范化——只需实现模型类并在注册表中登记即可。

## 11.3 加载器的策略体系

`vllm/model_executor/model_loader/` 下的多个 loader 文件构成了一个灵活的加载策略体系：

**DefaultLoader**（`default_loader.py`）是最常用的加载路径，负责从 HuggingFace Hub 或本地目录加载标准格式的模型权重。它调用 `weight_utils.py` 中的工具函数来处理 safetensors 和 PyTorch bin 格式的权重文件，逐层加载并映射到 vLLM 模型的参数中。

**ShardedStateLoader**（`sharded_state_loader.py`）处理已经预先分片的权重检查点。在 Tensor Parallelism 场景下，如果每次启动都要重新切分权重，会浪费大量时间。ShardedStateLoader 允许用户预先将权重按并行策略切分保存，启动时直接加载对应分片，大幅缩短冷启动时间。

**BitsAndBytesLoader**（`bitsandbytes_loader.py`）专门处理通过 bitsandbytes 库量化的模型，支持 INT8 和 NF4 等格式。**GGUFLoader**（`gguf_loader.py`）则处理来自 llama.cpp 生态的 GGUF 格式权重文件，使 vLLM 能够加载社区中大量以 GGUF 格式分发的量化模型。

此外还有 `tensorizer_loader.py` 支持通过 CoreWeave 的 Tensorizer 格式加速加载，`runai_streamer_loader.py` 支持流式加载，以及 `dummy_loader.py` 用于测试场景下快速构建占位模型。

## 11.4 权重转换与键映射

HuggingFace 模型和 vLLM 内部模型的参数命名往往不同。例如，HuggingFace 的 LLaMA 实现中注意力层的参数名可能是 `model.layers.0.self_attn.q_proj.weight`，而 vLLM 的实现可能将 Q、K、V 三个投影合并为一个参数 `qkv_proj`。

`weight_utils.py` 中的工具函数负责处理这种映射关系。每个模型实现类通常会定义一个 `load_weights()` 方法，在其中描述如何将 HuggingFace 的 state_dict 键转换为自身的参数结构。对于常见的合并模式——如将 `q_proj`、`k_proj`、`v_proj` 合并为 `qkv_proj`，或将 `gate_proj` 和 `up_proj` 合并为 `gate_up_proj`——vLLM 提供了标准化的工具来简化实现。

## 11.5 数据类型处理

vLLM 支持多种数据类型（dtype）：`float16`、`bfloat16`、`float32`，以及自动检测模式。自动检测会读取模型配置中的 `torch_dtype` 字段来决定精度。在实际部署中，`float16` 和 `bfloat16` 是最常用的选择——前者兼容性更广，后者在 Ampere 及以后的 GPU 上有更好的数值稳定性。

dtype 的选择不仅影响精度，还直接决定了显存占用。一个 7B 参数的模型，float16 下占用约 14GB，float32 则需要 28GB。因此正确的 dtype 配置对于最大化 KV Cache 的可用空间至关重要。

## 11.6 量化模型加载

对于量化模型，vLLM 在 `vllm/model_executor/layers/quantization/` 目录下实现了丰富的量化方案支持。主要包括：

- **AWQ**：逐通道的 4-bit 量化，保留重要权重通道的精度
- **GPTQ**：基于二阶信息的 4-bit 量化，精度损失小
- **FP8**：8-bit 浮点量化，在 Hopper 架构上有原生硬件支持

量化模型的加载流程与标准模型类似，但在 `vllm/model_executor/layers/linear.py` 等层实现中，会根据量化配置替换标准的线性层为对应的量化版本。量化层在推理时使用专门的 kernel 进行反量化和计算，在显著降低显存占用的同时维持可接受的推理精度。

## 11.7 层实现与显存预算

`vllm/model_executor/layers/` 目录包含了 vLLM 对各类神经网络层的实现。`linear.py` 实现了支持 Tensor Parallelism 的线性层（如 `ColumnParallelLinear` 和 `RowParallelLinear`），`layernorm.py` 实现了 RMSNorm 等归一化层，`rotary_embedding/` 目录实现了 RoPE 位置编码的多种变体，`vocab_parallel_embedding.py` 则实现了词表并行的 Embedding 层。

模型加载完成后，vLLM 需要确定还有多少显存可以用于 KV Cache。GPU Model Runner 中的 `determine_num_available_blocks()` 方法会执行一次显存分析（memory profiling）：它先将模型加载到 GPU，执行一次虚拟的前向传播以测量峰值显存占用，然后用 GPU 总显存减去模型占用和运行时开销，将剩余空间全部分配给 KV Cache 的物理块池。这个过程直接决定了系统能够并发处理的序列数量和最大上下文长度。

## 本章小结

模型加载是 vLLM 从静态权重文件到动态推理引擎的关键转换过程。vLLM 通过模型注册表将 HuggingFace 的模型架构映射到内部实现，通过策略化的 loader 体系支持多种权重来源和格式，通过精细的权重转换和 dtype 处理确保加载的正确性，通过量化支持降低显存门槛，最终通过显存分析为 KV Cache 分配最大可用空间。理解这一流程，是理解 vLLM 如何将一个预训练模型高效部署为在线服务的基础。
