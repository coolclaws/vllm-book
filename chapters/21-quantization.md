# 第 21 章：量化支持

> "Precision is not always a virtue; sometimes the art lies in knowing how much to let go." —— 在 LLM 推理中，用更少的比特表示权重和激活值，既是工程上的妥协，也是一种精妙的平衡艺术。

## 21.1 量化的动机

大语言模型的参数量动辄数十亿乃至数千亿，以 FP16 精度加载一个 70B 参数的模型就需要约 140GB 显存——这已经超出了单张 GPU 的容量。即便模型能够装入显存，KV Cache 的额外开销也会严重压缩可服务的并发请求数。

量化（Quantization）通过降低数值精度来缓解这一矛盾。其核心思路是：将原本以 FP16 或 BF16 存储的权重和/或激活值，压缩到更低的比特宽度（如 INT4、INT8 或 FP8）。这带来了两个直接好处：

- **显存占用大幅降低**：4-bit 量化可将模型体积缩减至原来的 1/4，使 70B 模型可在单张 80GB GPU 上运行
- **推理吞吐提升**：更小的数据搬运量意味着更低的 memory bandwidth 瓶颈，decode 阶段的吞吐量可获得显著提升

当然，量化并非没有代价——精度损失可能导致生成质量下降。不同的量化方法在压缩率与精度保持之间做出了不同的权衡。

## 21.2 量化架构总览

vLLM 的量化实现集中在 `vllm/model_executor/layers/quantization/` 目录下。该目录包含二十余种量化方法的实现，每种方法对应一个独立的 Python 文件。整体架构遵循策略模式（Strategy Pattern）：

```
QuantizationConfig (base_config.py)       # 抽象基类
  ├── AWQConfig (awq.py)                  # AWQ 量化配置
  ├── GPTQConfig (gptq.py)               # GPTQ 量化配置
  ├── Fp8Config (fp8.py)                  # FP8 量化配置
  ├── GGUFConfig (gguf.py)               # GGUF 格式支持
  ├── GPTQMarlinConfig (gptq_marlin.py)  # GPTQ + Marlin 内核
  ├── AWQMarlinConfig (awq_marlin.py)    # AWQ + Marlin 内核
  ├── BitsAndBytesConfig (bitsandbytes.py)
  └── ...更多方法
```

`QuantizationConfig` 基类定义在 `vllm/model_executor/layers/quantization/base_config.py` 中，声明了四个关键抽象方法：

- **`get_name()`**：返回量化方法的标识名称（如 `"awq"`、`"fp8"`）
- **`get_supported_act_dtypes()`**：返回该方法支持的激活值数据类型列表
- **`get_min_capability()`**：返回所需的最低 GPU 计算能力（如 Ampere 架构需要 80）
- **`get_quant_method(layer, prefix)`**：最核心的方法，为给定的网络层返回具体的量化处理器（`QuantizeMethodBase`），若该层不支持量化则返回 `None`

所有量化方法在 `vllm/model_executor/layers/quantization/__init__.py` 中统一注册，形成一个方法名到配置类的映射表。用户通过 `EngineArgs` 的 `quantization` 参数指定量化方法名称，引擎即可自动查找并加载对应的配置。

## 21.3 主流量化方法详解

### AWQ（Activation-aware Weight Quantization）

AWQ 是一种 4-bit 仅权重量化方法（weight-only quantization）。其核心观察是：权重中存在少量"显著"通道（salient channels），对模型输出影响极大。AWQ 通过保护这些显著通道来在 4-bit 精度下最大限度地保持模型质量。

实现位于 `vllm/model_executor/layers/quantization/awq.py`，其中 `AWQLinearMethod` 负责 Linear 层的量化推理。权重以 INT4 packed 格式存储（每个 INT32 中打包 8 个 4-bit 权重），推理时通过融合的反量化 + 矩阵乘法内核（fused dequant+matmul kernel）完成计算。vLLM 还提供了基于 Marlin 内核的加速版本 `AWQMarlinConfig`（`awq_marlin.py`），在支持的硬件上可进一步提升性能。

### GPTQ（Generative Pre-trained Transformer Quantization）

GPTQ 同样是 4-bit 权重量化，但采用了不同的量化策略——逐层校准（layer-wise calibration）。它使用少量校准数据，通过近似二阶信息（Hessian 矩阵）来最小化量化误差。

实现位于 `vllm/model_executor/layers/quantization/gptq.py`，`GPTQLinearMethod` 提供基础的 GPTQ 推理支持。在实际部署中，vLLM 优先使用 Marlin 内核加速的版本 `GPTQMarlinConfig`（`gptq_marlin.py`），它利用高度优化的 CUDA 内核来执行 4-bit 反量化与矩阵乘法，性能显著优于朴素实现。此外还有基于 ExLlama 内核的变体。

### FP8（8-bit 浮点量化）

FP8 量化代表了新一代 GPU（如 NVIDIA H100、H200）上的原生低精度支持。不同于 INT4 的仅权重量化，FP8 支持 W8A8 模式——权重和激活值均以 8-bit 浮点表示，实现了计算和访存的双重加速。

`Fp8Config`（`vllm/model_executor/layers/quantization/fp8.py`）的关键配置包括：

- **`is_checkpoint_fp8_serialized`**：模型权重是否已以 FP8 格式存储
- **`activation_scheme`**：激活值量化策略，`"dynamic"` 表示运行时动态计算缩放因子，`"static"` 表示使用预校准的固定缩放因子
- **`weight_block_size`**：可选的分块量化维度，支持更细粒度的缩放

`Fp8LinearMethod.apply()` 方法根据配置选择不同的计算路径：分块 FP8（`W8A8BlockFp8LinearOp`）、per-tensor FP8 线性层、或通过 Marlin 内核执行的仅权重 FP8 量化。

### GGUF 格式

GGUF 是 llama.cpp 社区广泛使用的模型格式，支持多种量化级别（Q2_K 到 Q8_0 等）。vLLM 通过 `GGUFConfig`（`vllm/model_executor/layers/quantization/gguf.py`）提供了对 GGUF 格式模型的直接加载支持，使得社区中大量现有的量化模型无需转换即可在 vLLM 中运行。

## 21.4 量化的 KV Cache

除了对模型权重进行量化，vLLM 还支持对 KV Cache 进行 FP8 量化。这一特性通过 `kv_cache_dtype` 参数控制，支持的值包括 `"auto"`（跟随模型精度）、`"fp8_e5m2"` 和 `"fp8_e4m3"`。

KV Cache 量化的实现位于 `vllm/model_executor/layers/quantization/kv_cache.py`。其核心逻辑是在 attention 计算前将 Key 和 Value 张量从高精度量化到 FP8 存入缓存，读取时再反量化回原精度。这样做可以将 KV Cache 的显存占用减半（从 FP16 的 2 字节/元素降至 FP8 的 1 字节/元素），从而支持更大的批处理规模或更长的上下文长度。

FP8 KV Cache 带来的精度影响通常很小，因为 attention 的 softmax 操作对 KV 值的微小扰动不太敏感。对于长上下文场景，这一优化的收益尤为显著。

## 21.5 模型加载与权重处理

量化模型的加载流程与标准模型有所不同。当 `quantization` 参数被指定时，引擎会执行以下步骤：

1. **配置解析**：根据量化方法名称从注册表中找到对应的 `QuantizationConfig` 子类
2. **权重映射**：量化模型的权重文件中通常包含 packed weights（压缩后的权重）、scales（缩放因子）和 zeros（零点）等额外张量
3. **层替换**：模型中的标准 `Linear` 层被替换为量化版本，`get_quant_method()` 为每一层分配合适的量化处理器
4. **权重解包**：加载时对 packed 权重进行解包或格式转换，使其适配 vLLM 的计算内核

对于 FP8 模型，若 checkpoint 本身已是 FP8 格式（`is_checkpoint_fp8_serialized=True`），则权重可以直接加载，无需额外的量化步骤。

## 21.6 使用量化模型

在 vLLM 中使用量化模型非常简单。对于离线推理：

```python
from vllm import LLM

# 使用预量化的 AWQ 模型
llm = LLM(model="TheBloke/Llama-2-7B-Chat-AWQ", quantization="awq")

# 使用 FP8 量化
llm = LLM(model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
           quantization="fp8")

# 启用 FP8 KV Cache
llm = LLM(model="meta-llama/Llama-3-8B", kv_cache_dtype="fp8_e5m2")
```

对于 API Server 启动，通过命令行参数指定：

```bash
vllm serve model_name --quantization awq
vllm serve model_name --kv-cache-dtype fp8_e5m2
```

选择量化方法时需要考虑几个因素：硬件支持（FP8 需要 H100 及以上）、模型可用性（是否有对应格式的预量化模型）、以及精度与性能的平衡。一般来说，FP8 在新硬件上是首选，AWQ 和 GPTQ 在 4-bit 场景下各有优势，GGUF 则适合直接使用社区已有的量化模型。

## 本章小结

量化是突破 GPU 显存瓶颈、提升推理效率的关键技术。vLLM 通过策略模式构建了灵活的量化框架，以 `QuantizationConfig` 基类和 `get_quant_method()` 方法为核心，支持了从 4-bit 整数量化（AWQ、GPTQ）到 8-bit 浮点量化（FP8）再到社区格式（GGUF）的广泛方法。KV Cache 的 FP8 量化进一步扩展了优化空间。整个量化系统的实现集中在 `vllm/model_executor/layers/quantization/` 目录下，遵循统一的抽象接口，使得新量化方法的接入变得规范而便捷。理解量化机制，对于在实际部署中平衡成本、性能和质量至关重要。
