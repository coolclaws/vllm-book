# 附录 B：核心类型速查

本附录汇总了 vLLM 中最常用的配置类和核心类型，方便在阅读源码或进行开发时快速查阅。所有类型信息基于 vLLM 主分支，具体字段可能随版本演进有所变化。

---

## B.1 EngineArgs

`EngineArgs` 是 vLLM 引擎的统一配置入口，定义在 `vllm/engine/arg_utils.py` 中。它将用户传入的参数解析并分发到各个子配置（`ModelConfig`、`CacheConfig`、`ParallelConfig` 等）。

### 模型相关参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `str` | — | 模型名称或路径，如 `"meta-llama/Llama-3-8B"` |
| `tokenizer` | `str \| None` | `None` | tokenizer 名称或路径，默认与 model 相同 |
| `dtype` | `ModelDType` | `"auto"` | 模型权重精度，可选 `"auto"`、`"float16"`、`"bfloat16"`、`"float32"` |
| `quantization` | `str \| None` | `None` | 量化方法名称，如 `"awq"`、`"gptq"`、`"fp8"` |
| `max_model_len` | `int \| None` | `None` | 最大序列长度（含 prompt + generation），默认从模型配置读取 |
| `trust_remote_code` | `bool` | `False` | 是否信任远程模型代码 |
| `tokenizer_mode` | `str` | `"auto"` | tokenizer 加载模式，可选 `"auto"`、`"slow"` |

### 并行与分布式参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tensor_parallel_size` | `int` | `1` | Tensor Parallelism 度数（TP） |
| `pipeline_parallel_size` | `int` | `1` | Pipeline Parallelism 度数（PP） |

### 缓存与内存参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gpu_memory_utilization` | `float` | `0.9` | GPU 显存使用比例，用于 KV Cache 分配 |
| `block_size` | `int \| None` | `None` | KV Cache 块大小（token 数），通常为 16 |
| `kv_cache_dtype` | `str` | `"auto"` | KV Cache 数据类型，可选 `"auto"`、`"fp8_e5m2"`、`"fp8_e4m3"` |

### 调度与批处理参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_num_batched_tokens` | `int \| None` | `None` | 每次迭代最大处理 token 数 |
| `max_num_seqs` | `int \| None` | `None` | 最大并发序列数 |
| `enable_chunked_prefill` | `bool \| None` | `None` | 是否启用分块预填充 |

### 执行相关参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enforce_eager` | `bool` | `False` | 强制使用 eager mode，禁用 CUDA Graph |

---

## B.2 SamplingParams

`SamplingParams` 定义在 `vllm/sampling_params.py` 中，控制文本生成的采样行为。每个推理请求都携带一份 `SamplingParams` 实例。

### 核心采样参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | `float` | `1.0` | 采样温度。0 表示贪心解码，越大越随机 |
| `top_p` | `float` | `1.0` | Nucleus sampling 的概率阈值 |
| `top_k` | `int` | `0` | Top-K sampling 的 K 值，0 表示不启用 |
| `n` | `int` | `1` | 每个 prompt 生成的序列数 |
| `seed` | `int \| None` | `None` | 随机种子，用于可复现的采样 |

### 长度控制参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_tokens` | `int \| None` | `16` | 最大生成 token 数 |
| `min_tokens` | `int` | `0` | 最小生成 token 数，在此之前不会生成 EOS |
| `stop` | `str \| list[str] \| None` | `None` | 停止字符串列表，匹配时终止生成 |

### 惩罚参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `presence_penalty` | `float` | `0.0` | 存在惩罚，鼓励新主题。范围 [-2.0, 2.0] |
| `frequency_penalty` | `float` | `0.0` | 频率惩罚，抑制高频词。范围 [-2.0, 2.0] |
| `repetition_penalty` | `float` | `1.0` | 重复惩罚系数。1.0 表示无惩罚，>1.0 抑制重复 |

### 输出控制参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `logprobs` | `int \| None` | `None` | 返回每个生成 token 的 top-N logprobs |
| `prompt_logprobs` | `int \| None` | `None` | 返回 prompt token 的 top-N logprobs |
| `skip_special_tokens` | `bool` | `True` | 输出中是否跳过特殊 token |
| `spaces_between_special_tokens` | `bool` | `True` | 特殊 token 之间是否添加空格 |

---

## B.3 SchedulerConfig

`SchedulerConfig` 控制调度器的行为，通常由 `EngineArgs` 生成。核心字段定义在 `vllm/config.py` 中。

| 字段 | 类型 | 说明 |
|------|------|------|
| `max_num_batched_tokens` | `int` | 每次调度迭代的 token 预算上限 |
| `max_num_seqs` | `int` | 每次调度迭代的序列预算上限 |
| `max_model_len` | `int` | 模型支持的最大序列长度 |

调度器在每次迭代中，受到 token budget（`max_num_batched_tokens`）和 sequence budget（`max_num_seqs`）的双重约束。详见第 4 章。

---

## B.4 核心枚举类型

### SequenceStatus

定义序列在其生命周期中的状态，相关逻辑贯穿调度器和 Worker。

| 枚举值 | 含义 |
|--------|------|
| `WAITING` | 在等待队列中，尚未开始推理 |
| `RUNNING` | 正在 GPU 上执行推理 |
| `SWAPPED` | 被抢占，KV Cache 已换出到 CPU |
| `FINISHED_STOPPED` | 生成了停止 token 或匹配了 stop 字符串 |
| `FINISHED_LENGTH_CAPPED` | 达到最大生成长度 |
| `FINISHED_ABORTED` | 被客户端取消或系统中止 |
| `FINISHED_IGNORED` | 因 prompt 过长等原因被跳过 |

### PreemptionMode

定义调度器在显存不足时的抢占策略。

| 枚举值 | 含义 |
|--------|------|
| `SWAP` | 将被抢占序列的 KV Cache 从 GPU 换出到 CPU 内存 |
| `RECOMPUTE` | 丢弃被抢占序列的 KV Cache，后续重新计算（重算 prefill） |

`SWAP` 模式需要足够的 CPU 内存和 PCIe 带宽，`RECOMPUTE` 模式则以重复计算换取更低的内存需求。调度器根据可用资源自动选择策略。详见第 4 章和第 5 章。

---

## B.5 其他重要类型

| 类型 | 定义位置 | 说明 |
|------|----------|------|
| `VllmConfig` | `vllm/config.py` | 顶层配置容器，聚合所有子配置 |
| `ModelConfig` | `vllm/config.py` | 模型相关配置（名称、精度、最大长度等） |
| `CacheConfig` | `vllm/config.py` | 缓存相关配置（块大小、显存比例、KV 精度等） |
| `ParallelConfig` | `vllm/config.py` | 并行相关配置（TP、PP 度数等） |
| `LoRAConfig` | `vllm/config.py` | LoRA 适配器相关配置 |
| `CompilationConfig` | `vllm/config.py` | 编译与 CUDA Graph 相关配置 |
