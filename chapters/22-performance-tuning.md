# 第 22 章：性能调优与 Profiling

> "Premature optimization is the root of all evil, but late optimization is the root of all failure." —— 在 LLM 推理系统中，性能调优既需要对底层机制的深刻理解，也需要系统化的度量手段。

## 22.1 CUDA Graph 优化

LLM 推理的 decode 阶段有一个显著特征：每次迭代只为每个序列生成一个 token，GPU 上执行的计算量很小，但 CUDA kernel launch 的开销（包括 CPU 端的调度、参数传递等）却是固定的。当 kernel 数量多而每个 kernel 的执行时间短时，launch overhead 就成为瓶颈。

CUDA Graph 正是为解决这一问题而生的技术。它的原理是：首先"录制"一次完整的 GPU 执行流程（包括所有 kernel 调用及其参数），然后在后续推理中直接"回放"录制好的计算图，绕过逐个 kernel 的 launch 开销。

在 vLLM 中，CUDA Graph 的管理通过 `CudagraphDispatcher` 和 `CUDAGraphWrapper` 实现，在 `vllm/v1/worker/gpu_model_runner.py` 中被初始化和使用。关键流程如下：

1. **预热阶段**：模型首次加载后，vLLM 会针对一组预设的 batch size 分别录制 CUDA Graph。这些 batch size 由 `compilation_config.cudagraph_capture_sizes` 配置
2. **录制过程**：`_build_attention_metadata()` 方法在 `for_cudagraph_capture=True` 模式下构建专用的 attention metadata，确保录制时的内存布局与回放时一致
3. **运行时分发**：`CudagraphDispatcher` 根据当前 batch 的实际大小，选择最接近的已录制图进行回放。若无匹配的图，则回退到 eager 模式执行

CUDA Graph 的主要限制是要求固定的输入形状——这正是为什么需要为不同 batch size 分别录制。动态控制流（如条件分支）也无法在图中表示。通过 `enforce_eager=True` 参数可以禁用 CUDA Graph，这在调试阶段非常有用。

## 22.2 torch.compile 集成

除了 CUDA Graph，vLLM 还集成了 PyTorch 2.x 的 `torch.compile` 功能来进一步优化模型前向传播。`torch.compile` 通过 TorchInductor 后端将 Python 模型代码编译为优化的 GPU 内核，实现：

- **算子融合（Kernel Fusion）**：将多个小算子合并为一个大内核，减少中间张量的内存读写
- **Python 开销消除**：编译后的代码不再经过 Python 解释器，消除了 frame evaluation 开销
- **自动调优**：TorchInductor 会为特定硬件和输入形状选择最优的内核实现

编译配置通过 `VllmConfig` 中的 `compilation_config` 传递。`vllm/v1/worker/gpu_model_runner.py` 中的 `compilation_config.static_forward_context` 存储了 attention 层的编译上下文，而 `compilation_counter` 用于追踪编译进度。

需要注意的是，首次编译会引入额外的启动延迟（数秒到数十秒），但后续推理会因此受益。在生产环境中，这一一次性开销通常是值得的。

## 22.3 关键性能参数

vLLM 提供了多个可调参数来适配不同的部署场景。理解这些参数的含义和影响是性能调优的基础。

### 内存相关参数

**`gpu_memory_utilization`**（默认 0.9）：控制 vLLM 使用的 GPU 显存比例。该值决定了分配给 KV Cache 的显存量。提高此值可以缓存更多 token，支持更大的并发量，但需要为其他 GPU 用途留有余量。计算逻辑位于 Worker 的 `determine_num_available_blocks()` 方法中：先通过 profile run 测量模型本身的显存开销，再将剩余显存的 `gpu_memory_utilization` 比例分配给 KV Cache。

**`block_size`**（可选值 8、16、32）：KV Cache 块的大小，即每个物理块存储的 token 数量。较大的块减少了 block table 的管理开销但可能增加内部碎片。默认值通常能获得良好的平衡。

**`kv_cache_dtype`**：KV Cache 的数据类型。设为 `"fp8_e5m2"` 或 `"fp8_e4m3"` 可将缓存显存减半，从而支持更多并发或更长上下文。

### 批处理相关参数

**`max_num_batched_tokens`**：每次调度迭代处理的最大 token 数。这是控制 prefill 阶段批处理规模的核心参数。较大的值提升吞吐量（GPU 利用率更高），但会增加单次迭代的延迟。对于延迟敏感的场景，适当降低此值并启用 chunked prefill 可以获得更好的 TTFT。

**`max_num_seqs`**：最大并发序列数。限制了同时处理的请求数量。增大此值可提高吞吐但也增加了调度和内存管理的开销。

### 执行相关参数

**`enforce_eager`**（默认 `False`）：设为 `True` 时禁用 CUDA Graph，所有推理使用 eager mode 执行。主要用于调试，或当 CUDA Graph 与某些模型/特性不兼容时。

**`enable_chunked_prefill`**：启用分块预填充。长 prompt 会被拆分为多个 chunk 逐步处理，避免长 prefill 阻塞 decode 请求。这对延迟敏感的在线服务至关重要。

## 22.4 基准测试工具

vLLM 仓库的 `benchmarks/` 目录提供了三个核心基准测试脚本：

**`benchmarks/benchmark_throughput.py`**——吞吐量测试。衡量离线批处理场景下的 token 生成速率（tokens/second）。这是评估系统最大处理能力的主要指标。典型用法：

```bash
python benchmarks/benchmark_throughput.py \
    --model meta-llama/Llama-3-8B \
    --input-len 512 --output-len 128 \
    --num-prompts 1000
```

**`benchmarks/benchmark_latency.py`**——延迟测试。测量单次推理的端到端延迟，包括 TTFT（Time To First Token，首 token 延迟）和 TPOT（Time Per Output Token，每 token 生成时间）。适合评估交互式场景的响应速度。

**`benchmarks/benchmark_serving.py`**——服务化测试。模拟真实的在线服务负载，按照指定的请求到达率（QPS）发送请求，测量 P50/P95/P99 延迟分位数和整体吞吐。这是最接近生产场景的测试方式。

## 22.5 Profiling 技术

当基准测试表明性能不达预期时，需要深入 profiling 来定位瓶颈。

### PyTorch Profiler

`torch.profiler` 提供了对 Python 层面和 CUDA kernel 的联合分析能力。vLLM 内部支持集成 `torch.profiler`，可以记录每次推理迭代中各个 kernel 的执行时间、GPU 利用率和内存分配情况。生成的 trace 文件可以用 Chrome Trace Viewer 或 TensorBoard 打开进行可视化分析。

### NVIDIA Nsight 工具

对于更底层的分析，NVIDIA 的 Nsight Systems 和 Nsight Compute 是不可或缺的工具：

- **Nsight Systems**：提供系统级时间线视图，展示 CPU-GPU 交互、kernel 排队等待、内存拷贝等全局信息。适合识别 CPU-GPU 同步瓶颈和 kernel launch overhead
- **Nsight Compute**：提供单个 CUDA kernel 的深度分析，包括 occupancy、memory throughput、compute throughput 等指标。适合优化特定的 attention 或 GEMM kernel

### 内存分析

`determine_num_available_blocks()` 方法是理解显存分配的关键入口。它通过以下步骤确定可用的 KV Cache 块数：执行一次 profile run 记录模型的显存峰值，计算剩余可用显存，再除以每个块的显存开销得到块数。当遇到 OOM 问题时，降低 `gpu_memory_utilization` 或减小 `max_num_batched_tokens` 是最直接的缓解手段。

## 22.6 常见优化模式

基于实际部署经验，以下是几种常见的优化策略：

**选择合适的 Tensor Parallelism 度数**：TP 度数并非越大越好。每增加一张 GPU，通信开销也随之增加。经验法则是：模型恰好能装入当前 GPU 数量时，TP 度数最优。例如 70B FP16 模型适合 TP=4（4×80GB），而 8B 模型单卡即可，使用 TP 反而增加不必要的通信。

**合理设置 max_num_batched_tokens**：对于吞吐优先的离线场景，增大该值（如 8192 或更高）以充分利用 GPU 算力；对于延迟敏感的在线服务，适度降低并启用 chunked prefill 以控制 TTFT。

**启用 FP8 KV Cache**：在 H100 等支持 FP8 的硬件上，使用 `kv_cache_dtype="fp8_e5m2"` 可以以极小的精度代价换取翻倍的缓存容量，这在长上下文或高并发场景中效果显著。

**Chunked Prefill 与 Decode 的平衡**：启用 `enable_chunked_prefill` 后，长 prompt 的 prefill 会被拆分为多个 chunk 与 decode 请求交替执行，避免长 prefill 造成的延迟尖峰。这对 SLA 严格的在线服务是必要的优化。

## 本章小结

性能调优是将 vLLM 的理论优势转化为实际生产收益的关键环节。CUDA Graph 消除了 decode 阶段的 kernel launch 开销，`torch.compile` 通过算子融合进一步提升效率。`gpu_memory_utilization`、`max_num_batched_tokens`、`max_num_seqs` 等参数提供了灵活的性能-延迟权衡空间。`benchmarks/` 目录下的三个基准测试脚本覆盖了吞吐、延迟和服务化三大测试维度，配合 PyTorch Profiler 和 NVIDIA Nsight 工具可以深入定位性能瓶颈。掌握这些工具和参数，是将 vLLM 成功部署到生产环境的必备技能。
