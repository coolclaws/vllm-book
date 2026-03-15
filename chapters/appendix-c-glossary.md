# 附录 C：名词解释

本附录收录了全书涉及的核心术语，按字母顺序排列。每个条目包含简要定义和相关章节索引，方便快速查阅。

---

**AllReduce**
分布式集合通信操作，将所有参与进程的张量进行规约（如求和）并将结果广播回所有进程。在 Tensor Parallelism 中用于合并各 GPU 上的部分计算结果。vLLM 使用 NCCL 和自定义 AllReduce 内核实现。参见第 16 章、第 18 章。

**Attention Backend**
vLLM 的 attention 计算后端抽象。不同后端实现了相同的 attention 接口但使用不同的底层内核，包括 Flash Attention、FlashInfer 等。用户可通过环境变量选择后端。参见第 7 章。

**AWQ（Activation-aware Weight Quantization）**
一种 4-bit 仅权重量化方法。通过识别并保护对输出影响最大的"显著通道"来在极低比特下保持模型质量。实现位于 `vllm/model_executor/layers/quantization/awq.py`。参见第 21 章。

**Block**
KV Cache 的基本管理单元。每个 block 存储固定数量（如 16 个）token 的 Key 和 Value 向量。Block 是 PagedAttention 内存管理的核心概念，类比操作系统中的物理页框。参见第 5 章、第 6 章。

**Block Table**
从逻辑块编号到物理块编号的映射表，每个序列维护一份。类比操作系统的页表，是 PagedAttention 实现虚拟-物理地址转换的关键数据结构。参见第 5 章。

**CacheEngine**
负责 KV Cache 物理内存的分配和管理的组件。它在 GPU 和 CPU 上分别维护缓存池，并执行块级别的拷贝、换入（swap in）和换出（swap out）操作。参见第 6 章。

**Chunked Prefill**
分块预填充技术。将长 prompt 的 prefill 计算拆分为多个 chunk，与 decode 请求交替执行，避免长 prefill 阻塞其他请求导致延迟尖峰。参见第 13 章。

**Continuous Batching**
连续批处理。不同于传统的静态 batching（等待一批请求全部完成再处理下一批），continuous batching 允许在每次调度迭代中动态加入新请求、移除已完成请求，最大化 GPU 利用率。参见第 4 章。

**Decode**
LLM 推理的自回归生成阶段。每次迭代为每个序列生成一个新 token，使用已缓存的 KV Cache 进行 attention 计算。Decode 阶段是 memory-bound 的，batch size 越大越能提高 GPU 利用率。参见第 3 章。

**Executor**
vLLM 的执行器抽象层，负责管理 Worker 的生命周期和任务分发。不同的 Executor 实现对应不同的部署模式：单进程（`UniProcExecutor`）、多进程（`MultiprocExecutor`）和 Ray 分布式（`RayDistributedExecutor`）。参见第 10 章。

**Flash Attention**
一种 IO-aware 的精确 attention 算法，通过分块计算和 kernel 融合大幅减少 HBM 访问次数，在不牺牲精度的前提下显著加速 attention 计算。vLLM 的多个 attention backend 基于 Flash Attention 实现。参见第 7 章。

**FP8（8-bit Floating Point）**
8 位浮点量化格式，NVIDIA H100 及以上 GPU 原生支持。vLLM 支持 FP8 权重量化（W8A8）和 FP8 KV Cache，分别用于减少模型显存和缓存显存。常见格式有 E5M2（范围大）和 E4M3（精度高）。参见第 21 章。

**GPTQ（Generative Pre-trained Transformer Quantization）**
一种基于二阶信息的 4-bit 权重量化方法。通过逐层校准最小化量化误差，通常需要校准数据集。实现位于 `vllm/model_executor/layers/quantization/gptq.py`。参见第 21 章。

**KV Cache**
Key-Value Cache，存储 LLM 自回归解码过程中历史 token 的 Key 和 Value 向量。避免重复计算是 LLM 高效推理的基础，KV Cache 的管理是 vLLM 的核心关注点。参见第 5 章、第 6 章。

**LoRA（Low-Rank Adaptation）**
一种参数高效微调方法，通过在预训练权重旁添加低秩矩阵来实现模型适配。vLLM 支持在推理时同时服务多个 LoRA 适配器，无需为每个适配器加载完整模型。参见第 15 章。

**ModelRunner**
模型执行器，负责管理模型的加载、输入准备和前向传播。v1 版本的 GPU ModelRunner 定义在 `vllm/v1/worker/gpu_model_runner.py` 中，是 Worker 内部的核心组件。参见第 10 章、第 11 章。

**NCCL（NVIDIA Collective Communications Library）**
NVIDIA 的集合通信库，提供高性能的 GPU 间通信原语（AllReduce、AllGather、Broadcast 等）。vLLM 在分布式推理中依赖 NCCL 进行 GPU 间数据同步。参见第 18 章。

**PagedAttention**
vLLM 的核心创新。借鉴操作系统虚拟内存的分页机制，将 KV Cache 分成固定大小的 block，按需分配，支持非连续存储和 Copy-on-Write 共享。将内存利用率从传统的 20%-40% 提升至接近 100%。参见第 1 章、第 5 章、第 7 章。

**Pipeline Parallelism（PP）**
流水线并行。将模型按层划分为多个 stage，分布在不同 GPU 上，数据以 micro-batch 的形式在 stage 间流水推进。适合模型过大无法通过 Tensor Parallelism 单独解决的场景。参见第 17 章。

**Preemption**
抢占。当 GPU 显存不足以容纳所有运行中的序列时，调度器选择部分序列暂停执行。被抢占的序列的 KV Cache 被换出到 CPU（SWAP 模式）或丢弃后续重算（RECOMPUTE 模式）。参见第 4 章、第 5 章。

**Prefill**
预填充阶段。处理输入 prompt 中所有 token 的 attention 计算，生成初始 KV Cache。Prefill 是 compute-bound 的，决定了 TTFT（首 token 延迟）。参见第 3 章。

**Quantization**
量化。通过降低数值精度（如 FP16 → INT4/INT8/FP8）来减少模型显存占用和提升推理效率的技术。vLLM 支持 AWQ、GPTQ、FP8、GGUF 等多种量化方法。参见第 21 章。

**SamplingParams**
采样参数配置，控制文本生成的行为。包括温度（temperature）、Top-P、Top-K、最大 token 数、停止条件、惩罚系数等。定义在 `vllm/sampling_params.py`。参见第 8 章、附录 B。

**Scheduler**
调度器，vLLM 引擎的"大脑"。每次迭代决定哪些请求参与推理、分配多少资源、是否需要抢占。受 token budget 和 sequence budget 双重约束。v1 实现位于 `vllm/v1/core/sched/scheduler.py`。参见第 4 章。

**Sequence Budget**
序列预算，即每次调度迭代允许的最大并发序列数，由 `max_num_seqs` 参数控制。与 Token Budget 共同约束每次迭代的工作量。参见第 4 章。

**Speculative Decoding**
推测解码。使用一个小的"草稿模型"（draft model）快速生成多个候选 token，再由大模型并行验证。通过一次前向传播接受多个 token 来减少大模型调用次数，降低延迟。参见第 14 章。

**Swap**
换出/换入。当序列被抢占时，其 KV Cache 从 GPU 显存拷贝到 CPU 内存（swap out）；恢复执行时再拷贝回来（swap in）。通过 PCIe 总线传输，延迟较高但避免了重算开销。参见第 4 章、第 5 章。

**Tensor Parallelism（TP）**
张量并行。将模型权重矩阵按列或行切分到多个 GPU 上，各 GPU 独立计算后通过 AllReduce 同步结果。是 LLM 推理中最常用的并行方式，通常在单节点内的多 GPU 间使用。参见第 16 章。

**Throughput**
吞吐量，衡量系统在单位时间内处理 token 的能力，通常以 tokens/second 表示。是离线批处理场景的核心指标。参见第 22 章。

**Token Budget**
Token 预算，即每次调度迭代允许处理的最大 token 数，由 `max_num_batched_tokens` 参数控制。该预算包括 prefill token 和 decode token（每个 decode 序列贡献 1 个 token）。参见第 4 章。

**TPOT（Time Per Output Token）**
每输出 token 时间，衡量 decode 阶段每生成一个 token 的耗时。是在线服务的重要延迟指标，直接影响用户感知的生成速度。参见第 22 章。

**TTFT（Time To First Token）**
首 token 延迟，从请求到达到第一个生成 token 返回的时间。主要由 prefill 计算时间和排队等待时间决定，是交互式场景最关键的延迟指标。参见第 22 章。

**Worker**
工作进程，vLLM 中管理单个 GPU 设备的组件。每个 Worker 拥有独立的模型副本（或模型分片）、KV Cache 和 ModelRunner。Worker 接收来自 Executor 的指令并在本地 GPU 上执行计算。v1 版本定义在 `vllm/v1/worker/gpu_worker.py`。参见第 10 章。
