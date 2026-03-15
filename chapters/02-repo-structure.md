# 第 2 章：Repo 结构与模块依赖

> "Any fool can write code that a computer can understand. Good programmers write code that humans can understand." —— Martin Fowler。vLLM 的代码组织方式，正是这句话的最佳注脚。

## 2.1 顶层目录总览

vLLM 的 GitHub 仓库（`vllm-project/vllm`）采用了清晰的顶层目录划分。克隆代码后，映入眼帘的核心目录包括：

```
vllm-project/vllm/
├── vllm/           # Python 主包，核心业务逻辑
├── csrc/           # C++/CUDA 高性能算子
├── benchmarks/     # 性能基准测试
├── docs/           # 项目文档
├── examples/       # 使用示例
├── tests/          # 测试用例
├── cmake/          # CMake 构建配置
├── docker/         # Docker 镜像定义
├── requirements/   # 依赖声明
├── tools/          # 开发辅助工具
├── CMakeLists.txt  # C++ 构建入口
└── pyproject.toml  # Python 项目配置
```

整个项目是一个典型的 Python + C++/CUDA 混合项目。Python 层负责调度、协调和 API 接口，C++/CUDA 层负责性能关键的计算内核。

## 2.2 vllm/ 主包结构

`vllm/` 目录是整个项目的核心，包含了数十个子模块。我们按照功能分组进行梳理：

### 引擎层（Engine）

`vllm/engine/` 目录是系统的控制中枢。`llm_engine.py` 定义了 `LLMEngine` 类，它是同步推理的主引擎；`async_llm_engine.py` 中的 `AsyncLLMEngine` 则封装了异步接口，适用于在线服务场景。引擎负责将用户请求传递给调度器，再将调度结果分发给 worker 执行。

值得注意的是，vLLM 正在经历从 v0 到 v1 架构的演进。新的 v1 引擎位于 `vllm/v1/engine/` 目录下，采用了更加模块化的设计。

### 核心调度层（Core）

v1 架构下，调度相关代码集中在 `vllm/v1/core/` 目录。其中最关键的组件是：

- **`sched/scheduler.py`**：核心调度器，决定每轮迭代中哪些请求参与 prefill、哪些继续 decode
- **`block_pool.py`**：物理块的分配与回收池
- **`kv_cache_manager.py`**：KV Cache 的逻辑管理，维护序列到物理块的映射
- **`kv_cache_coordinator.py`**：协调多层 KV Cache 的分配策略
- **`sched/request_queue.py`**：请求队列管理，支持优先级排序

### 执行层（Worker）

`vllm/v1/worker/` 目录实现了实际的模型执行逻辑：

- **`gpu_worker.py`**：GPU Worker 的主类，管理 GPU 设备和 KV Cache 的初始化
- **`gpu_model_runner.py`**：模型前向传播的执行器，处理输入准备、模型调用和输出收集
- **`cpu_worker.py` / `cpu_model_runner.py`**：CPU 后端实现
- **`block_table.py`**：在 worker 端维护的 block table 数据结构
- **`gpu_input_batch.py`**：管理 GPU 侧的输入 batch 组装

### 模型层（Model Executor）

`vllm/model_executor/` 目录负责模型的加载与执行：

- **`models/`**：包含数百个模型的具体实现（LLaMA、GPT、Qwen、Mistral 等）
- **`layers/`**：通用网络层的高性能实现，包括 linear、attention、RMSNorm 等
- **`model_loader/`**：模型权重加载器，支持多种格式和量化方案
- **`kernels/`**：Python 侧的内核封装

### Attention 模块

v1 架构的 attention 模块位于 `vllm/v1/attention/`：

- **`backends/`**：多种 attention 后端实现，包括 `flash_attn.py`（FlashAttention）、`flashinfer.py`（FlashInfer）、`rocm_attn.py`（AMD ROCm）、`triton_attn.py`（Triton）等
- **`selector.py`**：根据硬件和配置自动选择最优后端
- **`backend.py`**：后端的统一接口定义

### Executor 分布式层

`vllm/v1/executor/` 管理分布式执行策略：

- **`uniproc_executor.py`**：单进程执行器，适用于单卡场景
- **`multiproc_executor.py`**：多进程执行器，用于单机多卡 tensor parallelism
- **`ray_distributed_executor.py`**：基于 Ray 的分布式执行器，支持多机多卡

### API 入口层（Entrypoints）

`vllm/entrypoints/` 提供了丰富的对外接口：

- **`llm.py`**：离线推理的高层 API（`LLM` 类）
- **`openai/`**：OpenAI 兼容的 REST API 实现
- **`anthropic/`**：Anthropic API 兼容接口
- **`api_server.py`**：通用 API 服务器
- **`grpc_server.py`**：gRPC 服务端点

### 其他重要模块

- **`vllm/lora/`**：LoRA 适配器的运行时管理
- **`vllm/v1/spec_decode/`**：投机解码（Speculative Decoding）实现
- **`vllm/distributed/`**：分布式通信原语
- **`vllm/multimodal/`**：多模态输入处理（图像、音频等）
- **`vllm/transformers_utils/`**：与 HuggingFace Transformers 的集成工具

## 2.3 csrc/ 目录：高性能内核

`csrc/` 目录存放了用 C++ 和 CUDA 编写的高性能算子：

- **`csrc/attention/`**：PagedAttention 的 CUDA 内核，包括 `paged_attention_v1.cu`、`paged_attention_v2.cu` 以及辅助的数据类型头文件
- **`csrc/quantization/`**：量化相关的高性能内核
- **`csrc/moe/`**：Mixture of Experts 的自定义内核
- **`csrc/core/`**：核心工具函数

## 2.4 分层架构与依赖关系

vLLM 的模块依赖呈现清晰的分层结构，自上而下为：

```
Entrypoints (API 层)
    ↓
Engine (引擎层)
    ↓
Scheduler + KV Cache Manager (调度与内存管理层)
    ↓
Executor (执行编排层)
    ↓
Worker + Model Runner (执行层)
    ↓
Model Executor + Attention Backends (模型与算子层)
    ↓
csrc/ CUDA Kernels (内核层)
```

每一层只依赖其下方的层，避免了循环依赖。引擎层是整个系统的粘合剂：它上接 API 请求，下调 Scheduler 进行决策，再通过 Executor 将工作分发到 Worker 执行。

## 本章小结

vLLM 的代码仓库采用了经典的分层架构设计。Python 主包 `vllm/` 按功能划分为引擎、调度、执行、模型、attention、API 入口等子模块，各层职责清晰、依赖关系单向流动。`csrc/` 目录承载了 CUDA 高性能内核的实现。项目正处于从 v0 到 v1 架构的演进中，v1 架构（`vllm/v1/`）在模块化和可扩展性方面做了进一步优化。理解这一整体结构，是深入阅读各模块源码的前提。
