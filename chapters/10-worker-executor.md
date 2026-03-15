# 第 10 章：Worker 与 Executor

> "好的架构不是消除复杂性，而是把复杂性放到正确的层次。" —— Executor 和 Worker 的分层设计，正是将分布式执行的复杂性封装在了引擎看不到的地方。

## 10.1 Executor 与 Worker 的分工

在 vLLM 的架构中，Engine 层负责"做什么"的决策，而 Executor 和 Worker 层负责"怎么做"的执行。这两层之间有明确的分工：

- **Executor**：执行器抽象层，屏蔽单 GPU 与多 GPU 的差异。Engine 只与 Executor 交互，不需要关心底层有几块 GPU、是否使用 Ray 分布式框架
- **Worker**：实际管理一块 GPU 设备的工作单元。每个 Worker 拥有独立的模型副本、KV Cache 和 CUDA 运行时

这种分层设计的好处是：引擎层的代码不需要为单机和分布式场景编写不同的逻辑，所有差异都被 Executor 层吸收。

## 10.2 Executor 抽象层

Executor 的基类定义在 `vllm/v1/executor/abstract.py`，核心接口包括：

```python
class Executor:
    def _init_executor(self) -> None: ...
    def collective_rpc(self, method, args) -> list: ...
    def check_health(self) -> None: ...
```

### 三种 Executor 实现

**UniProcExecutor**（`vllm/v1/executor/uniproc_executor.py`）：单进程执行器，用于单 GPU 场景。Worker 直接运行在主进程中，方法调用是普通的函数调用，没有进程间通信开销。这是最简单也是延迟最低的执行模式。

**MultiProcExecutor**（`vllm/v1/executor/multiproc_executor.py`）：多进程执行器，用于单机多 GPU 场景。每个 Worker 运行在独立的 Python 进程中，通过进程间通信传递数据。适用于 tensor parallelism 部署。

**RayDistributedExecutor**（`vllm/v1/executor/ray_distributed_executor.py`）：Ray 分布式执行器，用于多机多 GPU 场景。Worker 以 Ray Actor 的形式运行在 Ray 集群的各个节点上，通过 Ray 的 RPC 机制通信。适用于大规模分布式推理。

### collective_rpc：统一的远程调用接口

Executor 的核心方法 `collective_rpc()` 提供了一个统一的接口来调用所有 Worker 上的方法：

```python
# 在所有 Worker 上执行 load_model
executor.collective_rpc("load_model")

# 在所有 Worker 上执行 execute_model，传递调度输出
executor.collective_rpc("execute_model",
                        args=(scheduler_output,))
```

不同的 Executor 实现以不同方式分发这个调用：UniProcExecutor 直接调用本地方法，MultiProcExecutor 通过多进程消息队列分发，RayDistributedExecutor 通过 Ray remote call 分发。但对 Engine 层来说，调用方式完全相同。

`collective_rpc` 还支持非阻塞模式（`non_block=True`），返回 `Future` 对象，允许 Engine 在等待计算完成的同时做其他工作。

## 10.3 Worker：GPU 设备的管家

Worker 类定义在 `vllm/v1/worker/gpu_worker.py`（GPU 版本）。每个 Worker 管理一块 GPU 设备，是模型执行的实际场所。

### 初始化流程

Worker 的初始化分为几个关键步骤：

**init_device()**：配置 GPU 环境
- 设置 CUDA 设备（`torch.cuda.set_device()`）
- 初始化 NCCL 通信后端（用于 tensor parallelism）
- 创建 `GPUModelRunner` 实例
- 在数据并行场景中计算正确的 rank 和设备映射

**load_model()**：加载模型权重
- 将模型权重从磁盘或 HuggingFace Hub 加载到 GPU 显存
- 在适当的内存池中执行加载操作
- 处理弹性 tensor parallelism 的权重映射

**determine_available_memory()**：分析可用显存
- 执行一次 profiling 前向计算，测量峰值显存使用
- 估算 CUDA Graph 所需的额外显存
- 计算剩余可用于 KV Cache 的显存量
- 向用户报告优化建议

### execute_model()：前向计算的入口

`execute_model()` 是 Worker 最核心的方法：

```python
def execute_model(self, scheduler_output: SchedulerOutput):
    # 1. 更新请求状态
    # 2. 准备模型输入
    # 3. 执行前向计算
    # 4. 采样生成 token
    # 5. 返回结果
    return self.model_runner.execute_model(scheduler_output)
```

在 pipeline parallelism 场景中，Worker 还需要处理中间张量的异步传输：非最后一个 stage 的 Worker 将 hidden states 发送给下一个 stage，非第一个 stage 的 Worker 从上一个 stage 接收输入。

### 内存管理

Worker 提供了精细的内存管理功能：

- **sleep() / wake_up()**：将模型权重和 KV Cache 卸载到 CPU 或释放，需要时再恢复。这对于多模型切换场景非常有用
- **compile_or_warm_up_model()**：预热 kernel、编译 Triton JIT 代码、捕获 CUDA Graph

## 10.4 GPUModelRunner：模型执行的细节

`GPUModelRunner`（`vllm/v1/worker/gpu_model_runner.py`）是实际执行模型前向计算的组件。它比 Worker 更底层，直接操作 GPU 张量。

### 核心职责

**_update_states()**：将 `SchedulerOutput` 中的调度决策同步到本地缓存的请求状态。利用"持久化批次"优化——连续步骤中重复出现的请求不需要重新初始化，只需更新增量信息。

**_prepare_inputs()**：构造模型的输入张量，包括：
- `input_ids`：本步要处理的 token ID
- `positions`：每个 token 的位置编码
- `seq_lens`：每个请求的序列长度
- `query_start_loc`：每个请求在批次中的起始位置
- attention metadata：传递给 attention 层的元数据（块表、前缀长度等）

**execute_model()**：编排整个前向计算流程：

```
_update_states() → _prepare_inputs() → model.forward() → sampler()
```

### CUDA Graph 优化

decode 阶段（每次只处理一个新 token）的计算量较小，GPU kernel 启动开销占比较大。GPUModelRunner 通过 **CUDA Graph** 来优化：

- 在预热阶段，对不同的 batch size 捕获 CUDA Graph
- 运行时直接 replay 图，避免重复的 kernel 启动开销
- `cudagraph_dispatcher` 负责选择匹配当前 batch size 的图

### Input Batch 管理

`gpu_input_batch.py` 中的 `InputBatch` 类管理着 GPU 上的持久化缓冲区。它维护着所有活跃请求的状态，包括 token ID、采样参数等。这些缓冲区在步骤之间持续存在，只在请求加入或离开时更新，大幅减少了 CPU-GPU 数据传输。

## 10.5 多设备支持

除了 GPU，vLLM v1 的 Worker 层还支持其他设备：

- **cpu_worker.py / cpu_model_runner.py**：CPU 推理支持
- **xpu_worker.py / xpu_model_runner.py**：Intel XPU 支持

这些实现共享相同的 Worker 接口，通过不同的 ModelRunner 适配不同硬件的计算特性。

## 10.6 通信链路全景

请求从用户到 GPU 的完整通信链路如下：

```
用户请求
  │
  ▼
LLM / AsyncLLM（入口层）
  │ add_request()
  ▼
EngineCore（核心层）
  │ schedule() → SchedulerOutput
  ▼
Executor（执行器层）
  │ collective_rpc("execute_model", ...)
  ▼
Worker（设备层）
  │ execute_model()
  ▼
GPUModelRunner（计算层）
  │ _prepare_inputs() → model.forward() → sample()
  ▼
GPU 硬件执行
```

返回路径是这条链路的逆向：GPU 计算结果 → ModelRunner → Worker → Executor → EngineCore → OutputProcessor → 用户。

每一层都只与相邻层交互，形成了清晰的责任链。这种设计使得任何一层的实现变更都不会波及其他层——无论是替换调度策略、切换分布式框架，还是适配新的硬件平台。

## 10.7 Worker 的并行策略

Worker 层原生支持多种并行策略：

**Tensor Parallelism**：将模型的权重矩阵沿特定维度切分到多个 GPU 上。每个 Worker 持有模型的一个切片，前向计算时通过 NCCL AllReduce 通信同步中间结果。详见 `vllm/distributed/` 中的通信原语。

**Pipeline Parallelism**：将模型按层分割到多个 GPU 上。Worker 之间通过异步发送/接收 hidden states 来实现流水线执行。`execute_model()` 中的 `async_tp_comm` 和 pipeline 相关逻辑处理这一场景。

**Data Parallelism**：多个完整的模型副本处理不同的请求子集。Worker 初始化时通过 `dp_utils.py` 中的工具函数计算自己的数据并行 rank 和对应的设备。

## 本章小结

本章剖析了 vLLM 执行层的两个核心组件。Executor（`vllm/v1/executor/`）作为抽象层，提供 `UniProcExecutor`、`MultiProcExecutor` 和 `RayDistributedExecutor` 三种实现，通过统一的 `collective_rpc()` 接口屏蔽分布式差异。Worker（`vllm/v1/worker/gpu_worker.py`）管理单个 GPU 设备，负责模型加载、显存管理和前向计算。GPUModelRunner（`vllm/v1/worker/gpu_model_runner.py`）在最底层构造模型输入、执行前向计算、管理 CUDA Graph 优化。从 Engine 到 Executor 到 Worker 到 ModelRunner 的分层设计，使得 vLLM 能够以统一的代码路径支持从单 GPU 到多机多卡的各种部署场景。
