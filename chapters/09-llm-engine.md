# 第 9 章：LLMEngine 总览

> "A good architect builds not by piling up features, but by composing clean layers." —— LLMEngine 是 vLLM 的中枢协调者，它不亲自做任何计算，却让所有组件协同运转。

## 9.1 LLMEngine 的定位

如果 Scheduler 是大脑，Worker 是四肢，那 LLMEngine 就是神经系统——连接决策与执行的中枢。它是 vLLM 最核心的编排组件，负责协调请求的接收、调度、执行和输出。

在 v1 架构中，Engine 层经历了重大重构。核心实现分布在以下文件中：

- `vllm/engine/llm_engine.py`：同步引擎，封装了请求处理的主循环
- `vllm/v1/engine/async_llm.py`：异步引擎 `AsyncLLM`，面向在线服务场景
- `vllm/v1/engine/core.py`：`EngineCore`，包含 Scheduler 和核心状态管理
- `vllm/entrypoints/llm.py`：`LLM` 类，面向离线批量推理的高级 API

## 9.2 step() 方法：推理主循环

`LLMEngine.step()` 是整个推理流程的驱动入口，每次调用执行一个完整的推理步骤：

```
step() 的执行流程：

1. 从 EngineCore 获取输出 → engine_core.get_output()
2. 通过 OutputProcessor 处理输出 → 生成 RequestOutput
3. 终止已完成的请求 → abort finished requests
4. 记录统计信息 → log stats
```

在底层，`EngineCore` 封装了真正的调度和执行逻辑：

1. **调度阶段**：`Scheduler.schedule()` 生成 `SchedulerOutput`，决定本步处理哪些请求
2. **执行阶段**：`Executor.execute_model()` 将调度结果发送给 Worker，执行模型前向计算
3. **采样阶段**：Worker 中的 Sampler 从 logits 中采样出新 token
4. **更新阶段**：将新 token 追加到对应请求，检查停止条件

这个循环不断重复，直到所有请求都处理完毕。

## 9.3 两种引擎接口

vLLM 提供两种引擎接口来适应不同的使用场景：

### LLMEngine：同步接口

`LLMEngine`（`vllm/engine/llm_engine.py`）提供同步的推理接口，主要用于离线批量处理。调用 `step()` 会阻塞当前线程直到本步完成。

关键方法：
- `add_request()`：将新请求加入系统
- `step()`：执行一步推理
- `abort_request()`：取消正在处理的请求
- `from_engine_args()`：从命令行参数创建引擎实例

### AsyncLLM：异步接口

`AsyncLLM`（`vllm/v1/engine/async_llm.py`）实现了 `EngineClient` 协议，面向在线服务场景。它在后台运行一个输出处理循环（`_run_output_handler()`），通过 async/await 模式与 API 服务层交互。

```python
class AsyncLLM(EngineClient):
    async def generate(self, prompt, sampling_params, request_id):
        collector = await self.add_request(...)
        async for output in collector:
            yield output  # 流式返回每一步的生成结果
```

`AsyncLLM` 的核心循环 `_run_output_handler()` 持续运行：
1. 通过 `engine_core.get_output_async()` 异步获取输出
2. 分块处理输出，避免阻塞事件循环
3. 对已完成请求发送终止信号
4. 上报统计指标

## 9.4 add_request()：请求如何进入系统

`add_request()` 是请求进入 vLLM 系统的入口。它的处理流程包括：

1. **输入预处理**：通过 `InputProcessor` 将用户的原始输入（文本字符串、token ID 列表或多模态输入）转换为标准化的 `EngineCoreRequest`
2. **参数校验**：验证 `SamplingParams` 的合法性（如 temperature >= 0、max_tokens > 0 等）
3. **Fan-out 处理**：当 `n > 1`（并行采样）时，将一个请求拆分为多个子请求，通过 `ParentRequest` 机制关联
4. **入队**：将请求发送给 `EngineCore`，最终进入 Scheduler 的 waiting 队列

## 9.5 LLM 类：离线推理的入口

`LLM` 类（`vllm/entrypoints/llm.py`）是面向用户最友好的 API，适合批量推理场景：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-8B")
outputs = llm.generate(
    ["Hello, my name is", "The capital of France is"],
    SamplingParams(temperature=0.8, max_tokens=256)
)
```

`LLM` 类是 `LLMEngine` 的高层封装：

- 构造函数中通过 `LLMEngine.from_engine_args()` 创建引擎实例
- `generate()` 方法将所有 prompt 通过 `add_request()` 加入引擎，然后循环调用 `step()` 直到所有请求完成
- `chat()` 方法支持对话格式，自动应用 chat template
- `encode()` 方法支持 embedding 提取

## 9.6 输出处理

引擎的输出处理链路将内部表示转换为用户可用的格式：

**RequestOutput**：每个请求在每一步的输出，包含：
- `request_id`：请求标识
- `outputs`：生成的 `CompletionOutput` 列表
- `finished`：是否已完成
- `prompt`：原始 prompt 文本
- `prompt_token_ids`：prompt 的 token ID

**CompletionOutput**：单个生成结果，包含：
- `text`：生成的文本
- `token_ids`：生成的 token ID 列表
- `finish_reason`：完成原因（"stop"、"length" 等）
- `logprobs`：可选的 token 对数概率

`OutputProcessor` 负责将 Worker 返回的原始输出（token ID 和 logprobs）解码为文本，并检查停止条件。

## 9.7 abort_request()：请求取消

在线服务中，用户可能随时断开连接。`abort_request()` 提供了优雅的请求取消机制：

1. 通知 `OutputProcessor` 停止处理该请求的后续输出
2. 通知 `EngineCore` 将该请求从 Scheduler 中移除
3. 释放该请求占用的 KV Cache 和 Encoder Cache 资源

这一机制确保了断开连接的请求不会继续浪费 GPU 资源。

## 9.8 引擎初始化

引擎的初始化过程相当复杂，涉及多个组件的创建和配置：

1. **配置解析**：`EngineArgs` 解析命令行参数或 Python API 参数，生成 `VllmConfig` 配置对象
2. **Executor 创建**：根据配置选择合适的 Executor（单 GPU、多 GPU Ray、多进程等）
3. **Worker 初始化**：Executor 创建 Worker，Worker 初始化 GPU 设备
4. **模型加载**：Worker 将模型权重加载到 GPU 显存
5. **内存分析**：通过 profiling 确定可用于 KV Cache 的显存量
6. **KV Cache 分配**：根据可用显存创建 KV Cache 张量
7. **CUDA Graph 捕获**：可选地捕获 CUDA Graph 以加速 decode 阶段

这个过程由 `LLMEngine.from_engine_args()` 或 `LLMEngine.from_vllm_config()` 工厂方法驱动。

## 9.9 系统管理功能

除了核心推理功能，引擎还提供了一系列管理接口：

- **LoRA 管理**：`add_lora()` / `remove_lora()` / `list_loras()` 动态管理 LoRA 适配器
- **缓存控制**：`reset_prefix_cache()` / `reset_mm_cache()` 清理前缀缓存和多模态缓存
- **电源管理**：`sleep()` / `wake_up()` 支持将模型权重和 KV Cache 卸载到 CPU，释放 GPU 显存
- **性能分析**：`start_profile()` / `stop_profile()` 控制性能采集

## 9.10 EngineCore：隐藏的核心

值得特别提及的是 `EngineCore`（`vllm/v1/engine/core.py`）。在 v1 架构中，`LLMEngine` 和 `AsyncLLM` 都不直接持有 Scheduler 和 Executor 的引用——这些组件被封装在 `EngineCore` 内部。`EngineCore` 可以运行在主进程中，也可以运行在独立进程中，通过 `EngineCoreClient` 接口与上层引擎通信。

这种设计的好处是将计算密集型的调度和执行与 IO 密集型的请求处理解耦。在在线服务场景中，`AsyncLLM` 的事件循环可以专注于处理网络请求和输出流式传输，而不会被模型前向计算阻塞。`EngineCore` 的进程隔离还提供了更好的容错性——即使引擎进程出现问题，API 服务进程仍然可以优雅地处理错误。

## 9.11 Coordinator：多引擎协调

在数据并行场景中，多个 `EngineCore` 实例需要协调工作。`Coordinator`（`vllm/v1/engine/coordinator.py`）负责在多个引擎核心之间分配请求、汇总输出，确保请求在正确的数据并行分片上执行。这种多引擎架构使得 vLLM 能够在保持单个引擎简洁性的同时，通过水平扩展来处理更高的请求负载。

## 本章小结

本章全面介绍了 vLLM 的引擎架构。`LLMEngine` 作为中枢协调者，通过 `step()` 方法驱动"调度 → 执行 → 采样 → 更新"的推理循环。同步的 `LLMEngine` 适合离线批量处理，异步的 `AsyncLLM` 面向在线服务。`LLM` 类提供了最简洁的用户 API。请求通过 `add_request()` 进入系统，经过 `InputProcessor` 预处理后进入 Scheduler，执行结果通过 `OutputProcessor` 转换为用户可读的 `RequestOutput`。引擎的初始化涵盖了从配置解析到 CUDA Graph 捕获的完整流程，为下一章将要介绍的 Worker 与 Executor 层奠定了上层架构基础。
