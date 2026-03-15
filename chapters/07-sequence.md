# 第 7 章：序列管理

> "Every complex system is built on simple abstractions." —— 在 vLLM 中，Request 就是那个看似简单却承载着整个推理生命周期的核心抽象。

## 7.1 从 Sequence 到 Request

在 LLM 推理系统中，"序列"是最基本的处理单元——一串 token ID 从 prompt 开始，逐步生成新的 token，直到满足停止条件。在 vLLM v1 架构中，这一概念被统一封装为 `Request` 类（定义在 `vllm/v1/request.py`），取代了早期版本中 `Sequence`、`SequenceGroup`、`SequenceData` 等分散的抽象。

这一设计简化是 v1 架构的重要改进：一个 Request 对象完整地表示一个推理请求从到达到完成的全部状态。

## 7.2 Request 类的核心属性

`Request` 类承载着请求的全部信息：

**标识与优先级**：
- `request_id`：全局唯一标识符
- `client_index`：客户端侧索引
- `priority`：调度优先级，影响抢占决策
- `arrival_time`：到达时间戳，用于 FCFS 排序

**输入数据**：
- `prompt_token_ids`：经过分词后的 prompt token 列表
- `prompt_embeds`：可选的预计算 embedding
- `mm_features`：多模态特征（如图像、音频）
- `num_prompt_tokens`：prompt 的 token 数量

**生成控制**：
- `sampling_params`：采样参数，控制温度（temperature）、top_p、top_k、最大生成长度等
- `max_tokens`：最大生成 token 数
- `output_token_ids`：已生成的 token 列表
- `num_output_tokens`：已生成的 token 计数

**缓存相关**：
- `block_hashes`：用于前缀缓存的哈希值
- `num_computed_tokens`：已计算 KV Cache 的 token 数

**状态管理**：
- `status`：当前的 `RequestStatus` 枚举值

## 7.3 RequestStatus：状态机

`RequestStatus` 枚举定义了请求在生命周期中的所有可能状态。理解这个状态机是理解 vLLM 调度系统的关键：

```
                    ┌─────────────────────────────┐
                    │          WAITING             │
                    └─────────┬───────────────────┘
                              │ 被调度执行
          ┌───────────────────▼───────────────────┐
          │              RUNNING                   │
          └──┬──────┬──────┬──────────────────────┘
             │      │      │ 生成完成
             │      │      ├──→ FINISHED_STOPPED
             │      │      ├──→ FINISHED_LENGTH_CAPPED
             │      │      ├──→ FINISHED_ABORTED
             │      │      ├──→ FINISHED_IGNORED
             │      │      ├──→ FINISHED_ERROR
             │      │      └──→ FINISHED_REPETITION
             │      │
             │      └──→ PREEMPTED ──→ WAITING（重新排队）
             │
             └──→ WAITING_FOR_FSM / WAITING_FOR_REMOTE_KVS
```

核心状态转换路径：

- **WAITING → RUNNING**：Scheduler 决定调度此请求，为其分配 KV Cache 块
- **RUNNING → FINISHED_***：生成完成，可能是正常停止（遇到 EOS 或 stop 序列）、达到长度上限、被用户取消或遇到错误
- **RUNNING → PREEMPTED**：GPU 内存不足，请求被抢占，KV Cache 被释放
- **PREEMPTED → WAITING**：抢占后重新进入等待队列，`num_computed_tokens` 重置为 0

v1 还引入了几个等待子状态：
- **WAITING_FOR_FSM**：等待结构化输出的有限状态机编译完成
- **WAITING_FOR_REMOTE_KVS**：在分布式 KV Cache 场景中等待远程缓存传输
- **WAITING_FOR_STREAMING_REQ**：等待流式请求的后续数据到达

## 7.4 SamplingParams：生成行为的控制器

`SamplingParams`（定义在 `vllm/sampling_params.py`）控制着 token 生成的每一个细节：

- **temperature**：温度参数，值越大随机性越强。0 表示贪心解码
- **top_p**：nucleus sampling 阈值，只从累积概率达到 top_p 的 token 中采样
- **top_k**：只从概率最高的 k 个 token 中采样
- **max_tokens**：最大生成 token 数
- **stop**：停止序列列表，生成内容匹配任一序列时停止
- **presence_penalty / frequency_penalty**：重复惩罚参数
- **n**：并行采样数，一个 prompt 生成多个独立回复

当 `n > 1` 时，上层引擎会将一个用户请求拆分为多个 Request 对象（fan-out），共享相同的 prompt。这在 `AsyncLLM.add_request()` 中通过 `ParentRequest` 机制实现。

## 7.5 SchedulerOutput 中的请求表示

当请求被调度执行时，它的信息需要被传递给 Worker 层。`SchedulerOutput`（`vllm/v1/core/sched/output.py`）使用两种数据结构来表示不同状态的请求：

**NewRequestData**：首次被调度的请求，携带完整的 prompt token ID、采样参数、KV Cache 块表等信息。Worker 需要这些信息来初始化请求状态。

**CachedRequestData**：已经在运行中的请求，只携带增量信息——新分配的块 ID、继续生成所需的最少元数据。这种设计避免了重复传输不变的数据。

这种区分体现了 v1 架构的"持久化批次（Persistent Batch）"优化思想：连续的调度步骤之间，大部分请求是重复的，只需传递差异即可。

## 7.6 请求的完整生命周期

一个请求从到达到完成，经历以下阶段：

1. **到达**：用户通过 API 发送请求，`AsyncLLM.add_request()` 或 `LLMEngine.add_request()` 将其转换为 `EngineCoreRequest`
2. **预处理**：`InputProcessor` 对输入进行分词、多模态处理，生成 `Request` 对象
3. **入队**：Request 进入 Scheduler 的 waiting 队列，状态为 `WAITING`
4. **调度**：`schedule()` 方法选中该请求，为其分配 KV Cache 块，状态变为 `RUNNING`
5. **执行**：Worker 对请求执行前向计算，生成一个新 token
6. **迭代**：重复步骤 4-5，每步生成一个 token（或被抢占后重新从步骤 3 开始）
7. **完成**：满足停止条件，状态变为 `FINISHED_*`，释放 KV Cache 资源
8. **输出**：`OutputProcessor` 将结果解码为文本，通过 API 返回给用户

其中步骤 4-6 构成了核心的推理循环，可能执行数百甚至数千次。

## 7.7 Request 的比较与排序

`Request` 类实现了 `__lt__` 方法，使其可以在优先队列中正确排序：

```python
def __lt__(self, other: "Request") -> bool:
    # 基于 priority 和 arrival_time 的比较
    ...
```

这使得 Scheduler 的 waiting 队列能够自动按优先级和到达时间排序，FCFS 策略下先到的请求优先被调度，Priority 策略下高优先级请求优先被调度。

## 7.8 多模态请求的特殊处理

现代大语言模型越来越多地支持多模态输入（图像、音频等）。`Request` 类通过 `mm_features` 字段承载多模态特征规格。当请求包含多模态输入时，处理流程有几点不同：

- **Encoder Cache**：多模态输入需要经过 encoder 编码后缓存。`EncoderCacheManager`（`vllm/v1/core/encoder_cache_manager.py`）负责管理这些编码结果的缓存，避免重复编码
- **调度预算**：Scheduler 需要额外跟踪 encoder 的计算预算，确保 encoder 前向计算不会超出资源限制
- **前缀缓存**：多模态 token 也参与前缀缓存的哈希计算，相同的图文组合可以复用已有的 KV Cache

## 7.9 流式请求与 StreamingUpdate

v1 架构引入了**流式请求**（Streaming Request）的概念。对于可恢复的请求（`resumable=True`），`StreamingUpdate` 数据类允许在请求运行过程中动态更新其参数——例如追加新的 prompt token 或调整生成长度。

`Request` 内部维护一个 `streaming_queue`（双端队列），存储待处理的 `StreamingUpdate` 消息。Scheduler 在每一步调度时检查这个队列，将更新应用到请求中。这一机制为交互式对话、实时语音转文字等场景提供了基础支持，使请求不再是一个一次性提交的静态实体，而可以在执行过程中动态演进。

## 本章小结

本章深入分析了 vLLM 的序列管理机制。v1 架构将序列抽象统一为 `Request` 类（`vllm/v1/request.py`），通过 `RequestStatus` 枚举管理请求的状态机转换。`SamplingParams` 提供了精细的生成行为控制，`SchedulerOutput` 中的 `NewRequestData` 和 `CachedRequestData` 实现了高效的增量信息传递。理解 Request 的生命周期——从 WAITING 到 RUNNING 再到 FINISHED——是理解 vLLM 整体工作流程的基础，也为下一章将要讨论的抢占机制提供了必要的上下文。
