# vLLM 源码解析

> PagedAttention 与高吞吐量 LLM 推理引擎深度剖析

## 关于本书

[vLLM](https://github.com/vllm-project/vllm) 是一个高吞吐量、低延迟的大语言模型推理与服务引擎。它通过 PagedAttention 算法实现了近零浪费的 KV Cache 内存管理，结合 Continuous Batching 调度策略，将 LLM 推理的吞吐量提升了数倍。

本书从源码层面系统梳理 vLLM 的架构设计与实现细节，适合希望：

- 理解 PagedAttention 内存管理机制的开发者
- 学习高性能 LLM 推理引擎设计的工程师
- 希望基于 vLLM 进行深度定制或贡献代码的参与者
- 对分布式推理、量化部署、Continuous Batching 感兴趣的技术人员

## 目录

详见 [CONTENTS.md](./contents.md)

全书共 **22 章 + 3 附录**，分八个部分：

| 部分 | 章节 | 核心议题 |
|------|------|---------|
| 第一部分：宏观认知 | Ch 1–2 | 设计哲学、Repo 结构 |
| 第二部分：PagedAttention 核心 | Ch 3–5 | KV Cache、BlockManager、Attention Kernel |
| 第三部分：调度系统 | Ch 6–8 | Scheduler、序列管理、Preemption |
| 第四部分：执行引擎 | Ch 9–12 | LLMEngine、Worker、模型加载、采样器 |
| 第五部分：连续批处理 | Ch 13–15 | Continuous Batching、Chunked Prefill、Speculative Decoding |
| 第六部分：分布式推理 | Ch 16–18 | TP、PP、分布式 KV Cache |
| 第七部分：API 服务层 | Ch 19–20 | OpenAI API、LoRA 服务 |
| 第八部分：量化与优化 | Ch 21–22 | 量化支持、性能调优 |

## 在线阅读

[https://vllm-book.myhubs.dev/](https://vllm-book.myhubs.dev/)

## License

本书内容采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可证。
vLLM 项目本身采用 Apache 2.0 许可证。
