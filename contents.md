# 目录

## 第一部分：宏观认知

- [第 1 章　项目概览与设计哲学](/chapters/01-overview)
- [第 2 章　Repo 结构与模块依赖](/chapters/02-repo-structure)

## 第二部分：PagedAttention 核心

- [第 3 章　KV Cache 传统问题与 PagedAttention 设计](/chapters/03-kv-cache-paged-attention)
- [第 4 章　BlockManager 实现](/chapters/04-block-manager)
- [第 5 章　Attention 算子实现](/chapters/05-attention-kernel)

## 第三部分：调度系统

- [第 6 章　Scheduler 架构](/chapters/06-scheduler)
- [第 7 章　序列管理](/chapters/07-sequence)
- [第 8 章　Preemption 与 Swap](/chapters/08-preemption-swap)

## 第四部分：执行引擎

- [第 9 章　LLMEngine 总览](/chapters/09-llm-engine)
- [第 10 章　Worker 与 Executor](/chapters/10-worker-executor)
- [第 11 章　模型加载与权重](/chapters/11-model-loading)
- [第 12 章　采样器](/chapters/12-sampler)

## 第五部分：连续批处理

- [第 13 章　Continuous Batching 原理](/chapters/13-continuous-batching)
- [第 14 章　Chunked Prefill](/chapters/14-chunked-prefill)
- [第 15 章　Speculative Decoding](/chapters/15-speculative-decoding)

## 第六部分：分布式推理

- [第 16 章　Tensor Parallelism](/chapters/16-tensor-parallelism)
- [第 17 章　Pipeline Parallelism](/chapters/17-pipeline-parallelism)
- [第 18 章　分布式 KV Cache 与 P/D 分离](/chapters/18-distributed-kv-cache)

## 第七部分：API 服务层

- [第 19 章　OpenAI 兼容 API](/chapters/19-openai-api)
- [第 20 章　LoRA 与多 Adapter 服务](/chapters/20-lora-adapter)

## 第八部分：量化与优化

- [第 21 章　量化支持](/chapters/21-quantization)
- [第 22 章　性能调优与 Profiling](/chapters/22-performance-tuning)

## 附录

- [附录 A：推荐阅读路径](/chapters/appendix-a-reading-path)
- [附录 B：核心类型速查](/chapters/appendix-b-type-reference)
- [附录 C：名词解释](/chapters/appendix-c-glossary)
