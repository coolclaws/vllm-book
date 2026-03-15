# 第 1 章：项目概览与设计哲学

> "The best way to predict the future is to invent it." —— Alan Kay。vLLM 的诞生，正是一群研究者不满足于现状、重新发明 LLM 推理范式的故事。

## 1.1 vLLM 的诞生背景

2023 年，大语言模型（Large Language Model, LLM）的浪潮席卷整个 AI 行业。GPT-4、LLaMA、PaLM 等模型展现了惊人的能力，但随之而来的是一个严峻的工程问题：**如何高效地为数以百万计的用户提供 LLM 推理服务？**

UC Berkeley 的 LMSYS 团队在运营 Chatbot Arena 的过程中，深刻体会到了 LLM 服务化的痛点。他们发现，当时主流的推理框架——无论是 HuggingFace Transformers 还是 NVIDIA 的 FasterTransformer——都面临着严重的内存浪费问题。这一观察促使 Kwon Woosuk 等人发表了开创性的论文 *"Efficient Memory Management for Large Language Model Serving with PagedAttention"*，并同步开源了 vLLM 项目。

## 1.2 核心问题：KV Cache 的内存困境

LLM 的自回归解码（autoregressive decoding）过程中，每生成一个 token 都需要用到之前所有 token 的 Key 和 Value 向量，即 KV Cache。传统系统的做法是为每个请求预分配一块连续内存，大小按 `max_seq_len` 计算。这种方式导致了三重浪费：

- **内部碎片（Internal Fragmentation）**：实际序列长度远小于最大长度，预分配的空间大量闲置
- **外部碎片（External Fragmentation）**：已释放的内存块大小不一，无法拼成完整的大块
- **预留浪费（Reservation Waste）**：为可能用到的最大长度预留内存，实际利用率极低

研究表明，传统系统中 60%~80% 的 KV Cache 内存处于浪费状态。对于一块价值数万美元的 GPU 来说，这意味着巨大的成本损失。

## 1.3 PagedAttention：从操作系统汲取灵感

vLLM 的核心创新 PagedAttention 从操作系统的虚拟内存管理中借鉴了分页（Paging）思想。正如操作系统将进程的虚拟地址空间映射到不连续的物理页框，PagedAttention 将每个序列的 KV Cache 分成固定大小的 block（通常为 16 个 token），通过 block table 将逻辑块映射到 GPU 上的物理块。

这一设计带来了革命性的改进：

- **近零内存浪费**：按需分配，只在需要时才申请新的物理块
- **Copy-on-Write 共享**：beam search 等场景下，多个序列可以共享相同的 KV Cache 块
- **动态增长**：序列长度增加时，只需追加新块，无需重新分配

## 1.4 性能表现

在论文公布的基准测试中，vLLM 展现了令人瞩目的性能：

- 相比 HuggingFace Transformers，吞吐量提升 **2~4 倍**
- 在某些长序列场景下，相比 text-generation-inference 吞吐量提升高达 **24 倍**
- 在高并发场景中优势尤为显著，因为内存利用率的提升意味着可以同时服务更多请求

## 1.5 设计哲学与指导原则

vLLM 的架构设计遵循几个核心原则：

**内存效率优先（Memory Efficiency First）**：整个系统围绕 KV Cache 的高效管理展开。从调度器（`vllm/v1/core/sched/scheduler.py`）到 block 管理器（`vllm/v1/core/block_pool.py`），每个组件都以最大化内存利用率为首要目标。

**连续批处理（Continuous Batching）**：不同于传统的静态 batching，vLLM 的调度器可以在每个 iteration 动态调整 batch 的组成。新请求可以随时加入，已完成的请求立即释放资源。这一逻辑的核心在 `vllm/v1/core/sched/scheduler.py` 中实现。

**硬件无关的执行层（Hardware-Agnostic Execution）**：通过 executor 抽象层（`vllm/v1/executor/abstract.py`），vLLM 支持从单卡到多节点分布式的各种部署场景。具体的 executor 实现包括单进程（`uniproc_executor.py`）、多进程（`multiproc_executor.py`）和基于 Ray 的分布式执行器（`ray_distributed_executor.py`）。

**易用性与兼容性**：vLLM 提供了 OpenAI 兼容的 API 服务端点（`vllm/entrypoints/openai/`），用户几乎不需要修改代码即可从 OpenAI API 迁移到自部署的 vLLM 服务。

## 1.6 关键源码入口

理解 vLLM 的代码结构，从以下几个文件入手最为高效：

| 文件路径 | 职责 |
|---------|------|
| `vllm/entrypoints/llm.py` | 面向用户的高层 API，离线推理入口 |
| `vllm/engine/llm_engine.py` | 核心引擎，协调调度与执行 |
| `vllm/v1/core/sched/scheduler.py` | 请求调度，决定哪些请求参与本轮推理 |
| `vllm/v1/worker/gpu_worker.py` | GPU Worker，执行模型前向传播 |
| `vllm/v1/worker/gpu_model_runner.py` | 模型执行器，管理模型加载与推理细节 |

这些文件构成了 vLLM 的骨架，从用户请求的接收到最终 token 的生成，数据流贯穿其中。

## 本章小结

vLLM 诞生于对 LLM 服务化内存效率问题的深刻洞察。它借鉴操作系统虚拟内存的分页机制，提出了 PagedAttention 这一核心创新，将 KV Cache 的内存利用率提升到接近理论极限。项目遵循内存效率优先、连续批处理、硬件无关和易用性四大设计原则，在短短两年间从一篇论文发展为 LLM 推理领域最具影响力的开源项目之一。在接下来的章节中，我们将逐层深入 vLLM 的代码实现，从仓库结构到核心算法，揭示这套系统的精妙设计。
