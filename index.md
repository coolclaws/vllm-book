---
layout: home

hero:
  name: "vLLM 源码解析"
  text: "PagedAttention 与高吞吐量 LLM 推理引擎深度剖析"
  tagline: 从 PagedAttention 内存管理到 Continuous Batching 调度，全面解读 vLLM 的架构设计与实现细节
  actions:
    - theme: brand
      text: 开始阅读
      link: /chapters/01-overview
    - theme: alt
      text: 查看目录
      link: /contents
    - theme: alt
      text: GitHub
      link: https://github.com/coolclaws/vllm-book

features:
  - icon:
      src: /icons/memory.svg
    title: PagedAttention 内存管理
    details: 深入虚拟内存式 KV Cache 管理，解析 BlockManager、物理块分配、前缀共享的完整实现，理解 vLLM 近零浪费的显存利用机制。

  - icon:
      src: /icons/scheduler.svg
    title: 调度与批处理系统
    details: 剖析 Scheduler 三队列状态机、Continuous Batching 动态调度、Chunked Prefill 与 Speculative Decoding 的高吞吐推理架构。

  - icon:
      src: /icons/distributed.svg
    title: 分布式推理引擎
    details: 覆盖 Tensor Parallelism、Pipeline Parallelism、分布式 KV Cache 与 Prefill/Decode 分离，掌握多 GPU 推理的生产级实现。

  - icon:
      src: /icons/api.svg
    title: API 服务与量化优化
    details: 解读 OpenAI 兼容 API、LoRA 动态加载、AWQ/GPTQ/FP8 量化、CUDA Graph 与性能调优，从源码理解 vLLM 的工程化能力。
---
