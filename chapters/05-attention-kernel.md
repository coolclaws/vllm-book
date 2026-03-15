# 第 5 章：Attention 算子实现

> "The devil is in the details." —— Ludwig Mies van der Rohe。PagedAttention 的优雅设计最终必须落地为高效的 CUDA 内核，这一章我们将深入到 GPU 线程的层面，看看 attention 算子是如何在非连续的物理块上完成计算的。

## 5.1 从设计到实现：Attention 的挑战

前两章介绍了 PagedAttention 的内存管理设计。但设计只完成了一半——真正的挑战在于：**如何让 GPU 高效地在非连续的物理块上执行 attention 计算？**

传统 attention 内核假设 KV Cache 存储在一块连续内存中，可以用简单的指针偏移来访问。PagedAttention 打破了这一假设：一个序列的 KV Cache 散布在多个物理块中，内核必须通过 block table 间接寻址。这增加了内存访问的复杂度，但 vLLM 通过精心的内核设计将这一开销控制在极低水平。

## 5.2 AttentionBackend 抽象层

vLLM 在 Python 层定义了统一的 attention 后端抽象。在 v1 架构中，相关代码位于 `vllm/v1/attention/` 目录：

- **`backend.py`**：定义后端的统一接口
- **`selector.py`**：根据硬件平台、模型类型和用户配置自动选择最优后端
- **`backends/`**：各种后端的具体实现

### 后端注册与选择

`selector.py` 中的选择逻辑综合考虑多个因素：GPU 架构（NVIDIA Ampere/Hopper vs AMD CDNA）、是否安装了 FlashAttention 库、模型是否使用了特殊的 attention 模式（如 GQA、MQA）等。系统启动时完成一次选择，之后所有推理请求使用同一后端。

`backends/registry.py` 维护了后端的注册表，支持动态注册新的后端实现。

### 多种后端实现

vLLM v1 支持种类丰富的 attention 后端（均位于 `vllm/v1/attention/backends/` 目录下）：

| 后端文件 | 适用场景 |
|---------|---------|
| `flash_attn.py` | NVIDIA GPU 上的首选后端，基于 FlashAttention 库 |
| `flashinfer.py` | 高性能替代方案，由 FlashInfer 项目提供 |
| `triton_attn.py` | 基于 Triton 的纯 Python CUDA 内核 |
| `rocm_attn.py` | AMD ROCm 平台的 attention 实现 |
| `cpu_attn.py` | CPU 后端，用于开发和测试 |
| `flex_attention.py` | PyTorch 的 FlexAttention 接口 |

## 5.3 PagedAttention CUDA 内核

vLLM 自带的 PagedAttention CUDA 内核是项目的技术基石，代码位于 `csrc/attention/` 目录。

### 源码文件结构

```
csrc/attention/
├── attention_kernels.cuh      # 核心内核模板实现
├── attention_dtypes.h         # 数据类型定义
├── attention_generic.cuh      # 通用工具函数
├── attention_utils.cuh        # 辅助工具
├── dtype_float16.cuh          # FP16 特化
├── dtype_bfloat16.cuh         # BF16 特化
├── dtype_float32.cuh          # FP32 特化
├── dtype_fp8.cuh              # FP8 特化
├── paged_attention_v1.cu      # V1 版本入口
├── paged_attention_v2.cu      # V2 版本入口（大 context）
└── mla/                       # Multi-head Latent Attention
```

### Grid 与 Block 映射

PagedAttention 内核的并行策略如下：

- **Grid 维度**：`(num_heads, num_seqs, num_partitions)`——每个 attention head、每个序列分配一个独立的线程组
- **Block 维度**：每个 CUDA thread block 包含若干个 warp（通常 128~256 个线程）

这意味着不同序列、不同 attention head 的计算完全并行，且互不干扰。

### 内核执行流程

以 decode 阶段为例，单个 thread block 的执行流程：

**第一步：加载 Query 向量。** 当前 token 的 Q 向量从全局内存加载到共享内存（shared memory），由线程组协作完成。

**第二步：遍历物理块计算 QK^T。** 这是 PagedAttention 区别于传统内核的关键步骤。内核遍历当前序列的 block table，对于每个逻辑块：
1. 从 block table 中查到物理块 ID
2. 根据物理块 ID 计算出 K 向量在 GPU 显存中的实际地址
3. 加载该块中所有 token 的 K 向量
4. 计算 Q 与 K 的点积，得到注意力分数（attention score）

这一步的访存模式是非连续的——每个物理块可能位于显存的任意位置——但由于块内的 token 是连续存储的，实际的内存访问仍然具有良好的局部性。

**第三步：Softmax 归一化。** 所有块的 attention score 计算完成后，执行 softmax 操作。为了保证数值稳定性，内核采用 online softmax 技巧：在遍历块的过程中同步更新 running max 和 running sum，避免了两遍遍历。

**第四步：V 加权求和。** 再次遍历所有物理块，加载 V 向量，以 softmax 后的注意力权重做加权求和，得到最终的 attention 输出。

### V1 与 V2 的区别

`paged_attention_v1.cu` 适用于序列较短的场景，一个 thread block 处理一个完整序列的所有块。当序列非常长时，单个 thread block 的工作量过大，此时 `paged_attention_v2.cu` 登场——它将一个序列的块分成多个 partition，每个 partition 由一个 thread block 处理，最后通过 `merge_attn_states.cu` 中的归约内核合并各 partition 的结果。

## 5.4 AttentionMetadata：内核的输入契约

Attention 内核需要知道当前 batch 中每个序列的 block table、序列长度和 slot 映射等信息。这些信息被打包在 **AttentionMetadata** 数据结构中，由 `gpu_model_runner.py` 在每次推理前构建，传递给 attention 层。

核心字段包括：

- **`block_tables`**：形状为 `[batch_size, max_num_blocks]` 的张量，每行是一个序列的物理块 ID 列表
- **`seq_lens`**：每个序列的当前长度
- **`slot_mapping`**：将每个 token 映射到 KV Cache 中具体的物理 slot 位置，用于 prefill 阶段将新计算的 KV 写入缓存

## 5.5 Prefill 与 Decode 的分离

LLM 推理可分为两个截然不同的阶段：

**Prefill 阶段**：处理整个 prompt，一次性计算所有 prompt token 的 KV Cache。这一阶段是**计算密集型（compute-bound）**的，因为 Q、K、V 都是完整的矩阵，attention 计算退化为标准的矩阵乘法。此阶段通常使用 FlashAttention（`vllm/v1/attention/backends/flash_attn.py`）来加速，因为 FlashAttention 针对 dense attention 做了深度优化。

**Decode 阶段**：逐 token 生成，每步只有一个新 token 的 Q 向量，但需要与所有历史 token 的 KV Cache 做 attention。这一阶段是**内存密集型（memory-bound）**的，瓶颈在于从 GPU 显存中读取 KV Cache。此阶段 PagedAttention 内核大显身手，因为它的非连续访存设计不影响 decode 的吞吐。

vLLM 的调度器会将同一 batch 中的 prefill 和 decode 请求分开处理（或使用 chunked prefill 策略混合调度），确保每种请求都走最优的内核路径。在 `vllm/v1/worker/gpu_model_runner.py` 中可以看到这一分流逻辑的实现。

## 5.6 FlashAttention 与 FlashInfer 集成

虽然 vLLM 自带了 PagedAttention CUDA 内核，但在实际部署中，社区更常使用第三方高性能 attention 库：

**FlashAttention**（`flash_attn.py`）：由 Tri Dao 开发的高性能 attention 库，通过 IO-aware 的 tiling 策略最小化 HBM 访问。vLLM 的 FlashAttention 后端对原版接口做了适配，传入 paged KV cache 的布局信息，使其能在分页内存上工作。

**FlashInfer**（`flashinfer.py`）：专为 LLM serving 场景设计的 attention 库，原生支持 paged KV cache、prefix caching 和多种 attention 变体。它在 decode 阶段的小 batch 场景下表现尤为优秀。

这些后端的选择对用户透明——`selector.py` 会根据环境自动做出最优选择。

## 本章小结

Attention 算子是 vLLM 性能的最终落脚点。系统通过 `vllm/v1/attention/` 中的抽象层支持多种后端实现，包括自研的 PagedAttention CUDA 内核（`csrc/attention/`）和第三方的 FlashAttention、FlashInfer 等。PagedAttention 内核的核心技巧在于通过 block table 间接寻址实现非连续 KV Cache 上的高效 attention 计算，配合 online softmax 和 partition 归约处理长序列。Prefill 和 Decode 阶段分别走不同的内核路径，前者利用 FlashAttention 的矩阵乘法优化，后者利用 PagedAttention 的内存访问优化。`AttentionMetadata` 作为连接 Python 调度层与 CUDA 内核层的桥梁，确保了信息传递的准确与高效。
