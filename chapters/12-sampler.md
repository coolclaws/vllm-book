# 第 12 章：采样器

> "Randomness is not disorder. It is a different kind of order." —— Edward Lorenz。采样器正是在确定性的模型输出与多样性的文本生成之间，构建秩序的关键环节。

## 12.1 从 logits 到 token：采样器的角色

模型的前向传播结束后，输出的是一个形状为 `[batch_size, vocab_size]` 的 logits 张量——每个位置对应词表中一个 token 的未归一化得分。采样器（Sampler）的职责是将这些原始的 logits 转化为最终的 token 选择。这个看似简单的过程，实际上涉及温度缩放、概率截断、惩罚机制、beam search 等一系列精密操作。

采样器的核心实现位于 `vllm/model_executor/layers/` 目录下。在 vLLM 的 v1 架构中，采样逻辑被集成到 GPU Model Runner 的执行流程中，由 `vllm/v1/sample/sampler.py` 协调整个采样管线。

## 12.2 Logits 处理管线

在实际采样之前，原始 logits 需要经过一系列处理步骤。`vllm/model_executor/layers/logits_processor.py` 中的 `LogitsProcessor` 负责将模型最后一层的 hidden states 投影到词表维度，生成 logits。随后，`vllm/logits_process.py` 中定义了各类 logits 修改器，按顺序对 logits 进行变换。

**温度缩放（Temperature Scaling）** 是最基础的操作。logits 除以温度系数 `T`，得到 `logits / T`。当 `T < 1` 时，分布变得更尖锐，模型倾向于选择高概率的 token；当 `T > 1` 时，分布更平坦，生成更具随机性；`T = 0` 等价于贪心解码（greedy decoding），直接选择概率最高的 token。

**Top-k 采样** 只保留概率最高的 k 个 token，将其余 token 的概率置为 0。这避免了从极低概率的 token 中采样导致的文本退化。

**Top-p（核采样, Nucleus Sampling）** 则是动态地确定截断阈值：将 token 按概率从大到小排序，累计概率达到 p 时截断。与 Top-k 不同，Top-p 的候选集大小随上下文动态变化——在模型置信度高的位置，可能只保留少数几个 token；在不确定性高的位置，候选集会自然扩大。

**Min-p 采样** 是较新的采样策略，它设置一个相对阈值：只保留概率大于 `min_p * max_prob` 的 token，其中 `max_prob` 是当前最高概率。这种方式结合了 Top-k 的简洁和 Top-p 的自适应性。

## 12.3 惩罚机制

为了控制生成文本的重复性，vLLM 实现了三种惩罚机制，它们在采样前对 logits 进行调整：

**重复惩罚（Repetition Penalty）** 检查已生成的 token 序列，对出现过的 token 的 logits 施加惩罚。具体来说，如果一个 token 的 logit 为正，则除以惩罚系数；如果为负，则乘以惩罚系数。这确保了惩罚方向始终是降低该 token 被选中的概率。

**存在惩罚（Presence Penalty）** 更为直接——只要一个 token 在已生成序列中出现过，就从其 logit 中减去固定值。它不关心出现了多少次，只关心"是否出现过"，鼓励模型探索新的词汇。

**频率惩罚（Frequency Penalty）** 则与出现次数成正比，从 logit 中减去 `penalty * count`。出现次数越多的 token 受到越重的惩罚，有效地抑制高频重复模式。

这三种惩罚可以组合使用，用户通过 `SamplingParams` 配置类（定义在 `vllm/sampling_params.py` 中）灵活控制。

## 12.4 采样策略

经过 logits 处理和惩罚调整后，vLLM 支持多种采样策略：

**贪心采样（Greedy Sampling）** 即 `temperature=0`，直接通过 `argmax` 选择概率最高的 token。结果完全确定性，适合需要一致性的场景。

**随机采样（Random Sampling）** 基于处理后的概率分布使用多项式采样（multinomial sampling）选择 token。通过 `torch.multinomial` 或等效的 GPU kernel 实现，每次调用可能产生不同的结果。

**Beam Search** 维护多个候选序列（beam），每一步扩展所有 beam 并保留累积概率最高的 k 个。vLLM 中 beam search 与 PagedAttention 的 copy-on-write 机制紧密配合——多个 beam 可以共享相同的 KV Cache 块，只在产生分歧时才复制，大幅减少内存开销。

## 12.5 可复现采样与种子机制

在调试和评估场景中，生成的可复现性（reproducibility）至关重要。vLLM 通过 `SamplingParams` 中的 `seed` 参数支持确定性采样。当设置了 seed 时，采样器使用该 seed 初始化随机数生成器，确保相同的输入和参数总是产生完全相同的输出。

在批量处理中，不同请求可能有不同的 seed 设置。vLLM 会为每个请求维护独立的随机状态，避免请求之间的随机性相互干扰。

## 12.6 Log Probabilities 计算

许多应用场景需要获取采样过程中的概率信息。当用户设置 `logprobs` 参数时，vLLM 不仅返回采样得到的 token，还返回该 token 的对数概率以及概率最高的 top-k 个候选 token 及其对数概率。

此外，vLLM 还支持 **prompt logprobs** 计算——为输入 prompt 中的每个 token 计算其在模型眼中的对数概率。`vllm/logprobs.py` 中集中了 log probabilities 的计算逻辑。这对于评估模型的困惑度（perplexity）、进行模型选择、以及构建基于概率的过滤和排序系统都非常有用。

## 12.7 SamplerOutput 与下游集成

采样器的输出被封装为结构化的 `SamplerOutput` 对象，包含采样得到的 token ID、对数概率（如果请求了 logprobs）、以及相关元数据。这个输出被传回引擎层，与对应的请求关联，最终通过 API 层返回给用户。

整个采样管线的高效性对系统吞吐量有直接影响。虽然采样本身的计算量远小于模型前向传播，但在大批量和大词表的场景下，Top-k/Top-p 的排序操作和多项式采样仍然是不可忽视的开销。vLLM 通过在 GPU 上原生实现这些操作，避免了 GPU-CPU 之间的数据传输，保证了采样环节不成为性能瓶颈。

## 本章小结

采样器是 vLLM 推理管线的最后一环，将模型输出的原始 logits 转化为用户看到的生成文本。从温度缩放、Top-k/Top-p 截断到重复/存在/频率惩罚，每一步 logits 处理都精确地塑造着生成分布。贪心、随机、beam search 三种策略覆盖了从确定性到探索性的全部需求谱系。结合种子可复现机制和 log probabilities 输出，vLLM 的采样器为上层应用提供了完整而灵活的生成控制能力。
