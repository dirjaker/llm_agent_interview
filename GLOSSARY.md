# 📚 LLM & Agent 术语表

> 中英对照，面试必备术语速查
>
> 本表收录 120+ 核心术语，按领域分类，覆盖 LLM 基础、训练微调、RAG、Agent、推理部署、工程实践、多模态及前沿方向。

---

## 📖 使用说明

- **英文术语** 列为面试/论文中的标准写法，建议牢记
- **一句话解释** 快速建立直觉，深入学习请参考对应章节
- **相关章节** 指向本仓库内更详细的笔记

---

## 一、LLM 基础

| 英文术语 | 中文 | 一句话解释 | 相关章节 |
|---------|------|-----------|---------|
| Transformer | 变换器 | 基于自注意力机制的神经网络架构，是现代 LLM 的基石 | 第1章 |
| Self-Attention | 自注意力 | 让序列中每个 token 关注其他所有 token 的机制，计算相关性权重 | 第1章 |
| Multi-Head Attention | 多头注意力 | 并行运行多组注意力计算，捕获不同子空间的语义关系 | 第1章 |
| Tokenizer | 分词器 | 将原始文本拆分为 token 序列的工具，常见算法有 BPE、WordPiece、SentencePiece | 第1章 |
| Token | 词元 | 模型处理的最小文本单元，可能是子词、字符或字节 | 第1章 |
| Embedding | 嵌入 | 将离散 token 映射为连续高维向量的过程/结果 | 第1章 |
| Positional Encoding | 位置编码 | 为 Transformer 注入序列位置信息的技术（绝对/旋转/相对位置编码） | 第1章 |
| Prompt | 提示词 | 输入给 LLM 的文本指令，是与模型交互的主要方式 | 第1章 |
| System Prompt | 系统提示 | 设定模型角色、行为边界和输出格式的预设指令 | 第1章 |
| Few-Shot | 少样本 | 在提示中给出少量示例，引导模型完成任务 | 第1章 |
| Zero-Shot | 零样本 | 不给示例，直接让模型完成任务 | 第1章 |
| Chain-of-Thought (CoT) | 思维链 | 让模型逐步推理的技术，通过"让我们一步步思考"提升复杂任务准确率 | 第1章 |
| Temperature | 温度 | 控制采样随机性的参数，越高越多样，越低越确定 | 第1章 |
| Top-P (Nucleus Sampling) | 核采样 | 按概率从累积概率前 P 的 token 中采样，替代 Temperature 的另一种控制方式 | 第1章 |
| Top-K | Top-K 采样 | 仅从概率最高的 K 个 token 中采样 | 第1章 |
| Logit | 对数几率 | 模型输出层的原始分数，经 softmax 后变为概率 | 第1章 |
| Perplexity (PPL) | 困惑度 | 衡量语言模型预测能力的指标，越低越好 | 第1章 |
| Context Window | 上下文窗口 | 模型单次能处理的最大 token 数量 | 第1章 |
| Pre-training | 预训练 | 在海量无标注文本上通过自监督学习训练基础模型的阶段 | 第1章 |
| Next Token Prediction | 下一个词预测 | 自回归语言模型的核心训练目标 | 第1章 |
| Autoregressive | 自回归 | 逐个生成 token，每次将已生成序列作为输入的生成方式 | 第1章 |
| Decoding Strategy | 解码策略 | 从模型输出概率分布中选择 token 的方法（贪心/束搜索/采样） | 第1章 |
| Beam Search | 束搜索 | 维护多个候选序列的解码策略，平衡质量与多样性 | 第1章 |
| Hallucination | 幻觉 | 模型生成看似合理但事实错误的内容 | 第1章 |
| Stop Sequence | 停止序列 | 指定模型遇到特定文本时停止生成 | 第1章 |

---

## 二、训练与微调

| 英文术语 | 中文 | 一句话解释 | 相关章节 |
|---------|------|-----------|---------|
| Supervised Fine-Tuning (SFT) | 有监督微调 | 用标注好的指令-回答对训练模型遵循指令的能力 | 第2章 |
| Reinforcement Learning from Human Feedback (RLHF) | 基于人类反馈的强化学习 | 用人类偏好训练奖励模型，再用 PPO 优化语言模型 | 第2章 |
| Direct Preference Optimization (DPO) | 直接偏好优化 | 绕过奖励模型，直接从偏好数据优化策略模型的算法 | 第2章 |
| Group Relative Policy Optimization (GRPO) | 群组相对策略优化 | DeepSeek 提出的强化学习算法，无需价值模型，通过组内对比计算优势 | 第2章 |
| Proximal Policy Optimization (PPO) | 近端策略优化 | RLHF 中常用的强化学习算法，通过裁剪目标函数稳定训练 | 第2章 |
| Reward Model (RM) | 奖励模型 | 学习人类偏好、为模型输出打分的模型 | 第2章 |
| LoRA | 低秩适配 | 通过注入低秩分解矩阵来微调大模型，只训练极少量参数 | 第2章 |
| QLoRA | 量化低秩适配 | 在 4-bit 量化模型上应用 LoRA，大幅降低显存需求 | 第2章 |
| PEFT | 参数高效微调 | 一类只微调少量参数的方法总称，包括 LoRA、Adapter、Prefix Tuning 等 | 第2章 |
| Adapter | 适配器 | 在 Transformer 层间插入小型可训练模块的微调方法 | 第2章 |
| Prefix Tuning | 前缀调优 | 在输入前添加可训练的连续向量（虚拟 token）的微调方法 | 第2章 |
| Prompt Tuning | 提示调优 | 仅优化连续提示嵌入的极轻量微调方法 | 第2章 |
| Instruction Tuning | 指令调优 | 用多样化的指令数据训练模型遵循各种指令 | 第2章 |
| Alignment | 对齐 | 让模型行为符合人类价值观和期望的过程 | 第2章 |
| Constitutional AI (CAI) | 宪法 AI | Anthropic 提出的用一组规则（宪法）进行自我改进的对齐方法 | 第2章 |
| Distillation | 蒸馏 | 用大模型（教师）的输出训练小模型（学生）以转移知识 | 第2章 |
| Continual Pre-training | 持续预训练 | 在已有模型基础上用领域数据继续预训练 | 第2章 |
| Overfitting | 过拟合 | 模型在训练数据上表现好但泛化能力差 | 第2章 |
| Learning Rate | 学习率 | 控制模型参数更新步长的超参数 | 第2章 |
| Gradient Accumulation | 梯度累积 | 多个 mini-batch 累积梯度后再更新，模拟大 batch size | 第2章 |
| Mixed Precision Training | 混合精度训练 | 使用 FP16/BF16 与 FP32 混合计算以加速训练并减少显存 | 第2章 |
| DeepSpeed ZeRO | DeepSpeed ZeRO | 微软的分布式训练优化技术，通过分片减少每个 GPU 的显存占用 | 第2章 |

---

## 三、RAG（检索增强生成）

| 英文术语 | 中文 | 一句话解释 | 相关章节 |
|---------|------|-----------|---------|
| Retrieval-Augmented Generation (RAG) | 检索增强生成 | 在生成时先检索外部知识库，将结果注入上下文以提升回答准确性 | 第3章 |
| Chunking | 分块 | 将长文档切分为适合检索的小段落 | 第3章 |
| Vector Database (Vector DB) | 向量数据库 | 存储和检索高维向量的专用数据库，如 Milvus、Pinecone、Chroma | 第3章 |
| Similarity Search | 相似度搜索 | 在向量空间中找到与查询最相似的文档片段 | 第3章 |
| Cosine Similarity | 余弦相似度 | 衡量两个向量方向相似程度的指标 | 第3章 |
| Reranking | 重排序 | 对初步检索结果用更精细的模型重新排序以提升相关性 | 第3章 |
| Hybrid Search | 混合搜索 | 结合关键词（BM25）和语义（向量）搜索以兼顾精确匹配与语义理解 | 第3章 |
| BM25 | BM25 | 经典的基于词频的文档检索算法 | 第3章 |
| Reciprocal Rank Fusion (RRF) | 互惠排名融合 | 合并多个排序列表的融合算法 | 第3章 |
| Embedding Model | 嵌入模型 | 将文本转换为向量表示的专用模型（如 text-embedding-3-small） | 第3章 |
| Knowledge Base | 知识库 | RAG 系统中存储外部文档和数据的集合 | 第3章 |
| Agentic RAG | 智能体驱动的 RAG | Agent 自主决定何时检索、检索什么、是否需要多轮检索 | 第3章 |
| GraphRAG | 图 RAG | 基于知识图谱的 RAG，利用实体关系提升检索质量 | 第3章 |
| Context Window Injection | 上下文注入 | 将检索到的内容塞入模型上下文的技术 | 第3章 |
| Metadata Filtering | 元数据过滤 | 利用文档标签（时间、来源等）缩小检索范围 | 第3章 |
| Semantic Chunking | 语义分块 | 基于语义边界（而非固定长度）进行文档切分 | 第3章 |

---

## 四、Agent（智能体）

| 英文术语 | 中文 | 一句话解释 | 相关章节 |
|---------|------|-----------|---------|
| Agent | 智能体 | 能自主感知环境、做出决策并执行动作的 LLM 系统 | 第4章 |
| ReAct | 推理+行动 | 交替进行推理（Reasoning）和行动（Acting）的 Agent 范式 | 第4章 |
| Function Calling | 函数调用 | LLM 结构化地输出函数名和参数，由外部系统执行 | 第4章 |
| Tool Use | 工具使用 | Agent 调用外部工具（搜索、代码执行、API 等）完成任务的能力 | 第4章 |
| Model Context Protocol (MCP) | 模型上下文协议 | Anthropic 提出的标准化 LLM 与外部工具/数据源交互的开放协议 | 第4章 |
| Planning | 规划 | Agent 将复杂任务分解为可执行子步骤的能力 | 第4章 |
| Memory | 记忆 | Agent 存储和检索历史交互信息的机制（短期/长期/工作记忆） | 第4章 |
| Short-term Memory | 短期记忆 | 当前对话上下文内的信息保持 | 第4章 |
| Long-term Memory | 长期记忆 | 跨会话持久化的知识和经验 | 第4章 |
| Working Memory | 工作记忆 | Agent 在任务执行过程中维护的临时状态 | 第4章 |
| Multi-Agent System | 多智能体系统 | 多个 Agent 协作完成任务的架构 | 第4章 |
| Orchestrator | 编排器 | 协调多个 Agent 或工具调用顺序的控制组件 | 第4章 |
| Reflection | 反思 | Agent 评估自身输出质量并自我纠正的机制 | 第4章 |
| Self-Correction | 自我纠错 | Agent 检测并修正自身错误的能力 | 第4章 |
| Tool Schema | 工具模式 | 描述工具功能、参数和返回值的 JSON Schema | 第4章 |
| Agentic Workflow | 智能体工作流 | 由 Agent 驱动的自动化任务执行流程 | 第4章 |
| LangChain | LangChain | 流行的 LLM 应用开发框架，提供链式调用和 Agent 抽象 | 第4章 |
| LangGraph | LangGraph | LangChain 团队的图状态机框架，用于构建复杂 Agent 工作流 | 第4章 |
| CrewAI | CrewAI | 基于角色的多智能体协作框架 | 第4章 |
| OpenAI Assistants API | OpenAI 助手 API | OpenAI 提供的内置工具（代码解释器/文件检索/函数调用）的 Agent API | 第4章 |

---

## 五、推理与部署

| 英文术语 | 中文 | 一句话解释 | 相关章节 |
|---------|------|-----------|---------|
| Inference | 推理 | 模型训练完成后进行预测/生成的过程 | 第5章 |
| Quantization | 量化 | 将模型权重从高精度（FP32）压缩到低精度（INT8/INT4）以减少显存和加速 | 第5章 |
| GPTQ | GPTQ | 一种训练后量化方法，逐层量化权重，适合 GPU 推理 | 第5章 |
| AWQ | 激活感知权重量化 | 基于激活分布优化量化精度的方法，质量优于 GPTQ | 第5章 |
| GGUF (GGML) | GGUF | llama.cpp 使用的量化格式，适合 CPU/边缘设备推理 | 第5章 |
| KV Cache | KV 缓存 | 推理时缓存已计算的 Key/Value 张量以避免重复计算 | 第5章 |
| FlashAttention | 闪速注意力 | 通过分块计算和 IO 优化实现高效注意力的算法 | 第5章 |
| PagedAttention | 分页注意力 | vLLM 使用的类似操作系统虚拟内存的 KV Cache 管理技术 | 第5章 |
| vLLM | vLLM | 高性能 LLM 推理引擎，支持 PagedAttention 和连续批处理 | 第5章 |
| TensorRT-LLM | TensorRT-LLM | NVIDIA 的高性能 LLM 推理优化引擎 | 第5章 |
| Continuous Batching | 连续批处理 | 动态加入/移除请求的批处理策略，提高吞吐量 | 第5章 |
| Speculative Decoding | 投机解码 | 用小模型快速生成草稿，大模型验证，加速推理 | 第5章 |
| Tensor Parallelism | 张量并行 | 将单层的权重矩阵分片到多个 GPU 上并行计算 | 第5章 |
| Pipeline Parallelism | 流水线并行 | 将模型不同层分配到不同 GPU 上 | 第5章 |
| Model Parallelism | 模型并行 | 将模型拆分到多个设备上运行的策略总称 | 第5章 |
| ONNX Runtime | ONNX 运行时 | 微软的跨平台模型推理加速框架 | 第5章 |
| TorchScript | TorchScript | PyTorch 的 JIT 编译格式，用于生产环境部署 | 第5章 |
| Latency | 延迟 | 从请求发出到收到完整响应的时间 | 第5章 |
| Throughput | 吞吐量 | 单位时间内能处理的请求数或 token 数 | 第5章 |
| Time to First Token (TTFT) | 首 token 时间 | 从请求发出到收到第一个生成 token 的延迟 | 第5章 |
| Tokens per Second (TPS) | 每秒 token 数 | 衡量生成速度的核心指标 | 第5章 |
| Prefill | 预填充 | 处理输入 prompt 并计算 KV Cache 的阶段（计算密集） | 第5章 |
| Decode | 解码 | 逐 token 生成输出的阶段（内存密集） | 第5章 |

---

## 六、工程实践

| 英文术语 | 中文 | 一句话解释 | 相关章节 |
|---------|------|-----------|---------|
| Streaming | 流式输出 | 模型逐 token 实时返回结果，而非等待全部生成完毕 | 第6章 |
| Server-Sent Events (SSE) | 服务端推送事件 | 基于 HTTP 的单向流式传输协议，常用于 LLM 流式输出 | 第6章 |
| Rate Limiting | 速率限制 | 控制 API 请求频率的机制，防止过载 | 第6章 |
| Circuit Breaker | 熔断器 | 当下游服务故障率超阈值时自动切断请求以保护系统 | 第6章 |
| Idempotency | 幂等性 | 同一请求多次执行结果相同，防止重复操作 | 第6章 |
| Retry with Backoff | 退避重试 | 请求失败后等待逐渐增长的时间再重试 | 第6章 |
| Prompt Injection | 提示注入 | 攻击者通过精心构造的输入劫持模型行为 | 第6章 |
| Guard Rails | 护栏 | 限制和引导模型行为的安全机制 | 第6章 |
| Content Filtering | 内容过滤 | 检测和拦截不当内容的机制 | 第6章 |
| Token Counting | Token 计数 | 统计输入/输出 token 数量，用于成本控制和上下文管理 | 第6章 |
| API Gateway | API 网关 | 统一管理 API 流量、认证、限流和路由的中间层 | 第6章 |
| Observability | 可观测性 | 通过日志、指标和追踪监控系统运行状态 | 第6章 |
| Tracing | 链路追踪 | 记录请求在各组件间的完整调用路径 | 第6章 |
| Evaluation (Eval) | 评估 | 用指标和数据集系统性衡量模型质量 | 第6章 |
| A/B Testing | A/B 测试 | 并行对比不同模型/提示版本效果的实验方法 | 第6章 |
| Semantic Versioning | 语义版本 | 用 MAJOR.MINOR.PATCH 格式管理 Prompt/模型版本 | 第6章 |

---

## 七、多模态

| 英文术语 | 中文 | 一句话解释 | 相关章节 |
|---------|------|-----------|---------|
| Vision Encoder | 视觉编码器 | 将图像转换为模型可理解的向量表示的模块 | 第7章 |
| CLIP | 对比语言-图像预训练 | OpenAI 的图文对比学习模型，建立视觉与语言的对齐 | 第7章 |
| Vision Transformer (ViT) | 视觉 Transformer | 将图像切分为 patch 后用 Transformer 处理的架构 | 第7章 |
| Optical Character Recognition (OCR) | 光学字符识别 | 从图像中识别和提取文字 | 第7章 |
| Text-to-Speech (TTS) | 文字转语音 | 将文本转换为自然语音 | 第7章 |
| Speech-to-Text (STT) | 语音转文字 | 将语音转换为文本 | 第7章 |
| Voice Activity Detection (VAD) | 语音活动检测 | 检测音频中哪些片段包含人声 | 第7章 |
| Multimodal LLM | 多模态大语言模型 | 能同时处理文本、图像、音频等多种模态的 LLM | 第7章 |
| Grounding | 接地/锚定 | 将模型输出与可验证的外部信息源关联 | 第7章 |
| Image Generation | 图像生成 | 根据文本描述生成图像（如 DALL·E、Stable Diffusion） | 第7章 |
| Diffusion Model | 扩散模型 | 通过逐步去噪过程生成图像/音频的生成模型 | 第7章 |

---

## 八、前沿方向

| 英文术语 | 中文 | 一句话解释 | 相关章节 |
|---------|------|-----------|---------|
| World Model | 世界模型 | 对物理世界运行规律建模的 AI 系统 | 第8章 |
| Synthetic Data | 合成数据 | 由模型自动生成的训练数据，用于补充真实数据不足 | 第8章 |
| Model Merging | 模型合并 | 将多个微调模型的参数融合为单一模型（SLERP/TIES/DARE） | 第8章 |
| Test-Time Compute (TTC) | 测试时计算 | 在推理时投入更多计算（如多次采样/搜索）提升输出质量 | 第8章 |
| Mixture of Experts (MoE) | 混合专家 | 通过路由机制激活部分专家网络，平衡模型容量与计算效率 | 第8章 |
| State Space Model (SSM) | 状态空间模型 | 如 Mamba，线性复杂度的序列建模架构，替代 Transformer 的新方向 | 第8章 |
| Long Context | 长上下文 | 支持 100K+ 甚至百万 token 上下文的技术趋势 | 第8章 |
| Retrieval-Interleaved Generation | 检索交织生成 | 边检索边生成，在生成过程中动态调用检索 | 第8章 |
| AI Agent Framework | AI 智能体框架 | 构建自主 Agent 的软件框架和工具链 | 第8章 |
| Structured Output | 结构化输出 | 让模型输出符合 JSON Schema 等结构的数据格式 | 第8章 |
| Knowledge Distillation | 知识蒸馏 | 用大模型的输出训练小模型的压缩技术 | 第8章 |
| Reinforcement Learning from AI Feedback (RLAIF) | AI 反馈强化学习 | 用 AI（而非人类）作为偏好标注者的训练方法 | 第8章 |
| Reward Hacking | 奖励攻击 | 模型找到奖励模型的漏洞，获得高分但实际质量差 | 第8章 |
| Sycophancy | 讨好倾向 | 模型倾向于迎合用户而非给出正确答案的问题 | 第8章 |
| Scaling Law | 缩放定律 | 揭示模型性能与参数量/数据量/计算量关系的幂律规律 | 第8章 |
| Emergent Ability | 涌现能力 | 模型规模达到一定阈值后突然出现的新能力 | 第8章 |
| Reasoning Model | 推理模型 | 专门强化复杂推理能力的模型（如 o1、DeepSeek-R1） | 第8章 |
| Native Tool Calling | 原生工具调用 | 模型架构层面内置的工具调用能力，而非通过提示实现 | 第8章 |

---

## 九、常见缩写速查

| 缩写 | 全称 | 中文 |
|------|------|------|
| LLM | Large Language Model | 大语言模型 |
| NLP | Natural Language Processing | 自然语言处理 |
| RAG | Retrieval-Augmented Generation | 检索增强生成 |
| RLHF | Reinforcement Learning from Human Feedback | 基于人类反馈的强化学习 |
| MCP | Model Context Protocol | 模型上下文协议 |
| API | Application Programming Interface | 应用编程接口 |
| SDK | Software Development Kit | 软件开发工具包 |
| GPU | Graphics Processing Unit | 图形处理单元 |
| TPU | Tensor Processing Unit | 张量处理单元 |
| FLOPS | Floating Point Operations | 浮点运算次数 |
| DNN | Deep Neural Network | 深度神经网络 |
| NAS | Neural Architecture Search | 神经架构搜索 |

---

> 💡 **面试提示**：掌握术语只是第一步，能用自己的话解释原理并结合实际案例才是加分项。建议每个术语至少能说出 **是什么 → 为什么重要 → 怎么用** 三层。
