# 📋 LLM 应用 & Agent 面试速查表

> 面试前 30 分钟翻一遍，覆盖 26 个专题核心考点

---

## 目录

| 编号 | 专题 | 页内跳转 |
|------|------|----------|
| 01 | Transformer 架构基础 | [→](#ch01) |
| 02 | 预训练与微调 | [→](#ch02) |
| 03 | Prompt Engineering | [→](#ch03) |
| 04 | RAG 检索增强生成 | [→](#ch04) |
| 05 | Agent 架构与设计 | [→](#ch05) |
| 06 | ReAct & 工具调用 | [→](#ch06) |
| 07 | Function Calling | [→](#ch07) |
| 08 | 多智能体协作 | [→](#ch08) |
| 09 | 记忆机制 | [→](#ch09) |
| 10 | 向量数据库 | [→](#ch10) |
| 11 | Embedding 模型 | [→](#ch11) |
| 12 | 评估与基准测试 | [→](#ch12) |
| 13 | RLHF 与对齐 | [→](#ch13) |
| 14 | 推理优化 | [→](#ch14) |
| 15 | 模型量化 | [→](#ch15) |
| 16 | 分布式训练 | [→](#ch16) |
| 17 | 安全与红队测试 | [→](#ch17) |
| 18 | 代码生成 Agent | [→](#ch18) |
| 19 | 多模态模型 | [→](#ch19) |
| 20 | 长上下文处理 | [→](#ch20) |
| 21 | RAG 进阶 | [→](#ch21) |
| 22 | 生产部署 | [→](#ch22) |
| 23 | AI 编程与开发工具 | [→](#ch23) |
| 24 | MCP 协议 | [→](#ch24) |
| 25 | 扩散模型基础 | [→](#ch25) |
| 26 | 前沿趋势 | [→](#ch26) |

---

<a id="ch01"></a>
## 01 - Transformer 架构基础

**核心考点：**

- Self-Attention: `Attention(Q,K,V) = softmax(QK^T / √d_k) V`
- Multi-Head Attention: 多个 head 并行计算，拼接后线性变换
- 三种架构：Encoder-Only (BERT)、Decoder-Only (GPT)、Encoder-Decoder (T5)
- 位置编码：绝对位置编码 vs RoPE（旋转位置编码）vs ALiBi

| 架构 | 代表模型 | 适用任务 |
|------|----------|----------|
| Encoder-Only | BERT, RoBERTa | 分类、NER、匹配 |
| Decoder-Only | GPT, LLaMA, Qwen | 生成、对话、推理 |
| Encoder-Decoder | T5, BART | 翻译、摘要 |

**⚠️ 常见坑：**
- 为什么除以 √d_k？→ 防止 softmax 梯度消失（点积过大）
- Decoder-Only 的 causal mask 确保只能看到左侧 token
- LayerNorm 位置：Pre-Norm（现代主流）vs Post-Norm（原始 Transformer）

---

<a id="ch02"></a>
## 02 - 预训练与微调

**核心考点：**

- 预训练目标：CLM（因果语言模型）、MLM（掩码语言模型）
- 全量微调 vs 参数高效微调（PEFT）
- LoRA 原理：冻结原始权重，注入低秩矩阵 `ΔW = BA`，B∈R^{d×r}, A∈R^{r×k}

| 微调方法 | 参数量 | 显存需求 | 效果 |
|----------|--------|----------|------|
| 全量微调 | 100% | 极高 | 最好 |
| LoRA | 0.1-1% | 低 | 接近全量 |
| QLoRA | 0.1-1% | 更低 | 略低于LoRA |
| Prefix Tuning | ~0.1% | 低 | 一般 |
| Adapter | 1-5% | 中 | 良好 |

**⚠️ 常见坑：**
- LoRA rank 选择：通常 r=8~64，太小欠拟合，太大过拟合
- QLoRA = 4bit量化 + LoRA，显著降低显存但精度略降
- 学习率：微调通常 1e-5 ~ 5e-5，远小于预训练

**LoRA 配置代码：**
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                        # 低秩维度
    lora_alpha=32,               # 缩放系数
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, config)
# 可训练参数占比
model.print_trainable_parameters()
```

---

<a id="ch03"></a>
## 03 - Prompt Engineering

**核心考点：**

- Zero-shot / Few-shot / Chain-of-Thought (CoT)
- System Prompt 定义角色、约束、输出格式
- 结构化输出：JSON mode、XML tag、正则约束

| 技术 | 说明 | 适用场景 |
|------|------|----------|
| Zero-shot | 无示例直接提问 | 简单任务 |
| Few-shot | 提供 2-5 个示例 | 格式要求高 |
| CoT | 引导逐步推理 | 数学/逻辑推理 |
| Self-Consistency | 多次采样投票 | 高可靠性需求 |
| ReAct | 思考+行动交替 | 需要工具的任务 |

**⚠️ 常见坑：**
- Few-shot 示例顺序影响结果（recency bias）
- Prompt 过长会挤占上下文窗口，注意 token 预算
- "Let's think step by step" 是 Zero-shot CoT 的经典触发词

---

<a id="ch04"></a>
## 04 - RAG 检索增强生成

**核心考点：**

- 基本流程：Query → 检索 → 增强 Prompt → LLM 生成
- 检索方式：稀疏（BM25）、稠密（Embedding）、混合
- 分块策略：固定大小、语义分块、递归分块

| 组件 | 关键参数 | 注意事项 |
|------|----------|----------|
| Chunk Size | 256-1024 tokens | 太小丢上下文，太大噪音多 |
| Overlap | 10-20% | 防止跨块信息丢失 |
| Top-K | 3-10 | 太多引入噪音 |
| Reranker | Cross-Encoder | 比 Bi-Encoder 精度高但慢 |

**RAG Pipeline 伪代码：**
```python
def rag_pipeline(query):
    # 1. 检索
    chunks = vector_db.search(query, top_k=5)
    # 2. 重排序（可选）
    chunks = reranker.rank(query, chunks, top_n=3)
    # 3. 构建 prompt
    context = "\n\n".join([c.text for c in chunks])
    prompt = f"""基于以下参考资料回答问题。
    
参考资料：
{context}

问题：{query}
回答："""
    # 4. 生成
    return llm.generate(prompt)
```

**⚠️ 常见坑：**
- 检索质量是 RAG 瓶颈，"Garbage in, garbage out"
- 用户 query 可能需要改写（Query Rewriting）才能有效检索
- 需要做引用溯源，验证生成内容确实来自检索结果

---

<a id="ch05"></a>
## 05 - Agent 架构与设计

**核心考点：**

- Agent = LLM + 记忆 + 工具 + 规划
- 核心循环：感知 → 推理 → 行动 → 观察
- 规划方法：任务分解、子目标设定、反思与修正

| 架构模式 | 说明 | 代表 |
|----------|------|------|
| 单 Agent | 一个LLM完成所有 | LangChain Agent |
| 多 Agent | 分工协作 | AutoGen, CrewAI |
| 层级式 | Manager 分配任务给 Worker | MetaGPT |
| 辩论式 | 多个Agent互相审查 | ChatDev |

**⚠️ 常见坑：**
- Agent 循环需要设置最大迭代次数，防止无限循环
- 工具描述质量直接影响 Agent 能否正确调用工具
- 错误处理：工具调用失败时 Agent 应有 fallback 策略

---

<a id="ch06"></a>
## 06 - ReAct & 工具调用

**核心考点：**

- ReAct = Reasoning + Acting 交替进行
- 每步输出：Thought → Action → Observation
- 与纯 CoT 的区别：ReAct 能与外部环境交互

**ReAct 循环代码：**
```python
def react_loop(query, tools, max_steps=10):
    scratchpad = f"Question: {query}\n"
    for step in range(max_steps):
        # LLM 生成 Thought + Action
        response = llm.generate(REACT_PROMPT + scratchpad)
        thought, action = parse(response)
        scratchpad += f"Thought: {thought}\n"
        
        if action.tool == "finish":
            return action.input
        
        # 执行工具
        observation = tools[action.tool].run(action.input)
        scratchpad += f"Action: {action}\nObservation: {observation}\n"
    
    return "达到最大步数限制"
```

**⚠️ 常见坑：**
- ReAct 需要较强的模型能力，小模型容易格式混乱
- Observation 过长会撑爆上下文，需要截断策略
- 解析 Action 时建议用结构化格式而非纯文本

---

<a id="ch07"></a>
## 07 - Function Calling

**核心考点：**

- LLM 输出结构化的函数调用请求，由外部执行
- 与 ReAct 的区别：更标准化、JSON schema 定义、平台原生支持
- 关键：函数描述（description）和参数定义要清晰准确

| 平台 | 格式 | 特点 |
|------|------|------|
| OpenAI | tools + function | 成熟稳定 |
| Anthropic | tool_use | XML风格 |
| Gemini | function_declarations | Google生态 |

**⚠️ 常见坑：**
- 并行调用 vs 串行调用：模型可能一次返回多个 function call
- 函数描述写得不清楚是 function calling 失败的首要原因
- 需要处理模型"幻觉"调用不存在的函数的情况

---

<a id="ch08"></a>
## 08 - 多智能体协作

**核心考点：**

- 通信模式：共享黑板、消息传递、层级指令
- 角色分工：Planner / Executor / Critic / Coordinator
- 常见框架：AutoGen、CrewAI、LangGraph、MetaGPT

| 协作模式 | 优点 | 缺点 |
|----------|------|------|
| 辩论式 | 提高准确性 | Token 消耗大 |
| 流水线式 | 分工明确 | 延迟高 |
| 动态分配 | 灵活 | 协调复杂 |

**⚠️ 常见坑：**
- Agent 数量不是越多越好，通信开销会成为瓶颈
- 需要明确终止条件，防止 Agent 间无限对话
- 角色 prompt 要有明确边界，避免职责重叠

---

<a id="ch09"></a>
## 09 - 记忆机制

**核心考点：**

- 短期记忆：对话上下文（context window）
- 长期记忆：向量数据库存储历史摘要
- 工作记忆：当前任务的中间状态

| 记忆类型 | 存储方式 | 生命周期 |
|----------|----------|----------|
| 感知记忆 | 原始输入 | 单次交互 |
| 短期记忆 | 对话历史 | 单会话 |
| 长期记忆 | 向量DB/文件 | 持久化 |
| 情景记忆 | 事件摘要 | 跨会话 |

**⚠️ 常见坑：**
- 记忆检索的召回率和精确率需要权衡
- 压缩记忆（摘要）可能丢失关键细节
- 上下文窗口满了之后需要滑动窗口或摘要策略

---

<a id="ch10"></a>
## 10 - 向量数据库

**核心考点：**

- 核心操作：ANN（近似最近邻）搜索
- 索引算法：HNSW、IVF、PQ（乘积量化）

| 数据库 | 特点 | 适用场景 |
|--------|------|----------|
| Milvus | 分布式、高性能 | 大规模生产 |
| Pinecone | 全托管 SaaS | 快速上手 |
| Chroma | 轻量嵌入式 | 原型开发 |
| FAISS | Meta开源库 | 研究/单机 |
| Weaviate | 支持混合搜索 | 多模态 |
| Qdrant | Rust实现 | 高性能 |

**⚠️ 常见坑：**
- HNSW 参数 M 和 ef_construction 需要根据数据量调优
- 距离度量选择：余弦相似度（归一化后等价于点积）、欧氏距离
- 向量维度不是越高越好，高维会增加存储和检索成本

---

<a id="ch11"></a>
## 11 - Embedding 模型

**核心考点：**

- 将文本映射到稠密向量空间
- 训练方式：对比学习（Contrastive Learning）
- 评估指标：MTEB 排行榜

| 模型 | 维度 | 特点 |
|------|------|------|
| text-embedding-3-large | 3072 | OpenAI 最强 |
| bge-large-zh | 1024 | 中文优选 |
| e5-mistral-7b | 4096 | LLM-based |
| jina-embeddings-v3 | 1024 | 多语言 |

**⚠️ 常见坑：**
- Query 和 Document 可能需要不同的前缀（如 "query:" vs "passage:"）
- Embedding 模型也需要针对领域微调才能达到最佳效果
- 归一化后再计算余弦相似度是标准做法

---

<a id="ch12"></a>
## 12 - 评估与基准测试

**核心考点：**

- 通用基准：MMLU、HellaSwag、HumanEval、GSM8K
- LLM 评估方法：自动评估（BLEU/ROUGE）、LLM-as-Judge、人工评估
- RAG 评估指标：Context Relevance、Answer Faithfulness、Answer Relevance

| 基准 | 测试能力 | 难度 |
|------|----------|------|
| MMLU | 多学科知识 | 中 |
| HumanEval | 代码生成 | 中高 |
| GSM8K | 数学推理 | 中 |
| MATH | 高等数学 | 高 |
| ARC | 科学推理 | 中 |

**⚠️ 常见坑：**
- 数据污染（Data Contamination）：测试集可能在训练数据中出现过
- BLEU/ROUGE 不适合评估 LLM 开放式生成
- LLM-as-Judge 存在位置偏见和自我偏见

---

<a id="ch13"></a>
## 13 - RLHF 与对齐

**核心考点：**

- 训练流程：SFT → Reward Model → PPO/DPO
- RLHF：用人类偏好训练奖励模型，再用 PPO 优化策略
- DPO：直接从偏好数据优化，无需单独训练奖励模型

| 方法 | 是否需要 RM | 训练复杂度 | 效果 |
|------|-------------|------------|------|
| RLHF (PPO) | 是 | 高 | 强 |
| DPO | 否 | 低 | 接近RLHF |
| KTO | 否 | 低 | 略低于DPO |
| ORPO | 否 | 低 | 良好 |

**DPO 损失函数：**
```
L_DPO = -E[log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

**⚠️ 常见坑：**
- DPO 对偏好数据质量非常敏感
- β 参数控制偏离参考策略的程度，过大则过于保守
- 对齐税（Alignment Tax）：对齐可能降低模型在某些任务上的表现

---

<a id="ch14"></a>
## 14 - 推理优化

**核心考点：**

- KV Cache：缓存历史 token 的 K/V，避免重复计算
- FlashAttention：IO 感知的精确注意力算法，减少 HBM 访问
- PagedAttention (vLLM)：类似操作系统虚拟内存管理 KV Cache

| 技术 | 加速比 | 是否精确 |
|------|--------|----------|
| FlashAttention | 2-4x | ✅ 精确 |
| PagedAttention | 2-4x | ✅ 精确 |
| Speculative Decoding | 2-3x | ✅ 精确 |
| Continuous Batching | 吞吐量↑ | ✅ 精确 |

**⚠️ 常见坑：**
- Prefill 阶段是 compute-bound，Decode 阶段是 memory-bound
- Speculative Decoding 的关键是 draft model 与 target model 分布接近
- Continuous Batching 提升吞吐但不降低单请求延迟

---

<a id="ch15"></a>
## 15 - 模型量化

**核心考点：**

- 量化精度：FP16 → INT8 → INT4 → GPTQ/AWQ/GGUF
- 权重量化 vs 权重-激活量化
- 后训练量化（PTQ）vs 量化感知训练（QAT）

| 方法 | 精度 | 速度 | 适用 |
|------|------|------|------|
| GPTQ | INT4 | 中 | GPU |
| AWQ | INT4 | 快 | GPU |
| GGUF | 多种 | 中 | CPU/GPU |
| bitsandbytes | INT8/INT4 | 中 | GPU |

**⚠️ 常见坑：**
- INT4 量化对大模型（>7B）效果好，小模型精度损失明显
- 量化后需要重新评估目标任务的精度
- 混合精度量化（关键层保持高精度）效果更好

---

<a id="ch16"></a>
## 16 - 分布式训练

**核心考点：**

- 数据并行（DP/DDP）：每个 GPU 持有完整模型，分数据
- 模型并行（Tensor Parallel）：将权重矩阵切分到多个 GPU
- 流水线并行（Pipeline Parallel）：将模型层分到不同 GPU
- ZeRO 优化器：分片存储优化器状态/梯度/参数

| 策略 | 显存节省 | 通信开销 | 适用场景 |
|------|----------|----------|----------|
| DDP | 无 | 中 | 模型放得下单卡 |
| ZeRO-1/2/3 | 逐步增大 | 逐步增大 | 大模型训练 |
| TP | 大 | 高 | 单层超大 |
| PP | 大 | 中 | 模型很深 |

**⚠️ 常见坑：**
- DeepSpeed ZeRO-3 通信开销大，不一定比 TP 快
- FSDP (PyTorch 原生) 是 ZeRO-3 的等价实现
- 混合并行（TP+PP+DP）是超大模型训练的标配

---

<a id="ch17"></a>
## 17 - 安全与红队测试

**核心考点：**

- 攻击类型：Prompt Injection（直接/间接）、越狱（Jailbreak）、数据泄露
- 防御：输入过滤、输出审查、系统 prompt 隔离、权限最小化
- 间接注入：恶意内容嵌入工具返回结果或检索文档中

| 攻击类型 | 危害 | 防御策略 |
|----------|------|----------|
| 直接注入 | 绕过限制 | 输入过滤 + 输出审查 |
| 间接注入 | 执行恶意指令 | 内容消毒 + 隔离 |
| 越狱 | 泄露系统提示 | 多层防御 |
| 数据投毒 | 模型行为偏移 | 训练数据审查 |

**⚠️ 常见坑：**
- 间接 Prompt Injection 是 RAG/Agent 场景的最大安全风险
- 系统 prompt 不应包含真正的机密信息
- 安全和能力之间需要平衡，过度限制会导致体验下降

---

<a id="ch18"></a>
## 18 - 代码生成 Agent

**核心考点：**

- 代码生成流程：理解需求 → 规划 → 生成 → 测试 → 修复
- 关键能力：代码补全、调试、重构、测试生成
- 代表工具：GitHub Copilot、Cursor、Devin、SWE-Agent

| 评估指标 | 说明 |
|----------|------|
| pass@k | k 次采样中至少一次通过率 |
| CodeBLEU | 代码质量评分 |
| 人工审查率 | 需要人工修改的比例 |

**⚠️ 常见坑：**
- 代码幻觉：生成看起来正确但有 bug 的代码
- 上下文长度限制影响理解大型代码库的能力
- 安全性：生成的代码可能引入漏洞（SQL注入等）

---

<a id="ch19"></a>
## 19 - 多模态模型

**核心考点：**

- 架构：视觉编码器（ViT）+ LLM 融合
- 任务：图像理解、视频理解、图文生成
- 代表：GPT-4o、Claude 3.5、Qwen-VL、LLaVA

| 模型 | 能力 | 架构 |
|------|------|------|
| GPT-4o | 图文音视频 | 闭源 |
| LLaVA | 图文理解 | ViT + LLM |
| Qwen-VL | 图文视频 | ViT + Qwen |
| Gemini | 原生多模态 | 统一架构 |

**⚠️ 常见坑：**
- 视觉 Token 消耗远高于文本，一张图可能占 1000+ tokens
- OCR 能力不等于文档理解能力
- 视频理解需要帧采样策略，不可能处理全部帧

---

<a id="ch20"></a>
## 20 - 长上下文处理

**核心考点：**

- 位置编码外推：RoPE 外推、YaRN、NTK-aware Scaling
- 高效注意力：稀疏注意力、线性注意力、滑动窗口
- 上下文窗口扩展：从 4K → 128K → 1M+

| 技术 | 原理 | 效果 |
|------|------|------|
| RoPE 外推 | 频率缩放 | 4x~8x |
| YaRN | 非均匀插值 | 16x+ |
| Ring Attention | 序列并行 | 理论无限 |
| 滑动窗口 | 局部注意力 | 降低复杂度 |

**"大海捞针"测试（Needle in a Haystack）**：验证模型在长上下文中定位关键信息的能力。

**⚠️ 常见坑：**
- 窗口大 ≠ 有效利用：模型对中间位置信息的注意力弱（Lost in the Middle）
- 长上下文推理成本随长度平方增长
- 实际使用中很少需要超过 32K 的上下文

---

<a id="ch21"></a>
## 21 - RAG 进阶

**核心考点：**

- Advanced RAG 技术：Query Transformation、Self-RAG、CRAG
- Graph RAG：结合知识图谱的结构化检索
- Multi-modal RAG：图文混合检索

| 技术 | 说明 | 适用场景 |
|------|------|----------|
| HyDE | 用假设回答做检索 | Query 短且模糊 |
| Self-RAG | 模型自我判断是否需要检索 | 开放域问答 |
| CRAG | 纠正式 RAG，评估检索质量 | 需要高准确性 |
| Graph RAG | 知识图谱 + 向量检索 | 多跳推理 |
| Agentic RAG | Agent 控制检索流程 | 复杂查询 |

**⚠️ 常见坑：**
- 多跳问题单次检索不够，需要迭代检索
- 检索文档之间的矛盾信息需要消歧处理
- Graph RAG 构建成本高，维护复杂

---

<a id="ch22"></a>
## 22 - 生产部署

**核心考点：**

- 推理框架：vLLM、TGI、TensorRT-LLM、SGLang
- 部署架构：负载均衡、模型并行、A/B 测试
- 监控指标：延迟（TTFT、TPS）、吞吐量、错误率、成本

| 框架 | 特点 | 适用 |
|------|------|------|
| vLLM | PagedAttention | 通用 |
| TGI | HuggingFace 生态 | HF 模型 |
| TensorRT-LLM | NVIDIA 优化 | 极致性能 |
| SGLang | 结构化生成优化 | Agent 场景 |

**关键指标：**
- TTFT (Time To First Token)：首 token 延迟
- TPS (Tokens Per Second)：生成速度
- QPS (Queries Per Second)：系统吞吐量

**⚠️ 常见坑：**
- 生产环境必须有 rate limiting 和请求队列
- 模型版本管理：灰度发布、回滚策略
- 成本优化：缓存、模型路由（简单问题用小模型）

---

<a id="ch23"></a>
## 23 - AI 编程与开发工具

**核心考点：**

- 工具类型：IDE 插件、CLI 工具、Agent 型工具
- 核心能力：代码补全、对话式编程、代码库理解、自动修复
- 代表产品：GitHub Copilot、Cursor、Windsurf、Cline

| 工具 | 类型 | 特点 |
|------|------|------|
| Copilot | IDE 补全 | 行级/块级补全 |
| Cursor | AI-native IDE | 全局上下文理解 |
| Cline | VSCode Agent | 开源、工具调用 |
| Aider | CLI | Git 集成 |

**⚠️ 常见坑：**
- 上下文管理是 AI 编程工具的核心差异点
- 需要人工审查生成代码，不能盲目信任
- 大型重构任务需要分步进行，不能一次给太大的指令

---

<a id="ch24"></a>
## 24 - MCP 协议（Model Context Protocol）

**核心考点：**

- Anthropic 提出的开放协议，标准化 LLM 与外部工具/数据的连接
- 架构：Host（应用）→ Client（协议层）→ Server（工具/数据提供方）
- 传输方式：stdio（本地进程）、SSE/HTTP（远程服务）
- 支持能力：Tools（工具调用）、Resources（资源读取）、Prompts（提示模板）

| 概念 | 说明 |
|------|------|
| MCP Host | 发起请求的应用（如 Claude Desktop） |
| MCP Client | 协议客户端，管理与 Server 的连接 |
| MCP Server | 暴露工具/资源的服务端 |
| Tool | 可调用的函数（有输入schema） |
| Resource | 可读取的数据源（类似 GET） |

**⚠️ 常见坑：**
- MCP Server 需要处理好权限隔离，不能暴露危险操作
- 与 OpenAI Function Calling 互补而非替代关系
- 远程 MCP 涉及认证和安全问题需要额外处理

---

<a id="ch25"></a>
## 25 - 扩散模型基础

**核心考点：**

- 核心原理：前向加噪（逐步加高斯噪声）→ 反向去噪（学习去噪）
- 训练目标：预测噪声 ε（ε-prediction）或预测 x₀
- 关键组件：U-Net / DiT 去噪网络 + 噪声调度器

| 模型 | 类型 | 特点 |
|------|------|------|
| Stable Diffusion | Latent Diffusion | 潜空间操作，高效 |
| DALL-E 3 | 扩散 + T5 | 文本理解强 |
| FLUX | DiT 架构 | Transformer 替代 U-Net |
| Sora | 视频 DiT | 时空 Transformer |

**Classifier-Free Guidance (CFG)：**
```
ε_guided = ε_uncond + w * (ε_cond - ε_uncond)
# w > 1 增强条件引导，通常 w=7~12
```

**⚠️ 常见坑：**
- 扩散模型采样步骤多（20-50步），推理速度慢
- Latent Diffusion 在潜空间操作，避免直接处理像素
- 文本生成图片的语义对齐仍是难题（手指、文字等）

---

<a id="ch26"></a>
## 26 - 前沿趋势

**核心考点：**

- Test-time Compute：推理时增加计算（如 o1 的思考链）提升质量
- 小模型崛起：Phi、Gemma 等高效小模型，边缘部署成为可能
- Agent 生态成熟：MCP 协议标准化、多 Agent 框架百花齐放
- 合成数据：用 LLM 生成训练数据，缓解数据瓶颈

| 趋势 | 影响 | 代表 |
|------|------|------|
| 推理增强 | 推理能力质变 | o1, DeepSeek-R1 |
| 开源追赶 | 降低门槛 | LLaMA 3, Qwen 2.5 |
| 多模态融合 | 统一理解 | GPT-4o, Gemini |
| AI 编程 | 开发效率 | Cursor, Devin |

**⚠️ 常见坑：**
- MoE（Mixture of Experts）不是万能药，训练稳定性是挑战
- Test-time Compute 增加延迟和成本，需要权衡
- 关注开源模型进展，很多场景不需要闭源大模型

---

## 📊 横向对比速查表

### 主流 LLM 对比

| 模型 | 参数量 | 上下文 | 特长 |
|------|--------|--------|------|
| GPT-4o | 未公开 | 128K | 综合最强 |
| Claude 3.5 Sonnet | 未公开 | 200K | 编程、长文本 |
| Gemini 1.5 Pro | 未公开 | 1M+ | 超长上下文 |
| LLaMA 3.1 405B | 405B | 128K | 开源最强 |
| Qwen 2.5 72B | 72B | 128K | 中文优秀 |
| DeepSeek-V3 | 671B MoE | 128K | 性价比极高 |

### 推理框架对比

| 框架 | 核心技术 | 吞吐量 | 易用性 |
|------|----------|--------|--------|
| vLLM | PagedAttention | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| TGI | Continuous Batching | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| TensorRT-LLM | TensorRT优化 | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| SGLang | RadixAttention | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 🔧 高频代码片段

### 1. OpenAI Function Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"}
            },
            "required": ["city"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京天气如何？"}],
    tools=tools,
    tool_choice="auto"
)

# 处理 function call
if response.choices[0].message.tool_calls:
    call = response.choices[0].message.tool_calls[0]
    args = json.loads(call.function.arguments)
    result = get_weather(**args)
```

### 2. 简易 RAG Pipeline（LangChain 风格）

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 构建
vectordb = Chroma.from_documents(docs, OpenAIEmbeddings())
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=retriever,
    return_source_documents=True
)

# 使用
result = qa.invoke({"query": "什么是 RAG？"})
```

### 3. DPO 训练配置

```python
from trl import DPOTrainer, DPOConfig

config = DPOConfig(
    output_dir="./dpo_output",
    per_device_train_batch_size=4,
    learning_rate=5e-7,
    beta=0.1,               # KL 惩罚系数
    max_length=1024,
    max_prompt_length=512,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    config=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

### 4. vLLM 部署

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,    # 2卡并行
    max_model_len=8192,
    gpu_memory_utilization=0.9,
)

sampling = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(["解释什么是 RAG"], sampling)
print(outputs[0].outputs[0].text)
```

---

## 💡 面试心态提示

1. **不需要全知** - 面试官期望你展示思考过程，而非背诵答案
2. **承认边界** - "这个我不太确定，但我的理解是..." 比胡说好
3. **由浅入深** - 先给概括性回答，等面试官追问再展开细节
4. **结合项目** - 任何知识点都尽量联系实际项目经验
5. **主动提问** - 面试结尾的提问环节展示你的思考深度

---

## 🎯 常见追问应对策略

### "为什么不用 X 而用 Y？"

> 模板：**需求分析 → X 的局限 → Y 的优势 → 实际取舍**

示例："为什么用 RAG 而不是微调？"
- 我们的数据更新频繁（每天），微调成本高
- 需要引用原文出处，RAG 天然支持
- 领域知识通过检索注入，不需要改变模型权重
- 如果任务是固定领域的风格迁移，我会选择微调

### "这个技术有什么缺点？"

> 模板：**先说 2-3 个真实缺点，再说缓解方案**

示例："RAG 的缺点？"
- 检索质量是瓶颈，需要高质量分块和检索策略
- 增加了系统复杂度和延迟
- 缓解：混合检索 + Reranker + Query 改写

### "在你的项目中怎么做的？"

> 模板：**背景 → 问题 → 方案 → 结果 → 反思**

示例：
- 背景：内部知识库问答系统，10万+文档
- 问题：简单检索准确率只有 60%
- 方案：语义分块 + 混合检索 + Cross-Encoder Reranker
- 结果：准确率提升到 85%
- 反思：还需要做 Query 改写处理口语化表达

### "如果数据量增长 100 倍怎么办？"

> 模板：**识别瓶颈 → 水平扩展方案 → 成本权衡**

- 检索层：从 Chroma 切换到 Milvus 分布式集群
- 分块层：预计算 Embedding，异步更新索引
- 生成层：模型服务多实例 + 负载均衡
- 缓存层：热门问题缓存，减少重复计算

### "如何评估效果？"

> 模板：**自动指标 + 人工评估 + A/B 测试**

- 自动：检索召回率、LLM-as-Judge 评分
- 人工：标注员评估准确性和有用性
- 线上：A/B 测试用户满意度、任务完成率

---

## 📝 最后检查清单

面试前快速过一遍：

- [ ] Transformer Self-Attention 公式
- [ ] LoRA 的原理和配置参数
- [ ] RAG 完整流程（分块 → 检索 → 重排 → 生成）
- [ ] ReAct 循环的 Thought-Action-Observation
- [ ] RLHF vs DPO 的区别
- [ ] FlashAttention 和 KV Cache 的原理
- [ ] 量化的 GPTQ vs AWQ 区别
- [ ] 分布式训练的 TP/PP/DP/ZeRO
- [ ] Prompt Injection 的类型和防御
- [ ] MCP 协议的基本架构
- [ ] 主流 LLM 对比（至少能说出 3 个）

---

> 💪 **祝面试顺利！记住：展示思考过程比展示知识量更重要。**
