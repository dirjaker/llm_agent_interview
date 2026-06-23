# 31. Context Engineering（上下文工程）

> **"The art of context engineering is the art of building dynamic systems that provide the right information and tools, in the right format, at every step of the task."** — Andrej Karpathy, 2025

---

## 🌳 知识图谱

```
Context Engineering（上下文工程）
├── 1. 基础概念
│   ├── 上下文的定义与组成
│   ├── 上下文窗口（Context Window）
│   └── 与 Prompt Engineering 的区别
├── 2. 上下文窗口管理
│   ├── 短上下文（4K-32K）
│   ├── 长上下文（128K-200K）
│   └── 超长上下文（1M+）
├── 3. 信息检索策略
│   ├── RAG（检索增强生成）
│   ├── Long Context 直接塞入
│   └── RAG + Long Context 混合策略
├── 4. 动态上下文组装
│   ├── System Prompt 设计
│   ├── 工具定义与选择
│   ├── 历史对话管理
│   └── 外部知识注入
├── 5. 上下文压缩技术
│   ├── 摘要压缩（Summary Compression）
│   ├── 选择性遗忘（Selective Forgetting）
│   ├── Token 预算分配
│   └── 重要性评分与裁剪
├── 6. Lost in the Middle 问题
│   ├── 问题本质与实验验证
│   ├── 信息位置对性能的影响
│   └── 解决策略
├── 7. 上下文缓存与复用
│   ├── KV Cache 原理
│   ├── Prefix Caching
│   ├── Prompt Caching（API 层）
│   └── 语义缓存
├── 8. 多轮对话上下文管理
│   ├── 滑动窗口策略
│   ├── 对话状态追踪
│   └── 长期记忆机制
└── 9. 工程实践与面试考点
    ├── 典型系统设计题
    ├── 性能优化题
    └── 场景分析题
```

---

## 一、Context Engineering 定义与重要性 ⭐🔥

### Q1：什么是 Context Engineering？它和 Prompt Engineering 有什么区别？

**A：**

**Prompt Engineering** 关注的是如何写好一条指令（prompt），让模型在单次调用中产出更好的结果。

**Context Engineering** 是一个更宏观的工程学科，关注的是：在 Agent 的**每一步推理**中，如何动态地组装、压缩、管理上下文信息，使模型获得最优的决策依据。

```
┌─────────────────────────────────────────────────────┐
│              Context Engineering 全景                │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ System    │  │ 工具定义  │  │ 外部知识         │  │
│  │ Prompt    │  │ Function │  │ (RAG/Web/API)   │  │
│  └────┬─────┘  └────┬─────┘  └───────┬──────────┘  │
│       │              │                │             │
│       ▼              ▼                ▼             │
│  ┌─────────────────────────────────────────────┐    │
│  │         动态上下文组装引擎                    │    │
│  │  (Context Assembly Engine)                   │    │
│  └──────────────────┬──────────────────────────┘    │
│                     │                               │
│                     ▼                               │
│  ┌─────────────────────────────────────────────┐    │
│  │  ┌─────┐ ┌──────┐ ┌──────┐ ┌────────────┐ │    │
│  │  │压缩 │ │缓存  │ │去重  │ │位置优化    │ │    │
│  │  └─────┘ └──────┘ └──────┘ └────────────┘ │    │
│  └──────────────────┬──────────────────────────┘    │
│                     │                               │
│                     ▼                               │
│  ┌─────────────────────────────────────────────┐    │
│  │              LLM 推理引擎                     │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

**核心区别：**

| 维度 | Prompt Engineering | Context Engineering |
|------|-------------------|-------------------|
| 范围 | 单条 prompt 优化 | 整个上下文生命周期管理 |
| 动态性 | 静态模板为主 | 动态组装、实时更新 |
| 关注点 | 措辞、格式、示例 | 信息选择、压缩、排序、缓存 |
| 适用场景 | 单轮任务 | 多步 Agent 系统 |
| 工程化程度 | 低（手动调优） | 高（自动化系统） |

#### 🔍 追问场景

**追问1：为什么说 Context Engineering 比 Prompt Engineering 更重要？**

> 随着模型能力提升，"怎么问"（prompt wording）的影响在下降，而"给模型看什么"（context composition）的影响在上升。一个 Agent 系统每一步都要决定：调用哪些工具、注入哪些历史、检索哪些文档——这些决策叠加起来的影响远超单条 prompt 的措辞优化。

**追问2：Context Engineering 包含哪些关键环节？**

> ① **信息采集**：从工具调用结果、RAG、用户输入等多源收集信息  
> ② **信息筛选**：根据相关性、重要性、时效性过滤  
> ③ **上下文组装**：按最优顺序和格式拼装  
> ④ **压缩与截断**：在 token 预算内保留最大信息量  
> ⑤ **缓存与复用**：避免重复计算，降低延迟和成本

---

## 二、上下文窗口管理 ⭐⭐🔥🔥

### Q2：不同规模的上下文窗口对系统设计有什么影响？

**A：**

上下文窗口大小直接决定了系统架构的选择。从 4K 到 1M，设计范式发生了根本变化。

```
上下文窗口演进路线图
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  4K        32K        128K       200K       1M+
  │          │           │          │         │
  ▼          ▼           ▼          ▼         ▼
GPT-3.5   GPT-4      Claude 3   Gemini    Gemini 1.5
2022       2023       2024       2024      2024
│          │           │          │         │
│ 基础对话  │ 复杂推理   │ 文档分析  │ 代码库   │ 代码库
│ 简单问答  │ 多轮交互   │ 多文档QA  │ 全书阅读  │ 视频理解
```

**不同窗口下的设计策略：**

```python
def design_context_strategy(window_size: int, task_type: str) -> dict:
    if window_size <= 4_096:  # 4K
        return {
            "strategy": "minimal_context",
            "system_prompt": "精简到 200 tokens 以内",
            "history": "仅保留最近 3 轮对话",
            "retrieval": "RAG 检索 top-1~2",
            "tools": "最多 3 个工具定义",
            "compression": "aggressive_summary",
        }
    elif window_size <= 131_072:  # 128K
        return {
            "strategy": "balanced_context",
            "system_prompt": "1-2K tokens，包含角色和规则",
            "history": "最近 10-20 轮 + 关键摘要",
            "retrieval": "RAG 检索 top-5-10",
            "tools": "10-20 个工具定义",
            "compression": "selective_pruning",
        }
    elif window_size <= 1_000_000:  # 1M+
        return {
            "strategy": "full_context",
            "system_prompt": "完整系统指令",
            "history": "完整对话历史或大量文档",
            "retrieval": "可直接塞入整个文档集",
            "tools": "完整工具集",
            "compression": "minimal_or_none",
        }
```

**128K vs 1M 的本质差异：**

| 维度 | 128K 窗口 | 1M+ 窗口 |
|------|----------|----------|
| 架构选择 | 必须 RAG + 压缩 | 可直接全文注入 |
| 成本 | ~$0.1-1/次 | ~$1-10/次（线性增长） |
| 延迟 | 秒级 | 十秒到分钟级 |
| Attention 质量 | 注意力集中 | 可能出现 Lost in the Middle |
| 最佳场景 | 精准问答、工具调用 | 全文分析、多文档对比 |

#### 🔍 追问场景

**追问1：窗口越大越好吗？是否存在"上下文窗口陷阱"？**

> **不是越大越好**。存在三个陷阱：  
> ① **注意力稀释**：窗口越大，模型对关键信息的注意力越分散  
> ② **成本线性增长**：1M context 的推理成本可能是 128K 的 8-10 倍（attention 复杂度 O(n²)）  
> ③ **幻觉增加**：信息过多时模型更容易混淆不同文档的内容  
> 实验表明，对于需要精确检索的任务，128K + RAG 往往优于 1M 直接塞入。

**追问2：如何决定一个任务该用 RAG 还是 Long Context？**

> 核心判断标准是**信息密度**和**检索精度要求**：  
> - 信息稀疏（答案只在某几段）→ RAG 更优  
> - 信息密集（需要综合全文理解）→ Long Context 更优  
> - 最佳方案通常是 RAG + Long Context 混合

---

## 三、RAG vs Long Context 对比 ⭐⭐🔥

### Q3：RAG 和 Long Context 各自的优劣势是什么？如何选择？

**A：**

```
┌─────────────────────────────────────────────────────────┐
│                RAG vs Long Context 对比                  │
├──────────────┬──────────────────┬────────────────────────┤
│    维度       │      RAG        │     Long Context       │
├──────────────┼──────────────────┼────────────────────────┤
│ 信息来源      │ 外部检索（向量DB）│ 直接放入上下文          │
│ 知识更新      │ 实时（更新DB即可）│ 需重新编码/推理          │
│ 成本          │ 低（小窗口）      │ 高（大窗口推理）         │
│ 精度依赖      │ 检索质量          │ 模型长文本理解能力       │
│ 多跳推理      │ 困难              │ 较好                    │
│ 上下文连贯性  │ 可能碎片化        │ 完整连贯                │
│ 可解释性      │ 高（可追溯来源）   │ 低（黑盒注意力）         │
│ 延迟          │ 检索+生成         │ 纯生成（可能更慢）       │
└──────────────┴──────────────────┴────────────────────────┘
```

**经典论文对比（Google, 2024）：**

```
实验设置：问答任务，不同文档规模
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                准确率
  100% ┤
       │  ●━━━━━━━━━━━━━━━━●  RAG (top-k)
   90% ┤                       随 k 增长先升后降
       │  ■━━━━━━━━━●━━━━━■  Long Context
   80% ┤              ↓        随文档增多先升后降
       │          sweet spot
   70% ┤
       ┼────┼────┼────┼────┼──
       0   50  100  200  500  文档数
```

**混合策略（推荐方案）：**

```python
def hybrid_retrieval(query, documents, context_window=128_000):
    # Step 1: 用向量检索快速筛选相关文档
    candidates = vector_search(query, documents, top_k=50)
    
    # Step 2: 用 Reranker 精排
    ranked = reranker.rank(query, candidates, top_k=20)
    
    # Step 3: 在 token 预算内塞入尽可能多的精排结果
    budget = context_window - SYSTEM_PROMPT_TOKENS - HISTORY_TOKENS
    selected = []
    used_tokens = 0
    for doc in ranked:
        doc_tokens = count_tokens(doc)
        if used_tokens + doc_tokens <= budget:
            selected.append(doc)
            used_tokens += doc_tokens
        else:
            # 尝试压缩后加入
            compressed = compress(doc, target_tokens=budget - used_tokens)
            if compressed:
                selected.append(compressed)
            break
    
    # Step 4: 按最优顺序排列（重要文档放首尾，避免 Lost in the Middle）
    return arrange_for_attention(selected)
```

#### 🔍 追问场景

**追问1：如果知识库有 10 万份文档，该用 RAG 还是 Long Context？**

> 必须用 RAG。10 万份文档即使每份 1000 tokens，总量也达 1 亿 tokens，远超任何模型窗口。正确做法是：向量检索 → Reranker 精排 → 裁剪到窗口内。

**追问2：有没有 RAG 解决不了、必须用 Long Context 的场景？**

> 有。典型场景：  
> ① **跨文档关联分析**：需要同时理解 10 份合同的交叉条款  
> ② **全局统计任务**：如"统计这 200 篇论文中提到的所有方法"  
> ③ **长文一致性检查**：检查一份 50 页报告的前后逻辑是否矛盾

---

## 四、动态上下文组装策略 ⭐⭐🔥🔥

### Q4：如何设计一个 Agent 的动态上下文组装系统？

**A：**

上下文组装是 Context Engineering 的核心环节。一个 Agent 的上下文通常由以下部分组成：

```
Agent 单步推理的上下文结构
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────┐
│ 1. System Prompt (系统指令)              │  ~1-2K tokens
│    - 角色定义、行为规则、输出格式          │
├─────────────────────────────────────────┤
│ 2. Tool Definitions (工具定义)           │  ~2-5K tokens
│    - 可用工具的 JSON Schema               │
├─────────────────────────────────────────┤
│ 3. Retrieved Knowledge (检索知识)        │  ~2-20K tokens
│    - RAG 检索结果、API 返回数据           │
├─────────────────────────────────────────┤
│ 4. Conversation History (对话历史)       │  ~2-50K tokens
│    - 用户消息、助手回复、工具调用记录       │
├─────────────────────────────────────────┤
│ 5. Current Query (当前查询)              │  ~0.1-1K tokens
│    - 用户最新输入或当前子任务              │
├─────────────────────────────────────────┤
│ 6. Scratchpad (工作记忆)                 │  ~1-5K tokens
│    - 中间推理结果、待验证假设              │
└─────────────────────────────────────────┘
```

**动态组装引擎实现：**

```python
class ContextAssembler:
    """动态上下文组装引擎"""
    
    def __init__(self, max_tokens: int = 128_000):
        self.max_tokens = max_tokens
        self.budget_allocator = TokenBudgetAllocator()
        
    def assemble(self, agent_state: AgentState) -> str:
        # Step 1: 分配 token 预算
        budget = self.budget_allocator.allocate(
            total=self.max_tokens,
            has_tools=agent_state.available_tools is not None,
            has_rag=agent_state.retrieved_docs is not None,
            history_turns=len(agent_state.conversation_history),
        )
        
        # Step 2: 组装各模块
        sections = []
        
        # System Prompt — 最高优先级，不可压缩
        system = self._build_system_prompt(
            agent_state.role, 
            agent_state.rules,
            budget.system
        )
        sections.append(("system", system))
        
        # 工具定义 — 动态选择相关工具
        if agent_state.available_tools:
            tools = self._select_tools(
                agent_state.available_tools,
                agent_state.current_query,
                max_tokens=budget.tools
            )
            sections.append(("tools", tools))
        
        # 检索知识 — RAG 结果，按相关性排序
        if agent_state.retrieved_docs:
            docs = self._arrange_docs(
                agent_state.retrieved_docs,
                max_tokens=budget.retrieval
            )
            sections.append(("retrieval", docs))
        
        # 对话历史 — 压缩 + 滑动窗口
        history = self._manage_history(
            agent_state.conversation_history,
            max_tokens=budget.history
        )
        sections.append(("history", history))
        
        # 当前查询
        sections.append(("query", agent_state.current_query))
        
        return self._render(sections)
    
    def _select_tools(self, tools, query, max_tokens):
        """动态选择最相关的工具（不总是塞入全部工具定义）"""
        if count_tokens(tools) <= max_tokens:
            return tools  # 工具定义都在预算内，全部保留
        
        # 按与 query 的相关性排序
        scored = [(similarity(query, t.description), t) for t in tools]
        scored.sort(reverse=True)
        
        selected, used = [], 0
        for score, tool in scored:
            t_tokens = count_tokens(tool.schema)
            if used + t_tokens <= max_tokens:
                selected.append(tool)
                used += t_tokens
        return selected


class TokenBudgetAllocator:
    """Token 预算分配器"""
    
    # 默认分配比例
    DEFAULT_RATIO = {
        "system": 0.05,     # 5%
        "tools": 0.10,      # 10%
        "retrieval": 0.40,  # 40%
        "history": 0.35,    # 35%
        "query": 0.05,      # 5%
        "reserve": 0.05,    # 5% 预留（生成空间）
    }
    
    def allocate(self, total, **kwargs) -> dict:
        ratio = dict(self.DEFAULT_RATIO)
        
        # 动态调整：如果工具多，增加工具预算
        if kwargs.get("has_tools") and not kwargs.get("has_rag"):
            ratio["tools"] = 0.20
            ratio["retrieval"] = 0.25
        
        # 如果历史对话轮次多，增加历史预算
        if kwargs.get("history_turns", 0) > 20:
            ratio["history"] = 0.45
            ratio["retrieval"] = 0.30
            
        return {k: int(total * v) for k, v in ratio.items()}
```

**最优排列顺序（基于注意力分布研究）：**

```
推荐的上下文排列策略：
━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────┐
│  🔴 最重要指令（System Prompt）│  ← 开头（注意力最高）
├─────────────────────────────┤
│  🟡 工具定义                 │
├─────────────────────────────┤
│  🟢 检索文档（按相关性降序）   │
├─────────────────────────────┤
│  🔵 对话历史（最近的在后）     │
├─────────────────────────────┤
│  🔴 当前查询 + 关键约束       │  ← 末尾（注意力第二高）
└─────────────────────────────┘

注意力分布曲线：
高 ┤ ██                              ██
   ┤ ████                          ████
   ┤ ██████                      ██████
低 ┤ ██████████━━━━━━━━━━━━━━━██████████
   ┼────────────────────────────────────
   开头                              末尾
```

#### 🔍 追问场景

追问1：当工具定义占用大量 token 时怎么办？**

> 三种策略：  
> ① **按需加载**：只注入与当前任务相关的工具定义（用分类器预筛选）  
> ② **工具摘要**：先给模型看工具摘要列表，模型选择后再注入详细 schema  
> ③ **分层工具**：将工具组织为树形结构，先选类别再选具体工具

**追问2：如何处理检索文档与对话历史冲突的情况？**

> 需要明确优先级策略：  
> - 检索文档通常作为**事实依据**（高优先级）  
> - 对话历史作为**交互上下文**（中优先级）  
> - 当两者冲突时，在 system prompt 中明确指示"以检索结果为准，如有矛盾请指出"

---

## 五、上下文压缩技术 ⭐⭐⭐🔥🔥

### Q5：有哪些上下文压缩技术？各自的适用场景是什么？

**A：**

上下文压缩是 Context Engineering 中最具技术深度的环节。核心挑战：**在有限 token 内保留最大信息量**。

```
上下文压缩技术分类
━━━━━━━━━━━━━━━━━━
├── 无损压缩（不丢失语义）
│   ├── 去重合并（重复信息合并）
│   ├── 格式优化（JSON→简写格式）
│   └── 代词替换（用代词替代重复实体）
├── 有损压缩（可控信息损失）
│   ├── 摘要压缩（Summary）
│   ├── 关键句提取（Key Sentence）
│   ├── 选择性遗忘（Drop old messages）
│   └── 重要性评分裁剪（Importance Scoring）
└── 学习型压缩（模型驱动）
    ├── Gist Tokens（学习的压缩 token）
    ├── TL;DR 预计算
    └── 自适应压缩率
```

**实现代码：**

```python
class ContextCompressor:
    """上下文压缩器"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def compress_conversation(
        self, 
        messages: list[Message], 
        target_tokens: int
    ) -> list[Message]:
        """压缩对话历史"""
        current_tokens = sum(count_tokens(m) for m in messages)
        
        if current_tokens <= target_tokens:
            return messages  # 不需要压缩
        
        # 策略 1: 保留最近 N 轮 + 摘要旧对话
        recent_keep = 6  # 保留最近 6 轮
        recent = messages[-recent_keep:]
        old = messages[:-recent_keep]
        
        # 用 LLM 生成旧对话摘要
        summary = self.llm.generate(
            f"请用 200 字概括以下对话的关键信息：\n{format_messages(old)}"
        )
        
        summary_msg = Message(
            role="system",
            content=f"[对话摘要] {summary}"
        )
        
        result = [summary_msg] + recent
        
        # 如果还超预算，进一步压缩
        if sum(count_tokens(m) for m in result) > target_tokens:
            result = self._aggressive_compress(result, target_tokens)
        
        return result
    
    def compress_documents(
        self, 
        docs: list[Document], 
        target_tokens: int
    ) -> list[Document]:
        """压缩检索文档"""
        total = sum(count_tokens(d) for d in docs)
        
        if total <= target_tokens:
            return docs
        
        compressed = []
        used = 0
        for doc in docs:
            doc_tokens = count_tokens(doc)
            remaining_budget = target_tokens - used
            
            if doc_tokens <= remaining_budget:
                compressed.append(doc)
                used += doc_tokens
            else:
                # 尝试提取关键段落
                key_passages = self._extract_key_passages(
                    doc, 
                    max_tokens=remaining_budget
                )
                if key_passages:
                    compressed.append(key_passages)
                break  # 预算用完
        
        return compressed
    
    def _extract_key_passages(self, doc, max_tokens):
        """提取文档关键段落"""
        sentences = split_sentences(doc.content)
        # 用 embedding 相似度或 TF-IDF 评分
        scored = [(score_sentence(s), s) for s in sentences]
        scored.sort(reverse=True)
        
        selected, used = [], 0
        for score, sent in scored:
            s_tokens = count_tokens(sent)
            if used + s_tokens <= max_tokens:
                selected.append(sent)
                used += s_tokens
            else:
                break
        return Document(content="\n".join(selected))


class ImportanceScorer:
    """重要性评分器"""
    
    def score_message(self, message: Message, context: list) -> float:
        """为消息打重要性分数"""
        score = 0.0
        
        # 1. 时间衰减：越新越重要
        recency = 1.0 / (1 + message.age_turns * 0.1)
        score += recency * 0.3
        
        # 2. 信息密度：包含实体、数字、专有名词越多越重要
        info_density = count_entities(message.content) / len(message.content)
        score += info_density * 0.3
        
        # 3. 引用频率：被后续消息引用越多越重要
        references = count_references(message, context)
        score += min(references / 5, 1.0) * 0.2
        
        # 4. 工具调用结果通常比纯文本重要
        if message.role == "tool":
            score += 0.2
        
        return score
```

**选择性遗忘策略对比：**

```
策略                 优点              缺点              适用场景
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
滑动窗口             简单高效          丢失早期信息        短对话
摘要压缩             保留全局信息      摘要可能失真        长对话
重要性裁剪           保留关键信息      评分可能不准        多主题对话
分层压缩             平衡细节与全局    实现复杂            Agent 多步推理
```

#### 🔍 追问场景

**追问1：摘要压缩会引入信息损失，如何评估和控制？**

> ① **可控压缩率**：设定压缩比上限（如不超过 4:1）  
> ② **关键信息保留检查**：压缩后用 NLI 模型验证关键事实是否保留  
> ③ **分段摘要**：每 N 轮生成一个摘要，而非一次性压缩全部  
> ④ **可回溯设计**：压缩后的摘要包含索引，需要时可回溯原文

**追问2：Gist Tokens 是什么？**

> Gist Tokens 是一种**学习型压缩**方法（Stanford, 2024）。通过微调模型，让它学会用几个特殊的 gist token 来表示一段长 prompt 的语义。推理时，只需输入 gist token 就能复现原始 prompt 的效果。优势是压缩率极高（100:1），但需要额外训练。

---

## 六、Lost in the Middle 问题与解决 ⭐⭐🔥🔥

### Q6：什么是 Lost in the Middle？如何解决？

**A：**

**Lost in the Middle**（中间丢失）是 2023 年由 Stanford/UC Berkeley 提出的重要发现：LLM 在处理长上下文时，对**开头和结尾**的信息利用效率远高于**中间**部分。

```
Lost in the Middle 现象
━━━━━━━━━━━━━━━━━━━━━━━

模型对不同位置信息的利用率：

利用率
100% ┤ ■■■■                          ■■■■
     ┤ ■■■■■■                      ■■■■■■
 80% ┤ ■■■■■■■■                  ■■■■■■■■
     ┤ ■■■■■■■■■■              ■■■■■■■■■■
 60% ┤ ■■■■■■■■■■■■          ■■■■■■■■■■■■
     ┤ ■■■■■■■■■■■■■■      ■■■■■■■■■■■■■■
 40% ┤ ■■■■■■■■■■■■■■■■  ■■■■■■■■■■■■■■■■
     ┤ ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
 20% ┤ ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
     ┼──────────────────────────────────────
     位置1   位置3   位置5   位置7   位置9
     (开头)                        (结尾)
     高 ← ──────── 低 ──────── → 高

结论：关键信息放在开头或结尾，性能最佳
```

**根因分析：**

```
Lost in the Middle 的技术原因
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 注意力分布不均匀
   - Self-Attention 在长序列中倾向于关注局部和边界
   - Softmax 归一化导致中间位置的注意力权重被稀释

2. 训练数据偏差
   - 预训练数据中，重要信息通常在文章开头/结尾
   - 模型学习到了"开头结尾更重要"的先验

3. 位置编码衰减
   - RoPE 等位置编码对远距离 token 的注意力有衰减效应
   - 中间位置的 token 可能处于"注意力盲区"
```

**解决策略：**

```python
class LostInTheMiddleMitigator:
    """Lost in the Middle 缓解策略"""
    
    def arrange_documents(
        self, 
        docs: list[Document], 
        query: str
    ) -> list[Document]:
        """策略1：将最相关文档放在首尾"""
        # 按相关性评分
        scored = [(compute_relevance(doc, query), doc) for doc in docs]
        scored.sort(reverse=True)
        
        # 最相关的放首尾，次相关的放中间
        result = [None] * len(scored)
        result[0] = scored[0][1]           # 最相关 → 开头
        result[-1] = scored[1][1]          # 第二相关 → 结尾
        for i, (_, doc) in enumerate(scored[2:], start=1):
            result[i] = doc                # 其余按序放中间
        
        return result
    
    def chunk_and_distribute(
        self, 
        long_doc: Document, 
        n_chunks: int = 5
    ) -> list[Document]:
        """策略2：分块 + 关键信息前置"""
        chunks = split_into_chunks(long_doc, n_chunks)
        # 按信息重要性重排
        ranked = rank_by_importance(chunks)
        return ranked  # 最重要的 chunk 放在前面
    
    def use_structured_context(
        self, 
        docs: list[Document]
    ) -> str:
        """策略3：用结构化标记强化中间信息"""
        output = []
        for i, doc in enumerate(docs):
            # 添加显式的位置标记和重要性标签
            marker = f"[文档 {i+1}/{len(docs)}] [重要性: {doc.importance}]"
            output.append(f"{marker}\n{doc.content}\n")
        
        # 在末尾添加回顾提醒
        output.append(
            "\n[提醒] 请特别注意以上所有文档的内容，"
            "包括中间部分的文档，它们都包含重要信息。"
        )
        return "\n".join(output)
    
    def repeated_insertion(
        self, 
        key_info: str, 
        context: str, 
        interval: int = 3
    ) -> str:
        """策略4：关键信息重复插入"""
        parts = context.split("\n\n")
        result = []
        for i, part in enumerate(parts):
            result.append(part)
            if i % interval == interval - 1:
                result.append(f"[关键信息提醒] {key_info}")
        return "\n\n".join(result)
```

#### 🔍 追问场景

**追问1：最新的模型（如 Claude 3.5、GPT-4o）还存在 Lost in the Middle 吗？**

> 有所改善但**并未完全解决**。2024 年的 Needle in a Haystack 测试显示，大多数模型在超长上下文（>100K）的中间位置仍有性能下降，只是幅度比早期模型小。最佳实践仍然是将关键信息放在上下文的首尾。

**追问2：有没有在架构层面解决这个问题的方法？**

> 有研究方向：  
> ① **Ring Attention**：环形注意力机制，使每个位置都能均匀关注其他位置  
> ② **Landmark Attention**：在关键位置插入 landmark token，引导注意力  
> ③ **分块注意力**：将长文分块处理后用 cross-attention 融合

---

## 七、上下文缓存与复用 ⭐⭐⭐🔥🔥

### Q7：KV Cache 和 Prefix Caching 是什么？如何优化推理成本？

**A：**

```
KV Cache 原理
━━━━━━━━━━━━

自回归生成中，每生成一个新 token 都需要计算与所有之前 token 的 attention。
如果每次都重新计算，复杂度为 O(n²)。

KV Cache 的核心思想：缓存已计算的 Key 和 Value 向量，新 token 只需计算自己的 Q，
与缓存的 K、V 做 attention。

┌──────────────────────────────────────────────────┐
│ 第 1 步：计算所有 token 的 K, V                   │
│                                                  │
│  token_1 → K₁, V₁                                │
│  token_2 → K₂, V₂                                │
│  ...                                             │
│  token_n → Kₙ, Vₙ                                │
│                                                  │
│  缓存: KV_cache = [(K₁,V₁), (K₂,V₂), ...]       │
├──────────────────────────────────────────────────┤
│ 第 2 步：生成 token_{n+1}                         │
│                                                  │
│  只需: Q_{n+1} = W_Q · token_{n+1}               │
│  计算: Attention = softmax(Q_{n+1} · K^T) · V    │
│  （使用缓存的 K, V，无需重新计算）                  │
│                                                  │
│  复杂度: O(n) 而非 O(n²)                          │
└──────────────────────────────────────────────────┘
```

**Prefix Caching（前缀缓存）：**

```
Prefix Caching 场景
━━━━━━━━━━━━━━━━━━

场景：多个用户请求共享相同的 System Prompt

请求1: [SYSTEM_PROMPT] + "请分析这份合同..."
请求2: [SYSTEM_PROMPT] + "帮我总结这篇文章..."
请求3: [SYSTEM_PROMPT] + "翻译这段文字..."
         ↑
    相同的前缀！

Prefix Caching:
- 首次请求：计算 SYSTEM_PROMPT 的 KV Cache 并缓存
- 后续请求：直接复用缓存，只需计算不同的后半部分

收益：
- 首 Token 延迟（TTFT）大幅降低
- GPU 计算量减少 50-80%（取决于前缀占比）
- 成本降低（部分 API 提供商对 cached prefix 给折扣）
```

**API 层的 Prompt Caching：**

```python
# OpenAI Prompt Caching 示例（2024 年推出）
# 缓存的 prompt 前缀按 50% 折扣计费

import openai

# 前 1024+ tokens 自动被缓存（相同的前缀）
response1 = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": LONG_SYSTEM_PROMPT},  # 这部分会被缓存
        {"role": "user", "content": "问题1"}
    ]
)

# 第二次请求：LONG_SYSTEM_PROMPT 部分命中缓存，按 50% 计费
response2 = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": LONG_SYSTEM_PROMPT},  # 缓存命中！
        {"role": "user", "content": "问题2"}
    ]
)

# Anthropic Claude 的 Prompt Caching（2024）
# 需要显式标记缓存断点
# messages=[
#     {"role": "user", "content": [
#         {"type": "text", "text": LONG_DOC, "cache_control": {"type": "ephemeral"}}
#     ]}
# ]
```

**KV Cache 的显存占用计算：**

```python
def estimate_kv_cache_size(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 2  # FP16 = 2 bytes
) -> int:
    """估算 KV Cache 显存占用（bytes）"""
    # 每个 token 每层需要存储 K 和 V 向量
    per_token_per_layer = 2 * num_kv_heads * head_dim * dtype_bytes
    total = num_layers * seq_len * per_token_per_layer
    return total

# 示例：LLaMA-70B, 128K 上下文
# 80 层, 8 KV heads (GQA), 128 dim/head
size = estimate_kv_cache_size(
    num_layers=80,
    num_kv_heads=8,
    head_dim=128,
    seq_len=128_000
)
# 结果: ~26 GB — 这就是为什么长上下文需要大量显存
```

#### 🔍 追问场景

**追问1：GQA（Grouped Query Attention）对 KV Cache 有什么影响？**

> GQA 将多个 Query Head 共享一组 KV Head，显著减少 KV Cache 大小。例如 LLaMA-70B 用 8 个 KV Head（而非 64 个 Q Head），KV Cache 缩小 8 倍。这是长上下文模型的标配优化。

**追问2：除了 Prefix Caching，还有哪些语义级别的缓存策略？**

> ① **语义缓存（Semantic Cache）**：用 embedding 相似度匹配历史请求，如果新问题与旧问题语义相近，直接复用旧答案  
> ② **部分缓存**：即使前缀不完全相同，缓存公共部分（如相同的 RAG 文档）  
> ③ **推测性缓存**：预测用户可能的下一个问题，预计算 KV Cache

---

## 八、多轮对话的上下文管理 ⭐⭐🔥

### Q8：如何设计多轮对话的上下文管理系统？

**A：**

```
多轮对话上下文管理架构
━━━━━━━━━━━━━━━━━━━━━━

┌──────────────────────────────────────────────────┐
│                 对话上下文管理器                    │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ 工作记忆  │  │ 短期记忆  │  │ 长期记忆      │  │
│  │(当前轮次) │  │(近期对话) │  │(跨会话知识)   │  │
│  │          │  │          │  │               │  │
│  │ 当前查询  │  │ 最近5-10 │  │ 用户画像      │  │
│  │ 工具结果  │  │ 轮对话    │  │ 历史偏好      │  │
│  │ 临时状态  │  │ 对话摘要  │  │ 重要事实      │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
│       │              │               │           │
│       ▼              ▼               ▼           │
│  ┌─────────────────────────────────────────┐     │
│  │        上下文组装器 (Assembler)          │     │
│  └─────────────────────────────────────────┘     │
└──────────────────────────────────────────────────┘
```

**实现方案：**

```python
class ConversationContextManager:
    """多轮对话上下文管理"""
    
    def __init__(self, max_history_tokens: int = 50_000):
        self.max_history_tokens = max_history_tokens
        self.working_memory = {}   # 当前轮次临时信息
        self.short_term = []       # 近期消息列表
        self.long_term = LongTermMemory()  # 持久化记忆
    
    def add_message(self, message: Message):
        """添加新消息"""
        self.short_term.append(message)
        
        # 检查是否需要压缩
        total_tokens = sum(count_tokens(m) for m in self.short_term)
        if total_tokens > self.max_history_tokens:
            self._compress_history()
        
        # 提取长期记忆
        important_facts = self._extract_facts(message)
        if important_facts:
            self.long_term.store(important_facts)
    
    def _compress_history(self):
        """压缩对话历史"""
        # 策略：保留最近 6 轮 + 压缩旧轮次为摘要
        recent_count = 6
        recent = self.short_term[-recent_count:]
        old = self.short_term[:-recent_count]
        
        if old:
            summary = self._summarize(old)
            self.short_term = [
                Message(role="system", content=f"[历史摘要] {summary}")
            ] + recent
    
    def get_context(
        self, 
        current_query: str,
        retrieved_docs: list = None
    ) -> list[Message]:
        """组装当前轮次的完整上下文"""
        messages = []
        
        # 1. System prompt
        messages.append(self._get_system_prompt())
        
        # 2. 长期记忆（如果相关）
        relevant_memory = self.long_term.retrieve(current_query, top_k=5)
        if relevant_memory:
            messages.append(Message(
                role="system",
                content=f"[用户相关信息]\n{format_memories(relevant_memory)}"
            ))
        
        # 3. 检索文档
        if retrieved_docs:
            messages.append(Message(
                role="system", 
                content=f"[参考资料]\n{format_docs(retrieved_docs)}"
            ))
        
        # 4. 对话历史
        messages.extend(self.short_term)
        
        # 5. 当前查询
        messages.append(Message(role="user", content=current_query))
        
        return messages
    
    def _extract_facts(self, message: Message) -> list[str]:
        """从消息中提取应记住的事实"""
        # 用 LLM 或规则提取关键信息
        # 例如：用户说"我是Python开发者"→ 记住用户的技术栈
        prompt = f"从以下消息中提取应该长期记住的关键事实：\n{message.content}"
        facts = self.llm.generate(prompt)
        return facts


class LongTermMemory:
    """长期记忆存储"""
    
    def __init__(self):
        self.memories = []
        self.embeddings = []
    
    def store(self, facts: list[str]):
        for fact in facts:
            embedding = encode(fact)
            self.memories.append(fact)
            self.embeddings.append(embedding)
    
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        if not self.memories:
            return []
        query_emb = encode(query)
        scores = [cosine_sim(query_emb, e) for e in self.embeddings]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.memories[i] for i in top_indices]
```

#### 🔍 追问场景

**追问1：如何处理用户说"刚才说的那个"这类指代消解问题？**

> ① **显式指代消解**：在用户消息前插入最近提及的实体上下文  
> ② **工作记忆**：维护一个"当前话题栈"，记录最近讨论的实体  
> ③ **反问确认**：当指代模糊时，让模型主动反问确认

**追问2：对话历史中的工具调用记录应该如何保留？**

> 推荐保留工具调用的**摘要**而非完整原始数据。例如：  
> - ❌ 完整的 API 返回 JSON（可能几千 tokens）  
> - ✅ "查询天气 API，结果：北京今天 25°C，晴"（几十 tokens）  
> 但需要保留完整的 tool_call_id，以满足 API 格式要求。

---

## 九、面试高频考点总结 ⭐⭐🔥🔥

### Q9：Context Engineering 面试中常见的问题类型有哪些？

**A：**

```
┌──────────────────────────────────────────────────────────┐
│           Context Engineering 面试考点地图                 │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │  概念理解题  │  │  系统设计题  │  │  场景分析题     │  │
│  │             │  │             │  │                 │  │
│  │ Q: 什么是    │  │ Q: 设计一个  │  │ Q: 用户投诉    │  │
│  │ Context     │  │ Agent 上下文 │  │ 模型"忘记了    │  │
│  │ Engineering?│  │ 管理系统     │  │ 之前的对话"，   │  │
│  │             │  │             │  │ 如何排查？      │  │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
│         │                │                   │           │
│         ▼                ▼                   ▼           │
│  ┌─────────────────────────────────────────────────┐     │
│  │              评价维度                             │     │
│  │  ① 理解深度：是否掌握底层原理                     │     │
│  │  ② 实践经验：是否有真实系统经验                   │     │
│  │  ③ 权衡能力：能否分析不同方案的 tradeoff          │     │
│  │  ④ 创新思维：能否提出新方案                       │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### 高频考题清单

**Level 1：概念理解（⭐）**

| # | 问题 | 核心要点 |
|---|------|---------|
| 1 | 什么是 Context Engineering？ | 动态上下文组装、压缩、管理的工程学科 |
| 2 | 上下文窗口越大越好吗？ | 否，注意力稀释 + 成本线性增长 |
| 3 | KV Cache 的作用是什么？ | 避免重复计算 K,V，降低推理延迟 |

**Level 2：技术分析（⭐⭐）**

| # | 问题 | 核心要点 |
|---|------|---------|
| 4 | RAG vs Long Context 如何选？ | 信息密度、成本、精度的权衡 |
| 5 | 如何缓解 Lost in the Middle？ | 关键信息放首尾、结构化标记、重复插入 |
| 6 | 对话历史过长怎么处理？ | 摘要压缩 + 滑动窗口 + 长期记忆 |

**Level 3：系统设计（⭐⭐⭐）**

| # | 问题 | 核心要点 |
|---|------|---------|
| 7 | 设计一个支持 100K 文档的 Agent 上下文系统 | 分层检索 + 动态组装 + 渐进压缩 |
| 8 | 如何设计 Token 预算分配策略？ | 按任务类型动态分配，保留预留空间 |
| 9 | Prefix Caching 如何实现？需要考虑什么？ | 共享前缀 + 缓存失效策略 + 显存管理 |

---

### Q10：设计一个完整的 Agent 上下文管理系统（系统设计题）⭐⭐⭐🔥🔥

**A：**

```
完整系统设计：Agent Context Management System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用户请求
    │
    ▼
┌──────────────────────────────────────────────┐
│              请求路由器                        │
│  判断任务类型 → 选择上下文策略                  │
└──────────────┬───────────────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌──────┐  ┌──────┐  ┌──────────┐
│简单  │  │RAG类 │  │长文档    │
│对话  │  │问答  │  │分析      │
└──┬───┘  └──┬───┘  └──┬───────┘
   │         │         │
   ▼         ▼         ▼
┌──────────────────────────────────────────────┐
│              信息收集层                        │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌───────┐ │
│  │RAG检索 │ │API调用 │ │工具执行│ │历史查询│ │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬───┘ │
└──────┼──────────┼──────────┼──────────┼──────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
┌──────────────────────────────────────────────┐
│              上下文组装层                      │
│  ┌─────────────────────────────────────────┐ │
│  │ Token 预算分配器                         │ │
│  │  System: 5% | Tools: 10% | RAG: 40%    │ │
│  │  History: 35% | Query: 5% | Reserve: 5%│ │
│  └─────────────────┬───────────────────────┘ │
│                    │                          │
│  ┌─────────────────┼───────────────────────┐ │
│  │ 信息压缩器       │                       │ │
│  │  - 摘要压缩      │  - 关键句提取          │ │
│  │  - 去重合并      │  - 格式优化            │ │
│  └─────────────────┼───────────────────────┘ │
│                    │                          │
│  ┌─────────────────┼───────────────────────┐ │
│  │ 排列优化器       │                       │ │
│  │  - 首尾重要      │  - Lost in the Middle │ │
│  │  - 结构化标记    │  - 分段放置关键信息     │ │
│  └─────────────────┼───────────────────────┘ │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│              缓存管理层                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ KV Cache │ │ Prefix   │ │ 语义缓存     │ │
│  │ 管理     │ │ Caching  │ │ (相似问答)    │ │
│  └──────────┘ └──────────┘ └──────────────┘ │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│              LLM 推理引擎                      │
│              生成响应                          │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│              后处理层                          │
│  - 更新对话历史                               │
│  - 提取长期记忆                               │
│  - 更新缓存                                   │
│  - 收集质量指标                               │
└──────────────────────────────────────────────┘
```

**关键设计决策与 Tradeoff：**

```python
# 决策1: Token 预算分配策略
# 问：RAG 检索结果 vs 对话历史，谁该占更多预算？
# 答：取决于任务类型
BUDGET_PROFILES = {
    "qa":        {"retrieval": 0.50, "history": 0.20},  # 问答：检索为主
    "chat":      {"retrieval": 0.10, "history": 0.55},  # 闲聊：历史为主
    "analysis":  {"retrieval": 0.60, "history": 0.10},  # 分析：大量文档
    "coding":    {"retrieval": 0.30, "history": 0.30, "tools": 0.15},  # 编码
}

# 决策2: 压缩时机
# 问：应该在什么时候压缩？
# 选项A: 每轮结束时压缩（实时）
# 选项B: 超过阈值时压缩（懒加载）
# 选项C: 组装时压缩（按需）
# 最佳实践: B + C 组合 —— 懒加载 + 组装时按需压缩

# 决策3: 检索粒度
# 问：检索应该返回文档、段落还是句子？
# 答：分层检索
#   Stage 1: 文档级召回（top-100） → 粗筛
#   Stage 2: 段落级精排（top-20）  → 细筛
#   Stage 3: 句子级提取（top-K）   → 塞入上下文
```

---

### Q11：常见面试追问与参考回答 ⭐⭐🔥

**追问1：用户反馈"模型经常忘记之前说过的话"，如何排查？**

> 排查步骤：  
> ① **检查对话历史是否被过度压缩**：查看压缩后的摘要是否丢失关键信息  
> ② **检查 token 预算分配**：历史对话是否被其他模块（如 RAG）挤占了空间  
> ③ **检查 Lost in the Middle**：关键信息是否被放在了上下文中间  
> ④ **检查多会话隔离**：不同会话的记忆是否正确隔离  
> ⑤ **量化指标**：对比压缩前后的上下文召回率

**追问2：如何评估上下文管理系统的质量？**

> 评估指标体系：  
> - **上下文利用率**：塞入的 token 中有多少被模型有效利用  
> - **压缩保真度**：压缩后的信息与原始信息的一致性  
> - **延迟影响**：上下文组装的额外延迟  
> - **成本效率**：每美元能处理的有效 token 数  
> - **端到端质量**：最终任务准确率是否满足要求

**追问3：Context Caching 和 KV Cache 的区别是什么？**

> - **KV Cache**：模型内部的推理优化，缓存已计算的 Key/Value 张量，避免重复计算。对用户透明。  
> - **Context Caching / Prompt Caching**：API 层的优化，将相同的 prompt 前缀缓存起来，跨请求复用。用户可以通过合理组织 prompt 来利用。  
> - **关系**：Prompt Caching 的底层实现依赖于 KV Cache。API 提供商将共享前缀的 KV Cache 保存下来，供后续请求使用。

**追问4：如果要支持 1000 个并发用户，上下文系统需要考虑什么？**

> ① **KV Cache 显存管理**：1000 个用户的 KV Cache 可能需要数百 GB 显存，需要 PagedAttention (vLLM) 等技术  
> ② **Prefix Caching 共享**：如果用户共享相同的 system prompt，可以大幅减少显存占用  
> ③ **异步检索**：RAG 检索应该是异步的，在 LLM 推理前并行完成  
> ④ **上下文预计算**：对于确定性部分（如工具定义），可以预计算并缓存  
> ⑤ **队列管理**：上下文组装应该是无状态的，便于水平扩展

---

## 附录：核心参考文献

| 论文 | 年份 | 关键贡献 |
|------|------|---------|
| Lost in the Middle (Liu et al.) | 2023 | 发现长上下文中信息位置对性能的影响 |
| RAG (Lewis et al.) | 2020 | 检索增强生成基础架构 |
| Gist Tokens (Mu et al.) | 2024 | 学习型上下文压缩方法 |
| Ring Attention (Liu et al.) | 2023 | 分布式长上下文注意力 |
| Prompt Caching (Anthropic) | 2024 | API 层前缀缓存优化 |
| Infini-Attention (Munkhdalai et al.) | 2024 | 无限上下文注意力机制 |
| GQA (Ainslie et al.) | 2023 | 分组查询注意力，减少 KV Cache |
| vLLM / PagedAttention (Kwon et al.) | 2023 | 高效 KV Cache 内存管理 |

---

> **面试锦囊**：Context Engineering 是 2025 年 Agent 领域最热门的工程方向之一。面试官关注的不是你是否知道每个技术细节，而是你能否**系统性地思考**：给定一个场景（如 100K 文档问答、1000 并发 Agent），如何设计上下文管理策略，如何在**成本、延迟、质量**之间做权衡。掌握本文的框架性思维，比记住具体技术更重要。
