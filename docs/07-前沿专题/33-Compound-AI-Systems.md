# 第33章 复合AI系统 (Compound AI Systems)

> **核心观点**：现代AI产品的质量提升，越来越多地依赖于**多个模型、工具、策略的组合编排**，而非单一模型的规模扩展。复合AI系统是LLM Agent落地的核心架构范式。

---

## 知识图谱

```
复合AI系统 (Compound AI Systems)
├── 1. 定义与动机
│   ├── 单模型 vs 复合系统
│   ├── 为什么需要复合系统
│   └── Berkeley AI Research (BAIR) 的Compound AI定义
├── 2. 核心架构模式
│   ├── 模型路由 (Model Router)
│   │   ├── 基于规则的路由
│   │   ├── 基于分类器的路由
│   │   └── 动态路由 (学习型)
│   ├── 模型级联 (Cascade)
│   │   ├── 置信度阈值触发
│   │   ├── 成本-质量权衡
│   │   └── FrugalGPT 级联策略
│   ├── 模型集成 (Ensemble)
│   │   ├── 投票机制 (Majority Voting)
│   │   ├── 加权聚合
│   │   └── LLM-as-Judge 裁决
│   └── 工具编排 (Tool Orchestration)
│       ├── Function Calling
│       ├── MCP (Model Context Protocol)
│       └── 多Agent协作
├── 3. 评估与监控
│   ├── 端到端指标
│   ├── 组件级评估
│   ├── 路由准确率
│   └── 线上监控与告警
├── 4. 成本优化
│   ├── 路由策略降本
│   ├── 级联策略降本
│   ├── 缓存与去重
│   └── Token 预算管理
└── 5. 生产实践
    ├── RAG + Agent + 多模型系统
    ├── 路由器生产化
    └── 故障恢复与降级
```

---

## 一、复合AI系统定义 ⭐⭐ 🔥🔥🔥

### Q1: 什么是复合AI系统？它和单模型系统有什么区别？

**A：**

复合AI系统是指由**多个组件（模型、工具、检索器、规则引擎等）协同工作**来完成任务的系统，其整体能力超越任何单一组件。

| 维度 | 单模型系统 | 复合AI系统 |
|------|-----------|-----------|
| 架构 | 一个大模型端到端 | 多组件编排 |
| 质量提升路径 | 更大模型、更多数据 | 更好的组合与路由 |
| 成本控制 | 随模型线性增长 | 可精细优化 |
| 可解释性 | 黑盒 | 组件间可追踪 |
| 典型代表 | GPT-4 直接回答 | RAG系统、多模型路由 |

> **关键洞察（BAIR, 2024）**：2023年AI领域的突破（如AlphaCode 2）并非来自更大的模型，而是来自将模型与搜索、验证等组件组合。ChatGPT本身也是一个复合系统（LLM + RLHF + 安全过滤器）。

### 追问场景

**Q1.1: 复合系统是不是就是Pipeline？**

不完全是。Pipeline是复合系统的一种形式，但复合系统还包含：
- **循环结构**（如Agent的ReAct循环）
- **条件分支**（如路由、级联）
- **并行执行**（如Ensemble投票）
- **动态编排**（运行时决定下一步组件）

---

## 二、模型路由 (Model Router) ⭐⭐ 🔥🔥🔥

### Q2: 什么是模型路由？如何实现？

**A：**

模型路由是根据**输入的特征**，将其分配到最合适的模型处理，以平衡质量、成本和延迟。

```
┌─────────────────────────────────────────────┐
│                  Router                      │
│                                              │
│   Input ──→ [特征提取] ──→ [路由决策]        │
│                               │              │
│                    ┌──────────┼──────────┐   │
│                    ▼          ▼          ▼   │
│                GPT-4o     Claude-3.5   Llama │
│               (复杂推理)  (长文档)    (简单QA) │
│                    │          │          │   │
│                    ▼          ▼          ▼   │
│                  Response  Response  Response │
└─────────────────────────────────────────────┘
```

**三种路由策略：**

**1）基于规则的路由（最简单）**

```python
def rule_based_router(query: str, metadata: dict) -> str:
    # 基于规则硬编码
    if metadata.get("language") == "zh":
        return "qwen-max"
    if len(query) > 4000:
        return "claude-3-opus"  # 长上下文
    if metadata.get("task_type") == "code":
        return "deepseek-coder"
    return "gpt-4o-mini"  # 默认轻量模型
```

**2）基于分类器的路由**

```python
class ClassifierRouter:
    def __init__(self):
        # 用小模型做路由分类器
        self.classifier = train_router_classifier(
            # 特征：query长度、复杂度分数、领域标签、用户意图
            features=["query_len", "complexity_score", "domain", "intent"],
            # 标签：最优模型（基于历史数据标注）
            labels=["gpt-4o", "claude-3.5", "llama-3-70b", "qwen-turbo"]
        )
    
    def route(self, query: str) -> str:
        features = self.extract_features(query)
        model = self.classifier.predict(features)
        confidence = self.classifier.predict_proba(features).max()
        
        # 低置信度时选择高质量模型兜底
        if confidence < 0.6:
            return "gpt-4o"
        return model
```

**3）语义路由（Embedding Router）**

```python
class SemanticRouter:
    def __init__(self):
        # 每个模型对应一组示例query的embedding
        self.model_clusters = {
            "gpt-4o": embed(examples_complex_reasoning),
            "llama-3-8b": embed(examples_simple_qa),
            "deepseek-coder": embed(examples_code),
        }
    
    def route(self, query: str) -> str:
        q_emb = embed(query)
        best_model = max(
            self.model_clusters,
            key=lambda m: cosine_similarity(q_emb, self.model_clusters[m])
        )
        return best_model
```

### 追问场景

**Q2.1: 路由器本身的延迟怎么处理？**

路由器通常使用**极小的模型**（如BERT-base、100ms级别）或规则引擎（<10ms），其延迟远小于LLM推理延迟（1-10s）。关键优化：
- 用distilled的小分类器
- 特征提取可缓存（如用户级别特征）
- 并行：路由器决策时可以先启动默认模型的prefill

**Q2.2: 路由决策错误怎么办？** ⭐⭐⭐

需要设计**路由容错机制**：
- **AB评估**：线上持续对比路由选择vs直接用最强模型
- **置信度阈值**：低置信度时默认选最强模型
- **回退机制**：下游模型返回低置信度时，自动重试更强模型
- **人工标注反馈环**：收集bad case持续优化路由

---

## 三、模型级联 (Model Cascade) ⭐⭐ 🔥🔥

### Q3: 模型级联是什么？和路由有什么区别？

**A：**

级联是路由的一种特殊形式：**先用便宜的小模型处理，如果小模型置信度不足，再"升级"到更贵的大模型**。区别在于路由是一次性决策，级联是**逐级fallback**。

```
用户Query
    │
    ▼
┌──────────┐   置信度 ≥ 0.9
│ 小模型    │ ──────────→ 返回结果 ✅ (省钱!)
│ (Llama-3-8B) │
└──────────┘
    │ 置信度 < 0.9
    ▼
┌──────────┐   置信度 ≥ 0.8
│ 中模型    │ ──────────→ 返回结果 ✅
│ (GPT-4o-mini) │
└──────────┘
    │ 置信度 < 0.8
    ▼
┌──────────┐
│ 大模型    │ ──────────→ 返回结果 ✅ (保证质量)
│ (GPT-4o)  │
└──────────┘
```

**FrugalGPT（Stanford, 2023）核心思想：**

```python
class FrugalGPTCascade:
    def __init__(self):
        self.model_chain = [
            ("gpt-3.5-turbo", 0.85, 0.002),   # (模型, 置信度阈值, 每token成本)
            ("gpt-4o-mini",   0.80, 0.00015),
            ("gpt-4o",        0.00, 0.005),     # 最后一级，无阈值
        ]
    
    def query(self, prompt: str) -> str:
        for model, threshold, _ in self.model_chain:
            response, confidence = call_llm(model, prompt)
            if confidence >= threshold:
                return response
        return response  # 最后一级一定返回
```

**级联的置信度信号来源：**
- **Logprob自评**：模型输出的token概率均值
- **自我一致性**：多次采样看是否一致
- **外部验证器**：训练一个专门的小模型判断回答质量
- **格式/规则检查**：代码能否执行、SQL能否运行

### 追问场景

**Q3.1: 级联的置信度校准怎么做？**

直接用模型的logprob做置信度往往**不校准**（overconfident），需要：
- 在验证集上做**温度缩放 (Temperature Scaling)**
- 训练**外部校准模型**：输入(query, response)→输出质量分数
- 使用**一致性校准**：采样5次，一致率作为置信度代理

**Q3.2: 级联增加的延迟怎么优化？** ⭐⭐⭐

- **投机执行**：小模型推理时，大模型已在后台prefill，若小模型不足则无缝切换
- **异步预热**：根据query特征预判可能需要大模型，提前启动
- **并行pipeline**：小模型和大模型同时推理，小模型置信度够则取消大模型

---

## 四、模型集成 (Ensemble) ⭐⭐ 🔥

### Q4: 多模型集成在LLM场景下怎么做？

**A：**

集成方法让多个模型同时回答，通过投票或聚合得出最终结果，追求质量上限。

**1）多数投票 (Majority Voting)**

```python
def majority_vote(query: str, models: list[str]) -> str:
    responses = [call_llm(m, query) for m in models]
    
    # 语义聚类投票（非精确匹配）
    clusters = semantic_cluster(responses)
    best_cluster = max(clusters, key=len)
    
    # 从最大簇中选最高质量的回答
    return select_best(best_cluster)
```

**2）LLM-as-Judge 裁决**

```python
def judge_ensemble(query: str, models: list[str]) -> str:
    # 第一步：各模型独立回答
    responses = {m: call_llm(m, query) for m in models}
    
    # 第二步：用裁判模型评估
    judge_prompt = f"""
    问题：{query}
    回答A：{responses['gpt-4o']}
    回答B：{responses['claude-3.5']}
    回答C：{responses['qwen-max']}
    
    请选出最佳回答（输出A/B/C），并说明理由。
    """
    judgment = call_llm("gpt-4o", judge_prompt)
    return parse_winner(judgment, responses)
```

**3）加权聚合（适用于数值类任务）**

```python
def weighted_ensemble(query: str, models: dict[str, float]) -> float:
    # models = {"gpt-4o": 0.5, "claude-3.5": 0.3, "qwen-max": 0.2}
    total = 0.0
    for model, weight in models.items():
        response = call_llm(model, query)
        total += parse_number(response) * weight
    return total
```

### 追问场景

**Q4.1: 集成的成本是单模型的N倍，值得吗？**

不一定。实践中通常只在**关键场景**使用集成：
- 高价值决策（医疗、法律、金融）
- 评估阶段（离线benchmark）
- 可用**异构集成**：1个大模型 + 2个小模型，成本仅增加50%

---

## 五、工具编排 (Tool Orchestration) ⭐⭐ 🔥🔥

### Q5: 复合系统中的工具编排怎么做？MCP是什么？

**A：**

工具编排是复合AI系统的核心能力，让LLM能够调用外部工具扩展能力边界。

**Function Calling 流程：**

```
User Query
    │
    ▼
┌──────────────┐
│   LLM Router │ ── 判断是否需要工具
│   + Planner  │
└──────┬───────┘
       │ function_call: {"name": "search", "args": {"query": "..."}}
       ▼
┌──────────────┐
│  Tool        │ ── 执行工具（搜索、数据库、API等）
│  Executor    │
└──────┬───────┘
       │ tool_result
       ▼
┌──────────────┐
│   LLM        │ ── 整合工具结果，生成最终回答
│   Synthesizer│
└──────────────┘
```

**MCP (Model Context Protocol)：**

MCP是Anthropic提出的开放协议，标准化了LLM与工具/数据源的连接方式。

```python
# MCP Server 定义（工具提供方）
from mcp import Server

server = Server("database-tools")

@server.tool()
def query_database(sql: str) -> str:
    """执行只读SQL查询"""
    return db.execute_readonly(sql)

@server.tool()
def list_tables() -> list[str]:
    """列出所有可用表"""
    return db.list_tables()

# MCP Client 使用（LLM应用侧）
from mcp import Client

client = Client("database-tools")
tools = client.list_tools()  # 自动获取工具schema
result = client.call_tool("query_database", sql="SELECT * FROM users LIMIT 5")
```

**MCP的核心价值：**
- **标准化**：一个协议适配所有工具（类似USB标准）
- **解耦**：工具开发者和LLM应用开发者独立工作
- **可组合**：多个MCP Server可动态组合

### 追问场景

**Q5.1: Function Calling和MCP有什么区别？**

| 维度 | Function Calling | MCP |
|------|-----------------|-----|
| 层级 | 模型能力（API参数） | 协议/标准 |
| 范围 | 单次调用 | 完整的工具生命周期管理 |
| 动态性 | 需预先定义工具列表 | 运行时发现工具 |
| 生态 | 各厂商私有 | 开放标准 |

Function Calling是MCP的底层能力之一，MCP在其之上增加了工具发现、认证、上下文管理等。

---

## 六、评估与监控 ⭐⭐ 🔥🔥

### Q6: 如何评估复合AI系统的质量？

**A：**

复合系统的评估比单模型复杂得多，需要**分层评估**：

```
┌────────────────────────────────────────────┐
│              端到端评估                      │
│   (最终用户体验: 准确率/满意度/任务完成率)    │
├────────────────────────────────────────────┤
│         组件级评估                           │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐   │
│  │Router│  │Retriev│  │  LLM  │  │ Tools │  │
│  │准确率 │  │Recall │  │质量分 │  │成功率  │  │
│  └──────┘  └──────┘  └──────┘  └──────┘   │
├────────────────────────────────────────────┤
│         流程级评估                           │
│  路由准确率 │ 级联触发率 │ 工具调用正确率     │
└────────────────────────────────────────────┘
```

**关键评估指标：**

```python
class CompoundSystemEvaluator:
    def evaluate(self, test_set):
        metrics = {
            # 端到端
            "task_accuracy": self.task_accuracy(test_set),
            "avg_latency": self.avg_latency(test_set),
            "avg_cost": self.avg_cost_per_query(test_set),
            
            # 路由组件
            "router_accuracy": self.router_accuracy(test_set),  # 路由是否选对模型
            "router_overhead": self.router_latency(test_set),   # 路由延迟开销
            
            # 级联组件
            "cascade_rate": self.cascade_trigger_rate(test_set), # 多少query需要升级
            "cost_savings": self.cost_vs_baseline(test_set),     # 相比全用大模型省了多少
            
            # 质量-成本帕累托
            "quality_cost_ratio": self.task_accuracy() / self.avg_cost(),
        }
        return metrics
```

### 追问场景

**Q6.1: 复合系统的bug定位怎么做？**

- **Trace ID贯穿**：每个请求生成唯一ID，贯穿Router→Model→Tool所有组件
- **组件级日志**：记录每一步的输入输出和决策依据
- **Ablation测试**：逐步替换组件（如将路由换成固定模型），定位质量下降点
- **回放测试**：用历史trace做回归测试

---

## 七、成本优化 ⭐⭐ 🔥🔥🔥

### Q7: 如何用复合系统架构降低API成本？

**A：**

核心思路：**让简单问题用便宜模型，只有复杂问题才用昂贵模型**。

**成本优化策略全景：**

```
成本优化
├── 1. 路由降本
│   ├── 规则路由：简单query → 小模型
│   └── 学习路由：分类器自动分流
├── 2. 级联降本
│   ├── 小模型处理70%的简单query
│   └── 仅30%复杂query升级到大模型
├── 3. 缓存降本
│   ├── 语义缓存：相似query命中缓存
│   └── KV Cache复用：相同前缀共享prefill
├── 4. 压缩降本
│   ├── Prompt压缩：减少输入token
│   └── 上下文裁剪：只传必要信息
└── 5. 批处理降本
    └── 异步批处理API（OpenAI Batch API 50%折扣）
```

**实际成本计算示例：**

```python
# 场景：日均10万请求，原方案全部用GPT-4o
# 优化方案：级联 = 70% GPT-4o-mini + 25% Claude-3.5 + 5% GPT-4o

# 原方案成本
original = 100_000 * 0.005  # $500/天

# 级联方案成本
cascade = (
    70_000 * 0.00015 +   # GPT-4o-mini: $10.5
    25_000 * 0.003 +     # Claude-3.5:  $75
    5_000 * 0.005        # GPT-4o:      $25
)  # = $110.5/天

# 节省: 78%!
print(f"节省: {(1 - cascade/original)*100:.0f}%")
```

### 追问场景

**Q7.1: 语义缓存怎么做？**

```python
class SemanticCache:
    def __init__(self, threshold=0.95):
        self.cache = {}  # embedding -> response
        self.threshold = threshold
    
    def get(self, query: str):
        q_emb = embed(query)
        for cached_emb, response in self.cache.items():
            if cosine_sim(q_emb, cached_emb) > self.threshold:
                return response  # 缓存命中
        return None
    
    def put(self, query: str, response: str):
        self.cache[embed(query)] = response
```

注意语义缓存的风险：**相似query不等于相同答案**（如"今天天气"每天不同），需要结合query特征做缓存失效策略。

---

## 八、实际案例：生产级复合AI系统 ⭐⭐⭐ 🔥🔥🔥

### Q8: 请描述一个完整的生产级复合AI系统架构

**A：**

以**智能客服系统**为例：

```
用户消息
    │
    ▼
┌─────────────────────────────────────────────────┐
│              意图识别 & 路由层                     │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │简单FAQ匹配│  │意图分类器  │  │复杂度评估器   │  │
│  │(缓存/规则)│  │(BERT-small)│  │(小模型)      │  │
│  └─────┬────┘  └─────┬─────┘  └──────┬───────┘  │
│        │             │               │           │
│    直接返回      路由到对应模型      标记优先级    │
└────────┼─────────────┼───────────────┼───────────┘
         │             │               │
         ▼             ▼               ▼
    ┌─────────────────────────────────────────┐
    │           模型调度层 (Dispatcher)         │
    │                                          │
    │   简单 ──→ Qwen-Turbo (快+便宜)          │
    │   中等 ──→ GPT-4o-mini (平衡)            │
    │   复杂 ──→ GPT-4o + RAG (高质量)         │
    │   关键 ──→ 多模型Ensemble + 人工审核       │
    └────────────────┬────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────┐
    │           工具执行层                      │
    │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐   │
    │  │订单查询│ │退款处理│ │知识库 │ │转人工│   │
    │  └──────┘ └──────┘ └──────┘ └──────┘   │
    └────────────────┬────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────┐
    │           质量保障层                      │
    │  • 安全过滤 (敏感信息/合规)               │
    │  • 回答质量评分                           │
    │  • 低分自动升级到更强模型                  │
    └────────────────┬────────────────────────┘
                     │
                     ▼
                  用户回复
```

**核心代码框架：**

```python
class ProductionCompoundSystem:
    def __init__(self):
        self.cache = SemanticCache(threshold=0.95)
        self.router = ClassifierRouter()
        self.models = {
            "fast": QwenTurbo(),
            "balanced": GPT4oMini(),
            "premium": GPT4o(),
        }
        self.tools = ToolRegistry()
        self.safety = SafetyFilter()
        self.monitor = SystemMonitor()
    
    async def handle(self, user_msg: str, context: dict) -> str:
        trace_id = generate_trace_id()
        
        # 1. 缓存检查
        cached = self.cache.get(user_msg)
        if cached:
            self.monitor.record(trace_id, "cache_hit")
            return cached
        
        # 2. 路由决策
        route = self.router.route(user_msg, context)
        self.monitor.record(trace_id, "route", route)
        
        # 3. 模型调用（带级联）
        response = await self._call_with_cascade(route, user_msg, context, trace_id)
        
        # 4. 工具调用（如果模型返回了function_call）
        if response.has_tool_calls:
            tool_results = await self.tools.execute(response.tool_calls)
            response = await self.models[route].continue_with_tools(
                user_msg, tool_results
            )
        
        # 5. 安全过滤
        response = self.safety.filter(response)
        
        # 6. 质量检查 & 可选升级
        quality_score = self._assess_quality(response)
        if quality_score < 0.7 and route != "premium":
            response = await self.models["premium"].generate(user_msg, context)
        
        # 7. 缓存 & 监控
        self.cache.put(user_msg, response)
        self.monitor.record(trace_id, "quality", quality_score)
        
        return response
    
    async def _call_with_cascade(self, route, msg, ctx, trace_id):
        model_chain = self._get_cascade_chain(route)
        for model in model_chain:
            response, confidence = await model.generate_with_confidence(msg, ctx)
            self.monitor.record(trace_id, f"cascade_{model.name}", confidence)
            if confidence >= model.threshold:
                return response
        return response  # 最后一级
```

### 追问场景

**Q8.1: 复合系统中某个组件挂了怎么办？** ⭐⭐⭐

```python
class ResilientCompoundSystem:
    """带故障恢复的复合系统"""
    
    async def call_model_with_fallback(self, model_name, prompt):
        fallback_chain = ["gpt-4o", "claude-3.5", "qwen-max", "local-llama"]
        start_idx = fallback_chain.index(model_name)
        
        for model in fallback_chain[start_idx:]:
            try:
                return await asyncio.wait_for(
                    self.models[model].generate(prompt),
                    timeout=10.0  # 超时控制
                )
            except (TimeoutError, APIError, RateLimitError):
                self.monitor.record_error(model)
                continue
        
        # 所有模型都失败，返回兜底响应
        return "抱歉，系统暂时繁忙，请稍后重试。"
```

关键原则：
- 每个组件都有降级方案
- 超时控制（不能让一个慢组件阻塞整个链路）
- 熔断器模式：某组件连续失败N次后，短路跳过

---

## 九、面试高频考点总结 🔥🔥🔥

### 高频问题速查表

| # | 问题 | 难度 | 关键点 |
|---|------|------|--------|
| 1 | 什么是复合AI系统？ | ⭐ | 多组件协同 > 单模型，BAIR定义 |
| 2 | 模型路由怎么做？ | ⭐⭐ | 规则/分类器/语义路由，路由器开销 |
| 3 | 级联和路由的区别？ | ⭐⭐ | 级联是逐级fallback，路由是一次性决策 |
| 4 | 如何做级联的置信度校准？ | ⭐⭐⭐ | 温度缩放、外部验证器、一致性 |
| 5 | Function Calling vs MCP？ | ⭐⭐ | 模型能力 vs 协议标准 |
| 6 | 复合系统怎么评估？ | ⭐⭐ | 端到端 + 组件级 + Trace贯穿 |
| 7 | 如何降低复合系统成本？ | ⭐⭐ | 路由+级联+缓存+压缩 |
| 8 | 生产系统的故障恢复？ | ⭐⭐⭐ | 降级链、超时、熔断器 |
| 9 | 语义缓存的风险？ | ⭐⭐ | 相似≠相同，缓存失效策略 |
| 10 | 多模型集成什么时候值得？ | ⭐⭐ | 高价值场景、离线评估 |

### 系统设计题模板

> **题目**：设计一个日均百万请求的智能问答系统，要求准确率>90%，成本< $100/天。

**答题框架：**

```
1. 需求分析
   - 流量分布：80%简单FAQ + 15%中等 + 5%复杂
   
2. 架构设计
   - 路由层：意图分类器分流
   - 模型层：3级级联（小→中→大）
   - 工具层：RAG检索 + Function Calling
   - 缓存层：语义缓存命中30%请求
   
3. 成本估算
   - 缓存命中30万：$0
   - 小模型56万：56万 × $0.0001 = $56
   - 中模型12万：12万 × $0.001 = $120  ← 超预算！
   
4. 优化调整
   - 提高缓存命中率到40%（优化embedding模型）
   - 提高小模型置信度阈值（处理更多）
   - 最终：缓存40万 + 小模型48万 + 中模型10万 + 大模型2万
   
5. 监控与迭代
   - 实时监控路由分布、级联率、成本
   - A/B测试持续优化
```

---

## 十、进阶：前沿方向 ⭐⭐⭐

### Q9: 复合AI系统的最新研究方向有哪些？

**A：**

**1）DSPy（Stanford）— 自动优化复合系统**

```python
# DSPy让复合系统的"编排"也能被优化
import dspy

class RAGModule(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# DSPy自动优化prompt和few-shot示例
optimizer = dspy.BootstrapFewShot(metric=exact_match)
optimized_rag = optimizer.compile(RAGModule(), trainset=train_data)
```

**2）RouteLLM — 开源路由框架**

```python
# 基于人类偏好数据训练路由器
from routellm.controller import Controller

controller = Controller(
    routers=["mf"],  # Matrix Factorization路由器
    strong_model="gpt-4o",
    weak_model="gpt-4o-mini"
)

# 自动路由，threshold控制质量-成本权衡
response = controller.chat.completions.create(
    model="router-mf-0.5",  # 0.5是阈值
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
```

**3）Compound AI Systems的自进化**

- 自动发现最优组件组合（Neural Architecture Search for compound systems）
- 基于用户反馈的在线学习路由策略
- 自动A/B测试和流量分配

---

## 总结

复合AI系统的核心哲学：

> **不要试图训练一个全能的大模型，而是用聪明的方式组合多个专长模型。**

关键记忆点：
1. **路由** = 分流（一次决策），**级联** = 逐级fallback
2. **MCP** 是工具连接的USB标准
3. 成本优化的核心公式：`大部分简单请求用小模型 + 缓存 + 级联保底`
4. 评估要**分层**：端到端 + 组件级 + 流程级
5. 生产系统必须有**降级方案**：每个组件都能failover
