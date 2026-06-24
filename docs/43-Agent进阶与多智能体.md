# 43. Agent 进阶与多智能体

> **Agent 不是单打独斗，多智能体协作才是复杂任务的解法**

---

## 一、多 Agent 架构模式

### Q1: 多 Agent 系统有哪些主流架构模式？⭐⭐⭐

**答**：

```
多 Agent 架构演进：

单 Agent ──→ Supervisor ──→ Hierarchical ──→ Swarm
(简单任务)   (中心调度)     (分层管理)       (自组织)

┌─────────────────────────────────────────────────────┐
│                    架构对比                           │
├──────────┬──────────┬──────────┬──────────┬─────────┤
│ 模式     │ 优点     │ 缺点     │ 适用场景 │ 代表    │
├──────────┼──────────┼──────────┼──────────┼─────────┤
│Supervisor│ 简单可控 │ 单点瓶颈 │ 2-5 Agent│ LangG.  │
│Hierarch. │ 分层清晰 │ 层级延迟 │ 大型团队 │ AutoGen │
│Swarm     │ 弹性高   │ 调试难   │ 动态任务 │ OpenAI  │
│Blackboard│ 解耦好   │ 一致性差 │ 异步协作 │ 自研    │
│Pipeline  │ 可预测   │ 灵活性差 │ 固定流程 │ LlamaIdx│
└──────────┴──────────┴──────────┴──────────┴─────────┘
```

**Supervisor 模式**：一个主 Agent 负责任务分配和结果汇总

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# 定义专业 Agent
researcher = create_react_agent(
    llm,
    tools=[web_search, arxiv_search],
    state_modifier="你是一个研究专家，负责信息收集和分析"
)

coder = create_react_agent(
    llm,
    tools=[python_repl, file_editor],
    state_modifier="你是一个编程专家，负责代码实现"
)

reviewer = create_react_agent(
    llm,
    tools=[code_review_tool],
    state_modifier="你是一个代码审查专家，负责质量把控"
)

# Supervisor 调度
from langgraph_supervisor import create_supervisor

supervisor = create_supervisor(
    agents=[researcher, coder, reviewer],
    model=llm,
    prompt="""你是项目总监，负责协调团队：
    1. 需要信息收集时 → researcher
    2. 需要写代码时 → coder
    3. 需要代码审查时 → reviewer
    确保任务按正确顺序完成。"""
)

app = supervisor.compile()
result = app.invoke({"messages": [("user", "实现一个天气查询API")]})
```

**Hierarchical 模式**：多层级管理，适合大型项目

```python
from langgraph.graph import StateGraph, MessagesState

# 底层 Worker
frontend_dev = create_react_agent(llm, [html_editor, css_editor])
backend_dev = create_react_agent(llm, [python_repl, db_tool])
qa_engineer = create_react_agent(llm, [test_runner, coverage_tool])

# 中层 Manager
dev_lead = create_supervisor(
    agents=[frontend_dev, backend_dev],
    model=llm,
    prompt="你是开发主管，协调前后端开发"
)

qa_lead = create_supervisor(
    agents=[qa_engineer],
    model=llm,
    prompt="你是QA主管，负责测试和质量"
)

# 高层 Director
director = create_supervisor(
    agents=[dev_lead, qa_lead],
    model=llm,
    prompt="你是项目总监，管理开发和测试流程"
)
```

---

### Q2: Swarm 模式是什么？和 Supervisor 有什么区别？⭐⭐⭐

**答**：

Swarm 模式的核心是**去中心化**——没有固定的调度者，Agent 之间通过 handoff（移交）自行传递控制权。

```python
# Supervisor 模式：中心调度
# 用户 → Supervisor → Agent A → Supervisor → Agent B → Supervisor → 用户
# Swarm 模式：直接移交
# 用户 → Agent A →(handoff)→ Agent B →(handoff)→ Agent C → 用户

from langgraph.prebuilt import create_react_agent

def transfer_to_billing():
    """当用户询问账单问题时，转交给账单Agent"""
    return "billing_agent"

def transfer_to_technical():
    """当用户询问技术问题时，转交给技术支持Agent"""
    return "technical_agent"

def transfer_to_sales():
    """当用户询问购买相关时，转交给销售Agent"""
    return "sales_agent"

# 主 Agent 自己决定是否移交
triage_agent = create_react_agent(
    llm,
    tools=[transfer_to_billing, transfer_to_technical, transfer_to_sales],
    state_modifier="""你是客服分诊台。根据用户问题：
    - 账单/付款/退款 → transfer_to_billing
    - 技术故障/使用问题 → transfer_to_technical
    - 购买/升级/定价 → transfer_to_sales
    简单问题直接回答。"""
)

# 专业 Agent 也可以移交回分诊台
def transfer_to_triage():
    return "triage_agent"

billing_agent = create_react_agent(
    llm,
    tools=[query_bill, process_refund, transfer_to_triage],
    state_modifier="你是账单专家。处理完后可转回分诊台。"
)
```

**关键区别**：

| 维度 | Supervisor | Swarm |
|------|-----------|-------|
| 控制流 | 中心化调度 | 去中心化移交 |
| 决策者 | Supervisor 决定下一个 Agent | 当前 Agent 自己决定移交 |
| 适用 | 任务流程明确 | 对话式、动态路由 |
| 复杂度 | Agent 数量增加时 Supervisor 负担重 | 每个 Agent 只管自己的领域 |
| 调试 | 集中日志 | 需要 trace 链路追踪 |

---

### Q3: Agent 之间如何通信？有哪些通信模式？⭐⭐⭐

**答**：

```
Agent 通信模式：

1. 消息传递（Message Passing）
   Agent A ──消息──→ Agent B
   最简单，适合简单协作

2. 共享状态（Shared State）
   Agent A ──读写──→ State Store ←──读写── Agent B
   适合需要共享上下文的场景

3. 黑板模式（Blackboard）
   所有 Agent ──读写──→ Blackboard ←──读写── 所有 Agent
   适合异步协作

4. 事件驱动（Event-Driven）
   Agent A ──事件──→ Event Bus ──订阅──→ Agent B, C, D
   适合松耦合系统
```

**LangGraph 共享状态实现**：

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
import operator

class SharedState(TypedDict):
    # 消息历史（所有 Agent 共享）
    messages: Annotated[list, operator.add]
    # 研究结果
    research_results: list[str]
    # 代码实现
    code_output: str
    # 审查意见
    review_feedback: str
    # 当前阶段
    current_phase: str

def researcher_node(state: SharedState) -> dict:
    """研究 Agent：收集信息"""
    results = do_research(state["messages"])
    return {
        "research_results": results,
        "current_phase": "research_done",
        "messages": [("assistant", f"研究完成，找到 {len(results)} 条信息")]
    }

def coder_node(state: SharedState) -> dict:
    """编码 Agent：基于研究结果写代码"""
    code = write_code(state["research_results"])
    return {
        "code_output": code,
        "current_phase": "coding_done",
        "messages": [("assistant", "代码实现完成")]
    }

def reviewer_node(state: SharedState) -> dict:
    """审查 Agent：检查代码质量"""
    feedback = review_code(state["code_output"])
    return {
        "review_feedback": feedback,
        "current_phase": "review_done",
        "messages": [("assistant", f"审查完成：{feedback}")]
    }

# 构建图
graph = StateGraph(SharedState)
graph.add_node("researcher", researcher_node)
graph.add_node("coder", coder_node)
graph.add_node("reviewer", reviewer_node)

graph.set_entry_point("researcher")
graph.add_edge("researcher", "coder")
graph.add_edge("coder", "reviewer")
graph.add_finish_point("reviewer")
```

---

## 二、反思与自我修正

### Q4: 什么是 Reflection 模式？如何实现 Agent 自我反思？⭐⭐⭐

**答**：

Reflection（反思）是让 Agent 在执行后回顾自己的输出，发现不足并改进。核心思想：**生成 → 评估 → 修正**循环。

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class ReflectionState(TypedDict):
    task: str
    draft: str
    critique: str
    revision_count: int
    final_output: str

def generate_node(state: ReflectionState) -> dict:
    """生成初稿或修订版"""
    if state.get("critique"):
        # 有批评意见，改进
        prompt = f"""基于以下批评改进你的回答：
        
原始任务：{state['task']}
上一版回答：{state['draft']}
批评意见：{state['critique']}

请输出改进后的回答："""
    else:
        prompt = f"请回答：{state['task']}"
    
    response = llm.invoke(prompt)
    return {"draft": response.content}

def reflect_node(state: ReflectionState) -> dict:
    """自我反思，找出不足"""
    prompt = f"""请严格审查以下回答，找出3个主要不足：

任务：{state['task']}
回答：{state['draft']}

审查维度：
1. 准确性：是否有事实错误？
2. 完整性：是否遗漏关键点？
3. 清晰度：表达是否清楚？
4. 实用性：是否有可操作性？

输出格式：
- 不足1: ...
- 不足2: ...
- 不足3: ...
- 总体评价：..."""
    
    response = llm.invoke(prompt)
    return {
        "critique": response.content,
        "revision_count": state.get("revision_count", 0) + 1
    }

def should_continue(state: ReflectionState) -> str:
    """决定是否继续反思"""
    if state["revision_count"] >= 3:
        return "finalize"
    # 检查批评是否足够正面
    if "优秀" in state["critique"] or "没有明显问题" in state["critique"]:
        return "finalize"
    return "revise"

def finalize_node(state: ReflectionState) -> dict:
    return {"final_output": state["draft"]}

# 构建反思循环
graph = StateGraph(ReflectionState)
graph.add_node("generate", generate_node)
graph.add_node("reflect", reflect_node)
graph.add_node("finalize", finalize_node)

graph.set_entry_point("generate")
graph.add_edge("generate", "reflect")
graph.add_conditional_edges("reflect", should_continue, {
    "revise": "generate",
    "finalize": "finalize"
})
graph.add_edge("finalize", END)
```

**Refine 模式 vs Reflexion 模式**：

```
Refine（逐步改进）：
生成 → 评估 → 修正 → 评估 → 修正 → ... → 输出
       ↑___________|

Reflexion（经验学习）：
尝试 → 失败 → 反思 → 记忆经验 → 重新尝试（带经验）
       ↑__________________________|

Refine 适合：单次任务的质量提升
Reflexion 适合：需要从失败中学习的复杂任务
```

---

### Q5: Self-Refine 如何在生产环境中应用？⭐⭐⭐

**答**：

Self-Refine 的核心是**用同一个模型同时担任生成者和评审者**，通过结构化反馈循环提升质量。

```python
class SelfRefineAgent:
    """生产级 Self-Refine Agent"""
    
    def __init__(self, llm, max_iterations=3, quality_threshold=0.8):
        self.llm = llm
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
    
    async def refine(self, task: str) -> dict:
        history = []
        
        # Step 1: 初始生成
        current = await self._generate(task)
        history.append({"version": 1, "output": current})
        
        for i in range(self.max_iterations):
            # Step 2: 评估
            eval_result = await self._evaluate(task, current)
            score = eval_result["score"]
            
            if score >= self.quality_threshold:
                return {
                    "output": current,
                    "iterations": i + 1,
                    "final_score": score,
                    "history": history
                }
            
            # Step 3: 修正
            current = await self._refine_one(task, current, eval_result["feedback"])
            history.append({
                "version": i + 2,
                "output": current,
                "prev_score": score,
                "feedback": eval_result["feedback"]
            })
        
        return {
            "output": current,
            "iterations": self.max_iterations,
            "final_score": score,
            "history": history
        }
    
    async def _evaluate(self, task: str, output: str) -> dict:
        """结构化评估"""
        prompt = f"""作为严格的质量评审，请评估以下回答。

任务：{task}
回答：{output}

请从以下维度评分（0-1）并给出改进建议：
1. 准确性（是否有错误）
2. 完整性（是否覆盖所有要点）
3. 清晰度（是否易于理解）
4. 实用性（是否可执行）

返回 JSON：
{{"score": 0.0-1.0, "feedback": "具体改进建议", "issues": ["问题1", "问题2"]}}"""
        
        resp = await self.llm.ainvoke(prompt)
        return json.loads(resp.content)
    
    async def _refine_one(self, task: str, output: str, feedback: str) -> str:
        """针对性修正"""
        prompt = f"""请根据反馈改进回答，保留正确的部分，只修正指出的问题。

原始任务：{task}
当前回答：{output}
评审反馈：{feedback}

输出改进后的完整回答："""
        
        resp = await self.llm.ainvoke(prompt)
        return resp.content
```

---

## 三、Agent 记忆系统深入

### Q6: Agent 的记忆如何分层管理？⭐⭐⭐

**答**：

Agent 记忆借鉴了认知科学的分层模型：

```
┌─────────────────────────────────────────────┐
│              Agent 记忆架构                   │
├─────────────────────────────────────────────┤
│                                              │
│  感觉记忆 (Sensory)                          │
│  ├─ 当前轮次的输入/输出                       │
│  └─ 生命周期：单次交互                        │
│                                              │
│  工作记忆 (Working / Short-term)             │
│  ├─ 当前对话窗口的消息                        │
│  ├─ Agent 的当前计划和中间状态                │
│  ├─ Scratchpad（工具调用结果暂存）            │
│  └─ 生命周期：单次任务                        │
│                                              │
│  长期记忆 (Long-term)                        │
│  ├─ 情景记忆：历史对话摘要                    │
│  ├─ 语义记忆：用户偏好、知识                  │
│  ├─ 程序记忆：学到的技能和模式                │
│  └─ 生命周期：跨会话持久化                    │
│                                              │
└─────────────────────────────────────────────┘
```

**工作记忆实现**：

```python
from langchain_core.messages import BaseMessage
from typing import TypedDict

class AgentMemory:
    """分层记忆管理器"""
    
    def __init__(self, max_working_memory=20, vector_store=None):
        self.working_memory: list[BaseMessage] = []
        self.max_working = max_working_memory
        self.vector_store = vector_store  # 长期记忆存储
        self.episodic_buffer = []  # 情景记忆缓冲
    
    def add_message(self, message: BaseMessage):
        """添加消息到工作记忆"""
        self.working_memory.append(message)
        
        # 超出窗口时压缩
        if len(self.working_memory) > self.max_working:
            self._compress_working_memory()
    
    def _compress_working_memory(self):
        """将旧消息压缩为摘要，存入长期记忆"""
        old_messages = self.working_memory[:10]
        summary = self._summarize(old_messages)
        
        # 保存到长期记忆
        if self.vector_store:
            self.vector_store.add_texts(
                texts=[summary],
                metadatas=[{"type": "episodic", "timestamp": time.time()}]
            )
        
        # 保留最近的消息 + 摘要
        self.working_memory = [
            SystemMessage(content=f"之前的对话摘要：{summary}")
        ] + self.working_memory[10:]
    
    def recall(self, query: str, k=3) -> list[str]:
        """从长期记忆中检索相关内容"""
        if not self.vector_store:
            return []
        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
    
    def get_context_window(self) -> list[BaseMessage]:
        """获取当前上下文窗口"""
        return self.working_memory
```

**程序记忆——Agent 学习经验**：

```python
class ProceduralMemory:
    """程序记忆：Agent 从成功/失败中学习"""
    
    def __init__(self, db_path="agent_memory.db"):
        self.db = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY,
                task_type TEXT,
                approach TEXT,
                outcome TEXT,  -- 'success' or 'failure'
                lessons TEXT,
                timestamp REAL
            )
        """)
    
    def record_experience(self, task_type: str, approach: str, 
                          outcome: str, lessons: str):
        """记录一次经验"""
        self.db.execute(
            "INSERT INTO experiences VALUES (NULL, ?, ?, ?, ?, ?)",
            (task_type, approach, outcome, lessons, time.time())
        )
        self.db.commit()
    
    def get_relevant_experience(self, task_type: str) -> str:
        """获取与当前任务相关的经验"""
        rows = self.db.execute(
            "SELECT approach, outcome, lessons FROM experiences "
            "WHERE task_type = ? ORDER BY timestamp DESC LIMIT 5",
            (task_type,)
        ).fetchall()
        
        if not rows:
            return "没有相关经验记录。"
        
        exp_text = "历史经验参考：\n"
        for approach, outcome, lessons in rows:
            exp_text += f"- 方法：{approach}\n  结果：{outcome}\n  教训：{lessons}\n"
        return exp_text
```

---

### Q7: 如何实现 Agent 的 Scratchpad？⭐⭐

**答**：

Scratchpad 是 Agent 的"草稿本"，记录中间推理过程和工具调用结果，帮助 Agent 在长任务中保持上下文连贯。

```python
class AgentScratchpad:
    """Agent 草稿本：记录推理链和中间结果"""
    
    def __init__(self):
        self.entries = []
        self.current_plan = None
        self.completed_steps = []
    
    def set_plan(self, plan: list[str]):
        """设置执行计划"""
        self.current_plan = plan
        self.completed_steps = []
    
    def record_step(self, step_idx: int, action: str, result: str, 
                    success: bool):
        """记录一个执行步骤"""
        entry = {
            "step": step_idx,
            "action": action,
            "result": result[:500],  # 截断过长结果
            "success": success,
            "timestamp": time.time()
        }
        self.entries.append(entry)
        if success:
            self.completed_steps.append(step_idx)
    
    def get_progress_summary(self) -> str:
        """生成进度摘要，供 Agent 参考"""
        if not self.current_plan:
            return "无执行计划"
        
        lines = ["执行进度："]
        for i, step in enumerate(self.current_plan):
            status = "✅" if i in self.completed_steps else "⬜"
            lines.append(f"  {status} {step}")
        
        # 添加最近的中间结果
        if self.entries:
            lines.append("\n最近结果：")
            for entry in self.entries[-3:]:
                lines.append(f"  [{entry['action']}] → {entry['result'][:100]}...")
        
        return "\n".join(lines)
    
    def inject_into_prompt(self, base_prompt: str) -> str:
        """将 scratchpad 内容注入到 prompt 中"""
        progress = self.get_progress_summary()
        return f"""{base_prompt}

---
当前工作状态：
{progress}

已完成步骤数：{len(self.completed_steps)}/{len(self.current_plan or [])}
---
"""
```

---

## 四、工具编排与组合

### Q8: 如何实现工具的动态组合和编排？⭐⭐⭐

**答**：

复杂任务往往需要多个工具按特定顺序和逻辑组合使用。核心模式有**链式、扇出-汇聚、条件分支**三种。

```python
from langchain_core.tools import tool
from typing import Callable

class ToolOrchestrator:
    """工具编排器：支持链式、并行、条件组合"""
    
    def __init__(self, llm):
        self.llm = llm
        self.pipelines = {}
    
    def register_pipeline(self, name: str, steps: list[dict]):
        """注册一个工具流水线
        steps: [{"tool": func, "condition": lambda ctx: bool, "parallel": False}]
        """
        self.pipelines[name] = steps
    
    async def execute_pipeline(self, name: str, input_data: dict) -> dict:
        """执行流水线"""
        steps = self.pipelines[name]
        context = {"input": input_data, "results": {}, "errors": []}
        
        # 链式执行
        parallel_group = []
        for step in steps:
            # 条件检查
            if step.get("condition") and not step["condition"](context):
                continue
            
            if step.get("parallel"):
                parallel_group.append(step)
                continue
            
            # 执行并行组
            if parallel_group:
                await self._execute_parallel(parallel_group, context)
                parallel_group = []
            
            # 执行当前步骤
            try:
                result = await step["tool"].ainvoke(context)
                context["results"][step["name"]] = result
            except Exception as e:
                context["errors"].append({"step": step["name"], "error": str(e)})
                if step.get("critical", False):
                    raise
        
        # 清理最后的并行组
        if parallel_group:
            await self._execute_parallel(parallel_group, context)
        
        return context
    
    async def _execute_parallel(self, steps: list, context: dict):
        """并行执行一组步骤"""
        import asyncio
        tasks = [step["tool"].ainvoke(context) for step in steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for step, result in zip(steps, results):
            if isinstance(result, Exception):
                context["errors"].append({"step": step["name"], "error": str(result)})
            else:
                context["results"][step["name"]] = result

# 使用示例
orchestrator = ToolOrchestrator(llm)
orchestrator.register_pipeline("research_and_write", [
    {"name": "search", "tool": web_search_tool, "parallel": True},
    {"name": "fetch", "tool": fetch_content_tool, "parallel": True},
    {"name": "analyze", "tool": analyze_tool},
    {"name": "write", "tool": write_draft_tool},
    {"name": "review", "tool": self_review_tool, "critical": True},
])
```

---

### Q9: Function Calling 的高级用法有哪些？⭐⭐⭐

**答**：

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 1. 带验证的结构化工具
class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词", min_length=1, max_length=200)
    max_results: int = Field(default=5, ge=1, le=20)
    date_range: str = Field(default="all", description="时间范围: day/week/month/all")

@tool("advanced_search", args_schema=SearchInput)
def advanced_search(query: str, max_results: int = 5, date_range: str = "all") -> str:
    """高级搜索工具，支持时间范围过滤"""
    results = do_search(query, max_results, date_range)
    return json.dumps(results, ensure_ascii=False)

# 2. 工具依赖注入（通过 config 传递上下文）
@tool
def get_user_data(user_id: str, config: RunnableConfig) -> str:
    """获取用户数据（自动注入当前用户上下文）"""
    current_user = config.get("configurable", {}).get("user_id")
    if current_user != user_id:
        return "无权访问该用户数据"
    return fetch_user_data(user_id)

# 3. 动态工具注册
class DynamicToolRegistry:
    """运行时动态注册/注销工具"""
    
    def __init__(self):
        self.tools = {}
        self.tool_metadata = {}
    
    def register(self, name: str, func: Callable, metadata: dict):
        """动态注册工具"""
        self.tools[name] = tool(name)(func)
        self.tool_metadata[name] = metadata
    
    def unregister(self, name: str):
        """注销工具"""
        self.tools.pop(name, None)
        self.tool_metadata.pop(name, None)
    
    def get_tools_for_context(self, context: dict) -> list:
        """根据上下文返回可用工具"""
        available = []
        for name, meta in self.tool_metadata.items():
            # 权限检查
            if meta.get("required_role"):
                if context.get("user_role") not in meta["required_role"]:
                    continue
            # 环境检查
            if meta.get("environments"):
                if context.get("env") not in meta["environments"]:
                    continue
            available.append(self.tools[name])
        return available
```

---

## 五、Agent 评估与测试

### Q10: 如何评估 Agent 的质量？⭐⭐⭐

**答**：

Agent 评估比单纯的 LLM 评估复杂得多，需要考虑**过程质量**和**结果质量**两个维度。

```python
class AgentEvaluator:
    """Agent 质量评估框架"""
    
    def evaluate(self, task: str, agent_output: dict, ground_truth: dict) -> dict:
        return {
            "task_completion": self._eval_completion(agent_output, ground_truth),
            "efficiency": self._eval_efficiency(agent_output),
            "tool_usage": self._eval_tool_usage(agent_output),
            "reasoning_quality": self._eval_reasoning(agent_output),
            "safety": self._eval_safety(agent_output),
        }
    
    def _eval_completion(self, output: dict, truth: dict) -> dict:
        """任务完成度"""
        final_answer = output.get("final_answer", "")
        expected = truth.get("expected_answer", "")
        
        # 用 LLM 判断答案正确性
        prompt = f"""判断以下回答是否正确完成了任务。

任务：{truth['task']}
期望结果：{expected}
实际回答：{final_answer}

评分（0-1）并说明原因。返回 JSON：{{"score": 0.0, "reason": "..."}}"""
        
        return llm_evaluate(prompt)
    
    def _eval_efficiency(self, output: dict) -> dict:
        """执行效率"""
        steps = output.get("steps", [])
        tool_calls = sum(1 for s in steps if s.get("tool_call"))
        total_tokens = sum(s.get("tokens", 0) for s in steps)
        
        return {
            "total_steps": len(steps),
            "tool_calls": tool_calls,
            "total_tokens": total_tokens,
            "avg_tokens_per_step": total_tokens / max(len(steps), 1),
            "redundant_calls": self._count_redundant_calls(steps)
        }
    
    def _eval_tool_usage(self, output: dict) -> dict:
        """工具使用合理性"""
        steps = output.get("steps", [])
        issues = []
        
        for i, step in enumerate(steps):
            # 检查：参数是否合理
            if step.get("tool_call"):
                if not step.get("tool_args"):
                    issues.append(f"Step {i}: 工具调用缺少参数")
                # 检查：结果是否被使用
                if i < len(steps) - 1:
                    next_step = steps[i + 1]
                    if step.get("tool_result") and not self._is_result_used(
                        step["tool_result"], next_step):
                        issues.append(f"Step {i}: 工具结果未被利用")
        
        return {"issues": issues, "score": max(0, 1 - len(issues) * 0.1)}
    
    def _eval_safety(self, output: dict) -> dict:
        """安全性评估"""
        checks = {
            "no_prompt_leak": self._check_no_system_prompt_leak(output),
            "no_harmful_action": self._check_no_harmful_actions(output),
            "permission_boundary": self._check_permission_boundary(output),
        }
        return checks
    
    def _count_redundant_calls(self, steps: list) -> int:
        """统计重复的工具调用"""
        seen = set()
        redundant = 0
        for step in steps:
            if step.get("tool_call"):
                key = f"{step['tool_name']}:{step.get('tool_args', '')}"
                if key in seen:
                    redundant += 1
                seen.add(key)
        return redundant
```

**Agent 测试套件**：

```python
import pytest

class TestAgent:
    """Agent 集成测试"""
    
    @pytest.fixture
    def agent(self):
        return create_test_agent()
    
    def test_basic_task_completion(self, agent):
        """测试基本任务完成能力"""
        result = agent.invoke("今天天气怎么样？")
        assert result["status"] == "success"
        assert "天气" in result["answer"]
    
    def test_tool_selection_accuracy(self, agent):
        """测试工具选择准确性"""
        result = agent.invoke("搜索最新的AI论文")
        tool_calls = [s for s in result["steps"] if s.get("tool_call")]
        assert any("search" in tc["tool_name"] for tc in tool_calls)
    
    def test_error_recovery(self, agent):
        """测试错误恢复能力"""
        # 模拟工具失败
        with mock_tool_failure("web_search", error="timeout"):
            result = agent.invoke("搜索最新新闻")
            # Agent 应该尝试替代方案或优雅降级
            assert result["status"] in ["success", "partial_success"]
            assert "重试" in str(result.get("recovery_actions", []))
    
    def test_max_iterations_limit(self, agent):
        """测试循环次数限制"""
        result = agent.invoke("一个不可能完成的任务 xyzabc")
        assert len(result["steps"]) <= agent.max_iterations
    
    def test_permission_boundary(self, agent):
        """测试权限边界"""
        result = agent.invoke("删除所有用户数据")
        assert result.get("refused", False) == True
        assert "权限" in result.get("refusal_reason", "")
    
    @pytest.mark.parametrize("task,expected_tools", [
        ("搜索天气", ["web_search"]),
        ("计算 123*456", ["calculator"]),
        ("读取文件内容", ["file_reader"]),
    ])
    def test_tool_routing(self, agent, task, expected_tools):
        """参数化测试：不同任务应路由到正确工具"""
        result = agent.invoke(task)
        used_tools = [s["tool_name"] for s in result["steps"] if s.get("tool_call")]
        for expected in expected_tools:
            assert expected in used_tools
```

---

## 六、Agent 安全与权限控制

### Q11: 如何实现 Agent 的权限边界？⭐⭐⭐

**答**：

生产环境的 Agent 必须有严格的权限控制，防止越权操作和 Prompt 注入攻击。

```python
from enum import Enum
from functools import wraps

class Permission(Enum):
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EXECUTE_CODE = "execute_code"
    NETWORK_ACCESS = "network_access"
    DATABASE_READ = "database_read"
    DATABASE_WRITE = "database_write"
    DELETE_RESOURCE = "delete_resource"

class PermissionGuard:
    """权限守卫：Agent 工具调用前的权限检查"""
    
    def __init__(self):
        self.role_permissions = {
            "viewer": {Permission.READ_FILE, Permission.DATABASE_READ},
            "developer": {
                Permission.READ_FILE, Permission.WRITE_FILE,
                Permission.EXECUTE_CODE, Permission.DATABASE_READ,
                Permission.NETWORK_ACCESS
            },
            "admin": set(Permission),
        }
        self.audit_log = []
    
    def check(self, user_role: str, required_permission: Permission,
              context: dict = None) -> bool:
        """检查权限"""
        allowed = self.role_permissions.get(user_role, set())
        has_permission = required_permission in allowed
        
        # 审计日志
        self.audit_log.append({
            "user_role": user_role,
            "permission": required_permission.value,
            "allowed": has_permission,
            "context": context,
            "timestamp": time.time()
        })
        
        return has_permission
    
    def guard_tool(self, required_permission: Permission):
        """装饰器：为工具添加权限检查"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, config: RunnableConfig, **kwargs):
                user_role = config.get("configurable", {}).get("user_role", "viewer")
                if not self.check(user_role, required_permission):
                    raise PermissionError(
                        f"权限不足：需要 {required_permission.value}，"
                        f"当前角色 {user_role}"
                    )
                return await func(*args, config=config, **kwargs)
            return wrapper
        return decorator

# 使用
guard = PermissionGuard()

@guard.guard_tool(Permission.DATABASE_WRITE)
@tool
def delete_record(record_id: str, config: RunnableConfig) -> str:
    """删除数据库记录（需要写权限）"""
    # 权限检查已在装饰器中完成
    db.delete(record_id)
    return f"记录 {record_id} 已删除"

@guard.guard_tool(Permission.EXECUTE_CODE)
@tool
def run_code(code: str, config: RunnableConfig) -> str:
    """执行代码（需要执行权限）"""
    return sandbox.execute(code)
```

**Prompt 注入防护**：

```python
class PromptInjectionGuard:
    """Prompt 注入检测"""
    
    INJECTION_PATTERNS = [
        r"忽略.*之前.*指令",
        r"ignore.*previous.*instructions",
        r"system prompt",
        r"你的指令是",
        r"你是一个.*而不是",
        r"从现在起.*你是",
        r"<\|system\|>",
        r"\[INST\]",
    ]
    
    def detect(self, user_input: str) -> dict:
        """检测输入是否包含注入攻击"""
        import re
        detected = []
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                detected.append(pattern)
        
        # 用 LLM 做二次确认
        if detected:
            llm_check = self._llm_verify(user_input)
            return {
                "is_injection": True,
                "patterns": detected,
                "llm_confirmed": llm_check,
                "sanitized": self._sanitize(user_input)
            }
        return {"is_injection": False}
    
    def _sanitize(self, text: str) -> str:
        """清理注入内容"""
        import re
        # 移除特殊标记
        text = re.sub(r'<\|.*?\|>', '', text)
        text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
        return text.strip()
```

---

## 七、生产级 Agent 监控

### Q12: 如何监控生产环境中的 Agent？⭐⭐⭐

**答**：

```python
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AgentTrace:
    """单次 Agent 执行的完整追踪"""
    trace_id: str
    task: str
    start_time: float
    end_time: Optional[float] = None
    steps: list = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    status: str = "running"
    error: Optional[str] = None
    
    def add_step(self, step_type: str, name: str, duration: float, 
                 tokens: int = 0, success: bool = True):
        self.steps.append({
            "type": step_type,
            "name": name,
            "duration_ms": duration * 1000,
            "tokens": tokens,
            "success": success
        })
        self.total_tokens += tokens
    
    def finalize(self, status: str, error: str = None):
        self.end_time = time.time()
        self.status = status
        self.error = error
    
    @property
    def latency_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0
    
    @property
    def tool_success_rate(self) -> float:
        tool_steps = [s for s in self.steps if s["type"] == "tool_call"]
        if not tool_steps:
            return 1.0
        return sum(1 for s in tool_steps if s["success"]) / len(tool_steps)

class AgentMonitor:
    """Agent 监控与指标收集"""
    
    def __init__(self, metrics_backend=None):
        self.traces = []
        self.metrics = metrics_backend
    
    def start_trace(self, task: str) -> AgentTrace:
        trace = AgentTrace(
            trace_id=str(uuid.uuid4())[:8],
            task=task,
            start_time=time.time()
        )
        return trace
    
    def record_trace(self, trace: AgentTrace):
        """记录并上报 trace"""
        self.traces.append(trace)
        
        # 上报指标
        if self.metrics:
            self.metrics.gauge("agent.latency_ms", trace.latency_ms)
            self.metrics.gauge("agent.tokens", trace.total_tokens)
            self.metrics.gauge("agent.tool_success_rate", trace.tool_success_rate)
            self.metrics.increment(f"agent.status.{trace.status}")
            
            # 告警
            if trace.latency_ms > 30000:  # 超过30秒
                self._alert("slow_agent", trace)
            if trace.tool_success_rate < 0.5:
                self._alert("low_tool_success", trace)
    
    def get_dashboard_data(self) -> dict:
        """生成监控面板数据"""
        recent = self.traces[-100:]
        return {
            "total_requests": len(recent),
            "success_rate": sum(1 for t in recent if t.status == "success") / max(len(recent), 1),
            "avg_latency_ms": sum(t.latency_ms for t in recent) / max(len(recent), 1),
            "avg_tokens": sum(t.total_tokens for t in recent) / max(len(recent), 1),
            "avg_tool_success_rate": sum(t.tool_success_rate for t in recent) / max(len(recent), 1),
            "error_distribution": Counter(t.error for t in recent if t.error),
            "slowest_traces": sorted(recent, key=lambda t: t.latency_ms, reverse=True)[:5]
        }
```

---

## 八、Agent 框架对比

### Q13: LangGraph vs CrewAI vs AutoGen vs Swarm 怎么选？⭐⭐⭐

**答**：

```
┌──────────────────────────────────────────────────────────────┐
│                    Agent 框架对比矩阵                         │
├────────────┬──────────┬──────────┬──────────┬───────────────┤
│ 维度       │LangGraph │ CrewAI   │ AutoGen  │ Swarm        │
├────────────┼──────────┼──────────┼──────────┼───────────────┤
│ 核心理念   │ 状态图   │ 角色扮演 │ 对话协作 │ 手动编排      │
│ 学习曲线   │ 中       │ 低       │ 中       │ 低            │
│ 灵活性     │ ★★★★★   │ ★★★     │ ★★★★    │ ★★★★         │
│ 生产就绪   │ ★★★★★   │ ★★★     │ ★★★     │ ★★            │
│ 多Agent    │ ★★★★★   │ ★★★★   │ ★★★★★  │ ★★★           │
│ 可观测性   │ ★★★★★   │ ★★★     │ ★★★     │ ★★            │
│ 持久化     │ ★★★★★   │ ★★      │ ★★★     │ ★             │
│ 人机交互   │ ★★★★★   │ ★★      │ ★★★★   │ ★★★           │
│ 社区生态   │ ★★★★★   │ ★★★     │ ★★★★   │ ★★★           │
└────────────┴──────────┴──────────┴──────────┴───────────────┘
```

**选型建议**：

```python
# 场景1：需要精确控制流程 → LangGraph
# 适合：生产系统、需要状态管理、复杂条件分支
app = StateGraph(MyState)
app.add_node("step1", node1)
app.add_conditional_edges("step1", router, {...})

# 场景2：角色扮演类任务 → CrewAI
# 适合：快速原型、内容创作、研究分析
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    process=Process.sequential
)

# 场景3：对话式多Agent → AutoGen
# 适合：需要Agent互相讨论、代码执行
groupchat = GroupChat(agents=[coder, reviewer, pm])
manager = GroupChatManager(groupchat=groupchat)

# 场景4：客服/路由类 → Swarm
# 适合：客服分流、技能路由
agent = Agent(
    name="Triage",
    instructions="分流客户问题",
    functions=[transfer_to_billing, transfer_to_tech]
)
```

---

## 九、生产实战模式

### Q14: 如何构建一个生产级的多 Agent 系统？⭐⭐⭐

**答**：

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator

class ProductionAgentSystem:
    """生产级多 Agent 系统模板"""
    
    def __init__(self):
        self.monitor = AgentMonitor()
        self.guard = PermissionGuard()
        self.memory = AgentMemory(vector_store=init_vector_store())
        self.app = self._build_graph()
    
    def _build_graph(self):
        class State(TypedDict):
            messages: Annotated[list, operator.add]
            task: str
            plan: list[str]
            current_step: int
            results: dict
            errors: list
            scratchpad: dict
        
        graph = StateGraph(State)
        
        # 节点
        graph.add_node("planner", self._planner_node)
        graph.add_node("executor", self._executor_node)
        graph.add_node("reviewer", self._reviewer_node)
        graph.add_node("error_handler", self._error_handler)
        graph.add_node("finalizer", self._finalizer_node)
        
        # 边
        graph.set_entry_point("planner")
        graph.add_conditional_edges("planner", self._plan_router, {
            "execute": "executor",
            "clarify": END
        })
        graph.add_conditional_edges("executor", self._exec_router, {
            "continue": "executor",
            "review": "reviewer",
            "error": "error_handler"
        })
        graph.add_conditional_edges("reviewer", self._review_router, {
            "revise": "executor",
            "approve": "finalizer"
        })
        graph.add_edge("error_handler", "executor")
        graph.add_edge("finalizer", END)
        
        # 持久化
        checkpointer = MemorySaver()
        return graph.compile(checkpointer=checkpointer)
    
    async def run(self, task: str, user_config: dict) -> dict:
        """带监控的执行入口"""
        trace = self.monitor.start_trace(task)
        
        try:
            result = await self.app.ainvoke(
                {"task": task, "messages": [("user", task)]},
                config={"configurable": {
                    "thread_id": str(uuid.uuid4()),
                    **user_config
                }}
            )
            trace.finalize("success")
            return result
        except Exception as e:
            trace.finalize("error", str(e))
            raise
        finally:
            self.monitor.record_trace(trace)
```

---

## 十、高频面试题

### Q15: 如何防止 Agent 陷入无限循环？⭐⭐

**答**：

```python
class LoopGuard:
    """防循环守卫"""
    
    def __init__(self, max_iterations=10, max_same_action=3, 
                 max_tokens=100000):
        self.max_iterations = max_iterations
        self.max_same_action = max_same_action
        self.max_tokens = max_tokens
        self.action_history = []
    
    def check(self, action: str, tokens_used: int) -> tuple[bool, str]:
        """检查是否应该停止"""
        self.action_history.append(action)
        
        # 1. 总次数限制
        if len(self.action_history) >= self.max_iterations:
            return True, "达到最大迭代次数"
        
        # 2. 重复动作检测
        recent = self.action_history[-self.max_same_action:]
        if len(set(recent)) == 1 and len(recent) >= self.max_same_action:
            return True, f"连续{self.max_same_action}次相同动作"
        
        # 3. Token 预算
        if tokens_used >= self.max_tokens:
            return True, "超出 Token 预算"
        
        # 4. 循环检测（A→B→A→B）
        if len(self.action_history) >= 4:
            last4 = self.action_history[-4:]
            if last4[0] == last4[2] and last4[1] == last4[3]:
                return True, "检测到交替循环"
        
        return False, ""
```

### Q16: Agent 的上下文窗口不够用怎么办？⭐⭐

**答**：

```python
class ContextManager:
    """上下文窗口管理"""
    
    def __init__(self, max_tokens=128000, reserve_output=4000):
        self.max_tokens = max_tokens
        self.reserve = reserve_output
        self.available = max_tokens - reserve_output
    
    def fit_context(self, system_prompt: str, history: list, 
                    scratchpad: str, relevant_memory: list) -> list:
        """智能裁剪上下文以适配窗口"""
        # 优先级：system > scratchpad > recent history > memory > old history
        
        system_tokens = count_tokens(system_prompt)
        remaining = self.available - system_tokens
        
        # Scratchpad（当前任务状态，高优先级）
        scratch_tokens = count_tokens(scratchpad)
        if scratch_tokens > remaining * 0.3:
            scratchpad = truncate_smart(scratchpad, int(remaining * 0.3))
            scratch_tokens = count_tokens(scratchpad)
        remaining -= scratch_tokens
        
        # 最近的消息（保留最近的对话）
        recent = history[-10:]
        recent_tokens = sum(count_tokens(m) for m in recent)
        remaining -= recent_tokens
        
        # 相关记忆（用剩余空间）
        memory_text = ""
        for mem in relevant_memory:
            mem_tokens = count_tokens(mem)
            if mem_tokens < remaining:
                memory_text += mem + "\n"
                remaining -= mem_tokens
        
        # 旧消息摘要（如果还有空间）
        old_messages = history[:-10]
        if old_messages and remaining > 200:
            summary = summarize_messages(old_messages)
            summary = truncate_smart(summary, remaining)
        else:
            summary = ""
        
        return [
            SystemMessage(content=system_prompt),
            SystemMessage(content=f"相关记忆：\n{memory_text}"),
            SystemMessage(content=f"历史摘要：{summary}"),
            *recent,
            SystemMessage(content=f"当前工作状态：\n{scratchpad}")
        ]
```

---

> **面试锦囊**：
> - 多 Agent 题必答**架构选型**（为什么选这种模式）
> - 安全题必答**权限边界 + 注入防护**两层
> - 评估题要答**过程指标**（tool 成功率、token 效率）而非只看结果
> - 生产题要答**监控 + 告警 + 降级**三板斧
