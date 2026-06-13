# 10. Agent 工程实践

> **难度标记**：⭐ 基础 ⭐⭐ 进阶 ⭐⭐⭐ 高级
> **热度**：🔥🔥🔥 2025 年 Agent 工程师核心考点

---

## 知识图谱

```
Agent 工程实践
├── 技能系统设计 ⭐⭐⭐ 必考
│   ├── Skill 定义与注册
│   ├── Skill 发现与选择
│   ├── Skill 组合与编排
│   └── Skill 版本管理
├── Agent 框架对比 ⭐⭐ 进阶
│   ├── LangChain / LangGraph
│   ├── CrewAI / AutoGen
│   ├── Hermes Agent 架构
│   └── 框架选型策略
├── 自主 Agent 模式 ⭐⭐⭐ 核心
│   ├── Goal-Driven Agent
│   ├── Self-Reflection
│   ├── Self-Healing
│   └── Human-in-the-Loop
├── 工作流引擎 ⭐⭐ 进阶
│   ├── DAG 工作流
│   ├── 状态机模式
│   ├── 事件驱动架构
│   └── 长时任务管理
└── 生产部署 ⭐⭐⭐
    ├── 可观测性
    ├── 成本控制
    ├── 安全边界
    └── 多租户隔离
```

---

## 一、技能系统设计

### 1. ⭐⭐⭐ Q: 什么是 Agent Skill？如何设计一个技能系统？

**答**：

Skill 是 Agent 的**可复用能力单元**，类似于面向对象中的「方法」或微服务中的「服务」。

**Skill 的核心特征**：
```
一个 Skill 包含：
├── 触发条件（Trigger）：什么时候使用这个技能
├── 输入规范（Input）：需要什么参数
├── 执行逻辑（Logic）：怎么做
├── 输出规范（Output）：返回什么结果
├── 依赖声明（Dependencies）：需要什么工具/权限
└── 元数据（Metadata）：版本、作者、描述
```

**Skill 系统架构**：
```python
from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum

class SkillStatus(Enum):
    AVAILABLE = "available"
    DISABLED = "disabled"
    DEPRECATED = "deprecated"

@dataclass
class SkillDefinition:
    """技能定义"""
    name: str                          # 技能名称（唯一标识）
    description: str                   # 技能描述（给 LLM 看）
    version: str = "1.0.0"            # 语义化版本
    triggers: list[str] = field(default_factory=list)     # 触发条件关键词
    input_schema: dict = field(default_factory=dict)      # 输入 JSON Schema
    output_schema: dict = field(default_factory=dict)     # 输出 JSON Schema
    dependencies: list[str] = field(default_factory=list) # 依赖的工具/技能
    permissions: list[str] = field(default_factory=list)  # 需要的权限
    timeout: int = 30                  # 超时时间（秒）
    max_retries: int = 3              # 最大重试次数
    tags: list[str] = field(default_factory=list)         # 分类标签

class SkillRegistry:
    """技能注册中心"""

    def __init__(self):
        self.skills: dict[str, SkillDefinition] = {}
        self.executors: dict[str, Callable] = {}

    def register(self, skill_def: SkillDefinition, executor: Callable):
        """注册技能"""
        self.skills[skill_def.name] = skill_def
        self.executors[skill_def.name] = executor

    def discover(self, query: str, top_k: int = 5) -> list[SkillDefinition]:
        """根据查询发现相关技能"""
        # 1. 关键词匹配
        keyword_matches = []
        for skill in self.skills.values():
            score = 0
            query_lower = query.lower()
            # 匹配触发条件
            for trigger in skill.triggers:
                if trigger.lower() in query_lower:
                    score += 2
            # 匹配标签
            for tag in skill.tags:
                if tag.lower() in query_lower:
                    score += 1
            # 匹配描述
            if any(word in skill.description.lower() for word in query_lower.split()):
                score += 0.5
            if score > 0:
                keyword_matches.append((skill, score))

        # 2. 语义匹配（可选，用 embedding）
        # embedding_matches = self._semantic_search(query)

        # 3. 排序返回
        keyword_matches.sort(key=lambda x: x[1], reverse=True)
        return [skill for skill, _ in keyword_matches[:top_k]]

    async def execute(self, skill_name: str, inputs: dict) -> Any:
        """执行技能"""
        if skill_name not in self.skills:
            raise SkillNotFoundError(f"技能 {skill_name} 不存在")

        skill = self.skills[skill_name]
        executor = self.executors[skill_name]

        # 验证输入
        self._validate_inputs(inputs, skill.input_schema)

        # 检查权限
        await self._check_permissions(skill.permissions)

        # 执行（带超时和重试）
        for attempt in range(skill.max_retries):
            try:
                result = await asyncio.wait_for(
                    executor(**inputs),
                    timeout=skill.timeout
                )
                return result
            except asyncio.TimeoutError:
                if attempt == skill.max_retries - 1:
                    raise SkillTimeoutError(f"技能 {skill_name} 执行超时")
            except Exception as e:
                if attempt == skill.max_retries - 1:
                    raise SkillExecutionError(f"技能 {skill_name} 执行失败: {e}")
                await asyncio.sleep(2 ** attempt)  # 指数退避
```

**使用示例**：
```python
# 定义技能
web_search_skill = SkillDefinition(
    name="web_search",
    description="在互联网上搜索信息，返回相关网页摘要",
    version="2.1.0",
    triggers=["搜索", "查找", "search", "look up"],
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {"type": "integer", "default": 5}
        },
        "required": ["query"]
    },
    dependencies=["search_api"],
    permissions=["network.read"],
    tags=["搜索", "信息获取", "实时数据"]
)

# 注册技能
registry.register(web_search_skill, web_search_executor)

# Agent 使用
results = registry.discover("帮我搜索最新的 AI 新闻")
# → 返回 [web_search_skill, news_aggregator_skill, ...]

result = await registry.execute("web_search", {"query": "AI news 2025"})
```

---

### 2. ⭐⭐⭐ Q: Skill 的组合模式有哪些？如何实现技能编排？

**答**：

**模式一：串行链式（Pipeline）**
```
用户: "帮我总结这篇英文论文的核心观点"

技能链:
translate_to_chinese → summarize → format_output

每个技能的输出是下一个的输入
```

**模式二：并行扇出（Fan-out）**
```
用户: "对比 React 和 Vue 的优缺点"

并行执行:
├── search("React 优缺点")
├── search("Vue 优缺点")
└── search("React vs Vue 对比")
然后合并结果
```

**模式三：条件分支（Router）**
```
用户: "帮我处理这个文件"

路由判断:
├── 如果是 PDF → extract_pdf_text
├── 如果是图片 → ocr_image
├── 如果是代码 → analyze_code
└── 如果是 CSV → parse_csv
```

**模式四：循环迭代（Loop）**
```
用户: "帮我写一篇高质量的文章"

循环:
draft → review → revise → review → ... → finalize
（直到 review 评分 > 8 分）
```

**实现代码**：
```python
class SkillOrchestrator:
    """技能编排器"""

    def __init__(self, registry: SkillRegistry, llm):
        self.registry = registry
        self.llm = llm

    async def pipeline(self, skills: list[str], initial_input: dict) -> Any:
        """串行管道"""
        result = initial_input
        for skill_name in skills:
            result = await self.registry.execute(skill_name, result)
        return result

    async def fan_out(self, tasks: list[dict]) -> list:
        """并行执行"""
        coroutines = [
            self.registry.execute(t["skill"], t["input"])
            for t in tasks
        ]
        return await asyncio.gather(*coroutines, return_exceptions=True)

    async def route(self, query: str, candidates: list[str]) -> Any:
        """LLM 路由选择技能"""
        # 构建技能描述
        skill_descriptions = "\n".join([
            f"- {name}: {self.registry.skills[name].description}"
            for name in candidates
        ])

        prompt = f"""根据用户查询，选择最合适的技能。

用户查询: {query}

可选技能:
{skill_descriptions}

只返回技能名称。"""

        chosen_skill = await self.llm.generate(prompt)
        return await self.registry.execute(chosen_skill.strip(), {"query": query})

    async def loop(self, skill: str, input_data: dict,
                   judge_fn: Callable, max_iterations: int = 5) -> Any:
        """循环执行直到满足条件"""
        result = None
        for i in range(max_iterations):
            result = await self.registry.execute(skill, input_data)

            # 判断是否满足条件
            if await judge_fn(result):
                return result

            # 更新输入（加入反馈）
            input_data["previous_result"] = result
            input_data["iteration"] = i + 1

        return result  # 达到最大迭代次数

    async def dag_execute(self, dag: dict) -> dict:
        """DAG 工作流执行"""
        results = {}
        in_degree = {node: 0 for node in dag["nodes"]}
        dependencies = {node: [] for node in dag["nodes"]}

        # 计算入度
        for edge in dag["edges"]:
            in_degree[edge["to"]] += 1
            dependencies[edge["to"]].append(edge["from"])

        # 拓扑排序执行
        queue = [n for n, d in in_degree.items() if d == 0]

        while queue:
            # 并行执行当前层所有节点
            current_tasks = []
            for node in queue:
                node_config = dag["nodes"][node]
                # 收集依赖结果
                dep_results = {dep: results[dep] for dep in dependencies[node]}
                current_tasks.append((node, node_config, dep_results))

            # 并行执行
            batch_results = await asyncio.gather(*[
                self._execute_node(config, dep_results)
                for _, config, dep_results in current_tasks
            ])

            # 更新结果和入度
            for (node, _, _), result in zip(current_tasks, batch_results):
                results[node] = result
                # 更新后续节点的入度
                for edge in dag["edges"]:
                    if edge["from"] == node:
                        in_degree[edge["to"]] -= 1

            # 找出新的可执行节点
            queue = [n for n, d in in_degree.items() if d == 0 and n not in results]

        return results
```

---

### 3. ⭐⭐ Q: 如何实现 Skill 的动态发现和自适应选择？

**答**：

传统方式是硬编码技能选择逻辑，高级方式是让 LLM **自主发现和选择**技能。

```python
class AdaptiveSkillSelector:
    """自适应技能选择器"""

    def __init__(self, registry: SkillRegistry, llm):
        self.registry = registry
        self.llm = llm
        self.usage_stats: dict[str, dict] = {}  # 技能使用统计

    async def select_and_execute(self, user_query: str) -> Any:
        """选择并执行技能"""

        # Step 1: 获取所有可用技能
        all_skills = self.registry.skills

        # Step 2: 构建技能目录（给 LLM 看）
        skill_catalog = self._build_catalog(all_skills)

        # Step 3: LLM 选择技能
        prompt = f"""你是一个智能助手，可以从以下技能中选择合适的来完成用户任务。

可用技能目录:
{skill_catalog}

用户任务: {user_query}

请输出 JSON:
{{
    "selected_skill": "技能名称",
    "reason": "选择原因",
    "inputs": {{"参数名": "参数值"}}
}}

如果没有合适的技能，输出:
{{"selected_skill": null, "reason": "原因", "fallback": "直接回答"}}"""

        decision = json.loads(await self.llm.generate(prompt))

        if decision["selected_skill"] is None:
            # 没有合适技能，直接用 LLM 回答
            return await self.llm.generate(user_query)

        # Step 4: 执行技能
        skill_name = decision["selected_skill"]
        inputs = decision["inputs"]

        try:
            result = await self.registry.execute(skill_name, inputs)
            self._record_usage(skill_name, success=True)
            return result
        except Exception as e:
            self._record_usage(skill_name, success=False)
            # 降级：直接用 LLM 回答
            return await self.llm.generate(user_query)

    def _build_catalog(self, skills: dict) -> str:
        """构建技能目录"""
        lines = []
        for name, skill in skills.items():
            # 添加成功率信息（帮助 LLM 选择）
            stats = self.usage_stats.get(name, {})
            success_rate = stats.get("success_rate", "N/A")

            lines.append(f"""
## {name} (v{skill.version})
- 描述: {skill.description}
- 触发条件: {', '.join(skill.triggers)}
- 输入: {json.dumps(skill.input_schema, ensure_ascii=False)}
- 成功率: {success_rate}
- 标签: {', '.join(skill.tags)}""")
        return "\n".join(lines)

    def _record_usage(self, skill_name: str, success: bool):
        """记录使用统计"""
        if skill_name not in self.usage_stats:
            self.usage_stats[skill_name] = {"success": 0, "fail": 0}

        if success:
            self.usage_stats[skill_name]["success"] += 1
        else:
            self.usage_stats[skill_name]["fail"] += 1

        total = self.usage_stats[skill_name]["success"] + self.usage_stats[skill_name]["fail"]
        self.usage_stats[skill_name]["success_rate"] = \
            f"{self.usage_stats[skill_name]['success'] / total * 100:.1f}%"
```

---

## 二、Agent 框架对比

### 4. ⭐⭐⭐ Q: 主流 Agent 框架有哪些？如何选型？

**答**：

**2025 年主流框架对比**：

| 框架 | 类型 | 核心特点 | 适用场景 | 学习曲线 |
|------|------|----------|----------|----------|
| **LangChain** | 通用框架 | 生态最大，组件最多 | 快速原型、RAG | 中 |
| **LangGraph** | 工作流引擎 | 图状态机，精确控制流 | 复杂工作流 | 高 |
| **CrewAI** | 多 Agent | 角色扮演，团队协作 | 多 Agent 协作 | 低 |
| **AutoGen** | 多 Agent | 对话式协作，微软出品 | 研究、多轮对话 | 中 |
| **Hermes Agent** | 自主 Agent | 技能系统，持久记忆，工具编排 | 个人助手、自动化 | 低 |
| **DSPy** | 编程框架 | 声明式 LM 编程，自动优化 | RAG 优化、研究 | 高 |
| **OpenAI Agents SDK** | 官方 SDK | 原生支持，简单直接 | OpenAI 生态 | 低 |

**选型决策树**：
```
你的需求是什么？
│
├── 快速原型验证？ ──▶ LangChain（生态大，示例多）
│
├── 复杂工作流？ ──▶ LangGraph（图状态机，精确控制）
│
├── 多 Agent 协作？ ──▶ CrewAI（简单）或 AutoGen（灵活）
│
├── 个人助手/自动化？ ──▶ Hermes Agent（技能+记忆+工具）
│
├── 优化 Prompt/检索？ ──▶ DSPy（自动优化）
│
└── 只用 OpenAI？ ──▶ OpenAI Agents SDK（最简单）
```

**Hermes Agent 架构特点**：
```
Hermes Agent
├── 技能系统（Skills）
│   ├── 可复用的 SKILL.md 文件
│   ├── 触发条件 + 执行步骤
│   ├── 动态加载和热更新
│   └── 版本管理和继承
├── 记忆系统（Memory）
│   ├── 持久化记忆（跨会话）
│   ├── 用户画像
│   └── 环境知识
├── 工具编排（Tools）
│   ├── 终端/文件/浏览器/搜索
│   ├── MCP 集成
│   └── 子任务委派
├── 会话管理
│   ├── 多轮对话
│   ├── 上下文压缩
│   └── 会话搜索
└── 多平台集成
    ├── Telegram / Discord / Feishu
    ├── CLI / Web
    └── 定时任务（Cron）
```

---

### 5. ⭐⭐ Q: LangGraph 的核心概念是什么？和 LangChain 有什么区别？

**答**：

LangGraph 是 LangChain 团队推出的**图状态机**框架，专为复杂 Agent 工作流设计。

**核心概念**：

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

# 1. 定义状态
class AgentState(TypedDict):
    messages: list
    current_tool: str
    iteration: int
    result: str

# 2. 定义节点（处理函数）
def think(state: AgentState) -> AgentState:
    """思考节点：决定下一步"""
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def use_tool(state: AgentState) -> AgentState:
    """工具节点：调用工具"""
    tool_call = extract_tool_call(state["messages"][-1])
    result = tools[tool_call.name].invoke(tool_call.args)
    return {
        "messages": state["messages"] + [ToolMessage(result)],
        "iteration": state["iteration"] + 1
    }

def should_continue(state: AgentState) -> str:
    """条件边：决定下一步"""
    if state["iteration"] > 5:
        return "end"
    if has_tool_calls(state["messages"][-1]):
        return "use_tool"
    return "end"

# 3. 构建图
graph = StateGraph(AgentState)

# 添加节点
graph.add_node("think", think)
graph.add_node("use_tool", use_tool)

# 添加边
graph.set_entry_point("think")
graph.add_conditional_edges("think", should_continue, {
    "use_tool": "use_tool",
    "end": END
})
graph.add_edge("use_tool", "think")  # 工具调用后回到思考

# 4. 编译执行
app = graph.compile()
result = app.invoke({"messages": [HumanMessage("...")], "iteration": 0})
```

**LangChain vs LangGraph**：

| 维度 | LangChain | LangGraph |
|------|-----------|-----------|
| 编程模型 | 链式调用（Chain） | 图状态机（Graph） |
| 控制流 | 线性，分支有限 | 任意图，支持循环 |
| 状态管理 | 隐式 | 显式（TypedDict） |
| 适用场景 | 简单 RAG、单步任务 | 复杂 Agent、多步推理 |
| 调试 | 困难 | 容易（可视化图） |

---

## 三、自主 Agent 模式

### 6. ⭐⭐⭐ Q: 什么是 Self-Reflection Agent？如何实现？

**答**：

Self-Reflection Agent 是指 Agent 能够**评估自己的输出**并**自我改进**。

```
普通 Agent:
任务 → 生成 → 输出（不管好坏）

Self-Reflection Agent:
任务 → 生成 → 评估 → 不满意 → 重新生成 → 评估 → 满意 → 输出
                    ↓
              反思：哪里不好？怎么改进？
```

**实现**：
```python
class SelfReflectingAgent:
    """自我反思 Agent"""

    def __init__(self, llm, max_reflections=3):
        self.llm = llm
        self.max_reflections = max_reflections

    async def run(self, task: str) -> str:
        """带自我反思的任务执行"""
        attempts = []
        feedback = None

        for i in range(self.max_reflections):
            # 1. 生成回答
            if feedback:
                prompt = f"""任务: {task}

你之前的回答:
{attempts[-1]}

收到的反馈:
{feedback}

请根据反馈改进你的回答。"""
            else:
                prompt = task

            response = await self.llm.generate(prompt)
            attempts.append(response)

            # 2. 自我评估
            evaluation = await self._evaluate(task, response)

            if evaluation["score"] >= 8:
                # 满意，返回
                return response

            # 3. 生成改进建议
            feedback = await self._reflect(task, response, evaluation)

        # 达到最大次数，返回最后一次
        return attempts[-1]

    async def _evaluate(self, task: str, response: str) -> dict:
        """评估回答质量"""
        prompt = f"""评估以下回答的质量（1-10 分）。

任务: {task}
回答: {response}

评估维度:
1. 准确性（信息是否正确）
2. 完整性（是否覆盖所有要点）
3. 清晰度（表达是否清楚）
4. 实用性（是否可直接使用）

输出 JSON: {{"score": 7, "reason": "..."}}"""

        return json.loads(await self.llm.generate(prompt))

    async def _reflect(self, task: str, response: str, evaluation: dict) -> str:
        """反思并生成改进建议"""
        prompt = f"""你是这个回答的审查者。请给出具体的改进建议。

任务: {task}
当前回答: {response[:500]}
评估结果: {evaluation['reason']}

请列出 2-3 个具体的改进点。"""

        return await self.llm.generate(prompt)
```

---

### 7. ⭐⭐⭐ Q: 什么是 Self-Healing Agent？如何处理错误？

**答**：

Self-Healing Agent 能够**自动检测错误、分析原因、尝试修复**。

```python
class SelfHealingAgent:
    """自我修复 Agent"""

    def __init__(self, llm, tools, max_retries=3):
        self.llm = llm
        self.tools = tools
        self.max_retries = max_retries
        self.error_history: list[dict] = []

    async def execute_with_healing(self, task: str) -> str:
        """带自我修复的执行"""

        for attempt in range(self.max_retries):
            try:
                # 1. 执行任务
                result = await self._execute(task)
                return result

            except ToolExecutionError as e:
                # 2. 分析错误
                error_analysis = await self._analyze_error(task, e, attempt)

                # 3. 决定修复策略
                if error_analysis["strategy"] == "retry":
                    # 简单重试
                    await asyncio.sleep(2 ** attempt)
                    continue

                elif error_strategy == "fix_args":
                    # 修复参数
                    task = await self._fix_task_args(task, error_analysis)
                    continue

                elif error_strategy == "use_alternative":
                    # 使用替代工具
                    alt_tool = error_analysis["alternative"]
                    task = await self._switch_tool(task, alt_tool)
                    continue

                elif error_strategy == "decompose":
                    # 分解任务
                    subtasks = await self._decompose_task(task)
                    results = []
                    for subtask in subtasks:
                        r = await self.execute_with_healing(subtask)
                        results.append(r)
                    return "\n".join(results)

                elif error_strategy == "ask_human":
                    # 请求人工帮助
                    return f"遇到无法解决的问题: {e}\n建议: {error_analysis['suggestion']}"

                else:
                    raise

            except Exception as e:
                # 未知错误
                self.error_history.append({
                    "task": task,
                    "error": str(e),
                    "attempt": attempt
                })
                raise

    async def _analyze_error(self, task: str, error: Exception, attempt: int) -> dict:
        """分析错误并生成修复策略"""
        prompt = f"""分析以下错误并建议修复策略。

任务: {task}
错误: {str(error)}
尝试次数: {attempt + 1}
历史错误: {json.dumps(self.error_history[-3:], ensure_ascii=False)}

可选策略:
1. retry - 简单重试（网络超时等临时错误）
2. fix_args - 修复参数（参数格式错误）
3. use_alternative - 使用替代工具
4. decompose - 分解任务（任务太复杂）
5. ask_human - 请求人工帮助（无法自动解决）

输出 JSON: {{"strategy": "...", "reason": "...", "alternative": "...", "suggestion": "..."}}"""

        return json.loads(await self.llm.generate(prompt))

    async def _fix_task_args(self, task: str, analysis: dict) -> str:
        """修复任务参数"""
        prompt = f"""原始任务: {task}
错误原因: {analysis['reason']}

请修正任务中的参数或格式，生成新的任务描述。"""
        return await self.llm.generate(prompt)
```

---

### 8. ⭐⭐ Q: Human-in-the-Loop 模式有哪些？何时需要人工介入？

**答**：

**人工介入的时机**：

```
必须人工确认：
├── 发送邮件/消息（不可撤回）
├── 删除数据
├── 支付/财务操作
├── 修改生产环境配置
└── 对外发布内容

建议人工审核：
├── 生成长文档（可能有错误）
├── 代码修改（可能引入 Bug）
├── 数据分析结论（可能有偏见）
└── 决策建议（需要人类判断）

可以自动执行：
├── 信息搜索和整理
├── 格式转换
├── 数据查询
└── 计算任务
```

**实现**：
```python
class HumanInTheLoopAgent:
    """支持人工介入的 Agent"""

    CONFIRM_ACTIONS = {"send_email", "delete_data", "payment", "deploy"}
    REVIEW_ACTIONS = {"write_document", "modify_code", "analyze_data"}

    async def execute(self, action: str, params: dict) -> str:
        # 必须确认的操作
        if action in self.CONFIRM_ACTIONS:
            confirmed = await self._request_confirmation(action, params)
            if not confirmed:
                return "操作已取消"

        # 建议审核的操作
        if action in self.REVIEW_ACTIONS:
            result = await self._execute(action, params)
            reviewed = await self._request_review(action, result)
            if not reviewed:
                # 用户要求修改
                return await self._revise(action, result, reviewed["feedback"])
            return result

        # 直接执行
        return await self._execute(action, params)

    async def _request_confirmation(self, action: str, params: dict) -> bool:
        """请求用户确认"""
        # 通过消息平台发送确认请求
        message = f"即将执行: {action}\n参数: {json.dumps(params, ensure_ascii=False)}\n确认？(y/n)"
        response = await self.platform.ask_user(message)
        return response.lower() in ["y", "yes", "确认"]
```

---

## 四、工作流引擎

### 9. ⭐⭐⭐ Q: 如何设计一个 Agent 工作流引擎？

**答**：

```python
from enum import Enum
from typing import Any, Callable

class NodeType(Enum):
    LLM = "llm"           # LLM 调用
    TOOL = "tool"         # 工具调用
    CONDITION = "condition"  # 条件判断
    HUMAN = "human"       # 人工介入
    PARALLEL = "parallel" # 并行执行

@dataclass
class WorkflowNode:
    id: str
    type: NodeType
    config: dict
    next_nodes: list[str] = field(default_factory=list)
    condition: Callable = None  # 用于 CONDITION 类型

class WorkflowEngine:
    """Agent 工作流引擎"""

    def __init__(self, llm, tools, skill_registry):
        self.llm = llm
        self.tools = tools
        self.skills = skill_registry

    async def execute_workflow(self, workflow: dict, inputs: dict) -> dict:
        """执行工作流"""
        nodes = {n["id"]: n for n in workflow["nodes"]}
        context = {**inputs}
        current_node_id = workflow["entry"]
        execution_log = []

        while current_node_id and current_node_id != "__end__":
            node = nodes[current_node_id]

            execution_log.append({
                "node": current_node_id,
                "timestamp": time.time(),
                "status": "running"
            })

            try:
                # 执行节点
                result = await self._execute_node(node, context)
                context[f"{current_node_id}_result"] = result

                execution_log[-1]["status"] = "completed"
                execution_log[-1]["result"] = str(result)[:200]

                # 决定下一个节点
                current_node_id = await self._get_next_node(node, context)

            except Exception as e:
                execution_log[-1]["status"] = "failed"
                execution_log[-1]["error"] = str(e)

                # 错误处理
                if node.get("on_error"):
                    current_node_id = node["on_error"]
                else:
                    raise

        return {
            "result": context,
            "execution_log": execution_log
        }

    async def _execute_node(self, node: dict, context: dict) -> Any:
        """执行单个节点"""
        node_type = node["type"]

        if node_type == "llm":
            # LLM 节点
            prompt = self._render_template(node["config"]["prompt"], context)
            return await self.llm.generate(prompt)

        elif node_type == "tool":
            # 工具节点
            tool_name = node["config"]["tool"]
            args = self._render_args(node["config"]["args"], context)
            return await self.tools[tool_name].execute(**args)

        elif node_type == "skill":
            # 技能节点
            skill_name = node["config"]["skill"]
            inputs = self._render_args(node["config"]["inputs"], context)
            return await self.skills.execute(skill_name, inputs)

        elif node_type == "condition":
            # 条件节点（不执行，只在 _get_next_node 中处理）
            return None

        elif node_type == "parallel":
            # 并行节点
            tasks = node["config"]["tasks"]
            results = await asyncio.gather(*[
                self._execute_node(t, context) for t in tasks
            ])
            return results

    async def _get_next_node(self, node: dict, context: dict) -> str:
        """获取下一个节点"""
        if node["type"] == "condition":
            # 评估条件
            condition_result = await self._evaluate_condition(
                node["config"]["condition"], context
            )
            return node["branches"][str(condition_result)]

        # 默认返回第一个 next_node
        return node.get("next_nodes", [None])[0]
```

---

### 10. ⭐⭐ Q: 如何管理长时间运行的 Agent 任务？

**答**：

```python
class LongRunningTaskManager:
    """长时任务管理器"""

    def __init__(self):
        self.tasks: dict[str, dict] = {}

    async def submit_task(self, task_id: str, agent_fn: Callable, inputs: dict):
        """提交长时任务"""
        self.tasks[task_id] = {
            "status": "pending",
            "created_at": time.time(),
            "progress": 0,
            "result": None,
            "error": None,
            "checkpoints": []
        }

        # 异步执行
        asyncio.create_task(self._run_task(task_id, agent_fn, inputs))

    async def _run_task(self, task_id: str, agent_fn: Callable, inputs: dict):
        """执行任务（支持检查点）"""
        task = self.tasks[task_id]
        task["status"] = "running"

        try:
            # 执行，传入进度回调
            async def progress_callback(progress: int, checkpoint: dict):
                task["progress"] = progress
                task["checkpoints"].append(checkpoint)
                # 可以在这里通知用户
                await self._notify_progress(task_id, progress)

            result = await agent_fn(inputs, progress_callback)
            task["status"] = "completed"
            task["result"] = result
            task["completed_at"] = time.time()

        except Exception as e:
            task["status"] = "failed"
            task["error"] = str(e)

    async def get_status(self, task_id: str) -> dict:
        """查询任务状态"""
        task = self.tasks.get(task_id)
        if not task:
            return {"error": "任务不存在"}

        return {
            "task_id": task_id,
            "status": task["status"],
            "progress": task["progress"],
            "elapsed": time.time() - task["created_at"],
            "result": task["result"] if task["status"] == "completed" else None
        }

    async def resume_from_checkpoint(self, task_id: str):
        """从检查点恢复"""
        task = self.tasks[task_id]
        if not task["checkpoints"]:
            raise ValueError("没有可用的检查点")

        last_checkpoint = task["checkpoints"][-1]
        # 从检查点状态恢复执行
        ...
```

---

## 五、生产部署

### 11. ⭐⭐⭐ Q: Agent 系统的可观测性如何设计？

**答**：

```python
from opentelemetry import trace, metrics

tracer = trace.get_tracer("agent")
meter = metrics.get_meter("agent")

class ObservableAgent:
    """可观测的 Agent"""

    def __init__(self):
        # 指标
        self.task_counter = meter.create_counter("agent.tasks")
        self.task_duration = meter.create_histogram("agent.task.duration")
        self.tool_calls = meter.create_counter("agent.tool.calls")
        self.llm_tokens = meter.create_counter("agent.llm.tokens")
        self.errors = meter.create_counter("agent.errors")

    async def run(self, task: str) -> str:
        with tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("task", task[:100])
            start = time.time()

            try:
                result = await self._execute(task)

                duration = time.time() - start
                self.task_duration.record(duration)
                self.task_counter.add(1, {"status": "success"})

                span.set_attribute("duration", duration)
                span.set_attribute("result_length", len(result))

                return result

            except Exception as e:
                self.task_counter.add(1, {"status": "error"})
                self.errors.add(1, {"error_type": type(e).__name__})
                span.record_exception(e)
                raise

    async def call_tool(self, tool_name: str, args: dict) -> Any:
        with tracer.start_as_current_span("agent.tool") as span:
            span.set_attribute("tool", tool_name)
            span.set_attribute("args", json.dumps(args, ensure_ascii=False)[:200])

            result = await self.tools[tool_name].execute(**args)

            self.tool_calls.add(1, {"tool": tool_name})
            span.set_attribute("result_size", len(str(result)))

            return result

# 关键监控指标
MONITORING = {
    "业务指标": {
        "agent.task.success_rate": "任务成功率",
        "agent.task.duration_p95": "P95 延迟",
        "agent.task.iterations": "平均迭代次数",
    },
    "技术指标": {
        "agent.llm.tokens_per_task": "每任务 Token 消耗",
        "agent.llm.cost_per_task": "每任务成本",
        "agent.tool.error_rate": "工具调用错误率",
    },
    "资源指标": {
        "agent.concurrent_tasks": "并发任务数",
        "agent.queue_depth": "队列深度",
        "agent.memory_usage": "内存使用",
    }
}
```

---

### 12. ⭐⭐⭐ Q: Agent 系统如何做成本控制？

**答**：

```python
class CostController:
    """Agent 成本控制器"""

    def __init__(self, budget_config: dict):
        self.budgets = budget_config
        self.usage: dict[str, float] = {}

    async def check_budget(self, user_id: str, action: str, estimated_cost: float) -> bool:
        """检查预算"""
        user_budget = self.budgets.get(user_id, self.budgets["default"])

        # 检查单次任务预算
        if estimated_cost > user_budget["max_per_task"]:
            raise BudgetExceededError(
                f"单次任务成本 ${estimated_cost:.4f} 超过限制 ${user_budget['max_per_task']:.4f}"
            )

        # 检查日预算
        daily_usage = self.usage.get(f"{user_id}:daily", 0)
        if daily_usage + estimated_cost > user_budget["daily_limit"]:
            raise BudgetExceededError(
                f"日预算 ${daily_usage:.4f} + ${estimated_cost:.4f} 超过限制 ${user_budget['daily_limit']:.4f}"
            )

        return True

    async def record_usage(self, user_id: str, cost: float):
        """记录使用量"""
        key = f"{user_id}:daily"
        self.usage[key] = self.usage.get(key, 0) + cost

    def estimate_cost(self, task: str, model: str) -> float:
        """估算任务成本"""
        # 基于任务复杂度估算 token 数
        estimated_tokens = len(task) * 2  # 粗略估算

        # 不同模型的价格
        prices = {
            "gpt-4o": {"input": 2.5, "output": 10},
            "gpt-4o-mini": {"input": 0.15, "output": 0.6},
            "deepseek-chat": {"input": 0.14, "output": 0.28},
        }

        price = prices.get(model, prices["gpt-4o-mini"])
        return (estimated_tokens * price["input"] + estimated_tokens * 2 * price["output"]) / 1_000_000

# 成本优化策略
COST_STRATEGIES = {
    "模型降级": "简单任务用小模型，复杂任务用大模型",
    "缓存复用": "相同查询使用缓存结果",
    "批量处理": "多个任务合并处理",
    "截断控制": "限制输入/输出 token 数",
    "提前终止": "达到满意结果立即停止",
}
```

---

## 总结

### 面试高频追问

1. **"Skill 和 Tool 什么区别？"** → Tool 是原子操作，Skill 是 Tool 的组合 + 业务逻辑
2. **"Agent 框架怎么选？"** → 看需求复杂度：简单用 SDK，中等用 LangChain，复杂用 LangGraph
3. **"Self-Reflection 会不会死循环？"** → 会，需要设置最大迭代次数和质量阈值
4. **"长时任务怎么做？"** → 异步执行 + 检查点 + 进度通知 + 状态查询
5. **"Agent 成本怎么控制？"** → 模型路由 + 缓存 + 预算限制 + 提前终止
