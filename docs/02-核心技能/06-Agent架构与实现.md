# 06. Agent 架构与实现

> 面向大模型应用工程师 / Agent 开发工程师的高频面试题

---

## 一、Agent 基础概念

### Q: 什么是 AI Agent？和普通的 LLM 调用有什么区别？⭐

**答：**

AI Agent 是一个能够**自主感知环境、做出决策、执行行动**来完成目标的智能系统。你可以把它想象成一个"有手有脚"的大脑——普通 LLM 只能"想"，而 Agent 还能"做"。

**普通 LLM 调用**是"一问一答"模式：你给一段 Prompt，模型返回一段文本，事情就结束了。就像你问一个百科全书一个问题，翻到答案就合上了。

**AI Agent** 则是一个循环系统：它会**观察**当前状态 → **思考**下一步该做什么 → **执行**一个动作（比如调用工具、查询数据库）→ **观察**执行结果 → **继续思考**……直到任务完成。就像一个实习生，不只是回答问题，而是拿到需求后自己去查资料、写代码、测试、修改，直到交付。

核心区别可以用代码来说明：

```python
# 普通 LLM 调用：一问一答
response = llm.chat("北京今天天气怎么样？")
print(response)  # "抱歉，我没有实时数据" — 结束

# Agent：感知-思考-行动循环
def agent_loop(task):
    context = {"task": task, "history": []}
    for step in range(MAX_STEPS):
        # 1. 思考：下一步做什么？
        thought = llm.chat(build_prompt(context))
        # 2. 行动：执行工具调用
        if thought.action == "call_tool":
            result = tools[thought.tool_name](thought.tool_input)
            context["history"].append((thought, result))
        elif thought.action == "finish":
            return thought.answer
    return "达到最大步数，任务未完成"

# Agent 会自主：查天气API → 解析结果 → 格式化输出 → 返回
result = agent_loop("北京今天天气怎么样？")
```

**追问：**
- Agent 和"LLM + 工具链（Pipeline）"有什么区别？Pipeline 是预定义的固定流程，Agent 的执行路径是动态决策的。
- 为什么 Agent 是"有状态"的？因为它维护了历史记忆（对话历史、工具调用结果），每次决策都基于累积的上下文。

---

### Q: Agent 的核心能力有哪些？⭐⭐

**答：**

Agent 有四大核心能力，可以用"一个侦探办案"来类比：

| 能力 | 类比 | 说明 |
|------|------|------|
| **感知（Perception）** | 侦探看现场 | 接收用户输入、环境信息、工具返回结果 |
| **推理（Reasoning）** | 侦探分析案情 | 基于感知信息，规划下一步行动 |
| **行动（Action）** | 侦探去抓人 | 调用工具、查询数据库、发送请求 |
| **记忆（Memory）** | 侦探的笔记本 | 记住历史信息，支持长短期记忆 |

```python
class Agent:
    def __init__(self, llm, tools, memory):
        self.llm = llm          # 推理引擎
        self.tools = tools      # 行动能力
        self.memory = memory    # 记忆系统

    def perceive(self, user_input, tool_results):
        """感知：收集所有信息"""
        return {
            "user_input": user_input,
            "tool_results": tool_results,
            "history": self.memory.retrieve()
        }

    def reason(self, perception):
        """推理：决定下一步"""
        prompt = self.build_prompt(perception)
        return self.llm.chat(prompt)  # 输出：思考过程 + 下一步动作

    def act(self, decision):
        """行动：执行具体操作"""
        if decision.tool_name:
            return self.tools.execute(decision.tool_name, decision.args)
        return decision.answer  # 任务完成

    def remember(self, step_info):
        """记忆：存储本次交互"""
        self.memory.store(step_info)
```

这四个能力缺一不可：没有感知，Agent 是"瞎子"；没有推理，是"无头苍蝇"；没有行动，是"空想家"；没有记忆，是"金鱼脑"。

**追问：**
- 四个能力中哪个最难实现？记忆。短期记忆（上下文窗口）容易，但长期记忆的存储、检索、遗忘机制很复杂。
- 感知能力除了文本还能包括什么？多模态 Agent 可以感知图像、音频、视频，甚至传感器数据。

---

### Q: 什么是 Agentic Design？为什么是 AI 应用的下一个范式？⭐⭐

**答：**

Agentic Design（智能体化设计）是一种应用架构思想：**把 AI 从"工具"变成"协作者"**。传统 AI 应用是"人驱动 AI"——人一步步指示 AI 做事；Agentic Design 是"AI 驱动自己"——给一个目标，AI 自己拆解、规划、执行、验证。

为什么说它是下一个范式？三个原因：

1. **从"单次推理"到"持续执行"**：传统 LLM 应用是一次性调用，Agentic 应用是一个持续运行的进程。就像从"计算器"进化到"计算机"。
2. **从"被动响应"到"主动行动"**：Agent 可以主动查询信息、发起请求、监控事件，不需要人类每一步都触发。
3. **从"单一能力"到"组合能力"**：通过工具调用和多 Agent 协作，一个 Agent 系统可以完成远超单一模型能力的复杂任务。

```python
# 传统范式：人驱动 AI
user_step1 = "帮我查一下这个客户的订单"
result1 = llm.chat(user_step1)
user_step2 = "看看有没有退货"  # 人来决定下一步
result2 = llm.chat(user_step2)

# Agentic 范式：AI 自驱动
agent.run("帮我处理客户张三的售后问题，确保他满意")
# Agent 自主：查订单 → 查退货记录 → 判断是否在保修期 → 生成处理方案 → 发送邮件
```

**追问：**
- Agentic Design 和传统的自动化脚本（如 RPA）有什么区别？RPA 是固定流程，Agentic Design 是动态决策，能处理未预见的情况。
- Agentic Design 的挑战是什么？可控性、可解释性、成本控制、安全边界。

---

## 二、ReAct 架构

### Q: ReAct 循环是怎么工作的？⭐⭐

**答：**

ReAct（Reasoning + Acting）是目前最经典的 Agent 架构，核心思想是**交替进行"思考"和"行动"**。就像一个人解决问题时的思维过程：想一步、做一步、看结果、再想下一步。

工作流程可以类比为"剥洋葱"——每一轮循环，Agent 都会：
1. **Thought（思考）**：分析当前情况，决定下一步做什么
2. **Action（行动）**：调用一个工具执行具体操作
3. **Observation（观察）**：获取行动的结果
4. 重复上述步骤，直到得出最终答案

```python
def react_loop(task, tools, llm, max_steps=10):
    """ReAct 核心循环"""
    scratchpad = f"Task: {task}\n"

    for step in range(max_steps):
        # 1. Reasoning：让 LLM 思考下一步
        prompt = f"""{scratchpad}
        你有以下工具可用：{tools.describe()}
        
        请用以下格式思考和行动：
        Thought: <你的思考>
        Action: <工具名称>
        Action Input: <工具参数>
        
        如果任务已完成：
        Thought: <总结>
        Final Answer: <最终答案>
        """
        response = llm.chat(prompt)

        # 2. 解析 LLM 输出
        thought, action, action_input = parse_react_output(response)

        # 3. 判断是否结束
        if action == "Final Answer":
            return action_input

        # 4. Acting：执行工具
        observation = tools.execute(action, action_input)

        # 5. 记录到 scratchpad
        scratchpad += f"Thought: {thought}\n"
        scratchpad += f"Action: {action}\n"
        scratchpad += f"Action Input: {action_input}\n"
        scratchpad += f"Observation: {observation}\n"

    return "达到最大步骤数限制"
```

一个完整的 ReAct 执行示例：

```
Task: 帮我查一下特斯拉最近一个季度的营收和同比增长

Thought: 我需要先查特斯拉最近的财报数据
Action: search
Action Input: "特斯拉 2025 Q4 营收"
Observation: 特斯拉2025年Q4营收为251.7亿美元...

Thought: 我有了Q4数据，现在需要查去年同期数据来算同比增长
Action: search
Action Input: "特斯拉 2024 Q4 营收"
Observation: 特斯拉2024年Q4营收为251.67亿美元...

Thought: 现在我有了两个数据，可以计算并给出答案
Final Answer: 特斯拉2025年Q4营收为251.7亿美元，同比增长约0.01%。
```

**追问：**
- ReAct 和 Chain-of-Thought 有什么区别？CoT 只有"思考"没有"行动"，ReAct 加入了工具调用能力。
- ReAct 中的 Thought 有什么作用？它让模型的推理过程显式化，便于调试和理解，也能引导模型做出更好的决策。

---

### Q: ReAct 的伪代码是什么？每一步做什么？⭐⭐

**答：**

ReAct 的完整伪代码如下，每一步都有明确的职责：

```python
def react_agent(question, tools, llm):
    """
    ReAct Agent 完整实现
    
    Args:
        question: 用户问题
        tools: 可用工具集合
        llm: 大语言模型
    
    Returns:
        最终答案
    """
    # 初始化：构建初始 prompt，包含系统指令和问题
    trajectory = []  # 轨迹记录
    max_iterations = 5
    
    # 构建工具描述
    tool_descriptions = "\n".join([
        f"- {t.name}: {t.description}" for t in tools
    ])
    
    for i in range(max_iterations):
        # === STEP 1: 构建 Prompt ===
        # 把所有历史轨迹拼接到 prompt 中
        history_text = format_trajectory(trajectory)
        prompt = f"""Answer the following question step by step.

Available tools:
{tool_descriptions}

Question: {question}

{history_text}
Thought:"""
        
        # === STEP 2: LLM 生成 Thought ===
        thought = llm.generate(prompt, stop=["\nAction:"])
        trajectory.append({"type": "thought", "content": thought})
        
        # === STEP 3: LLM 生成 Action ===
        action_prompt = f"{prompt}\nThought: {thought}\nAction:"
        action_line = llm.generate(action_prompt, stop=["\nAction Input:"])
        tool_name = parse_tool_name(action_line)
        
        # === STEP 4: LLM 生成 Action Input ===
        input_prompt = f"{action_prompt}{action_line}\nAction Input:"
        tool_input = llm.generate(input_prompt, stop=["\n"])
        
        trajectory.append({
            "type": "action",
            "tool": tool_name,
            "input": tool_input
        })
        
        # === STEP 5: 执行工具，获取 Observation ===
        try:
            observation = execute_tool(tool_name, tool_input)
        except Exception as e:
            observation = f"Error: {str(e)}"
        
        trajectory.append({"type": "observation", "content": observation})
        
        # === STEP 6: 判断是否结束 ===
        if should_finish(thought, observation):
            break
    
    # === STEP 7: 生成最终答案 ===
    final_prompt = build_final_prompt(trajectory)
    return llm.generate(final_prompt)
```

每一步的职责：

| 步骤 | 职责 | 关键点 |
|------|------|--------|
| 构建 Prompt | 把历史轨迹和工具描述组装成上下文 | 上下文管理是关键 |
| 生成 Thought | 模型分析当前状况 | 这是"推理"的核心 |
| 生成 Action | 模型选择要调用的工具 | 需要输出工具名称 |
| 生成 Action Input | 模型生成工具调用参数 | 参数格式要正确 |
| 执行工具 | 真正运行工具获取结果 | 需要错误处理 |
| 判断结束 | 检查是否可以得出答案 | 可以是 LLM 判断，也可以是规则判断 |
| 生成最终答案 | 综合所有信息给出答案 | 可选步骤 |

**追问：**
- 为什么 Thought、Action、Action Input 分三次生成而不是一次？早期实现确实是一次生成全部，但分开生成可以更好地控制输出格式，也便于在每一步做校验。
- 这和 OpenAI 的 Function Calling 有什么区别？Function Calling 是模型原生支持的结构化输出，更可靠；ReAct 是通过 prompt engineering 实现的文本解析，更灵活但更脆弱。

---

### Q: ReAct 的局限性？什么场景下 ReAct 不够用？⭐⭐⭐

**答：**

ReAct 虽然经典，但有四个明显的局限性：

**1. 串行执行，效率低**

ReAct 每次只能调用一个工具，当多个子任务相互独立时，无法并行。比如"同时查3个城市的天气"，ReAct 需要循环3次。

**2. 无全局规划，容易走弯路**

ReAct 是"走一步看一步"的贪心策略，缺乏全局视角。复杂任务（如"帮我规划一次旅行"）需要先制定整体计划，ReAct 可能中途发现方向错误，浪费大量 token。

**3. 上下文窗口压力大**

每轮循环都会增加历史记录长度，经过多轮后，上下文可能超出模型的窗口限制，导致信息丢失或性能下降。

**4. 错误累积，难以回溯**

如果某一步的 Thought 或 Action 出错，后续步骤会基于错误的前提继续推理，而且 ReAct 没有内置的"回退"机制。

```python
# ReAct 的局限性示例

# 局限1：串行执行
Task: "对比特斯拉和比亚迪的Q4营收"
# ReAct 需要 4 步：查特斯拉 → 记录 → 查比亚迪 → 对比
# 而理想情况是并行查询，2 步完成

# 局限2：无全局规划
Task: "帮我分析这个代码仓库的安全漏洞"
# ReAct: 先扫描A文件 → 发现漏洞1 → 扫描B文件 → ...
# 理想：先制定扫描策略（静态分析→依赖检查→权限审计），再分步执行

# 局限3：上下文膨胀
# 10轮循环后，scratchpad 可能已经 5000+ tokens
# 模型可能忘记前面的关键信息
```

**什么场景需要更高级的架构？**

| 场景 | 推荐架构 | 原因 |
|------|----------|------|
| 需要全局规划 | Plan-and-Execute | 先规划再执行，避免走弯路 |
| 子任务可并行 | DAG/Graph-based | 提升执行效率 |
| 需要自我纠错 | Reflexion | 加入反思和重试机制 |
| 长任务 | Multi-Agent | 分工协作，降低单个Agent压力 |

**追问：**
- 有没有混合方案？有，比如 Plan-and-Execute 中，执行每个子步骤时内部使用 ReAct 循环。
- ReAct 在什么场景下仍然最好用？简单到中等复杂度的任务，步骤不超过5步，依赖链明确的场景。

---

### Q: 如何控制 ReAct 的循环次数？防止无限循环？⭐⭐

**答：**

防止无限循环是 Agent 开发中必须处理的问题。实际开发中，我通常用**三层防护**：

**第一层：硬性步数限制**

最基础的保护，设置最大循环次数。

```python
class SafeReActAgent:
    def __init__(self, max_steps=10, max_tokens=4000):
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.step_count = 0
        self.total_tokens = 0

    def run(self, task):
        for step in range(self.max_steps):
            result = self.one_step(task)
            self.step_count += 1

            # 检查是否完成
            if result.is_final:
                return result.answer

            # 检查 token 预算
            self.total_tokens += result.tokens_used
            if self.total_tokens > self.max_tokens:
                return self.force_finish(task)

        # 达到最大步数，强制总结
        return self.force_finish(task)
```

**第二层：重复检测**

防止 Agent 在相同的工具调用之间反复跳转（"转圈"）。

```python
class RepeatedActionDetector:
    def __init__(self, window_size=3):
        self.history = []
        self.window_size = window_size

    def check_loop(self, action, action_input):
        """检测是否陷入了循环"""
        self.history.append((action, action_input))

        # 检测完全重复
        if len(self.history) >= self.window_size:
            recent = self.history[-self.window_size:]
            if len(set(recent)) == 1:
                return True  # 连续 N 次完全相同的操作

        # 检测振荡（ABAB模式）
        if len(self.history) >= 4:
            if (self.history[-1] == self.history[-3] and
                self.history[-2] == self.history[-4]):
                return True  # ABAB 振荡

        return False

    def suggest_escape(self, task):
        """当检测到循环时，建议换一种方式"""
        return f"你已经尝试了相同的方法多次，请换一种思路解决：{task}"
```

**第三层：LLM 自主判断**

让模型自己判断是否在浪费时间，主动改变策略。

```python
def should_continue_prompt(task, history):
    """让 LLM 自己判断是否应该继续"""
    return f"""你正在执行任务：{task}

到目前为止的步骤：
{format_history(history)}

请判断：
1. 继续当前方向是否还有意义？
2. 是否需要换一种方法？
3. 是否应该直接给出当前最佳答案？

回答格式：
- CONTINUE: 如果应该继续
- CHANGE_STRATEGY: 如果需要换方法，附上新策略
- FINISH: 如果应该结束，附上当前最佳答案"""
```

**追问：**
- 实际生产中，步数限制通常设多少？简单任务 5-8 步，复杂任务 10-15 步，超过 20 步通常说明任务需要被拆解。
- 除了步数限制，还有什么成本控制手段？Token 预算限制、总耗时限制、API 调用次数限制。

---

## 三、Plan-and-Execute

### Q: Plan-and-Execute 和 ReAct 有什么区别？⭐⭐

**答：**

Plan-and-Execute（规划-执行）和 ReAct 的核心区别在于**有没有"全局规划"阶段**。可以类比为：

- **ReAct**：边走边看的"即兴旅行"——走到哪算哪
- **Plan-and-Execute**：先看地图规划路线，再按计划出行的"计划旅行"

```python
# ReAct：每一步都重新思考
# 问题 → 思考 → 行动 → 观察 → 思考 → 行动 → 观察 → ... → 答案

# Plan-and-Execute：先规划，再执行
# 问题 → 制定计划 [步骤1, 步骤2, 步骤3, ...] → 逐步执行 → 答案
```

具体架构对比：

```python
# ========== ReAct ==========
def react(task):
    context = task
    for _ in range(MAX_STEPS):
        # 每一步都从全局视角思考
        action = llm.chat(f"当前任务：{task}\n历史：{context}\n下一步？")
        result = execute(action)
        context += result
    return context

# ========== Plan-and-Execute ==========
def plan_and_execute(task):
    # 阶段1：规划（Planner）
    plan = llm.chat(f"为以下任务制定详细执行计划：{task}")
    # plan = ["步骤1: 查XXX", "步骤2: 分析XXX", "步骤3: 总结XXX"]

    # 阶段2：逐步执行（Executor）
    results = []
    for step in plan:
        step_result = llm.chat(f"执行：{step}\n已有结果：{results}")
        results.append(step_result)

    # 阶段3：汇总
    return llm.chat(f"根据以下结果回答问题{task}：{results}")
```

| 对比维度 | ReAct | Plan-and-Execute |
|----------|-------|------------------|
| 规划方式 | 无显式规划，每步即时决策 | 先制定全局计划 |
| 适用场景 | 简单、步骤少的任务 | 复杂、步骤多的任务 |
| 效率 | 可能走弯路 | 整体路径更优 |
| 灵活性 | 高，随时可调整 | 中，需要重规划机制 |
| 实现复杂度 | 低 | 高 |

**追问：**
- Plan-and-Execute 中，谁来做规划，谁来执行？通常用一个能力更强的模型做规划（Planner），用同一个或更轻量的模型做执行（Executor）。
- 计划执行到一半发现原计划不可行怎么办？需要"动态重规划"机制，这正是下一个问题。

---

### Q: 什么时候用 Plan-Execute 而不是 ReAct？⭐⭐

**答：**

选择 ReAct 还是 Plan-and-Execute，取决于任务的**复杂度、步骤数和依赖关系**。我总结了一个简单的决策框架：

**用 ReAct 的场景：**
- 任务步骤 ≤ 3 步
- 任务路径不确定，需要根据中间结果灵活调整
- 实时性要求高（ReAct 可以更快返回第一个有用的结果）

**用 Plan-and-Execute 的场景：**
- 任务步骤 ≥ 5 步
- 子任务之间有依赖关系，需要全局优化
- 需要让用户先看到执行计划，获得确认后再执行
- 资源有限，需要提前评估成本

```python
# 用 ReAct 的场景示例
"今天北京天气怎么样？"          # 1步：查天气API
"这篇文章有多少个错别字？"      # 2步：读取→检查
"把这段话翻译成英文"            # 1步：直接翻译

# 用 Plan-and-Execute 的场景示例
"帮我写一份竞品分析报告"
# 计划：1.确定竞品列表 2.收集各竞品信息 3.对比分析 4.生成报告

"帮我把这个Python项目从Flask迁移到FastAPI"
# 计划：1.分析现有路由 2.创建FastAPI项目 3.逐个迁移路由 4.迁移中间件 5.测试验证
```

```python
class TaskRouter:
    """根据任务特征选择架构"""

    def route(self, task: str) -> str:
        complexity = self.estimate_complexity(task)

        if complexity <= 2:
            return "react"         # 简单任务用 ReAct
        elif complexity <= 4:
            return "react_with_reflection"  # 中等任务加反思
        else:
            return "plan_and_execute"       # 复杂任务用规划

    def estimate_complexity(self, task: str) -> int:
        """估算任务复杂度（1-5）"""
        # 可以用 LLM 来判断，也可以用规则
        signals = {
            "多个": 1, "对比": 1, "分析": 1,
            "报告": 2, "迁移": 2, "重构": 2,
            "首先然后最后": 1, "步骤": 1,
        }
        score = sum(v for k, v in signals.items() if k in task)
        return min(score, 5)
```

**追问：**
- 能不能两种架构混合使用？可以。Plan-and-Execute 的每个"执行"步骤内部，可以用 ReAct 来完成。
- 有没有自适应的架构？有，比如 AutoGPT 类的 Agent 会在运行时动态决定是否需要重新规划。

---

### Q: 如何实现动态重规划？计划失败了怎么办？⭐⭐⭐

**答：**

动态重规划是 Plan-and-Execute 架构中最重要的机制。核心思想是：**计划不是一成不变的，执行过程中如果遇到意外，要能调整计划。**

类比一下：你计划了周末出游路线，但到了第一个景点发现关门了——你不会直接回家，而是拿出手机重新规划剩余路线。

```python
class DynamicPlanAndExecute:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def run(self, task):
        # 1. 初始规划
        plan = self.create_plan(task)
        results = {}
        completed_steps = []

        while plan:
            current_step = plan.pop(0)

            # 2. 执行当前步骤
            try:
                result = self.execute_step(current_step, results)
                results[current_step.id] = result
                completed_steps.append(current_step)

            except StepFailedError as e:
                # 3. 步骤失败 → 判断是否需要重规划
                replan_decision = self.should_replan(
                    task=task,
                    failed_step=current_step,
                    error=e,
                    completed=completed_steps,
                    remaining=plan
                )

                if replan_decision.should_replan:
                    # 4. 动态重规划
                    plan = self.replan(
                        original_task=task,
                        completed=completed_steps,
                        failed_step=current_step,
                        remaining=plan
                    )
                else:
                    # 5. 尝试补救（重试/跳过/替代方案）
                    result = self.recover(current_step, e)
                    results[current_step.id] = result

        # 6. 汇总结果
        return self.summarize(task, results)

    def replan(self, original_task, completed, failed_step, remaining):
        """动态重规划"""
        prompt = f"""原始任务：{original_task}

已完成的步骤：
{format_steps(completed)}

失败的步骤：{failed_step}，错误：{failed_step.error}

原定剩余步骤：
{format_steps(remaining)}

请根据实际情况，制定新的执行计划。
已成功的步骤不需要重复，新计划应该从下一步开始。
输出格式：JSON 数组，每个元素包含 id, description, dependencies"""

        new_plan = self.llm.chat(prompt)
        return parse_plan(new_plan)

    def should_replan(self, task, failed_step, error, completed, remaining):
        """判断是否需要重规划 vs 简单重试"""
        prompt = f"""步骤 "{failed_step.description}" 执行失败，错误：{error}

这是一个需要简单重试的临时错误，还是需要重新规划的系统性错误？

回答 RETRY 或 REPLAN，并说明理由。"""

        decision = self.llm.chat(prompt)
        return ReplanDecision(
            should_replan="REPLAN" in decision,
            reason=decision
        )
```

**重规划的三种策略：**

| 策略 | 适用场景 | 示例 |
|------|----------|------|
| **局部调整** | 当前步骤失败，但不影响整体计划 | API 超时，换一个 API |
| **跳过步骤** | 当前步骤非关键，可以跳过 | 某个数据源不可用，用已有数据继续 |
| **全面重规划** | 整体方向需要调整 | 发现用户需求理解错误 |

**追问：**
- 重规划会不会导致无限重规划？会，所以需要设置重规划次数上限（通常 2-3 次）。
- 如何避免重规划浪费已完成的工作？关键是把"已完成步骤"作为上下文传给 Planner，让它在此基础上规划。

---

## 四、工具调用 / Function Calling

### Q: Function Calling 的实现原理是什么？⭐⭐

**答：**

Function Calling 是让 LLM 能够**结构化地调用外部工具**的机制。它的原理并不是模型真的在"调用函数"，而是模型**生成了一个结构化的工具调用描述**，由应用层代码来执行实际调用。

类比一下：模型就像一个指挥官，它说"我要用望远镜看东边"，但真正拿起望远镜的是旁边的士兵（应用层代码）。

```python
# Function Calling 的完整流程

# 1. 定义工具 schema（告诉模型有哪些工具）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如'北京'"
                    },
                    "date": {
                        "type": "string",
                        "description": "日期，格式 YYYY-MM-DD，默认今天"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 2. 调用模型（模型决定是否需要调用工具）
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
    tool_choice="auto"  # auto / none / required
)

# 3. 模型返回工具调用请求（不是直接执行）
# response.choices[0].message.tool_calls = [
#     {
#         "id": "call_abc123",
#         "function": {
#             "name": "get_weather",
#             "arguments": '{"city": "北京", "date": "2025-01-15"}'
#         }
#     }
# ]

# 4. 应用层执行实际调用
tool_call = response.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)
result = get_weather(**args)  # 真正的函数调用

# 5. 把结果返回给模型
response2 = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "北京今天天气怎么样？"},
        response.choices[0].message,  # 包含 tool_calls 的 assistant 消息
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result, ensure_ascii=False)
        }
    ]
)
```

**底层原理：**

模型在训练时，学会了当遇到工具 schema 时，在应该调用工具的地方输出一个特殊的 token 序列（如 `<tool_call>`），包含工具名和参数的 JSON。这本质上是一种**受控文本生成**，只不过输出格式被约束为结构化的 JSON。

**追问：**
- Function Calling 和 ReAct 的文本解析有什么区别？Function Calling 是模型原生支持的，输出格式更可靠；ReAct 的文本解析依赖正则表达式，容易因为模型输出格式变化而失败。
- 为什么有些模型不支持 Function Calling？需要专门的微调训练，小模型或开源模型可能没有经过这种训练。可以通过 prompt engineering 模拟，但效果不如原生支持。

---

### Q: 如何设计好的工具描述？让模型准确选择工具？⭐⭐

**答：**

工具描述的质量**直接决定** Agent 的效果。我总结了五个设计原则，按重要性排序：

**原则一：描述要"像给人看的文档"**

```python
# ❌ 差的描述
{
    "name": "search_db",
    "description": "搜索数据库",
    "parameters": {
        "query": {"type": "string"}
    }
}

# ✅ 好的描述
{
    "name": "search_product_database",
    "description": "在产品数据库中搜索商品信息。支持按商品名称、类别、价格范围搜索。返回商品的名称、价格、库存数量和描述。适用于需要查找商品详情或库存状态的场景。",
    "parameters": {
        "query": {
            "type": "string",
            "description": "搜索关键词，如'蓝牙耳机'、'运动鞋'"
        },
        "category": {
            "type": "string",
            "enum": ["electronics", "clothing", "food", "other"],
            "description": "商品类别筛选，可选"
        },
        "price_range": {
            "type": "object",
            "properties": {
                "min": {"type": "number", "description": "最低价格"},
                "max": {"type": "number", "description": "最高价格"}
            },
            "description": "价格范围筛选，可选"
        }
    },
    "required": ["query"]
}
```

**原则二：明确边界——说明"不能做什么"**

```python
# 加上边界说明
"description": """搜索产品数据库。
能力：按名称、类别、价格搜索在售商品。
限制：
- 不能查询已下架商品
- 不能执行修改/删除操作
- 不包含用户订单信息（请用 search_orders 工具）
返回：商品名称、价格、库存、描述的JSON"""
```

**原则三：用枚举约束参数值**

```python
# 让模型从有限选项中选择，减少自由文本的错误
"parameters": {
    "time_range": {
        "type": "string",
        "enum": ["today", "this_week", "this_month", "this_year"],
        "description": "时间范围"
    }
}
```

**原则四：提供参数示例**

```python
"query": {
    "type": "string",
    "description": "SQL查询语句。示例：SELECT * FROM users WHERE age > 18 LIMIT 10"
}
```

**原则五：工具数量控制在合理范围**

```python
# 工具太多时，用"路由"模式
class ToolRouter:
    def __init__(self, all_tools):
        self.all_tools = all_tools

    def get_relevant_tools(self, task, max_tools=5):
        """根据任务动态选择相关工具"""
        tool_names = self.llm.chat(f"""
            从以下工具中选择与任务最相关的{max_tools}个：
            任务：{task}
            工具列表：{[t.name for t in self.all_tools]}
            返回工具名称列表，JSON格式
        """)
        return [t for t in self.all_tools if t.name in tool_names]
```

**追问：**
- 工具数量多了会怎样？模型容易"挑花眼"，选错工具。一般建议不超过 10-15 个工具同时暴露。
- 中文描述和英文描述哪个效果好？英文描述通常效果更好（模型训练数据以英文为主），但如果用户输入是中文，工具描述也用中文更一致。

---

### Q: 工具调用失败了怎么处理？有哪些重试策略？⭐⭐

**答：**

工具调用失败是 Agent 开发中最常见的问题。我通常用**分类处理**策略：不同类型的错误，用不同的方式应对。

```python
class ToolExecutor:
    def __init__(self, max_retries=3):
        self.max_retries = max_retries

    def execute_with_retry(self, tool_name, tool_input):
        """带重试和错误恢复的工具执行"""

        for attempt in range(self.max_retries):
            try:
                result = self.tools[tool_name](**tool_input)
                return {"status": "success", "result": result}

            except TimeoutError:
                # 类型1：临时错误 — 重试
                wait_time = 2 ** attempt  # 指数退避
                time.sleep(wait_time)
                continue

            except ValidationError as e:
                # 类型2：参数错误 — 让 LLM 修正参数
                corrected_input = self.fix_parameters(
                    tool_name, tool_input, str(e)
                )
                tool_input = corrected_input
                continue

            except PermissionError:
                # 类型3：权限错误 — 不重试，直接报告
                return {
                    "status": "error",
                    "message": f"没有权限调用 {tool_name}"
                }

            except Exception as e:
                # 类型4：未知错误 — 记录并让 LLM 决定
                return {
                    "status": "error",
                    "message": str(e),
                    "suggest_retry": attempt < self.max_retries - 1
                }

        return {"status": "error", "message": "达到最大重试次数"}

    def fix_parameters(self, tool_name, original_input, error_msg):
        """让 LLM 根据错误信息修正参数"""
        prompt = f"""调用工具 {tool_name} 失败。

原始参数：{json.dumps(original_input)}
错误信息：{error_msg}

请修正参数，返回修正后的JSON。"""
        corrected = self.llm.chat(prompt)
        return json.loads(corrected)
```

**五种重试策略：**

| 策略 | 适用场景 | 实现方式 |
|------|----------|----------|
| **简单重试** | 网络超时、服务临时不可用 | 等待后重试 |
| **指数退避** | API 限流（429错误） | 每次等待时间翻倍 |
| **参数修正** | 参数格式错误 | 让 LLM 修正参数 |
| **工具降级** | 主工具不可用 | 切换到备选工具 |
| **跳过** | 非关键步骤失败 | 跳过并继续 |

**追问：**
- 如何区分"值得重试"和"不该重试"的错误？HTTP 状态码是好指标：408/429/500/502/503 值得重试，400/401/403/404 不该重试。
- 如何避免重试风暴？设置全局重试预算（比如总共最多重试 10 次），使用指数退避 + 随机抖动。

---

### Q: 如何实现工具的并行调用？⭐⭐⭐

**答：**

当 Agent 需要调用多个**相互独立**的工具时，并行执行可以大幅提升效率。实现方式取决于你用的框架和模型 API。

**方式一：模型原生并行调用（推荐）**

OpenAI 等 API 支持模型在一次响应中返回多个 tool_calls：

```python
# 模型可能返回多个并行的工具调用
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": "对比北京、上海、深圳今天的天气"
    }],
    tools=weather_tools
)

# response.choices[0].message.tool_calls 可能包含 3 个调用：
# [
#   {"function": {"name": "get_weather", "arguments": '{"city":"北京"}'}},
#   {"function": {"name": "get_weather", "arguments": '{"city":"上海"}'}},
#   {"function": {"name": "get_weather", "arguments": '{"city":"深圳"}'}}
# ]
```

**方式二：应用层并行执行**

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelToolExecutor:
    def __init__(self, tools, max_workers=5):
        self.tools = tools
        self.max_workers = max_workers

    async def execute_parallel(self, tool_calls):
        """并行执行多个工具调用"""
        tasks = []
        for call in tool_calls:
            task = asyncio.create_task(
                self.execute_single(call)
            )
            tasks.append(task)

        # 等待所有任务完成，设置超时
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 组装结果
        tool_results = []
        for call, result in zip(tool_calls, results):
            if isinstance(result, Exception):
                tool_results.append({
                    "tool_call_id": call["id"],
                    "content": f"Error: {str(result)}"
                })
            else:
                tool_results.append({
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, ensure_ascii=False)
                })

        return tool_results

    async def execute_single(self, tool_call):
        """执行单个工具调用"""
        func_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
        return await asyncio.to_thread(
            self.tools[func_name], **args
        )
```

**方式三：DAG 依赖分析 + 并行**

更高级的方式，自动分析哪些步骤可以并行：

```python
class DAGScheduler:
    """基于依赖关系的并行调度器"""

    def __init__(self):
        self.dependency_graph = {}

    def build_from_plan(self, plan_steps):
        """从计划中构建依赖图"""
        for step in plan_steps:
            self.dependency_graph[step.id] = {
                "step": step,
                "dependencies": step.depends_on,
                "status": "pending"
            }

    def get_ready_steps(self):
        """获取可以并行执行的步骤（所有依赖已完成）"""
        ready = []
        for step_id, info in self.dependency_graph.items():
            if info["status"] != "pending":
                continue
            deps_done = all(
                self.dependency_graph[d]["status"] == "done"
                for d in info["dependencies"]
            )
            if deps_done:
                ready.append(info["step"])
        return ready

    async def execute(self):
        """执行整个 DAG"""
        while True:
            ready = self.get_ready_steps()
            if not ready:
                break  # 全部完成

            # 并行执行所有 ready 的步骤
            results = await asyncio.gather(*[
                self.execute_step(step) for step in ready
            ])

            for step, result in zip(ready, results):
                self.dependency_graph[step.id]["status"] = "done"
                self.dependency_graph[step.id]["result"] = result
```

**追问：**
- 并行调用会不会有并发问题？如果工具是只读的（查询），没问题。如果有写操作，需要注意并发安全。
- 如何决定什么时候并行、什么时候串行？如果后一步依赖前一步的结果，必须串行。可以用 DAG 自动分析依赖关系。

---

### Q: 如何限制 Agent 的工具使用权限？⭐⭐

**答：**

安全是 Agent 开发中最重要的问题之一。一个不受限制的 Agent 可能删除数据库、发送垃圾邮件、泄露敏感信息。我通常用**四层权限控制**：

```python
class ToolPermissionManager:
    def __init__(self):
        self.permissions = {}      # 工具权限配置
        self.rate_limits = {}      # 频率限制
        self.approval_required = set()  # 需要人工审批的工具
        self.audit_log = []        # 审计日志

    # 第一层：工具白名单
    def get_allowed_tools(self, user_role):
        """根据用户角色返回可用工具"""
        role_permissions = {
            "viewer": ["search", "read_file", "get_weather"],
            "editor": ["search", "read_file", "write_file", "get_weather"],
            "admin": ["*"]  # 所有工具
        }
        allowed = role_permissions.get(user_role, [])
        if "*" in allowed:
            return self.all_tools
        return [t for t in self.all_tools if t.name in allowed]

    # 第二层：参数校验（防止注入）
    def validate_tool_call(self, tool_name, params):
        """校验工具调用参数"""
        # 1. SQL 注入防护
        if tool_name == "execute_sql":
            if any(keyword in params["query"].upper()
                   for keyword in ["DROP", "DELETE", "TRUNCATE", "ALTER"]):
                raise PermissionError("不允许执行破坏性SQL")

        # 2. 路径遍历防护
        if tool_name == "read_file":
            if ".." in params["path"] or params["path"].startswith("/etc"):
                raise PermissionError("不允许访问该路径")

        # 3. URL 白名单
        if tool_name == "http_request":
            allowed_domains = ["api.example.com", "openai.com"]
            domain = urlparse(params["url"]).netloc
            if domain not in allowed_domains:
                raise PermissionError(f"不允许访问 {domain}")

    # 第三层：人工审批（Human-in-the-loop）
    def needs_approval(self, tool_name, params):
        """判断是否需要人工审批"""
        high_risk_tools = {"delete_file", "send_email", "execute_sql", "make_payment"}
        if tool_name in high_risk_tools:
            return True
        if self.is_destructive_action(tool_name, params):
            return True
        return False

    # 第四层：频率限制
    def check_rate_limit(self, tool_name):
        """检查是否超出频率限制"""
        now = time.time()
        window = 60  # 1 分钟窗口
        limit = self.rate_limits.get(tool_name, 10)

        recent_calls = [
            t for t in self.audit_log
            if t["tool"] == tool_name and now - t["time"] < window
        ]

        if len(recent_calls) >= limit:
            raise RateLimitError(f"{tool_name} 超出频率限制 ({limit}/min)")

    # 审计日志
    def log_execution(self, tool_name, params, result, user):
        self.audit_log.append({
            "time": time.time(),
            "tool": tool_name,
            "params": params,
            "result_summary": str(result)[:200],
            "user": user
        })
```

**追问：**
- Human-in-the-loop 在 Agent 中怎么实现？在执行高风险操作前，暂停 Agent，把操作描述发给用户确认，用户确认后才执行。
- 有没有渐进式授权？有，类似 Android 的权限模型：先给基础权限，Agent 需要更多权限时再向用户申请。

---

## 五、错误恢复与鲁棒性

### Q: Agent 执行过程中遇到错误怎么处理？⭐⭐

**答：**

Agent 执行中的错误处理需要分层设计，就像一个"洋葱"——从内到外，每一层处理不同的错误类型：

```python
class RobustAgent:
    def run(self, task):
        try:
            # 最外层：全局异常捕获
            return self._run_with_recovery(task)
        except BudgetExceededError:
            return self.graceful_degradation(task, "预算超限")
        except ModelUnavailableError:
            return self.fallback_response(task)

    def _run_with_recovery(self, task):
        for step in range(self.max_steps):
            try:
                # 中间层：步骤级错误处理
                result = self.execute_step(task)
                if result.is_final:
                    return result.answer
            except ToolExecutionError as e:
                # 工具执行失败 → 让 LLM 决定如何处理
                recovery = self.handle_tool_error(e, task)
                if recovery.should_skip:
                    continue
                elif recovery.should_retry:
                    result = self.retry_step(task, max_attempts=2)
            except ContextLengthError:
                # 上下文过长 → 压缩历史
                self.compress_history()
            except RateLimitError as e:
                # API 限流 → 等待后重试
                time.sleep(e.retry_after)

        return "任务执行超时，请尝试简化问题。"

    def handle_tool_error(self, error, task):
        """让 LLM 参与错误恢复决策"""
        prompt = f"""在执行任务 "{task}" 时遇到错误：

工具：{error.tool_name}
参数：{error.params}
错误：{error.message}

请判断：
1. SKIP - 跳过此步骤继续
2. RETRY - 重试（可能是临时错误）
3. ALTERNATIVE - 用其他方式完成

回答格式：DECISION: <SKIP/RETRY/ALTERNATIVE>
理由：<简要说明>"""

        response = self.llm.chat(prompt)
        return parse_recovery_decision(response)
```

**追问：**
- 要不要把错误信息完全暴露给 LLM？不建议，错误信息要精简。原始错误堆栈可能有几百行，应该提取关键信息（错误类型、原因、建议修复方式）。
- 如何避免错误处理本身消耗过多 token？设置错误处理的 token 预算，超限后用简单的规则处理（如默认跳过）。

---

### Q: 如何实现 Agent 的自我纠错？⭐⭐⭐

**答：**

自我纠错是高级 Agent 的核心能力。基本思路是：**执行 → 检查 → 如果错误 → 分析原因 → 修正 → 重新执行**。

```python
class SelfCorrectingAgent:
    def run_with_correction(self, task, max_corrections=3):
        """带自我纠错的 Agent 执行"""

        for attempt in range(max_corrections + 1):
            # 1. 执行任务
            result = self.execute(task)

            # 2. 自我检查
            check_result = self.verify_result(task, result)

            if check_result.is_valid:
                return result  # 通过检查，返回结果

            # 3. 分析错误原因
            error_analysis = self.analyze_error(
                task=task,
                result=result,
                error=check_result.error
            )

            # 4. 修正策略
            task = self.refine_task(
                original_task=task,
                previous_result=result,
                error_analysis=error_analysis
            )

        return result  # 达到最大纠错次数

    def verify_result(self, task, result):
        """验证结果是否正确"""
        prompt = f"""请检查以下回答是否正确和完整：

问题：{task}
回答：{result}

检查项：
1. 事实是否正确？
2. 逻辑是否自洽？
3. 是否完整回答了问题？
4. 是否有遗漏？

回答 VALID 或 INVALID，并说明原因。"""
        check = self.llm.chat(prompt)
        return VerificationResult(
            is_valid="VALID" in check,
            error=check
        )

    def refine_task(self, original_task, previous_result, error_analysis):
        """根据错误分析，生成修正后的任务描述"""
        return f"""原始任务：{original_task}
上次执行结果：{previous_result}
发现的问题：{error_analysis}
请修正以上问题，重新执行任务。"""
```

**更高级的方案：使用 Code Interpreter 验证**

```python
class CodeVerifyingAgent:
    """用代码执行来验证结果"""

    def verify_with_code(self, task, result):
        """生成验证代码并执行"""
        verification_code = self.llm.chat(f"""
任务：{task}
回答：{result}

请写一段 Python 代码来验证这个回答是否正确。
例如：如果是数学计算，写代码重新算一遍；
如果是数据查询，写代码验证数据一致性。

只返回 Python 代码。""")

        try:
            exec_result = execute_code_safely(verification_code)
            return exec_result  # True/False + 详情
        except Exception as e:
            return {"verified": False, "reason": f"验证代码执行失败: {e}"}
```

**追问：**
- 自我纠错和 Reflexion 有什么区别？自我纠错是同一轮内的即时修正；Reflexion 是跨轮次的长期学习。
- 如何避免"纠错循环"——模型反复犯同样的错？记录已经尝试过的方案，避免重复；设置最大纠错次数。

---

### Q: 什么是 Reflexion？让 Agent 从失败中学习？⭐⭐⭐

**答：**

Reflexion 是一种让 Agent **通过"反思"来改进**的架构。核心思想是：Agent 执行完任务后，如果失败了，它会**生成一段反思（Reflection）**，总结失败原因和改进方向，然后在下次尝试时参考这些反思。

类比：就像学生做题——做错了，老师讲解后写下"错题笔记"，下次遇到类似题目时翻开笔记参考。

```python
class ReflexionAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.reflections = []  # 存储反思笔记

    def run(self, task, max_trials=3):
        """带 Reflexion 的执行循环"""

        for trial in range(max_trials):
            # 1. 执行任务（带上历史反思）
            result = self.execute(task)

            # 2. 检查是否成功
            if self.evaluate(task, result):
                return result

            # 3. 反思：为什么失败了？
            reflection = self.reflect(task, result)

            # 4. 存储反思，供下次使用
            self.reflections.append({
                "trial": trial,
                "task": task,
                "result": result,
                "reflection": reflection
            })

        return result

    def execute(self, task):
        """执行任务，参考历史反思"""
        reflection_context = ""
        if self.reflections:
            reflection_context = "\n\n## 历史反思\n"
            for r in self.reflections:
                reflection_context += f"- 第{r['trial']+1}次尝试失败原因：{r['reflection']}\n"
                reflection_context += f"  改进建议：{r['improvement']}\n"

        prompt = f"""任务：{task}
{reflection_context}

请注意避免之前犯过的错误。执行任务："""

        return react_loop(prompt, self.tools, self.llm)

    def reflect(self, task, result):
        """生成反思"""
        prompt = f"""你刚刚尝试完成以下任务，但没有成功。

任务：{task}
你的回答：{result}

请反思：
1. 为什么会失败？（具体原因）
2. 你遗漏了什么信息？
3. 下次应该怎么做？
4. 有哪些错误应该避免？

用简洁的语言总结（3-5句话）。"""

        reflection = self.llm.chat(prompt)

        # 提取改进建议
        improvement = self.llm.chat(f"""基于以下反思，给出具体的改进建议：
{reflection}

用一句话总结下次应该怎么做。""")

        return reflection, improvement
```

Reflexion 的完整流程图：

```
Trial 1: 执行 → 失败 → 反思("我应该先查数据库，而不是直接猜")
                        ↓
Trial 2: 执行（参考反思）→ 失败 → 反思("查数据库时应该用模糊匹配")
                                ↓
Trial 3: 执行（参考所有反思）→ 成功 ✓
```

**追问：**
- Reflexion 的反思存储在哪里？短期可以放在 prompt 中，长期应该存入向量数据库，类似经验库。
- Reflexion 适合什么场景？编程竞赛、问答系统、复杂推理任务。不太适合实时性要求高的场景（因为要多次尝试）。

---

### Q: 如何处理模型输出格式错误？⭐⭐

**答：**

模型输出格式错误是 Agent 开发中最"日常"的头疼问题。模型可能输出不合法的 JSON、格式不匹配的文本、多余的解释性文字等。我总结了一套**四步防护体系**：

```python
class OutputParser:
    """健壮的输出解析器"""

    def parse_with_retry(self, prompt, expected_format, max_retries=3):
        """带重试的输出解析"""

        for attempt in range(max_retries):
            # 1. 获取原始输出
            raw_output = self.llm.chat(prompt)

            # 2. 清洗输出
            cleaned = self.sanitize(raw_output)

            # 3. 尝试解析
            try:
                result = self.parse(cleaned, expected_format)
                return result
            except ParseError as e:
                # 4. 解析失败，用更强的约束重试
                prompt = self.strengthen_prompt(prompt, raw_output, e)
                continue

        # 所有重试都失败，使用兜底策略
        return self.fallback_parse(raw_output, expected_format)

    def sanitize(self, raw_output):
        """清洗输出"""
        # 去除 markdown 代码块标记
        raw_output = re.sub(r'```(?:json|python)?\s*', '', raw_output)
        raw_output = re.sub(r'```\s*$', '', raw_output)

        # 去除前后多余文字
        # 尝试提取 JSON 部分
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            return json_match.group()

        return raw_output.strip()

    def parse(self, text, expected_format):
        """根据期望格式解析"""
        if expected_format == "json":
            return json.loads(text)
        elif expected_format == "react":
            return self.parse_react(text)
        elif expected_format == "xml":
            return self.parse_xml(text)
        else:
            raise ValueError(f"未知格式：{expected_format}")

    def strengthen_prompt(self, original_prompt, bad_output, error):
        """在重试时加强格式约束"""
        return f"""{original_prompt}

注意：你的上次输出格式有误。
错误：{error}
你的输出：{bad_output}

请严格按以下 JSON schema 输出，不要添加任何其他文字：
```json
&#123;&#123;"key": "value"&#125;&#125;
```"""

    def fallback_parse(self, raw_output, expected_format):
        """兜底解析策略"""
        # 策略1：让 LLM 帮忙转换格式
        convert_prompt = f"""请将以下文本转换为标准 JSON 格式：
{raw_output}

只返回 JSON，不要其他文字。"""
        converted = self.llm.chat(convert_prompt)
        try:
            return json.loads(converted)
        except:
            # 策略2：返回原始文本，标记解析失败
            return {"raw_output": raw_output, "parse_failed": True}
```

**使用约束解码（Constrained Decoding）是更根本的解决方案：**

```python
# 使用 Instructor 库（基于 Pydantic 模型约束输出）
import instructor
from pydantic import BaseModel

class ToolCall(BaseModel):
    thought: str
    tool_name: str
    tool_input: dict

client = instructor.from_openai(OpenAI())

# 模型输出会被强制约束为 ToolCall 格式
result = client.chat.completions.create(
    model="gpt-4",
    response_model=ToolCall,  # 输出必然是合法的 ToolCall
    messages=[{"role": "user", "content": prompt}]
)
```

**追问：**
- JSON Mode 和 Function Calling 能解决格式问题吗？能解决大部分，但不是 100%。模型仍然可能生成合法但语义不正确的 JSON。
- XML 格式和 JSON 格式哪个更好？JSON 更通用，但 XML 在嵌套复杂结构时更不容易出格式错误（有闭合标签做校验）。

---

## 六、状态管理

### Q: 如何管理 Agent 的执行状态？⭐⭐

**答：**

Agent 的状态管理决定了它的可靠性和可恢复性。一个 Agent 的完整状态包括：**当前任务、执行历史、中间结果、工具状态、配置信息**。

类比：Agent 的状态就像一个"手术记录单"——记录了手术（任务）的每一步操作、每一步的发现、以及下一步计划。

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import json

class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentState:
    """Agent 完整状态"""
    task_id: str
    task_description: str
    status: AgentStatus = AgentStatus.IDLE

    # 执行历史
    steps: list = field(default_factory=list)
    current_step: int = 0

    # 中间结果
    context: dict = field(default_factory=dict)

    # 资源追踪
    token_used: int = 0
    tool_calls_count: int = 0
    start_time: float = 0

    def add_step(self, thought, action, observation):
        """记录一步执行"""
        self.steps.append({
            "step_id": len(self.steps),
            "thought": thought,
            "action": action,
            "observation": observation,
            "timestamp": time.time()
        })
        self.current_step = len(self.steps)

    def get_context_window(self, max_tokens=4000):
        """获取适配上下文窗口的历史"""
        # 优先保留最近的步骤，必要时压缩早期步骤
        total_tokens = 0
        selected_steps = []

        for step in reversed(self.steps):
            step_tokens = count_tokens(step)
            if total_tokens + step_tokens > max_tokens:
                # 早期步骤压缩为摘要
                summary = self.summarize_steps(self.steps[:step["step_id"]])
                selected_steps.insert(0, {"summary": summary})
                break
            selected_steps.insert(0, step)
            total_tokens += step_tokens

        return selected_steps

    def to_dict(self):
        """序列化为字典，用于持久化"""
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "status": self.status.value,
            "steps": self.steps,
            "context": self.context,
            "token_used": self.token_used,
        }

    @classmethod
    def from_dict(cls, data):
        """从字典恢复状态"""
        state = cls(
            task_id=data["task_id"],
            task_description=data["task_description"]
        )
        state.status = AgentStatus(data["status"])
        state.steps = data["steps"]
        state.context = data["context"]
        return state
```

**状态持久化方案：**

```python
class StateStore:
    """状态持久化存储"""

    def save(self, state: AgentState):
        """保存状态"""
        data = state.to_dict()
        # 方案1：Redis（适合分布式）
        redis_client.set(f"agent:{state.task_id}", json.dumps(data))
        # 方案2：本地文件（适合单机）
        with open(f"state/{state.task_id}.json", "w") as f:
            json.dump(data, f)

    def load(self, task_id: str) -> AgentState:
        """恢复状态"""
        data = redis_client.get(f"agent:{task_id}")
        return AgentState.from_dict(json.loads(data))
```

**追问：**
- 状态管理和记忆（Memory）有什么区别？状态是当前执行的即时信息，记忆是跨会话的长期信息。状态是短期的、精确的；记忆是长期的、可能被压缩的。
- 分布式环境下如何保证状态一致性？用 Redis + 分布式锁，或使用数据库事务。

---

### Q: 什么是 Agent 的 Scratchpad？⭐⭐

**答：**

Scratchpad（草稿纸）是 Agent 的**工作记忆区**，存放当前任务执行过程中的所有中间信息。就像数学考试时你打草稿的那张纸——记录每一步的计算过程，方便后续参考。

在 ReAct 架构中，Scratchpad 就是拼接到 prompt 中的完整历史轨迹。

```python
class Scratchpad:
    """Agent 的草稿纸"""

    def __init__(self, max_size=8000):
        self.entries = []
        self.max_size = max_size  # 最大 token 数

    def add(self, entry_type, content):
        """添加一条记录"""
        self.entries.append({
            "type": entry_type,  # thought/action/observation/note
            "content": content,
            "timestamp": time.time(),
            "tokens": count_tokens(content)
        })
        self.trim()  # 超出大小时自动裁剪

    def trim(self):
        """当超出大小限制时，压缩早期内容"""
        total = sum(e["tokens"] for e in self.entries)
        if total <= self.max_size:
            return

        # 保留最近的条目，压缩早期的
        recent = []
        early_content = []
        early_tokens = 0

        for entry in reversed(self.entries):
            if sum(e["tokens"] for e in recent) + entry["tokens"] > self.max_size * 0.7:
                early_content.insert(0, entry["content"])
                early_tokens += entry["tokens"]
            else:
                recent.insert(0, entry)

        # 用 LLM 压缩早期内容为摘要
        if early_content:
            summary = self.summarize(early_content)
            self.entries = [{"type": "summary", "content": summary, "tokens": count_tokens(summary)}] + recent

    def to_prompt(self):
        """转换为 prompt 文本"""
        parts = []
        for entry in self.entries:
            if entry["type"] == "thought":
                parts.append(f"Thought: {entry['content']}")
            elif entry["type"] == "action":
                parts.append(f"Action: {entry['content']}")
            elif entry["type"] == "observation":
                parts.append(f"Observation: {entry['content']}")
            elif entry["type"] == "summary":
                parts.append(f"[之前的步骤摘要] {entry['content']}")
            elif entry["type"] == "note":
                parts.append(f"Note: {entry['content']}")
        return "\n".join(parts)
```

**Scratchpad 的几种使用模式：**

```python
# 模式1：完整历史（小任务）
scratchpad.to_prompt()  # 把所有历史都放进 prompt

# 模式2：滑动窗口（中等任务）
scratchpad.get_recent(n=5)  # 只保留最近5步

# 模式3：压缩历史（长任务）
scratchpad.get_compressed()  # 早期步骤压缩为摘要

# 模式4：结构化（需要精确控制）
scratchpad.get_structured()  # 分类展示：工具结果/思考/错误
```

**追问：**
- Scratchpad 和 Conversation History 有什么区别？Scratchpad 是任务级别的，通常只在一次任务执行期间存在；Conversation History 是会话级别的，跨多个任务。
- Scratchpad 放在 prompt 里不会太浪费 token 吗？会，所以需要压缩策略。实际生产中，通常把工具返回的原始数据存到外部存储，Scratchpad 只存摘要和关键信息。

---

### Q: 如何实现 Agent 的断点续传？⭐⭐⭐

**答：**

断点续传让 Agent 在中断（进程崩溃、超时、用户暂停）后，能从上次中断的地方继续执行，而不是从头开始。这在长时间运行的任务中非常重要。

```python
class CheckpointManager:
    """检查点管理器"""

    def __init__(self, state_store):
        self.state_store = state_store

    def save_checkpoint(self, agent_state: AgentState):
        """保存检查点"""
        checkpoint = {
            "task_id": agent_state.task_id,
            "status": agent_state.status.value,
            "completed_steps": agent_state.steps,
            "current_step": agent_state.current_step,
            "context": agent_state.context,
            "scratchpad": agent_state.scratchpad.to_dict(),
            "metadata": {
                "token_used": agent_state.token_used,
                "timestamp": time.time()
            }
        }
        self.state_store.save(checkpoint)
        return checkpoint

    def load_checkpoint(self, task_id):
        """加载检查点"""
        return self.state_store.load(task_id)


class ResumableAgent:
    """支持断点续传的 Agent"""

    def __init__(self, llm, tools, checkpoint_manager):
        self.llm = llm
        self.tools = tools
        self.checkpoint_mgr = checkpoint_manager

    def run(self, task, task_id=None, resume_from=None):
        """执行任务，支持断点续传"""

        if resume_from:
            # 从检查点恢复
            state = self.checkpoint_mgr.load_checkpoint(resume_from)
            print(f"从步骤 {state.current_step} 恢复执行")
        else:
            # 新建任务
            state = AgentState(
                task_id=task_id or str(uuid.uuid4()),
                task_description=task
            )

        # 构建执行计划
        if not state.context.get("plan"):
            plan = self.create_plan(task)
            state.context["plan"] = plan

        # 从当前步骤开始执行
        plan = state.context["plan"]

        for step_idx in range(state.current_step, len(plan)):
            step = plan[step_idx]
            state.current_step = step_idx

            try:
                # 执行步骤
                result = self.execute_step(step, state)
                state.add_step(step.thought, step.action, result)

                # 每步保存检查点
                self.checkpoint_mgr.save_checkpoint(state)

            except KeyboardInterrupt:
                # 用户暂停
                state.status = AgentStatus.PAUSED
                self.checkpoint_mgr.save_checkpoint(state)
                print(f"任务已暂停，task_id={state.task_id}")
                return {"status": "paused", "task_id": state.task_id}

            except Exception as e:
                # 执行失败
                state.status = AgentStatus.FAILED
                self.checkpoint_mgr.save_checkpoint(state)
                raise

        state.status = AgentStatus.COMPLETED
        self.checkpoint_mgr.save_checkpoint(state)
        return self.summarize(state)
```

**使用示例：**

```python
# 首次执行
agent = ResumableAgent(llm, tools, checkpoint_mgr)
result = agent.run("帮我分析这个代码仓库的安全性", task_id="task_001")

# 中断后恢复
result = agent.run("", resume_from="task_001")  # task 从检查点加载
```

**追问：**
- 如何处理检查点时正在执行的工具调用？需要在工具调用前后都保存检查点，恢复时判断上次是在调用前还是调用后中断的。
- 检查点数据会很大吗？可能会，特别是工具返回了大量数据时。建议工具返回结果存到外部存储，检查点只存引用。

---

## 七、实际开发

### Q: 如何调试 Agent？有哪些调试技巧？⭐⭐

**答：**

Agent 调试比普通程序难得多，因为**执行路径是动态的**，而且涉及 LLM 的"黑盒"推理。以下是我常用的调试技巧：

**技巧一：完整的 Trace 日志**

```python
import logging
import json
from datetime import datetime

class AgentTracer:
    """Agent 执行追踪器"""

    def __init__(self, log_file="agent_trace.jsonl"):
        self.traces = []
        self.log_file = log_file

    def log_step(self, step_info):
        """记录每一步的完整信息"""
        trace = {
            "timestamp": datetime.now().isoformat(),
            "step": step_info["step_num"],
            "prompt": step_info["prompt"][:500],  # 截断避免日志过大
            "llm_output": step_info["llm_output"],
            "tool_called": step_info.get("tool_name"),
            "tool_input": step_info.get("tool_input"),
            "tool_output": str(step_info.get("tool_output", ""))[:300],
            "tokens_used": step_info.get("tokens"),
            "latency_ms": step_info.get("latency"),
        }
        self.traces.append(trace)

        # 实时写入文件
        with open(self.log_file, "a") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

        # 关键信息打印到控制台
        print(f"[Step {trace['step']}] "
              f"Tool: {trace['tool_called']} | "
              f"Tokens: {trace['tokens_used']} | "
              f"Latency: {trace['latency_ms']}ms")
```

**技巧二：可视化执行流程**

```python
def visualize_trace(trace_file):
    """将 trace 可视化为可读格式"""
    with open(trace_file) as f:
        traces = [json.loads(line) for line in f]

    print("=" * 60)
    print("AGENT 执行追踪")
    print("=" * 60)

    for t in traces:
        print(f"\n📍 Step {t['step']} [{t['timestamp']}]")
        print(f"  💭 Thought: {t['llm_output'][:100]}...")
        if t['tool_called']:
            print(f"  🔧 Tool: {t['tool_called']}")
            print(f"  📥 Input: {json.dumps(t['tool_input'], ensure_ascii=False)[:100]}")
            print(f"  📤 Output: {t['tool_output'][:100]}")
        print(f"  ⏱️ Latency: {t['latency_ms']}ms | Tokens: {t['tokens_used']}")
```

**技巧三：Prompt 测试框架**

```python
class AgentTestRunner:
    """Agent 测试框架"""

    def __init__(self, agent):
        self.agent = agent
        self.results = []

    def test_case(self, name, task, expected_tools=None, expected_answer_contains=None):
        """运行一个测试用例"""
        print(f"\n🧪 Running: {name}")
        result = self.agent.run(task, trace=True)

        passed = True
        issues = []

        # 检查是否调用了正确的工具
        if expected_tools:
            actual_tools = [s["tool"] for s in result.steps if s.get("tool")]
            for tool in expected_tools:
                if tool not in actual_tools:
                    passed = False
                    issues.append(f"期望调用 {tool}，实际未调用")

        # 检查答案是否包含关键词
        if expected_answer_contains:
            for keyword in expected_answer_contains:
                if keyword not in result.answer:
                    passed = False
                    issues.append(f"答案中缺少 '{keyword}'")

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}")
        if issues:
            for issue in issues:
                print(f"    ⚠️ {issue}")

        self.results.append({"name": name, "passed": passed, "result": result})
        return passed

# 使用
tester = AgentTestRunner(agent)
tester.test_case(
    "天气查询",
    "北京今天天气怎么样？",
    expected_tools=["get_weather"],
    expected_answer_contains=["北京", "℃"]
)
```

**追问：**
- LLM 推理过程不可控怎么办？可以把温度设为 0，减少随机性，让调试更可复现。
- 如何快速定位是 Prompt 问题还是工具问题？对比"直接给模型正确的工具输出"和"让模型自己调用工具"的结果差异。

---

### Q: Agent 的延迟优化有哪些手段？⭐⭐

**答：**

Agent 的延迟来源主要有三个：**LLM 推理、工具调用、多轮循环**。优化也需要从这三个方向入手。

```python
class OptimizedAgent:
    """延迟优化的 Agent"""

    # 优化1：并行工具调用
    async def execute_tools_parallel(self, tool_calls):
        """独立的工具调用并行执行"""
        tasks = [self.execute_tool(tc) for tc in tool_calls]
        return await asyncio.gather(*tasks)

    # 优化2：流式输出
    async def stream_response(self, task):
        """流式返回结果，提升用户感知速度"""
        async for chunk in self.llm.stream_chat(task):
            yield chunk  # 逐 token 输出

    # 优化3：缓存
    def cached_tool_call(self, tool_name, params, ttl=300):
        """工具调用结果缓存"""
        cache_key = f"{tool_name}:{hash(json.dumps(params, sort_keys=True))}"
        cached = self.cache.get(cache_key)
        if cached and time.time() - cached["time"] < ttl:
            return cached["result"]  # 缓存命中

        result = self.tools[tool_name](**params)
        self.cache.set(cache_key, {"result": result, "time": time.time()})
        return result

    # 优化4：选择合适的模型
    def select_model(self, task_complexity):
        """根据任务复杂度选择模型"""
        if task_complexity == "simple":
            return "gpt-4o-mini"    # 快、便宜
        elif task_complexity == "medium":
            return "gpt-4o"         # 平衡
        else:
            return "o1"             # 强推理能力

    # 优化5：减少不必要的步骤
    def smart_plan(self, task):
        """优化计划，合并可合并的步骤"""
        plan = self.create_plan(task)
        optimized = self.llm.chat(f"""以下执行计划中，哪些步骤可以合并或跳过？

原始计划：
{format_plan(plan)}

输出优化后的计划。""")
        return parse_plan(optimized)
```

**延迟优化检查清单：**

| 优化手段 | 效果 | 实现难度 |
|----------|------|----------|
| 并行工具调用 | 减少 50-70% 工具调用延迟 | 低 |
| 流式输出 | 用户感知延迟降低 80% | 低 |
| 结果缓存 | 重复查询零延迟 | 低 |
| 模型路由 | 简单任务快 3-5 倍 | 中 |
| 步骤合并 | 减少 LLM 调用次数 | 中 |
| 提前终止 | 避免不必要的后续步骤 | 中 |
| Speculative Execution | 预执行可能的下一步 | 高 |

**追问：**
- 流式输出和 Function Calling 能同时使用吗？可以，但实现比较复杂，需要处理流式解析 tool_calls 的问题。
- 缓存的 key 怎么设计？需要考虑语义相似性——"北京天气"和"北京市今天天气"应该命中同一缓存。

---

### Q: 如何评估 Agent 的效果？⭐⭐

**答：**

评估 Agent 比评估普通 LLM 复杂得多，因为 Agent 涉及**多步推理、工具调用、最终结果**三个维度。我通常用以下评估框架：

```python
class AgentEvaluator:
    """Agent 效果评估框架"""

    def evaluate(self, agent, test_cases):
        """全面评估"""
        results = {
            "accuracy": [],      # 答案正确率
            "efficiency": [],    # 执行效率
            "robustness": [],    # 鲁棒性
            "cost": [],          # 成本
        }

        for case in test_cases:
            result = agent.run(case["task"])

            results["accuracy"].append(
                self.check_accuracy(result, case["expected"])
            )
            results["efficiency"].append({
                "steps": len(result.steps),
                "tokens": result.token_used,
                "latency": result.total_time
            })
            results["cost"].append(result.token_used * COST_PER_TOKEN)

        return self.summarize_results(results)

    def check_accuracy(self, result, expected):
        """多维度准确性检查"""
        scores = {}

        # 1. 最终答案匹配
        scores["answer_match"] = self.fuzzy_match(
            result.answer, expected["answer"]
        )

        # 2. 工具调用路径匹配
        if expected.get("expected_tools"):
            actual_tools = [s.tool for s in result.steps]
            scores["tool_path_match"] = (
                actual_tools == expected["expected_tools"]
            )

        # 3. 关键信息覆盖
        if expected.get("must_contain"):
            scores["coverage"] = all(
                keyword in result.answer
                for keyword in expected["must_contain"]
            )

        # 综合得分
        scores["overall"] = (
            scores["answer_match"] * 0.5 +
            scores.get("tool_path_match", 1.0) * 0.2 +
            scores.get("coverage", 1.0) * 0.3
        )
        return scores

    def evaluate_robustness(self, agent, perturbation_cases):
        """鲁棒性评估：用模糊/错误输入测试"""
        robustness_scores = []
        for case in perturbation_cases:
            result = agent.run(case["noisy_task"])
            still_correct = self.check_accuracy(result, case["expected"])
            robustness_scores.append(still_correct)
        return robustness_scores
```

**评估维度详解：**

| 维度 | 指标 | 说明 |
|------|------|------|
| **准确性** | 答案正确率、关键信息覆盖率 | 最终结果是否正确 |
| **效率** | 步骤数、Token 消耗、延迟 | 完成任务的效率 |
| **工具使用** | 工具选择准确率、参数正确率 | 是否正确使用工具 |
| **鲁棒性** | 噪声输入下的表现 | 面对模糊/错误输入的能力 |
| **安全性** | 越权操作次数、敏感信息泄露 | 是否遵守安全边界 |

**追问：**
- 如何构建测试数据集？从真实用户 query 中采样，人工标注期望答案和执行路径。
- 评估中的"工具使用路径"指标重要吗？非常重要。同样的正确答案，走 2 步和走 10 步的成本差距巨大。

---

## 八、实战难题

### 难题一：Agent 陷入"死循环"——反复调用同一个失败的工具

**场景：** Agent 在查询数据库时，SQL 语法错误导致查询失败，但 Agent 不断重试相同的 SQL，陷入死循环。

**根因分析：** Agent 没有"失败记忆"，每次循环都重新生成同样的错误 SQL。

**解决方案：**

```python
class FailureAwareAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.failure_log = []  # 记录失败的尝试

    def execute_step(self, task, scratchpad):
        # 在 prompt 中加入失败历史
        failure_context = ""
        if self.failure_log:
            failure_context = "\n\n## 已尝试但失败的方法（请勿重复）：\n"
            for f in self.failure_log[-3:]:  # 最近3次
                failure_context += f"- 工具 {f['tool']}，参数 {f['input']}，错误：{f['error']}\n"

        prompt = build_prompt(task, scratchpad, failure_context)
        response = self.llm.chat(prompt)

        # 执行并记录结果
        try:
            result = self.tools.execute(response.tool_name, response.tool_input)
            return result
        except Exception as e:
            self.failure_log.append({
                "tool": response.tool_name,
                "input": response.tool_input,
                "error": str(e)
            })

            # 连续失败 2 次同一工具，强制换策略
            recent_failures = [f for f in self.failure_log[-2:]
                              if f["tool"] == response.tool_name]
            if len(recent_failures) >= 2:
                return self.escalate(task, response.tool_name, str(e))

            raise
```

**教训：** Agent 的 prompt 中必须包含失败历史，否则模型会"健忘"地重复犯错。

---

### 难题二：上下文窗口溢出——长任务执行到一半，上下文满了

**场景：** 执行一个需要 15+ 步的复杂任务，到第 10 步时，scratchpad 已经把上下文窗口占满了。

**解决方案：**

```python
class ContextManager:
    """智能上下文管理"""

    def __init__(self, max_tokens=8000):
        self.max_tokens = max_tokens
        self.full_history = []  # 完整历史存到外部

    def build_context(self, task, recent_steps, early_steps):
        """构建适配上下文窗口的 prompt"""
        # 计算各部分 token 数
        task_tokens = count_tokens(task)
        system_tokens = count_tokens(SYSTEM_PROMPT)
        budget = self.max_tokens - task_tokens - system_tokens - 500  # 留 500 buffer

        # 策略：最近的步骤完整保留，早期步骤压缩
        recent_text = format_steps(recent_steps)
        recent_tokens = count_tokens(recent_text)

        if recent_tokens > budget * 0.7:
            # 最近步骤也太多，只保留最近 3 步
            recent_text = format_steps(recent_steps[-3:])
            recent_tokens = count_tokens(recent_text)

        early_budget = budget - recent_tokens
        if early_steps:
            early_summary = self.summarize_early_steps(early_steps, early_budget)
        else:
            early_summary = ""

        return f"""{SYSTEM_PROMPT}

任务：{task}

{early_summary}

最近的执行记录：
{recent_text}
"""

    def summarize_early_steps(self, steps, token_budget):
        """用 LLM 压缩早期步骤"""
        prompt = f"""请将以下执行历史压缩为摘要（不超过 {token_budget} tokens），
保留关键发现和结论，省略中间过程：

{format_steps(steps)}"""
        return self.llm.chat(prompt)
```

**教训：** 上下文管理要在项目初期就设计好，不要等到溢出才处理。

---

### 难题三：工具返回结果太长，撑爆 prompt

**场景：** Agent 调用搜索工具，返回了 10 页的搜索结果，直接放进 prompt 会超出上下文限制。

**解决方案：**

```python
class ToolResultProcessor:
    """工具结果后处理"""

    def __init__(self, max_result_tokens=500):
        self.max_result_tokens = max_result_tokens

    def process(self, tool_name, raw_result):
        """处理工具返回结果"""

        # 策略1：截断
        if count_tokens(str(raw_result)) > self.max_result_tokens:
            processed = self.truncate(tool_name, raw_result)

        # 策略2：提取关键信息
        if tool_name == "search":
            processed = self.extract_key_info(raw_result)

        # 策略3：结构化
        if tool_name == "database_query":
            processed = self.format_as_table(raw_result)

        return processed

    def extract_key_info(self, search_results):
        """从搜索结果中提取关键信息"""
        prompt = f"""从以下搜索结果中提取与问题最相关的关键信息（3-5 条）：

{search_results}

每条信息格式：[来源] 关键内容"""
        return self.llm.chat(prompt)

    def truncate(self, tool_name, result):
        """智能截断：保留开头和结尾"""
        text = str(result)
        tokens = count_tokens(text)
        if tokens > self.max_result_tokens:
            # 保留前 70% 和后 30%
            keep_chars = int(len(text) * self.max_result_tokens / tokens)
            head = text[:int(keep_chars * 0.7)]
            tail = text[-int(keep_chars * 0.3):]
            return f"{head}\n...[中间部分省略]...\n{tail}"
        return text
```

**教训：** 所有外部数据在进入 prompt 之前，都要经过"过滤器"处理。

---

### 难题四：多 Agent 协作中的信息丢失

**场景：** 两个 Agent 协作完成任务，Agent A 把结果传给 Agent B，但 B 理解不了 A 的输出格式，导致信息丢失。

**解决方案：**

```python
class AgentCommunicationProtocol:
    """Agent 间通信协议"""

    @dataclass
    class Message:
        sender: str
        receiver: str
        task: str
        data: dict
        schema: str  # 数据格式说明
        confidence: float  # 发送方对结果的信心度

    def send_with_schema(self, sender, receiver, data, task):
        """带 schema 的消息发送"""
        # 自动生成数据描述
        schema_desc = self.describe_data(data)

        message = self.Message(
            sender=sender,
            receiver=receiver,
            task=task,
            data=data,
            schema=schema_desc,
            confidence=self.estimate_confidence(data)
        )

        # 接收方验证
        if not self.validate_message(message):
            # 数据不兼容，请求发送方补充
            return self.request_clarification(message)

        return message

    def describe_data(self, data):
        """自动描述数据结构"""
        if isinstance(data, dict):
            return {k: type(v).__name__ for k, v in data.items()}
        return type(data).__name__

    def format_for_receiver(self, message, receiver_preferences):
        """根据接收方偏好格式化数据"""
        if receiver_preferences.get("format") == "json":
            return json.dumps(message.data, ensure_ascii=False)
        elif receiver_preferences.get("format") == "natural_language":
            return self.to_natural_language(message.data)
        return str(message.data)
```

**教训：** 多 Agent 系统需要定义清晰的"通信协议"，就像 API 需要文档一样。

---

### 难题五：Agent 在生产环境中成本失控

**场景：** Agent 上线后，某些复杂任务的 token 消耗远超预期，一天烧掉了几天的预算。

**解决方案：**

```python
class CostController:
    """Agent 成本控制器"""

    def __init__(self, daily_budget=100.0, per_task_limit=5.0):
        self.daily_budget = daily_budget      # 每日预算（美元）
        self.per_task_limit = per_task_limit   # 单任务上限
        self.daily_cost = 0
        self.task_costs = {}

    def check_budget(self, task_id, estimated_tokens):
        """执行前检查预算"""
        estimated_cost = estimated_tokens * COST_PER_TOKEN

        # 检查单任务限制
        current_task_cost = self.task_costs.get(task_id, 0)
        if current_task_cost + estimated_cost > self.per_task_limit:
            raise BudgetExceededError(
                f"任务 {task_id} 已接近成本上限 "
                f"(${current_task_cost:.2f} / ${self.per_task_limit})"
            )

        # 检查每日预算
        if self.daily_cost + estimated_cost > self.daily_budget * 0.9:
            # 接近预算上限，降级到便宜模型
            return "gpt-4o-mini"  # 切换到便宜模型

        if self.daily_cost + estimated_cost > self.daily_budget:
            raise BudgetExceededError("每日预算已耗尽")

        return None  # 正常执行

    def record_cost(self, task_id, tokens_used, model):
        """记录消耗"""
        cost = tokens_used * MODEL_COSTS[model]
        self.daily_cost += cost
        self.task_costs[task_id] = self.task_costs.get(task_id, 0) + cost

    def get_cost_report(self):
        """成本报告"""
        return {
            "daily_cost": f"${self.daily_cost:.2f}",
            "budget_remaining": f"${self.daily_budget - self.daily_cost:.2f}",
            "top_tasks": sorted(
                self.task_costs.items(), key=lambda x: x[1], reverse=True
            )[:5]
        }
```

**成本控制的几条铁律：**
1. 每个任务必须有 token 预算上限
2. 设置每日/每月总预算，超限后降级或停止
3. 监控异常：同一任务 token 消耗突然翻倍要报警
4. 简单任务用小模型，复杂任务才用大模型
5. 缓存一切可以缓存的结果

**教训：** 成本控制不是事后优化，要在架构设计阶段就考虑。

---

## 总结

| 主题 | 核心要点 |
|------|----------|
| Agent 基础 | 自主感知-推理-行动-记忆循环 |
| ReAct | 思考-行动-观察交替，简单高效但无全局规划 |
| Plan-and-Execute | 先规划再执行，适合复杂任务，需配合动态重规划 |
| Function Calling | 模型输出结构化工具调用，应用层负责执行 |
| 错误恢复 | 分层处理：重试→参数修正→工具降级→跳过 |
| 状态管理 | Scratchpad + 检查点 + 持久化 |
| 实战核心 | 循环防护、上下文管理、成本控制、通信协议 |

---

## 二、Agent 高级架构与工程实践

### Q: 什么是 LATS（Language Agent Tree Search）？和 ReAct、Reflexion 有什么区别？⭐⭐⭐

**答：**

LATS（Language Agent Tree Search）是将 **蒙特卡洛树搜索（MCTS）** 思想引入 LLM Agent 的框架。与 ReAct 的线性"思考-行动"链不同，LATS 构建一棵**搜索树**，每个节点代表一个状态，每条边代表一个动作，通过探索多条路径找到最优解。

**核心思想对比：**

| 方法 | 搜索策略 | 反馈机制 | 适用场景 |
|------|---------|---------|---------|
| **ReAct** | 贪心（只走一条路） | 环境观察 | 简单推理任务 |
| **Reflexion** | 单链 + 事后反思 | 自我语言反馈 | 需要纠错的任务 |
| **LATS** | 树搜索（多路径探索） | 价值函数 + LLM启发式 | 复杂决策/博弈 |

LATS 的核心流程：**选择（Select）→ 扩展（Expand）→ 评估（Evaluate）→ 回溯（Backpropagate）**，与 MCTS 完全对应。

```python
class LATSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.is_terminal = False

    def uct_score(self, exploration=1.4):
        """UCB1 公式，平衡探索与利用"""
        if self.visits == 0:
            return float('inf')
        exploit = self.value / self.visits
        explore = exploration * (2 * (self.parent.visits if self.parent else 0) / self.visits) ** 0.5
        return exploit + explore

class LATSAgent:
    def __init__(self, llm, environment, max_simulations=10):
        self.llm = llm
        self.env = environment
        self.max_simulations = max_simulations

    def search(self, root_state):
        root = LATSNode(root_state)
        for _ in range(self.max_simulations):
            # 1. 选择：沿 UCT 最高的路径向下
            node = self._select(root)
            # 2. 扩展：LLM 生成候选动作，创建子节点
            if not node.is_terminal:
                node = self._expand(node)
            # 3. 评估：LLM 对叶节点状态打分
            value = self._evaluate(node)
            # 4. 回溯：将价值沿路径向上传播
            self._backpropagate(node, value)
        # 返回价值最高的子树路径
        return self._best_action_sequence(root)

    def _expand(self, node):
        """LLM 作为启发式搜索：生成多个候选动作"""
        actions = self.llm.chat(
            f"当前状态：{node.state}\n"
            f"请给出 3 个可能的下一步动作。"
        )
        for action in actions:
            child_state = self.env.simulate(node.state, action)
            child = LATSNode(child_state, parent=node, action=action)
            node.children.append(child)
        return node.children[0]  # 先探索第一个

    def _evaluate(self, node):
        """LLM 评估状态价值：0~1 分"""
        score = self.llm.chat(
            f"状态：{node.state}\n"
            f"评估当前状态的优劣（0-10分）："
        )
        return score / 10.0
```

**与 ReAct/Reflexion 的本质区别：** ReAct 是"走一步看一步"，Reflexion 是"走错了再反思"，LATS 是"同时走多条路，选最好的那条"。LATS 的代价是更多的 LLM 调用，适合高价值决策场景。

**追问：**
- LATS 的价值函数如何设计？可以用 LLM 作为评判（LLM-as-a-Judge），也可以用任务特定的奖励模型。
- LATS 什么时候比 ReAct 差？当动作空间极小（2-3步就能完成）时，树搜索的开销不划算，ReAct 的贪心策略更高效。
- LATS 与 MCTS 的最大区别？MCTS 的模拟（Simulation）阶段用随机 rollout，LATS 用 LLM 生成有语义的动作，每个节点的"模拟"本身就是一次推理。

---

### Q: AutoGPT 的架构原理？为什么 AutoGPT 容易陷入死循环？⭐⭐⭐

**答：**

AutoGPT 是早期最知名的自主 Agent 框架，核心理念是"给一个目标，让 AI 自己完成所有步骤"。它的架构是一个**目标驱动的自主循环**：设定目标 → 生成计划 → 执行任务 → 评估结果 → 更新计划 → 继续执行。

**架构核心组件：**

```python
class AutoGPTAgent:
    def __init__(self, llm, tools, memory, goals):
        self.llm = llm
        self.tools = tools
        self.memory = memory      # 短期 + 长期记忆
        self.goals = goals         # 用户设定的目标
        self.plan = []             # 生成的计划
        self.step_count = 0
        self.max_steps = 50

    def run(self):
        self.plan = self._create_plan(self.goals)
        while not self._is_goal_achieved():
            if self.step_count >= self.max_steps:
                return "达到最大步数限制"
            # 1. 从计划中取下一步
            current_task = self._get_next_task()
            # 2. 执行
            result = self._execute(current_task)
            # 3. 自我评估
            assessment = self._self_evaluate(current_task, result)
            # 4. 根据评估更新计划
            if assessment.status == "failed":
                self._replan(result, assessment.reason)
            elif assessment.status == "partial":
                self._adjust_plan(assessment.suggestion)
            # 5. 存入记忆
            self.memory.store(current_task, result, assessment)
            self.step_count += 1

    def _self_evaluate(self, task, result):
        """自我反馈：LLM 评估自己的执行结果"""
        return self.llm.chat(
            f"目标：{self.goals}\n"
            f"当前任务：{task}\n"
            f"执行结果：{result}\n"
            f"评估：任务是否完成？有何问题？下一步建议？"
        )

    def _replan(self, failed_result, reason):
        """失败后重新规划"""
        new_plan = self.llm.chat(
            f"原计划：{self.plan}\n"
            f"执行失败：{failed_result}\n"
            f"失败原因：{reason}\n"
            f"请重新制定计划。"
        )
        self.plan = new_plan
```

**为什么 AutoGPT 容易陷入死循环？**

1. **自我评估不可靠：** LLM 评估自己的输出，容易"自我欺骗"——认为没完成的任务已完成，或在错误方向上反复尝试。
2. **缺乏终止信号：** 目标是模糊的自然语言，LLM 难以判断"什么时候算做完"。
3. **重规划退化：** 每次失败后的重规划可能生成几乎相同的计划，导致"原地打转"。
4. **记忆膨胀：** 随着历史变长，上下文窗口塞满后，LLM 推理质量急剧下降。

**改进方案：**

```python
class ImprovedAutoGPT:
    def __init__(self):
        self.loop_detector = LoopDetector(window_size=5, similarity_threshold=0.8)

    def run_with_safeguards(self):
        while not self._is_done():
            action = self._decide()
            # 循环检测：最近N步相似度过高则强制换策略
            if self.loop_detector.detect_loop(self.recent_actions):
                action = self._force_exploration("检测到循环，尝试全新策略")
            # 进度惩罚：鼓励"前进"而非重复
            progress_score = self._measure_progress()
            if progress_score < 0.1:
                self._escalate_to_human("长时间无进展，请人工指导")
            self._execute(action)
```

**追问：**
- 循环检测的具体实现？可以用 action embedding 的余弦相似度，或检查连续 N 步的工具调用模式。
- 为什么不用 Plan-and-Execute 替代 AutoGPT？Plan-and-Execute 更结构化，但 AutoGPT 的自主性更强，适合探索性任务。
- AutoGPT 和 BabyAGI 的区别？BabyAGI 更侧重任务队列管理，AutoGPT 更侧重完整的执行循环。

---

### Q: Agent 框架对比？LangChain Agent vs LlamaIndex Agent vs AutoGPT vs CrewAI？⭐⭐

**答：**

主流 Agent 框架各有侧重，选型需要根据具体场景决定。

| 框架 | 核心定位 | 架构特点 | 适用场景 |
|------|---------|---------|---------|
| **LangChain Agent** | 通用 Agent 工具链 | 链式调用 + 工具注册 + ReAct/Plan-and-Execute | 快速原型、工具集成 |
| **LlamaIndex Agent** | 数据驱动的 RAG Agent | 索引 + 检索 + 查询引擎 + 子Agent路由 | 知识库问答、文档分析 |
| **AutoGPT** | 自主任务执行 | 目标驱动 + 自我反馈 + 记忆 | 探索性任务、创意生成 |
| **CrewAI** | 多 Agent 协作 | 角色定义 + 任务分配 + 流程编排 | 复杂工作流、团队模拟 |

```python
# LangChain Agent：工具调用为主
from langchain.agents import AgentExecutor, create_react_agent
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5)
result = executor.invoke({"input": "查询今天的天气"})

# LlamaIndex Agent：数据查询为主
from llama_index.core.agent import ReActAgent
agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)
response = agent.chat("总结这份文档的要点")

# CrewAI：多角色协作
from crewai import Agent, Task, Crew
researcher = Agent(role="研究员", goal="收集资料", tools=[search_tool])
writer = Agent(role="写手", goal="撰写报告", tools=[file_tool])
research_task = Task(description="调研Agent框架", agent=researcher)
write_task = Task(description="撰写对比报告", agent=writer)
crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
result = crew.kickoff()
```

**架构差异的核心：**
- **LangChain** 的核心抽象是 `Chain`（链），Agent 是一种特殊的 Chain——带循环的链。
- **LlamaIndex** 的核心抽象是 `Index`（索引），Agent 围绕"如何高效检索和利用数据"展开。
- **CrewAI** 的核心抽象是 `Crew`（团队），强调 Agent 之间的协作与通信。

**追问：**
- LangGraph 和 LangChain Agent 的关系？LangGraph 是 LangChain 的升级版，用图（Graph）替代链，支持更复杂的循环和分支。
- 生产环境选哪个？LangGraph 最成熟，LlamaIndex 在数据密集场景最强，CrewAI 适合多角色场景。
- 这些框架的共同瓶颈？都是 LLM 调用延迟和成本控制，框架层的差异远不如模型能力的差异重要。

---

### Q: 如何实现 Agent 的 Human-in-the-Loop？⭐⭐⭐

**答：**

Human-in-the-Loop（HITL）是指在 Agent 执行过程中引入人工审批、确认或干预的机制。这是生产环境 Agent 的**必备能力**——不能让 AI 在无人监督下执行高风险操作（如转账、删除数据、发送邮件）。

**核心设计模式：**

```python
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# 1. 基于 LangGraph 的 interrupt_before 实现
def build_hitl_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("think", think_node)
    workflow.add_node("act", act_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("execute", execute_tool_node)

    workflow.add_edge("think", "act")
    # 高风险操作前插入人工审批节点
    workflow.add_conditional_edges("act", route_by_risk, {
        "low_risk": "execute",      # 低风险直接执行
        "high_risk": "human_review"  # 高风险需人工审批
    })
    workflow.add_edge("human_review", "execute")
    workflow.add_edge("execute", "think")

    # 关键：interrupt_before 在进入节点前暂停
    app = workflow.compile(
        checkpointer=SqliteSaver.from_conn_string(":memory:"),
        interrupt_before=["human_review"]  # 进入审批前暂停
    )
    return app

# 2. 断点续传实现
def run_with_hitl(app, user_input, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    # 运行到断点自动暂停
    result = app.invoke({"input": user_input}, config)

    # 检查是否暂停在人工审批节点
    state = app.get_state(config)
    if state.next == ("human_review",):
        # 展示给用户审批
        print(f"Agent 建议执行：{state.values['pending_action']}")
        approval = input("是否批准？(yes/no/modify): ")

        if approval == "yes":
            app.update_state(config, {"approved": True})
        elif approval == "no":
            app.update_state(config, {"approved": False, "reason": "用户拒绝"})
        else:
            new_action = input("请提供修改后的操作：")
            app.update_state(config, {"pending_action": new_action, "approved": True})

        # 从断点继续执行
        result = app.invoke(None, config)
    return result
```

**人工干预的三个层级：**
1. **审批（Approval）：** 执行前确认，最常见
2. **编辑（Edit）：** 人工修改 Agent 的参数或计划后再执行
3. **接管（Takeover）：** 人工直接接管执行，Agent 退为观察者

**追问：**
- 如何决定哪些操作需要人工审批？基于风险等级标签（risk_level: low/medium/high），高风险操作必须审批。
- 如何避免 HITL 拖慢系统？异步审批 + 超时机制——审批请求发到 Slack/飞书，超时未响应则自动拒绝。
- LangGraph 的 interrupt 和 interrupt_before 区别？`interrupt_before` 在节点执行前暂停，`interrupt` 可以在节点内部任意位置暂停。

---

### Q: 什么是 Agent 的 Observability？如何实现？⭐⭐

**答：**

Observability（可观测性）是指能够**追踪、监控和调试** Agent 内部运行状态的能力。Agent 不同于传统的 API �用——一次用户请求可能触发 10+ 次 LLM 调用、20+ 次工具调用，没有 Observability 就是在"盲人摸象"。

**三大支柱：**

| 支柱 | 含义 | 工具示例 |
|------|------|---------|
| **Trace（追踪）** | 一次完整调用的全链路 | LangSmith、Phoenix |
| **Log（日志）** | 每一步的详细输入输出 | 结构化日志 |
| **Metrics（指标）** | 延迟、成本、成功率等聚合数据 | Prometheus、Grafana |

```python
# 1. 基于 LangSmith 的 Trace 实现
import langsmith
from langsmith import traceable

@traceable(name="agent_step")
def agent_step(input_text: str) -> str:
    """自动被 LangSmith 追踪：输入、输出、延迟、token数"""
    thought = llm.chat(f"思考：{input_text}")
    if thought.tool_call:
        result = execute_tool(thought.tool_name, thought.tool_input)
        return llm.chat(f"基于结果回答：{result}")
    return thought.answer

# 2. 自定义 Trace 追踪
class AgentTracer:
    def __init__(self):
        self.spans = []

    def trace_step(self, step_name):
        """装饰器：追踪每个 Agent 步骤"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                span = {
                    "name": step_name,
                    "input": str(args),
                    "start_time": time.time(),
                }
                try:
                    result = func(*args, **kwargs)
                    span["output"] = str(result)[:500]
                    span["status"] = "success"
                    return result
                except Exception as e:
                    span["status"] = "error"
                    span["error"] = str(e)
                    raise
                finally:
                    span["duration_ms"] = (time.time() - span["start_time"]) * 1000
                    self.spans.append(span)
            return wrapper
        return decorator

# 3. 核心指标监控
class AgentMetrics:
    def __init__(self):
        self.metrics = defaultdict(list)

    def record(self, step, latency_ms, tokens, cost, success):
        self.metrics[step].append({
            "latency": latency_ms,
            "tokens": tokens,
            "cost": cost,
            "success": success
        })

    def get_summary(self):
        return {
            step: {
                "avg_latency": np.mean([m["latency"] for m in vals]),
                "total_tokens": sum(m["tokens"] for m in vals),
                "total_cost": sum(m["cost"] for m in vals),
                "success_rate": np.mean([m["success"] for m in vals])
            }
            for step, vals in self.metrics.items()
        }
```

**生产环境 Observability 清单：**
- 每次 LLM 调用记录：prompt、response、model、tokens、latency
- 每次工具调用记录：tool_name、input、output、latency、error
- Agent 整体记录：总步数、总耗时、总成本、是否达成目标
- 异常记录：重试次数、降级次数、超时次数

**追问：**
- Trace 和传统日志的区别？Trace 是一次完整请求的全链路关联，日志是离散的事件记录。Trace 能看到"第3步的输入来自第2步的输出"。
- Phoenix 和 LangSmith 的区别？Phoenix（Arize）是开源的，侧重 LLM 特定的评估（Hallucination 检测）；LangSmith 是 LangChain 官方的 SaaS，与 LangChain 生态深度集成。
- 如何在不引入额外依赖的情况下实现基础 Observability？用 Python logging + 结构化 JSON 输出 + 请求 ID 串联即可。

---

### Q: 如何处理 Agent 的长任务？⭐⭐⭐

**答：**

长任务是 Agent 系统的一大挑战。一个复杂的调研任务可能需要 30+ 分钟、100+ 步，期间可能遇到超时、Token 耗尽、进程崩溃等问题。核心策略是**分治 + 持久化 + 异步**。

```python
import asyncio
from dataclasses import dataclass, field
from typing import Optional
import json

@dataclass
class TaskCheckpoint:
    """检查点：支持断点续传"""
    task_id: str
    total_steps: int
    completed_steps: int
    current_step: dict
    results: list = field(default_factory=list)
    status: str = "in_progress"  # in_progress / paused / failed / completed

class LongTaskManager:
    def __init__(self, checkpoint_store, timeout_seconds=1800):
        self.checkpoint_store = checkpoint_store
        self.timeout = timeout_seconds

    async def execute_long_task(self, task_id, subtasks, agent):
        """长任务执行器：带断点续传"""
        # 1. 恢复已有进度（如有）
        checkpoint = self.checkpoint_store.load(task_id)
        if checkpoint:
            start_idx = checkpoint.completed_steps
            results = checkpoint.results
            print(f"从第 {start_idx} 步恢复执行")
        else:
            start_idx = 0
            results = []
            checkpoint = TaskCheckpoint(
                task_id=task_id, total_steps=len(subtasks),
                completed_steps=0, current_step={}
            )

        # 2. 逐步执行
        for i, subtask in enumerate(subtasks[start_idx:], start=start_idx):
            checkpoint.current_step = {"index": i, "task": subtask}

            try:
                # 带超时的单步执行
                result = await asyncio.wait_for(
                    agent.execute_step(subtask),
                    timeout=self.timeout / len(subtasks)
                )
                results.append(result)
                checkpoint.completed_steps = i + 1
                checkpoint.results = results

                # 3. 定期保存检查点
                if (i + 1) % 5 == 0:
                    self.checkpoint_store.save(checkpoint)
                    self._report_progress(task_id, i + 1, len(subtasks))

            except asyncio.TimeoutError:
                checkpoint.status = "failed"
                checkpoint.current_step["error"] = "timeout"
                self.checkpoint_store.save(checkpoint)
                raise TaskTimeoutError(f"步骤 {i} 超时")

        checkpoint.status = "completed"
        self.checkpoint_store.save(checkpoint)
        return results

    def _report_progress(self, task_id, completed, total):
        """进度上报：写入 Redis/数据库，前端可轮询"""
        progress = {"task_id": task_id, "completed": completed, "total": total}
        self.checkpoint_store.set_progress(task_id, json.dumps(progress))


# 任务分解：把大任务拆成可独立执行的子任务
class TaskDecomposer:
    def decompose(self, complex_task: str, llm) -> list[dict]:
        """LLM 自动分解大任务"""
        prompt = f"""请将以下任务分解为独立的子任务（每个子任务可独立完成）：
        任务：{complex_task}
        输出格式：JSON 列表，每个元素包含 step_id, description, dependencies"""
        response = llm.chat(prompt)
        subtasks = json.loads(response)
        # 按依赖关系排序
        return self._topological_sort(subtasks)
```

**关键设计原则：**
1. **任务分解：** 大任务 → 小子任务，每个子任务有明确的输入输出
2. **检查点持久化：** 每 N 步保存一次，崩溃后可恢复
3. **超时分层：** 全局超时 + 单步超时 + API 调用超时
4. **异步执行：** 无依赖的子任务并行执行，有依赖的串行
5. **进度上报：** 用户能看到实时进度，而非"请稍候…"

**追问：**
- 如何处理子任务之间的依赖关系？用 DAG（有向无环图）建模，拓扑排序后执行，无依赖的任务并行。
- 检查点应该存在哪里？轻量级用 SQLite/本地文件，生产环境用 Redis 或 PostgreSQL。
- 如果 LLM 调用频繁超时怎么办？分级重试（快速重试 3 次 → 换模型重试 → 降级处理），配合指数退避。

---

### Q: 什么是 Tool Use 的 Parallel Function Calling？⭐⭐

**答：**

Parallel Function Calling（并行函数调用）是指模型在**一次响应中同时输出多个独立的工具调用**，应用层可以并行执行这些调用，从而显著降低端到端延迟。

**核心思想：** 当多个工具调用之间没有依赖关系时，为什么要串行等待？

```python
# 场例：用户问"北京和上海今天天气怎么样？"
# 串行（2次调用，总延迟 = 延迟1 + 延迟2）
weather_beijing = get_weather("北京")
weather_shanghai = get_weather("上海")

# 并行（2次调用，总延迟 = max(延迟1, 延迟2)）
weather_beijing, weather_shanghai = await asyncio.gather(
    get_weather("北京"),
    get_weather("上海")
)

# OpenAI Parallel Function Calling 示例
import openai

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京和上海今天天气如何？"}],
    tools=[weather_tool_schema],
    parallel_tool_calls=True  # 关键参数
)

# 响应中包含多个 tool_call
# message.tool_calls = [
#     {id: "call_1", function: {name: "get_weather", arguments: '{"city": "北京"}'}},
#     {id: "call_2", function: {name: "get_weather", arguments: '{"city": "上海"}'}}
# ]

# 应用层并行执行
async def execute_parallel_tool_calls(tool_calls):
    """并行执行模型返回的多个工具调用"""
    # 1. 依赖分析：哪些调用可以并行？
    #    简单策略：无数据依赖的调用全部并行
    independent_calls = []
    dependent_calls = []
    # 这里简化处理：假设所有调用独立
    independent_calls = tool_calls

    # 2. 并行执行
    tasks = [
        execute_single_tool(tc.function.name, json.loads(tc.function.arguments))
        for tc in independent_calls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 3. 构建工具结果消息
    tool_messages = []
    for tc, result in zip(tool_calls, results):
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(result) if not isinstance(result, Exception) else f"Error: {result}"
        })
    return tool_messages
```

**依赖分析与结果合并：**

```python
class ToolCallScheduler:
    """工具调用调度器：分析依赖，编排执行顺序"""

    def __init__(self):
        self.dependency_graph = {}

    def analyze_dependencies(self, tool_calls):
        """分析工具调用之间的依赖关系"""
        # 方法1：基于参数的静态分析
        # 如果 call_2 的输入依赖 call_1 的输出，则有依赖
        # 方法2：让 LLM 判断（更准确但更慢）
        groups = []  # 每组内的调用可以并行
        visited = set()

        for tc in tool_calls:
            if tc.id not in visited:
                # 找出所有无依赖的调用，放入同一组
                parallel_group = self._find_independent_calls(tc, tool_calls, visited)
                groups.append(parallel_group)
                visited.update(c.id for c in parallel_group)

        return groups  # 组间串行，组内并行

    async def execute_scheduled(self, groups):
        """按依赖顺序执行"""
        all_results = []
        for group in groups:
            # 组内并行
            results = await asyncio.gather(*[
                self._execute(tc) for tc in group
            ])
            all_results.extend(results)
        return all_results
```

**追问：**
- 所有模型都支持 Parallel Function Calling 吗？OpenAI GPT-4o 支持，Claude 也支持多工具调用，但不一定同时返回。
- 并行调用的副作用问题？如果并行调用中包含写操作（如"发送邮件"+"写入数据库"），需要考虑部分失败的补偿机制。
- 如何判断哪些调用可以并行？最简单的方案：同一工具的不同参数可以并行；不同工具的调用需要分析输入输出依赖。

---

## 更新后的总结

| 主题 | 核心要点 |
|------|----------|
| Agent 基础 | 自主感知-推理-行动-记忆循环 |
| ReAct | 思考-行动-观察交替，简单高效但无全局规划 |
| Plan-and-Execute | 先规划再执行，适合复杂任务，需配合动态重规划 |
| Function Calling | 模型输出结构化工具调用，应用层负责执行 |
| 错误恢复 | 分层处理：重试→参数修正→工具降级→跳过 |
| 状态管理 | Scratchpad + 检查点 + 持久化 |
| 实战核心 | 循环防护、上下文管理、成本控制、通信协议 |
| LATS | 树搜索 + MCTS 思想，多路径探索最优解 |
| AutoGPT | 目标驱动自主循环，需循环检测防止死循环 |
| 框架选型 | LangChain(通用链) / LlamaIndex(数据) / CrewAI(多Agent) |
| HITL | 人工审批/编辑/接管，LangGraph interrupt 实现 |
| Observability | Trace + Log + Metrics，LangSmith/Phoenix |
| 长任务 | 任务分解 + 检查点 + 超时分层 + 异步执行 |
| 并行工具调用 | 依赖分析 + 并行执行 + 结果合并 |
