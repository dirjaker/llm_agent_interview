# 39 - Agent 开发实战

---

## 一、ReAct 框架

### Q1：什么是 ReAct 框架？它与 Chain-of-Thought 有什么区别？ ⭐⭐

**ReAct = Reasoning + Acting**，由 Yao et al. (2022) 提出，核心思想是让 LLM **交替进行推理（Thought）和行动（Action）**，而不是只做纯推理或纯行动。

**与 Chain-of-Thought (CoT) 的区别：**

| 维度 | Chain-of-Thought | ReAct |
|------|-------------------|-------|
| 核心 | 只推理（Thought） | 推理 + 行动（Thought + Action） |
| 能否使用工具 | 不能 | 能 |
| 能否获取外部信息 | 不能（纯靠模型知识） | 能（通过工具查询实时数据） |
| 典型场景 | 数学推理、逻辑推演 | 需要外部信息的复杂任务 |

**ReAct 循环：**

```
Thought: 我需要查找北京今天的天气
Action: search("北京今天天气")
Observation: 北京今天晴，最高温度 32°C
Thought: 已经获取到天气信息，可以回答用户了
Action: finish("北京今天天气晴朗，最高温度32°C")
```

---

### Q2：请设计一个 ReAct 的 Prompt 模板 ⭐⭐

```python
REACT_PROMPT = """你是一个智能助手，可以使用以下工具：

{tool_descriptions}

请按照以下格式回答问题：

Thought: 分析当前情况，决定下一步
Action: 工具名[参数]
Observation: 工具返回的结果
... (可以重复 Thought/Action/Observation)
Thought: 分析最终结果
Final Answer: 最终回答

注意：
1. 每次只能调用一个工具
2. 等待 Observation 后再做下一步 Thought
3. 确定答案后用 Final Answer 给出最终回答

问题: {question}
"""
```

**追问：ReAct 和 Function Calling 的关系？**

ReAct 是**框架/范式**（定义了推理和行动的循环模式），Function Calling 是**实现机制**（模型输出结构化的工具调用 JSON）。现代实践中，Function Calling 取代了 ReAct 中手动解析 `Action: tool[arg]` 的方式，但思维模式仍是 ReAct。

---

### Q3：请用 Python 实现一个完整的 ReAct Agent ⭐⭐⭐

```python
import openai
import json
import re
from typing import Callable, Any

class ReactAgent:
    """基于 ReAct 框架的 Agent"""

    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI()
        self.model = model
        self.tools: dict[str, Callable] = {}
        self.max_iterations = 10

    def register_tool(self, name: str, func: Callable, description: str):
        self.tools[name] = {"func": func, "description": description}

    def _build_tool_descriptions(self) -> str:
        lines = []
        for name, info in self.tools.items():
            lines.append(f"- {name}: {info['description']}")
        return "\n".join(lines)

    def _build_prompt(self, question: str, scratchpad: str) -> str:
        return f"""你是一个智能助手，可以使用以下工具：

{self._build_tool_descriptions()}

请按照 Thought / Action / Observation 的格式逐步解决问题。
Action 格式: 工具名[参数]
确定答案后输出: Final Answer: 你的答案

问题: {question}
{scratchpad}
Thought:"""

    def _parse_action(self, text: str) -> tuple[str, str] | None:
        match = re.search(r"Action:\s*(\w+)\[(.+?)\]", text)
        if match:
            return match.group(1), match.group(2)
        return None

    def run(self, question: str) -> str:
        scratchpad = ""
        for i in range(self.max_iterations):
            prompt = self._build_prompt(question, scratchpad)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            output = resp.choices[0].message.content.strip()
            scratchpad += f"\nThought: {output}\n"

            # 检查是否有最终答案
            if "Final Answer:" in output:
                return output.split("Final Answer:")[-1].strip()

            # 解析并执行 Action
            action = self._parse_action(output)
            if action is None:
                scratchpad += "Observation: 未识别到有效动作，请重试\n"
                continue

            tool_name, tool_input = action
            if tool_name not in self.tools:
                scratchpad += f"Observation: 工具 {tool_name} 不存在\n"
                continue

            try:
                result = self.tools[tool_name]["func"](tool_input)
            except Exception as e:
                result = f"工具执行错误: {e}"

            scratchpad += f"Observation: {result}\n"

        return "达到最大迭代次数，未能得出最终答案"


# 使用示例
def search(query: str) -> str:
    return f"搜索结果: {query}的相关信息..."

agent = ReactAgent()
agent.register_tool("search", search, "搜索互联网获取信息")
answer = agent.run("Python 3.12 有什么新特性？")
```

---

## 二、Function Calling / Tool Use

### Q4：Function Calling 的工作原理是什么？ ⭐⭐

Function Calling 的工作流程：

```
1. 开发者定义工具列表（JSON Schema 格式）
2. 用户发送消息 + 工具列表 → LLM
3. LLM 判断是否需要调用工具：
   - 不需要 → 直接生成文本回复
   - 需要 → 输出结构化的工具调用 JSON（函数名 + 参数）
4. 开发者执行工具调用，将结果返回给 LLM
5. LLM 根据工具结果生成最终回复
```

**关键点：LLM 本身不执行函数，只是生成调用指令，由外部代码执行。**

---

### Q5：如何使用 OpenAI Function Calling API？请写出完整代码 ⭐⭐

```python
import openai
import json

client = openai.OpenAI()

# 1. 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 2. 工具实现
def get_weather(city: str, unit: str = "celsius") -> str:
    return json.dumps({"city": city, "temp": 28, "unit": unit, "condition": "晴"})

# 3. 对话循环
messages = [{"role": "user", "content": "北京今天天气怎么样？"}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto"  # auto / none / required / 指定某个工具
)

msg = response.choices[0].message

# 4. 处理工具调用
if msg.tool_calls:
    messages.append(msg)
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        result = get_weather(**args)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result
        })
    # 5. 让模型根据工具结果生成最终回复
    final = client.chat.completions.create(
        model="gpt-4o", messages=messages, tools=tools
    )
    print(final.choices[0].message.content)
```

---

### Q6：工具描述（JSON Schema）的最佳实践有哪些？ ⭐⭐

```python
# ❌ 差的工具描述
{
    "name": "search",
    "description": "搜索",
    "parameters": {
        "type": "object",
        "properties": {
            "q": {"type": "string"}
        }
    }
}

# ✅ 好的工具描述
{
    "name": "search_products",
    "description": "在商品数据库中搜索商品。当用户询问商品信息、价格、库存时使用此工具。不适用于订单查询。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，如商品名称或类别"
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "food"],
                "description": "商品类别筛选（可选）"
            },
            "max_results": {
                "type": "integer",
                "description": "返回结果数量，默认10，最大50"
            }
        },
        "required": ["query"]
    }
}
```

**最佳实践清单：**
1. **命名清晰**：用 `search_products` 而非 `search` 或 `func1`
2. **描述要说明用途和边界**：何时用、何时不用
3. **每个参数都有 description**
4. **用 enum 约束可选值**
5. **标注 required 字段**
6. **描述中包含示例值**

---

### Q7：什么是并行工具调用？如何处理？ ⭐⭐

并行工具调用指 LLM 在一次响应中同时请求调用多个工具（无需等待前一个完成）。

```python
# LLM 返回多个 tool_calls
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    parallel_tool_calls=True  # 默认开启
)

msg = response.choices[0].message
# msg.tool_calls 可能包含多个调用：
# [tool_call_1, tool_call_2, tool_call_3]

# 并行执行
import asyncio

async def execute_tool_calls(tool_calls):
    async def run_one(tc):
        args = json.loads(tc.function.arguments)
        result = await call_tool_async(tc.function.name, args)
        return {"role": "tool", "tool_call_id": tc.id, "content": result}

    tasks = [run_one(tc) for tc in tool_calls]
    return await asyncio.gather(*tasks)
```

---

### Q8：工具调用返回错误怎么办？ ⭐⭐⭐

```python
def handle_tool_call_with_retry(messages, tools, max_tool_retries=2):
    """工具调用错误处理策略"""
    for attempt in range(max_tool_retries + 1):
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=tools
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        messages.append(msg)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
                result = execute_tool(tc.function.name, args)
                # 将结果返回给模型
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result)
                })
            except json.JSONDecodeError:
                # 参数解析失败 → 告知模型参数格式错误
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "错误: 参数格式不合法，请检查 JSON 格式后重试"
                })
            except Exception as e:
                # 工具执行失败 → 告知模型错误信息
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"错误: {type(e).__name__}: {str(e)}"
                })

    return "工具调用多次失败，请人工介入。"
```

**关键原则：**
- 将错误信息**原样返回**给模型，让模型自行修正
- 区分**可重试错误**（网络超时）和**不可重试错误**（参数无效）
- 限制重试次数，避免无限循环

**追问：如何让模型知道什么时候该调用工具？**

1. **工具描述中写清触发条件**：如"当用户询问天气时使用"
2. **在 system prompt 中给出指引**：如"遇到需要查询实时数据的问题时使用工具"
3. **tool_choice 参数控制**：`"auto"` 让模型自主判断，`"required"` 强制调用，或指定具体工具

---

## 三、工具调用死循环处理 ⭐⭐⭐（重点）

### Q9：工具调用死循环产生的原因有哪些？

```python
# 场景 1：工具返回结果不满足预期 → 模型反复重试
# Thought: 搜索结果不够详细，我再搜一次
# Action: search("Python教程")
# Observation: 返回10条结果
# Thought: 还是不够，再搜一次  ← 无限循环

# 场景 2：工具描述歧义 → 模型选错工具
# 用户问"今天几号"
# Action: search("今天几号")  ← 应该用 get_date，但描述不清

# 场景 3：多工具循环依赖
# Action: get_user_info(123) → 需要 department_id
# Action: get_department("engineering") → 需要 user_id
# Action: get_user_info(123) → 循环...
```

---

### Q10：请实现一个完整的防死循环 Agent 框架 ⭐⭐⭐

```python
from dataclasses import dataclass, field
from collections import Counter
import time
import json

@dataclass
class AgentConfig:
    max_iterations: int = 15           # 硬上限
    max_consecutive_errors: int = 3    # 连续错误上限
    max_same_call_repeat: int = 2      # 相同调用重复上限
    max_total_tokens: int = 50000      # token 预算
    max_tool_calls: int = 30           # 工具调用总次数上限
    timeout_seconds: int = 120         # 超时时间

@dataclass
class LoopDetector:
    """死循环检测器"""
    call_history: list = field(default_factory=list)
    consecutive_errors: int = 0
    total_tokens: int = 0
    total_tool_calls: int = 0

    def record_call(self, tool_name: str, args: str, success: bool):
        key = f"{tool_name}:{args}"
        self.call_history.append(key)
        self.total_tool_calls += 1
        if success:
            self.consecutive_errors = 0
        else:
            self.consecutive_errors += 1

    def check_loop(self, config: AgentConfig) -> str | None:
        """检查是否触发死循环，返回 None 表示正常，否则返回原因"""

        # 1. 连续错误检测
        if self.consecutive_errors >= config.max_consecutive_errors:
            return f"连续错误 {self.consecutive_errors} 次，停止执行"

        # 2. 相同调用重复检测
        if len(self.call_history) >= config.max_same_call_repeat:
            recent = self.call_history[-config.max_same_call_repeat:]
            if len(set(recent)) == 1:
                return f"相同工具调用重复 {config.max_same_call_repeat} 次，疑似死循环"

        # 3. Token 预算检测
        if self.total_tokens >= config.max_total_tokens:
            return f"Token 消耗已达上限 ({config.total_tokens})"

        # 4. 工具调用总数检测
        if self.total_tool_calls >= config.max_tool_calls:
            return f"工具调用次数已达上限 ({self.total_tool_calls})"

        return None

class SafeAgent:
    """带防死循环保护的 Agent"""

    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.client = openai.OpenAI()
        self.tools = {}
        self.detector = LoopDetector()

    def run(self, messages: list, tools_schema: list) -> str:
        start_time = time.time()

        for iteration in range(self.config.max_iterations):
            # 超时检测
            if time.time() - start_time > self.config.timeout_seconds:
                return "⏰ 执行超时，请人工介入"

            # 死循环检测
            loop_reason = self.detector.check_loop(self.config)
            if loop_reason:
                return f"🛑 死循环检测: {loop_reason}"

            # 调用模型
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools_schema
            )
            msg = resp.choices[0].message
            self.detector.total_tokens += resp.usage.total_tokens

            # 无工具调用 → 返回最终回答
            if not msg.tool_calls:
                return msg.content

            messages.append(msg)
            for tc in msg.tool_calls:
                args_str = tc.function.arguments
                try:
                    args = json.loads(args_str)
                    result = self._execute_tool(tc.function.name, args)
                    success = True
                except Exception as e:
                    result = f"错误: {e}"
                    success = False

                self.detector.record_call(tc.function.name, args_str, success)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result)
                })

        return f"达到最大迭代次数 ({self.config.max_iterations})，停止执行"
```

**追问：生产环境中 Agent 的最大迭代次数设多少合适？**

| 场景 | 建议值 | 理由 |
|------|--------|------|
| 简单问答 | 3-5 | 大多 1-2 轮就能解决 |
| 通用助手 | 8-15 | 需要多轮工具调用 |
| 复杂工作流 | 15-25 | 可能需要大量步骤 |
| 开放式探索 | 20-30 | 但要搭配严格的退出条件 |

**实际建议：不只靠 max_iterations，更要靠多层防护（循环检测 + 错误累积 + token 预算）。**

---

## 四、Agent 状态管理

### Q11：Agent 的短期记忆和长期记忆有什么区别？ ⭐⭐

| 维度 | 短期记忆 | 长期记忆 |
|------|----------|----------|
| 存储位置 | Context Window（对话历史） | 外部存储（数据库、向量库） |
| 生命周期 | 当前会话 | 跨会话持久化 |
| 容量 | 受 context window 限制 | 无限制 |
| 访问速度 | 快（直接在 prompt 中） | 慢（需要检索） |
| 典型实现 | messages 列表 | Redis / SQLite / 向量数据库 |

```python
class AgentMemory:
    def __init__(self, max_context_tokens=8000):
        self.short_term = []        # 对话历史
        self.max_context = max_context_tokens
        self.long_term_db = None    # 向量数据库连接

    def add_message(self, msg: dict):
        self.short_term.append(msg)
        # 超出窗口时压缩
        if self._count_tokens() > self.max_context:
            self._compress()

    def _compress(self):
        """压缩策略: 保留 system + 最近N轮 + 中间部分摘要"""
        system = [m for m in self.short_term if m["role"] == "system"]
        recent = self.short_term[-6:]  # 保留最近3轮
        old = self.short_term[len(system):-6]

        # 对旧消息做摘要
        summary = self._summarize(old)
        summary_msg = {"role": "system", "content": f"历史摘要: {summary}"}

        # 存入长期记忆
        self._store_long_term(old)

        self.short_term = system + [summary_msg] + recent

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """从长期记忆中检索相关信息"""
        # 向量检索实现
        return self.long_term_db.similarity_search(query, k=top_k)
```

---

### Q12：Agent 执行到一半断了怎么恢复？ ⭐⭐⭐

```python
import json
import sqlite3
from datetime import datetime

class AgentCheckpoint:
    """Agent 检查点 - 支持断点恢复"""

    def __init__(self, db_path: str = "agent_state.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                task_id TEXT PRIMARY KEY,
                messages TEXT NOT NULL,
                tool_calls_history TEXT,
                step_index INTEGER,
                status TEXT DEFAULT 'running',
                created_at TEXT,
                updated_at TEXT
            )
        """)

    def save(self, task_id: str, messages: list, step: int, tool_history: list):
        """每步保存检查点"""
        self.conn.execute(
            """INSERT OR REPLACE INTO checkpoints
               VALUES (?, ?, ?, ?, 'running', ?, ?)""",
            (task_id, json.dumps(messages), json.dumps(tool_history),
             step, datetime.now().isoformat(), datetime.now().isoformat())
        )
        self.conn.commit()

    def load(self, task_id: str) -> dict | None:
        """加载检查点用于恢复"""
        row = self.conn.execute(
            "SELECT messages, tool_calls_history, step_index FROM checkpoints WHERE task_id=?",
            (task_id,)
        ).fetchone()
        if row:
            return {
                "messages": json.loads(row[0]),
                "tool_history": json.loads(row[1]),
                "step": row[2]
            }
        return None

# 恢复逻辑
def resume_agent(task_id: str):
    cp = AgentCheckpoint()
    state = cp.load(task_id)
    if state:
        print(f"从步骤 {state['step']} 恢复执行")
        agent.run_from(state["messages"], state["step"])
    else:
        print("无检查点，从头开始")
```

---

## 五、多 Agent 编排

### Q13：多 Agent 编排有哪些模式？各自的适用场景是什么？ ⭐⭐⭐

#### 1. Supervisor 模式（管理者模式）

```
              ┌──────────┐
              │Supervisor│  ← 决策者，分配任务
              └────┬─────┘
         ┌─────────┼─────────┐
         ▼         ▼         ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │Agent A │ │Agent B │ │Agent C │
    │(搜索)  │ │(编码)  │ │(分析)  │
    └────────┘ └────────┘ └────────┘
```

```python
def supervisor_node(state):
    """Supervisor 决定下一步由哪个 agent 处理"""
    response = llm.invoke(
        f"""你是一个任务管理者。根据当前状态决定下一步行动。
        可用的 agent: {list(agents.keys())}
        当前状态: {state}
        回复一个 agent 名称，或 FINISH 表示完成。"""
    )
    return response.strip()  # "search_agent" / "code_agent" / "FINISH"
```

**适用场景：** 任务可明确拆分、需要中央协调

#### 2. Peer 模式（对等协商）

```
    ┌────────┐    ┌────────┐    ┌────────┐
    │Agent A │◄──►│Agent B │◄──►│Agent C │
    └────────┘    └────────┘    └────────┘
       平等协商，轮流发言，最终达成共识
```

**适用场景：** 辩论、头脑风暴、需要多方意见的场景

#### 3. Pipeline 模式（流水线）

```
    ┌────────┐   ┌────────┐   ┌────────┐
    │Agent A │──►│Agent B │──►│Agent C │──► 最终结果
    │(收集)  │   │(处理)  │   │(输出)  │
    └────────┘   └────────┘   └────────┘
```

**适用场景：** 任务有明确的先后顺序（如：搜索 → 分析 → 撰写报告）

#### 对比总结

| 模式 | 控制方式 | 灵活性 | 复杂度 | 适用场景 |
|------|----------|--------|--------|----------|
| Supervisor | 中心化 | 高 | 中 | 任务可拆分，需协调 |
| Peer | 去中心化 | 最高 | 高 | 多角度分析、辩论 |
| Pipeline | 线性 | 低 | 低 | 固定流程、顺序执行 |

---

### Q14：用 LangGraph 如何实现 Supervisor 模式？ ⭐⭐⭐

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class AgentState(TypedDict):
    messages: list
    next_agent: str
    task: str

def supervisor(state: AgentState) -> AgentState:
    """Supervisor 节点：决定下一步"""
    response = llm.invoke(
        f"任务: {state['task']}\n"
        f"当前进度: {state['messages'][-1] if state['messages'] else '刚开始'}\n"
        f"选择下一个执行者: research / code / write / FINISH"
    )
    return {**state, "next_agent": response.strip()}

def researcher(state: AgentState) -> AgentState:
    """搜索 Agent"""
    result = search_agent.invoke(state["task"])
    return {**state, "messages": state["messages"] + [f"搜索结果: {result}"]}

def coder(state: AgentState) -> AgentState:
    """编码 Agent"""
    result = code_agent.invoke(state["task"])
    return {**state, "messages": state["messages"] + [f"代码: {result}"]}

def writer(state: AgentState) -> AgentState:
    """写作 Agent"""
    result = write_agent.invoke(state["task"])
    return {**state, "messages": state["messages"] + [f"报告: {result}"]}

def route(state: AgentState) -> Literal["researcher", "coder", "writer", "__end__"]:
    return state["next_agent"]

# 构建图
graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("coder", coder)
graph.add_node("writer", writer)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route)
graph.add_edge("researcher", "supervisor")
graph.add_edge("coder", "supervisor")
graph.add_edge("writer", "supervisor")

app = graph.compile()
result = app.invoke({"messages": [], "next_agent": "", "task": "分析并报告..."})
```

**追问：多 Agent 系统最大的挑战是什么？**

1. **通信开销**：Agent 之间传递信息消耗大量 token
2. **错误传播**：一个 Agent 出错，后续全部受影响
3. **状态一致性**：多个 Agent 共享状态时容易冲突
4. **调试困难**：非确定性 + 多步骤 = 难以复现问题
5. **成本控制**：多个 Agent 同时调用 LLM，成本快速膨胀

---

## 六、Agent 评估

### Q15：如何评估 Agent 的表现？ ⭐⭐⭐

**Agent 评估的难点：**
- 非确定性：同样的输入可能走不同路径
- 多步骤：中间步骤的错误可能被放大
- 工具依赖：外部工具的行为不可控

**评估维度：**

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| 任务完成率 | 最终是否正确完成 | 正确完成数 / 总任务数 |
| 步骤效率 | 平均用了多少步 | 总步骤数 / 任务数 |
| 工具调用准确率 | 工具选择和参数是否正确 | 正确调用数 / 总调用数 |
| 成本效率 | 每个任务消耗多少 token | 总 token / 任务数 |
| 平均延迟 | 任务完成时间 | 总时间 / 任务数 |

**评估方式：**

```python
class AgentEvaluator:
    def evaluate(self, agent, test_cases: list) -> dict:
        results = {
            "total": len(test_cases),
            "completed": 0,
            "correct": 0,
            "total_steps": 0,
            "total_tokens": 0,
            "tool_accuracy": {"correct": 0, "total": 0}
        }

        for case in test_cases:
            output = agent.run(case["input"])
            stats = agent.get_run_stats()

            results["total_steps"] += stats["steps"]
            results["total_tokens"] += stats["tokens"]

            if output is not None:
                results["completed"] += 1

            # 自动评估：精确匹配 / LLM-as-Judge
            if self._check_answer(output, case["expected"]):
                results["correct"] += 1

        results["completion_rate"] = results["completed"] / results["total"]
        results["accuracy"] = results["correct"] / results["total"]
        results["avg_steps"] = results["total_steps"] / results["total"]
        return results

    def _check_answer(self, output, expected):
        """使用 LLM 作为评判者"""
        judge_prompt = f"判断以下回答是否正确。\n预期: {expected}\n实际: {output}\n回答 yes/no"
        judge_resp = llm.invoke(judge_prompt)
        return "yes" in judge_resp.lower()
```

**追问：如何评估生产环境中的 Agent？**

1. **在线指标**：用户满意度评分、任务完成率、人工介入率
2. **日志分析**：记录每步 Thought/Action/Observation，定期抽样审查
3. **A/B 测试**：新版本 vs 旧版本在真实流量上的表现对比
4. **告警机制**：异常高迭代次数、高错误率自动告警
5. **成本监控**：跟踪每个会话的 token 消耗，设置预算上限

---

## 七、Agent 开发面试高频题汇总

### Q16：什么是 Agent？和普通的 LLM 应用有什么区别？ ⭐

**Agent = LLM + 记忆 + 工具 + 规划能力**。普通 LLM 应用只能做单轮问答，Agent 能自主规划、调用工具、根据结果调整策略、多步完成复杂任务。

---

### Q17：ReAct、Plan-and-Execute、Reflexion 三种 Agent 架构有什么区别？ ⭐⭐

| 架构 | 核心思路 | 适用场景 |
|------|----------|----------|
| ReAct | 交替思考和行动（边想边做） | 通用任务 |
| Plan-and-Execute | 先规划完整计划，再逐步执行 | 步骤明确的复杂任务 |
| Reflexion | 执行后自我反思、总结失败经验 | 需要迭代改进的任务 |

---

### Q18：如何防止 Agent 调用危险工具（如删除文件）？ ⭐⭐

1. **权限分级**：将工具分为只读/写入/危险三类
2. **白名单/黑名单**：限制可调用的工具范围
3. **人工确认**：危险操作前暂停等待用户确认
4. **沙箱执行**：工具在隔离环境中运行
5. **输入校验**：对工具参数做安全检查（如防止 SQL 注入）

---

### Q19：如何处理 Agent 的 Context Window 溢出问题？ ⭐⭐

1. **对话摘要**：旧对话压缩为摘要
2. **滑动窗口**：只保留最近 N 轮对话
3. **重要信息提取**：关键信息存入长期记忆
4. **工具结果压缩**：长结果只保留关键部分
5. **RAG 检索**：历史信息存入向量库，按需检索

---

### Q20：Tool Calling 和 JSON Mode 有什么区别？ ⭐⭐

| 维度 | Tool Calling | JSON Mode |
|------|-------------|-----------|
| 目的 | 调用外部函数 | 输出结构化 JSON |
| 结构 | 函数名 + 参数（固定格式） | 任意 JSON |
| 是否执行代码 | 是（由外部执行） | 否（纯输出） |
| 典型用途 | Agent 工具调用 | 信息提取、分类 |

---

### Q21：多 Agent 通信有哪些方式？ ⭐⭐

1. **共享状态**：所有 Agent 读写同一个状态对象（LangGraph 方式）
2. **消息传递**：Agent 之间通过消息队列通信
3. **黑板模式**：共享"黑板"，Agent 自行读取和写入
4. **函数调用**：一个 Agent 直接调用另一个 Agent 作为工具

---

### Q22：如何降低 Agent 的运行成本？ ⭐⭐

1. **小模型路由**：简单任务用小模型，复杂任务用大模型
2. **缓存**：相同工具调用结果缓存
3. **减少无用工具**：精简工具列表，减少模型选择负担
4. **提前终止**：设置合理的退出条件
5. **Prompt 优化**：减少 system prompt 中的冗余描述

---

### Q23：什么是 Agentic RAG？和普通 RAG 有什么区别？ ⭐⭐

| 维度 | 普通 RAG | Agentic RAG |
|------|----------|-------------|
| 流程 | 固定：查询 → 检索 → 生成 | 动态：Agent 自主决定检索策略 |
| 检索次数 | 1 次 | 可多轮检索、改写查询 |
| 数据源 | 单一向量库 | 可切换多个数据源 |
| 自我纠错 | 无 | 检索结果不满意可重新检索 |

---

### Q24：MCP (Model Context Protocol) 是什么？解决什么问题？ ⭐⭐

MCP 是 Anthropic 提出的标准化协议，解决**工具接入碎片化**问题。类似于 USB-C 标准：工具开发者只需实现一次 MCP Server，所有支持 MCP 的 Agent 客户端都能直接使用，无需为每个 Agent 单独适配。

---

### Q25：生产环境部署 Agent 需要注意什么？ ⭐⭐⭐

1. **超时与熔断**：单次请求超时、工具服务不可用时快速失败
2. **成本控制**：设置 token 上限、并发限制
3. **日志与可观测性**：记录每步 Thought/Action/Observation
4. **降级策略**：Agent 不可用时降级为简单问答
5. **并发隔离**：不同用户的 Agent 会话互不干扰
6. **安全审计**：工具调用记录可追溯，敏感操作需审批

---

> **本章总结：** Agent 开发的核心是平衡**能力**与**安全**——既要让 Agent 自主完成复杂任务，又要通过防死循环、权限控制、成本预算等机制确保可控。面试重点：ReAct 原理、Function Calling 实现、死循环防护策略、多 Agent 编排模式。
