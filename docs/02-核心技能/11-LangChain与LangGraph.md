# 11. LangChain 与 LangGraph

> **Agent 开发主流框架**：LangChain 生态最大，LangGraph 是复杂 Agent 的首选

---

## 一、LangChain 核心概念

### 1. ⭐⭐ Q: LangChain 的核心组件有哪些？

**答**：

```
LangChain 核心组件：
├── Models（模型）
│   ├── Chat Models —— 对话模型（GPT-4, Claude, Qwen）
│   ├── LLMs —— 文本补全模型（旧接口）
│   └── Embeddings —— 向量模型
│
├── Prompts（提示）
│   ├── PromptTemplate —— 模板
│   ├── ChatPromptTemplate —— 对话模板
│   └── FewShotPromptTemplate —— 少样本模板
│
├── Chains（链）
│   ├── LLMChain —— 基础链
│   ├── SequentialChain —— 顺序链
│   └── RouterChain —— 路由链
│
├── Memory（记忆）
│   ├── ConversationBufferMemory —— 完整历史
│   ├── ConversationSummaryMemory —— 摘要
│   └── ConversationBufferWindowMemory —— 滑动窗口
│
├── Retrievers（检索器）
│   ├── VectorStoreRetriever —— 向量检索
│   ├── BM25Retriever —— 关键词检索
│   └── EnsembleRetriever —— 混合检索
│
├── Tools（工具）
│   ├── @tool 装饰器
│   ├── BaseTool 基类
│   └── StructuredTool
│
└── Agents（代理）
    ├── ReAct Agent
    ├── OpenAI Functions Agent
    └── Plan-and-Execute Agent
```

---

### 2. ⭐⭐⭐ Q: LangChain Expression Language (LCEL) 是什么？

**答**：

LCEL 是 LangChain 的声明式组合语法，用 `|` 管道符连接组件：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 基础 LCEL 链
prompt = ChatPromptTemplate.from_template(
    "请用一句话解释什么是{topic}"
)
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 用 | 连接
chain = prompt | model | parser

# 执行
result = chain.invoke({"topic": "机器学习"})

# LCEL 的优势：
# 1. 自动支持 stream, batch, ainvoke
# 2. 自动重试和回退
# 3. 支持并行执行
# 4. 支持配置和调试

# 流式输出
for chunk in chain.stream({"topic": "机器学习"}):
    print(chunk, end="")

# 批量处理
results = chain.batch([
    {"topic": "机器学习"},
    {"topic": "深度学习"},
    {"topic": "强化学习"},
])

# 并行链
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    summary=prompt_summary | model | parser,
    keywords=prompt_keywords | model | parser,
    translation=prompt_translate | model | parser,
)

result = chain.invoke({"text": "..."})
# result = {"summary": "...", "keywords": "...", "translation": "..."}
```

---

### 3. ⭐⭐⭐ Q: 如何用 LangChain 实现 RAG？

**答**：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 文档加载和分块
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " "]
)
chunks = text_splitter.split_documents(documents)

# 2. 向量存储
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. RAG 链
template = """基于以下上下文回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文:
{context}

问题: {question}

回答:"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-4o-mini")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 4. 使用
answer = rag_chain.invoke("什么是机器学习？")

# 5. 带来源的 RAG
from langchain_core.runnables import RunnableParallel

rag_chain_with_sources = RunnableParallel(
    context=retriever | format_docs,
    question=RunnablePassthrough()
) | RunnableParallel(
    answer=prompt | model | StrOutputParser(),
    sources=retriever
)

result = rag_chain_with_sources.invoke("什么是机器学习？")
print(result["answer"])
print(result["sources"])  # 检索到的文档
```

---

### 4. ⭐⭐⭐ Q: LangChain 的 Tool 和 Agent 怎么实现？

**答**：

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent

# 1. 定义工具
@tool
def search_web(query: str) -> str:
    """搜索互联网获取最新信息"""
    results = web_search(query)
    return "\n".join(results)

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)  # 生产环境应使用安全的表达式解析
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    weather_data = fetch_weather(city)
    return f"{city}天气: {weather_data['condition']}, 温度: {weather_data['temp']}°C"

# 2. 创建 Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [search_web, calculate, get_weather]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手，可以使用工具来回答问题。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)

# 3. 创建 Agent 执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
)

# 4. 运行
result = agent_executor.invoke({
    "input": "北京今天天气怎么样？如果温度超过30度，帮我计算开空调8小时的电费（每度电0.5元，空调功率1.5kW）"
})
print(result["output"])
```

---

## 二、LangGraph 核心概念

### 5. ⭐⭐⭐ Q: LangGraph 的核心概念是什么？和 LangChain 有什么区别？

**答**：

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator

# 1. 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # 消息列表（自动累加）
    next_step: str                            # 下一步
    iterations: int                           # 迭代次数

# 2. 定义节点（函数）
def think(state: AgentState) -> AgentState:
    """思考节点"""
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "iterations": state["iterations"] + 1
    }

def use_tool(state: AgentState) -> AgentState:
    """工具节点"""
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    
    result = tools[tool_call["name"]].invoke(tool_call["args"])
    
    return {
        "messages": [ToolMessage(content=str(result), tool_call_id=tool_call["id"])]
    }

def should_continue(state: AgentState) -> str:
    """条件边"""
    last_message = state["messages"][-1]
    
    if state["iterations"] > 5:
        return "end"
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tool"
    return "end"

# 3. 构建图
graph = StateGraph(AgentState)

# 添加节点
graph.add_node("think", think)
graph.add_node("use_tool", use_tool)

# 设置入口
graph.set_entry_point("think")

# 添加边
graph.add_conditional_edges(
    "think",
    should_continue,
    {
        "use_tool": "use_tool",
        "end": END
    }
)
graph.add_edge("use_tool", "think")  # 工具调用后回到思考

# 4. 编译
app = graph.compile()

# 5. 运行
result = app.invoke({
    "messages": [HumanMessage(content="北京天气怎么样？")],
    "iterations": 0
})
```

**LangChain vs LangGraph**：

| 维度 | LangChain | LangGraph |
|------|-----------|-----------|
| 模型 | 链式（Chain） | 图（Graph） |
| 控制流 | 线性 | 任意（支持循环） |
| 状态 | 隐式 | 显式（TypedDict） |
| 适用 | 简单 RAG、单步任务 | 复杂 Agent、多步推理 |
| 可视化 | 困难 | 容易（可导出图） |

---

### 6. ⭐⭐⭐ Q: 如何用 LangGraph 实现复杂的多步 Agent？

**答**：

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, List
import operator

class ResearchState(TypedDict):
    """研究 Agent 状态"""
    topic: str                              # 研究主题
    search_results: List[str]              # 搜索结果
    analysis: str                           # 分析结果
    outline: str                            # 大纲
    draft: str                              # 初稿
    final: str                              # 终稿
    messages: Annotated[list, operator.add] # 消息历史
    next_step: str                          # 下一步

# 节点函数
def search(state: ResearchState) -> ResearchState:
    """搜索节点"""
    results = []
    queries = generate_search_queries(state["topic"])
    
    for query in queries:
        result = web_search(query)
        results.append(result)
    
    return {
        "search_results": results,
        "messages": [SystemMessage(content=f"搜索完成，找到 {len(results)} 条结果")]
    }

def analyze(state: ResearchState) -> ResearchState:
    """分析节点"""
    analysis = llm.invoke(
        f"分析以下搜索结果，提取关于'{state['topic']}'的关键信息：\n\n"
        + "\n".join(state["search_results"])
    )
    
    return {
        "analysis": analysis.content,
        "messages": [SystemMessage(content="分析完成")]
    }

def create_outline(state: ResearchState) -> ResearchState:
    """生成大纲"""
    outline = llm.invoke(
        f"基于以下分析，为'{state['topic']}'生成文章大纲：\n\n{state['analysis']}"
    )
    
    return {
        "outline": outline.content,
        "messages": [SystemMessage(content="大纲生成完成")]
    }

def write_draft(state: ResearchState) -> ResearchState:
    """撰写初稿"""
    draft = llm.invoke(
        f"基于以下大纲和分析，撰写详细文章：\n\n"
        f"大纲：\n{state['outline']}\n\n"
        f"分析：\n{state['analysis']}"
    )
    
    return {
        "draft": draft.content,
        "messages": [SystemMessage(content="初稿完成")]
    }

def review_draft(state: ResearchState) -> ResearchState:
    """审阅初稿"""
    review = llm.invoke(
        f"审阅以下文章，提出改进建议：\n\n{state['draft']}"
    )
    
    # 判断是否需要修改
    if "需要修改" in review.content or "建议修改" in review.content:
        return {
            "next_step": "revise",
            "messages": [SystemMessage(content=f"审阅建议：{review.content}")]
        }
    else:
        return {
            "next_step": "finalize",
            "messages": [SystemMessage(content="审阅通过")]
        }

def revise_draft(state: ResearchState) -> ResearchState:
    """修改初稿"""
    revised = llm.invoke(
        f"根据以下建议修改文章：\n\n"
        f"原文：\n{state['draft']}\n\n"
        f"建议：\n{state['messages'][-1].content}"
    )
    
    return {
        "draft": revised.content,
        "messages": [SystemMessage(content="修改完成")]
    }

def finalize(state: ResearchState) -> ResearchState:
    """定稿"""
    return {
        "final": state["draft"],
        "messages": [SystemMessage(content="文章完成")]
    }

# 条件边
def should_revise(state: ResearchState) -> str:
    if state.get("next_step") == "revise":
        return "revise"
    return "finalize"

# 构建图
graph = StateGraph(ResearchState)

# 添加节点
graph.add_node("search", search)
graph.add_node("analyze", analyze)
graph.add_node("outline", create_outline)
graph.add_node("draft", write_draft)
graph.add_node("review", review_draft)
graph.add_node("revise", revise_draft)
graph.add_node("finalize", finalize)

# 添加边
graph.set_entry_point("search")
graph.add_edge("search", "analyze")
graph.add_edge("analyze", "outline")
graph.add_edge("outline", "draft")
graph.add_edge("draft", "review")
graph.add_conditional_edges("review", should_revise, {
    "revise": "revise",
    "finalize": "finalize"
})
graph.add_edge("revise", "review")  # 修改后再审阅
graph.add_edge("finalize", END)

# 编译
app = graph.compile()

# 运行
result = app.invoke({
    "topic": "2025年AI发展趋势",
    "search_results": [],
    "analysis": "",
    "outline": "",
    "draft": "",
    "final": "",
    "messages": [],
    "next_step": ""
})

print(result["final"])
```

---

### 7. ⭐⭐⭐ Q: LangGraph 如何实现 Human-in-the-Loop？

**答**：

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

# 1. 创建带检查点的图
checkpointer = MemorySaver()

graph = StateGraph(AgentState)
# ... 添加节点和边 ...

app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute_dangerous_action"],  # 在执行前暂停
)

# 2. 运行到中断点
config = {"configurable": {"thread_id": "user-123"}}

result = app.invoke(
    {"messages": [HumanMessage(content="删除所有数据")]},
    config=config
)

# 3. 检查当前状态
current_state = app.get_state(config)
print(current_state.values)  # 当前状态
print(current_state.next)    # 下一个节点

# 4. 人工审核后继续
# 方式一：直接继续
app.invoke(None, config=config)

# 方式二：修改状态后继续
app.update_state(
    config,
    {"messages": [HumanMessage(content="用户确认删除")]}
)
app.invoke(None, config=config)

# 方式三：跳过危险操作
app.update_state(
    config,
    {"next_step": "safe_alternative"}
)
app.invoke(None, config=config)
```

---

### 8. ⭐⭐ Q: LangGraph 的持久化和恢复机制？

**答**：

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# SQLite 持久化
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# PostgreSQL 持久化
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/dbname"
)

# 编译时启用
app = graph.compile(checkpointer=checkpointer)

# 每次调用都需要 thread_id
config = {"configurable": {"thread_id": "conversation-123"}}

# 第一轮对话
result1 = app.invoke(
    {"messages": [HumanMessage(content="你好")]},
    config=config
)

# 第二轮对话（自动加载历史状态）
result2 = app.invoke(
    {"messages": [HumanMessage(content="继续之前的话题")]},
    config=config
)

# 查看历史
history = list(app.get_state_history(config))
for state in history:
    print(f"Step: {state.metadata['step']}, Messages: {len(state.values['messages'])}")

# 分支（fork）
# 从某个历史点创建新分支
fork_config = app.fork(
    config,
    source_thread_id="conversation-123",
    source_step=3
)
result = app.invoke(
    {"messages": [HumanMessage(content="换个方向")]},
    config=fork_config
)
```

---

### 9. ⭐⭐⭐ Q: CrewAI 的核心概念是什么？

**答**：

CrewAI 是多 Agent 协作框架，核心概念：

```python
from crewai import Agent, Task, Crew, Process

# 1. 定义 Agent（角色）
researcher = Agent(
    role="高级研究分析师",
    goal="发现关于{topic}的最新趋势和见解",
    backstory="你是一位经验丰富的研究分析师，擅长发现隐藏的模式和趋势",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool, web_scraper],
    llm=ChatOpenAI(model="gpt-4o")
)

writer = Agent(
    role="技术博客作家",
    goal="撰写引人入胜的技术文章",
    backstory="你是一位才华横溢的技术作家，能将复杂概念转化为易懂的文章",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(model="gpt-4o")
)

editor = Agent(
    role="资深编辑",
    goal="确保文章质量和准确性",
    backstory="你是一位严谨的编辑，注重细节和事实准确性",
    verbose=True,
    allow_delegation=True,  # 可以委派任务给其他 Agent
    llm=ChatOpenAI(model="gpt-4o")
)

# 2. 定义 Task（任务）
research_task = Task(
    description="研究{topic}的最新发展，包括关键技术、主要玩家、市场趋势",
    expected_output="详细的研究报告，包含数据和来源",
    agent=researcher
)

writing_task = Task(
    description="基于研究报告撰写一篇技术博客文章",
    expected_output="一篇 2000 字的技术博客，结构清晰，易于理解",
    agent=writer,
    context=[research_task]  # 依赖研究任务的结果
)

editing_task = Task(
    description="审阅和编辑博客文章，确保质量和准确性",
    expected_output="最终版本的博客文章，附带修改建议",
    agent=editor,
    context=[writing_task]
)

# 3. 组建 Crew（团队）
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,  # 顺序执行
    verbose=True,
    memory=True,  # 启用记忆
)

# 4. 执行
result = crew.kickoff(inputs={"topic": "大模型Agent"})
print(result)
```

**CrewAI vs LangGraph**：

| 维度 | CrewAI | LangGraph |
|------|--------|-----------|
| 核心概念 | 角色 + 任务 | 状态 + 图 |
| 协作模式 | 顺序/并行/层级 | 任意图 |
| 易用性 | 高（声明式） | 中（需理解图） |
| 灵活性 | 中 | 高 |
|| 适用场景 | 多角色协作 | 复杂工作流 |

---

## LangGraph 状态图深入

### StateGraph vs MessageGraph

```python
from langgraph.graph import StateGraph, MessageGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# ========== MessageGraph（旧版，仅消息列表） ==========
# 状态 = List[BaseMessage]，节点接收消息列表，返回新消息
msg_graph = MessageGraph()

def echo(state):
    return AIMessage(content=f"你说了: {state[-1].content}")

msg_graph.add_node("echo", echo)
msg_graph.set_entry_point("echo")
msg_graph.add_edge("echo", END)

app = msg_graph.compile()
result = app.invoke([HumanMessage(content="你好")])
# result = [HumanMessage(...), AIMessage("你说了: 你好")]

# ========== StateGraph（推荐，自定义状态） ==========
from typing import TypedDict, Annotated
import operator

class MyState(TypedDict):
    messages: Annotated[list, operator.add]  # 消息自动累加
    step_count: int
    metadata: dict

graph = StateGraph(MyState)

def process(state: MyState):
    return {
        "messages": [AIMessage(content="处理完毕")],
        "step_count": state["step_count"] + 1
    }

graph.add_node("process", process)
graph.set_entry_point("process")
graph.add_edge("process", END)

app = graph.compile()
result = app.invoke({
    "messages": [HumanMessage(content="你好")],
    "step_count": 0,
    "metadata": {}
})
```

**区别总结**：
| 特性 | MessageGraph | StateGraph |
|------|-------------|------------|
| 状态类型 | `List[BaseMessage]` | 自定义 TypedDict |
| 状态合并 | 追加消息 | 自定义 Reducer |
| 灵活性 | 低 | 高 |
| 推荐程度 | 已过时 | **推荐** |

---

### 条件分支 (conditional_edges)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class RouterState(TypedDict):
    input: str
    category: str
    result: str

# 分类节点：决定路由
def classify(state: RouterState) -> RouterState:
    category = llm.invoke(f"将以下内容分类为 [技术/商务/其他]: {state['input']}")
    return {"category": category.content.strip()}

# 不同处理节点
def handle_tech(state: RouterState) -> RouterState:
    return {"result": f"[技术处理] {state['input']}"}

def handle_business(state: RouterState) -> RouterState:
    return {"result": f"[商务处理] {state['input']}"}

def handle_general(state: RouterState) -> RouterState:
    return {"result": f"[通用处理] {state['input']}"}

# 路由函数：返回目标节点名
def route_by_category(state: RouterState) -> Literal["tech", "business", "general"]:
    mapping = {"技术": "tech", "商务": "business"}
    return mapping.get(state["category"], "general")

# 构建图
graph = StateGraph(RouterState)
graph.add_node("classify", classify)
graph.add_node("tech", handle_tech)
graph.add_node("business", handle_business)
graph.add_node("general", handle_general)

graph.set_entry_point("classify")

# 条件边：根据路由函数的结果选择下一个节点
graph.add_conditional_edges(
    "classify",               # 源节点
    route_by_category,        # 路由函数
    {
        "tech": "tech",       # 返回值 -> 目标节点
        "business": "business",
        "general": "general"
    }
)

graph.add_edge("tech", END)
graph.add_edge("business", END)
graph.add_edge("general", END)

app = graph.compile()
result = app.invoke({"input": "如何部署 Kubernetes？", "category": "", "result": ""})
```

---

### 人工审批 (interrupt_before / interrupt_after)

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator

class ApprovalState(TypedDict):
    messages: Annotated[list, operator.add]
    action: str
    approved: bool

def plan_action(state: ApprovalState):
    return {
        "messages": [AIMessage(content="计划: 删除过期数据")],
        "action": "delete_expired_data"
    }

def execute_action(state: ApprovalState):
    # 实际执行操作
    return {"messages": [AIMessage(content=f"已执行: {state['action']}")]}

def needs_approval(state: ApprovalState) -> str:
    if state.get("approved", False):
        return "approved"
    return "needs_review"

# 带人工审批的图
checkpointer = MemorySaver()
graph = StateGraph(ApprovalState)
graph.add_node("plan", plan_action)
graph.add_node("execute", execute_action)
graph.set_entry_point("plan")
graph.add_edge("plan", "execute")
graph.add_edge("execute", END)

app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute"],  # 执行前暂停，等待人工确认
    # interrupt_after=["plan"],     # 也可在某节点执行后暂停
)

# 第一次运行：在 execute 前暂停
config = {"configurable": {"thread_id": "approval-1"}}
state = app.invoke(
    {"messages": [HumanMessage(content="清理过期数据")], "action": "", "approved": False},
    config=config
)

# 查看暂停状态
snapshot = app.get_state(config)
print("暂停在:", snapshot.next)        # ('execute',)
print("当前状态:", snapshot.values)

# 人工审批后继续
app.update_state(config, {"approved": True})
result = app.invoke(None, config=config)  # None 表示继续执行
print(result)
```

---

### 子图 (Subgraph)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# ---- 子图：搜索流程 ----
class SearchState(TypedDict):
    query: str
    results: list
    messages: Annotated[list, operator.add]

def web_search_node(state: SearchState):
    results = ["搜索结果1", "搜索结果2"]
    return {"results": results, "messages": [AIMessage(content="搜索完成")]}

def filter_results(state: SearchState):
    filtered = state["results"][:1]  # 取第一条
    return {"results": filtered}

search_subgraph = StateGraph(SearchState)
search_subgraph.add_node("search", web_search_node)
search_subgraph.add_node("filter", filter_results)
search_subgraph.set_entry_point("search")
search_subgraph.add_edge("search", "filter")
search_subgraph.add_edge("filter", END)
search_app = search_subgraph.compile()

# ---- 主图：将子图作为一个节点 ----
class MainState(TypedDict):
    query: str
    results: list
    answer: str
    messages: Annotated[list, operator.add]

def research_node(state: MainState):
    # 调用子图
    sub_result = search_app.invoke({"query": state["query"], "results": [], "messages": []})
    return {
        "results": sub_result["results"],
        "messages": sub_result["messages"]
    }

def answer_node(state: MainState):
    answer = f"基于 {state['results']} 的回答"
    return {"answer": answer, "messages": [AIMessage(content=answer)]}

main_graph = StateGraph(MainState)
main_graph.add_node("research", research_node)    # 节点内部调用子图
main_graph.add_node("answer", answer_node)
main_graph.set_entry_point("research")
main_graph.add_edge("research", "answer")
main_graph.add_edge("answer", END)

# 也可以直接将子图作为节点添加
# main_graph.add_node("research", search_app)  # 直接传入编译后的子图

app = main_graph.compile()
result = app.invoke({"query": "AI Agent", "results": [], "answer": "", "messages": []})
print(result["answer"])
```

---

### 持久化 (Checkpointing)

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# ---- 内存检查点（开发/测试） ----
memory_checkpointer = MemorySaver()

# ---- SQLite 持久化（单机） ----
import aiosqlite
async def get_sqlite_checkpointer():
    conn = aiosqlite.connect("checkpoints.db")
    return AsyncSqliteSaver(conn)

# ---- PostgreSQL 持久化（生产） ----
async def get_pg_checkpointer():
    return AsyncPostgresSaver.from_conn_string(
        "postgresql://user:pass@localhost:5432/langgraph"
    )

# ---- 使用检查点 ----
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

class ChatState(TypedDict):
    messages: Annotated[list, operator.add]

def chatbot(state: ChatState):
    return {"messages": [AIMessage(content="你好！")]}

graph = StateGraph(ChatState)
graph.add_node("chat", chatbot)
graph.set_entry_point("chat")
graph.add_edge("chat", END)

app = graph.compile(checkpointer=memory_checkpointer)

# 线程隔离：不同 thread_id 互不干扰
config_1 = {"configurable": {"thread_id": "user-alice"}}
config_2 = {"configurable": {"thread_id": "user-bob"}}

app.invoke({"messages": [HumanMessage(content="我叫Alice")]}, config=config_1)
app.invoke({"messages": [HumanMessage(content="我叫Bob")]}, config=config_2)

# 查看历史状态
for snapshot in app.get_state_history(config_1):
    print(f"Step {snapshot.metadata.get('step', '?')}: {snapshot.values['messages'][-1].content}")

# 从某个 checkpoint 分支（fork）
# 获取第2步的状态
history = list(app.get_state_history(config_1))
fork_point = history[2]  # 第3个状态点
fork_config = {"configurable": {"thread_id": "alice-fork", **fork_point.config["configurable"]}}
app.invoke({"messages": [HumanMessage(content="换个思路")]}, config=fork_config)
```

---

## LCEL 自定义组件

### RunnablePassthrough 与 RunnableLambda

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

# ---- RunnablePassthrough：透传输入 ----
# 常用于并行链中，将原始输入传递到下游
chain = RunnablePassthrough()
result = chain.invoke({"key": "value"})
# result = {"key": "value"}  原样返回

# 配合 assign 添加新字段
chain = RunnablePassthrough.assign(
    upper=lambda x: x["text"].upper()
)
result = chain.invoke({"text": "hello"})
# result = {"text": "hello", "upper": "HELLO"}

# ---- RunnableLambda：包装任意函数 ----
def preprocess(text: str) -> str:
    return text.strip().lower()

def word_count(text: str) -> int:
    return len(text.split())

# 方式一：直接传函数
count_chain = RunnableLambda(word_count)
result = count_chain.invoke("Hello World")
# result = 2

# 方式二：带错误处理的 Lambda
def risky_function(input_data):
    if not input_data:
        raise ValueError("输入不能为空")
    return process(input_data)

safe_chain = RunnableLambda(risky_function).with_fallbacks(
    [RunnableLambda(lambda x: "默认输出")]
)

# 实际 RAG 场景中的组合使用
rag_chain = {
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
} | prompt | llm | StrOutputParser()
```

---

### 自定义 Runnable

```python
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.outputs import ChatGenerationChunk
from typing import Any, Optional, Iterator

class WordCounter(Runnable[str, dict]):
    """自定义 Runnable：统计词数和字数"""

    @property
    def InputType(self):
        return str

    @property
    def OutputType(self):
        return dict

    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> dict:
        return {
            "text": input,
            "word_count": len(input.split()),
            "char_count": len(input),
            "line_count": len(input.splitlines())
        }

    def stream(self, input: str, config: Optional[RunnableConfig] = None) -> Iterator[dict]:
        """流式输出：逐词返回统计"""
        words = input.split()
        for i, word in enumerate(words):
            yield {
                "word": word,
                "index": i,
                "partial_count": i + 1,
                "total": len(words)
            }

# 使用
counter = WordCounter()
result = counter.invoke("Hello World from LangChain")
# {"text": "...", "word_count": 4, "char_count": 27, "line_count": 1}

# 组合到链中
chain = counter | RunnableLambda(lambda d: f"共 {d['word_count']} 个词")
result = chain.invoke("Hello World")
# "共 2 个词"

# 批量处理
results = counter.batch(["Hello", "Hello World", "Hi there friend"])
```

---

### 链式调用原理 (invoke / batch / stream)

```python
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate

# ---- invoke: 单次调用 ----
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "AI"})  # 同步调用
# 调用链: prompt.invoke() -> llm.invoke() -> parser.invoke()

# ---- batch: 并行批量调用 ----
inputs = [{"topic": "AI"}, {"topic": "ML"}, {"topic": "DL"}]
results = chain.batch(inputs)
# 底层使用 asyncio.gather 并行执行，大幅提升吞吐

# 自定义并发控制
results = chain.batch(inputs, config={"max_concurrency": 2})

# ---- stream: 流式输出 ----
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="|")
# 每个 chunk 是部分输出，实现逐 token 返回

# ---- ainvoke / abatch / astream: 异步版本 ----
import asyncio

async def async_call():
    result = await chain.ainvoke({"topic": "AI"})
    results = await chain.abatch(inputs)

    async for chunk in chain.astream({"topic": "AI"}):
        print(chunk, end="")

asyncio.run(async_call())

# ---- 内部调度原理 ----
# pipe 运算符 | 调用 RunnableSequence.__or__()
# 创建 RunnableSequence(first, middle, last)
# invoke 时依次调用: first.invoke() -> middle.invoke() -> last.invoke()
# stream 时: first.stream() -> 逐 chunk -> middle.stream() -> ...
# batch 时: 使用 gather 并行调用每个 input 的 invoke
```

---

### 错误处理 (with_fallbacks / with_retry)

```python
from langchain_core.runnables import RunnableLambda
import time

# ---- with_fallbacks: 回退策略 ----
def primary_llm(input_data):
    """主模型（可能失败）"""
    raise ConnectionError("API 限流")

def fallback_llm(input_data):
    """备用模型"""
    return f"[Fallback] 处理结果: {input_data}"

# 主模型失败时自动切换到备用
chain = RunnableLambda(primary_llm).with_fallbacks(
    [RunnableLambda(fallback_llm)],
    exceptions_to_handle=(ConnectionError, TimeoutError)  # 只捕获特定异常
)
result = chain.invoke("test")
# result = "[Fallback] 处理结果: test"

# 多级回退
chain = (
    RunnableLambda(gpt4_call)           # 优先 GPT-4
    .with_fallbacks([
        RunnableLambda(claude_call),     # 回退 Claude
        RunnableLambda(local_llm_call),  # 再回退本地模型
    ])
)

# ---- with_retry: 重试策略 ----
def unreliable_api(input_data):
    if time.time() % 2 < 1:  # 随机失败
        raise ConnectionError("临时错误")
    return "成功"

chain = RunnableLambda(unreliable_api).with_retry(
    retry_if_exception_type=(ConnectionError,),
    wait_exponential_jitter=True,  # 指数退避 + 抖动
    stop_after_attempt=3,          # 最多重试3次
)
result = chain.invoke("test")

# ---- 组合使用：先重试，再回退 ----
chain = (
    RunnableLambda(primary_api)
    .with_retry(stop_after_attempt=3, retry_if_exception_type=(ConnectionError,))
    .with_fallbacks([RunnableLambda(fallback_api)])
)

# 在链中使用
from langchain_openai import ChatOpenAI

primary_model = ChatOpenAI(model="gpt-4o", max_retries=3)
fallback_model = ChatOpenAI(model="gpt-4o-mini")

model_with_fallback = primary_model.with_fallbacks([fallback_model])
robust_chain = prompt | model_with_fallback | StrOutputParser()
```

---

## LangChain vs LangGraph vs CrewAI 对比

### 架构对比表

| 维度 | LangChain | LangGraph | CrewAI |
|------|-----------|-----------|--------|
| **核心抽象** | Chain（链） | Graph（状态图） | Agent + Task + Crew |
| **控制流** | 线性管道（LCEL） | 任意图（含循环） | 顺序/并行/层级 |
| **状态管理** | 隐式（Memory） | 显式（TypedDict + Reducer） | 隐式（内置 Memory） |
| **多 Agent** | 需手动编排 | 图中多节点 | 原生支持（角色分工） |
| **人工介入** | 回调方式 | interrupt_before/after | 无原生支持 |
| **持久化** | 无原生支持 | Checkpointing（SQLite/PG） | 内置 Memory |
| **可视化** | 困难 | Mermaid/Graphviz 导出 | 无 |
| **学习曲线** | 中 | 高（需理解图概念） | 低（声明式） |
| **生态** | 最大（集成多） | 同 LangChain 生态 | 独立生态 |
| **适用规模** | 中小 | 大（生产级） | 中小 |
| **流式支持** | LCEL 原生 | 原生（含 token 级） | 有限 |
| **调试能力** | LangSmith | LangSmith + 图可视化 | 内置 verbose |

### 适用场景

```
场景选择指南：

1. 简单 RAG / 文档问答
   → LangChain（LCEL 链即可）

2. 单 Agent + 工具调用
   → LangChain AgentExecutor 或 LangGraph

3. 复杂多步推理 Agent（需循环/分支）
   → LangGraph（唯一选择）

4. 需要人工审批的流程
   → LangGraph（interrupt 机制）

5. 多角色协作（如：研究员+写手+编辑）
   → CrewAI（最简单）或 LangGraph（更灵活）

6. 生产级部署 + 可观测性
   → LangGraph + LangSmith

7. 快速原型 / MVP
   → CrewAI（最快上手）

8. 自定义复杂工作流
   → LangGraph（子图 + 条件边 + 持久化）
```

### 选型建议

```python
# ---- 决策流程 ----
"""
1. 你的任务是否需要循环/条件分支/人工审批？
   ├─ 是 → LangGraph
   └─ 否 → 继续判断

2. 是否需要多 Agent 角色协作？
   ├─ 是 → 需要精细控制？
   │   ├─ 是 → LangGraph
   │   └─ 否 → CrewAI（更快上手）
   └─ 否 → 继续判断

3. 是否为简单的链式调用（RAG / 分类 / 翻译）？
   ├─ 是 → LangChain（LCEL）
   └─ 否 → LangGraph（通用选择）
"""

# ---- 混合使用示例 ----
# LangGraph 中使用 LangChain 组件
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o")
tools = [search_tool, calculator_tool]

# 一行代码创建 ReAct Agent（LangGraph 预构建）
agent = create_react_agent(llm, tools)
result = agent.invoke({"messages": [HumanMessage(content="...")]})

# CrewAI Agent 包装为 LangGraph 节点
def crew_node(state):
    crew = Crew(agents=[...], tasks=[...], process=Process.sequential)
    result = crew.kickoff(inputs={"topic": state["topic"]})
    return {"result": result}

graph = StateGraph(MyState)
graph.add_node("crew_research", crew_node)  # CrewAI 作为图的一个节点
graph.add_node("human_review", review_node)
graph.add_edge("crew_research", "human_review")
```

**总结**：
- **LangChain**：基础设施层，提供模型/工具/检索等组件
- **LangGraph**：编排层，处理复杂控制流，生产级 Agent 首选
- **CrewAI**：应用层，多角色协作场景最快落地
- 三者不互斥，可在同一项目中混合使用
