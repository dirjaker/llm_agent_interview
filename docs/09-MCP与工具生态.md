# 09. MCP 与工具生态

> **难度标记**：⭐ 基础 ⭐⭐ 进阶 ⭐⭐⭐ 高级
> **热度**：🔥🔥🔥 2025 年最高频新增考点

---

## 知识图谱

```
MCP 与工具生态
├── MCP 协议基础 ⭐⭐⭐ 必考
│   ├── 协议架构（Client-Server-Host）
│   ├── 核心能力（Tools / Resources / Prompts）
│   ├── 传输层（stdio / SSE / Streamable HTTP）
│   └── 与 OpenAI Function Calling 的区别
├── 工程实现 ⭐⭐ 进阶
│   ├── MCP Server 开发
│   ├── MCP Client 集成
│   ├── 安全与权限控制
│   └── 错误处理与重试
├── 工具生态 ⭐⭐ 进阶
│   ├── 工具发现与注册
│   ├── 工具编排（Tool Orchestration）
│   ├── 工具组合模式
│   └── 工具版本管理
└── 实战难题 ⭐⭐⭐
    ├── 多工具并发调用
    ├── 工具结果缓存
    ├── 工具沙箱与隔离
    └── 跨平台工具标准化
```

---

## 一、MCP 协议基础

### 1. ⭐⭐⭐ Q: 什么是 MCP（Model Context Protocol）？解决什么问题？

**答**：

MCP 是 Anthropic 于 2024 年 11 月发布的**开放协议标准**，定义了 LLM 应用与外部数据源、工具之间的通信方式。

**解决的核心问题**：

```
传统方式（M × N 问题）：
┌──────┐     ┌──────┐     ┌──────┐
│App 1 │────▶│Tool A│     │App 2 │────▶ Tool A（重复集成）
│      │────▶│Tool B│     │      │────▶ Tool B（重复集成）
│      │────▶│Tool C│     │      │────▶ Tool C（重复集成）
└──────┘     └──────┘     └──────┘
每个 App 都要写 N 个工具适配器 = M × N 个适配器

MCP 方式（M + N 问题）：
┌──────┐     ┌──────┐     ┌──────────┐
│App 1 │────▶│MCP   │────▶│Server A  │（工具实现一次）
│App 2 │────▶│Client│────▶│Server B  │（工具实现一次）
│App 3 │────▶│      │────▶│Server C  │（工具实现一次）
└──────┘     └──────┘     └──────────┘
每个 App 只需 1 个 MCP Client = M + N 个适配器
```

**类比**：MCP 之于 LLM 工具，就像 USB 之于外设 —— 统一接口，即插即用。

**与 Function Calling 的区别**：

| 维度 | Function Calling | MCP |
|------|-----------------|-----|
| 层级 | 模型层能力 | 应用层协议 |
| 定义 | 模型识别调用意图 | 标准化工具通信 |
| 范围 | 单次调用 | 完整生命周期（发现→调用→结果） |
| 依赖 | 绑定特定模型 API | 模型无关，跨平台 |
| 生态 | 各厂商私有格式 | 开放标准，统一生态 |

---

### 2. ⭐⭐⭐ Q: MCP 的架构由哪几部分组成？各自职责是什么？

**答**：

MCP 采用 **Host → Client → Server** 三层架构：

```
┌─────────────────────────────────────────────┐
│                  Host 层                     │
│  (Claude Desktop / IDE / 你的Agent应用)       │
│  ┌─────────────┐  ┌─────────────┐           │
│  │ MCP Client 1│  │ MCP Client 2│           │
│  └──────┬──────┘  └──────┬──────┘           │
└─────────┼────────────────┼──────────────────┘
          │                │
    ┌─────▼─────┐    ┌─────▼─────┐
    │MCP Server │    │MCP Server │
    │  (GitHub) │    │  (Slack)  │
    │  Tools:   │    │  Tools:   │
    │ - create_ │    │ - send_   │
    │   issue   │    │   message │
    │ - search  │    │ - list    │
    └───────────┘    └───────────┘
```

**Host（宿主）**：
- 运行 LLM 的应用程序（如 Claude Desktop、Cursor、你的 Agent）
- 管理 MCP Client 的生命周期
- 控制权限和安全策略

**Client（客户端）**：
- 与 MCP Server 建立一对一连接
- 处理协议协商和能力交换
- 转发工具调用请求和结果

**Server（服务端）**：
- 暴露具体的工具（Tools）、资源（Resources）、提示（Prompts）
- 处理实际的业务逻辑
- 可以是本地进程（stdio）或远程服务（SSE/HTTP）

---

### 3. ⭐⭐⭐ Q: MCP 的三大核心能力是什么？举例说明。

**答**：

MCP Server 可以暴露三种类型的能力：

**① Tools（工具）** —— 模型可以调用的函数
```json
{
  "name": "create_issue",
  "description": "Create a GitHub issue",
  "inputSchema": {
    "type": "object",
    "properties": {
      "title": { "type": "string" },
      "body": { "type": "string" },
      "labels": { "type": "array", "items": { "type": "string" } }
    },
    "required": ["title"]
  }
}
```
特点：**模型主动调用**，需要用户确认（安全敏感操作）

**② Resources（资源）** —— 应用可以读取的数据
```json
{
  "uri": "file:///project/README.md",
  "name": "Project README",
  "mimeType": "text/markdown"
}
```
特点：**应用层读取**，类似 GET 请求，可以订阅更新

**③ Prompts（提示模板）** —— 预定义的交互模板
```json
{
  "name": "code_review",
  "description": "Review code for best practices",
  "arguments": [
    { "name": "code", "required": true },
    { "name": "language", "required": false }
  ]
}
```
特点：**用户/应用选择触发**，生成结构化的消息序列

**类比理解**：
- Tools = 函数调用（动词）
- Resources = 文件/数据库（名词）
- Prompts = 提示模板（动词+名词的组合）

---

### 4. ⭐⭐ Q: MCP 有哪些传输方式？各自适用场景？

**答**：

MCP 定义了三种传输机制：

**① stdio（标准输入输出）**
```
Host 进程 ←──stdin/stdout──→ MCP Server 子进程
```
- 适用：**本地工具**，如文件系统、本地数据库
- 优点：零网络开销，进程隔离
- 缺点：仅限单机，每个 Client 独占一个进程

**② SSE（Server-Sent Events）**
```
Client ────HTTP POST───▶ Server（请求）
Client ◀──SSE Stream──── Server（响应流）
```
- 适用：**远程服务**，如 GitHub API、Slack API
- 优点：支持流式，跨网络
- 缺点：需要维护长连接，扩展性有限

**③ Streamable HTTP（2025 新增）**
```
Client ────HTTP POST───▶ Server
Client ◀──HTTP Response── Server（支持流式或一次性返回）
```
- 适用：**生产环境**，支持无状态和有状态模式
- 优点：兼容标准 HTTP，支持负载均衡，可恢复会话
- 缺点：较新，生态支持还在完善

---

### 5. ⭐⭐⭐ Q: 如何实现一个 MCP Server？以 Python 为例。

**答**：

```python
# server.py — 一个简单的 MCP Server 示例
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import json

# 1. 创建 Server 实例
server = Server("my-tool-server")

# 2. 定义工具列表
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="calculate",
            description="执行数学计算",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2 + 3 * 4'"
                    }
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="get_weather",
            description="获取城市天气",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": { "type": "string", "description": "城市名" }
                },
                "required": ["city"]
            }
        )
    ]

# 3. 实现工具调用逻辑
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "calculate":
        try:
            result = eval(arguments["expression"])  # 生产环境应使用安全的表达式解析
            return [TextContent(type="text", text=str(result))]
        except Exception as e:
            return [TextContent(type="text", text=f"计算错误: {e}")]
    
    elif name == "weather":
        city = arguments["city"]
        # 调用天气 API
        weather = await fetch_weather(city)
        return [TextContent(type="text", text=json.dumps(weather, ensure_ascii=False))]
    
    raise ValueError(f"未知工具: {name}")

# 4. 定义资源（可选）
@server.list_resources()
async def list_resources():
    return [
        {
            "uri": "config://app/settings",
            "name": "应用配置",
            "mimeType": "application/json"
        }
    ]

@server.read_resource()
async def read_resource(uri: str):
    if uri == "config://app/settings":
        return json.dumps({"theme": "dark", "language": "zh"})
    raise ValueError(f"未知资源: {uri}")

# 5. 启动服务器（stdio 模式）
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**关键步骤**：
1. 创建 `Server` 实例
2. 用 `@server.list_tools()` 注册工具定义
3. 用 `@server.call_tool()` 实现工具逻辑
4. 选择传输方式启动（stdio / SSE / HTTP）

---

### 6. ⭐⭐⭐ Q: 如何在 Agent 应用中集成 MCP Client？

**答**：

```python
# client.py — 在 Agent 中使用 MCP Client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPAgent:
    def __init__(self):
        self.tools = []      # 从 MCP Server 获取的工具列表
        self.session = None  # MCP 会话
    
    async def connect(self, server_script: str):
        """连接到 MCP Server"""
        # 1. 启动 Server 子进程
        server_params = StdioServerParameters(
            command="python",
            args=[server_script]
        )
        
        # 2. 建立连接
        self.read, self.write = await stdio_client(server_params).__aenter__()
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        
        # 3. 初始化（能力协商）
        await self.session.initialize()
        
        # 4. 发现工具
        response = await self.session.list_tools()
        self.tools = response.tools
        print(f"发现 {len(self.tools)} 个工具: {[t.name for t in self.tools]}")
    
    async def run(self, user_query: str):
        """处理用户查询"""
        # 1. 将 MCP 工具转换为 LLM 的 function calling 格式
        llm_tools = self._convert_tools(self.tools)
        
        # 2. 调用 LLM
        response = await call_llm(
            messages=[{"role": "user", "content": user_query}],
            tools=llm_tools
        )
        
        # 3. 如果 LLM 选择调用工具
        if response.tool_calls:
            for tool_call in response.tool_calls:
                # 4. 通过 MCP 调用工具
                result = await self.session.call_tool(
                    name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments)
                )
                
                # 5. 将结果返回给 LLM
                response = await call_llm(
                    messages=[
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "tool_calls": [tool_call]},
                        {"role": "tool", "content": result.content[0].text}
                    ],
                    tools=llm_tools
                )
        
        return response.content
    
    def _convert_tools(self, mcp_tools):
        """将 MCP 工具格式转换为 OpenAI function calling 格式"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in mcp_tools
        ]
    
    async def disconnect(self):
        """断开连接"""
        if self.session:
            await self.session.__aexit__(None, None, None)

# 使用示例
async def main():
    agent = MCPAgent()
    await agent.connect("server.py")
    
    result = await agent.run("帮我查一下北京的天气")
    print(result)
    
    await agent.disconnect()
```

---

### 7. ⭐⭐ Q: MCP 的安全模型是怎样的？

**答**：

MCP 在多个层面实现安全控制：

**① Host 层权限控制**
```python
# Host 可以限制 Client 的能力
server_params = StdioServerParameters(
    command="python",
    args=["server.py"],
    env={"API_KEY": "***"}  # 注入敏感信息
)

# Host 可以要求用户确认危险操作
@server.call_tool()
async def call_tool(name, arguments):
    if name in HIGH_RISK_TOOLS:
        # Host 弹出确认对话框
        if not await host.confirm(f"确认执行 {name}?"):
            return [TextContent(type="text", text="用户取消")]
```

**② 传输层安全**
```
stdio: 本地进程隔离，无网络暴露
SSE/HTTP: TLS 加密，Token 认证
```

**③ 工具层安全**
```python
# 工具实现中的安全措施
@server.call_tool()
async def call_tool(name, arguments):
    # 1. 输入验证
    validate_input(arguments)
    
    # 2. 权限检查
    if not has_permission(current_user, name):
        raise PermissionError("无权限")
    
    # 3. 速率限制
    if rate_limiter.is_limited(name):
        raise RateLimitError("请求过于频繁")
    
    # 4. 沙箱执行
    result = sandbox_execute(name, arguments)
    
    # 5. 输出过滤
    return sanitize_output(result)
```

**④ 最小权限原则**
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxx"  // 使用 fine-grained token，只给必要权限
      }
    }
  }
}
```

---

## 二、工程实践

### 8. ⭐⭐⭐ Q: MCP 与 OpenAI Function Calling、Tool Use 有什么区别？什么时候用哪个？

**答**：

| 维度 | OpenAI Function Calling | Anthropic Tool Use | MCP |
|------|------------------------|-------------------|-----|
| 层级 | 模型 API 能力 | 模型 API 能力 | 应用层协议 |
| 范围 | 单次调用 | 单次调用 | 完整生命周期 |
| 标准化 | 各厂商私有 | 各厂商私有 | 开放标准 |
| 工具发现 | 手动定义 | 手动定义 | 自动发现 |
| 状态管理 | 无 | 无 | 支持会话 |
| 生态 | OpenAI 生态 | Anthropic 生态 | 跨平台 |

**选择策略**：

```
需要跨平台/跨模型？ ──是──▶ 用 MCP
        │
        否
        ▼
只用单一模型？ ──是──▶ 用该模型的原生 Function Calling
        │
        否
        ▼
需要工具复用？ ──是──▶ 用 MCP（写一次，到处用）
        │
        否
        ▼
简单场景 ──▶ Function Calling 足够
```

**实际项目中的最佳实践**：
```
┌─────────────────────────────────────────┐
│              你的 Agent 应用              │
│  ┌───────────────────────────────────┐  │
│  │         统一工具抽象层             │  │
│  │  (将 MCP 工具转为 Function Call)   │  │
│  └─────────────┬─────────────────────┘  │
│                │                         │
│    ┌───────────┼───────────┐            │
│    ▼           ▼           ▼            │
│  OpenAI    Anthropic    本地模型         │
│  Function  Tool Use     MCP Tools       │
│  Calling                              │
└─────────────────────────────────────────┘
```

---

### 9. ⭐⭐ Q: 如何实现 MCP 工具的动态发现和热更新？

**答**：

```python
class MCPToolRegistry:
    """MCP 工具注册中心 — 支持动态发现和热更新"""
    
    def __init__(self):
        self.servers: dict[str, MCPServerConnection] = {}
        self.tools: dict[str, Tool] = {}
        self._watchers: list[Callable] = []
    
    async def register_server(self, name: str, config: dict):
        """注册一个 MCP Server"""
        conn = MCPServerConnection(config)
        await conn.connect()
        self.servers[name] = conn
        
        # 获取该 Server 的所有工具
        tools = await conn.list_tools()
        for tool in tools:
            tool_id = f"{name}/{tool.name}"  # 命名空间隔离
            self.tools[tool_id] = tool
        
        # 启动工具变更监听
        asyncio.create_task(self._watch_server(name, conn))
    
    async def _watch_server(self, name: str, conn: MCPServerConnection):
        """监听 Server 工具变更"""
        while True:
            try:
                # 定期轮询或订阅变更通知
                await asyncio.sleep(30)
                new_tools = await conn.list_tools()
                
                # 检测变更
                old_ids = {t.name for t in self.tools.values() if t.name.startswith(name)}
                new_ids = {t.name for t in new_tools}
                
                added = new_ids - old_ids
                removed = old_ids - new_ids
                
                if added or removed:
                    # 更新工具注册表
                    for tool in new_tools:
                        self.tools[f"{name}/{tool.name}"] = tool
                    for r in removed:
                        self.tools.pop(f"{name}/{r}", None)
                    
                    # 通知观察者
                    for watcher in self._watchers:
                        await watcher(name, added, removed)
                    
                    logger.info(f"Server {name} 工具变更: +{added} -{removed}")
            
            except Exception as e:
                logger.error(f"监听 Server {name} 失败: {e}")
                await asyncio.sleep(5)
    
    def on_tools_changed(self, callback: Callable):
        """注册工具变更回调"""
        self._watchers.append(callback)
    
    async def call_tool(self, tool_id: str, arguments: dict):
        """调用工具"""
        if tool_id not in self.tools:
            raise ToolNotFoundError(f"工具 {tool_id} 不存在")
        
        server_name = tool_id.split("/")[0]
        tool_name = tool_id.split("/", 1)[1]
        
        conn = self.servers[server_name]
        return await conn.call_tool(tool_name, arguments)
    
    def get_tools_for_llm(self) -> list[dict]:
        """导出为 LLM function calling 格式"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool_id.replace("/", "__"),  # 避免命名冲突
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool_id, tool in self.tools.items()
        ]
```

---

### 10. ⭐⭐⭐ Q: 多个 MCP Server 的工具如何编排？有哪些模式？

**答**：

**模式一：串行链式调用**
```
用户: "帮我创建一个 Jira ticket，并在 Slack 通知"

Agent 执行链:
1. MCP(Server=jira) → create_issue(title, description)
2. 拿到 issue_id
3. MCP(Server=slack) → send_message(channel, f"新 Issue: {issue_id}")
```

**模式二：并行调用**
```
用户: "查一下 GitHub 和 Jira 的最新状态"

Agent 执行:
┌──────────────────────┬──────────────────────┐
│ MCP(Server=github)   │ MCP(Server=jira)     │
│ → list_recent_prs()  │ → list_my_issues()   │
└──────────┬───────────┴──────────┬───────────┘
           └──────────┬───────────┘
                      ▼
              合并结果返回给用户
```

**模式三：条件分支**
```
用户: "帮我处理这个错误"

Agent 判断:
├── 如果是代码 Bug → MCP(Server=github) → create_issue()
├── 如果是线上告警 → MCP(Server=pagerduty) → create_incident()
└── 如果是用户反馈 → MCP(Server=intercom) → reply_message()
```

**实现代码**：
```python
class MCPToolOrchestrator:
    """MCP 工具编排器"""
    
    def __init__(self, registry: MCPToolRegistry):
        self.registry = registry
    
    async def parallel_call(self, calls: list[dict]) -> list:
        """并行调用多个工具"""
        tasks = [
            self.registry.call_tool(c["tool_id"], c["arguments"])
            for c in calls
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def chain_call(self, calls: list[dict]) -> any:
        """串行链式调用，前一个结果传给下一个"""
        result = None
        for call in calls:
            # 支持动态参数（引用前一步结果）
            arguments = self._resolve_args(call["arguments"], result)
            result = await self.registry.call_tool(call["tool_id"], arguments)
        return result
    
    async def conditional_call(self, condition: str, branches: dict) -> any:
        """条件分支调用"""
        # 让 LLM 判断走哪个分支
        branch = await self._evaluate_condition(condition, branches)
        return await self.registry.call_tool(
            branches[branch]["tool_id"],
            branches[branch]["arguments"]
        )
```

---

## 三、高级话题

### 11. ⭐⭐⭐ Q: MCP 的工具描述（Tool Description）如何优化以提高调用准确率？

**答**：

工具描述是 LLM 决定是否调用、如何调用的关键。优化策略：

**① 描述要具体，不要泛泛**
```json
// ❌ 差
{
  "name": "search",
  "description": "搜索数据"
}

// ✅ 好
{
  "name": "search_documents",
  "description": "在知识库中搜索文档。返回最相关的文档片段，按相关度排序。适用于查找技术文档、产品手册、FAQ 等。不适用于实时数据查询。",
  "inputSchema": {
    "properties": {
      "query": {
        "description": "搜索查询，用自然语言描述你要找的内容。示例: '如何配置 OAuth2 认证'"
      },
      "top_k": {
        "description": "返回结果数量，默认 5。如果需要全面分析可以设为 10-20"
      }
    }
  }
}
```

**② 提供参数约束和示例**
```json
{
  "name": "create_issue",
  "inputSchema": {
    "properties": {
      "priority": {
        "type": "string",
        "enum": ["low", "medium", "high", "critical"],
        "description": "优先级。low=不影响使用, medium=有 workaround, high=影响核心功能, critical=系统不可用"
      },
      "labels": {
        "type": "array",
        "items": { "type": "string" },
        "description": "标签列表。常用: ['bug', 'feature', 'docs', 'performance']"
      }
    }
  }
}
```

**③ 说明使用场景和限制**
```json
{
  "name": "execute_sql",
  "description": "执行只读 SQL 查询。仅支持 SELECT 语句，不支持 INSERT/UPDATE/DELETE。查询超时 30 秒。返回前 1000 行。",
  "inputSchema": {
    "properties": {
      "query": {
        "description": "SQL 查询语句。必须是 SELECT 语句。支持 JOIN、子查询、聚合函数。"
      }
    }
  }
}
```

---

### 12. ⭐⭐⭐ Q: MCP 在生产环境中如何做监控和可观测性？

**答**：

```python
import time
from opentelemetry import trace

tracer = trace.get_tracer("mcp-client")

class ObservableMCPClient:
    """带可观测性的 MCP Client"""
    
    async def call_tool(self, tool_id: str, arguments: dict):
        # 1. 创建 Span
        with tracer.start_as_current_span(
            "mcp.call_tool",
            attributes={
                "mcp.tool_id": tool_id,
                "mcp.server": tool_id.split("/")[0],
                "mcp.tool_name": tool_id.split("/", 1)[1],
            }
        ) as span:
            start = time.time()
            
            try:
                # 2. 调用工具
                result = await self._do_call(tool_id, arguments)
                
                # 3. 记录成功指标
                duration = time.time() - start
                span.set_attribute("mcp.duration_ms", duration * 1000)
                span.set_attribute("mcp.result_size", len(str(result)))
                span.set_status(trace.Status(trace.StatusCode.OK))
                
                # 4. 发送指标
                metrics.histogram("mcp.tool.duration", duration, tags={
                    "tool": tool_id,
                    "status": "success"
                })
                metrics.increment("mcp.tool.calls", tags={"tool": tool_id})
                
                return result
            
            except Exception as e:
                # 5. 记录失败
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                metrics.increment("mcp.tool.errors", tags={
                    "tool": tool_id,
                    "error": type(e).__name__
                })
                raise

# 监控 Dashboard 关键指标
MONITORING_METRICS = {
    "mcp.tool.duration":       "工具调用延迟 (P50/P95/P99)",
    "mcp.tool.calls":          "工具调用 QPS",
    "mcp.tool.errors":         "工具调用错误率",
    "mcp.tool.result_size":    "结果大小分布",
    "mcp.server.connected":    "Server 连接状态",
    "mcp.tool.cache.hit_rate": "工具结果缓存命中率",
}
```

---

### 13. ⭐⭐ Q: MCP 与 A2A（Agent-to-Agent）协议是什么关系？

**答**：

两者解决不同层次的问题：

```
┌─────────────────────────────────────────────────────┐
│                    应用层                             │
│  ┌──────────────────────────────────────────────┐   │
│  │              A2A 协议                         │   │
│  │  Agent 之间发现、协商、协作                    │   │
│  │  (Agent Card → Task → Artifact)              │   │
│  └──────────────────┬───────────────────────────┘   │
│                     │                                │
│  ┌──────────────────▼───────────────────────────┐   │
│  │              MCP 协议                         │   │
│  │  Agent 与工具/数据源的标准化连接              │   │
│  │  (Tools → Resources → Prompts)               │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

| 维度 | MCP | A2A |
|------|-----|-----|
| 发布者 | Anthropic | Google |
| 解决问题 | Agent ↔ 工具/数据 | Agent ↔ Agent |
| 核心概念 | Tools, Resources, Prompts | Agent Card, Task, Artifact |
| 通信模式 | 请求-响应 | 异步任务，支持长时间运行 |
| 关系 | 互补，不是竞争 | 互补，不是竞争 |

**实际应用中两者结合**：
```
用户 → Orchestrator Agent
         │
         ├──A2A──▶ Research Agent ──MCP──▶ Web Search Tool
         │                                    │
         ├──A2A──▶ Code Agent ────MCP──▶ GitHub Tool
         │                                    │
         └──A2A──▶ Report Agent ──MCP──▶ Docs Tool
```

---

### 14. ⭐⭐⭐ Q: 如何测试 MCP Server？有哪些测试策略？

**答**：

```python
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class TestMCPServer:
    """MCP Server 测试套件"""
    
    @pytest.fixture
    async def session(self):
        """创建测试会话"""
        params = StdioServerParameters(
            command="python",
            args=["server.py"]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    
    # 1. 工具发现测试
    async def test_list_tools(self, session):
        """测试工具列表是否正确"""
        result = await session.list_tools()
        assert len(result.tools) > 0
        
        tool_names = [t.name for t in result.tools]
        assert "create_issue" in tool_names
        assert "search" in tool_names
    
    # 2. 工具 Schema 验证
    async def test_tool_schema(self, session):
        """测试工具 Schema 是否合法"""
        result = await session.list_tools()
        for tool in result.tools:
            assert tool.inputSchema.get("type") == "object"
            assert "properties" in tool.inputSchema
    
    # 3. 正常调用测试
    async def test_call_tool_success(self, session):
        """测试正常工具调用"""
        result = await session.call_tool(
            "calculate",
            {"expression": "2 + 3"}
        )
        assert result.content[0].text == "5"
    
    # 4. 错误处理测试
    async def test_call_tool_invalid_input(self, session):
        """测试无效输入"""
        with pytest.raises(Exception):
            await session.call_tool(
                "calculate",
                {"expression": "invalid"}
            )
    
    # 5. 未知工具测试
    async def test_call_unknown_tool(self, session):
        """测试调用不存在的工具"""
        with pytest.raises(Exception):
            await session.call_tool("nonexistent_tool", {})
    
    # 6. 资源读取测试
    async def test_read_resource(self, session):
        """测试资源读取"""
        result = await session.read_resource("config://app/settings")
        assert "theme" in result.contents[0].text
    
    # 7. 性能测试
    async def test_tool_latency(self, session):
        """测试工具调用延迟"""
        import time
        start = time.time()
        await session.call_tool("calculate", {"expression": "1+1"})
        latency = time.time() - start
        assert latency < 1.0  # 延迟应小于 1 秒
    
    # 8. 并发测试
    async def test_concurrent_calls(self, session):
        """测试并发调用"""
        import asyncio
        tasks = [
            session.call_tool("calculate", {"expression": f"{i}+1}"})
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 10
```

---

## 四、实战难题

### 15. ⭐⭐⭐ Q: 如何设计一个支持多 MCP Server 的 Agent 框架？

**答**：

```python
class MultiMCPAgent:
    """支持多 MCP Server 的 Agent 框架"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.registry = MCPToolRegistry()
        self.tool_call_log = []  # 工具调用日志
    
    async def add_server(self, name: str, config: dict):
        """添加 MCP Server"""
        await self.registry.register_server(name, config)
        logger.info(f"已连接 Server: {name}, 工具数: {len(self.registry.tools)}")
    
    async def run(self, user_message: str, max_iterations: int = 10) -> str:
        """运行 Agent 循环"""
        messages = [{"role": "user", "content": user_message}]
        tools = self.registry.get_tools_for_llm()
        
        for i in range(max_iterations):
            # 1. 调用 LLM
            response = await self.llm.chat(messages=messages, tools=tools)
            
            # 2. 如果没有工具调用，返回结果
            if not response.tool_calls:
                return response.content
            
            # 3. 执行工具调用
            messages.append(response.message)
            
            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # 转换命名空间
                tool_id = tool_name.replace("__", "/")
                
                try:
                    result = await self.registry.call_tool(tool_id, arguments)
                    tool_result = result.content[0].text
                    
                    # 记录调用日志
                    self.tool_call_log.append({
                        "tool": tool_id,
                        "arguments": arguments,
                        "result": tool_result[:500]  # 截断
                    })
                
                except Exception as e:
                    tool_result = f"工具调用失败: {e}"
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
        
        return "达到最大迭代次数"
    
    def get_tool_usage_stats(self) -> dict:
        """获取工具使用统计"""
        from collections import Counter
        tool_counts = Counter(log["tool"] for log in self.tool_call_log)
        return {
            "total_calls": len(self.tool_call_log),
            "tool_distribution": dict(tool_counts),
            "avg_result_size": sum(
                len(log["result"]) for log in self.tool_call_log
            ) / max(len(self.tool_call_log), 1)
        }
```

---

### 16. ⭐⭐⭐ Q: MCP 工具调用失败如何做优雅降级？

**答**：

```python
class ResilientMCPClient:
    """带容错的 MCP Client"""
    
    def __init__(self):
        self.servers: dict[str, MCPServerConnection] = {}
        self.fallback_strategies: dict[str, Callable] = {}
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
    
    async def call_tool_with_fallback(
        self, 
        tool_id: str, 
        arguments: dict,
        max_retries: int = 3
    ) -> str:
        """带重试和降级的工具调用"""
        
        # 1. 熔断器检查
        breaker = self.circuit_breakers.get(tool_id)
        if breaker and breaker.is_open():
            return await self._fallback(tool_id, arguments)
        
        # 2. 重试逻辑
        last_error = None
        for attempt in range(max_retries):
            try:
                result = await asyncio.wait_for(
                    self._do_call(tool_id, arguments),
                    timeout=10.0  # 超时控制
                )
                
                # 成功，重置熔断器
                if breaker:
                    breaker.record_success()
                
                return result
            
            except asyncio.TimeoutError:
                last_error = "工具调用超时"
                logger.warning(f"{tool_id} 超时 (尝试 {attempt+1}/{max_retries})")
            
            except ServerDisconnectedError:
                last_error = "Server 断开连接"
                logger.warning(f"{tool_id} Server 断开 (尝试 {attempt+1}/{max_retries})")
                # 尝试重连
                await self._reconnect(tool_id)
            
            except ToolExecutionError as e:
                last_error = str(e)
                # 工具执行错误不重试
                break
            
            # 指数退避
            await asyncio.sleep(2 ** attempt)
        
        # 3. 所有重试失败，触发熔断
        if breaker:
            breaker.record_failure()
        
        # 4. 降级处理
        return await self._fallback(tool_id, arguments, last_error)
    
    async def _fallback(self, tool_id: str, arguments: dict, error: str = None) -> str:
        """降级策略"""
        
        # 策略 1: 使用缓存结果
        cached = await self.cache.get(f"{tool_id}:{hash(str(arguments))}")
        if cached:
            logger.info(f"{tool_id} 使用缓存结果")
            return cached
        
        # 策略 2: 使用备用 Server
        backup = self.backup_servers.get(tool_id)
        if backup:
            try:
                return await backup.call_tool(tool_id.split("/", 1)[1], arguments)
            except Exception:
                pass
        
        # 策略 3: 返回友好错误信息
        return json.dumps({
            "error": True,
            "message": f"工具 {tool_id} 暂时不可用，请稍后再试",
            "suggestion": "你可以手动完成这个操作，或者换个方式提问"
        })
```

---

### 17. ⭐⭐ Q: MCP 工具结果如何做缓存？缓存策略有哪些？

**答**：

```python
from datetime import timedelta

class MCPToolCache:
    """MCP 工具结果缓存"""
    
    def __init__(self):
        self.cache = {}
        
        # 不同工具的缓存策略
        self.strategies = {
            # 实时数据：不缓存
            "weather/get_current": {"ttl": 0, "key_fields": ["city"]},
            
            # 半静态数据：短期缓存
            "github/list_repos": {"ttl": 300, "key_fields": ["owner"]},
            
            # 静态数据：长期缓存
            "docs/search": {"ttl": 3600, "key_fields": ["query"]},
            
            # 计算结果：永久缓存
            "calculator/evaluate": {"ttl": -1, "key_fields": ["expression"]},
        }
    
    def _build_cache_key(self, tool_id: str, arguments: dict) -> str:
        """构建缓存键"""
        strategy = self.strategies.get(tool_id, {"key_fields": list(arguments.keys())})
        
        # 只用关键字段作为缓存键
        key_parts = []
        for field in strategy["key_fields"]:
            key_parts.append(f"{field}={arguments.get(field)}")
        
        return f"{tool_id}|{'&'.join(key_parts)}"
    
    async def get_or_call(self, tool_id: str, arguments: dict, caller: Callable) -> str:
        """获取缓存结果或调用工具"""
        strategy = self.strategies.get(tool_id, {"ttl": 0})
        
        # 不缓存
        if strategy["ttl"] == 0:
            return await caller(tool_id, arguments)
        
        cache_key = self._build_cache_key(tool_id, arguments)
        
        # 检查缓存
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if strategy["ttl"] == -1 or entry["expires"] > time.time():
                return entry["value"]
        
        # 缓存未命中，调用工具
        result = await caller(tool_id, arguments)
        
        # 写入缓存
        self.cache[cache_key] = {
            "value": result,
            "expires": time.time() + strategy["ttl"] if strategy["ttl"] > 0 else float('inf'),
            "created": time.time()
        }
        
        return result
```

---

### 18. ⭐⭐⭐ Q: MCP Server 如何做版本管理和向后兼容？

**答**：

```python
from enum import Enum

class ToolVersion(Enum):
    V1 = "1.0"
    V2 = "2.0"

# 工具版本管理
TOOLS_V1 = {
    "name": "search",
    "description": "搜索文档",
    "inputSchema": {
        "properties": {
            "query": {"type": "string"}
        }
    }
}

TOOLS_V2 = {
    "name": "search",
    "description": "搜索文档（支持语义搜索）",
    "inputSchema": {
        "properties": {
            "query": {"type": "string"},
            "mode": {
                "type": "string",
                "enum": ["keyword", "semantic", "hybrid"],
                "description": "搜索模式，默认 hybrid"
            },
            "filters": {
                "type": "object",
                "properties": {
                    "date_range": {"type": "string"},
                    "doc_type": {"type": "string"}
                }
            }
        }
    }
}

class VersionedMCPServer:
    """支持版本管理的 MCP Server"""
    
    def __init__(self):
        self.versions = {
            "1.0": {"search": self._search_v1},
            "2.0": {"search": self._search_v2},
        }
        self.current_version = "2.0"
    
    @server.list_tools()
    async def list_tools(self, protocol_version: str = None):
        """根据协议版本返回工具列表"""
        version = protocol_version or self.current_version
        
        if version.startswith("1."):
            return [TOOLS_V1]
        else:
            return [TOOLS_V2]
    
    @server.call_tool()
    async def call_tool(self, name: str, arguments: dict, protocol_version: str = None):
        version = protocol_version or self.current_version
        
        handler = self.versions.get(version, {}).get(name)
        if not handler:
            # 降级到最近的兼容版本
            handler = self._find_compatible_handler(name, version)
        
        return await handler(arguments)
    
    async def _search_v1(self, args):
        """V1 搜索（关键词）"""
        return keyword_search(args["query"])
    
    async def _search_v2(self, args):
        """V2 搜索（支持语义）"""
        mode = args.get("mode", "hybrid")
        if mode == "semantic":
            return semantic_search(args["query"])
        elif mode == "keyword":
            return keyword_search(args["query"])
        else:
            return hybrid_search(args["query"], args.get("filters"))
```

---

## 总结

### 面试高频追问

1. **"MCP 和 Function Calling 什么关系？"** → MCP 是应用层协议，Function Calling 是模型层能力，两者互补
2. **"MCP 怎么保证安全？"** → Host 层权限 + 传输加密 + 最小权限 + 沙箱执行
3. **"多个 MCP Server 怎么管理？"** → 工具注册中心 + 命名空间隔离 + 动态发现
4. **"MCP 工具调用失败怎么办？"** → 重试 + 熔断 + 降级（缓存/备用/友好提示）
5. **"MCP 和 A2A 什么关系？"** → MCP 解决 Agent↔Tool，A2A 解决 Agent↔Agent，互补

### 关键数字

| 指标 | 参考值 |
|------|--------|
| MCP 工具调用延迟 | <100ms (本地 stdio), <500ms (远程 SSE) |
| 工具描述长度 | 建议 50-200 字 |
| 重试次数 | 2-3 次，指数退避 |
| 缓存 TTL | 静态 1h, 半静态 5min, 实时不缓存 |
