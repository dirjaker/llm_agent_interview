# 32 - Agentic Coding（代码Agent） 🔥🔥🔥

> **热度指示：🔥🔥🔥** — 2024-2025年最热门的AI应用方向之一，几乎所有大厂和创业公司都在布局

---

## 知识图谱

```
Agentic Coding（代码Agent）
├── 1. 范式演进
│   ├── 代码补全（Copilot v1）
│   ├── 对话式编码（ChatGPT）
│   ├── Agent式编码（Claude Code / Cursor Agent）
│   └── 自主编码（Devin / OpenHands）
│
├── 2. 核心架构
│   ├── 规划层（Planning）
│   │   ├── 任务分解
│   │   ├── 文件定位
│   │   └── 执行策略
│   ├── 工具层（Tool Use）
│   │   ├── 文件读写
│   │   ├── 终端执行
│   │   ├── 搜索/索引
│   │   └── LSP/DAP协议
│   ├── 上下文管理
│   │   ├── 代码库索引（Codebase Indexing）
│   │   ├── RAG for Code
│   │   └── 上下文窗口压缩
│   └── 反馈循环
│       ├── 测试执行
│       ├── Lint/类型检查
│       └── 错误修复迭代
│
├── 3. 代表性产品
│   ├── Claude Code（Anthropic）
│   ├── Cursor（Anysphere）
│   ├── GitHub Copilot（Microsoft/GitHub）
│   ├── Cline（开源）
│   ├── Windsurf（Codeium）
│   ├── Devin（Cognition）
│   └── OpenHands（开源）
│
├── 4. 质量控制
│   ├── 测试驱动开发（TDD）
│   ├── 静态分析 & Lint
│   ├── 类型检查
│   └── AI Review / 自我审查
│
├── 5. 安全与边界
│   ├── 沙箱隔离
│   ├── 权限控制
│   ├── 代码审计
│   └── 人在环中（Human-in-the-loop）
│
└── 6. 前沿趋势
    ├── 多Agent协作编码
    ├── 全自主软件工程（SWE-bench）
    ├── 长上下文 + 代码理解
    └── Agentic IDE 融合
```

---

## 一、Agentic Coding 范式概览 ⭐⭐ 🔥🔥

### Q1：什么是 Agentic Coding？与传统代码生成有什么区别？

**A：**

Agentic Coding 是指 LLM 以 **Agent 模式** 进行编码活动——不再是简单的"输入提示→输出代码片段"，而是具备 **自主规划、工具调用、环境交互、迭代修复** 能力的端到端编码系统。

| 维度 | 传统代码生成 | Agentic Coding |
|------|-------------|----------------|
| 交互模式 | 单轮：prompt → code | 多轮：规划→执行→观察→修正 |
| 上下文范围 | 当前文件/片段 | 整个代码库 |
| 工具使用 | 无 | 文件读写、终端、搜索、LSP |
| 错误处理 | 用户手动修复 | Agent自动检测并迭代修复 |
| 任务粒度 | 函数/代码块 | Feature/Bugfix/跨文件重构 |
| 代表产品 | Copilot v1 补全 | Claude Code, Cursor Agent |

```
演进路线：

代码补全 → 对话编码 → Agent编码 → 全自主编码
(2021)     (2023)     (2024)      (2025+)
Copilot    ChatGPT    Claude Code  Devin
Tab补全     问答式      工具调用+规划  端到端自主
```

**追问场景：**
- Q：Agentic Coding 和普通的多轮对话编码有什么本质区别？
- A：本质区别在于 **工具使用（Tool Use）** 和 **自主规划（Autonomous Planning）**。多轮对话仍需用户逐步引导，而 Agentic Coding 的 Agent 能自主决定"下一步做什么"——读哪些文件、运行什么命令、如何修复错误。

---

### Q2：主流代码Agent产品有哪些？各自的架构特点是什么？

**A：** ⭐⭐ 🔥🔥

| 产品 | 类型 | 核心特点 | 架构模式 |
|------|------|---------|---------|
| **Claude Code** | CLI Agent | Headless模式、MCP协议、子Agent委派 | 终端原生、深度集成 |
| **Cursor** | IDE Agent | 内嵌VSCode、Tab+Agent混合、代码库索引 | IDE原生、RAG增强 |
| **GitHub Copilot** | IDE插件 | Workspace索引、Agent模式（2025） | 插件式、逐步Agent化 |
| **Cline** | VSCode插件 | 开源、MCP支持、自主文件操作 | 开源Agent、社区驱动 |
| **Windsurf** | IDE | Cascade流式Agent、深度上下文 | IDE原生、流式执行 |
| **Devin** | 全自主Agent | 浏览器+终端+编辑器、端到端 | 沙箱化、全自主 |
| **OpenHands** | 开源框架 | Docker沙箱、多Agent、可扩展 | 开源、可定制 |

**追问场景：**
- Q：Cursor 和 Claude Code 的核心设计差异是什么？
- A：Cursor 是 **IDE原生**，将Agent能力嵌入编辑器体验中（Tab补全 + Chat + Agent三种模式共存）；Claude Code 是 **终端原生**，以CLI交互为核心，通过MCP协议扩展能力，更强调"无头（headless）"自动化和脚本集成。Cursor更注重日常编码体验，Claude Code更注重深度自动化和CI/CD集成。

---

## 二、代码Agent架构设计 ⭐⭐ 🔥🔥

### Q3：代码Agent的核心架构包含哪些模块？

**A：**

```
┌─────────────────────────────────────────────┐
│                 用户意图                      │
│          "修复登录页面的bug"                   │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│              规划层 (Planner)                 │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐  │
│  │任务分解  │→│文件定位   │→│执行策略制定   │  │
│  └─────────┘ └──────────┘ └──────────────┘  │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│              工具层 (Tools)                   │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌───────────┐  │
│  │文件   │ │终端   │ │搜索   │ │LSP/类型    │  │
│  │读写   │ │执行   │ │索引   │ │检查        │  │
│  └──────┘ └──────┘ └──────┘ └───────────┘  │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│            上下文管理 (Context)                │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐  │
│  │代码库索引  │ │RAG检索   │ │上下文压缩    │  │
│  └──────────┘ └──────────┘ └─────────────┘  │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│            反馈循环 (Feedback Loop)            │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌───────────┐  │
│  │测试   │→│Lint  │→│类型   │→│迭代修复    │  │
│  │执行   │ │检查   │ │检查   │ │           │  │
│  └──────┘ └──────┘ └──────┘ └───────────┘  │
└─────────────────────────────────────────────┘
```

**追问场景：**
- Q：如果让你从零设计一个代码Agent，你会怎么组织工具层？
- A：我会设计以下核心工具：(1) `read_file` — 读取文件内容；(2) `write_file` / `patch` — 写入或增量修改文件；(3) `search_files` — 基于ripgrep的代码搜索；(4) `execute_command` — 终端命令执行；(5) `list_files` — 目录浏览；(6) `run_tests` — 测试执行。每个工具都应有清晰的输入/输出schema，并且所有副作用操作需要用户确认或在沙箱中执行。

---

### Q4：代码Agent如何处理文件操作？有哪些设计模式？

**A：** ⭐⭐

代码Agent的文件操作有三种主要模式：

**1. 全量覆写（Full Write）**
```python
# Agent生成完整文件内容
def write_file(path: str, content: str):
    with open(path, 'w') as f:
        f.write(content)
```
- 优点：简单可靠
- 缺点：上下文消耗大、容易丢失未读取的内容

**2. 增量补丁（Patch/Diff）**
```python
# Agent生成统一diff格式
def apply_patch(path: str, patch: str):
    # 使用 fuzzy matching 定位
    # 应用 unified diff 格式的修改
    apply_unified_diff(path, patch)
```
- 优点：精确修改、上下文效率高
- 缺点：需要fuzzy matching容错

**3. 搜索替换（Search & Replace）**
```python
# Agent指定搜索字符串和替换字符串
def search_replace(path: str, old: str, new: str):
    content = read_file(path)
    # 模糊匹配定位 old_string
    updated = fuzzy_replace(content, old, new)
    write_file(path, updated)
```
- 优点：直观、平衡精确与灵活
- 缺点：搜索字符串不唯一时可能出错

**追问场景：**
- Q：为什么 Cursor/Claude Code 都倾向于使用增量补丁而非全量覆写？
- A：核心原因是 **上下文窗口效率**。全量覆写需要Agent在上下文中持有整个文件内容，对于大文件（1000+行）非常浪费token。增量补丁只需指定修改部分，节省70-90%的token。此外，增量补丁降低了"意外删除"的风险，因为Agent只修改明确指定的区域。但增量补丁需要 **fuzzy matching** 来容忍缩进差异等小问题。

---

### Q5：终端执行工具如何设计？需要注意什么？

**A：** ⭐⭐

```python
class TerminalTool:
    def execute(self, command: str, timeout: int = 30) -> dict:
        """
        执行终端命令并返回结果
        """
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=self.working_dir
        )
        return {
            "stdout": result.stdout[-5000:],  # 截断防止溢出
            "stderr": result.stderr[-2000:],
            "returncode": result.returncode,
            "timed_out": False
        }
```

**关键设计考量：**

| 考量 | 设计策略 |
|------|---------|
| 输出过长 | 截断到固定长度，保留首尾 |
| 超时控制 | 默认30s，长任务（构建）可延长 |
| 交互式命令 | 禁止或自动提供输入（如 `yes \|`） |
| 后台进程 | 支持 `&` 和进程管理 |
| 环境隔离 | 固定工作目录、环境变量控制 |
| 安全性 | 禁止危险命令或需要确认 |

**追问场景：**
- Q：Agent执行 `rm -rf /` 怎么办？
- A：需要多层防护：(1) **命令黑名单** — 预定义危险命令模式匹配；(2) **路径白名单** — 只允许在项目目录内操作；(3) **确认机制** — 高风险操作需用户确认；(4) **沙箱隔离** — 在Docker容器中执行；(5) **操作日志** — 所有命令可审计回溯。

---

## 三、Claude Code 的设计哲学 ⭐⭐⭐ 🔥🔥🔥

### Q6：Claude Code 的架构设计有哪些核心理念？

**A：** ⭐⭐⭐ 🔥🔥

Claude Code 是 Anthropic 推出的终端原生代码Agent，其设计哲学有几个核心点：

**1. 终端原生（Terminal-Native）**
```
$ claude "帮我重构 auth 模块，添加 JWT 支持"

Claude Code 不是IDE插件，而是终端工具。
这意味着它可以：
- 在CI/CD中运行
- 被其他脚本调用
- 在远程服务器上工作
- 集成到任何开发流程
```

**2. Headless 模式**
```bash
# 非交互式，适合自动化
claude -p "运行所有测试并修复失败的" --output-format json

# 在CI中使用
claude -p "检查PR #123的代码质量" --headless

# 管道式使用
echo "分析这段代码" | claude -p
```

Headless模式使Claude Code成为 **可编程的编码Agent**，而非仅仅是交互式工具。

**3. MCP（Model Context Protocol）集成**
```json
// Claude Code 通过MCP连接外部工具
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "..." }
    },
    "database": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-postgres"],
      "env": { "DATABASE_URL": "..." }
    }
  }
}
```

MCP让Claude Code的能力可以 **无限扩展** — 连接数据库、API、云服务等。

**4. 子Agent委派（Delegation）**
```
主Agent (Claude Code)
├── 子Agent 1: 负责前端组件重构
├── 子Agent 2: 负责后端API修改
└── 子Agent 3: 负责测试更新
```

复杂任务可以委派给子Agent并行处理。

**追问场景：**
- Q：Claude Code 为什么选择终端而非IDE作为主要交互界面？
- A：(1) **普适性** — 终端在所有环境中都可用，不依赖特定IDE；(2) **可组合性** — 遵循Unix哲学，可与其他工具管道组合；(3) **自动化友好** — 天然支持headless/脚本化运行；(4) **开发运维融合** — 同一工具可用于编码和部署；(5) **避免IDE耦合** — 不受VSCode等IDE的插件API限制。

---

### Q7：MCP 协议在代码Agent中起什么作用？

**A：** ⭐⭐⭐ 🔥🔥

MCP（Model Context Protocol）是 Anthropic 提出的开放协议，为LLM Agent提供标准化的外部工具/数据接入方式。

```
┌──────────────┐     MCP      ┌──────────────────┐
│  Claude Code │◄────────────►│  MCP Server      │
│  (Client)    │   JSON-RPC   │  (工具提供方)      │
└──────────────┘              ├──────────────────┤
                              │ • GitHub API     │
                              │ • Database       │
                              │ • File System    │
                              │ • Slack          │
                              │ • Custom Tools   │
                              └──────────────────┘
```

**MCP 核心概念：**
- **Resources** — 模型可读取的数据源（类似GET）
- **Tools** — 模型可调用的操作（类似POST）
- **Prompts** — 预定义的提示模板

**在代码Agent中的应用：**
```python
# MCP Server 示例：提供代码库搜索能力
@server.tool()
def search_codebase(query: str, file_pattern: str = "*") -> list:
    """在代码库中搜索代码"""
    results = ripgrep_search(query, file_pattern)
    return [{"file": r.path, "line": r.line, "content": r.text} for r in results]

@server.tool()
def get_git_history(file_path: str, limit: int = 10) -> list:
    """获取文件的git历史"""
    return git_log(file_path, limit)
```

**追问场景：**
- Q：MCP 和 OpenAI 的 Function Calling / Tool Use 有什么区别？
- A：Function Calling 是 **模型级别的协议** — 定义模型如何输出工具调用请求。MCP 是 **系统级别的协议** — 定义Agent如何 **发现、连接、使用** 外部工具服务。简单说：Function Calling 是"模型说我要调工具"，MCP 是"Agent如何找到并连接工具"。MCP更像是Agent的"USB接口"，而Function Calling是"通信语言"。

---

## 四、代码生成质量控制 ⭐⭐ 🔥

### Q8：如何保证Agent生成代码的质量？

**A：**

代码质量控制是多层防线的设计：

```
质量控制层次：

Layer 1: 生成时控制
├── System Prompt 约束（代码风格、最佳实践）
├── Few-shot 示例引导
└── 结构化输出约束

Layer 2: 静态检查
├── Lint（ESLint, Pylint, Ruff）
├── 类型检查（TypeScript, mypy, pyright）
└── 格式化（Prettier, Black）

Layer 3: 动态验证
├── 单元测试执行
├── 集成测试
└── 类型推断验证

Layer 4: 迭代修复
├── 错误信息反馈给Agent
├── Agent分析错误原因
└── 自动修复并重新验证

Layer 5: 人工审查
├── Diff Review
├── 关键逻辑确认
└── 合并前审批
```

**自动化流水线示例：**
```python
class QualityPipeline:
    def validate(self, code: str, context: dict) -> ValidationResult:
        # Step 1: Lint
        lint_result = self.run_lint(code)
        if lint_result.has_errors:
            return self.retry_with_feedback(lint_result.errors)

        # Step 2: Type Check
        type_result = self.run_type_check(code)
        if type_result.has_errors:
            return self.retry_with_feedback(type_result.errors)

        # Step 3: Tests
        test_result = self.run_tests(context["test_files"])
        if test_result.has_failures:
            return self.retry_with_feedback(test_result.failures)

        return ValidationResult(success=True)
```

**追问场景：**
- Q：Agent生成的代码通过了测试，但逻辑是错的（测试本身有漏洞），怎么处理？
- A：这正是纯自动化方案的局限。应对策略：(1) **测试质量监控** — 使用mutation testing检测测试覆盖率的有效性；(2) **多角度验证** — 除了功能测试，还做类型检查、lint、代码复杂度分析；(3) **关键路径人工review** — 核心业务逻辑必须人工确认；(4) **Agent自我审查** — 让Agent先写代码，再换一个"审查者"角色Agent来review。

---

### Q9：什么是测试驱动的Agentic Coding？

**A：** ⭐⭐ 🔥

```
TDD + Agent 工作流：

1. 用户描述需求
      ↓
2. Agent 生成测试用例
      ↓
3. 用户确认测试覆盖正确
      ↓
4. Agent 编写实现代码
      ↓
5. 运行测试 → 失败
      ↓
6. Agent 分析失败原因 → 修改代码
      ↓
7. 循环 5-6 直到全部通过
      ↓
8. Agent 重构优化代码
      ↓
9. 再次运行测试确认无回归
```

**这种方式的优势：**
- 测试即 **验收标准**，先确认标准再实现
- Agent的每次修改都有 **客观验证**
- 避免"看起来对但实际错"的问题
- 生成的代码天然具有可测试性

**追问场景：**
- Q：Agent 怎么知道要写哪些测试用例？
- A：Agent 通过分析需求描述生成测试用例，但更好的方式是：(1) 用户先确认边界条件和edge case；(2) Agent参考项目中已有测试的风格和模式；(3) 使用 **属性测试（Property-based Testing）** 自动生成边界用例；(4) 结合类型系统推断可能的边界情况（null、空集合、溢出等）。

---

## 五、大型代码库理解 ⭐⭐⭐ 🔥🔥

### Q10：代码Agent如何理解和处理大型代码库？

**A：** ⭐⭐⭐ 🔥🔥

大型代码库（10万+行）的理解是代码Agent的核心挑战。

**1. 代码库索引（Codebase Indexing）**

```
索引构建流程：

源代码文件
    ↓
┌───────────────────────┐
│  分块策略               │
│  ├── 按函数/类分块       │
│  ├── 按文件分块          │
│  └── 滑动窗口分块        │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│  向量化                 │
│  ├── Code Embedding    │
│  │   (text-embedding)  │
│  ├── AST 结构编码       │
│  └── Symbol Graph编码  │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│  存储                   │
│  ├── Vector Store      │
│  ├── Symbol Index      │
│  └── File Graph        │
└───────────────────────┘
```

**2. RAG for Code 的特殊挑战**

| 挑战 | 说明 | 解决方案 |
|------|------|---------|
| 语义 vs 结构 | 代码有语法结构，不纯是自然语言 | AST-aware chunking |
| 依赖关系 | 代码跨文件引用 | Import Graph + Symbol Index |
| 多语言 | 一个项目多种语言 | 语言特定的解析器 |
| 动态语义 | 同名函数在不同上下文含义不同 | 上下文感知的embedding |
| 增量更新 | 代码频繁变更 | 增量索引更新 |

**3. Cursor 的索引方案**
```
Cursor 使用远端索引 + 本地缓存：

1. 上传代码到远端（隐私保护处理）
2. 远端构建向量索引 + 符号索引
3. 查询时：本地缓存 + 远端RAG
4. 结合：语义搜索 + 符号跳转 + 文件图
```

**追问场景：**
- Q：RAG for Code 和普通的文本RAG有什么关键区别？
- A：(1) **分块策略不同** — 代码应按AST节点（函数/类）分块，而非固定长度窗口；(2) **需要结构信息** — 光靠语义相似度不够，需要知道import关系、类型定义、调用链；(3) **多跳推理** — "修改这个函数需要更新哪些文件"需要图遍历而非单次检索；(4) **精确性要求更高** — 代码检索召回错误文件会导致生成不可编译的代码；(5) **需要Symbol级别的索引** — 函数名、类名、变量名的精确定位。

---

### Q11：如何在有限的上下文窗口中处理大型代码库？

**A：** ⭐⭐⭐

```
上下文管理策略：

┌─────────────────────────────────────┐
│         上下文窗口 (200K tokens)      │
│                                     │
│  ┌─────────────────────────────┐   │
│  │  System Prompt (~2K)         │   │
│  ├─────────────────────────────┤   │
│  │  用户请求 + 对话历史 (~5K)    │   │
│  ├─────────────────────────────┤   │
│  │  核心文件内容 (~50K)          │   │
│  │  (当前编辑的文件)             │   │
│  ├─────────────────────────────┤   │
│  │  RAG 检索结果 (~30K)         │   │
│  │  (相关文件片段)              │   │
│  ├─────────────────────────────┤   │
│  │  工具调用结果 (~30K)          │   │
│  │  (搜索/测试输出)             │   │
│  ├─────────────────────────────┤   │
│  │  代码库概览 (~10K)           │   │
│  │  (目录树、关键文件摘要)       │   │
│  ├─────────────────────────────┤   │
│  │  预留空间 (~73K)             │   │
│  │  (迭代过程中的新增内容)       │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

**关键策略：**
1. **渐进式加载** — 先看目录结构，再看相关文件，按需深入
2. **摘要压缩** — 对不直接编辑的文件使用摘要而非全文
3. **上下文淘汰** — 移除过时的工具调用结果
4. **分层索引** — 目录级→文件级→函数级，逐层深入

**追问场景：**
- Q：当Agent需要修改一个被20个文件import的公共接口时，如何处理？
- A：这是最考验代码Agent能力的场景。策略：(1) 先用符号索引找出所有引用方；(2) 按引用类型分类（类型引用 vs 实际调用）；(3) 制定修改计划，确定哪些文件需要同步修改；(4) 分批修改，每批后运行类型检查和测试；(5) 如果上下文不够，使用子Agent并行处理不同文件组。

---

## 六、自主编码 vs 人机协作模式 ⭐⭐ 🔥

### Q12：自主编码和人机协作模式各有什么优劣？

**A：**

```
自主程度光谱：

完全人工 ◄────────────────────────────────────► 完全自主
   │          │           │           │           │
 纯手写    Tab补全     Chat辅助    Agent模式    全自主
           (Copilot)   (ChatGPT)  (Claude Code) (Devin)
```

| 维度 | 人机协作模式 | 自主编码模式 |
|------|------------|------------|
| 代表 | Cursor Agent, Copilot | Devin, OpenHands |
| 交互频率 | 高频确认 | 低频/事后审查 |
| 适用场景 | 复杂决策、核心逻辑 | 重复性任务、明确定义的任务 |
| 代码质量 | 高（人工把关） | 中等（依赖测试覆盖） |
| 效率 | 中等（等待确认） | 高（并行、无阻塞） |
| 风险 | 低 | 中高（可能走偏） |
| 信任要求 | 低 | 高 |

**追问场景：**
- Q：在生产环境中应该选择哪种模式？
- A：**分层策略**：(1) 重复性任务（格式化、迁移、样板代码）→ 自主模式；(2) 功能开发 → 人机协作，Agent实现+人审查；(3) 核心逻辑/安全相关 → 人工主导+Agent辅助；(4) Bug修复 → Agent定位+修复，人确认。关键是根据 **任务风险等级** 和 **可逆性** 来决定自主程度。

---

## 七、代码Agent的安全边界 ⭐⭐⭐ 🔥🔥

### Q13：代码Agent有哪些安全风险？如何防范？

**A：** ⭐⭐⭐ 🔥

```
安全威胁模型：

┌──────────────────────────────────────┐
│            安全威胁层次               │
├──────────────────────────────────────┤
│ L1: 代码注入                         │
│   ├── Prompt Injection via 代码注释   │
│   ├── 恶意依赖引入                    │
│   └── 代码执行中的命令注入             │
├──────────────────────────────────────┤
│ L2: 数据泄露                         │
│   ├── 代码发送到外部API               │
│   ├── 密钥/凭据暴露                   │
│   └── 私有代码泄露                    │
├──────────────────────────────────────┤
│ L3: 系统破坏                         │
│   ├── 误删文件                       │
│   ├── 执行危险命令                    │
│   └── 修改系统配置                    │
├──────────────────────────────────────┤
│ L4: 供应链攻击                       │
│   ├── 引入恶意npm/pip包              │
│   ├── 修改CI/CD配置                  │
│   └── 注入后门代码                    │
└──────────────────────────────────────┘
```

**防范措施：**

```python
class SecurityPolicy:
    # 1. 沙箱隔离
    sandbox = DockerSandbox(
        network="none",          # 禁止网络（可选白名单）
        read_only_paths=["/"],   # 系统目录只读
        writable_paths=["/workspace"],  # 只允许写项目目录
        resource_limits={"memory": "2GB", "cpu": "2"}
    )

    # 2. 命令过滤
    blocked_commands = [
        r"rm\s+-rf\s+/",        # 禁止递归删除根目录
        r"curl.*\|.*sh",        # 禁止管道执行
        r"chmod\s+777",         # 禁止过度权限
        r"sudo",                # 禁止提权
    ]

    # 3. 文件保护
    protected_paths = [
        ".env",
        "*.key",
        "*.pem",
        "secrets/",
        ".git/config",
    ]

    # 4. 确认机制
    require_confirmation_for = [
        "package_install",      # 安装依赖需确认
        "file_delete",          # 删除文件需确认
        "git_push",            # 推送代码需确认
        "config_change",       # 修改配置需确认
    ]
```

**追问场景：**
- Q：如何防止代码Agent被prompt injection攻击（通过代码注释注入恶意指令）？
- A：(1) **输入消毒** — 对读取的代码内容进行安全标记，区分"指令"和"数据"；(2) **最小权限** — Agent只拥有完成任务所需的最小权限；(3) **行为监控** — 检测Agent的异常行为模式（突然请求读取.env文件）；(4) **输出过滤** — Agent的工具调用需经过白名单验证；(5) **沙箱隔离** — 即使被注入，也无法造成外部影响。

---

## 八、前沿趋势 ⭐⭐⭐ 🔥🔥

### Q14：Agentic Coding 的未来发展方向是什么？

**A：**

**1. 多Agent协作编码**
```
Orchestrator Agent
├── Architect Agent: 系统设计、接口定义
├── Frontend Agent: 前端实现
├── Backend Agent: 后端实现
├── Test Agent: 测试编写与执行
└── Review Agent: 代码审查与质量把关
```

**2. SWE-bench 推动的全自主编码**
- SWE-bench / SWE-bench Verified 成为标准benchmark
- 目标：端到端解决真实GitHub Issue
- 当前SOTA已超过50%解决率

**3. 长上下文 + 代码理解融合**
- 1M+ token上下文窗口
- 整个代码库可装入上下文
- 但仍需要智能的注意力分配

**4. Agentic IDE 深度融合**
- IDE不再只是编辑器，而是Agent的工作环境
- Agent可感知IDE状态（打开的文件、光标位置、诊断信息）
- 人机交互更加自然（边编码边对话）

**追问场景：**
- Q：代码Agent会取代程序员吗？
- A：短期不会，但会深刻改变工作方式。(1) **重复性工作** 将被大幅自动化；(2) 程序员角色从"写代码"转向 **"定义需求+审查结果"**；(3) 程序员需要新技能：如何有效指导Agent、如何审查AI生成的代码；(4) **初级开发者的技能栈** 会发生变化，更多关注系统设计和问题分解；(5) 代码Agent更像是"超级结对编程伙伴"而非替代者。

---

## 九、面试高频考点总结 ⭐⭐ 🔥🔥

### 必考知识点

| # | 考点 | 难度 | 热度 | 考察形式 |
|---|------|------|------|---------|
| 1 | Agentic Coding vs 传统代码生成 | ⭐ | 🔥🔥 | 概念辨析 |
| 2 | 代码Agent的工具层设计 | ⭐⭐ | 🔥🔥 | 系统设计 |
| 3 | 文件操作的三种模式对比 | ⭐⭐ | 🔥 | 方案对比 |
| 4 | Claude Code 的 Headless 模式 | ⭐⭐ | 🔥🔥 | 产品理解 |
| 5 | MCP 协议与 Function Calling 区别 | ⭐⭐⭐ | 🔥🔥 | 协议理解 |
| 6 | RAG for Code 的特殊挑战 | ⭐⭐⭐ | 🔥🔥 | 技术深度 |
| 7 | 代码Agent的安全边界设计 | ⭐⭐⭐ | 🔥🔥 | 安全设计 |
| 8 | 测试驱动的Agentic Coding | ⭐⭐ | 🔥 | 方法论 |
| 9 | 大型代码库的上下文管理 | ⭐⭐⭐ | 🔥🔥 | 系统设计 |
| 10 | 自主 vs 协作模式选择策略 | ⭐⭐ | 🔥 | 产品思维 |

### 高频面试问题速查

1. **"设计一个代码Agent"** → 工具层 + 规划层 + 上下文管理 + 反馈循环
2. **"如何处理大代码库"** → 索引 + RAG + 渐进式加载 + 上下文压缩
3. **"如何保证代码质量"** → 多层质量门：Lint → 类型检查 → 测试 → Review
4. **"MCP是什么"** → Agent连接外部工具的标准协议，类似USB接口
5. **"代码Agent安全怎么保证"** → 沙箱 + 权限控制 + 命令过滤 + 确认机制
6. **"Cursor和Claude Code区别"** → IDE原生 vs 终端原生，不同设计哲学
7. **"代码Agent会取代程序员吗"** → 短期不会，改变角色而非取代

---

## 实战思考题

1. 如果让你为一个100万行的monorepo设计代码索引系统，你会怎么做？
2. 如何设计一个能在PR review中自动发现安全漏洞的Agent？
3. 多Agent协作编码时，如何处理Agent之间的代码冲突？
4. 如何评估一个代码Agent在真实项目中的ROI（投入产出比）？
5. 设计一个方案，让代码Agent能够"学习"团队的编码风格并保持一致。

---

> **下一章预告：** [33-Compound AI Systems](./33-Compound-AI-Systems.md) — 复合AI系统、多Agent协作、任务编排
