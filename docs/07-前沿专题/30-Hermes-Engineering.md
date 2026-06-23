# 30. Hermes Engineering (代理工程设计哲学)

> **难度标记**：⭐⭐⭐ 高级
> **热度**：🔥🔥🔥 2025-2026 Agent 系统设计核心考点
> **关键词**：Skill System、Memory、Delegation、Context Engineering、Production Agent

---

## 知识图谱

```
Hermes Engineering
├── 设计哲学 ⭐⭐⭐ 核心理念
│   ├── 渐进式复杂度
│   ├── 工具优先 (Tool-First)
│   ├── 技能系统 (Skill System)
│   └── 人在回路 (Human-in-the-Loop)
├── 技能系统设计 ⭐⭐⭐ 必考
│   ├── Skill 定义与元数据
│   ├── Skill 发现与加载
│   ├── Skill 组合与编排
│   └── Skill 生命周期管理
├── 记忆系统架构 ⭐⭐⭐ 核心
│   ├── 短期记忆 (Session Context)
│   ├── 长期记忆 (Persistent Memory)
│   ├── 程序性记忆 (Skills/Procedures)
│   └── 记忆注入与检索
├── 任务委派 ⭐⭐ 进阶
│   ├── 子代理 (Sub-Agent)
│   ├── 并行委派
│   ├── 编排器模式
│   └── 结果聚合
├── 上下文工程 ⭐⭐⭐ 生产必备
│   ├── 系统提示设计
│   ├── 动态上下文组装
│   ├── 上下文预算管理
│   └── 上下文压缩策略
└── 生产化工程 ⭐⭐ 工程实践
    ├── 错误处理与重试
    ├── 成本控制
    ├── 可观测性
    └── 安全边界
```

---

## 题目 30.1：Agent 技能系统设计

**Q：请设计一个 Agent 技能系统，支持技能的定义、发现、加载和组合。**

### 核心答案

```
技能系统架构

├── Skill 定义 (SKILL.md)
│   ├── 元数据：name, description, category
│   ├── 触发条件：什么时候使用这个技能
│   ├── 执行步骤：具体操作指南
│   ├── 引用文件：references/, templates/, scripts/
│   └── 版本管理
│
├── Skill 发现
│   ├── 静态注册：配置文件声明
│   ├── 动态发现：运行时扫描
│   ├── 语义匹配：Embedding 相似度
│   └── 关键词匹配：触发条件匹配
│
├── Skill 加载
│   ├── 懒加载：只在需要时加载
│   ├── 按需注入：注入到系统提示
│   ├── 分层加载：核心技能常驻，辅助技能按需
│   └── 依赖解析：技能间依赖管理
│
└── Skill 组合
    ├── 串行组合：A → B → C
    ├── 并行组合：A || B || C
    ├── 条件组合：if A then B else C
    └── 嵌套组合：技能内调用其他技能
```

### Skill 文件结构

```
~/.hermes/skills/
├── github-pr-workflow/
│   ├── SKILL.md              # 主技能文件
│   ├── references/
│   │   ├── gh-cli-ref.md     # gh 命令参考
│   │   └── git-workflow.md   # Git 工作流
│   ├── templates/
│   │   └── pr-template.md    # PR 模板
│   └── scripts/
│       └── check-pr.sh       # 检查脚本
├── python-debugging/
│   ├── SKILL.md
│   └── references/
│       └── pdb-cheatsheet.md
└── ...
```

### SKILL.md 格式

```markdown
---
name: github-pr-workflow
description: GitHub PR lifecycle: branch, commit, open, CI, merge
category: software-development
version: 1.0.0
triggers:
  - "create pull request"
  - "open PR"
  - "merge branch"
---

# GitHub PR Workflow

## When to Use
- User asks to create a pull request
- User wants to merge a feature branch
- User needs to review PR changes

## Steps

### 1. Create Branch
```bash
git checkout -b feature/xxx
```

### 2. Make Changes & Commit
```bash
git add .
git commit -m "feat: description"
```

### 3. Push & Create PR
```bash
git push origin feature/xxx
gh pr create --title "..." --body "..."
```

## Pitfalls
- Always use `dev` branch, not `main`
- Check CI status before merging
- Squash commits for clean history
```

### 追问场景

**面试官**：技能系统和传统的工具注册有什么区别？

**回答要点**：
| 维度 | 工具注册 | 技能系统 |
|------|----------|----------|
| 粒度 | 单个函数/API | 完整工作流 |
| 知识 | 参数签名 | 步骤 + 最佳实践 + 踩坑 |
| 组合 | 手动编排 | 声明式组合 |
| 学习 | 静态 | 可动态学习新技能 |
| 上下文 | 无 | 包含参考文档和模板 |

---

## 题目 30.2：Agent 记忆系统架构

**Q：请设计一个 Agent 的分层记忆系统，包括短期、长期和程序性记忆。**

### 核心答案

```
三层记忆架构

├── 短期记忆 (Working Memory / Session Context)
│   ├── 当前对话历史
│   ├── 当前任务状态
│   ├── 工具调用结果
│   ├── 生命周期：单次会话
│   ├── 存储位置：上下文窗口
│   └── 容量限制：模型上下文长度
│
├── 长期记忆 (Long-Term Memory / Episodic Memory)
│   ├── 用户偏好
│   ├── 环境信息
│   ├── 历史决策
│   ├── 跨会话持久化
│   ├── 存储位置：文件/数据库
│   └── 检索方式：相关性匹配
│
└── 程序性记忆 (Procedural Memory / Skills)
    ├── 操作步骤
    ├── 最佳实践
    ├── 踩坑经验
    ├── 存储位置：技能文件
    └── 加载方式：触发条件匹配
```

### 记忆注入机制

```python
class MemorySystem:
    def __init__(self):
        self.short_term = SessionContext()
        self.long_term = PersistentMemory()
        self.procedural = SkillRegistry()
    
    def build_context(self, user_message: str) -> str:
        """构建注入到系统提示的记忆上下文"""
        context_parts = []
        
        # 1. 长期记忆（用户偏好、环境信息）
        long_term_mem = self.long_term.retrieve()
        if long_term_mem:
            context_parts.append(
                f"## User Profile\n{long_term_mem}"
            )
        
        # 2. 程序性记忆（相关技能）
        relevant_skills = self.procedural.find_relevant(
            user_message
        )
        if relevant_skills:
            for skill in relevant_skills:
                context_parts.append(
                    f"## Skill: {skill.name}\n{skill.content}"
                )
        
        # 3. 短期记忆（对话历史）由模型自动处理
        
        return "\n\n".join(context_parts)
    
    def save_memory(self, key: str, value: str, target: str):
        """保存记忆"""
        if target == "user":
            self.long_term.save_user_profile(key, value)
        elif target == "memory":
            self.long_term.save_system_note(key, value)
```

### 记忆管理原则

```
记忆写入原则
├── 主动保存：用户纠正、偏好、重要事实
├── 去重：避免重复信息
├── 精简：保持条目简洁
└── 时效：过期信息及时清理

记忆读取原则
├── 相关性：只检索相关内容
├── 优先级：用户偏好 > 环境事实 > 历史
├── 压缩：注入时精简表述
└── 预算：控制注入的记忆量
```

### 追问场景

**面试官**：如何决定哪些信息应该存入长期记忆？

**回答要点**：
1. **用户偏好**：语言、格式、工具偏好 → 必存
2. **环境信息**：OS、项目结构、已安装工具 → 必存
3. **纠正信息**：用户说"不要这样做" → 必存
4. **任务结果**：完成的任务 → 不存（临时状态）
5. **调试过程**：排查步骤 → 技能化存储（不存原始日志）

---

## 题目 30.3：任务委派与子代理

**Q：请设计 Agent 的任务委派机制，支持并行子代理和结果聚合。**

### 核心答案

```
委派架构模式

├── 模式 1：叶子委派 (Leaf Delegation)
│   ├── 主代理 → 子代理（不可再委派）
│   ├── 子代理独立执行
│   ├── 返回结果给主代理
│   └── 适合：独立子任务
│
├── 模式 2：编排器委派 (Orchestrator)
│   ├── 编排器 → 多个子代理
│   ├── 子代理可继续委派
│   ├── 编排器协调和聚合
│   └── 适合：复杂多步骤任务
│
├── 模式 3：并行委派 (Parallel)
│   ├── 主代理 → [子代理A, 子代理B, 子代理C]
│   ├── 并行执行
│   ├── 结果聚合
│   └── 适合：独立并行任务
│
└── 模式 4：竞争委派 (Race)
    ├── 主代理 → [子代理A, 子代理B]
    ├── 最快成功的返回
    └── 适合：高可靠性场景
```

### 委派实现

```python
class DelegationManager:
    def __init__(self, max_concurrent=3, max_depth=2):
        self.max_concurrent = max_concurrent
        self.max_depth = max_depth
    
    async def delegate(
        self,
        tasks: List[Task],
        parent_context: str
    ) -> List[Result]:
        """并行委派多个任务"""
        # 限制并发数
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def run_with_limit(task):
            async with semaphore:
                return await self.run_subagent(
                    task, parent_context
                )
        
        results = await asyncio.gather(*[
            run_with_limit(task) for task in tasks
        ])
        
        return results
    
    async def run_subagent(
        self, task: Task, parent_context: str
    ) -> Result:
        """运行子代理"""
        subagent = SubAgent(
            # 子代理继承父代理的工具集
            tools=self.get_allowed_tools(task),
            # 子代理有独立的上下文
            context=self.build_subagent_context(
                task, parent_context
            ),
            # 限制子代理不能再委派（叶子模式）
            can_delegate=self.depth < self.max_depth,
        )
        
        return await subagent.execute(task)
```

### 子代理上下文隔离

```
子代理设计原则

├── 隔离性
│   ├── 独立对话上下文
│   ├── 独立工具集（按需授权）
│   ├── 独立工作目录
│   └── 不共享中间状态
│
├── 约束性
│   ├── 最大执行时间
│   ├── 最大 token 消耗
│   ├── 工具白名单
│   └── 委派深度限制
│
└── 通信
    ├── 父 → 子：任务描述 + 上下文
    ├── 子 → 父：执行结果 + 摘要
    └── 子不能访问父的完整历史
```

### 追问场景

**面试官**：子代理失败了怎么办？如何处理部分失败？

**回答要点**：
1. **重试**：同一子代理重试（带错误信息）
2. **降级**：换一个更简单的策略
3. **回退**：主代理自己执行
4. **部分成功**：返回已完成部分，标记失败部分
5. **用户通知**：严重失败时通知用户

---

## 题目 30.4：上下文工程 (Context Engineering)

**Q：什么是上下文工程？如何设计一个高效的动态上下文组装系统？**

### 核心答案

```
上下文工程 = 管理 LLM 在每次调用时"看到"的信息

上下文组成
├── 系统提示 (System Prompt)
│   ├── 角色定义
│   ├── 行为规则
│   ├── 工具说明
│   └── 约束条件
│
├── 动态注入 (Dynamic Injection)
│   ├── 用户记忆/偏好
│   ├── 相关技能
│   ├── 环境信息
│   └── 任务上下文
│
├── 对话历史 (Conversation History)
│   ├── 用户消息
│   ├── 助手回复
│   ├── 工具调用结果
│   └── 系统事件
│
└── 工具定义 (Tool Definitions)
    ├── 函数签名
    ├── 参数描述
    └── 使用示例
```

### 动态上下文组装

```python
class ContextAssembler:
    def __init__(self, max_tokens: int = 128000):
        self.max_tokens = max_tokens
        self.budget = TokenBudget(max_tokens)
    
    def assemble(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: List[Message],
        memory: MemorySystem,
        skills: SkillRegistry,
    ) -> str:
        """动态组装上下文"""
        
        # 1. 固定开销（必须保留）
        self.budget.reserve("system_prompt", system_prompt)
        self.budget.reserve("tools", self.get_tools_definition())
        self.budget.reserve("current_message", user_message)
        
        # 2. 动态分配剩余预算
        remaining = self.budget.remaining()
        
        # 3. 按优先级分配
        # 优先级 1：用户记忆（精简版）
        user_mem = memory.get_user_profile()
        self.budget.allocate("memory", user_mem, priority=1)
        
        # 优先级 2：相关技能
        relevant_skills = skills.find_relevant(user_message)
        for skill in relevant_skills:
            self.budget.allocate(
                f"skill:{skill.name}", 
                skill.content, 
                priority=2
            )
        
        # 优先级 3：对话历史（滑动窗口）
        history = self.trim_history(
            conversation_history, 
            self.budget.remaining()
        )
        self.budget.allocate("history", history, priority=3)
        
        return self.budget.build_context()
    
    def trim_history(
        self, history: List[Message], budget: int
    ) -> str:
        """滑动窗口 + 摘要压缩"""
        # 保留最近 N 轮
        recent = history[-10:]  # 最近 10 轮
        
        # 如果超出预算，压缩旧历史
        if self.count_tokens(recent) > budget:
            old = history[:-10]
            summary = self.summarize(old)
            return summary + "\n" + self.format_messages(recent)
        
        return self.format_messages(recent)
```

### 上下文预算管理

```
Token 预算分配示例 (128K 上下文)

├── 系统提示: 2K (固定)
├── 工具定义: 5K (固定)
├── 当前消息: 1K (固定)
├── 用户记忆: 1K
├── 相关技能: 5-15K (按需)
├── 对话历史: 50-80K (滑动窗口)
└── 预留空间: 20K (工具结果、回复)

关键策略
├── 固定部分预分配
├── 动态部分按优先级分配
├── 超出预算时压缩低优先级
└── 始终预留回复空间
```

### 追问场景

**面试官**：上下文太长会影响模型表现吗？如何平衡信息量和模型能力？

**回答要点**：
1. **Lost in the Middle**：模型对上下文中间部分的关注度较低
2. **信息密度**：精炼 > 冗长，避免注入无关信息
3. **分层策略**：核心信息放开头/结尾，辅助信息放中间
4. **RAG vs 长上下文**：长上下文适合需要全局理解的任务，RAG 适合精确检索
5. **实际测试**：不同模型的长上下文能力差异很大

---

## 题目 30.5：Agent 安全边界设计

**Q：如何设计 Agent 的安全边界？如何防止 Agent 执行危险操作？**

### 核心答案

```
安全边界层次

├── 层 1：权限控制
│   ├── 工具白名单
│   ├── 文件路径限制
│   ├── 命令执行限制
│   └── 网络访问限制
│
├── 层 2：输入验证
│   ├── 参数范围检查
│   ├── 路径遍历防护
│   ├── 命令注入防护
│   └── 敏感信息过滤
│
├── 层 3：执行确认
│   ├── 危险操作需用户确认
│   ├── 批量操作需二次确认
│   ├── 不可逆操作需明确同意
│   └── 自动审批白名单
│
├── 层 4：资源限制
│   ├── 最大执行时间
│   ├── 最大 token 消耗
│   ├── 最大文件大小
│   └── 最大并发数
│
└── 层 5：审计日志
    ├── 所有工具调用记录
    ├── 用户确认记录
    ├── 错误和异常记录
    └── 可回溯和复盘
```

### 安全实现示例

```python
class SafetyGuard:
    def __init__(self, config):
        self.dangerous_commands = [
            "rm -rf", "sudo rm", "mkfs", "dd if=",
            "chmod 777", "> /dev/sda"
        ]
        self.protected_paths = [
            "/etc", "/boot", "/sys", "/proc"
        ]
        self.auto_approve = config.get("auto_approve", [])
    
    def check_tool_call(
        self, tool: str, args: dict
    ) -> tuple[bool, str]:
        """检查工具调用是否安全"""
        
        # 命令执行检查
        if tool == "terminal":
            cmd = args.get("command", "")
            for dangerous in self.dangerous_commands:
                if dangerous in cmd:
                    return False, f"危险命令: {dangerous}"
        
        # 文件操作检查
        if tool in ["write_file", "read_file", "patch"]:
            path = args.get("path", "")
            for protected in self.protected_paths:
                if path.startswith(protected):
                    return False, f"受保护路径: {protected}"
        
        # 是否需要用户确认
        if tool not in self.auto_approve:
            return None, "需要用户确认"  # None = 待确认
        
        return True, "安全"
```

### 追问场景

**面试官**：Agent 有执行代码的能力，如何防止它被 prompt injection 攻击？

**回答要点**：
1. **输入清洗**：过滤用户输入中的恶意指令
2. **工具隔离**：代码执行在沙箱环境中
3. **权限最小化**：只授予必要的权限
4. **确认机制**：敏感操作需用户确认
5. **监控告警**：异常行为实时告警

---

## 面试高频考点总结

```
✅ 必须掌握
├── 技能系统的完整设计
├── 三层记忆架构的实现
├── 任务委派的模式和隔离性
├── 上下文工程的动态组装
└── 安全边界的层次设计

✅ 加分项
├── 技能发现和组合的算法
├── 记忆压缩和检索策略
├── 子代理失败处理机制
├── 上下文预算管理策略
└── 防 prompt injection 的具体措施
```