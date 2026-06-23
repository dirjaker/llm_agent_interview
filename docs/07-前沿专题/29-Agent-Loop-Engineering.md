# 29. Agent Loop Engineering (循环工程)

> **难度标记**：⭐⭐⭐ 高级
> **热度**：🔥🔥🔥 2025-2026 Agent 架构核心考点
> **关键词**：ReAct、Reflexion、迭代优化、循环控制、错误恢复

---

## 知识图谱

```
Agent Loop Engineering
├── 基础循环模式 ⭐⭐⭐ 必知
│   ├── ReAct (Reasoning + Acting)
│   ├── Observe-Orient-Decide-Act (OODA)
│   ├── Plan-Execute-Reflect
│   └── 事件驱动循环
├── 迭代优化模式 ⭐⭐⭐ 核心
│   ├── Reflexion (自我反思)
│   ├── Self-Refine (自我精炼)
│   ├── Tree of Thoughts (思维树)
│   └── 迭代 RAG
├── 循环控制策略 ⭐⭐⭐ 生产必备
│   ├── 终止条件设计
│   ├── 最大迭代限制
│   ├── 代价预算控制
│   └── 质量收敛检测
├── 错误恢复机制 ⭐⭐ 进阶
│   ├── 自我纠错 (Self-Correction)
│   ├── 回滚策略 (Rollback)
│   ├── 降级方案 (Fallback)
│   └── 人类干预 (Human-in-the-Loop)
├── 并行循环 ⭐⭐ 高级
│   ├── 并行分支探索
│   ├── 投票/聚合决策
│   └── 竞争式执行
└── 循环模式对比 ⭐⭐ 选型
    ├── 单步 vs 多步
    ├── 同步 vs 异步
    └── 串行 vs 并行
```

---

## 题目 29.1：ReAct 循环详解

**Q：请详细解释 ReAct (Reasoning + Acting) 的工作原理，以及它为什么是 Agent 的基础范式。**

### 核心答案

```
ReAct 循环流程
┌─────────────────────────────────────┐
│  1. Thought (思考)                   │
│     模型分析当前状态，规划下一步        │
│     "我需要查询用户的订单信息"         │
├─────────────────────────────────────┤
│  2. Action (行动)                    │
│     选择并执行工具                    │
│     → tool_call: get_order(user_id) │
├─────────────────────────────────────┤
│  3. Observation (观察)               │
│     获取工具返回结果                  │
│     → {order_id: "123", status: "..."}│
├─────────────────────────────────────┤
│  4. 循环判断                         │
│     ├── 任务完成 → 返回结果           │
│     └── 未完成 → 回到步骤 1           │
└─────────────────────────────────────┘
```

**ReAct 为什么是基础范式**：

| 特性 | 说明 |
|------|------|
| **可解释性** | Thought 过程清晰展示推理链 |
| **灵活性** | 可调用任意工具，不限于预定义流程 |
| **容错性** | 每步观察结果，可动态调整策略 |
| **通用性** | 适用于问答、代码、数据分析等各种场景 |

### 追问场景

**面试官**：ReAct 的 Thought 和 CoT 的 Chain-of-Thought 有什么区别？

**回答要点**：
- **CoT**：纯推理，不与环境交互，一次性生成答案
- **ReAct Thought**：推理 + 行动规划，需要根据 Observation 动态调整
- **关键区别**：ReAct 的 Thought 是**交互式**的，每步都依赖环境反馈

---

## 题目 29.2：Reflexion 自我反思循环

**Q：请解释 Reflexion 框架的工作原理，它如何通过反思提升 Agent 表现？**

### 核心答案

```
Reflexion 三阶段循环

阶段 1：执行 (Actor)
├── Agent 尝试完成任务
├── 生成行动轨迹 τ = [a1, o1, a2, o2, ...]
└── 输出结果或失败

阶段 2：评估 (Evaluator)
├── 评估任务是否成功
├── 生成评估信号（成功/失败/分数）
└── 提供具体错误信息

阶段 3：反思 (Self-Reflection)
├── 分析失败原因
├── 生成自然语言反思
├── 存入长期记忆 (memory buffer)
└── 下次执行时参考历史反思

循环：反思 → 改进执行 → 重新评估 → 更深反思
```

**Reflexion 的核心创新**：

```
传统 Agent：执行 → 失败 → 重试（无记忆）

Reflexion：执行 → 失败 → 反思 → 存储反思 → 重试（带反思记忆）

反思记忆示例：
├── 尝试 1："用 requests 库访问 API，超时了"
├── 反思 1："API 可能有速率限制，应该添加重试和退避"
├── 尝试 2："添加了 retry 和 exponential backoff"
├── 反思 2："重试解决了超时，但响应格式解析错误"
└── 尝试 3："检查 API 文档，使用正确的解析方式"
```

### 代码示例：Reflexion 实现

```python
class ReflexionAgent:
    def __init__(self, llm, max_attempts=3):
        self.llm = llm
        self.max_attempts = max_attempts
        self.reflections = []  # 长期记忆
    
    def run(self, task: str) -> str:
        for attempt in range(self.max_attempts):
            # 1. 执行（带反思记忆）
            result, trajectory = self.execute(task)
            
            # 2. 评估
            success = self.evaluate(task, result)
            if success:
                return result
            
            # 3. 反思
            reflection = self.reflect(task, trajectory, result)
            self.reflections.append(reflection)
        
        return "达到最大尝试次数"
    
    def execute(self, task):
        # 构建包含历史反思的 prompt
        prompt = f"任务: {task}\n"
        if self.reflections:
            prompt += "历史反思:\n"
            for r in self.reflections:
                prompt += f"- {r}\n"
        
        return self.llm.generate_with_tools(prompt)
    
    def reflect(self, task, trajectory, result):
        prompt = f"""任务: {task}
执行轨迹: {trajectory}
结果: {result}
请分析失败原因，总结经验教训："""
        return self.llm.generate(prompt)
```

### 追问场景

**面试官**：Reflexion 的反思记忆会不会无限增长？如何控制？

**回答要点**：
1. **滑动窗口**：只保留最近 N 条反思
2. **重要性过滤**：只保留关键反思，丢弃琐碎的
3. **总结压缩**：定期将多条反思合并为总结
4. **相关性检索**：使用 RAG 检索相关反思，而非全部注入

---

## 题目 29.3：循环终止条件设计

**Q：在生产环境中，如何设计 Agent 循环的终止条件？如何防止无限循环？**

### 核心答案

```
终止条件层次设计

├── 硬性终止（必须有）
│   ├── 最大迭代次数 (max_iterations)
│   │   ├── 简单任务：5-10 次
│   │   ├── 复杂任务：20-50 次
│   │   └── 代码生成：50-100 次
│   ├── 最大 token 消耗 (max_tokens)
│   │   └── 防止成本失控
│   ├── 最大执行时间 (timeout)
│   │   └── 防止阻塞
│   └── 最大工具调用次数
│       └── 防止工具滥用
│
├── 软性终止（推荐有）
│   ├── 任务完成检测
│   │   ├── LLM 自判断
│   │   ├── 规则匹配
│   │   └── 评估函数
│   ├── 质量收敛检测
│   │   ├── 连续 N 步结果无改进
│   │   └── 分数变化 < 阈值
│   └── 重复检测
│       ├── 相同工具 + 相同参数
│       └── 循环模式识别
│
└── 紧急终止（安全网）
    ├── 错误累积阈值
    ├── 异常模式检测
    └── 人工干预触发
```

### 终止条件实现

```python
class LoopController:
    def __init__(self, config):
        self.max_iterations = config.get("max_iterations", 20)
        self.max_tokens = config.get("max_tokens", 100000)
        self.max_time = config.get("max_time", 300)  # 秒
        self.convergence_window = config.get("convergence_window", 3)
        self.iteration = 0
        self.total_tokens = 0
        self.start_time = time.time()
        self.recent_scores = []
    
    def should_continue(self, result, score=None) -> tuple[bool, str]:
        # 硬性检查
        if self.iteration >= self.max_iterations:
            return False, "达到最大迭代次数"
        
        if self.total_tokens >= self.max_tokens:
            return False, "达到 token 预算上限"
        
        if time.time() - self.start_time > self.max_time:
            return False, "超时"
        
        # 软性检查
        if score is not None:
            self.recent_scores.append(score)
            if len(self.recent_scores) >= self.convergence_window:
                recent = self.recent_scores[-self.convergence_window:]
                if max(recent) - min(recent) < 0.01:
                    return False, "质量收敛，无改进"
        
        # 重复检测
        if self._detect_loop(result):
            return False, "检测到循环模式"
        
        self.iteration += 1
        return True, "继续"
    
    def _detect_loop(self, result) -> bool:
        # 检测最近 N 步是否重复
        # ...
        pass
```

### 追问场景

**面试官**：如果 Agent 在循环中反复调用同一个工具但参数不同，算不算循环？

**回答要点**：
- **不算循环**，但需要监控
- 如果参数在**有意义地变化**（如不同搜索词），是正常的探索
- 如果参数**微小变化**（如只是格式不同），可能是无效循环
- 建议：设置**工具调用多样性阈值**，检测参数相似度

---

## 题目 29.4：并行循环与竞争执行

**Q：请介绍 Agent 的并行循环模式，以及如何在多个候选方案中选择最优。**

### 核心答案

```
并行循环模式

├── 模式 1：并行探索 (Fan-Out)
│   ├── 同时执行多个独立子任务
│   ├── 各分支独立循环
│   └── 结果聚合
│   例：同时搜索 3 个数据源，合并结果
│
├── 模式 2：竞争执行 (Race)
│   ├── 同一任务多策略并行
│   ├── 最先成功的返回
│   └── 其他分支取消
│   例：同时尝试 3 种代码方案，用第一个通过测试的
│
├── 模式 3：投票决策 (Ensemble)
│   ├── 多个 Agent 独立回答
│   ├── 投票或加权聚合
│   └── 提高答案可靠性
│   例：3 个 Agent 分别判断，多数票决定
│
└── 模式 4：辩论模式 (Debate)
    ├── 多个 Agent 提出观点
    ├── 互相质疑和改进
    └── 达成共识
    例：架构评审，正反方辩论
```

### 实现示例

```python
import asyncio
from typing import List

class ParallelAgentLoop:
    async def race(self, task: str, strategies: List[str]) -> str:
        """竞争模式：最快成功的返回"""
        tasks = [
            self.execute_with_strategy(task, strategy)
            for strategy in strategies
        ]
        
        # return_when=FIRST_COMPLETED
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )
        
        # 取消未完成的
        for task in pending:
            task.cancel()
        
        return done.pop().result()
    
    async def ensemble(self, task: str, n_agents: int = 3) -> str:
        """投票模式：多数票决定"""
        results = await asyncio.gather(*[
            self.agent_execute(task) for _ in range(n_agents)
        ])
        
        # 投票
        from collections import Counter
        votes = Counter(results)
        return votes.most_common(1)[0][0]
    
    async def fan_out(self, subtasks: List[str]) -> List[str]:
        """扇出模式：并行执行子任务"""
        results = await asyncio.gather(*[
            self.agent_execute(task) for task in subtasks
        ])
        return results
```

### 追问场景

**面试官**：并行执行会增加成本，如何决定什么时候用并行？

**回答要点**：
1. **独立子任务** → 并行（如同时查多个数据源）
2. **高风险决策** → 投票（如医疗诊断、金融决策）
3. **时间敏感** → 竞争（如实时响应）
4. **简单任务** → 串行（并行开销不值得）
5. **成本敏感** → 串行优先，必要时并行

---

## 题目 29.5：循环中的状态管理

**Q：Agent 循环执行过程中，如何管理状态？如何支持中断恢复？**

### 核心答案

```
Agent 状态管理

├── 会话状态 (Session State)
│   ├── 当前对话历史
│   ├── 工具调用记录
│   ├── 中间结果
│   └── 生命周期：单次会话
│
├── 任务状态 (Task State)
│   ├── 任务目标
│   ├── 当前进度
│   ├── 子任务状态
│   └── 生命周期：单个任务
│
├── 持久状态 (Persistent State)
│   ├── 长期记忆
│   ├── 学到的技能
│   ├── 用户偏好
│   └── 生命周期：跨会话
│
└── 检查点 (Checkpoint)
    ├── 循环快照
    ├── 支持回滚
    └── 支持恢复
```

### 检查点与恢复

```python
class CheckpointManager:
    def save_checkpoint(self, agent_state: dict) -> str:
        """保存检查点"""
        checkpoint = {
            "timestamp": time.time(),
            "iteration": agent_state["iteration"],
            "tool_history": agent_state["tool_history"],
            "intermediate_results": agent_state["results"],
            "context": agent_state["context"],
        }
        checkpoint_id = str(uuid.uuid4())
        self.storage.save(checkpoint_id, checkpoint)
        return checkpoint_id
    
    def restore(self, checkpoint_id: str) -> dict:
        """恢复检查点"""
        return self.storage.load(checkpoint_id)
    
    def rollback(self, agent_state: dict) -> dict:
        """回滚到上一个检查点"""
        checkpoints = self.storage.list_recent(
            agent_state["session_id"]
        )
        if len(checkpoints) >= 2:
            return self.restore(checkpoints[-2].id)
        return agent_state
```

### 追问场景

**面试官**：在长循环任务中，如何避免上下文窗口溢出？

**回答要点**：
1. **滑动窗口**：只保留最近 N 轮对话
2. **摘要压缩**：定期将历史压缩为摘要
3. **工具结果缓存**：历史工具结果存储在外部，按需检索
4. **分层记忆**：工作记忆（上下文）+ 长期记忆（外部存储）
5. **阶段性重置**：子任务完成后重置上下文，只保留关键结论

---

## 面试高频考点总结

```
✅ 必须掌握
├── ReAct 循环的完整流程
├── Reflexion 的反思机制
├── 循环终止条件的层次设计
├── 并行循环的四种模式
└── 状态管理与检查点机制

✅ 加分项
├── 循环中的成本控制策略
├── 防止无限循环的具体实现
├── 并行 vs 串行的决策框架
└── 长循环任务的上下文管理
```