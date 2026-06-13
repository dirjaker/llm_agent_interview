# 💰 LLM 成本计算器

> 各主流模型定价 + 成本优化策略（2026 年最新）

---

## 目录

1. [2026 年主流模型定价表](#1-2026-年主流模型定价表)
2. [成本计算公式](#2-成本计算公式)
3. [场景成本估算](#3-场景成本估算)
4. [10 条成本优化策略](#4-10-条成本优化策略)
5. [成本监控代码示例](#5-成本监控代码示例-python)
6. [月度成本预算模板](#6-月度成本预算模板)

---

## 1. 2026 年主流模型定价表

> 价格单位：**美元 / 百万 tokens**（即 $/1M tokens）

### 1.1 顶级旗舰模型

| 模型 | 提供商 | 输入价格 | 输出价格 | 上下文窗口 | 备注 |
|------|--------|----------|----------|-----------|------|
| GPT-4o | OpenAI | $2.50 | $10.00 | 128K | 多模态旗舰 |
| GPT-4o-mini | OpenAI | $0.15 | $0.60 | 128K | 高性价比 |
| o3 | OpenAI | $10.00 | $40.00 | 200K | 推理模型 |
| o4-mini | OpenAI | $1.10 | $4.40 | 200K | 轻量推理 |
| Claude Opus 4 | Anthropic | $15.00 | $75.00 | 200K | 最强推理 |
| Claude Sonnet 4 | Anthropic | $3.00 | $15.00 | 200K | 均衡首选 |
| Claude Haiku 3.5 | Anthropic | $0.80 | $4.00 | 200K | 快速轻量 |

### 1.2 中国主流模型

| 模型 | 提供商 | 输入价格 | 输出价格 | 上下文窗口 | 备注 |
|------|--------|----------|----------|-----------|------|
| DeepSeek-V3 | DeepSeek | $0.27 | $1.10 | 128K | 国产旗舰 |
| DeepSeek-R1 | DeepSeek | $0.55 | $2.19 | 128K | 推理模型 |
| Qwen3-235B | 阿里云 | $0.40 | $1.20 | 128K | MoE 架构 |
| Qwen3-30B | 阿里云 | $0.10 | $0.30 | 128K | 轻量级 |
| GLM-4-Plus | 智谱 | $0.70 | $2.80 | 128K | 多模态 |
| ERNIE 4.5 | 百度 | $0.56 | $2.24 | 128K | 国产旗舰 |
| 豆包-pro | 字节 | $0.35 | $1.40 | 128K | 高性价比 |

### 1.3 Google 模型

| 模型 | 提供商 | 输入价格 | 输出价格 | 上下文窗口 | 备注 |
|------|--------|----------|----------|-----------|------|
| Gemini 2.5 Pro | Google | $1.25 / $2.50 | $10.00 / $15.00 | 1M | ≤/>200K 差价 |
| Gemini 2.5 Flash | Google | $0.15 / $0.30 | $0.60 / $1.20 | 1M | 快速经济 |
| Gemini 2.0 Flash | Google | $0.10 | $0.40 | 1M | 便宜大窗口 |

### 1.4 开源 / 自部署模型

| 模型 | 参数量 | 推荐 GPU | 自部署成本估算 | 备注 |
|------|--------|---------|--------------|------|
| Llama 4 Maverick | 400B (MoE 17B active) | 8&times; A100 80G | ~$8/h | Meta 开源 |
| Llama 4 Scout | 109B (MoE 17B active) | 4&times; A100 80G | ~$4/h | 轻量版 |
| Qwen3-30B-A3B | 30B (MoE 3B active) | 1&times; A100 | ~$1.5/h | 极致性价比 |
| DeepSeek-V3 | 671B (MoE 37B active) | 8&times; H100 | ~$12/h | 自部署旗舰 |

---

## 2. 成本计算公式

### 2.1 基础公式

```
单次调用成本 = (输入 tokens × 输入价格 + 输出 tokens × 输出价格) / 1,000,000
```

用数学公式表示：

```
Cost = (N_in × P_in + N_out × P_out) / 1,000,000
```

其中：
- `N_in` = 输入 token 数量
- `N_out` = 输出 token 数量
- `P_in` = 每百万输入 token 价格（$）
- `P_out` = 每百万输出 token 价格（$）

### 2.2 计算示例

```
场景：使用 Claude Sonnet 4 处理一个 RAG 请求
- 输入：系统提示 500 tokens + 检索文档 3000 tokens + 用户问题 200 tokens = 3,700 tokens
- 输出：模型回答 800 tokens

Cost = (3,700 × $3.00 + 800 × $15.00) / 1,000,000
     = ($11,100 + $12,000) / 1,000,000
     = $23,100 / 1,000,000
     = $0.0231 ≈ ¥0.17
```

### 2.3 月度成本公式

```
月度成本 = 日均请求量 × 单次平均成本 × 30
```

```
示例：日均 10,000 次请求，平均每次 $0.02
月度成本 = 10,000 × $0.02 × 30 = $6,000/月
```

### 2.4 与 Token 等价换算

| 语言 | 1000 tokens ≈ |
|------|---------------|
| 英文 | 750 个单词 |
| 中文 | 500-700 个汉字 |
| 代码 | 100-150 行 Python |

---

## 3. 场景成本估算

### 3.1 场景对比总览

| 场景 | 平均输入 tokens | 平均输出 tokens | GPT-4o 成本 | DeepSeek-V3 成本 | Claude Sonnet 4 成本 |
|------|----------------|----------------|------------|-----------------|---------------------|
| 简单对话 | 500 | 200 | $0.003 | $0.0004 | $0.005 |
| RAG 问答 | 3,500 | 500 | $0.014 | $0.002 | $0.018 |
| Agent 单轮 | 5,000 | 1,500 | $0.028 | $0.003 | $0.038 |
| Agent 多轮(5步) | 25,000 | 7,500 | $0.14 | $0.015 | $0.19 |
| 代码生成 | 2,000 | 2,000 | $0.025 | $0.003 | $0.036 |
| 文档摘要(10K字) | 6,000 | 1,000 | $0.025 | $0.003 | $0.033 |
| 批量分类(每条) | 300 | 50 | $0.001 | $0.0001 | $0.002 |

### 3.2 对话场景

```
典型配置：
- 系统提示：200 tokens
- 历史对话（10轮）：2,000 tokens
- 用户最新消息：100 tokens
- 模型回复：300 tokens

每日 5,000 用户 × 10 轮对话 = 50,000 次调用
使用 GPT-4o-mini：
  日成本 = 50,000 × (2,300 × $0.15 + 300 × $0.60) / 1M = $26.25/天
  月成本 = ~$788

使用 DeepSeek-V3：
  日成本 = 50,000 × (2,300 × $0.27 + 300 × $1.10) / 1M = $47.55/天
  月成本 = ~$1,427

使用 GPT-4o：
  日成本 = 50,000 × (2,300 × $2.50 + 300 × $10.00) / 1M = $437.50/天
  月成本 = ~$13,125
```

### 3.3 RAG 场景

```
典型配置：
- 系统提示：300 tokens
- 检索 Top-5 文档片段：3,000 tokens
- 用户问题：200 tokens
- 对话历史：1,000 tokens
- 模型回复：500 tokens

每日 2,000 次 RAG 查询
使用 Claude Sonnet 4：
  日成本 = 2,000 × (4,500 × $3.00 + 500 × $15.00) / 1M = $42.00/天
  月成本 = ~$1,260

使用 Qwen3-235B：
  日成本 = 2,000 × (4,500 × $0.40 + 500 × $1.20) / 1M = $4.80/天
  月成本 = ~$144
```

### 3.4 Agent 场景

```
典型配置（单次任务平均 5 步工具调用）：
- 每步输入：系统 300 + 工具描述 500 + 历史 2,000 + 当前观察 1,000 = 3,800 tokens
- 每步输出：思考 + 工具调用 600 tokens
- 最终回复：800 tokens

总 tokens：输入 3,800 × 5 + 800 = 19,800；输出 600 × 5 + 800 = 3,800

每日 500 个 Agent 任务
使用 GPT-4o：
  日成本 = 500 × (19,800 × $2.50 + 3,800 × $10.00) / 1M = $43.75/天
  月成本 = ~$1,313

使用 DeepSeek-V3：
  日成本 = 500 × (19,800 × $0.27 + 3,800 × $1.10) / 1M = $4.76/天
  月成本 = ~$143
```

### 3.5 批量处理场景

```
场景：对 100 万条数据做分类
- 每条输入：指令 100 + 数据 200 = 300 tokens
- 每条输出：分类标签 20 tokens

使用 GPT-4o-mini：
  总成本 = 1,000,000 × (300 × $0.15 + 20 × $0.60) / 1M = $57.00

使用 DeepSeek-V3：
  总成本 = 1,000,000 × (300 × $0.27 + 20 × $1.10) / 1M = $103.00

使用批处理 API（通常 50% 折扣）：
  GPT-4o-mini 批量 = $28.50
```

---

## 4. 10 条成本优化策略

### 策略 1：模型路由分层（Model Routing） ⭐ 节省 40-70%

```
思路：简单任务用便宜模型，复杂任务用贵模型
路由规则：
  - 简单分类/提取 → GPT-4o-mini / Qwen3-30B
  - 标准问答 → DeepSeek-V3 / Qwen3-235B
  - 复杂推理/代码 → Claude Sonnet 4 / GPT-4o
  - 关键决策 → Claude Opus 4 / o3

效果：整体成本下降 40-70%
```

### 策略 2：Prompt 压缩 ⭐ 节省 20-50%

```
方法：
  - 精简系统提示（去除冗余指令）
  - 压缩检索文档（摘要替代原文）
  - 滑动窗口管理对话历史
  - 使用 Prompt 压缩库（如 LLMLingua）

效果：输入 token 减少 20-50%
```

### 策略 3：缓存策略 ⭐ 节省 30-60%

```
方法：
  - 精确匹配缓存：相同输入直接返回
  - 语义缓存：相似问题复用结果
  - Prompt 缓存：利用 API 的 prompt caching（如 OpenAI 自动缓存前缀）

实现：
  - Redis + 哈希缓存（精确匹配）
  - 向量数据库 + 相似度阈值（语义缓存）

效果：命中率 30-60% 时节省等比例成本
```

### 策略 4：批量处理 API ⭐ 节省 50%

```
方法：
  - 使用 OpenAI Batch API（50% 折扣，24h 内返回）
  - 使用 Anthropic Message Batches API
  - 非实时任务统一走批量接口

适用场景：数据标注、文档分类、内容生成
效果：直接 50% 折扣
```

### 策略 5：流式输出 + 提前终止 ⭐ 节省 10-30%

```
方法：
  - 设置 max_tokens 限制输出长度
  - 检测到完整答案后提前终止
  - 对结构化输出使用 JSON mode 减少冗余

效果：输出 token 减少 10-30%
```

### 策略 6：微调小模型替代大模型 ⭐ 节省 60-90%

```
方法：
  - 用大模型生成训练数据
  - 微调 GPT-4o-mini 或开源小模型
  - 特定任务上小模型 + 微调 ≈ 大模型效果

成本对比：
  大模型 $10/M tokens → 小模型微调后 $0.15/M tokens
  微调成本：一次性 $50-500

效果：长期成本降低 60-90%
```

### 策略 7：结构化输出 + 减少重试 ⭐ 节省 10-20%

```
方法：
  - 使用 JSON Schema 约束输出格式
  - 使用 Instructor / Outlines 库
  - 减少因格式错误导致的重试

效果：减少 10-20% 无效调用
```

### 策略 8：Token 计数预检 ⭐ 节省 5-15%

```
方法：
  - 发送前用 tiktoken 预估 token 数
  - 超出预期时自动截断/压缩
  - 设置预算告警

效果：避免意外的高成本调用
```

### 策略 9：混合本地 + 云端 ⭐ 节省 50-80%

```
方法：
  - 简单任务本地部署小模型（Qwen3-30B-A3B 等）
  - 复杂任务调用云端 API
  - 使用 Ollama / vLLM 本地部署

效果：可将 70% 的简单请求卸载到本地
```

### 策略 10：异步 + 重试优化 ⭐ 节省 5-10%

```
方法：
  - 使用指数退避重试避免限流浪费
  - 并发控制避免超出 RPM 限制
  - 使用 Semaphore 控制并发数

效果：减少因限流导致的无效调用
```

### 优化策略汇总

| # | 策略 | 预估节省 | 实施难度 | 优先级 |
|---|------|---------|---------|-------|
| 1 | 模型路由分层 | 40-70% | ⭐⭐ | 🔴 高 |
| 2 | Prompt 压缩 | 20-50% | ⭐⭐ | 🔴 高 |
| 3 | 缓存策略 | 30-60% | ⭐⭐⭐ | 🔴 高 |
| 4 | 批量处理 API | 50% | ⭐ | 🟡 中 |
| 5 | 流式 + 提前终止 | 10-30% | ⭐ | 🟡 中 |
| 6 | 微调小模型 | 60-90% | ⭐⭐⭐⭐ | 🟡 中 |
| 7 | 结构化输出 | 10-20% | ⭐ | 🟢 低 |
| 8 | Token 预检 | 5-15% | ⭐ | 🟢 低 |
| 9 | 本地 + 云端混合 | 50-80% | ⭐⭐⭐ | 🟡 中 |
| 10 | 异步重试优化 | 5-10% | ⭐ | 🟢 低 |

---

## 5. 成本监控代码示例 (Python)

### 5.1 基础 Token 计数与成本计算

```python
"""
LLM 成本监控器 - 基础版本
"""

import tiktoken
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ========== 定价配置 ==========
PRICING = {
    "gpt-4o":            {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
    "o3":                {"input": 10.00, "output": 40.00},
    "o4-mini":           {"input": 1.10,  "output": 4.40},
    "claude-opus-4":     {"input": 15.00, "output": 75.00},
    "claude-sonnet-4":   {"input": 3.00,  "output": 15.00},
    "claude-haiku-3.5":  {"input": 0.80,  "output": 4.00},
    "deepseek-v3":       {"input": 0.27,  "output": 1.10},
    "deepseek-r1":       {"input": 0.55,  "output": 2.19},
    "qwen3-235b":        {"input": 0.40,  "output": 1.20},
    "qwen3-30b":         {"input": 0.10,  "output": 0.30},
    "gemini-2.5-pro":    {"input": 1.25,  "output": 10.00},
    "gemini-2.5-flash":  {"input": 0.15,  "output": 0.60},
}


@dataclass
class UsageRecord:
    """单次调用记录"""
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    metadata: dict = field(default_factory=dict)


class CostTracker:
    """LLM 成本追踪器"""

    def __init__(self, monthly_budget: Optional[float] = None):
        self.records: list[UsageRecord] = []
        self.monthly_budget = monthly_budget  # 美元

    def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """估算 token 数量"""
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> tuple[float, float, float]:
        """计算单次调用成本，返回 (input_cost, output_cost, total_cost)"""
        if model not in PRICING:
            raise ValueError(f"未知模型: {model}，可用: {list(PRICING.keys())}")

        pricing = PRICING[model]
        input_cost = (input_tokens * pricing["input"]) / 1_000_000
        output_cost = (output_tokens * pricing["output"]) / 1_000_000
        return input_cost, output_cost, input_cost + output_cost

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[dict] = None,
    ) -> UsageRecord:
        """记录一次调用"""
        input_cost, output_cost, total_cost = self.calculate_cost(
            model, input_tokens, output_tokens
        )

        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            metadata=metadata or {},
        )
        self.records.append(record)

        # 预算检查
        if self.monthly_budget:
            monthly_total = self.get_monthly_total()
            usage_pct = (monthly_total / self.monthly_budget) * 100
            if usage_pct > 80:
                print(f"⚠️  警告：本月已用 {usage_pct:.1f}% 预算 (${monthly_total:.2f}/${self.monthly_budget:.2f})")
            if usage_pct > 100:
                print(f"🚨 超出预算！本月已花费 ${monthly_total:.2f}")

        return record

    def get_monthly_total(self) -> float:
        """获取本月总成本"""
        now = datetime.now()
        month_prefix = now.strftime("%Y-%m")
        return sum(
            r.total_cost for r in self.records
            if r.timestamp.startswith(month_prefix)
        )

    def get_summary(self) -> dict:
        """获取汇总统计"""
        if not self.records:
            return {"total_cost": 0, "total_calls": 0}

        from collections import defaultdict
        model_stats = defaultdict(lambda: {"calls": 0, "cost": 0, "tokens": 0})

        for r in self.records:
            stats = model_stats[r.model]
            stats["calls"] += 1
            stats["cost"] += r.total_cost
            stats["tokens"] += r.input_tokens + r.output_tokens

        return {
            "total_cost": sum(r.total_cost for r in self.records),
            "total_calls": len(self.records),
            "total_tokens": sum(r.input_tokens + r.output_tokens for r in self.records),
            "by_model": dict(model_stats),
        }

    def print_report(self):
        """打印成本报告"""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("📊 LLM 成本报告")
        print("=" * 60)
        print(f"总调用次数: {summary['total_calls']:,}")
        print(f"总 Token 数: {summary['total_tokens']:,}")
        print(f"总成本: ${summary['total_cost']:.4f}")
        print(f"\n按模型统计:")
        print("-" * 60)
        for model, stats in summary.get("by_model", {}).items():
            avg = stats["cost"] / stats["calls"] if stats["calls"] else 0
            print(f"  {model:25s} | {stats['calls']:>6} 次 | "
                  f"${stats['cost']:>10.4f} | 均价 ${avg:.6f}")
        if self.monthly_budget:
            usage = (self.get_monthly_total() / self.monthly_budget) * 100
            bar = "█" * int(usage / 5) + "░" * (20 - int(usage / 5))
            print(f"\n预算使用: [{bar}] {usage:.1f}%")
        print("=" * 60)


# ========== 使用示例 ==========
if __name__ == "__main__":
    tracker = CostTracker(monthly_budget=1000.0)

    # 模拟各种调用
    tracker.record("gpt-4o", input_tokens=3500, output_tokens=500,
                   metadata={"task": "rag_qa", "user": "user_001"})
    tracker.record("deepseek-v3", input_tokens=2000, output_tokens=300,
                   metadata={"task": "chat", "user": "user_002"})
    tracker.record("claude-sonnet-4", input_tokens=5000, output_tokens=1500,
                   metadata={"task": "agent", "user": "user_001"})
    tracker.record("gpt-4o-mini", input_tokens=300, output_tokens=50,
                   metadata={"task": "classification"})

    tracker.print_report()
```

### 5.2 与 OpenAI API 集成的实时成本追踪

```python
"""
与 OpenAI SDK 集成，自动追踪每次调用的成本
"""

from openai import OpenAI


class CostAwareClient:
    """带成本追踪的 OpenAI 客户端"""

    def __init__(self, tracker: CostTracker, **kwargs):
        self.client = OpenAI(**kwargs)
        self.tracker = tracker

    def chat(self, model: str, messages: list, **kwargs) -> str:
        """带成本追踪的聊天调用"""
        response = self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

        usage = response.usage
        self.tracker.record(
            model=model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            metadata={"model": model},
        )

        return response.choices[0].message.content

    def chat_with_budget_check(
        self, model: str, messages: list, max_cost: float = 0.10, **kwargs
    ) -> str:
        """带预算检查的调用 - 超过单次上限时降级到便宜模型"""
        # 预估成本
        input_text = " ".join(m["content"] for m in messages if isinstance(m["content"], str))
        est_input = self.tracker.count_tokens(input_text, model)
        est_output = kwargs.get("max_tokens", 500)
        _, _, est_cost = self.tracker.calculate_cost(model, est_input, est_output)

        if est_cost > max_cost:
            # 降级到便宜模型
            fallback = "gpt-4o-mini"
            print(f"⚡ 预估 ${est_cost:.4f} > 上限 ${max_cost}，降级到 {fallback}")
            model = fallback

        return self.chat(model, messages, **kwargs)
```

---

## 6. 月度成本预算模板

### 6.1 预算表模板

```markdown
| 项目 | 模型 | 日均请求量 | 平均输入/次 | 平均输出/次 | 单次成本 | 日成本 | 月成本 |
|------|------|-----------|-----------|-----------|---------|-------|-------|
| 客服对话 | GPT-4o-mini | 5,000 | 2,000 | 300 | $0.0005 | $2.48 | $74 |
| RAG 知识库 | DeepSeek-V3 | 2,000 | 4,500 | 500 | $0.002 | $3.31 | $99 |
| Agent 工作流 | Claude Sonnet 4 | 200 | 19,800 | 3,800 | $0.12 | $23.40 | $702 |
| 数据分类 | GPT-4o-mini (批量) | 50,000 | 300 | 20 | $0.00006 | $2.85 | $86 |
| 内容生成 | GPT-4o | 100 | 3,000 | 2,000 | $0.028 | $2.75 | $83 |
| **合计** | | | | | | **$34.79** | **$1,044** |
```

### 6.2 预算分级建议

```
┌─────────────────────────────────────────────────────────┐
│  🟢 基础级：$500/月以内                                    │
│  适用：小型项目、个人开发、MVP 验证                            │
│  策略：全部使用 DeepSeek/Qwen + 缓存                       │
├─────────────────────────────────────────────────────────┤
│  🟡 成长级：$500 - $5,000/月                               │
│  适用：中型产品、企业内部工具                                  │
│  策略：模型分层 + 缓存 + 批量处理                             │
├─────────────────────────────────────────────────────────┤
│  🔴 规模级：$5,000 - $50,000/月                             │
│  适用：大型 SaaS、高频调用场景                                │
│  策略：全部优化手段 + 自部署混合                               │
├─────────────────────────────────────────────────────────┤
│  ⚫ 企业级：$50,000+/月                                     │
│  适用：平台级产品、亿级用户                                   │
│  策略：自部署为主 + API 补充 + 定制优化                        │
└─────────────────────────────────────────────────────────┘
```

### 6.3 成本监控仪表盘指标

```yaml
# 推荐监控指标
核心指标:
  - 日均成本 (daily_cost)
  - 月度累计成本 (monthly_cost)
  - 剩余预算天数 (runway_days)
  - 成本/请求 (cost_per_request)

维度分析:
  - 按模型分布 (cost_by_model)
  - 按业务线分布 (cost_by_service)
  - 按用户分布 (cost_by_user)
  - 按时间段分布 (cost_by_hour)

告警规则:
  - 日成本超过日均 200% → 即时告警
  - 月度预算使用超过 80% → 预警
  - 月度预算使用超过 100% → 紧急告警
  - 单次调用成本超过 $1.00 → 异常检测
```

---

## 附录：快速对比卡片

```
🏆 最便宜旗舰：DeepSeek-V3 ($0.27/$1.10)
🏆 最便宜轻量：Qwen3-30B ($0.10/$0.30)
🏆 最大上下文：Gemini 2.5 (1M tokens)
🏆 最强推理：Claude Opus 4 ($15/$75)
🏆 最佳性价比：GPT-4o-mini ($0.15/$0.60)
🏆 最便宜推理：DeepSeek-R1 ($0.55/$2.19)
🏆 自部署性价比王：Qwen3-30B-A3B (1×A100)
```

---

> 📅 最后更新：2026 年 6 月
> ⚠️ 价格可能随时调整，请以各厂商官方定价页面为准
> 🔗 OpenAI: https://openai.com/pricing
> 🔗 Anthropic: https://www.anthropic.com/pricing
> 🔗 DeepSeek: https://platform.deepseek.com/pricing
> 🔗 阿里云: https://help.aliyun.com/zh/model-studio/pricing
