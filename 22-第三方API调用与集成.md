# 22. 第三方 API 调用与集成

> 本章聚焦 LLM 应用中调用第三方 API（大模型 API、工具 API、数据服务 API）的核心工程问题。涵盖超时重试、熔断降级、限流、密钥安全、流式处理、多模型 Fallback、成本控制等高频面试考点。

---

## LLM API 调用基础

### Q: 调用第三方大模型 API（如 OpenAI、DeepSeek）时，需要注意哪些关键问题？ ⭐⭐⭐

**答：** 调用第三方大模型 API 需要关注 **6 个维度**，可以用口诀 **"超重熔限签缓"** 记忆：

**1. 超时（Timeout）**
- LLM 生成耗时远超普通 API（几秒到几十秒），必须设置合理的超时
- 区分 **连接超时**（TCP 握手，5-10s）和 **读取超时**（等待生成，30-120s）
- 流式请求的读取超时需要更长（逐 token 输出，间隔可能较大）

**2. 重试（Retry）**
- 网络错误、5xx、429（限流）可以重试
- 4xx（除 429）是客户端错误，不应重试（如参数错误、认证失败）
- 使用 **指数退避 + 随机抖动**，避免惊群效应
- 遵守服务端返回的 `Retry-After` 头

**3. 熔断（Circuit Breaker）**
- 当下游 API 连续失败超过阈值，自动"熔断"——后续请求直接快速失败
- 防止一个故障的 API 拖垮整个系统
- 冷却后自动进入半开状态试探恢复

**4. 限流（Rate Limiting）**
- 客户端主动限流，不等服务端返回 429
- 使用令牌桶或滑动窗口控制并发
- 多租户场景下按用户/项目分配配额

**5. 签名与认证（Auth）**
- API Key 不能硬编码，用环境变量或 Secret Manager
- 支持 Key 轮转（Rotation），新旧 Key 并行期
- 请求签名防篡改（HMAC-SHA256）

**6. 缓存（Cache）**
- 相同 Prompt 的结果可以缓存（开发/测试阶段尤其有用）
- 使用语义缓存（Embedding 相似度匹配）减少调用次数
- 注意缓存 TTL，模型更新后缓存可能失效

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 生产级 LLM API 调用模板
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
)
async def call_llm(prompt: str, model: str = "deepseek-chat") -> str:
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        response = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        )
        
        # 429: 限流，尊重 Retry-After
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 5))
            raise httpx.HTTPStatusError(
                f"Rate limited, retry after {retry_after}s",
                request=response.request, response=response,
            )
        
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
```

---

## 超时与重试

### Q: 第三方 API 调用超时了，系统会怎样？如何设计合理的超时策略？ ⭐⭐⭐

**答：** 超时的影响和应对分三层：

**超时的连锁反应：**
```
API 超时 → 请求堆积 → 线程/连接耗尽 → 服务雪崩
```
如果超时设置太长，一个慢 API 会占满所有连接池，导致整个服务不可用。如果太短，正常请求被误杀。

**超时分层设计：**

| 层级 | 超时类型 | 推荐值 | 说明 |
|------|----------|--------|------|
| 连接超时 | `connect_timeout` | 5-10s | TCP 握手时间 |
| 读取超时 | `read_timeout` | 30-120s | 等待 LLM 生成 |
| 总超时 | `total_timeout` | 60-300s | 整个请求生命周期 |
| 链路超时 | 逐级递减 | A>B>C | 调用链中每层减少 10-20% |

**关键原则：**
- **链路超时递减**：A 调用 B 调用 C，A 的超时 > B 的超时 > C 的超时
- **重试预算**：总超时 = 单次超时 × (1 + 重试次数)，不能超过上游给的超时
- **LLM 特殊处理**：流式响应用 `stream_timeout`（两个 chunk 之间的最大间隔）

```python
import httpx

# 分场景超时配置
TIMEOUT_CONFIGS = {
    "chat": httpx.Timeout(120.0, connect=5.0),        # 对话生成，允许较长
    "embedding": httpx.Timeout(30.0, connect=5.0),     # 向量化，较快
    "tool_call": httpx.Timeout(15.0, connect=3.0),     # 工具调用，要快
    "stream": httpx.Timeout(300.0, connect=5.0, read=30.0),  # 流式，总超时长但 chunk 间隔短
}
```

### Q: 重试策略有哪些？为什么用指数退避而不是固定间隔？ ⭐⭐

**答：**

**三种重试策略对比：**

| 策略 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| 固定间隔 | 每次等 2s | 简单 | 多客户端同时重试造成流量尖峰 |
| 指数退避 | 1s → 2s → 4s → 8s | 逐渐拉开间隔 | 仍可能同时重试 |
| 指数退避 + 抖动 | 1s → 1.8s → 3.2s → 7.1s（随机） | 最佳分散效果 | 略复杂 |

**为什么需要抖动（Jitter）？**

假设 100 个客户端同时请求失败，都用固定 2s 间隔重试：
- 2s 后 → 100 个请求同时到达（惊群效应）
- 服务端刚恢复又被打垮

加抖动后，重试时间分散在 [0, 4s] 范围内，流量平滑。

**哪些错误该重试？**

| 错误类型 | 是否重试 | 原因 |
|----------|----------|------|
| 网络超时/连接失败 | ✅ 重试 | 通常是瞬时问题 |
| 500/502/503/504 | ✅ 重试 | 服务端临时故障 |
| 429 | ✅ 重试（等待 Retry-After） | 限流，等一下就好 |
| 400/401/403/404 | ❌ 不重试 | 客户端错误，重试也没用 |
| 408 | ⚠️ 可选 | 超时，可能成功也可能失败 |

```python
import random
import asyncio

async def retry_with_backoff(func, max_retries=3, base_delay=1.0, max_delay=60.0):
    """指数退避 + 完全抖动"""
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if attempt == max_retries:
                raise
            
            # 指数退避
            exp_delay = min(base_delay * (2 ** attempt), max_delay)
            # 完全抖动：在 [0, exp_delay] 均匀随机
            delay = random.uniform(0, exp_delay)
            
            # 如果服务端返回 Retry-After，优先使用
            if hasattr(e, 'response') and e.response:
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    delay = max(delay, float(retry_after))
            
            await asyncio.sleep(delay)
```

---

## 熔断器模式

### Q: 什么是熔断器模式？和重试有什么区别？ ⭐⭐⭐

**答：**

**核心区别：**
- **重试**：针对 **单次请求** 的容错——这次失败了，再试一次
- **熔断**：针对 **服务整体** 的保护——这个服务已经不行了，别再试了

**类比**：重试是你打一个人电话打不通再打一次；熔断是发现这个人的电话一直打不通，暂时不打了，过一会儿再试。

**三种状态：**
```
关闭（正常） → 打开（熔断） → 半开（试探） → 关闭（恢复）
     ↑                                    |
     └────────────────────────────────────┘
```

| 状态 | 行为 | 触发条件 |
|------|------|----------|
| **关闭** | 请求正常通过 | 初始状态 |
| **打开** | 请求直接失败（快速失败） | 连续失败 ≥ 阈值（如 5 次） |
| **半开** | 允许少量请求试探 | 冷却时间到了（如 30s） |

**为什么需要同时使用重试 + 熔断？**

```
场景：下游 API 完全挂了

只有重试：每个请求都重试 3 次 × 每次等 60s = 每个用户等 3 分钟才失败
有熔断器：连续 5 个请求失败后熔断打开，后续请求直接返回降级结果（毫秒级）
```

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"       # 正常
    OPEN = "open"           # 熔断
    HALF_OPEN = "half_open" # 半开

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30, success_threshold=2):
        self.state = CircuitState.CLOSED
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

    async def call(self, func):
        if self.state == CircuitState.OPEN:
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("熔断器打开，请求被拒绝")
        
        try:
            result = await func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        else:
            self.failure_count = 0

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

---

## 限流处理

### Q: 收到 429（Too Many Requests）响应后应该怎么做？如何在客户端做限流？ ⭐⭐

**答：**

**收到 429 后的处理步骤：**
1. **立即停止发送新请求**
2. **读取 `Retry-After` 头**，按它指定的秒数等待
3. **如果没有 `Retry-After` 头**，用指数退避计算等待时间
4. **降低后续请求速率**（自适应限流）

**客户端限流 vs 服务端限流：**

| 维度 | 客户端限流 | 服务端限流 |
|------|-----------|-----------|
| 目的 | 保护自己不过度调用 | 保护服务不被打垮 |
| 位置 | 调用方 | 提供方 |
| 算法 | 令牌桶、滑动窗口 | 令牌桶、漏桶、固定窗口 |
| 响应 | 本地排队/延迟 | 返回 429 |

**常见限流算法：**

| 算法 | 原理 | 特点 |
|------|------|------|
| **令牌桶** | 固定速率生成令牌，请求消耗令牌 | 允许突发流量 |
| **漏桶** | 固定速率处理请求 | 平滑流量，不允许突发 |
| **滑动窗口** | 统计最近 N 秒内的请求数 | 精确但内存开销大 |

**令牌桶 vs 漏桶的区别：**
- 漏桶：水（请求）从桶底固定速率流出，满了就溢出（拒绝）
- 令牌桶：桶里装令牌，请求来了取令牌，桶空了就等待
- **关键区别**：令牌桶允许桶里积累令牌，短时间可以处理突发流量

```python
import time
import asyncio

class TokenBucket:
    """令牌桶限流器"""
    def __init__(self, rate: float, capacity: int):
        self.rate = rate          # 每秒生成的令牌数
        self.capacity = capacity  # 桶的最大容量
        self.tokens = capacity
        self.last_refill = time.monotonic()

    async def acquire(self, tokens=1):
        while True:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            # 计算等待时间
            wait = (tokens - self.tokens) / self.rate
            await asyncio.sleep(wait)

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

# 使用示例：限制每秒最多 10 次 LLM 调用
limiter = TokenBucket(rate=10, capacity=20)

async def call_with_limit(prompt):
    await limiter.acquire()  # 超限时自动等待
    return await call_llm(prompt)
```

---

## API Key 安全管理

### Q: 如何安全管理第三方 API Key？泄露了怎么办？ ⭐⭐⭐

**答：**

**安全存储的优先级（从高到低）：**
1. **Secret Manager**（Vault、AWS Secrets Manager、阿里云 KMS）—— 生产环境首选
2. **环境变量** —— 开发/测试可用
3. **加密配置文件** —— 需要管理解密密钥
4. ~~硬编码在代码中~~ —— **绝对禁止**

**API Key 管理最佳实践：**

| 实践 | 说明 |
|------|------|
| **最小权限** | 只授予必要的权限（只读、特定模型） |
| **定期轮转** | 每 30-90 天更换一次 Key |
| **分离环境** | 开发/测试/生产用不同的 Key |
| **审计日志** | 记录每次 Key 的使用（谁、何时、调了什么） |
| **IP 白名单** | 限制 Key 只能从特定 IP 调用 |
| **额度限制** | 设置每日/每月的调用上限 |

**Key 泄露的应急流程：**
```
1. 立即吊销泄露的 Key
2. 生成新 Key 并更新所有使用处
3. 审计日志：检查泄露 Key 的所有调用记录
4. 排查泄露途径（Git 历史、日志、网络抓包）
5. 通知 API 提供方（有些平台会帮你审查异常调用）
6. 复盘：引入 Secret Manager，CI 扫描硬编码 Key
```

```python
import os
from functools import lru_cache

class APIKeyManager:
    """API Key 管理器，支持多供应商和轮转"""
    
    def __init__(self):
        self._keys = {}
    
    def register(self, provider: str, key: str, expires_in: int = None):
        """注册 Key，可设置过期时间"""
        self._keys[provider] = {
            "key": key,
            "expires_at": time.time() + expires_in if expires_in else None,
        }
    
    def get(self, provider: str) -> str:
        """获取 Key，过期则抛异常"""
        info = self._keys.get(provider)
        if not info:
            raise KeyError(f"未注册: {provider}")
        if info["expires_at"] and time.time() > info["expires_at"]:
            raise PermissionError(f"Key 已过期: {provider}")
        return info["key"]
    
    def rotate(self, provider: str, new_key: str, overlap_seconds: int = 3600):
        """
        Key 轮转：新旧 Key 并行期
        为什么需要并行期？分布式系统中不同节点可能缓存了旧 Key，
        并行期允许旧请求仍然被接受，避免轮转瞬间的请求失败。
        """
        self.register(provider, new_key)

# 使用环境变量
manager = APIKeyManager()
manager.register("deepseek", os.environ["DEEPSEEK_API_KEY"])
manager.register("openai", os.environ["OPENAI_API_KEY"])
```

---

## 流式响应处理

### Q: 如何正确处理大模型 API 的流式响应（Streaming）？ ⭐⭐⭐

**答：**

**为什么需要流式？**
- LLM 生成一个完整回复可能需要 5-30 秒
- 非流式：用户盯着空白页等 10 秒 → 体验极差
- 流式：第一个 token 在 0.5-1 秒内出现 → 用户感觉"在思考"

**流式响应的工程挑战：**

| 挑战 | 说明 | 解决方案 |
|------|------|----------|
| 超时判断 | 两个 chunk 之间间隔多长算超时？ | 用 chunk 间超时（如 30s），不用总超时 |
| 错误处理 | 流到一半报错了怎么办？ | 已输出的内容无法撤回，追加错误提示 |
| Token 计数 | 流式模式下怎么统计 token？ | 用 tiktoken 本地计数，或等 `usage` 字段 |
| 断线重连 | WebSocket/SSE 断了怎么办？ | 记录已生成长度，重新请求后跳过 |
| 并发控制 | 多用户同时流式请求 | 每个请求独立协程，共享连接池 |

```python
import httpx
from typing import AsyncIterator

async def stream_llm(prompt: str, model: str = "deepseek-chat") -> AsyncIterator[str]:
    """流式调用 LLM，逐 token 返回"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0, read=30.0)) as client:
        async with client.stream(
            "POST",
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                # SSE 格式：data: {...}\n\n
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        import json
                        data = json.loads(line[6:])
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]

# 使用示例
async def main():
    async for token in stream_llm("解释量子计算"):
        print(token, end="", flush=True)
```

---

## 多模型 Fallback

### Q: 如何设计多模型 Fallback 机制？ ⭐⭐⭐

**答：**

**为什么要 Fallback？**
- 单一 API 供应商可能宕机、限流、涨价
- 不同模型擅长不同任务（代码用 DeepSeek，创意用 GPT-4o）
- 成本优化：简单任务用便宜模型，复杂任务用强模型

**Fallback 策略：**

```
主模型（DeepSeek）→ 备用模型1（Qwen）→ 备用模型2（GPT-4o-mini）→ 降级（缓存/默认值）
```

**设计要点：**
1. **优先级链**：按成本/质量排序，便宜的在前
2. **快速切换**：主模型超时后立即切备用，不要等太久
3. **健康检查**：定期探测各模型的可用性
4. **状态追踪**：记录每个模型的成功率/延迟，动态调整优先级

```python
from dataclasses import dataclass
from typing import Optional
import httpx
import logging

@dataclass
class ModelConfig:
    name: str
    api_url: str
    api_key: str
    model_id: str
    priority: int              # 越小越优先
    timeout: float = 60.0
    max_retries: int = 1
    is_healthy: bool = True
    success_rate: float = 1.0

class ModelRouter:
    """多模型 Fallback 路由器"""
    
    def __init__(self, models: list[ModelConfig]):
        self.models = sorted(models, key=lambda m: m.priority)
    
    async def call(self, messages: list[dict], **kwargs) -> str:
        """按优先级尝试调用，失败则 Fallback"""
        last_error = None
        
        for model in self.models:
            if not model.is_healthy:
                logging.info(f"跳过不健康的模型: {model.name}")
                continue
            
            try:
                result = await self._call_model(model, messages, **kwargs)
                model.success_rate = model.success_rate * 0.95 + 0.05  # 滑动平均
                return result
            except Exception as e:
                last_error = e
                model.success_rate = model.success_rate * 0.95  # 降低成功率
                if model.success_rate < 0.5:
                    model.is_healthy = False  # 标记不健康
                logging.warning(f"模型 {model.name} 调用失败: {e}，尝试下一个")
        
        raise Exception(f"所有模型都失败，最后错误: {last_error}")
    
    async def _call_model(self, model: ModelConfig, messages: list[dict], **kwargs) -> str:
        async with httpx.AsyncClient(timeout=model.timeout) as client:
            response = await client.post(
                model.api_url,
                headers={"Authorization": f"Bearer {model.api_key}"},
                json={"model": model.model_id, "messages": messages, **kwargs},
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

# 使用示例
router = ModelRouter([
    ModelConfig("deepseek", "https://api.deepseek.com/v1/chat/completions",
                DEEPSEEK_KEY, "deepseek-chat", priority=1, timeout=60.0),
    ModelConfig("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                QWEN_KEY, "qwen-plus", priority=2, timeout=30.0),
    ModelConfig("gpt-4o-mini", "https://api.openai.com/v1/chat/completions",
                OPENAI_KEY, "gpt-4o-mini", priority=3, timeout=30.0),
])

answer = await router.call([{"role": "user", "content": "什么是 RAG？"}])
```

---

## 成本控制

### Q: 如何控制第三方 LLM API 的调用成本？ ⭐⭐

**答：**

**成本构成：**
```
总成本 = 调用次数 × 单次 Token 数 × 单价
```

**控制手段（从效果最大到最小）：**

| 手段 | 节省比例 | 说明 |
|------|----------|------|
| **Prompt 压缩** | 30-50% | 去掉冗余指令、缩短 System Prompt |
| **语义缓存** | 20-40% | 相似问题直接返回缓存结果 |
| **模型路由** | 30-60% | 简单任务用小模型（GPT-4o-mini vs GPT-4o） |
| **Context Caching** | 50-90% | 缓存重复的 System Prompt/知识库（DeepSeek/Gemini 支持） |
| **Token 上限** | 防溢出 | 设置 `max_tokens` 防止模型"话痨" |
| **批量请求** | 50% | Batch API 价格减半（OpenAI Batch） |

```python
from pydantic import BaseModel
import hashlib

class CostTracker:
    """LLM 调用成本追踪器"""
    
    # 各模型定价（每百万 Token，美元）
    PRICING = {
        "deepseek-chat": {"input": 0.14, "output": 0.28},
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "qwen-plus": {"input": 0.4, "output": 1.2},
    }
    
    def __init__(self):
        self.daily_cost = 0.0
        self.daily_budget = 10.0  # 每日预算（美元）
    
    def calculate(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = self.PRICING.get(model, {"input": 1.0, "output": 3.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        self.daily_cost += cost
        
        if self.daily_cost > self.daily_budget:
            logging.warning(f"日成本 ${self.daily_cost:.4f} 超出预算 ${self.daily_budget}")
        
        return cost

class SemanticCache:
    """语义缓存：相似问题命中缓存，减少调用"""
    
    def __init__(self, threshold=0.95):
        self.cache = {}
        self.threshold = threshold  # 相似度阈值
    
    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[str]:
        key = self._hash(prompt)
        # 精确匹配（生产中用向量相似度匹配）
        return self.cache.get(key)
    
    def set(self, prompt: str, response: str):
        key = self._hash(prompt)
        self.cache[key] = response
```

---

## 错误处理与降级

### Q: 第三方 API 返回错误时，如何优雅降级？ ⭐⭐⭐

**答：**

**降级策略（从最好到最差）：**

```
正常调用 → 重试 → Fallback 模型 → 缓存结果 → 默认值 → 报错
```

**不同场景的降级方案：**

| 场景 | 降级方案 | 用户体验 |
|------|----------|----------|
| 对话 API 挂了 | Fallback 到备用模型 | 几乎无感知 |
| 搜索 API 挂了 | 返回缓存的搜索结果 | 结果可能稍旧 |
| 推荐 API 挂了 | 返回热门默认推荐 | 不够个性化 |
| 图片生成 API 挂了 | 返回占位图 + "稍后重试" | 功能受限 |
| 所有 API 都挂了 | 返回 "服务暂时不可用" | 明确告知 |

```python
from enum import Enum

class DegradationLevel(Enum):
    NORMAL = 1      # 正常
    FALLBACK = 2    # 备用模型
    CACHED = 3      # 缓存结果
    DEFAULT = 4     # 默认值
    ERROR = 5       # 报错

class GracefulDegradation:
    """优雅降级管理器"""
    
    async def call_with_degradation(
        self,
        primary_func,
        fallback_func=None,
        cache=None,
        default_value=None,
    ) -> tuple[any, DegradationLevel]:
        """按降级链尝试调用"""
        
        # 1. 尝试主调用
        try:
            result = await primary_func()
            return result, DegradationLevel.NORMAL
        except Exception as e:
            logging.warning(f"主调用失败: {e}")
        
        # 2. 尝试 Fallback
        if fallback_func:
            try:
                result = await fallback_func()
                return result, DegradationLevel.FALLBACK
            except Exception as e:
                logging.warning(f"Fallback 失败: {e}")
        
        # 3. 尝试缓存
        if cache:
            cached = cache.get()
            if cached is not None:
                return cached, DegradationLevel.CACHED
        
        # 4. 返回默认值
        if default_value is not None:
            return default_value, DegradationLevel.DEFAULT
        
        # 5. 彻底失败
        raise Exception("所有降级方案都失败")
```

---

## 并发调用优化

### Q: 需要批量调用第三方 API 时，如何优化并发性能？ ⭐⭐

**答：**

**问题**：假设有 100 篇文档需要 Embedding，逐个调用太慢，并发太高会被限流。

**并发控制方案：**

| 方案 | 适用场景 | 实现方式 |
|------|----------|----------|
| `asyncio.gather` | 小批量，无限制 | 全部并发 |
| `asyncio.Semaphore` | 控制并发上限 | 信号量 |
| `asyncio.as_completed` | 需要逐个处理结果 | 按完成顺序返回 |
| 批量 API | 支持批量的接口 | 一次请求多条 |

```python
import asyncio
from typing import Callable

async def batch_call_with_semaphore(
    items: list,
    func: Callable,
    max_concurrent: int = 10,
    rate_per_second: float = 50.0,
) -> list:
    """
    带并发控制和限流的批量调用
    
    Args:
        items: 待处理的数据列表
        func: 异步调用函数
        max_concurrent: 最大并发数
        rate_per_second: 每秒最大请求数
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    rate_limiter = TokenBucket(rate=rate_per_second, capacity=max_concurrent)
    results = [None] * len(items)
    
    async def _call_one(index: int, item):
        async with semaphore:
            await rate_limiter.acquire()
            try:
                result = await func(item)
                results[index] = result
            except Exception as e:
                results[index] = e
    
    tasks = [_call_one(i, item) for i, item in enumerate(items)]
    await asyncio.gather(*tasks)
    
    return results

# 使用示例：批量 Embedding
async def embed_text(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENAI_KEY}"},
            json={"model": "text-embedding-3-small", "input": text},
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

# 100 篇文档，并发 10，限速 50 req/s
texts = ["文档1内容", "文档2内容", "..."]
embeddings = await batch_call_with_semaphore(texts, embed_text, max_concurrent=10)
```

---

## 综合实战

### Q: 设计一个生产级的第三方 API 调用层，需要包含哪些组件？ ⭐⭐⭐

**答：** 一个生产级的 API 调用层需要 **7 个核心组件**，可以画成洋葱模型：

```
请求入口
  → [1] 认证 & 签名
    → [2] 限流（令牌桶）
      → [3] 缓存检查（命中则返回）
        → [4] 熔断器（打开则快速失败）
          → [5] 重试（指数退避）
            → [6] HTTP 请求（连接池 + 超时）
              → [7] 响应验证（Pydantic）
            → [5] 重试（失败时）
          → [4] 熔断器（记录失败）
        → [3] 缓存写入（成功时）
      → [2] 限流（释放令牌）
    → [1] 记录审计日志
  → 返回结果
```

**每个组件的职责：**

| 组件 | 职责 | 关键配置 |
|------|------|----------|
| 认证 | API Key 管理、请求签名 | Key 轮转、环境变量 |
| 限流 | 控制调用频率 | 令牌桶、按用户/模型分配 |
| 缓存 | 减少重复调用 | TTL、语义缓存、LRU |
| 熔断 | 保护系统不被拖垮 | 失败阈值、冷却时间 |
| 重试 | 处理瞬时故障 | 指数退避、抖动、Retry-After |
| HTTP | 网络通信 | 连接池、超时、流式 |
| 验证 | 确保响应正确 | Pydantic、状态码、Content-Type |

**这个架构面试时画出来，能覆盖 80% 的追问。**

---

## 面试速查

### 高频追问清单

| 问题 | 关键点 |
|------|--------|
| API 调用超时怎么办？ | 重试 → 熔断 → Fallback → 降级 |
| 什么是惊群效应？ | 多客户端同时重试，用抖动分散 |
| 熔断器三种状态？ | 关闭→打开→半开→关闭 |
| 令牌桶 vs 漏桶？ | 令牌桶允许突发，漏桶平滑 |
| 429 响应怎么处理？ | 遵守 Retry-After，客户端主动限流 |
| API Key 泄露怎么办？ | 立即吊销→换新→审计→排查→复盘 |
| 为什么要流式？ | 减少首 token 延迟，提升用户体验 |
| 怎么做语义缓存？ | Embedding 相似度匹配，超阈值命中 |
| 多模型 Fallback 怎么设计？ | 优先级链 + 健康检查 + 动态调整 |
| 批量调用怎么优化？ | Semaphore 控制并发 + 令牌桶限速 |

### 速记口诀

```
超重熔限签，缓幂日验连
超时 → 重试 → 熔断 → 限流 → 签名
缓存 → 幂等 → 日志 → 验证 → 连接池
```
