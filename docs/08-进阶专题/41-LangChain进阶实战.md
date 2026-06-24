# 41. LangChain 进阶实战

> 本文涵盖 LangChain 框架的进阶知识点，包括自定义组件、可观测性、流式输出、异步编程、错误处理、生产部署等核心主题。适合有一定 LangChain 基础、准备中高级 LLM/AI 工程师面试的读者。

---

## 一、自定义 Runnables 与 RunnableConfig

### Q: LangChain 中的 Runnable 接口是什么？它统一了哪些操作？ ⭐⭐⭐

**答**：

Runnable 是 LangChain v0.1+ 中所有组件实现的核心抽象接口。它将 LLM、Prompt、Parser、Retriever、Tool 等异构组件统一到同一协议下，支持以下标准方法：

| 方法 | 用途 |
|------|------|
| `invoke(input)` | 单次同步调用 |
| `batch(inputs)` | 批量调用（自动并行化） |
| `stream(input)` | 流式输出 |
| `ainvoke(input)` | 异步单次调用 |
| `abatch(inputs)` | 异步批量调用 |
| `astream(input)` | 异步流式输出 |

这种设计的核心价值在于**可组合性**：任意 Runnable 可以通过 `|`（pipe 运算符）串联成链（Chain），构成 LCEL（LangChain Expression Language）。

```python
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("用一句话解释：{topic}")
llm = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# LCEL 链式组合
chain = prompt | llm | parser
result = chain.invoke({"topic": "量子计算"})
```

---

### Q: 如何自定义一个 Runnable？请给出完整示例。 ⭐⭐⭐

**答**：

自定义 Runnable 有两种方式：继承 `RunnableLambda` 或实现 `RunnableSerializable`。推荐使用 `RunnableLambda` 包装普通函数：

```python
from langchain_core.runnables import RunnableLambda, RunnableConfig
import re

# 方式一：简单包装函数
def clean_text(input_data: dict) -> dict:
    """清洗文本：去除多余空格和特殊字符"""
    text = input_data.get("text", "")
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return {**input_data, "text": cleaned}

cleaner = RunnableLambda(clean_text)

# 方式二：带类型签名的 Runnable（推荐，LangChain 可自动推断 schema）
from langchain_core.runnables import RunnableLambda
from typing import TypedDict

class CleanInput(TypedDict):
    text: str

class CleanOutput(TypedDict):
    text: str
    original_length: int

def clean_and_count(input_data: CleanInput) -> CleanOutput:
    text = input_data["text"]
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return {"text": cleaned, "original_length": len(text)}

typed_cleaner = RunnableLambda(clean_and_count)

# 验证：可以正常使用 LCEL 组合
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    RunnableLambda(lambda x: {"topic": x["topic"].strip()})
    | ChatPromptTemplate.from_template("解释：{topic}")
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)
print(chain.invoke({"topic": " RAG 检索增强生成 "}))
```

---

### Q: RunnableConfig 的作用是什么？如何在链中传递运行时配置？ ⭐⭐⭐

**答**：

`RunnableConfig` 是 LangChain 的运行时配置对象，允许在调用时注入元数据，而无需修改链的结构。它支持以下关键字段：

- `tags`：为调用打标签，便于 LangSmith 筛选
- `metadata`：附加自定义元数据
- `callbacks`：注册回调函数（用于日志、追踪）
- `run_name`：自定义运行名称
- `max_concurrency`：控制并发数

```python
from langchain_core.runnables import RunnableConfig, RunnableLambda

def process_with_config(input_data: dict, config: RunnableConfig) -> dict:
    """在 Runnable 中访问 config"""
    metadata = config.get("metadata", {})
    user_id = metadata.get("user_id", "unknown")
    tags = config.get("tags", [])

    # 根据用户等级选择不同模型
    if "vip" in tags:
        model = "gpt-4o"
    else:
        model = "gpt-4o-mini"

    return {"user_id": user_id, "model_used": model, "input": input_data}

processor = RunnableLambda(process_with_config)

# 调用时传入 config
result = processor.invoke(
    {"query": "你好"},
    config=RunnableConfig(
        tags=["vip", "production"],
        metadata={"user_id": "user_123", "session_id": "sess_456"},
        run_name="user_query_processor",
    )
)
print(result)
# {'user_id': 'user_123', 'model_used': 'gpt-4o', 'input': {'query': '你好'}}
```

---

### Q: 如何用 RunnablePassthrough 和 RunnableParallel 构建复杂数据流？ ⭐⭐⭐

**答**：

`RunnablePassthrough` 直接透传输入数据（可附加额外字段），`RunnableParallel` 并行执行多个 Runnable 并合并结果。它们是构建 RAG、Multi-Chain 等复杂架构的基础：

```python
from langchain_core.runnables import (
    RunnablePassthrough, RunnableParallel, RunnableLambda
)

# 模拟 Retriever
def fake_retrieve(query: str) -> str:
    return f"检索到的文档：关于 {query} 的相关内容..."

# 构建 RAG 流程
rag_chain = (
    RunnableParallel(
        context=RunnableLambda(fake_retrieve),       # 检索上下文
        question=RunnablePassthrough(),               # 原始问题透传
    )
    | RunnableLambda(lambda x: {
        "prompt": f"基于以下内容回答问题：\n{x['context']}\n问题：{x['question']}",
        "has_context": bool(x["context"]),
    })
)

result = rag_chain.invoke("什么是 Transformer")
print(result)
```

**核心区别**：
- `RunnablePassthrough()` — 直接透传输入，也可 `.assign()` 附加字段
- `RunnableParallel({...})` — 并行执行多个分支，结果合并为 dict
- `RunnableLambda(fn)` — 包装任意 Python 函数

---

### Q: Runnable 的 `.assign()`、`.pick()`、`.map()` 分别有什么用？ ⭐⭐⭐

**答**：

这三个方法用于操作 Runnable 的数据流：

```python
from langchain_core.runnables import RunnableLambda

# .assign() — 在已有输出上追加新字段
chain_with_score = RunnableLambda(lambda x: {"text": x}).assign(
    length=lambda x: len(x["text"]),
    upper=lambda x: x["text"].upper(),
)
print(chain_with_score.invoke("hello"))
# {'text': 'hello', 'length': 5, 'upper': 'HELLO'}

# .pick() — 从输出中选取指定字段
only_text = chain_with_score.pick(["text", "length"])
print(only_text.invoke("hello"))
# {'text': 'hello', 'length': 5}

# .map() — 对列表中每个元素执行 Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

summarize = (
    ChatPromptTemplate.from_template("一句话总结：{text}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# 批量总结多段文本
batch_summarize = summarize.map()
results = batch_summarize.invoke([
    {"text": "LangChain 是一个 LLM 应用开发框架"},
    {"text": "RAG 结合了检索和生成两种能力"},
    {"text": "Agent 让 LLM 具备工具使用能力"},
])
print(results)  # ['LangChain...', 'RAG...', 'Agent...']
```

---

## 二、LangSmith 可观测性与调试

### Q: LangSmith 是什么？它解决了 LLM 应用开发中的哪些痛点？ ⭐⭐

**答**：

LangSmith 是 LangChain 官方的可观测性平台，核心功能：

1. **Trace 追踪**：记录每次 LLM 调用的完整调用链（输入、输出、耗时、token 用量）
2. **在线评估（Evaluators）**：自动对输出质量评分
3. **数据集管理**：构建测试数据集，支持回归测试
4. **Prompt 版本管理**：追踪 Prompt 的迭代历史
5. **调试**：可视化查看每一步的中间结果

它解决的核心痛点：LLM 应用是"黑盒"的，调试困难，LangSmith 提供了类似 APM（应用性能监控）的能力。

---

### Q: 如何在项目中集成 LangSmith？有哪些关键配置？ ⭐⭐⭐

**答**：

集成只需设置环境变量，无需修改代码：

```bash
# 环境变量配置
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="lsv2_pt_xxxx"
export LANGCHAIN_PROJECT="my-rag-project"        # 项目名称
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
```

```python
import os

# 代码中设置（等价于环境变量）
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_pt_xxxx"

# 所有 LangChain 调用自动被追踪
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o")
chain = ChatPromptTemplate.from_template("说一个关于{topic}的笑话") | llm
result = chain.invoke({"topic": "编程"})  # 此调用自动上报到 LangSmith
```

还可以通过 `RunnableConfig` 传递更细粒度的元数据：

```python
from langchain_core.runnables import RunnableConfig

result = chain.invoke(
    {"topic": "编程"},
    config=RunnableConfig(
        tags=["test", "v2"],
        metadata={"user": "alice", "experiment": "prompt_v3"},
        run_name="joke_generator_v2",
    )
)
```

---

### Q: 如何在 LangSmith 中创建评估数据集并运行自动评估？ ⭐⭐⭐

**答**：

LangSmith 支持通过 SDK 创建数据集、上传测试用例、运行评估：

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 1. 创建数据集
dataset = client.create_dataset(
    dataset_name="rag_qa_eval_v1",
    description="RAG 系统问答质量评估集",
)

# 2. 添加测试用例
examples = [
    {"inputs": {"question": "什么是 RAG？"},
     "outputs": {"expected": "RAG 是检索增强生成的缩写..."}},
    {"inputs": {"question": "LangChain 的核心概念是什么？"},
     "outputs": {"expected": "LangChain 的核心概念包括..."}},
]
for ex in examples:
    client.create_example(
        inputs=ex["inputs"],
        outputs=ex["outputs"],
        dataset_id=dataset.id,
    )

# 3. 定义评估函数
def correctness_evaluator(run, example):
    """评估回答是否正确"""
    prediction = run.outputs.get("output", "")
    reference = example.outputs.get("expected", "")
    # 简单的相似度评估（生产中可用 LLM-as-Judge）
    score = 1.0 if reference[:10] in prediction else 0.0
    return {"key": "correctness", "score": score}

# 4. 运行评估
def target_function(inputs: dict) -> dict:
    """被评估的目标函数（你的 RAG 链）"""
    result = my_rag_chain.invoke({"question": inputs["question"]})
    return {"output": result}

experiment_results = evaluate(
    target_function,
    data="rag_qa_eval_v1",
    evaluators=[correctness_evaluator],
    experiment_prefix="rag_v2",
)
print(f"平均分：{experiment_results['aggregate_metrics']}")
```

---

## 三、流式输出（Streaming）深度实践

### Q: LangChain 的 Streaming 有哪些模式？各自的适用场景是什么？ ⭐⭐⭐

**答**：

LangChain 提供 4 种流式模式：

| 模式 | 方法 | 适用场景 |
|------|------|---------|
| `stream()` | 同步流式 | 终端工具、同步 Web 服务 |
| `astream()` | 异步流式 | FastAPI、WebSocket 异步服务 |
| `astream_events()` | 异步事件流 | 需要监控链内部各步骤的场景 |
| `astream_log()` | 异步日志流 | 调试和监控（逐步被 astream_events 替代） |

```python
# 基本流式输出
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

chain = (
    ChatPromptTemplate.from_template("写一首关于{topic}的诗")
    | ChatOpenAI(model="gpt-4o", streaming=True)
)

# 同步流式
for chunk in chain.stream({"topic": "月亮"}):
    print(chunk.content, end="", flush=True)
```

---

### Q: 如何使用 `astream_events` 实现细粒度的流式监控？ ⭐⭐⭐

**答**：

`astream_events` 返回链执行过程中的所有事件，包括每个 Runnable 的开始、结束、流式 chunk 等：

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("详细解释：{topic}")
    | ChatOpenAI(model="gpt-4o", streaming=True)
    | StrOutputParser()
)

async def stream_with_events():
    async for event in chain.astream_events(
        {"topic": "注意力机制"},
        version="v2",  # 推荐使用 v2
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            # LLM 的流式 token
            content = event["data"]["chunk"].content
            if content:
                print(f"[LLM Token]: {content}")

        elif kind == "on_chain_start":
            print(f"\n[Chain 开始] {event['name']}")

        elif kind == "on_chain_end":
            print(f"\n[Chain 结束] {event['name']}")

        elif kind == "on_parser_start":
            print(f"[Parser 开始]")

        elif kind == "on_parser_stream":
            print(f"[Parser 输出]: {event['data']['chunk']}")

asyncio.run(stream_with_events())
```

**关键事件类型**：
- `on_chain_start` / `on_chain_end`：链节点开始/结束
- `on_chat_model_start` / `on_chat_model_stream`：LLM 调用和流式 token
- `on_tool_start` / `on_tool_end`：工具调用
- `on_retriever_start` / `on_retriever_end`：检索器调用

---

### Q: 如何在 FastAPI 中实现 SSE（Server-Sent Events）流式响应？ ⭐⭐⭐

**答**：

这是生产中最常见的流式模式——将 LangChain 的流式输出通过 SSE 推送给前端：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import asyncio
import json

app = FastAPI()

chain = (
    ChatPromptTemplate.from_template("回答：{question}")
    | ChatOpenAI(model="gpt-4o", streaming=True)
    | StrOutputParser()
)

async def event_generator(question: str):
    """SSE 事件生成器"""
    async for chunk in chain.astream({"question": question}):
        # SSE 格式
        yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

@app.get("/chat/stream")
async def chat_stream(question: str):
    return StreamingResponse(
        event_generator(question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx 透传
        },
    )
```

**前端调用示例**：

```javascript
const evtSource = new EventSource("/chat/stream?question=什么是LangChain");
evtSource.onmessage = (e) => {
  if (e.data === "[DONE]") {
    evtSource.close();
    return;
  }
  const { content } = JSON.parse(e.data);
  document.getElementById("output").textContent += content;
};
```

---

### Q: 如何实现"第一个 token 延迟"（TTFT）优化？ ⭐⭐⭐

**答**：

Time To First Token（TTFT）是流式体验的关键指标。优化策略：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

# 策略1：使用更快的模型生成开头
fast_llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
full_llm = ChatOpenAI(model="gpt-4o", streaming=True)

prompt = ChatPromptTemplate.from_template("回答：{question}")

# 策略2：减少 Prompt 长度（更少 token = 更快首 token）
# 策略3：合理设置 max_tokens，避免生成过长
llm = ChatOpenAI(
    model="gpt-4o",
    streaming=True,
    max_tokens=1024,      # 限制输出长度
    request_timeout=30,    # 设置超时
)

chain = prompt | llm | StrOutputParser()

# 测量 TTFT
async def measure_ttft(question: str):
    first_token_time = None
    start = time.time()

    async for chunk in chain.astream({"question": question}):
        if first_token_time is None:
            first_token_time = time.time() - start
            print(f"TTFT: {first_token_time:.3f}s")
        print(chunk, end="", flush=True)

    total = time.time() - start
    print(f"\n总耗时: {total:.3f}s")

import asyncio
asyncio.run(measure_ttft("什么是 RAG"))
```

---

## 四、异步编程（async/await）最佳实践

### Q: 为什么 LLM 应用强烈推荐使用异步编程？ ⭐⭐

**答**：

LLM 调用是典型的 **I/O 密集型** 操作（网络请求等待模型推理），异步编程的价值在于：

1. **高并发处理**：单线程可同时处理数千个请求，避免为每个请求分配线程
2. **资源效率**：不需要大量线程/进程，内存占用低
3. **可组合性**：LangChain 的 `Runnable` 原生支持 `ainvoke`、`abatch`、`astream`

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o")
chain = ChatPromptTemplate.from_template("一句话说说{topic}") | llm | StrOutputParser()

# 同步方式：串行执行（慢）
def sync_batch(questions):
    return [chain.invoke(q) for q in questions]  # 逐个等待

# 异步方式：并发执行（快）
async def async_batch(questions):
    tasks = [chain.ainvoke(q) for q in questions]
    return await asyncio.gather(*tasks)  # 并发执行

# 使用 abatch（推荐，内置并发控制）
async def optimized_batch(questions):
    return await chain.abatch(questions, config={"max_concurrency": 10})
```

---

### Q: 如何在异步环境中优雅地处理并发限制和限流？ ⭐⭐⭐

**答**：

生产环境中 API 有 rate limit，需要控制并发：

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

llm = ChatOpenAI(model="gpt-4o")

# 方法1：使用 Semaphore 控制并发
async def process_with_semaphore(questions: list[str], max_concurrent: int = 5):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_invoke(question: str):
        async with semaphore:
            return await llm.ainvoke(question)

    tasks = [limited_invoke(q) for q in questions]
    return await asyncio.gather(*tasks, return_exceptions=True)

# 方法2：使用 RunnableConfig 的 max_concurrency
async def process_with_config(questions: list[str]):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    chain = (
        ChatPromptTemplate.from_template("{q}")
        | llm
        | StrOutputParser()
    )
    # abatch 内置并发控制
    return await chain.abatch(
        [{"q": q} for q in questions],
        config=RunnableConfig(max_concurrency=5),
    )

# 方法3：指数退避重试（处理 429 错误）
from langchain_core.runnables import RunnableLambda
import random

async def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"限流，等待 {wait:.1f}s 后重试...")
                await asyncio.sleep(wait)
            else:
                raise
```

---

### Q: 如何在异步链中处理上下文传播（如用户身份信息）？ ⭐⭐⭐

**答**：

使用 `contextvars` 或 `RunnableConfig` 传播上下文：

```python
import asyncio
import contextvars
from langchain_core.runnables import RunnableConfig, RunnableLambda

# 方法1：使用 contextvars（Python 原生）
user_ctx = contextvars.ContextVar("user", default="anonymous")

async def middle_aware(input_data: dict, config: RunnableConfig) -> dict:
    # 从 config 中获取用户信息
    metadata = config.get("metadata", {})
    user = metadata.get("user", "anonymous")
    user_ctx.set(user)  # 设置上下文变量

    # 所有后续 async 调用自动继承此上下文
    result = await some_llm_call(input_data)

    return {"result": result, "processed_by": user_ctx.get()}

async def some_llm_call(data: dict) -> str:
    # 可以在此处读取上下文
    user = user_ctx.get()
    return f"处理结果（用户: {user}）"

# 方法2：通过 config 传递（推荐，更显式）
aware_chain = RunnableLambda(middle_aware)

async def main():
    result = await aware_chain.ainvoke(
        {"query": "hello"},
        config=RunnableConfig(metadata={"user": "alice"}),
    )
    print(result)

asyncio.run(main())
```

---

## 五、错误处理与重试机制

### Q: LangChain 提供了哪些内置的错误处理和重试机制？ ⭐⭐⭐

**答**：

LangChain 提供多层错误处理：

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
import logging

# 1. LLM 层级：with_fallbacks — 主模型失败时切换备选模型
primary_llm = ChatOpenAI(model="gpt-4o", max_retries=3)
fallback_llm = ChatOpenAI(model="gpt-4o-mini", max_retries=3)

llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])

# 2. 链层级：with_retry — 自定义重试策略
from langchain_core.runnables import Runnable

def risky_operation(input_data: dict) -> dict:
    """可能失败的操作"""
    if random.random() < 0.5:
        raise ValueError("随机失败")
    return {"result": "成功"}

import random
risky = RunnableLambda(risky_operation)

# 配置重试
reliable = risky.with_retry(
    retry_if_exception_type=(ValueError, TimeoutError),
    wait_exponential_jitter=True,  # 指数退避 + 抖动
    stop_after_attempt=5,
)

# 3. 链层级：with_fallbacks — 整条链的 fallback
def backup_operation(input_data: dict) -> dict:
    return {"result": "备用方案"}

backup_chain = RunnableLambda(backup_operation)
chain_with_fallback = risky.with_fallbacks([backup_chain])

# 测试
try:
    print(chain_with_fallback.invoke({"input": "test"}))
except Exception as e:
    print(f"所有方案失败: {e}")
```

---

### Q: 如何实现自定义的全局错误处理中间件？ ⭐⭐⭐

**答**：

通过自定义 Runnable 包装器实现统一的错误处理、日志和指标上报：

```python
from langchain_core.runnables import Runnable, RunnableConfig
from typing import Any, Optional
import time
import logging
import traceback

logger = logging.getLogger(__name__)

class ErrorHandlingWrapper(Runnable):
    """全局错误处理包装器"""

    def __init__(self, inner: Runnable, fallback_value: Any = None):
        self.inner = inner
        self.fallback_value = fallback_value

    @property
    def InputType(self):
        return self.inner.InputType

    @property
    def OutputType(self):
        return self.inner.OutputType

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        start_time = time.time()
        run_name = config.get("run_name", "unknown") if config else "unknown"

        try:
            result = self.inner.invoke(input, config)
            elapsed = time.time() - start_time
            logger.info(f"[{run_name}] 成功，耗时 {elapsed:.2f}s")
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[{run_name}] 失败，耗时 {elapsed:.2f}s\n"
                f"错误类型: {type(e).__name__}\n"
                f"错误信息: {str(e)}\n"
                f"堆栈:\n{traceback.format_exc()}"
            )

            if self.fallback_value is not None:
                logger.warning(f"[{run_name}] 使用 fallback 值")
                return self.fallback_value
            raise

# 使用
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("回答：{q}")
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

safe_chain = ErrorHandlingWrapper(chain, fallback_value="抱歉，服务暂时不可用，请稍后重试。")

result = safe_chain.invoke(
    {"q": "你好"},
    config=RunnableConfig(run_name="chat_endpoint"),
)
```

---

### Q: 生产环境中如何处理 LLM 的 token 限制和截断问题？ ⭐⭐⭐

**答**：

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import tiktoken

class TokenSafeLLM:
    """Token 安全的 LLM 包装器"""

    def __init__(self, model: str = "gpt-4o", max_input_tokens: int = 100000):
        self.llm = ChatOpenAI(model=model)
        self.max_input_tokens = max_input_tokens
        self.encoding = tiktoken.encoding_for_model("gpt-4o")

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def truncate_messages(self, messages: list) -> list:
        """截断消息列表，保留系统消息和最近的消息"""
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        # 优先保留系统消息
        total_tokens = sum(self.count_tokens(m.content) for m in system_msgs)
        kept_messages = list(system_msgs)

        # 从最新消息开始保留
        for msg in reversed(other_msgs):
            msg_tokens = self.count_tokens(msg.content)
            if total_tokens + msg_tokens <= self.max_input_tokens:
                kept_messages.append(msg)
                total_tokens += msg_tokens
            else:
                # 截断当前消息
                remaining = self.max_input_tokens - total_tokens
                if remaining > 100:  # 至少保留 100 token
                    truncated = self.encoding.decode(
                        self.encoding.encode(msg.content)[:remaining]
                    )
                    kept_messages.append(
                        HumanMessage(content=truncated + "\n[已截断]")
                    )
                break

        return kept_messages

    def invoke(self, messages: list):
        safe_messages = self.truncate_messages(messages)
        return self.llm.invoke(safe_messages)

# 使用
safe_llm = TokenSafeLLM(max_input_tokens=8000)
long_messages = [
    SystemMessage(content="你是一个助手"),
    HumanMessage(content="很长的文本..." * 1000),
]
result = safe_llm.invoke(long_messages)
```

---

## 六、自定义 Retriever 实现

### Q: 如何实现一个自定义的 Retriever？ ⭐⭐⭐

**答**：

继承 `BaseRetriever` 并实现 `_get_relevant_documents` 方法：

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List
import hashlib

class HybridRetriever(BaseRetriever):
    """混合检索器：结合向量搜索和关键词搜索"""

    vector_store: object       # 向量数据库
    keyword_store: dict        # 关键词索引 {keyword: [doc_ids]}
    documents: dict            # 文档存储 {doc_id: Document}
    k: int = 4                 # 返回文档数
    vector_weight: float = 0.7 # 向量搜索权重
    keyword_weight: float = 0.3 # 关键词搜索权重

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # 1. 向量搜索
        vector_results = self.vector_store.similarity_search_with_score(
            query, k=self.k * 2
        )
        vector_scores = {}
        for doc, score in vector_results:
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
            vector_scores[doc_id] = (doc, score)

        # 2. 关键词搜索
        keyword_scores = {}
        query_keywords = set(query.lower().split())
        for keyword in query_keywords:
            if keyword in self.keyword_store:
                for doc_id in self.keyword_store[keyword]:
                    doc = self.documents[doc_id]
                    keyword_scores[doc_id] = (doc,
                        keyword_scores.get(doc_id, (None, 0))[1] + 1.0
                    )

        # 3. 融合排序
        all_doc_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
        scored_docs = []
        for doc_id in all_doc_ids:
            v_score = vector_scores.get(doc_id, (None, 0))[1]
            k_score = keyword_scores.get(doc_id, (None, 0))[1]
            combined = self.vector_weight * v_score + self.keyword_weight * k_score
            doc = vector_scores.get(doc_id, keyword_scores.get(doc_id))[0]
            if doc:
                scored_docs.append((doc, combined))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:self.k]]

# 使用
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_texts(
    ["LangChain 是框架", "RAG 是技术", "Agent 是模式"],
    OpenAIEmbeddings(),
)

retriever = HybridRetriever(
    vector_store=vectorstore,
    keyword_store={"langchain": ["doc1"], "rag": ["doc2"]},
    documents={"doc1": Document(page_content="LangChain 是框架"),
               "doc2": Document(page_content="RAG 是技术")},
    k=2,
)
docs = retriever.invoke("LangChain RAG")
```

---

### Q: 如何为 Retriever 添加压缩和去重后处理？ ⭐⭐⭐

**答**：

使用 `DocumentCompressorPipeline` 和 `BaseDocumentTransformer`：

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from typing import List

class Deduplicator(BaseDocumentTransformer):
    """去除重复文档"""

    def __init__(self, similarity_threshold: float = 0.9):
        self.threshold = similarity_threshold

    def transform_documents(self, documents: List[Document]) -> List[Document]:
        seen_contents = set()
        unique = []
        for doc in documents:
            # 简单的基于内容哈希的去重
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique.append(doc)
        return unique

# 组合后处理管道
embeddings = OpenAIEmbeddings()
compressor_pipeline = DocumentCompressorPipeline(
    transformers=[
        CharacterTextSplitter(chunk_size=500, chunk_overlap=50),  # 重新分块
        EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7),  # 相似度过滤
        Deduplicator(),  # 去重
    ]
)

# 包装 Retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor_pipeline,
    base_retriever=some_base_retriever,
)

docs = compression_retriever.invoke("什么是 LangChain")
```

---

## 七、LangChain v0.3 核心变化

### Q: LangChain v0.3 相比 v0.2 有哪些重大变化？迁移时需要注意什么？ ⭐⭐⭐

**答**：

LangChain v0.3（2024年发布）的核心变化：

1. **全面转向 Pydantic v2**：不再兼容 Pydantic v1，所有自定义组件需要迁移
2. **移除 langchain.community 中的旧代码**：社区集成迁移到独立包
3. **统一 Runnable 接口**：所有组件必须实现 Runnable 协议
4. **langchain-core 精简**：移除不必要的依赖

```python
# ❌ v0.2 写法（已废弃）
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

# ✅ v0.3 写法
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Pydantic v2 迁移示例
from pydantic import BaseModel, Field  # 使用 Pydantic v2

class MyTool(BaseModel):
    name: str = Field(description="工具名称")
    description: str = Field(description="工具描述")

    # v2: 使用 model_validator 替代 validator
    from pydantic import model_validator

    @model_validator(mode="after")
    def validate_name(self):
        if len(self.name) < 2:
            raise ValueError("名称至少2个字符")
        return self
```

**迁移清单**：
```bash
# 更新包
pip install langchain>=0.3 langchain-core>=0.3 langchain-openai>=0.2

# 检查 Pydantic v1 兼容性
pip install langchain-core --upgrade

# 常见修改
# 1. import 路径变更
# 2. Pydantic v1 -> v2
# 3. .dict() -> .model_dump()
# 4. .schema() -> .model_json_schema()
```

---

### Q: LangChain v0.3 中 packages 拆分策略是什么？为什么这样设计？ ⭐⭐⭐

**答**：

v0.3 采用更细粒度的包拆分：

```
langchain-core        # 核心抽象（Runnable, Message, Document 等）
langchain-openai      # OpenAI 集成
langchain-anthropic   # Anthropic 集成
langchain-community   # 社区贡献的集成（大型包）
langchain-text-splitters # 文本分割器
langchain             # 主包（依赖 langchain-core）
```

设计原因：
1. **依赖隔离**：不需要 Anthropic SDK 就不用安装它
2. **独立版本管理**：各集成包可独立发版
3. **安全审计**：每个包的依赖链更短，更易审计
4. **减少安装冲突**：不同 SDK 的依赖冲突概率降低

```python
# 最小安装
# pip install langchain-core langchain-openai

# 完整安装
# pip install langchain langchain-openai langchain-community

# 推荐的 requirements.txt
"""
langchain-core>=0.3.0
langchain-openai>=0.2.0
langchain-text-splitters>=0.3.0
"""
```

---

## 八、生产部署模式

### Q: 如何将 LangChain 应用部署到生产环境？有哪些推荐架构？ ⭐⭐⭐

**答**：

推荐的生产部署架构：

```python
# 生产级 FastAPI 应用结构
# app/main.py
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import redis
import hashlib
import json

# 全局资源管理
class AppState:
    chain = None
    redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化
    AppState.chain = (
        ChatPromptTemplate.from_template("回答：{q}")
        | ChatOpenAI(
            model="gpt-4o",
            max_tokens=2048,
            request_timeout=30,
            max_retries=2,
        )
        | StrOutputParser()
    )
    AppState.redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    yield
    # 关闭时清理
    await AppState.chain.ainvoke.__wrapped__  # 清理连接

app = FastAPI(lifespan=lifespan)

# 缓存层
async def get_cached_response(query: str) -> str | None:
    cache_key = f"chat:{hashlib.md5(query.encode()).hexdigest()}"
    cached = AppState.redis_client.get(cache_key)
    return cached

async def set_cached_response(query: str, response: str, ttl: int = 3600):
    cache_key = f"chat:{hashlib.md5(query.encode()).hexdigest()}"
    AppState.redis_client.setex(cache_key, ttl, response)

@app.post("/chat")
async def chat(request: dict):
    question = request.get("question", "")
    if not question:
        raise HTTPException(400, "question 不能为空")

    # 缓存检查
    cached = await get_cached_response(question)
    if cached:
        return {"answer": cached, "cached": True}

    # 调用 LLM
    try:
        result = await AppState.chain.ainvoke({"q": question})
        await set_cached_response(question, result)
        return {"answer": result, "cached": False}
    except Exception as e:
        raise HTTPException(500, f"LLM 调用失败: {str(e)}")
```

---

### Q: 如何实现 LangChain 应用的健康检查和优雅降级？ ⭐⭐⭐

**答**：

```python
from fastapi import FastAPI, Response
from enum import Enum
import asyncio
import time

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthChecker:
    def __init__(self):
        self.last_check = 0
        self.status = HealthStatus.HEALTHY
        self.check_interval = 30  # 秒

    async def check_llm(self) -> bool:
        """检查 LLM 是否可用"""
        try:
            result = await asyncio.wait_for(
                AppState.chain.ainvoke({"q": "ping"}),
                timeout=10.0,
            )
            return len(result) > 0
        except Exception:
            return False

    async def check_redis(self) -> bool:
        """检查 Redis 是否可用"""
        try:
            return AppState.redis_client.ping()
        except Exception:
            return False

    async def get_health(self) -> dict:
        now = time.time()
        if now - self.last_check > self.check_interval:
            llm_ok = await self.check_llm()
            redis_ok = await self.check_redis()

            if llm_ok and redis_ok:
                self.status = HealthStatus.HEALTHY
            elif llm_ok:
                self.status = HealthStatus.DEGRADED
            else:
                self.status = HealthStatus.UNHEALTHY

            self.last_check = now

        return {
            "status": self.status,
            "llm": "ok" if self.status != HealthStatus.UNHEALTHY else "error",
            "cache": "ok" if self.status == HealthStatus.HEALTHY else "degraded",
        }

health_checker = HealthChecker()

@app.get("/health")
async def health_check(response: Response):
    result = await health_checker.get_health()
    if result["status"] == HealthStatus.UNHEALTHY:
        response.status_code = 503
    return result

# 优雅降级：LLM 不可用时返回缓存或默认回答
@app.post("/chat")
async def chat_with_fallback(request: dict):
    question = request.get("question", "")

    # 优先缓存
    cached = await get_cached_response(question)
    if cached:
        return {"answer": cached, "source": "cache"}

    # 检查 LLM 健康状态
    health = await health_checker.get_health()
    if health["status"] == HealthStatus.UNHEALTHY:
        return {
            "answer": "服务暂时不可用，请稍后重试。",
            "source": "fallback",
        }

    try:
        result = await AppState.chain.ainvoke({"q": question})
        await set_cached_response(question, result)
        return {"answer": result, "source": "llm"}
    except Exception:
        return {"answer": "处理失败，请重试。", "source": "error"}
```

---

## 九、性能优化技巧

### Q: LangChain 应用有哪些关键性能优化手段？ ⭐⭐⭐

**答**：

```python
# 1. Prompt 压缩 — 减少输入 token
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LongLLMLinguaCompressor

compressor = LongLLMLinguaCompressor()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# 2. 批量处理 — 使用 batch 代替多次 invoke
questions = [{"q": q} for q in ["问题1", "问题2", "问题3"]]
# ❌ 低效
results = [chain.invoke(q) for q in questions]
# ✅ 高效（内部自动并行）
results = chain.batch(questions, config={"max_concurrency": 10})

# 3. 语义缓存 — 相似问题命中缓存
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# 首次调用
result1 = llm.invoke("什么是AI")  # 调用 API
# 相同问题
result2 = llm.invoke("什么是AI")  # 命中缓存，不调用 API

# 4. 嵌入缓存 — 避免重复计算 Embedding
from langchain_community.embeddings import CacheBackedEmbeddings
from langchain_community.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

underlying_embeddings = OpenAIEmbeddings()
store = LocalFileStore("./embedding_cache/")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace="openai"
)

# 5. 流式 + 早返回 — 减少用户感知延迟
# 前面已介绍，使用 astream 替代 ainvoke

# 6. 模型选择策略
from langchain_openai import ChatOpenAI

# 简单任务用小模型
simple_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=512)
# 复杂任务用大模型
complex_llm = ChatOpenAI(model="gpt-4o", max_tokens=4096)

def route_by_complexity(question: str) -> ChatOpenAI:
    """根据问题复杂度路由到不同模型"""
    complexity_indicators = ["分析", "推理", "比较", "评估", "综合"]
    if any(ind in question for ind in complexity_indicators):
        return complex_llm
    return simple_llm
```

---

### Q: 如何监控和优化 LangChain 应用的 token 消耗成本？ ⭐⭐⭐

**答**：

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

    # GPT-4o 定价（per 1M tokens）
    COST_MAP = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
    }

    def add(self, model: str, prompt: int, completion: int):
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        costs = self.COST_MAP.get(model, (0, 0))
        self.total_cost += (prompt * costs[0] + completion * costs[1]) / 1_000_000

class TokenTrackerCallback(BaseCallbackHandler):
    """Token 消耗追踪回调"""

    def __init__(self):
        self.usage = TokenUsage()
        self.model = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.model = serialized.get("kwargs", {}).get("model_name", "unknown")

    def on_llm_end(self, response, **kwargs):
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.usage.add(
                self.model or "unknown",
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )

    def get_report(self) -> dict:
        return {
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
            "estimated_cost_usd": round(self.usage.total_cost, 4),
        }

# 使用
tracker = TokenTrackerCallback()
llm = ChatOpenAI(model="gpt-4o", callbacks=[tracker])

# 执行多次调用后查看报告
for i in range(10):
    llm.invoke(f"问题 {i}")

print(tracker.get_report())
# {'prompt_tokens': 1500, 'completion_tokens': 2000, 'total_tokens': 3500, 'estimated_cost_usd': 0.0238}
```

---

## 十、LangChain vs LlamaIndex 选型

### Q: LangChain 和 LlamaIndex 各自的定位和核心优势是什么？ ⭐⭐

**答**：

| 维度 | LangChain | LlamaIndex |
|------|-----------|------------|
| **定位** | 通用 LLM 应用开发框架 | 专注于数据连接和索引 |
| **核心优势** | 链式编排、Agent、工具生态 | RAG 深度优化、索引结构 |
| **学习曲线** | 较陡（概念多） | 较平缓（聚焦 RAG） |
| **适用场景** | 复杂 Agent、多步骤工作流 | 知识库问答、文档搜索 |
| **抽象层次** | 高（LCEL、Runnable） | 中（Index、Query Engine） |
| **社区生态** | 最大（集成 700+） | 大（专注 RAG 生态） |

---

### Q: 什么场景下应该选择 LangChain 而非 LlamaIndex？反之呢？ ⭐⭐⭐

**答**：

**选择 LangChain 的场景**：
1. 需要构建复杂的 Agent（多工具、多步骤推理）
2. 需要与多种外部系统集成（数据库、API、文件系统）
3. 需要 LCEL 的链式编排能力
4. 团队已有 LangChain 经验
5. 需要 LangSmith 的完整可观测性

**选择 LlamaIndex 的场景**：
1. 核心需求是文档问答/RAG
2. 需要复杂的索引结构（树索引、知识图谱索引）
3. 需要精细的检索策略（递归检索、子问题分解）
4. 数据源连接是核心需求
5. 需要快速搭建原型

**两者结合使用**：

```python
# 使用 LlamaIndex 的高级索引 + LangChain 的 Agent
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# LlamaIndex 构建索引
documents = SimpleDirectoryReader("./data/").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# 将 LlamaIndex 封装为 LangChain Tool
@tool
def search_knowledge_base(query: str) -> str:
    """搜索知识库获取相关信息"""
    response = query_engine.query(query)
    return str(response)

# 在 LangChain Agent 中使用
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手，使用工具回答问题"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, [search_knowledge_base], prompt)
executor = AgentExecutor(agent=agent, tools=[search_knowledge_base], verbose=True)
result = executor.invoke({"input": "公司的年假政策是什么？"})
```

---

### Q: 如果面试官问"你会如何评估一个 RAG 系统的质量"，如何用 LangChain + LangSmith 回答？ ⭐⭐⭐

**答**：

RAG 系统质量评估是一个系统工程，需要从多个维度评测：

```python
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI

client = Client()
judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 评估维度1：答案相关性（Answer Relevance）
def relevance_evaluator(run, example):
    """评估回答是否与问题相关"""
    question = example.inputs["question"]
    answer = run.outputs.get("answer", "")
    prompt = f"""评估回答是否与问题相关。
    问题：{question}
    回答：{answer}
    评分（0-1）："""
    score = float(judge_llm.invoke(prompt).content.strip())
    return {"key": "relevance", "score": score}

# 评估维度2：忠实度（Faithfulness）— 回答是否基于检索到的文档
def faithfulness_evaluator(run, example):
    """评估回答是否忠实于上下文"""
    answer = run.outputs.get("answer", "")
    context = run.outputs.get("context", "")
    prompt = f"""评估回答是否完全基于给定上下文。
    上下文：{context}
    回答：{answer}
    是否有上下文不支持的信息？评分（0-1，1=完全忠实）："""
    score = float(judge_llm.invoke(prompt).content.strip())
    return {"key": "faithfulness", "score": score}

# 评估维度3：上下文精确度 — 检索到的文档是否相关
def context_precision_evaluator(run, example):
    """评估检索质量"""
    question = example.inputs["question"]
    context = run.outputs.get("context", "")
    prompt = f"""评估检索到的上下文是否与问题相关。
    问题：{question}
    上下文：{context}
    评分（0-1）："""
    score = float(judge_llm.invoke(prompt).content.strip())
    return {"key": "context_precision", "score": score}

# 评估维度4：答案正确性
def correctness_evaluator(run, example):
    """评估答案是否正确"""
    answer = run.outputs.get("answer", "")
    reference = example.outputs.get("expected_answer", "")
    prompt = f"""评估回答是否正确。
    参考答案：{reference}
    实际回答：{answer}
    评分（0-1）："""
    score = float(judge_llm.invoke(prompt).content.strip())
    return {"key": "correctness", "score": score}

# 运行评估
def rag_target(inputs: dict) -> dict:
    docs = retriever.invoke(inputs["question"])
    context = "\n".join(d.page_content for d in docs)
    answer = chain.invoke({"question": inputs["question"], "context": context})
    return {"answer": answer, "context": context}

results = evaluate(
    rag_target,
    data="rag_eval_dataset",
    evaluators=[
        relevance_evaluator,
        faithfulness_evaluator,
        context_precision_evaluator,
        correctness_evaluator,
    ],
    experiment_prefix="rag_v3_eval",
)
```

**评估指标体系**：

| 指标 | 含义 | 衡量什么 |
|------|------|---------|
| Answer Relevance | 答案与问题的相关性 | 生成质量 |
| Faithfulness | 答案是否忠实于上下文 | 幻觉检测 |
| Context Precision | 检索结果的精确度 | 检索质量 |
| Context Recall | 检索结果的召回率 | 检索覆盖 |
| Answer Correctness | 答案的正确性 | 端到端质量 |

---

### Q: 如何在 LangChain 中实现一个生产级的多轮对话系统？ ⭐⭐⭐

**答**：

多轮对话需要管理会话状态、消息历史和上下文窗口：

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
import json
import redis
from typing import List

class ConversationManager:
    """生产级多轮对话管理器"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.max_history = 20  # 最多保留的轮数
        self.ttl = 3600 * 24   # 会话过期时间（24小时）

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="你是一个专业的助手。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        self.llm = ChatOpenAI(model="gpt-4o", streaming=True)
        self.chain = self.prompt | self.llm

    def _session_key(self, session_id: str) -> str:
        return f"chat:session:{session_id}"

    def get_history(self, session_id: str) -> List:
        """获取会话历史"""
        key = self._session_key(session_id)
        raw = self.redis.lrange(key, 0, -1)
        messages = []
        for item in raw:
            data = json.loads(item)
            if data["role"] == "human":
                messages.append(HumanMessage(content=data["content"]))
            elif data["role"] == "ai":
                messages.append(AIMessage(content=data["content"]))
        return messages[-self.max_history:]  # 截断到最大轮数

    def save_message(self, session_id: str, role: str, content: str):
        """保存消息到 Redis"""
        key = self._session_key(session_id)
        self.redis.rpush(key, json.dumps({"role": role, "content": content}))
        self.redis.expire(key, self.ttl)

    async def chat(self, session_id: str, user_input: str) -> str:
        """多轮对话"""
        history = self.get_history(session_id)
        self.save_message(session_id, "human", user_input)

        result = await self.chain.ainvoke({
            "history": history,
            "input": user_input,
        })

        self.save_message(session_id, "ai", result.content)
        return result.content

    def clear_session(self, session_id: str):
        """清除会话"""
        self.redis.delete(self._session_key(session_id))

# FastAPI 集成
from fastapi import FastAPI

app = FastAPI()
conv_manager = ConversationManager()

@app.post("/chat/{session_id}")
async def chat_endpoint(session_id: str, request: dict):
    answer = await conv_manager.chat(session_id, request["message"])
    return {"answer": answer, "session_id": session_id}

@app.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    conv_manager.clear_session(session_id)
    return {"status": "cleared"}
```

---

### Q: LangChain Expression Language (LCEL) 的底层执行机制是什么？它如何优化执行性能？ ⭐⭐⭐

**答**：

LCEL 的 pipe 运算符（`|`）构建的是一个有向无环图（DAG），执行时有以下优化：

1. **自动批量并行化**：`batch()` 方法对独立分支自动并行执行
2. **流式传播**：上游的流式输出自动传播到下游（通过 `transform` 方法）
3. **惰性执行**：只有在 `invoke`/`batch`/`stream` 时才真正执行
4. **自动类型推断**：通过 `InputType`/`OutputType` 自动验证输入输出

```python
from langchain_core.runnables import (
    RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableSequence
)

# LCEL 构建的实际上是 RunnableSequence
chain = prompt | llm | parser
print(type(chain))  # <class 'langchain_core.runnables.RunnableSequence'>

# 等价于显式构建
chain_explicit = RunnableSequence(first=prompt, middle=[llm], last=parser)

# 并行分支自动优化
parallel = RunnableParallel(
    summary=RunnableLambda(lambda x: summarize(x)) | llm,
    keywords=RunnableLambda(lambda x: extract_keywords(x)) | llm,
    sentiment=RunnableLambda(lambda x: analyze_sentiment(x)) | llm,
)
# batch 时，三个分支会并发执行，而非串行

# 流式传播优化
streaming_chain = prompt | ChatOpenAI(streaming=True) | parser
# parser 会逐 chunk 处理 LLM 的流式输出，而非等待完整响应
```

---

### Q: 如何在 LangChain 中实现复杂的条件分支和路由逻辑？ ⭐⭐⭐

**答**：

```python
from langchain_core.runnables import (
    RunnableBranch, RunnableLambda, RunnablePassthrough
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 路由器：根据问题类型选择不同处理链
def classify_query(input_data: dict) -> str:
    """简单分类器"""
    query = input_data["query"].lower()
    if any(kw in query for kw in ["价格", "费用", "多少钱"]):
        return "pricing"
    elif any(kw in query for kw in ["怎么", "如何", "教程"]):
        return "howto"
    elif any(kw in query for kw in ["投诉", "问题", "bug"]):
        return "support"
    return "general"

def route(input_data: dict) -> RunnableLambda:
    category = classify_query(input_data)
    routes = {
        "pricing": pricing_chain,
        "howto": howto_chain,
        "support": support_chain,
        "general": general_chain,
    }
    return routes.get(category, general_chain)

# 各分支链
llm = ChatOpenAI(model="gpt-4o")

pricing_chain = (
    ChatPromptTemplate.from_template("你是一个定价专家。回答：{query}")
    | llm | StrOutputParser()
)
howto_chain = (
    ChatPromptTemplate.from_template("你是一个教程作者。回答：{query}")
    | llm | StrOutputParser()
)
support_chain = (
    ChatPromptTemplate.from_template("你是一个客服。回答：{query}")
    | llm | StrOutputParser()
)
general_chain = (
    ChatPromptTemplate.from_template("回答：{query}")
    | llm | StrOutputParser()
)

# 方式1：使用 RunnableBranch
branch = RunnableBranch(
    (lambda x: classify_query(x) == "pricing", pricing_chain),
    (lambda x: classify_query(x) == "howto", howto_chain),
    (lambda x: classify_query(x) == "support", support_chain),
    general_chain,  # 默认分支
)

result = branch.invoke({"query": "这个产品多少钱？"})
print(result)

# 方式2：使用自定义路由函数
from langchain_core.runnables import Runnable

class Router(Runnable):
    """自定义路由 Runnable"""

    def invoke(self, input_data, config=None):
        category = classify_query(input_data)
        chain_map = {
            "pricing": pricing_chain,
            "howto": howto_chain,
            "support": support_chain,
        }
        selected = chain_map.get(category, general_chain)
        return selected.invoke(input_data, config=config)

router = Router()
result = router.invoke({"query": "如何使用这个 API？"})
```

---

### Q: LangChain 中如何实现 Structured Output（结构化输出）？有哪些最佳实践？ ⭐⭐⭐

**答**：

结构化输出是将 LLM 的自由文本输出转为结构化数据（JSON、Pydantic 对象）的技术：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field
from typing import List

# 方法1：使用 Pydantic + with_structured_output（推荐）
class MovieReview(BaseModel):
    """电影评价结构"""
    title: str = Field(description="电影名称")
    rating: float = Field(description="评分，1-10")
    summary: str = Field(description="一句话总结")
    pros: List[str] = Field(description="优点列表")
    cons: List[str] = Field(description="缺点列表")
    recommend: bool = Field(description="是否推荐")

llm = ChatOpenAI(model="gpt-4o")

# with_structured_output 自动使用 function calling / JSON mode
structured_llm = llm.with_structured_output(MovieReview)

result = structured_llm.invoke("评价一下电影《流浪地球》")
print(type(result))  # <class 'MovieReview'>
print(result.rating)
print(result.pros)

# 方法2：使用 JsonOutputParser（不依赖 function calling）
parser = JsonOutputParser(pydantic_object=MovieReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个影评人。{format_instructions}"),
    ("human", "评价电影：{movie}"),
])

chain = prompt | llm | parser
result = chain.invoke({
    "movie": "流浪地球",
    "format_instructions": parser.get_format_instructions(),
})

# 方法3：使用 PydanticToolsParser（多结构化输出）
from langchain_core.utils.function_calling import tool
from langchain_core.output_parsers import PydanticToolsParser

@tool
def review_movie(
    title: str = Field(description="电影名称"),
    rating: float = Field(description="评分 1-10"),
    summary: str = Field(description="一句话总结"),
) -> dict:
    """生成电影评价"""
    return {"title": title, "rating": rating, "summary": summary}

structured_llm_v2 = llm.with_structured_output(
    MovieReview, method="json_mode"  # 使用 JSON mode 而非 function calling
)
```

---

### Q: 如何在 LangChain 中实现自定义 Callback Handler 进行调试和监控？ ⭐⭐⭐

**答**：

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List, Optional, Union
from langchain_core.messages import BaseMessage
import time
import logging
import json

logger = logging.getLogger(__name__)

class ProductionCallbackHandler(BaseCallbackHandler):
    """生产级回调处理器"""

    def __init__(self):
        self.metrics = {
            "total_llm_calls": 0,
            "total_tokens": 0,
            "total_errors": 0,
            "total_latency_ms": 0,
        }
        self._start_times: Dict[str, float] = {}

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id=None,
        **kwargs,
    ):
        self._start_times[str(run_id)] = time.time()
        self.metrics["total_llm_calls"] += 1
        logger.info(f"LLM 调用开始: {serialized.get('name', 'unknown')}")

    def on_llm_end(self, response: LLMResult, *, run_id=None, **kwargs):
        start = self._start_times.pop(str(run_id), time.time())
        latency = (time.time() - start) * 1000
        self.metrics["total_latency_ms"] += latency

        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            tokens = usage.get("total_tokens", 0)
            self.metrics["total_tokens"] += tokens
            logger.info(f"LLM 调用完成: {latency:.0f}ms, {tokens} tokens")

    def on_llm_error(self, error: BaseException, *, run_id=None, **kwargs):
        self.metrics["total_errors"] += 1
        logger.error(f"LLM 调用失败: {error}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any],
        *, run_id=None, **kwargs
    ):
        chain_name = serialized.get("name", "unknown")
        logger.debug(f"Chain 开始: {chain_name}")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str,
        *, run_id=None, **kwargs
    ):
        tool_name = serialized.get("name", "unknown")
        logger.info(f"Tool 调用: {tool_name}, 输入: {input_str[:100]}")

    def on_tool_end(self, output: str, *, run_id=None, **kwargs):
        logger.info(f"Tool 完成: 输出长度 {len(output)}")

    def get_metrics(self) -> dict:
        avg_latency = (
            self.metrics["total_latency_ms"] / self.metrics["total_llm_calls"]
            if self.metrics["total_llm_calls"] > 0 else 0
        )
        return {**self.metrics, "avg_latency_ms": round(avg_latency, 2)}

# 使用
callback = ProductionCallbackHandler()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", callbacks=[callback])

# 所有调用自动通过 callback 记录
for i in range(5):
    llm.invoke(f"测试问题 {i}")

print(json.dumps(callback.get_metrics(), indent=2, ensure_ascii=False))
# {
#   "total_llm_calls": 5,
#   "total_tokens": 1250,
#   "total_errors": 0,
#   "total_latency_ms": 3500.0,
#   "avg_latency_ms": 700.00
# }
```

---

### Q: 面试总结：如何系统性地向面试官展示你对 LangChain 的深度理解？ ⭐⭐⭐

**答**：

建议按以下框架组织回答：

1. **架构认知**：说明 LangChain 的分层架构（langchain-core → langchain-xxx → langchain），强调 LCEL 和 Runnable 的设计哲学

2. **实践经验**：举具体例子说明你在生产中如何解决：
   - 流式输出（FastAPI SSE）
   - 错误处理（fallback + retry + 熔断）
   - 性能优化（缓存、批量、模型路由）
   - 可观测性（LangSmith 集成）

3. **选型判断**：能客观对比 LangChain vs LlamaIndex vs 自研，说明各场景的最优选择

4. **问题意识**：能指出 LangChain 的局限性：
   - 抽象层过厚导致调试困难
   - 版本迭代频繁，API 不稳定
   - 某些场景下直接用 SDK 更简单

5. **演进视野**：了解 LangChain 的发展方向（LCEL 2.0、langgraph、langsmith 深度集成）

> **面试黄金话术**："LangChain 是一个很好的起点，但在生产环境中，我们会根据实际需求做裁剪——简单场景直接用 OpenAI SDK + 自定义封装，复杂场景才引入 LangChain 的全套抽象。核心原则是：**抽象服务于需求，而非为了抽象而抽象**。"
