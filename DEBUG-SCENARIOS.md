# 🔧 生产环境故障排查场景

> 10 个真实故障场景，考察排查思路和解决问题的能力

---

## 场景 1：RAG 系统返回无关结果 — Embedding 模型版本不一致

### 场景描述

某知识问答系统上线三个月，近期用户频繁反馈"搜索结果答非所问"。系统采用 RAG 架构，文档已入库，查询和文档分别经过 Embedding 模型编码后做相似度检索。团队近期进行过一次模型升级，将 Embedding 模型从 `text-embedding-ada-002` 迁移到 `text-embedding-3-small`，但数据库中的历史向量未重新生成。

### 症状

- 用户查询"退货政策"，返回的 Top-5 结果是完全无关的技术文档
- 召回率（Recall@5）从 92% 暴跌至 34%
- 新入库的文档检索正常，老文档检索异常
- 同一个查询在 staging 环境正常，production 环境异常

### 排查步骤

1. **检查检索日志**：查看查询向量和返回文档的相似度分数，发现分数普遍低于 0.3（正常应在 0.7+），说明向量空间不对齐
2. **对比 Embedding 模型版本**：检查 pipeline 配置，发现查询侧使用 `text-embedding-3-small`，但数据库中大量文档仍由 `text-embedding-ada-002` 编码
3. **验证向量维度**：`ada-002` 输出 1536 维，`3-small` 输出 1536 维，维度相同但语义空间不同，导致余弦相似度计算无意义
4. **检查数据入库流水线**：确认新文档入库时自动使用了新模型，但历史数据迁移脚本未执行
5. **确认 staging 环境**：staging 是全量重建的，所以没有问题

### 根因

Embedding 模型升级时只更新了查询侧的模型，未对已入库的向量做全量重建。不同版本的 Embedding 模型输出的向量空间不兼容，导致新旧向量无法正确比较。

### 解决方案

```python
# 方案一：全量重建向量索引
async def rebuild_vectors():
    documents = await db.get_all_documents()
    for batch in chunk(documents, size=100):
        new_embeddings = await embedding_model.encode(
            [doc.content for doc in batch],
            model="text-embedding-3-small"
        )
        await vector_db.upsert(batch.ids, new_embeddings)

# 方案二：灰度切换，同时支持两套向量
class DualVectorRetriever:
    async def retrieve(self, query, top_k=5):
        new_results = await self.new_index.search(query)
        old_results = await self.old_index.search(query)
        return self.merge_and_rerank(new_results, old_results, top_k)
```

### 预防措施

- 建立 Embedding 模型版本管理机制，向量数据中记录生成模型版本号
- 模型升级时制定完整的迁移计划，包含全量重建步骤
- 在监控中增加检索质量指标（如 Recall@5、MRR），设置告警阈值
- 灰度发布：先在小流量验证检索质量，再全量切换

---

## 场景 2：Agent 陷入死循环 — ReAct 循环没有退出条件

### 场景描述

一个基于 ReAct（Reasoning + Acting）模式的客服 Agent，负责处理用户的退换货请求。某日运维发现某台服务器 CPU 持续 100%，日志量暴增。排查发现一个 Agent 实例在同一个 Thought-Action-Observation 循环中跑了 4 小时，消耗了大量 token 和计算资源。

### 症状

- 单个 Agent 会话持续时间异常（正常 < 30 秒，异常 > 4 小时）
- 该会话的 API 调用次数超过 500 次
- CPU 和内存使用率持续偏高
- 日志中可见重复的 Thought → Action → Observation 模式
- 用户早已断开连接，但 Agent 仍在运行

### 排查步骤

1. **检查日志流**：发现 Agent 反复调用同一个工具 `search_order`，每次返回"订单不存在"，但 Agent 持续尝试
2. **分析 ReAct 循环**：Agent 的 Thought 步骤不断变化措辞但本质相同——"让我再试一次搜索"
3. **检查退出条件**：代码中没有设置最大循环次数限制
4. **检查工具返回格式**：工具返回的 Observation 没有明确的"失败"标识，Agent 无法判断是否应该放弃
5. **检查超时机制**：设置了 HTTP 请求超时，但没有设置整个会话的超时

### 根因

ReAct 循环缺少三个关键保护机制：① 最大迭代次数限制；② 工具调用失败的明确反馈；③ 整体会话超时。Agent 在工具持续返回非致命错误时，无法做出"放弃并回复用户"的决策。

### 解决方案

```python
class ReActAgent:
    def __init__(self, max_iterations=15, session_timeout=120):
        self.max_iterations = max_iterations
        self.session_timeout = session_timeout

    async def run(self, user_query):
        start_time = time.time()
        iterations = 0
        seen_actions = set()

        while iterations < self.max_iterations:
            # 会话超时检查
            if time.time() - start_time > self.session_timeout:
                return self.fallback_response("处理超时，请稍后重试")

            thought, action = await self.think(user_query, history)
            result = await self.execute(action)

            # 检测重复动作（防止循环）
            action_key = self._action_signature(action)
            if action_key in seen_actions:
                return self.fallback_response("无法完成操作，转人工客服")
            seen_actions.add(action_key)

            # 工具返回明确失败时，终止循环
            if result.is_error and result.retryable is False:
                break

            iterations += 1

        return self.fallback_response("已达最大尝试次数")
```

### 预防措施

- 所有 Agent 循环必须设置 `max_iterations` 上限（推荐 10-20）
- 设置会话级别的超时（推荐 60-180 秒）
- 实现重复动作检测，连续 2-3 次相同动作时强制退出
- 工具接口标准化，区分可重试错误和不可重试错误
- 添加监控告警：单次会话 token 消耗超过阈值时触发

---

## 场景 3：LLM API 间歇性超时 — 连接池耗尽

### 场景描述

一个高并发的 LLM 应用，每天处理 10 万+ 请求。上线半年一直稳定运行，但近期出现间歇性 API 调用超时。超时集中在每小时的第 15 分钟和第 45 分钟，持续约 2-3 分钟后自行恢复。非高峰期偶尔也会出现，但概率较低。

### 症状

- API 调用返回 `TimeoutError` 或 `ConnectionPoolExhausted` 错误
- 错误呈现周期性模式（每 30 分钟约 2-3 分钟的故障窗口）
- 高峰期超时率约 5%，低峰期约 0.1%
- 重试后大概率成功
- LLM 服务端监控显示 CPU/GPU 正常，未过载

### 排查步骤

1. **检查 LLM 服务端**：服务端指标正常，排除模型推理瓶颈
2. **分析超时时间分布**：发现与定时任务时间吻合——每 30 分钟有一个批量摘要任务在跑
3. **检查 HTTP 连接池配置**：`httpx.AsyncClient` 的连接池大小为 10，最大保持连接数为 20
4. **查看并发量**：正常并发约 15 个请求，批量任务会瞬间增加 50 个并发
5. **连接池等待队列**：当连接池满时，新请求进入等待队列，队列默认无超时，积压导致后续请求连锁超时
6. **检查 Keep-Alive 配置**：连接复用超时设置过长（300s），空闲连接未及时释放

### 根因

HTTP 连接池大小不足以应对突发并发，且批量定时任务未做流量整形。连接池耗尽后新请求排队等待，形成级联超时。Keep-Alive 超时过长导致空闲连接无法及时释放。

### 解决方案

```python
import httpx

# 优化连接池配置
client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=100,           # 总连接数
        max_keepalive_connections=50,  # 保持连接数
        keepalive_expiry=30,          # 空闲连接回收时间
    ),
    timeout=httpx.Timeout(
        connect=5.0,    # 连接超时
        read=60.0,      # 读取超时
        write=10.0,     # 写入超时
        pool=10.0,      # 从连接池获取连接的超时
    ),
)

# 批量任务添加限流
class RateLimitedBatch:
    def __init__(self, max_concurrent=10):
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process(self, items):
        async def _process_one(item):
            async with self.semaphore:
                return await call_llm(item)
        return await asyncio.gather(*[_process_one(i) for i in items])
```

### 预防措施

- 连接池大小根据并发量合理设置，预留 2-3 倍余量
- 批量任务与在线流量隔离，使用独立的连接池和客户端实例
- 设置连接池等待超时（`pool timeout`），避免无限排队
- 监控连接池使用率，设置告警（>80% 使用率）
- 实现自适应限流，根据响应延迟动态调整并发度

---

## 场景 4：流式响应卡顿 — Token 生成速度不均匀

### 场景描述

某聊天应用使用 SSE（Server-Sent Events）流式返回 LLM 响应。用户反馈"回答像是卡了一下又突然蹦出来"，体验远不如竞品流畅。技术团队实测发现前 3-5 秒完全没有输出，之后突然快速吐出大量 token。

### 症状

- 流式响应前 3-5 秒空白，无任何 token 输出
- 空白期后 token 以极快速度批量输出
- 用户感知体验远差于非流式模式
- 不同长度的输出，空白时间相对固定
- GPU 利用率在空白期很低，之后突然飙高

### 排查步骤

1. **抓包分析 SSE 事件时序**：前 3-5 秒无数据包，之后瞬间出现大量数据
2. **检查 LLM 推理引擎**：使用 vLLM 部署，检查发现开启了一个 `prefill` 优化选项，会先完成所有 prompt 的预填充再开始解码
3. **检查 prompt 长度**：System prompt + 历史消息的 prompt 约 2000 token，prefill 阶段耗时 3-4 秒
4. **检查流式传输配置**：服务端缓冲区设为 4KB，小 token 量不足以触发 flush
5. **对比测试**：缩短 prompt 后空白时间明显减少，确认与 prefill 相关
6. **检查 chunk 传输**：Nginx 的 proxy_buffering 未关闭，存在额外缓冲

### 根因

两个因素叠加：① vLLM 的 prefill 阶段需要处理完整的 prompt 才能开始解码输出，长 prompt 导致较长的首次 token 延迟（TTFT）；② Nginx 代理层的 proxy_buffering 开启，小块数据被缓冲后才发送给客户端。

### 解决方案

```nginx
# Nginx 配置：关闭 proxy buffering
location /api/chat {
    proxy_pass http://llm_backend;
    proxy_buffering off;              # 关键：关闭缓冲
    proxy_cache off;
    chunked_transfer_encoding on;
    proxy_set_header Connection '';
    proxy_http_version 1.1;
}
```

```python
# 服务端：确保立即 flush
async def stream_response(prompt):
    async for chunk in llm.generate_stream(prompt):
        yield f"data: {json.dumps({'text': chunk})}\n\n"
        await asyncio.sleep(0)  # 让出控制权，触发 flush

# vLLM 配置优化
engine_args = {
    "enable_chunked_prefill": True,    # 分块 prefill
    "max_num_batched_tokens": 512,     # 控制 prefill chunk 大小
    "max_num_seqs": 256,
}
```

### 预防措施

- 开启 `chunked_prefill` 让 prefill 阶段也能输出中间结果
- 关闭所有代理层的 buffering（Nginx、CDN 等）
- 监控 TTFT（Time to First Token）指标，P95 应 < 1 秒
- 控制 prompt 长度，减少不必要的 system prompt
- 前端实现 streaming 打字机效果，对小延迟做平滑处理

---

## 场景 5：模型幻觉率突增 — Prompt 模板被覆盖

### 场景描述

某医疗问答系统上线后一直运行良好，幻觉率控制在 2% 以下。周一上线后突然收到大量投诉——模型开始编造药物剂量和不存在的医学术语。团队紧急回滚了代码，但 diff 显示最近一周没有代码变更。

### 症状

- 模型回答中出现虚构的药物名称和剂量
- 幻觉率从 1.8% 飙升至 23%
- 只有医疗场景受影响，通用问答正常
- 周一凌晨 2:00 后开始出现异常
- 回滚代码到上周五版本后问题依旧

### 排查步骤

1. **检查代码变更**：Git 日志显示周末无人提交代码，排除代码变更
2. **检查 Prompt 模板**：在数据库中发现 system prompt 模板在周日凌晨被修改
3. **排查模板管理流程**：运营团队通过后台 CMS 修改了 Prompt 模板，添加了"请用专业、自信的语气回答"，删除了"不确定时请明确告知用户"
4. **检查审核流程**：Prompt 模板的修改没有代码审查，运营人员直接修改数据库
5. **对比新旧模板**：旧模板包含 `"If you are not certain, explicitly state that you don't know"`，新模板中该指令被删除

### 根因

运营人员通过 CMS 后台修改了 Prompt 模板，无意中删除了关键的反幻觉指令（"不确定时明确告知"），并添加了鼓励自信回答的措辞。Prompt 模板变更没有经过审核流程，也未纳入版本控制。

### 解决方案

```python
# Prompt 模板版本管理
class PromptManager:
    def __init__(self):
        self.template_registry = {}

    def get_template(self, template_name, version=None):
        if version:
            return self.template_registry[template_name][version]
        return self.template_registry[template_name]["latest"]

    def update_template(self, template_name, new_content, author):
        # 强制审核流程
        review_id = self.create_review(
            template_name=template_name,
            new_content=new_content,
            author=author,
            requires_approval=True,  # 必须审核
        )
        # 模板中必须包含安全指令
        required_segments = [
            "If you are not certain",
            "explicitly state",
        ]
        for segment in required_segments:
            if segment.lower() not in new_content.lower():
                raise TemplateValidationError(
                    f"Missing required safety instruction: {segment}"
                )
        return review_id
```

### 预防措施

- Prompt 模板纳入 Git 版本控制，变更必须经过 Code Review
- 保留"安全锚点指令"（Safety Anchor），不允许运营人员修改
- 建立 Prompt 模板的自动化测试：输入已知问题，检查幻觉率
- 每次 Prompt 变更后自动运行评估基准测试
- 实现灰度发布：新 Prompt 先对 1% 流量生效，观察指标

---

## 场景 6：向量数据库查询变慢 — 索引碎片化

### 场景描述

系统使用 Milvus 向量数据库存储 500 万条文档向量。初期查询延迟 < 50ms，运行半年后查询延迟逐渐升至 500ms-2s。期间不断有新文档入库，旧文档被标记删除但未物理清理。

### 症状

- 向量检索 P95 延迟从 50ms 升至 800ms
- 内存使用量持续增长
- 数据库大小（磁盘）比实际文档数量大 40%
- 偶发的超时错误，特别是在高并发时
- CPU 使用率正常，瓶颈在 I/O

### 排查步骤

1. **监控分析**：查询延迟呈线性增长趋势，与数据量增长不成比例（实际活跃数据仅 300 万条）
2. **检查删除策略**：发现使用了软删除（标记 `is_deleted=True`），但没有物理清理任务
3. **分析索引状态**：HNSW 索引中包含了大量已删除的节点，这些节点在搜索时仍需遍历
4. **检查 segment 状态**：Milvus 中存在大量小 segment（碎片），由频繁的小批量插入造成
5. **Compaction 状态**：自动 compaction 未生效，手动触发后延迟下降但未恢复到初始水平
6. **索引参数检查**：`ef_search` 参数过低，碎片化后需要更大的搜索范围

### 根因

两个问题叠加：① 软删除后未做物理清理，HNSW 索引中死节点持续增长，搜索时遍历无效节点；② 频繁小批量写入导致 segment 碎片化，降低了索引效率。

### 解决方案

```python
# 1. 定期物理清理
async def cleanup_deleted_vectors():
    """定期清理已标记删除的向量"""
    deleted_count = await milvus.delete_entities(
        expr="is_deleted == true",
        physical=True  # 物理删除
    )
    logger.info(f"Cleaned up {deleted_count} deleted vectors")

# 2. 定期重建索引
async def rebuild_index():
    """在低峰期重建索引"""
    collection = milvus.get_collection("documents")
    # 停止服务写入
    collection.release()
    # 删除旧索引
    collection.drop_index()
    # 重建索引，调整参数
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {
                "M": 32,
                "efConstruction": 256,
            }
        }
    )
    collection.load()

# 3. 批量写入优化
class BatchWriter:
    def __init__(self, batch_size=10000, flush_interval=60):
        self.buffer = []
        self.batch_size = batch_size

    async def add(self, vectors):
        self.buffer.extend(vectors)
        if len(self.buffer) >= self.batch_size:
            await self.flush()
```

### 预防措施

- 定期执行物理清理（建议每周一次，低峰期）
- 监控索引碎片率，超过阈值时自动触发 compaction
- 使用批量写入，避免小批量频繁插入
- 监控查询延迟趋势，设置渐进式告警
- 为 Milvus 配置自动 compaction 策略

---

## 场景 7：多 Agent 系统消息丢失 — 消息队列积压

### 场景描述

一个由 5 个 Agent 组成的协作系统（规划 Agent、研究 Agent、写作 Agent、审核 Agent、发布 Agent），通过 RabbitMQ 传递任务消息。用户报告"任务提交后没有结果返回"，发生率约 5%。

### 症状

- 约 5% 的任务提交后无响应
- 问题任务的 Agent 链在中途断开
- RabbitMQ 管理界面显示某队列消息数持续增长
- 部分消息的 delivery count > 1（被重复投递但未 ACK）
- 没有错误日志，消息似乎"静默消失"

### 排查步骤

1. **检查消息队列监控**：`agent_writing_queue` 的消息数从正常的 0 增长到 12000+
2. **检查消费者状态**：Writing Agent 实例数为 2，但其中一个已断开连接（网络闪断后未自动重连）
3. **分析消息投递**：消息被投递给断开的消费者后，等待 ACK 超时后重新入队，但重试次数达到上限后消息被路由到死信队列
4. **检查死信队列**：死信队列（DLQ）未配置消费者，消息堆积无人处理
5. **检查错误处理**：Writing Agent 处理消息时如果遇到 LLM API 超时，会抛异常但不 NACK 消息，导致消息既没 ACK 也没 NACK
6. **检查消息 TTL**：消息 TTL 设置为 1 小时，超时后消息被丢弃

### 根因

多重因素导致：① 消费者实例网络闪断后未自动重连，消息被投递到不可用的消费者；② 错误处理不当，异常导致消息既不 ACK 也不 NACK；③ 死信队列无消费者，丢失的消息无法恢复；④ 消息 TTL 过短，超过 1 小时未处理的消息被永久丢弃。

### 解决方案

```python
# 健壮的消费者实现
class RobustAgentConsumer:
    def __init__(self, queue_name, agent):
        self.queue_name = queue_name
        self.agent = agent
        self.max_retries = 5

    async def start(self):
        while True:  # 自动重连
            try:
                connection = await aio_pika.connect_robust(
                    "amqp://localhost",
                    reconnect_interval=5,  # 5 秒重连
                )
                channel = await connection.channel()
                await channel.set_qos(prefetch_count=10)
                queue = await channel.declare_queue(
                    self.queue_name,
                    arguments={
                        "x-dead-letter-exchange": "dlx",
                        "x-message-ttl": 86400000,  # 24 小时 TTL
                    }
                )
                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        await self.process_message(message)
            except aio_pika.exceptions.AMQPConnectionError:
                logger.warning("Connection lost, reconnecting...")
                await asyncio.sleep(5)

    async def process_message(self, message):
        try:
            result = await self.agent.process(
                json.loads(message.body)
            )
            await message.ack()
        except Exception as e:
            retry_count = message.headers.get("x-retry-count", 0)
            if retry_count < self.max_retries:
                # 重新发布，增加重试计数
                await self.republish(message, retry_count + 1)
            await message.ack()  # 确保 ACK，避免阻塞队列
```

### 预防措施

- 消费者必须实现自动重连机制（`connect_robust`）
- 所有消息必须最终被 ACK 或 NACK，不能有中间状态
- 配置死信队列并添加消费者，用于人工检查和恢复
- 设置合理的消息 TTL（建议 24 小时+）
- 监控队列深度，消息积压超过阈值时告警

---

## 场景 8：模型部署 OOM — KV Cache 内存泄漏

### 场景描述

使用 vLLM 部署了一个 70B 参数的模型，配有 4 张 A100 80GB GPU。系统运行正常约 2 周后，频繁出现 OOM（Out of Memory）错误，导致推理服务崩溃。每次重启后能恢复正常，但 2-3 天后再次出现。

### 症状

- 推理服务运行 2-3 天后 OOM 崩溃
- 重启后恢复正常，问题周期性出现
- GPU 显存使用量随时间单调递增
- 服务承载的并发量没有明显变化
- nvidia-smi 显示已用显存从启动时的 60GB 增长到 79GB+

### 排查步骤

1. **监控显存趋势**：绘制显存使用量随时间的变化曲线，确认是单调递增而非突发
2. **分析 KV Cache 分配**：vLLM 的 `GPU memory utilization` 设置为 0.9，理论应有足够空间
3. **检查请求模式**：发现存在大量超长上下文请求（32K+ token），长上下文的 KV Cache 占用远超预期
4. **检查 Cache 淘汰策略**：vLLM 的 preemption 策略为 `recompute`，但某些请求的 `max_model_len` 被设为超过模型最大长度
5. **检查请求超时**：部分长时间运行的请求（超 60 秒）未正常返回，其 KV Cache 一直被占用
6. **内存碎片分析**：频繁分配和释放不同大小的 KV Cache 导致显存碎片化，虽然总量够用但无法分配连续空间

### 根因

KV Cache 内存泄漏的根本原因是：① 长上下文请求的 KV Cache 未在请求结束后正确释放（vLLM bug，特定版本已知问题）；② 显存碎片化加剧了问题，使得即使有剩余空间也无法满足新请求的连续分配需求；③ 请求超时配置过长，僵尸请求长期占用 Cache。

### 解决方案

```python
# vLLM 启动配置优化
engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-3-70B",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.85,    # 留出更多余量
    max_model_len=8192,             # 限制最大上下文长度
    max_num_seqs=64,                # 限制并发序列数
    enforce_eager=True,             # 避免 CUDA graph 的额外内存碎片
    enable_prefix_caching=False,    # 关闭 prefix caching 减少内存开销
)

# 请求级超时保护
class RequestTimeoutMiddleware:
    async def __call__(self, request):
        try:
            result = await asyncio.wait_for(
                self.engine.generate(request),
                timeout=45.0,  # 45 秒超时
            )
            return result
        except asyncio.TimeoutError:
            # 强制终止请求，释放 KV Cache
            await self.engine.abort(request.request_id)
            return error_response("Request timeout")

# 定期内存健康检查
async def memory_health_check():
    while True:
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        fragmentation = 1 - (allocated / reserved)

        if fragmentation > 0.3:  # 碎片率 > 30%
            logger.warning(f"High memory fragmentation: {fragmentation:.2%}")
            # 优雅重启
            await graceful_restart()

        await asyncio.sleep(300)  # 每 5 分钟检查
```

### 预防措施

- 升级 vLLM 到已修复 KV Cache 泄漏的版本
- 限制最大上下文长度，超长请求需特殊处理
- 设置请求级超时，超时后强制释放资源
- 监控 GPU 显存使用趋势，设置递增告警
- 定期（如每天凌晨）做滚动重启
- 预留 15% 的 GPU 显存作为安全余量

---

## 场景 9：用户反馈回答质量下降 — 数据漂移

### 场景描述

一个基于微调模型的金融分析系统，部署时评估指标 F1=0.91。运行 6 个月后，用户满意度从 4.2 分降至 3.1 分（5 分制）。模型的自动评估指标仍显示 F1=0.88，看似只下降了 3%，但用户体验断崖式下降。

### 症状

- 用户满意度评分从 4.2 降至 3.1
- 自动评估指标（F1）仅从 0.91 降至 0.88
- 模型在旧测试集上表现正常
- 用户投诉集中在"回答过时""不理解新概念"
- 新出现的金融术语（如新型衍生品名称）模型无法处理

### 排查步骤

1. **分析用户投诉**：收集 top-50 被投诉的回答，发现 80% 涉及最近 6 个月新出现的金融产品和术语
2. **检查训练数据**：训练数据截止日期为 6 个月前，新概念完全缺失
3. **对比输入分布**：分析近 30 天的查询，发现 35% 的查询包含训练数据中不存在的术语
4. **数据漂移检测**：用 KL 散度对比新旧查询的 embedding 分布，发现显著漂移
5. **检查评估数据集**：自动评估用的是旧测试集，无法检测到新概念上的失败
6. **分析具体案例**：如"ESG 债券"概念在最近半年才广泛使用，模型将其误分类为普通债券

### 根因

典型的数据漂移（Data Drift）问题。用户的查询分布随时间变化，新金融概念和术语不断出现，但模型的训练数据有截止日期，无法覆盖新概念。自动评估使用的是静态测试集，掩盖了模型在新数据上的退化。

### 解决方案

```python
# 1. 数据漂移检测系统
class DriftDetector:
    def __init__(self, reference_embeddings):
        self.reference_dist = self._compute_distribution(
            reference_embeddings
        )

    def detect_drift(self, recent_embeddings, threshold=0.15):
        current_dist = self._compute_distribution(recent_embeddings)
        kl_div = scipy.stats.entropy(
            current_dist, self.reference_dist
        )
        return {
            "drift_detected": kl_div > threshold,
            "kl_divergence": kl_div,
            "threshold": threshold,
        }

    def detect_ood_queries(self, query, known_terms):
        """检测超出分布范围的查询"""
        tokens = tokenize(query)
        unknown_ratio = len(
            [t for t in tokens if t not in known_terms]
        ) / len(tokens)
        return unknown_ratio > 0.3

# 2. 动态评估流水线
class DynamicEvaluator:
    def __init__(self):
        self.eval_sets = {
            "static": load_static_test_set(),
            "recent": load_recent_queries(days=7),
            "complaints": load_complaint_queries(),
        }

    def evaluate(self, model):
        results = {}
        for name, data in self.eval_sets.items():
            results[name] = {
                "f1": compute_f1(model, data),
                "sample_size": len(data),
            }
        return results

# 3. 增量学习管道
async def incremental_update_pipeline():
    # 收集近期高质量问答对
    new_data = await collect_recent_high_quality_qa(days=30)
    # 标注和审核
    annotated = await human_review(new_data)
    # 增量微调
    model = await fine_tune_incremental(
        base_model=current_model,
        new_data=annotated,
        epochs=3,
    )
    # A/B 测试
    await ab_test(model, traffic_ratio=0.1)
```

### 预防措施

- 建立数据漂移检测系统，每日监控查询分布变化
- 动态评估集必须包含近期真实查询，不能只用静态测试集
- 设置漂移告警阈值，超过时自动触发模型更新流程
- 定期（如每月）进行增量微调，引入新领域数据
- 建立用户反馈闭环，投诉数据自动进入评估和训练流程
- 对于 OOD（Out-of-Distribution）查询，模型应识别并提示"该概念超出我的知识范围"

---

## 场景 10：API 成本异常飙升 — 重试风暴

### 场景描述

某 LLM 应用使用 OpenAI API，月度账单通常在 $5,000 左右。某月账单突然飙升至 $45,000，增长了 9 倍。团队排查发现 3 天的消耗量就超过了平时一个月的量。OpenAI 后台显示 API 调用量是平时的 8 倍。

### 症状

- 月度 API 费用从 $5,000 飙升至 $45,000
- API 调用量是平时的 8 倍
- 用户量和业务量没有明显变化
- 某些高价值模型（如 GPT-4）的调用量增长更明显（12 倍）
- 监控面板显示大量 `500` 和 `503` 错误对应的重试请求

### 排查步骤

1. **分析 API 调用日志**：发现同一请求平均被调用 6-8 次，说明存在大量重试
2. **检查重试逻辑**：代码中对所有 HTTP 错误（包括 429、500、503）进行无限重试
3. **分析错误分布**：错误集中在 OpenAI 的 rate limit（429）和偶发 500 错误
4. **检查退避策略**：重试间隔为固定 1 秒（无指数退避），且没有最大重试次数限制
5. **检查并发控制**：多线程/多进程环境下，每个 worker 独立重试，重试量被倍增
6. **检查幂等性**：重试的请求是幂等的查询还是会产生实际消耗的生成请求

### 根因

经典的"重试风暴"（Retry Storm）问题。重试策略存在三个致命缺陷：① 没有最大重试次数限制；② 没有指数退避和抖动（jitter），所有重试几乎同时触发；③ 未区分可重试错误和不可重试错误。当 OpenAI 返回 rate limit 时，大量客户端同时重试，形成放大效应，加剧了 rate limit，形成恶性循环。

### 解决方案

```python
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

# 正确的重试策略
@retry(
    stop=stop_after_attempt(3),  # 最多重试 3 次
    wait=wait_exponential_jitter(
        initial=1,      # 初始等待 1 秒
        max=60,          # 最大等待 60 秒
        jitter=5,        # 随机抖动 ±5 秒
    ),
    retry=retry_if_exception_type((
        openai.RateLimitError,
        openai.InternalServerError,
        openai.APITimeoutError,
    )),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying API call, attempt {retry_state.attempt_number}"
    ),
)
async def call_llm(prompt, model="gpt-4"):
    return await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

# 全局速率限制器
class TokenBucketRateLimiter:
    def __init__(self, rpm_limit=500):
        self.rpm_limit = rpm_limit
        self.semaphore = asyncio.Semaphore(rpm_limit)
        self.tokens = rpm_limit
        self.last_refill = time.time()

    async def acquire(self):
        self._refill()
        if self.tokens <= 0:
            wait_time = (1 / self.rpm_limit) * 60
            await asyncio.sleep(wait_time + random.uniform(0, 1))
            self._refill()
        self.tokens -= 1

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.rpm_limit,
            self.tokens + elapsed * (self.rpm_limit / 60)
        )
        self.last_refill = now

# 成本监控和熔断
class CostCircuitBreaker:
    def __init__(self, daily_budget=200):
        self.daily_budget = daily_budget
        self.current_cost = 0

    async def check_and_record(self, api_call):
        if self.current_cost >= self.daily_budget:
            raise CircuitBreakerOpen(
                f"Daily budget ${self.daily_budget} exceeded"
            )
        result = await api_call()
        cost = self._calculate_cost(result)
        self.current_cost += cost

        # 预警阈值
        if self.current_cost > self.daily_budget * 0.8:
            await self.send_alert("API cost approaching daily budget")

        return result
```

### 预防措施

- 所有 API 调用必须有重试上限（推荐 2-3 次）
- 使用指数退避 + 随机抖动（Exponential Backoff with Jitter）
- 区分可重试和不可重试的错误类型
- 实现全局限流器，防止并发重试超过 API 限制
- 设置每日/每月 API 预算上限，超过后熔断
- 实时监控 API 成本，设置阶梯告警（50%、80%、100%）
- 分离在线流量和批量任务，使用不同的 API Key 和配额
- 定期审查 API 使用日志，识别异常模式

---

## 📋 排查方法论总结

### 通用排查框架

```
1. 确认现象 → 收集可观测数据（指标、日志、链路追踪）
2. 缩小范围 → 是全局还是局部？是突变还是渐变？
3. 定位根因 → 时间线分析、变更关联、对比实验
4. 修复验证 → 修复 → 灰度验证 → 全量
5. 复盘改进 → 根因分析文档 → 预防措施 → 监控补充
```

### 常用排查工具

| 场景 | 工具 |
|------|------|
| 日志分析 | ELK Stack, Loki, CloudWatch |
| 指标监控 | Prometheus + Grafana, Datadog |
| 链路追踪 | Jaeger, Zipkin, OpenTelemetry |
| 性能分析 | py-spy, nvidia-smi, torch.profiler |
| 向量数据库 | Milvus Dashboard, Pinecone Metrics |
| 消息队列 | RabbitMQ Management, Kafka UI |
| 成本分析 | OpenAI Dashboard, 自建 Token 计数器 |

### 关键原则

1. **可观测性优先**：没有监控就无法排查，先建设再优化
2. **变更关联**：90% 的故障与最近的变更相关，优先排查变更
3. **最小化复现**：找到最小复现路径才能快速定位根因
4. **防御性编程**：超时、重试、熔断、限流是生产环境的标配
5. **自动化恢复**：系统应能自动检测异常并触发恢复流程
