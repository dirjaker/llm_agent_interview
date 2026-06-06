# 26. Embedding 与向量数据库

> 本章深入讲解 Embedding 原理、向量数据库核心技术、索引算法和检索优化。Embedding 和向量数据库是 RAG 系统的基础设施。

---

## Embedding 基础

### Q: 什么是 Embedding？为什么需要向量化表示？⭐⭐

**答：** Embedding 是将离散的、高维的数据（文本、图片、音频）映射到连续的、低维的向量空间的过程。

**为什么需要 Embedding？**
- 文本是离散符号（"猫"和"猫咪"是完全不同的字符串）
- 向量可以表示**语义相似度**（"猫"和"猫咪"的向量很近）
- 向量支持**高效检索**（最近邻搜索，O(log N) 复杂度）

```
文本 → Embedding Model → [0.12, -0.34, 0.56, ..., 0.78] (1536维)
                              ↓
                         语义空间中的一个点
                         与相似文本距离近
```

### Q: 常见的 Embedding 模型有哪些？如何选型？⭐⭐⭐

**答：**

| 模型 | 维度 | 中文 | 价格 | 推荐场景 |
|------|:----:|:----:|------|----------|
| **text-embedding-3-small** | 1536 | ✅ | $0.02/M tokens | 通用，性价比最高 |
| **text-embedding-3-large** | 3072 | ✅ | $0.13/M tokens | 高精度需求 |
| **BGE-large-zh-v1.5** | 1024 | ✅✅ | 免费本地 | 中文最佳，本地部署 |
| **BGE-M3** | 1024 | ✅✅ | 免费本地 | 多语言+多粒度 |
| **Jina-embeddings-v3** | 1024 | ✅ | $0.02/M tokens | 长文本(8K) |
| **Cohere embed-v3** | 1024 | ✅ | $0.1/M tokens | 多语言最佳 |
| **GTE-Qwen2** | 1536 | ✅✅ | 免费本地 | 阿里开源，中文强 |

**选型决策树：**
```
需要本地部署？
  ├─ 是 → 中文为主？ → BGE-large-zh / GTE-Qwen2
  │       多语言？ → BGE-M3
  └─ 否 → 预算敏感？ → text-embedding-3-small
           精度优先？ → text-embedding-3-large / Cohere
           长文本？ → Jina-embeddings-v3
```

```python
# OpenAI Embedding API
import httpx

async def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": model, "input": text}
        )
        return resp.json()["data"][0]["embedding"]

# 本地 BGE 模型
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
embeddings = model.encode(["什么是RAG？", "检索增强生成是什么？"])
# 两句话的余弦相似度 > 0.9
```

### Q: 余弦相似度 vs 点积 vs 欧氏距离有什么区别？⭐⭐

**答：**

| 度量 | 公式 | 范围 | 特点 | 适用场景 |
|------|------|:----:|------|----------|
| **余弦相似度** | cos(θ) = A·B / (‖A‖·‖B‖) | [-1, 1] | 只看方向，不看长度 | 文本相似度（最常用） |
| **点积** | A·B = Σai·bi | [-∞, +∞] | 方向+长度 | 已归一化的向量 |
| **欧氏距离** | ‖A-B‖₂ | [0, +∞] | 空间直线距离 | 聚类、异常检测 |

```python
import numpy as np

def cosine_similarity(a, b):
    """余弦相似度：最常用，不受向量长度影响"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dot_product(a, b):
    """点积：向量已归一化时等价于余弦相似度"""
    return np.dot(a, b)

def euclidean_distance(a, b):
    """欧氏距离：越小越相似"""
    return np.linalg.norm(a - b)

# 文本 Embedding 通常已归一化，所以余弦相似度 ≈ 点积
a = get_embedding("什么是RAG？")
b = get_embedding("检索增强生成的原理")
print(f"余弦相似度: {cosine_similarity(a, b):.4f}")  # 0.85+
```

---

## 向量数据库

### Q: 向量数据库的核心原理是什么？⭐⭐⭐

**答：** 向量数据库的核心是 **近似最近邻搜索（ANN）**——在百万级向量中快速找到与查询向量最相似的 Top-K 个。

**为什么不直接暴力搜索？**
- 暴力搜索：O(N×D)，100 万 × 1536 维 = 15 亿次计算
- ANN 索引：O(log N × D)，100 万只需要几千次计算
- 代价：牺牲少量精度（召回率 95-99%）

**主流 ANN 索引算法：**

| 算法 | 原理 | 优点 | 缺点 | 代表 |
|------|------|------|------|------|
| **HNSW** | 分层可导航小世界图 | 精度高、查询快 | 内存大 | Milvus, Qdrant |
| **IVF** | 倒排索引 + 聚类 | 内存小 | 精度稍低 | Milvus, Faiss |
| **PQ** | 乘积量化压缩 | 超省内存 | 精度损失 | Faiss |
| **ScaNN** | Google 的各向异性量化 | 速度快 | 生态小 | Google |

### Q: HNSW 索引算法的原理是什么？⭐⭐⭐

**答：** HNSW（Hierarchical Navigable Small World）是目前最流行的 ANN 算法。

**核心思想：** 多层跳表 + 小世界图

```
Layer 3 (最稀疏):  A ←→ D           ← 全局导航，快速定位区域
                   ↓
Layer 2:           A ←→ B ←→ D      ← 中间层
                   ↓
Layer 1 (最密集):  A ←→ B ←→ C ←→ D ←→ E  ← 精确搜索
```

**搜索过程：**
1. 从最高层的入口点开始
2. 在当前层贪心搜索最近邻
3. 向下一层移动，继续搜索
4. 在最底层返回 Top-K 结果

**关键参数：**
- `M`：每个节点的最大邻居数（越大精度越高，内存越大）
- `ef_construction`：建索引时的搜索宽度（越大索引质量越好）
- `ef_search`：查询时的搜索宽度（越大精度越高，速度越慢）

```python
# Milvus HNSW 索引配置
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {
        "M": 16,              # 邻居数，推荐 16-64
        "efConstruction": 200  # 建索引搜索宽度，推荐 100-500
    }
}

# 查询时
search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 64}  # 查询搜索宽度，越大越精确
}
```

### Q: Milvus vs Pinecone vs Chroma vs Weaviate 如何选型？⭐⭐⭐

**答：**

| 特性 | Milvus | Pinecone | Chroma | Weaviate | Qdrant |
|------|--------|----------|--------|----------|--------|
| **部署** | 自托管/云 | 纯云 | 本地/嵌入 | 自托管/云 | 自托管/云 |
| **规模** | 十亿级 | 十亿级 | 百万级 | 十亿级 | 十亿级 |
| **索引** | HNSW/IVF/DiskANN | 专有 | HNSW | HNSW | HNSW |
| **混合检索** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **多租户** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **开源** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **学习曲线** | 中 | 低 | 低 | 中 | 低 |
| **推荐场景** | 大规模生产 | 快速上线 | 原型验证 | 全功能 | 轻量生产 |

**选型建议：**
- **原型开发** → Chroma（最简单，pip install）
- **中小规模生产** → Qdrant 或 Weaviate
- **大规模生产** → Milvus（分布式，十亿级）
- **不想运维** → Pinecone（全托管）

```python
# Chroma（最简单的向量数据库）
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")

collection.add(
    documents=["什么是RAG？", "什么是Agent？"],
    ids=["doc1", "doc2"]
)

results = collection.query(query_texts=["检索增强"], n_results=2)

# Milvus（生产级）
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

connections.connect("default", host="localhost", port="19530")

fields = [
    FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema("text", DataType.VARCHAR, max_length=65535),
]
schema = CollectionSchema(fields)
collection = Collection("documents", schema)
```

---

## 检索优化

### Q: 如何优化向量检索的精度和速度？⭐⭐⭐

**答：**

| 优化维度 | 方法 | 效果 |
|----------|------|------|
| **索引参数** | 增大 M/ef_search | 精度 ↑，速度 ↓ |
| **量化压缩** | PQ/SQ8 量化 | 内存 ↓ 75%，精度 ↓ 2-5% |
| **过滤检索** | 先过滤再搜索 | 减少搜索空间 |
| **Rerank** | 粗检索 + 精排序 | 精度 ↑↑，延迟 +50ms |
| **分片策略** | 按时间/类别分片 | 减少扫描范围 |
| **缓存** | 热点查询缓存 | 延迟 ↓ 90% |

```python
class OptimizedRetriever:
    """粗检索 + Rerank 两阶段方案"""

    def __init__(self):
        self.vector_db = MilvusClient()
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        # 阶段 1：向量粗检索（取 20 个候选）
        query_emb = get_embedding(query)
        candidates = self.vector_db.search(
            collection="docs",
            query_vector=query_emb,
            top_k=20,
            search_params={"ef": 128}
        )

        # 阶段 2：Cross-Encoder 精排序
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        # 合并排序
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)

        results = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_k]
```

### Q: 混合检索（向量+关键词）如何实现？⭐⭐⭐

**答：** 单独用向量检索会丢失精确匹配能力（比如搜"ERR-404"错误码），混合检索结合两者优势。

**RRF（Reciprocal Rank Fusion）融合公式：**
```
RRF_score(d) = Σ 1 / (k + rank_i(d))
其中 k=60（常数），rank_i 是第 i 个检索器的排名
```

```python
from collections import defaultdict

class HybridRetriever:
    """向量 + BM25 混合检索"""

    def __init__(self):
        self.vector_db = MilvusClient()
        self.bm25 = BM25Index()

    async def search(self, query: str, top_k: int = 5, k: int = 60) -> list[dict]:
        # 向量检索 Top-20
        query_emb = get_embedding(query)
        vector_results = self.vector_db.search("docs", query_emb, top_k=20)

        # BM25 关键词检索 Top-20
        bm25_results = self.bm25.search(query, top_k=20)

        # RRF 融合
        scores = defaultdict(float)
        for rank, doc in enumerate(vector_results):
            scores[doc["id"]] += 1 / (k + rank + 1)
        for rank, doc in enumerate(bm25_results):
            scores[doc["id"]] += 1 / (k + rank + 1)

        # 按融合分数排序
        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]

        # 返回结果
        all_docs = {d["id"]: d for d in vector_results + bm25_results}
        return [all_docs[doc_id] for doc_id in sorted_ids]
```

---

## Embedding 微调

### Q: Embedding 模型的微调方法？⭐⭐

**答：** 当通用 Embedding 在特定领域效果不好时（如医疗、法律），需要微调。

**微调方式：**

| 方式 | 数据需求 | 效果 | 复杂度 |
|------|----------|------|--------|
| **对比学习微调** | 正负样本对 | 最好 | 高 |
| **LoRA 微调** | 少量标注 | 好 | 中 |
| **提示工程** | 无 | 一般 | 低 |

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 加载预训练模型
model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

# 准备训练数据（正样本对）
train_examples = [
    InputExample(texts=["什么是RAG？", "检索增强生成的原理介绍"], label=0.9),
    InputExample(texts=["什么是RAG？", "今天天气怎么样"], label=0.1),
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# 微调
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="./bge-finetuned-medical"
)
```

---

## 面试速查

| 问题 | 关键点 |
|------|--------|
| Embedding 是什么？ | 离散数据→连续向量，支持语义相似度计算 |
| 余弦 vs 点积？ | 余弦看方向、点积看方向+长度，归一化后等价 |
| HNSW 原理？ | 分层可导航小世界图，贪心搜索逐层下降 |
| 向量数据库选型？ | 原型用 Chroma，生产用 Milvus/Qdrant，托管用 Pinecone |
| 混合检索？ | 向量 + BM25 + RRF 融合，兼顾语义和精确匹配 |
| 精度优化？ | Rerank（Cross-Encoder）、增大 ef_search、混合检索 |
| 成本优化？ | 量化压缩、分片、缓存、选小维度模型 |
