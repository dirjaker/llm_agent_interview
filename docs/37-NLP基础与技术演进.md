# 37. NLP 基础与技术演进

> 从 TF-IDF 到 Transformer 的完整演进路线，面试高频考点
> 每个知识点：问题 → 答案 → 追问，难度标记：⭐基础 ⭐⭐进阶 ⭐⭐⭐高级

---

## 一、文本表示的演进

### Q: 文本表示经历了哪些范式转变？⭐⭐

**答：**

文本表示的演进可以分为三个时代：

| 时代 | 方法 | 核心思想 | 局限性 |
|------|------|---------|--------|
| 离散表示 | One-Hot → Bag of Words → TF-IDF | 词是独立符号 | 无法捕获语义关系 |
| 分布式表示 | Word2Vec → GloVe → FastText | 语义相近的词向量相近 | 一词多义问题 |
| 上下文表示 | ELMo → BERT → GPT | 同一个词在不同语境有不同向量 | 计算成本高 |

**类比理解：**

想象你要描述一个人：
- **One-Hot**：给每个人一个身份证号（纯编号，不包含任何个人信息）
- **TF-IDF**：给每个人一个标签（"程序员""男性""北京"），但同一标签的人完全一样
- **Word2Vec**：根据社交关系给每个人一组特征值（相似社交圈的人特征值接近）
- **BERT**：同一个人在不同场合有不同的特征（工作中严肃的你 vs 朋友聚会中的你）

```python
# 三种范式的代码对比
import numpy as np

# === 1. One-Hot 编码 ===
vocab = ["我", "喜欢", "学习", "深度", "学习"]  # 注意"学习"出现两次
vocab_unique = ["我", "喜欢", "学习", "深度"]
def one_hot(word, vocab):
    vec = [0] * len(vocab)
    vec[vocab.index(word)] = 1
    return vec

print("One-Hot '喜欢':", one_hot("喜欢", vocab_unique))
# [0, 1, 0, 0] — 完全独立，无法知道"喜欢"和"热爱"相近

# === 2. TF-IDF ===
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ["我喜欢学习", "学习深度学习", "我喜欢深度学习"]
vectorizer = TfidfVectorizer()
tfidf_matrix = fit_transform = vectorizer.fit_transform(corpus)
print("TF-IDF 特征:", vectorizer.get_feature_names_out())
# 仍然是稀疏高维向量，词与词之间无语义关联

# === 3. Word2Vec ===
from gensim.models import Word2Vec
sentences = [["我", "喜欢", "学习"], ["深度", "学习", "很", "有趣"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
print("Word2Vec '学习':", model.wv["学习"].shape)  # (100,) — 密集向量
print("相似词:", model.wv.most_similar("学习", topn=3))
# 语义相近的词向量距离近："学习" ≈ "研究"，"喜欢" ≈ "热爱"
```

**追问：为什么离散表示不行？**

离散表示的三大致命缺陷：
1. **维度灾难**：词表 10 万 → 每个词是 10 万维稀疏向量
2. **无语义关系**："喜欢"和"热爱"的距离 = "喜欢"和"桌子"的距离
3. **词袋假设**：忽略词序，"狗咬人"和"人咬狗"表示相同

---

### Q: TF-IDF 的数学推导与直觉？⭐⭐

**答：**

**TF（词频）**：一个词在当前文档中出现的频率

$$TF(t, d) = \frac{词t在文档d中出现的次数}{文档d的总词数}$$

**IDF（逆文档频率）**：一个词在所有文档中出现的频率越低，区分度越高

$$IDF(t) = log\frac{文档总数}{包含词t的文档数 + 1}$$

**TF-IDF** = TF × IDF

**直觉理解：**

- "的""是""在"在每个文档都高频出现 → IDF 低 → TF-IDF 低
- "Transformer"只在技术文档中出现 → IDF 高 → TF-IDF 高
- TF-IDF 的本质：**一个词越在当前文档重要，越在其他文档罕见，它就越重要**

```python
import math
from collections import Counter

def compute_tfidf(corpus):
    """手动实现 TF-IDF"""
    # corpus: list of documents, each document is a list of words
    
    # 1. 计算每个文档的 TF
    tf_list = []
    for doc in corpus:
        word_count = Counter(doc)
        total = len(doc)
        tf = {word: count / total for word, count in word_count.items()}
        tf_list.append(tf)
    
    # 2. 计算全局 IDF
    doc_count = len(corpus)
    df = Counter()  # document frequency
    for doc in corpus:
        for word in set(doc):  # set: 每个词在文档中只计一次
            df[word] += 1
    
    idf = {word: math.log(doc_count / (freq + 1)) for word, freq in df.items()}
    
    # 3. TF-IDF = TF × IDF
    tfidf_list = []
    for tf in tf_list:
        tfidf = {word: tf_val * idf.get(word, 0) for word, tf_val in tf.items()}
        tfidf_list.append(tfidf)
    
    return tfidf_list

# 示例
corpus = [
    ["机器", "学习", "是", "人工智能", "的", "分支"],
    ["深度", "学习", "是", "机器", "学习", "的", "子领域"],
    ["自然语言", "处理", "使用", "深度", "学习", "技术"],
]

results = compute_tfidf(corpus)
for i, tfidf in enumerate(results):
    sorted_words = sorted(tfidf.items(), key=lambda x: -x[1])[:5]
    print(f"文档{i+1} 关键词: {sorted_words}")
```

**追问：TF-IDF 有什么局限性？**

1. **忽略词序**："不好"和"好"的 TF-IDF 只看词频
2. **无法处理同义词**："电脑"和"计算机"是两个独立特征
3. **稀疏向量**：词表大时维度极高，计算浪费
4. **无上下文**："苹果"在所有文档中是同一个向量

**面试技巧：** TF-IDF 虽然"古老"，但在**关键词提取、搜索引擎、作为 baseline** 时仍然非常实用。面试中能说出 TF-IDF 的数学公式 + 局限性 + 适用场景即可。

---

### Q: Bag of Words 和 N-gram 模型有什么区别？⭐

**答：**

| 方法 | 思想 | 维度 | 能否捕获词序 |
|------|------|------|------------|
| Bag of Words | 只看词频，不管顺序 | \|V\| | ❌ |
| N-gram | 看连续 N 个词的组合 | \|V\|^N | ✅（局部） |

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["I love NLP", "NLP loves me"]

# Unigram (BoW)
uni = CountVectorizer(ngram_range=(1, 1))
print("Unigram:", uni.fit_transform(corpus).toarray())
# 特征: [I, love, me, nlp, loves] — 无词序

# Bigram
bi = CountVectorizer(ngram_range=(2, 2))
print("Bigram:", bi.fit_transform(corpus).toarray())
# 特征: [i love, love nlp, nlp loves, loves me] — 捕获局部词序
```

**N-gram 的问题：** N 越大 → 维度指数增长（|V|^N），数据稀疏严重。实践中通常只用 bigram 或 trigram。

---

## 二、词嵌入（Word Embedding）

### Q: Word2Vec 的两种训练方式有什么区别？⭐⭐

**答：**

Word2Vec 有两种架构：

**CBOW（Continuous Bag of Words）**：用上下文预测中心词

```
输入: [The, cat, on, the, ___]  
目标: 预测 "mat"
```

**Skip-gram**：用中心词预测上下文

```
输入: [cat]  
目标: 预测 [The, mat, on, the]
```

| 对比 | CBOW | Skip-gram |
|------|------|-----------|
| 训练速度 | 快（一次前向预测一个词） | 慢（预测多个上下文词） |
| 高频词 | 效果好 | 效果稍差 |
| 低频词/罕见词 | 效果一般 | 效果好 |
| 数据量小 | ✅ 更好 | ❌ 需要更多数据 |
| 实践选择 | 小数据集、高频词场景 | 大数据集、需要处理罕见词 |

```python
from gensim.models import Word2Vec

# CBOW: sg=0 (default)
model_cbow = Word2Vec(sentences, vector_size=100, window=5, sg=0)

# Skip-gram: sg=1
model_sg = Word2Vec(sentences, vector_size=100, window=5, sg=1)

# 经典类比测试
# king - man + woman ≈ queen
result = model_sg.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
print(result)  # [("queen", 0.9)]
```

**追问：Word2Vec 为什么能捕获 "king - man + woman = queen" 这种语义关系？**

Word2Vec 的训练目标是"预测上下文"。出现在相似上下文中的词，向量会趋近。"king"和"queen"经常出现在类似的上下文中（"The ___ ruled the kingdom"），所以它们的向量在空间中的**偏移方向**可以编码语义关系（如性别、时态、国家-首都）。

数学上，Word2Vec 学到的向量空间具有**线性结构**——语义关系可以表示为向量的加减。

---

### Q: 负采样（Negative Sampling）为什么必要？⭐⭐⭐

**答：**

**问题：** Word2Vec 的原始训练目标是 Softmax：

$$P(w_o | w_i) = \frac{exp(v_{w_o} \cdot v_{w_i})}{\sum_{w=1}^{|V|} exp(v_w \cdot v_{w_i})}$$

分母需要对**整个词表**（10万+词）求和，计算量巨大。

**负采样的解决思路：** 不用完整的 Softmax，而是把问题转化为**二分类**：

- 正样本：(中心词, 真实上下文词) → 标签 1
- 负样本：(中心词, 随机采样的非上下文词) → 标签 0

$$log \sigma(v_{w_o} \cdot v_{w_i}) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)} [log \sigma(-v_{w_k} \cdot v_{w_i})]$$

其中 K 是负样本数量（通常 5-20），$P_n(w) \propto f(w)^{3/4}$（按词频的 3/4 次方采样）。

```python
# 负采样的直观理解
# 原始: 预测 "cat" 的上下文 → Softmax over 100,000 词 → 太慢
# 负采样: 
#   正样本: ("cat", "mat") → 1    (真实的上下文)
#   负样本: ("cat", "democracy") → 0  (随机采的非上下文)
#   负样本: ("cat", "pizza") → 0
#   只需计算 1+K=6 次 sigmoid，而不是 100,000 次！

# Gensim 自动使用负采样
model = Word2Vec(
    sentences,
    vector_size=100,
    negative=5,        # 5个负样本
    ns_exponent=0.75,  # 采样分布指数
)
```

**为什么采样分布用 $f(w)^{3/4}$？**
- 纯按词频 $f(w)$：高频词（"的""是"）被过度采样为负样本
- 均匀采样：低频词（"Transformer"）被选为负样本的概率太低，学不好
- $f(w)^{3/4}$ 是一个折中：压低高频词权重，抬高低频词权重

---

### Q: GloVe 和 Word2Vec 的本质区别？⭐⭐

**答：**

| 维度 | Word2Vec | GloVe |
|------|----------|-------|
| 训练方式 | 预测式（Predictive） | 计数式（Count-based） |
| 核心思想 | 预测上下文 | 拟合共现矩阵 |
| 输入 | 局部上下文窗口 | 全局词共现统计 |
| 数学目标 | Skip-gram/CBOW 目标函数 | 加权最小二乘拟合共现概率比 |

**GloVe 的核心洞察：**

词向量应该编码词与词之间的共现概率比：

$$\frac{P(k|ice)}{P(k|steam)} = \begin{cases} \gg 1 & \text{if k = solid (与ice相关)} \\ \approx 1 & \text{if k = water (两者都相关)} \\ \ll 1 & \text{if k = gas (与steam相关)} \end{cases}$$

GloVe 的损失函数：

$$J = \sum_{i,j=1}^{|V|} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - log X_{ij})^2$$

其中 $X_{ij}$ 是词 i 和词 j 的共现次数，$f(x)$ 是权重函数（高频共现对权重有上限）。

**实践中选哪个？** 效果差异不大。Word2Vec 更常用（Gensim 实现成熟），GloVe 提供了预训练向量（如 glove-twitter-200）。

---

### Q: FastText 的子词嵌入解决了什么问题？⭐⭐

**答：**

**Word2Vec 的问题：** 每个词是一个独立的向量。对于训练时没见过的词（OOV，Out-of-Vocabulary），无法给出向量。

**FastText 的创新：** 把每个词拆成字符 n-gram，词向量 = 子词向量之和。

```
"where" 的 3-gram: <wh, whe, her, ere, re>
词向量 "where" = v(<wh) + v(whe) + v(her) + v(ere) + v(re>) + v(where)
```

**优势：**
1. **处理 OOV**：即使 "wherever" 没见过，它的子词（<wh, whe, her...）可能已经学过
2. **形态学知识**：英语中 "-ing"、"-ed"、"-tion" 等后缀有语义，子词嵌入能捕获
3. **多语言**：对中文等没有天然分词的语言也有效

```python
from gensim.models import FastText

model = FastText(sentences, vector_size=100, window=5, min_n=3, max_n=6)

# OOV 词也能获得向量！
vector = model.wv["unseenword"]  # 通过子词组合得到
```

---

## 三、语言模型基础

### Q: 什么是困惑度（Perplexity）？⭐⭐

**答：**

困惑度是衡量语言模型好坏的核心指标。

**直觉定义：** 模型在预测下一个词时平均有多少个等概率的候选词。

$$PP(W) = P(w_1 w_2 ... w_N)^{-\frac{1}{N}} = exp\left(-\frac{1}{N}\sum_{i=1}^{N} log P(w_i | w_1...w_{i-1})\right)$$

**类比理解：**
- 困惑度 = 1：模型 100% 确定下一个词（完美）
- 困惑度 = 10：模型平均在 10 个词之间犹豫
- 困惑度 = 1000：模型完全不知道下一个词是什么

| 模型 | 困惑度（Penn Treebank） |
|------|----------------------|
| 3-gram | ~150 |
| LSTM | ~80 |
| Transformer | ~60 |
| GPT-3 | ~20 |

```python
import torch
import torch.nn.functional as F

def perplexity(logits, targets):
    """计算困惑度
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len)
    """
    # 交叉熵损失
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction='mean'
    )
    # 困惑度 = exp(交叉熵)
    return torch.exp(loss).item()

# 交叉熵越低 → 困惑度越低 → 模型越好
```

**追问：困惑度和交叉熵的关系？**

$$PP = exp(H) = exp\left(-\frac{1}{N}\sum log P(w_i|context)\right)$$

交叉熵 $H$ 是困惑度的对数。优化交叉熵 = 优化困惑度。

---

### Q: CLM vs MLM 有什么区别？为什么各自的选择？⭐⭐⭐

**答：**

| 维度 | CLM（因果语言模型） | MLM（掩码语言模型） |
|------|-------------------|-------------------|
| 代表模型 | GPT 系列 | BERT 系列 |
| 训练方式 | 预测下一个词 | 预测被掩码的词 |
| 上下文 | 只看左边（单向） | 看左右两边（双向） |
| 训练效率 | 每个 token 都参与预测 | 只有 15% 被掩码的 token 参与 |
| 生成能力 | ✅ 天然适合 | ❌ 不适合生成 |
| 理解能力 | 一般 | ✅ 更强 |

**CLM（GPT 的选择）：**

```
输入: "今天天气"
训练目标: P("真" | "今天天气")
         P("好" | "今天天气真")
         P("</s>" | "今天天气真好")
```

每个 token 都是一个训练样本 → 训练效率高 → 适合大规模预训练 → 天然支持文本生成。

**MLM（BERT 的选择）：**

```
输入: "今天 [MASK] 气真好"
训练目标: P("天" | "今天 [MASK] 气真好")  ← 同时看左右上下文
```

能看到双向上下文 → 对语义理解更强 → 但不适合生成（推理时没有 [MASK]）。

**追问：为什么生成任务几乎都用 CLM？**

因为生成任务的本质就是**从左到右逐词预测**。CLM 的训练目标和推理过程完全一致。MLM 在推理时需要特殊处理（如 BERT 的 [MASK]），无法自然地生成流畅文本。

---

## 四、Seq2Seq 与 Attention

### Q: Attention 机制是怎么发明的？解决了什么问题？⭐⭐

**答：**

**问题背景（Seq2Seq 的瓶颈）：**

传统的 Encoder-Decoder 架构将整个输入句子压缩成一个**固定长度的向量**（context vector），然后用它生成输出。

```
输入: "我今天去了北京大学图书馆看书"  →  Encoder  → [0.3, -0.1, ...]（一个固定向量）
输出: "I went to the library at Peking University to read"  ←  Decoder  ← 这一个向量
```

**问题：** 句子越长，信息压缩越严重。长句子的后半部分信息几乎丢失。

**Attention 的核心思想（2014, Bahdanau）：**

不压缩成一个向量，而是让 Decoder 在每一步**"回头看"** Encoder 的所有隐状态，自动找到最相关的部分。

```python
import torch
import torch.nn.functional as F

def attention(query, key, value):
    """
    query: Decoder 当前状态 (batch, d_model)
    key: Encoder 所有隐状态 (batch, seq_len, d_model)
    value: Encoder 所有隐状态 (batch, seq_len, d_model)
    """
    # 1. 计算注意力分数
    scores = torch.bmm(key, query.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)
    
    # 2. Softmax 归一化
    attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
    
    # 3. 加权求和
    context = torch.bmm(attn_weights.unsqueeze(1), value).squeeze(1)  # (batch, d_model)
    
    return context, attn_weights

# 生成 "library" 时，attention 权重集中在 "图书馆" 上
# 生成 "Peking University" 时，attention 权重集中在 "北京大学" 上
```

**两种 Attention 的区别：**

| 类型 | 公式 | 特点 |
|------|------|------|
| Bahdanau (加性) | $score = v^T tanh(W_1 h_{enc} + W_2 h_{dec})$ | 引入可学习参数 |
| Luong (乘性) | $score = h_{enc}^T W h_{dec}$ 或直接点积 | 计算更高效 |

---

### Q: Q/K/V 的直觉理解是什么？⭐⭐⭐

**答：**

Attention 的核心公式：

$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}}) V$$

**图书馆类比：**

- **Q（Query，查询）**：你要找什么？你的问题（"这本书讲深度学习吗？"）
- **K（Key，键）**：每本书的标题/目录（快速判断相关性）
- **V（Value，值）**：每本书的实际内容（提取有用信息）

**过程：**
1. 你的 Query 和每本书的 Key 做匹配 → 得到相关性分数
2. 分数归一化（Softmax）→ 得到注意力权重
3. 用权重对 Value 加权求和 → 得到你需要的信息

```python
# 简单示例
import torch

# 3个词的序列，d_model=4
Q = torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # 查询：想找什么
K = torch.tensor([[1.0, 0.0, 0.0, 0.0],   # key1
                   [0.0, 1.0, 0.0, 0.0],   # key2
                   [1.0, 0.0, 1.0, 0.0]])  # key3 — 和Q最匹配！
V = torch.tensor([[0.1, 0.2, 0.3, 0.4],   # value1
                   [0.5, 0.6, 0.7, 0.8],   # value2
                   [0.9, 1.0, 1.1, 1.2]])  # value3

# Q 和 K 的点积 → 注意力分数
scores = torch.mm(Q, K.T)  # [[1.0, 0.0, 2.0]]
# key3 得分最高！

weights = torch.softmax(scores, dim=-1)  # [[0.12, 0.02, 0.86]]
# 86% 的注意力在 key3 上

output = torch.mm(weights, V)  # [[0.79, 0.88, 0.97, 1.06]]
# 输出主要是 value3 的内容
```

**追问：为什么除以 $\sqrt{d_k}$？**

当 $d_k$ 很大时，$QK^T$ 的点积结果方差会很大 → Softmax 输出接近 one-hot → 梯度消失。除以 $\sqrt{d_k}$ 使方差保持在合理范围内，让 Softmax 的梯度更平滑。

---

## 五、Transformer 架构详解

### Q: Transformer 的完整架构包含哪些组件？⭐⭐⭐

**答：**

```
┌─────────────────────────────────────────┐
│              Transformer                │
│                                         │
│  ┌─────────────┐    ┌─────────────┐     │
│  │   Encoder    │    │   Decoder    │    │
│  │  (N layers)  │    │  (N layers)  │    │
│  │              │    │              │    │
│  │ Self-Attn    │    │ Masked       │    │
│  │ + FFN        │───▶│ Self-Attn    │    │
│  │ + LayerNorm  │    │ Cross-Attn   │    │
│  │ + Residual   │    │ + FFN        │    │
│  └─────────────┘    │ + LayerNorm  │    │
│        ↑             │ + Residual   │    │
│   Input Embedding    └─────────────┘    │
│   + Positional            ↑             │
│   Encoding          Output Embedding    │
│                     + Positional         │
│                     Encoding             │
└─────────────────────────────────────────┘
```

**每一层 Encoder 的计算流程：**

```python
import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_ff=2048, dropout=0.1):
        super().__init__()
        # 1. Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 2. Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),  # 或 GELU
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        # 3. LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src_mask=None):
        # Pre-Norm 方式（现代 LLM 的选择）
        # Step 1: Self-Attention + Residual
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, attn_mask=src_mask)
        x = x + self.dropout(attn_out)  # 残差连接
        
        # Step 2: FFN + Residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)  # 残差连接
        
        return x
```

**关键组件解析：**

| 组件 | 作用 |
|------|------|
| Multi-Head Attention | 让模型从不同角度关注不同位置 |
| FFN | 非线性变换，增加模型容量（2层MLP） |
| LayerNorm | 稳定训练，加速收敛 |
| Residual Connection | 缓解梯度消失，支持训练深层网络 |
| Positional Encoding | 注入位置信息（Attention 本身无位置感知） |

---

### Q: Multi-Head Attention 为什么要多头？⭐⭐⭐

**答：**

**单头 Attention 的问题：** 只能关注一种模式。

**Multi-Head 的思想：** 每个头独立学习不同的关注模式，然后拼接。

```python
# Multi-Head Attention 的计算
d_model = 512
nhead = 8
d_k = d_model // nhead  # 64

# 输入
Q = torch.randn(batch, seq_len, d_model)  # (batch, 10, 512)

# 1. 线性投影到每个头
W_q = torch.randn(d_model, d_model)  # (512, 512)
Q_proj = Q @ W_q  # (batch, 10, 512)

# 2. 拆分成 8 个头
Q_heads = Q_proj.view(batch, seq_len, nhead, d_k).transpose(1, 2)
# (batch, 8, 10, 64) — 每个头独立计算

# 3. 每个头独立做 Attention
attn_outputs = []
for i in range(nhead):
    q_i = Q_heads[:, i]  # (batch, 10, 64)
    # Attention(Q_i, K_i, V_i) — 每个头学到不同的关注模式
    attn_i = scaled_dot_product_attention(q_i, k_i, v_i)
    attn_outputs.append(attn_i)

# 4. 拼接 + 线性投影
concat = torch.cat(attn_outputs, dim=-1)  # (batch, 10, 512)
output = concat @ W_o  # (batch, 10, 512)
```

**不同头关注什么？**（来自实际训练的 Transformer 可视化）

| 头 | 关注模式 |
|----|---------|
| Head 1 | 关注前一个词（局部语法） |
| Head 2 | 关注句子主语（长距离依赖） |
| Head 3 | 关注标点（句子边界） |
| Head 4 | 关注同义词（语义关联） |
| ... | 各种语言学模式 |

**追问：为什么 d_k = d_model / nhead？**

保持总计算量不变。单头 Attention 的计算量 = $O(n^2 \cdot d_{model})$，多头分成 h 个头后每个头 $d_k = d_{model}/h$，总计算量仍然是 $O(n^2 \cdot d_{model})$。多头不增加计算量，但增加了表达能力。

---

### Q: 位置编码有哪些方案？各自的优缺点？⭐⭐⭐

**答：**

Transformer 的 Self-Attention 是**置换不变**的——不关心词序。位置编码注入位置信息。

| 方案 | 模型 | 原理 | 优势 | 局限 |
|------|------|------|------|------|
| 正弦位置编码 | 原始 Transformer | $PE_{(pos,2i)} = sin(pos/10000^{2i/d})$ | 简单、可外推 | 固定不可学习 |
| 可学习位置编码 | BERT、GPT-2 | 每个位置一个可学习向量 | 灵活 | 不能外推到训练长度之外 |
| RoPE | LLaMA、Qwen | 旋转矩阵编码相对位置 | 相对位置、支持外推 | 实现稍复杂 |
| ALiBi | BLOOM | 注意力分数加位置偏置 | 无需额外参数、可外推 | 效果略逊于 RoPE |

**RoPE 的核心思想：**

把词向量分成若干对二维向量，每对做一个旋转。旋转角度与位置成正比。

$$RoPE(x, pos) = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{pmatrix} \otimes \begin{pmatrix} cos(pos \cdot \theta_1) \\ cos(pos \cdot \theta_1) \\ cos(pos \cdot \theta_2) \\ cos(pos \cdot \theta_2) \end{pmatrix} + \begin{pmatrix} -x_2 \\ x_1 \\ -x_4 \\ x_3 \end{pmatrix} \otimes \begin{pmatrix} sin(pos \cdot \theta_1) \\ sin(pos \cdot \theta_1) \\ sin(pos \cdot \theta_2) \\ sin(pos \cdot \theta_2) \end{pmatrix}$$

关键性质：$RoPE(x_m, m)^T RoPE(x_n, n) = f(x_m, x_n, m-n)$

两个位置的点积**只与相对距离 (m-n) 有关**，与绝对位置无关。这是 RoPE 能支持长度外推的根本原因。

```python
import torch

def precompute_rope_freqs(dim, max_seq_len, theta=10000.0):
    """预计算 RoPE 频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim/2)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    """应用 RoPE 旋转"""
    # x: (batch, seq_len, dim)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    
    # 旋转
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    
    return torch.cat([out1, out2], dim=-1)
```

---

## 六、预训练模型时代

### Q: BERT 和 GPT 的核心区别？⭐⭐⭐

**答：**

| 维度 | BERT | GPT |
|------|------|-----|
| 架构 | Transformer Encoder | Transformer Decoder |
| 注意力 | 双向（看左右） | 单向（只看左） |
| 训练任务 | MLM + NSP | CLM（预测下一个词） |
| 适用场景 | 理解任务（分类、NER、QA） | 生成任务（对话、写作、代码） |
| 微调方式 | 加任务头微调 | Prompt / In-Context Learning |
| 代表应用 | 搜索引擎、情感分析、NER | ChatGPT、代码生成、创意写作 |

**BERT 的训练任务：**

```python
# MLM（Masked Language Model）
input:  "今天 [MASK] 气真好"    → 预测 "天"
input:  "今天天 [MASK] 真好"    → 预测 "气"
# 每次随机掩码 15% 的 token

# NSP（Next Sentence Prediction）
input: "[CLS] 今天天气好 [SEP] 我去公园散步" → IsNext = True
input: "[CLS] 今天天气好 [SEP] 原子由质子组成" → IsNext = False
```

**GPT 的训练：**

```python
# CLM（Causal Language Model）
input:  "今天天气" → 预测 "真"
input:  "今天天气真" → 预测 "好"
input:  "今天天气真好" → 预测 "!"
# 每个 token 都是训练样本，效率更高
```

**追问：2024-2025 的趋势是哪个方向胜出？**

GPT 路线（自回归生成）胜出。原因：
1. **规模定律**：CLM 训练效率更高，更容易 scaling
2. **通用性**：生成模型可以通过 in-context learning 做理解任务
3. **涌现能力**：足够大的 CLM 展现出 CoT 推理、代码生成等能力

BERT 路线没有消失，而是在**嵌入模型**（如 BGE、E5）和**重排序**场景中继续发挥作用。

---

## 七、分词算法

### Q: BPE（Byte Pair Encoding）算法的完整流程？⭐⭐⭐

**答：**

BPE 是目前最主流的子词分词算法（GPT、LLaMA、Qwen 都使用）。

**算法流程：**

```
初始: 每个字符是一个 token
迭代: 找到最频繁的相邻 token 对 → 合并为新 token
重复: 直到达到目标词表大小
```

**完整示例：**

```
训练语料: "low low low low low lowest lowest newer newer newer wider"

Step 0: 字符级
  l o w → [l, o, w]
  lowest → [l, o, w, e, s, t]
  
字符频率: {l:10, o:9, w:10, e:8, s:4, t:5, n:6, r:6, i:2, d:2}

Step 1: 最频繁对 = (l, o) 出现 7 次 → 合并为 "lo"
  [l, o, w, e, s, t] → [lo, w, e, s, t]

Step 2: 最频繁对 = (lo, w) 出现 7 次 → 合并为 "low"
  [lo, w, e, s, t] → [low, e, s, t]

Step 3: 最频繁对 = (e, r) 出现 6 次 → 合并为 "er"
  ... 继续合并 ...

最终词表: [l, o, w, e, s, t, n, r, i, d, lo, low, er, new, est, ...]
```

```python
# 手动实现 BPE
def learn_bpe(corpus, num_merges):
    """学习 BPE 合并规则"""
    # 初始化：字符级
    vocab = {}
    for word in corpus:
        chars = list(word) + ["</w>"]
        key = " ".join(chars)
        vocab[key] = vocab.get(key, 0) + 1
    
    merges = []
    for i in range(num_merges):
        # 统计相邻对频率
        pairs = {}
        for word, freq in vocab.items():
            symbols = word.split()
            for j in range(len(symbols) - 1):
                pair = (symbols[j], symbols[j+1])
                pairs[pair] = pairs.get(pair, 0) + freq
        
        if not pairs:
            break
        
        # 找最频繁的对
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        
        # 合并
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = word.replace(" ".join(best_pair), "".join(best_pair))
            new_vocab[new_word] = freq
        vocab = new_vocab
    
    return merges, vocab

# 示例
corpus = ["low"] * 5 + ["lowest"] * 2 + ["newer"] * 3 + ["wider"] * 2
merges, final_vocab = learn_bpe(corpus, num_merges=10)
print("合并规则:", merges)
```

**追问：BPE vs WordPiece vs Unigram 有什么区别？**

| 算法 | 使用模型 | 方向 | 核心区别 |
|------|---------|------|---------|
| BPE | GPT、LLaMA | 自底向上 | 合并最频繁对 |
| WordPiece | BERT | 自底向上 | 合并使语言模型概率最大的对 |
| Unigram | T5、ALBERT | 自顶向下 | 从大词表开始，逐步删除低贡献 token |
| SentencePiece | 多语言模型 | 框架 | 不依赖预分词，直接处理原始文本 |

---

## 八、文本生成的解码策略

### Q: Greedy、Beam Search、Top-k、Top-p 的区别？⭐⭐⭐

**答：**

```python
import torch
import torch.nn.functional as F

def greedy_decode(logits):
    """贪心解码：每步选概率最大的词"""
    return torch.argmax(logits, dim=-1)

def beam_search(logits_seq, beam_size=5):
    """束搜索：维护 beam_size 个候选序列"""
    # 每步保留得分最高的 beam_size 个序列
    # 最终选总得分最高的
    pass  # 实现较复杂，略

def top_k_sampling(logits, k=50):
    """Top-k 采样：只在概率最高的 k 个词中采样"""
    # 1. 取 top-k
    top_k_logits, top_k_indices = torch.topk(logits, k)
    # 2. 其余设为 -inf
    logits_filtered = torch.full_like(logits, float('-inf'))
    logits_filtered.scatter_(1, top_k_indices, top_k_logits)
    # 3. 采样
    probs = F.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, 1)

def top_p_sampling(logits, p=0.9):
    """Nucleus 采样：选择累积概率达到 p 的最小词集"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 找到累积概率超过 p 的位置
    cutoff_idx = torch.searchsorted(cumulative_probs, p)
    
    # 将超出 p 的词概率设为 0
    sorted_logits[cutoff_idx + 1:] = float('-inf')
    
    # 恢复原始顺序
    logits_filtered = sorted_logits.scatter(1, sorted_indices.argsort(1), sorted_logits)
    probs = F.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, 1)

# Temperature 的作用
def apply_temperature(logits, temperature=1.0):
    """温度控制随机性"""
    # temperature < 1: 更确定（趋向贪心）
    # temperature = 1: 原始分布
    # temperature > 1: 更随机（趋向均匀）
    return logits / temperature
```

| 策略 | 特点 | 适用场景 |
|------|------|---------|
| Greedy | 确定性、最快 | 翻译、摘要等确定性任务 |
| Beam Search | 比贪心好、但可能重复 | 翻译、短文本生成 |
| Top-k | 固定候选集大小 | 通用生成 |
| Top-p | 动态候选集（概率低时自动缩小） | 开放式对话、创意写作 |
| Temperature | 控制随机性 | 配合 Top-k/Top-p 使用 |

**追问：Top-k 和 Top-p 哪个更好？**

Top-p 通常更好：
- Top-k 的问题：当模型很确定时（概率集中在 1-2 个词），k=50 强制引入噪声
- Top-p 的优势：当模型确定时，候选集自动缩小到几个词；当模型不确定时，候选集自动扩大

**实际组合使用：** `Top-p (0.9) + Temperature (0.7)` 是对话系统的常见配置。

---

## 九、NLP 评估方法

### Q: BLEU 分数的原理和局限性？⭐⭐

**答：**

**BLEU（Bilingual Evaluation Understudy）** 是机器翻译的标准评估指标。

**核心思想：** 计算机器翻译结果与人工参考译文的 n-gram 重合度。

$$BLEU = BP \cdot exp\left(\sum_{n=1}^{4} w_n \cdot log(p_n)\right)$$

其中：
- $p_n$ = n-gram 精确率（候选译文的 n-gram 在参考译文中出现的比例）
- $BP$ = 简短惩罚（防止翻译过短）
- $w_n = 1/4$（通常 1-gram 到 4-gram 等权）

```python
from collections import Counter

def compute_bleu(candidate, references, max_n=4):
    """简化版 BLEU 计算"""
    
    def get_ngrams(text, n):
        words = text.split()
        return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
    
    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(candidate, n)
        # 取所有参考译文中每个 n-gram 的最大出现次数
        max_ref = Counter()
        for ref in references:
            ref_ngrams = get_ngrams(ref, n)
            for ng, count in ref_ngrams.items():
                max_ref[ng] = max(max_ref[ng], count)
        
        # 裁剪计数
        clipped = sum(min(count, max_ref[ng]) for ng, count in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        precisions.append(clipped / total if total > 0 else 0)
    
    # 几何平均
    import math
    log_avg = sum(0.25 * math.log(max(p, 1e-10)) for p in precisions)
    
    # 简短惩罚
    bp = min(1, math.exp(1 - len(references[0].split()) / len(candidate.split())))
    
    return bp * math.exp(log_avg)

# 示例
candidate = "the cat is on the mat"
reference = ["there is a cat on the mat"]
print(f"BLEU: {compute_bleu(candidate, reference):.3f}")
```

**BLEU 的局限性：**

1. **不考虑语义**："好天气" vs "天气好" BLEU 很低但语义正确
2. **依赖参考译文**：没有参考就无法评估
3. **n-gram 匹配太严格**：同义词替换会扣分
4. **不适合开放式生成**：对话、创意写作没有标准答案

**追问：还有什么替代指标？**

| 指标 | 适用场景 | 改进点 |
|------|---------|--------|
| ROUGE | 文本摘要 | 召回率导向（BLEU 是精确率导向） |
| BERTScore | 通用 | 用 BERT 嵌入计算语义相似度 |
| METEOR | 翻译 | 考虑同义词和词干匹配 |
| 人工评估 | 所有场景 | 最可靠但成本高 |

---

## 十、NLP 基础面试高频题汇总

### 高频题快速参考

**Q1: 为什么 Transformer 能并行而 RNN 不能？⭐⭐**
RNN 的每个时间步依赖前一步的输出（串行）。Transformer 的 Self-Attention 对所有位置同时计算（并行）。训练时 Transformer 可以一次性处理整个序列。

**Q2: Attention 的时间复杂度？⭐⭐**
$O(n^2 \cdot d)$，其中 n 是序列长度，d 是维度。n² 是因为每个位置都要和其他所有位置计算注意力分数。

**Q3: 为什么 BERT 用 [CLS] 做分类？⭐⭐**
[CLS] 是 BERT 在序列开头添加的特殊 token。经过多层 Self-Attention 后，[CLS] 的隐状态聚合了整个序列的信息，适合作为分类特征。

**Q4: Word2Vec 和 BERT 的词向量有什么区别？⭐⭐**
Word2Vec：每个词一个固定向量（静态嵌入）。BERT：同一个词在不同上下文中向量不同（上下文嵌入）。"苹果好吃"和"苹果发布"中的"苹果"向量不同。

**Q5: 为什么 Softmax 用在 Attention 中？⭐⭐**
将注意力分数归一化为概率分布（和为1）。可以用其他函数替代（如 ReLU Attention），但 Softmax 最常用因为输出是概率分布、可微分。

**Q6: Transformer 中 FFN 的作用？⭐⭐⭐**
FFN 是两层 MLP（$d_{model} \to d_{ff} \to d_{model}$），提供非线性变换。研究表明 FFN 存储了事实知识（key-value memory 视角），Attention 负责路由，FFN 负责存储和检索知识。

**Q7: 为什么 LLM 使用 RMSNorm 而不是 LayerNorm？⭐⭐**
RMSNorm 去掉了均值中心化，只做方差归一化：$RMSNorm(x) = \frac{x}{\sqrt{mean(x^2) + \epsilon}} \cdot \gamma$。计算量减少约 10%，实验表明效果几乎不变。

**Q8: KV Cache 是什么？为什么重要？⭐⭐⭐**
自回归生成时，每生成一个 token 都要计算所有之前 token 的 K 和 V。KV Cache 缓存已计算的 K/V，避免重复计算。将生成复杂度从 $O(n^2)$ 降到 $O(n)$。

**Q9: BPE 分词对中文有什么特殊处理？⭐⭐**
中文没有天然的空格分隔。SentencePiece 直接处理原始字符流，不依赖预分词。中文的 BPE 可能会学到常见词组（如"人工智能"），但基础单元是字符。

**Q10: 如何评估一个 LLM 的好坏？⭐⭐**
多维度评估：困惑度（语言流畅性）、Benchmark（MMLU/HumanEval/GSM8K）、人工评估（有用性/安全性/准确性）、特定任务（翻译用BLEU、摘要用ROUGE）。

**Q11: 什么是 Prefix LM？和 CLM/MLM 有什么区别？⭐⭐⭐**
Prefix LM（如 T5、UniLM）：输入部分双向注意力（如 BERT），输出部分单向注意力（如 GPT）。结合了理解能力和生成能力。U-PaLM 和很多现代模型使用这种混合方式。

**Q12: Attention 有哪些高效变体？⭐⭐⭐**
- FlashAttention：IO 感知的精确注意力，不改变数学结果但更快更省显存
- Linear Attention：用核函数近似 Softmax，复杂度降到 $O(n)$
- Sparse Attention：只关注局部窗口 + 全局 token
- Multi-Query Attention：多头共享 K/V，减少 KV Cache 大小

**Q13: 为什么 Transformer 中要加残差连接？⭐⭐**
- 缓解梯度消失：梯度可以通过残差路径直接回传
- 支持训练更深的网络（原始 Transformer 6层 → 现代 LLM 100+ 层）
- 恒等映射基准：网络只需学习"增量修改"，而不是从头表示

**Q14: GPT 的 In-Context Learning 是怎么工作的？⭐⭐⭐**
GPT 通过 prompt 中的示例"隐式学习"任务。不更新参数，只通过注意力机制在推理时动态调整行为。可能的机制：Transformer 层内的梯度下降（隐式微调假说）。

**Q15: 什么是 Chain-of-Thought (CoT) Prompting？⭐⭐**
在 prompt 中加入推理步骤的示例，引导模型逐步推理。"Let's think step by step" 能显著提升数学和逻辑推理能力。只在足够大的模型（>100B）上有效（涌现能力）。

---

*最后更新：2026年6月*
