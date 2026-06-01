# 01 - LLM 基础原理高频面试题

> 面向：大模型应用工程师 / Agent 开发工程师
> 格式：问题 → 答案 → 追问 | 难度标记：⭐基础 ⭐⭐进阶 ⭐⭐⭐高级

---

## 一、Transformer 架构

---

### Q1: Transformer 的核心思想是什么？为什么取代了 RNN？ ⭐⭐

**答：** Transformer 的核心思想是 **"注意力机制"（Attention）**——让模型在处理序列数据时，能够直接"看到"序列中的任意位置，而不需要像 RNN 那样一步步地传递信息。

**RNN 的问题：** 想象你正在读一本很长的书。RNN 就像一个记忆力很差的人，必须从第一页开始，一页一页往后读，每读一页就更新一下脑子里的记忆。等读到第 500 页时，第 1 页的内容已经忘得差不多了——这就是 **长距离依赖问题**。更关键的是，RNN 是串行处理的，第 100 个 token 必须等第 99 个处理完才能开始，无法并行计算。

**Transformer 的解法：** Transformer 就像一个"开了全局视角"的读者，拿到书之后可以同时看到所有页面的内容，然后通过注意力机制决定重点关注哪些部分。这种"全局并行"的能力带来了两个巨大优势：

1. **完全并行计算**：所有 token 可以同时处理，训练速度大幅提升
2. **任意距离依赖**：第 1 个 token 和第 1000 个 token 之间的交互路径长度为 O(1)

```python
# RNN: 必须串行，h_t 依赖 h_{t-1}
for t in range(seq_len):
    h_t = rnn_cell(x[t], h[t-1])  # 无法并行

# Transformer: 所有位置同时计算
# Attention(Q, K, V) 一次矩阵乘法搞定所有位置对之间的关系
output = softmax(Q @ K.T / sqrt(d_k)) @ V  # 并行！
```

**追问：**
- Q: Transformer 的计算复杂度是多少？这是它的一个缺点吗？
- A: Self-Attention 的复杂度是 O(n²)，n 是序列长度。当序列很长时（如 100K tokens），计算和显存开销会非常大。这是 Transformer 的主要瓶颈，也是 Flash Attention、稀疏注意力等技术出现的原因。

---

### Q2: Self-Attention 的计算过程？为什么要除以 √(d_k)？ ⭐⭐

**答：** Self-Attention 的核心思想是：对序列中的每个 token，计算它与所有其他 token 的"相关性得分"，然后用这些得分对所有 token 的表示做加权求和。

**计算过程分 4 步：**

1. **线性投影**：每个 token 的向量 x 通过三个矩阵 W_Q、W_K、W_V 得到 Query、Key、Value
2. **计算注意力分数**：用 Q 和 K 做点积，得到每对 token 之间的相关性
3. **缩放 + Softmax**：除以 √(d_k) 后做 softmax，得到注意力权重
4. **加权求和**：用注意力权重对 V 做加权求和

```python
import torch
import torch.nn.functional as F

def self_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    # 第1步: 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1))
    # 第2步: 缩放 —— 这是关键！
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    # 第3步: 可选的 mask（decoder 中遮盖未来位置）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    # 第4步: softmax 归一化
    attn_weights = F.softmax(scores, dim=-1)
    # 第5步: 加权求和
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

# 示例
seq_len, d_k, d_v = 4, 8, 8
Q = torch.randn(1, seq_len, d_k)
K = torch.randn(1, seq_len, d_k)
V = torch.randn(1, seq_len, d_v)
output, weights = self_attention(Q, K, V)
print(f"输出形状: {output.shape}")       # [1, 4, 8]
print(f"注意力权重: {weights}")           # 每行和为 1
```

**为什么要除以 √(d_k)？** 假设 Q 和 K 的每个分量都是均值为 0、方差为 1 的独立随机变量，那么点积 Q·K 的方差就是 d_k。当 d_k 很大时（比如 64 或 128），点积值的绝对值会很大，导致 softmax 的输入落入梯度饱和区——softmax 输出接近 one-hot，梯度接近于 0，模型难以训练。除以 √(d_k) 将方差归一化回 1，让 softmax 的输出更"柔和"，梯度流动更健康。

**追问：**
- Q: 除了缩放点积注意力，还有其他计算注意力的方式吗？
- A: 有。加性注意力（Additive Attention）用一个小 MLP 来计算分数，适合 Q 和 K 维度不同的场景。不过在 Transformer 中，缩放点积注意力因为可以用矩阵乘法高效实现，所以更常用。

---

### Q3: Multi-Head Attention 的作用？为什么不用一个大 Head？ ⭐⭐

**答：** Multi-Head Attention 就是把 Q、K、V 投影到多个不同的子空间，分别计算注意力，最后拼接起来。类比来说：你在评判一幅画时，一个人可能只关注颜色，另一个人关注构图，还有人关注笔触——把多个"专家"的意见综合起来，比一个人从单一角度评判更全面。

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个 head 的维度

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        # 线性投影 + 拆分成多个 head
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 每个 head 独立计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        # 拼接所有 head + 输出投影
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_O(context)
```

**为什么不用一个大 Head（比如 d_k=512）？** 有两个核心原因：

1. **子空间多样性**：一个大 head 只能在一个表示空间中计算注意力，而多个小 head 可以在不同的子空间中捕获不同类型的关系。比如一个 head 学到语法依赖，另一个学到语义相似性，另一个学到位置关系。
2. **计算量相当但表达力更强**：8 个 head × d_k=64 的参数量和 1 个 head × d_k=512 差不多，但多头注意力的表达能力更强。这就像一个团队中有多样化背景的成员，比所有人都来自同一个专业更能解决复杂问题。

**追问：**
- Q: 所有 head 学到的东西一样怎么办？
- A: 实际中确实有研究发现某些 head 是冗余的。一些模型压缩技术（如 head pruning）会剪掉不重要的 head。另外 GQA（Grouped Query Attention）和 MQA 就是通过减少 K/V head 数量来节省显存和加速推理的技术。

---

### Q4: 位置编码有哪些方案？RoPE 为什么比绝对位置编码好？ ⭐⭐⭐

**答：** Transformer 的 Self-Attention 本身是"无序"的——打乱输入 token 的顺序，注意力的计算结果不变。所以必须显式地注入位置信息。主流方案有三种：

**1. 绝对位置编码（Sinusoidal，原始 Transformer）：** 给每个位置一个固定的向量，加到 token embedding 上。用 sin/cos 函数生成，不同位置有唯一编码。

**2. 可学习位置编码（Learned PE，GPT-2/BERT）：** 把位置编码当作可训练参数，通过学习得到。简单有效，但泛化到训练时未见过的长度时效果差。

**3. 旋转位置编码（RoPE，LLaMA/Qwen 等主流模型）：** 不修改 embedding，而是在计算注意力时，对 Q 和 K 的向量做旋转变换——把向量看成二维平面上的点，按位置角度旋转。

```python
import torch

def apply_rope(q, k, positions, d_model):
    """RoPE 的简化实现"""
    # 生成频率
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    # 位置 × 频率 = 角度
    angles = positions.unsqueeze(-1) * freqs  # [seq_len, d_model//2]
    # 旋转矩阵
    cos = angles.cos()
    sin = angles.sin()
    # 把 q, k 拆成前半和后半，做二维旋转
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q_rot = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
    k_rot = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
    return q_rot, k_rot
```

**RoPE 为什么好？** 核心优势是 **相对位置感知**：两个 token 做注意力时，得分只取决于它们的相对距离，而不是绝对位置。数学上可以证明，经过 RoPE 变换后，`Q_i · K_j` 的值只和 `i - j` 有关。这意味着：

- 更好的长度外推能力（通过 NTK-aware scaling 等技术可扩展到更长序列）
- 天然的相对位置编码，不需要额外参数
- 和 Linear Attention 等高效变体兼容

**踩坑经验：** 实际微调模型时，如果想扩展上下文长度（比如从 4K 到 32K），RoPE 的 `base frequency` 需要调整（如 YaRN、NTK-aware interpolation），否则超过训练长度后效果急剧下降。

**追问：**
- Q: ALiBi 和 RoPE 比怎么样？
- A: ALiBi（Attention with Linear Biases）直接在注意力分数上加一个和距离成正比的偏置，实现更简单，但效果上 RoPE 配合 NTK 缩放的长度外推能力更强。目前主流开源模型（LLaMA、Qwen、Mistral）都选择 RoPE。

---

### Q5: 为什么 Decoder-only 架构（GPT）比 Encoder-Decoder（T5）更流行？ ⭐⭐

**答：** 从 2022 年 ChatGPT 以来，几乎所有主流大模型（GPT-4、LLaMA、Qwen、Mistral、DeepSeek）都选择了 Decoder-only 架构。原因是多方面的：

**1. 训练效率更高：** Decoder-only 用因果语言模型（Causal LM）的目标——预测下一个 token。每个训练样本的每个位置都是一个训练信号，数据利用率高。而 Encoder-Decoder 的 Seq2Seq 任务中，只有 decoder 部分是生成目标，encoder 部分的训练信号较少。

**2. 天然的通用接口：** 任何任务都可以转化为"给定前文，续写后文"的形式。分类、翻译、问答、代码生成——统统可以续写。而 Encoder-Decoder 需要区分"输入"和"输出"，灵活性差一些。

**3. In-Context Learning 能力更强：** Decoder-only 的因果注意力模式使得模型在 few-shot prompting 中自然地将示例和查询放在同一个序列中处理，无需特殊的架构设计。

**4. 工程简洁性：** 只有一个模型，推理时只需要自回归循环。Encoder-Decoder 需要维护两套参数，KV Cache 管理也更复杂。

```python
# Decoder-only: 所有任务统一为续写
prompt = "将以下英文翻译成中文:\nHello, how are you?\n翻译:"
# → 模型直接续写 "你好，你怎么样？"

# Encoder-Decoder: 需要区分 source/target
source = "translate English to Chinese: Hello, how are you?"
# encoder 编码 source → decoder 生成 target
```

**追问：**
- Q: 那 T5/BART 这类 Encoder-Decoder 模型就完全没用了吗？
- A: 不是。在特定任务（如翻译、摘要）上，Encoder-Decoder 仍然可以表现很好。另外 Google 的 PaLM 2 据说也融合了 Encoder-Decoder 的思想。但在通用大模型领域，Decoder-only 目前是绝对主流。不过 Google 的 T5 仍在工业界广泛用于中小规模的任务模型。

---

## 二、Tokenization

---

### Q6: BPE 算法的原理？为什么不用字级别分词？ ⭐

**答：** BPE（Byte Pair Encoding，字节对编码）是目前大模型最主流的分词算法。它的核心思想是：**从单个字符开始，反复合并出现频率最高的相邻字符对，直到达到目标词表大小。**

**为什么不用字级别分词？** 假设用单个字符/字作为 token：
- 英文 "tokenization" 变成 12 个 token，序列超长，计算量暴增
- 模型需要学会从零拼出常见词，学习难度大
- 无法利用词汇层面的语义信息（如 "unhappy" 整体含义 ≠ "un" + "happy" 的简单叠加）

BPE 的好处是：**高频词保持完整（如 "the"、"is"），罕见词拆分为子词（如 "antidisestablishmentarianism" 拆成 "anti" + "dis" + "establish" + ...）**，在词表大小和序列长度之间取得平衡。

```python
# BPE 训练过程（简化演示）
from collections import Counter

def get_pairs(word_freqs):
    """统计所有相邻字符对的频率"""
    pairs = Counter()
    for word, freq in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

# 初始词表: 每个字符是一个 token
# 语料: "low" x5, "lower" x2, "newest" x6, "widest" x3
word_freqs = {
    "l o w": 5, "l o w e r": 2,
    "n e w e s t": 6, "w i d e s t": 3,
}
# 第1轮: 最高频对是 ("e", "s") → 合并为 "es"
# 第2轮: 最高频对是 ("es", "t") → 合并为 "est"
# 第3轮: 最高频对是 ("l", "o") → 合并为 "lo"
# ... 一直合并到词表大小达到目标

# 使用 HuggingFace tokenizers 库
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=1000, special_tokens=["[PAD]", "[UNK]"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)
print(tokenizer.encode("tokenization").tokens)
# 输出类似: ['tok', 'en', 'ization']
```

**追问：**
- Q: GPT 用的是 BPE，那 LLaMA 用的是什么？
- A: LLaMA 用的是 SentencePiece，底层算法也是 BPE（或 Unigram）。SentencePiece 的特点是直接在原始文本上训练，不依赖预分词（pre-tokenization），对多语言支持更好。GPT 系列使用 tiktoken 库，本质也是 BPE。

---

### Q7: 中文分词和英文分词有什么区别？为什么中文 token 效率低？ ⭐⭐

**答：** 这是实际开发中经常遇到的痛点。

**英文分词：** 英文天然有空格分隔单词，BPE 的 pre-tokenization 步骤先按空格拆分，再在词内做子词合并。一个常见英文单词通常 1-3 个 token。

**中文分词：** 中文没有空格，且 BPE 的初始单元是字节（byte）。早期模型（如 GPT-2）的词表主要基于英文语料训练，中文字符在词表中的覆盖率低，一个汉字可能被拆成 2-3 个 byte-level token。

```python
# 对比演示（使用 GPT-2 tokenizer）
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print(len(tokenizer.encode("Hello, how are you?")))      # ~6 tokens
print(len(tokenizer.encode("你好，你怎么样？")))            # ~15 tokens！

# 使用中文优化的 tokenizer（如 Qwen）
from transformers import AutoTokenizer
qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")
print(len(qwen_tok.encode("你好，你怎么样？")))             # ~6 tokens
```

**中文 token 效率低的原因：**

1. **词表训练语料偏向英文**：BPE 合并规则基于英文语料学到的，中文高频组合没被合并
2. **字符集巨大**：常用汉字有 6000+，加上生僻字、繁体字、日韩汉字，字表庞大
3. **字节回退机制**：对于词表中没有的字符，回退到 UTF-8 字节编码，一个中文字符 = 3 个字节 = 3 个 token

**实际影响：** 同样的上下文窗口（如 4K），中文能放的内容只有英文的一半甚至更少。API 价格也更高（按 token 计费）。

**解决方案：**
- 使用中文优化的模型和 tokenizer（Qwen、GLM、Baichuan 等）
- 如果自训 tokenizer，确保训练语料中中文占比足够高
- 推理前做 prompt 精简，减少不必要的 token 消耗

**追问：**
- Q: 怎么检查一个模型的中文分词效率？
- A: 简单方法：取一段中文文本，用 tokenizer 编码后看 token 数量。`len(tokenizer.encode("你"))` 如果等于 1 说明效率好，如果等于 2 或 3 说明效率差。可以对中英文平行文本做对比测试。

---

### Q8: Tokenizer 对模型性能有什么影响？ ⭐⭐

**答：** Tokenizer 是模型的"输入层"，它的质量直接影响模型的方方面面，但经常被忽视。

**1. 影响上下文利用率：** Tokenizer 的压缩率决定了同样长度的文本消耗多少 token。压缩率差的 tokenizer 会导致有效上下文长度缩短。比如一个 4K 窗口的模型，如果中文平均每个字 2.5 个 token，那么大约只能放 1600 个汉字，而压缩率好的 tokenizer 可以放 4000 个字。

**2. 影响模型训练和推理成本：** 序列长度直接决定 Self-Attention 的计算量（O(n²)）和显存占用。tokenizer 压缩率好 2 倍，训练和推理的计算量可以降低近 4 倍。

**3. 影响模型学习能力：** 如果 tokenizer 把一个完整的语义单元切碎了（比如把"北京"拆成"北"+"京"），模型就需要额外学习这些碎片组合的含义，增加了学习难度。

**4. 影响下游任务表现：** 代码生成任务中，如果 tokenizer 不认识缩进模式或特定语法结构，会严重影响生成质量。好的代码 tokenizer（如 StarCoder 使用的）会有专门的代码 token。

```python
# 同一个模型，不同 tokenizer 对比
from transformers import AutoTokenizer

# LLaMA-1 tokenizer: 词表 32K
tok1 = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
# LLaMA-3 tokenizer: 词表 128K（大幅扩充）
tok2 = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

text = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
print(f"LLaMA-2: {len(tok1.encode(text))} tokens")
print(f"LLaMA-3: {len(tok2.encode(text))} tokens")
# LLaMA-3 的 token 数明显更少，效率更高
```

**踩坑经验：**
- 微调模型时，不要更换 tokenizer！新 tokenizer 的词表和预训练不匹配，会导致 embedding 层完全失效
- 如果需要扩充词表（比如加入特殊 token），新增 token 的 embedding 需要初始化（通常用已有 token 的均值），然后继续训练
- 多语言模型的 tokenizer 一定要检查目标语言的效率

**追问：**
- Q: 词表大小怎么选？
- A: 一般在 32K-150K 之间权衡。词表越大：embedding 层参数越多（词表 × 隐藏维度），但序列越短，推理越快。词表越小：embedding 参数少，但序列长，推理慢。LLaMA-3 从 32K 扩到 128K 就是为了提高多语言和代码的 token 效率。

---

## 三、训练范式

---

### Q9: 预训练 → SFT → RLHF 三阶段分别做什么？ ⭐⭐

**答：** 这是当前大模型训练的标准流水线，三个阶段各有分工，类比为"大学教育"：

**阶段 1：预训练（Pre-training）—— 读万卷书**
- **目标：** 让模型学会语言和知识
- **数据：** 互联网文本（万亿 token 级别）
- **方法：** 自回归语言建模——给定前文，预测下一个 token
- **类比：** 像一个人从小到大读了无数的书和文章，积累了海量知识，但不知道怎么和人好好对话

```python
# 预训练目标：Causal Language Modeling
# 输入: "今天天气"
# 目标: "天天气真" (shift 一个 token)
import torch
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()
# logits: [batch, seq_len, vocab_size]
# labels: [batch, seq_len] — 每个位置的真实下一个 token
loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
```

**阶段 2：SFT（Supervised Fine-Tuning）—— 名师指点**
- **目标：** 让模型学会遵循指令、进行对话
- **数据：** 高质量的（指令, 回答）对，通常几万到几十万条
- **方法：** 用人类标注的对话数据做监督学习
- **类比：** 读了很多书的人开始接受对话训练，学会理解问题并给出有条理的回答

```python
# SFT 数据格式
sft_example = {
    "instruction": "请用简洁的语言解释什么是量子计算",
    "input": "",
    "output": "量子计算是利用量子力学原理进行计算的技术..."
}
# 训练时只计算 response 部分的 loss，instruction 部分不计算
# 通过 label masking 实现
```

**阶段 3：RLHF / DPO —— 人类反馈强化学习**
- **目标：** 让模型的回答更符合人类偏好（有用、安全、诚实）
- **数据：** 人类对多个回答的偏好排序
- **方法：** 训练一个奖励模型（Reward Model），然后用 PPO 优化策略
- **类比：** 在实际对话中接受用户反馈，不断调整回答风格

```python
# RLHF 的核心流程
# 1. 用 SFT 模型对同一个 prompt 生成多个回答
# 2. 人类标注员对回答排序: response_A > response_B > response_C
# 3. 训练奖励模型（Bradley-Terry 模型）
#    loss = -log(sigmoid(r_chosen - r_rejected))
# 4. 用 PPO 算法优化 LLM 策略，最大化奖励
#    objective = E[r(x,y)] - β * KL(π_θ || π_ref)
```

**追问：**
- Q: 一定要三个阶段吗？能不能跳过某个？
- A: 可以跳过 RLHF/DPO。很多模型（如 LLaMA-2-Chat）做了 RLHF，但也有不少优秀模型只做了 SFT（如 Mistral 的一些版本）。但一般来说，经过 RLHF/DPO 的模型在对话质量和安全性上更好。预训练阶段是必不可少的。

---

### Q10: RLHF 和 DPO 的区别？DPO 为什么更简单？ ⭐⭐⭐

**答：** RLHF 和 DPO 的目标相同——让模型的输出更符合人类偏好，但实现路径完全不同。

**RLHF（三步走）：**
1. 训练一个独立的 **奖励模型（RM）**：输入（prompt, response），输出一个标量分数
2. 用 **PPO 算法** 优化 LLM 策略：让模型生成高奖励的回答
3. 加 KL 散度约束：防止模型偏离原始策略太远（防止 reward hacking）

RLHF 的问题：需要同时维护 4 个模型（Policy、Reference、Reward、Value），显存占用巨大，PPO 训练不稳定，超参数敏感。

**DPO（Direct Preference Optimization，一步到位）：**

DPO 的核心洞察是：**可以直接从偏好数据中学到最优策略，不需要显式训练奖励模型。** 数学上证明了 RLHF 的最优解可以用闭式公式表示，然后把这个公式代入损失函数，直接优化。

```python
# DPO 的损失函数
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps, beta=0.1):
    """
    policy_xxx_logps: 当前策略模型对 chosen/rejected 回答的 log 概率
    reference_xxx_logps: 参考模型（SFT 模型）对 chosen/rejected 的 log 概率
    beta: 温度参数，控制偏离参考策略的程度
    """
    # 计算隐式奖励差
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    # 损失：让 chosen 的隐式奖励高于 rejected
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss

# DPO 数据格式（非常简单！）
dpo_data = {
    "prompt": "如何学习编程？",
    "chosen": "建议从 Python 入手，先学基础语法，再做小项目...",
    "rejected": "随便学，学什么都行，不用想太多。"
}
```

**DPO 为什么更简单？**

| 对比维度 | RLHF | DPO |
|---------|------|-----|
| 需要的模型数量 | 4个（Policy, Ref, RM, Value） | 2个（Policy, Ref） |
| 训练稳定性 | 不稳定，PPO 超参敏感 | 稳定，和 SFT 类似 |
| 实现复杂度 | 高（需要 PPO 框架） | 低（和普通 fine-tune 几乎一样） |
| 显存需求 | 非常高 | 较低 |
| 效果 | 略好（有专门 RM） | 接近（有时更好） |

**追问：**
- Q: DPO 有什么缺点吗？
- A: 主要有：1）DPO 假设偏好数据是确定性的，没有建模标注噪声；2）离线算法，不能像 PPO 那样在线探索新策略；3）当 chosen 和 rejected 差距很大时，梯度信号可能不够。后续的 IPO、KTO、ORPO 等方法在 DPO 基础上做了改进。

---

### Q11: 什么是 Scaling Law？对实际工作有什么指导意义？ ⭐⭐

**答：** Scaling Law（缩放定律）是指大模型的一个惊人规律：**模型的性能（以 loss 衡量）可以用一个简单的幂律公式来预测**，主要由三个因素决定：

**公式：** L(N, D, C) ≈ (N_c/N)^α_N + (D_c/D)^α_D + E

其中：
- N = 模型参数量
- D = 训练数据量
- C = 训练计算量（FLOPs）
- E = 数据的不可约损失（entropy of text）

**核心发现（Kaplan et al., 2020; Chinchilla, 2022）：**

1. 性能和 N、D、C 之间是 **对数线性关系**（双对数坐标下是一条直线）
2. **Chinchilla 最优比例**：给定计算预算，参数量和数据量应该等比例增长。即模型大 10 倍，数据也要多 10 倍
3. 很多模型是 **"过大过少数据"** 的（如 GPT-3 175B 只训练了 300B token）

```python
# Scaling Law 的简化演示
import numpy as np

def predict_loss(N, D, E=1.69, alpha_N=0.076, alpha_D=0.095,
                 N_c=8.8e13, D_c=5.4e13):
    """
    预测给定模型大小 N 和数据量 D 下的 loss
    N: 参数量（单位: 十亿）
    D: 训练 token 数（单位: 十亿）
    """
    loss = E + (N_c / (N * 1e9)) ** alpha_N + (D_c / (D * 1e9)) ** alpha_D
    return loss

# 对比不同规模
print(f"7B, 1T tokens:  loss = {predict_loss(7, 1000):.4f}")
print(f"70B, 1T tokens: loss = {predict_loss(70, 1000):.4f}")
print(f"70B, 2T tokens: loss = {predict_loss(70, 2000):.4f}")
```

**对实际工作的指导意义：**

1. **预估训练资源**：在训练前，可以用 Scaling Law 预估达到目标 loss 需要多少数据和算力
2. **模型选型**：给定预算，选择最优的模型大小和数据量组合
3. **判断是否值得继续**：如果实验中的 loss 趋势和 Scaling Law 预测不符，说明有问题

**追问：**
- Q: Scaling Law 有没有失效的时候？
- A: 有。在某些特定能力（如数学推理、代码生成）上，loss 的下降不一定转化为任务性能的提升——可能出现"涌现"现象。另外在模型非常大、数据非常多时，幂律关系可能开始偏离。

---

### Q12: 什么是涌现能力？为什么模型规模大了会突然学会某些能力？ ⭐⭐⭐

**答：** 涌现能力（Emergent Abilities）是指 **某些能力在小模型中完全不存在，但当模型规模超过某个阈值时突然表现出来。** 这就像水在 99°C 时还是液态，到了 100°C 突然沸腾——性质发生了质变。

**经典例子：**
- **思维链推理（Chain-of-Thought）**：GPT-3 13B 以下基本没有 CoT 能力，但 175B 配合 CoT prompting 能大幅提升数学和逻辑推理
- **多步算术**：小模型做不了三位数加法，大模型突然可以了
- **代码执行推理**：小模型完全不懂代码执行过程，大模型能追踪变量状态

```python
# 涌现能力的例子：Chain-of-Thought
# 小模型（< 100B）：
prompt = "Q: Roger has 5 tennis balls. He buys 2 more cans of 3. How many does he have now?\nA:"
# 小模型直接回答: "11" (经常算错)

# 大模型 + CoT：
prompt_cot = """Q: Roger has 5 tennis balls. He buys 2 more cans of 3. How many does he have now?
A: Roger started with 5 balls. 2 cans of 3 balls = 6 balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 for lunch and bought 6 more, how many do they have?
A:"""
# 大模型续写: "The cafeteria had 23 apples. They used 20, so 23 - 20 = 3. 
#              Then they bought 6 more, so 3 + 6 = 9. The answer is 9."
```

**为什么会出现涌现？** 有几种理论：

1. **度量选择理论**（Schaeffer et al., 2023）：涌现可能部分是评估指标的"错觉"——如果用非线性指标（如精确匹配准确率），性能的连续提升会看起来像突变。用平滑指标（如 token-level accuracy）可能看不到突变。
2. **组合能力假说**：复杂能力是多个基础能力的组合。只有当所有基础能力都达到阈值时，组合能力才突然出现。就像拼图，缺一块都是"不会"，拼上了就是"会了"。
3. **知识压缩阶段转变**：模型在某个规模下，可能突然找到了更高效的知识表示方式。

**追问：**
- Q: 涌现能力对实际应用有什么启示？
- A: 最重要的一点：**小模型上验证失败不代表大模型也不行。** 如果某个任务在 7B 模型上效果很差，不要急着放弃，试试 70B 或 130B。反之亦然：在大模型上看到的能力，不能期望小模型也具备。

---

## 四、推理基础

---

### Q13: Temperature、Top-p、Top-k 分别是什么？怎么调？ ⭐

**答：** 这三个参数控制模型生成文本时的"随机性"，是调用 LLM API 时最常用的参数。

**Temperature（温度）：** 控制概率分布的"平滑程度"。

```python
import torch
import torch.nn.functional as F

# 假设模型输出的 logits
logits = torch.tensor([2.0, 1.0, 0.5, 0.1, -1.0])

def apply_temperature(logits, temperature):
    """温度越低，概率越集中；温度越高，概率越平均"""
    return F.softmax(logits / temperature, dim=-1)

print("T=0.1 (非常确定):", apply_temperature(logits, 0.1))
# → [0.99, 0.01, 0.00, 0.00, 0.00]  几乎总是选最大值
print("T=1.0 (正常):   ", apply_temperature(logits, 1.0))
# → [0.52, 0.19, 0.12, 0.08, 0.03]  原始概率分布
print("T=2.0 (更随机): ", apply_temperature(logits, 2.0))
# → [0.34, 0.22, 0.17, 0.14, 0.09]  更加均匀
```

**Top-k：** 只在概率最高的 k 个 token 中采样。

```python
def top_k_sampling(logits, k=3):
    """只保留概率最高的 k 个 token"""
    values, indices = torch.topk(logits, k)
    probs = F.softmax(values, dim=-1)
    chosen_idx = torch.multinomial(probs, 1)
    return indices[chosen_idx]
```

**Top-p（Nucleus Sampling）：** 选择概率从高到低排列后，累计概率达到 p 的最小 token 集合。

```python
def top_p_sampling(logits, p=0.9):
    """选择累计概率达到 p 的最小 token 集合"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # 找到累计概率超过 p 的位置
    cutoff_idx = (cumulative_probs > p).nonzero()[0]
    # 把 cutoff 之后的 token 概率设为 -inf
    sorted_logits[cutoff_idx + 1:] = float('-inf')
    probs = F.softmax(sorted_logits, dim=-1)
    chosen_idx = torch.multinomial(probs, 1)
    return sorted_indices[chosen_idx]
```

**实际调参指南：**

| 场景 | Temperature | Top-p | Top-k |
|------|------------|-------|-------|
| 代码生成 | 0.0-0.2 | 0.95 | - |
| 知识问答 | 0.0-0.3 | 0.9 | - |
| 创意写作 | 0.7-1.0 | 0.9-0.95 | 50-100 |
| 数学推理 | 0.0 | - | - |
| 对话聊天 | 0.5-0.7 | 0.9 | - |

**踩坑经验：** 大多数生产环境应该用 **低温度（0~0.3）+ Top-p（0.9~0.95）**，除非明确需要创造性输出。高温度 + 低 Top-p 会导致输出质量不可控。

**追问：**
- Q: Temperature=0 就是 Greedy Decoding 吗？
- A: 实际上是的。Temperature → 0 时，softmax 退化为 argmax，等价于每次都选概率最高的 token。但不同 API 实现可能有微小差异（如 OpenAI 在 T=0 时可能仍有一点随机性）。

---

### Q14: Greedy Decoding vs Beam Search vs Sampling 的区别？ ⭐⭐

**答：** 这三种是文本生成中选择 token 的主要策略，各有适用场景。

**Greedy Decoding（贪心解码）：** 每步都选概率最高的 token。简单快速，但容易陷入重复，生成质量一般。

```python
def greedy_decode(model, input_ids, max_len=50):
    for _ in range(max_len):
        logits = model(input_ids)
        next_token = logits[:, -1, :].argmax(dim=-1)  # 每步选最高概率
        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
    return input_ids
```

**Beam Search（束搜索）：** 同时维护 k 条候选路径（beam），每步扩展所有路径，保留总概率最高的 k 条。

```python
def beam_search(model, input_ids, beam_width=5, max_len=50):
    beams = [(input_ids, 0.0)]  # (序列, log概率)
    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            logits = model(seq)
            top_k_probs, top_k_ids = logits[:, -1, :].topk(beam_width)
            for i in range(beam_width):
                new_seq = torch.cat([seq, top_k_ids[:, i:i+1]], dim=1)
                new_score = score + top_k_probs[:, i].log().item()
                candidates.append((new_seq, new_score))
        # 保留得分最高的 beam_width 条
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    return beams[0][0]  # 返回最优序列
```

**Sampling（采样）：** 按概率分布随机采样下一个 token。配合 Temperature、Top-p 使用，生成更多样化的文本。

**核心区别：**

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| Greedy | 快速、确定性 | 重复、质量一般 | 分类、提取任务 |
| Beam Search | 质量较优 | 慢、倾向保守 | 翻译、摘要 |
| Sampling | 多样、自然 | 不确定性 | 对话、创意写作 |

**LLM 时代的选择：** 大模型（7B+）几乎都用 Sampling，因为大模型的概率分布已经足够好，Sampling 能发挥其多样性优势。小模型可能需要 Beam Search 来保证基本质量。对话和 Agent 场景下，通常用 **Temperature=0（接近 Greedy）** 来保证可靠性。

**追问：**
- Q: 为什么 Beam Search 在对话中效果不好？
- A: Beam Search 优化的是整体序列概率，倾向于生成"安全"但无聊的回答（高频词、短句）。人类对话本来就是多样化的，低概率的词也可能是好回答。所以对话场景更适合 Sampling。

---

### Q15: 什么是 Hallucination？怎么缓解？ ⭐⭐⭐

**答：** 幻觉（Hallucination）是指模型 **生成了看似流畅合理、但实际上是错误的或无中生有的内容**。这是大模型最严重的问题之一，也是面试高频题。

**幻觉的类型：**

1. **事实性幻觉**：编造不存在的事实。如"爱因斯坦在 1921 年获得了诺贝尔化学奖"（应该是物理奖）
2. **忠实性幻觉**：回答偏离了给定的上下文/文档。如 RAG 场景中，文档说"收入增长 10%"，模型说"收入增长 20%"
3. **推理幻觉**：推理过程中引入了错误的中间步骤
4. **引用幻觉**：编造不存在的论文、书籍、URL

**幻觉产生的原因：**

1. **训练数据本身有错误**：互联网数据中有大量错误信息
2. **训练目标的局限**：语言模型优化的是"语言流畅度"而非"事实正确性"
3. **知识的模糊存储**：参数中的知识不像数据库那样精确，存在干扰和遗忘
4. **过度自信**：模型缺乏"说不知道"的能力

**缓解策略：**

```python
# 1. RAG（检索增强生成）—— 最实用的方案
def rag_query(question, retriever, llm):
    # 先检索相关文档
    relevant_docs = retriever.search(question, top_k=3)
    # 把文档作为上下文
    prompt = f"""基于以下文档回答问题。如果文档中没有相关信息，请说"我不确定"。

文档：
{chr(10).join(relevant_docs)}

问题：{question}
回答："""
    return llm.generate(prompt)

# 2. 加入"不确定"指令
system_prompt = """你是一个谨慎的助手。如果对答案不确定，请明确说"我不确定"或"我没有相关信息"。
不要编造事实。如果你的知识截止到某个日期，请说明。"""

# 3. 自我一致性检查（Self-Consistency）
def self_consistency_check(llm, question, n_samples=5):
    """多次采样，如果回答不一致则标记为不确定"""
    answers = [llm.generate(question, temperature=0.7) for _ in range(n_samples)]
    # 如果多数回答一致，可信度高
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0]
    if most_common[1] >= n_samples * 0.6:  # 超过 60% 一致
        return most_common[0], "high_confidence"
    else:
        return most_common[0], "low_confidence"

# 4. 引用溯源
prompt = """回答问题时，请在每个事实性陈述后标注来源。
格式：事实内容 [来源: 文档X] 或 [来源: 我的训练知识]
"""
```

**追问：**
- Q: 生产环境中，怎么系统性地检测幻觉？
- A: 常见方案：1）用另一个 LLM 做 fact-checking（LLM-as-Judge）；2）和知识库做自动比对；3）人工抽样审核；4）用 Faithfulness 评估指标（如 BERTScore、Rouge）检查回答和源文档的一致性。

---

### Q16: 什么是 Context Window？长上下文模型的核心技术是什么？ ⭐⭐

**答：** Context Window（上下文窗口）是指模型一次能处理的最大 token 数量。GPT-3.5 是 4K，GPT-4 Turbo 是 128K，Claude 3 支持 200K，Gemini 1.5 Pro 支持 1M。

**为什么原始 Transformer 只能处理短序列？**

1. **显存问题**：Self-Attention 需要存储 n×n 的注意力矩阵，n=4K 时约 16M 个元素，n=128K 时约 16B 个元素，显存爆炸
2. **位置编码泛化**：绝对位置编码在训练长度之外效果急剧下降
3. **注意力稀释**：序列太长时，注意力被分散到太多位置，重要信息被"淹没"

**长上下文的核心技术：**

```python
# 1. Flash Attention —— 降低显存（详见下一题）
# 不存储 n×n 的注意力矩阵，而是分块计算，显存从 O(n²) 降到 O(n)

# 2. RoPE 长度外推
# NTK-aware scaling: 修改 RoPE 的 base frequency
def ntk_aware_rope(base=10000, dim=128, scale=4):
    """将 base 增大，让模型"看到"更长的位置"""
    new_base = base * (scale ** (dim / (dim - 2)))
    freqs = 1.0 / (new_base ** (torch.arange(0, dim, 2).float() / dim))
    return freqs

# 3. YaRN (Yet another RoPE extensioN)
# 结合 NTK 缩放 + attention scaling + 温度调整

# 4. 滑动窗口注意力 (Sliding Window Attention)
# Mistral 使用：每个 token 只关注前后 W 个 token
def sliding_window_attention(Q, K, V, window_size=4096):
    """只在局部窗口内计算注意力"""
    seq_len = Q.size(1)
    # 为每个位置限制注意力范围
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = True
    scores = Q @ K.transpose(-2, -1) / (Q.size(-1) ** 0.5)
    scores[~mask] = float('-inf')
    return F.softmax(scores, dim=-1) @ V

# 5. Ring Attention / 序列并行
# 将长序列切分到多个 GPU 上，通过通信计算全局注意力
```

**实际使用长上下文的注意事项：**

1. **"Lost in the Middle" 问题**：研究表明，模型对上下文开头和结尾的信息记忆更好，中间的信息容易丢失
2. **成本**：128K 上下文的推理成本远高于 4K，因为 Attention 的计算量是 O(n²)
3. **实际有效长度**：虽然模型声称支持 128K，但实际对长文本中后半部分的信息检索准确率可能下降

**追问：**
- Q: RAG 和长上下文哪个更好？
- A: 目前最佳实践是两者结合：用 RAG 检索最相关的几段文本，放入上下文窗口。纯靠长上下文放全部文档，成本高且效果不一定好（Lost in the Middle 问题）。但如果文档不长（< 10K），直接放上下文可能更简单。

---

### Q17: Flash Attention 是什么？为什么能加速注意力计算？ ⭐⭐⭐

**答：** Flash Attention 是 2022 年由 Tri Dao 提出的一种 **IO 感知（IO-aware）** 的注意力计算算法。它不改变注意力的数学结果，但通过优化 GPU 内存访问模式，实现了 2-4 倍的加速和大幅显存节省。

**理解 Flash Attention，需要先理解 GPU 内存层次：**

```
GPU 内存层次（从快到慢）：
┌─────────────────────┐
│  SRAM (片上内存)      │  ~20 TB/s, ~20 MB  ← 极快但极小
├─────────────────────┤
│  HBM (高带宽显存)     │  ~2 TB/s, ~80 GB   ← 较快较大
├─────────────────────┤
│  系统内存 (CPU RAM)   │  ~100 GB/s         ← 慢
└─────────────────────┘
```

**标准注意力的问题：** 需要将完整的 n×n 注意力矩阵 S = QK^T 存入 HBM，然后读出来做 softmax，再存回去，再读出来和 V 相乘。大量的 **数据搬运（IO）** 成为瓶颈，而不是计算本身。

**Flash Attention 的核心思想：分块计算（Tiling）。**

```python
# 伪代码：Flash Attention 的核心逻辑
def flash_attention_forward(Q, K, V, block_size=256):
    """
    把 Q、K、V 分成小块，每次只加载一小块到 SRAM
    在 SRAM 中完成所有计算，结果写回 HBM
    """
    N = Q.size(0)  # 序列长度
    O = torch.zeros_like(Q)  # 输出
    l = torch.zeros(N)       # softmax 的分母（在线更新）
    m = torch.full((N,), float('-inf'))  # softmax 的最大值（在线更新）

    # 外层循环: 遍历 K, V 的块
    for j in range(0, N, block_size):
        Kj = K[j:j+block_size]  # 加载一块 K 到 SRAM
        Vj = V[j:j+block_size]  # 加载一块 V 到 SRAM

        # 内层循环: 遍历 Q 的块
        for i in range(0, N, block_size):
            Qi = Q[i:i+block_size]  # 加载一块 Q 到 SRAM
            # 在 SRAM 中计算局部注意力
            Sij = Qi @ Kj.T / (Qi.size(-1) ** 0.5)
            # 在线更新 softmax（关键技巧）
            mij_new = Sij.max(dim=-1).values
            # ... 省略复杂的在线 softmax 更新逻辑
            # 结果累加到 O 中
            O[i:i+block_size] += ...
    
    return O
```

**Flash Attention 的三个关键技巧：**

1. **分块（Tiling）**：将 Q、K、V 分成小块，每次只在 SRAM 中处理一小块，避免大量 HBM 读写
2. **在线 Softmax**：不需要先计算完所有分数再 softmax，而是边计算边更新（通过维护 running max 和 running sum）
3. **不存储注意力矩阵**：n×n 的注意力矩阵从不完整写入 HBM，显存从 O(n²) 降到 O(n)

**效果：**
- 训练速度提升 2-4 倍
- 显存占用从 O(n²) 降到 O(n)
- 支持更长的上下文长度
- Flash Attention 2 进一步优化，速度比 Flash Attention 1 再快 2 倍

```python
# 使用 Flash Attention（PyTorch 2.0+ 内置）
import torch.nn.functional as F

# 只需设置一个参数即可启用
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    is_causal=True  # 启用因果 mask
)
# PyTorch 会自动选择 Flash Attention 后端
```

**追问：**
- Q: Flash Attention 有没有缺点？
- A: 1）只在支持的硬件上有效（需要特定 GPU 架构）；2）实现复杂，CUDA kernel 难以修改；3）在序列较短（< 256）时优势不明显，因为 HBM 读写不是瓶颈。Flash Attention 3 针对 H100 的异步特性做了进一步优化。

---

## 五、实战难题

---

### 🔥 实战题 1：RAG 系统中，模型总是忽略检索到的文档，自己编答案

**问题描述：** 搭建了一个 RAG 系统，检索到了正确的文档片段，但模型经常忽略这些内容，用自己预训练的知识回答（可能是错误的）。

**原因分析：**
1. Prompt 中的指令不够强，没有明确要求"基于文档回答"
2. 检索到的文档和 query 的表述差距大，模型认为不相关
3. 模型的预训练知识和文档内容冲突，模型更相信自己

**解决方案：**

```python
# 方案1: 强化 Prompt 中的文档约束
prompt = """你是一个基于文档的问答助手。你必须且只能根据以下文档内容来回答问题。
如果文档中没有相关信息，你必须回答"根据提供的文档，我无法回答这个问题"。
绝对不要使用你自己的知识来补充。

文档内容：
{context}

问题：{question}

请严格基于上述文档回答："""

# 方案2: 先让模型提取相关段落，再生成答案（两步法）
step1_prompt = f"""从以下文档中，找出与问题最相关的段落，直接引用原文。
如果没有相关段落，说"无相关内容"。

文档：{context}
问题：{question}
相关段落："""

relevant = llm.generate(step1_prompt)
if "无相关内容" in relevant:
    return "根据提供的文档，我无法回答这个问题。"

step2_prompt = f"""基于以下相关段落回答问题。

相关段落：{relevant}
问题：{question}
回答："""
answer = llm.generate(step2_prompt)

# 方案3: 加入引用标记，强制模型引用原文
prompt = """回答问题时，必须在每句话后面用 [1][2] 等标记标注来自哪个文档段落。
...
"""
```

**效果提升：** 方案 2 的两步法通常效果最好，因为第一步强制模型先理解文档内容。

---

### 🔥 实战题 2：调用 LLM API 时，JSON 输出格式不稳定

**问题描述：** 让 LLM 输出 JSON 格式的结果，但经常格式不对——多了一个逗号、缺少引号、输出多余的文字。

**解决方案：**

```python
import json
import re

# 方案1: 使用 JSON Mode（OpenAI 支持）
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},  # 强制 JSON 输出
    messages=[
        {"role": "system", "content": "你必须输出有效的 JSON 格式"},
        {"role": "user", "content": "提取以下文本中的实体，输出JSON: ..."}
    ]
)
result = json.loads(response.choices[0].message.content)

# 方案2: 使用结构化输出库（如 Instructor）
import instructor
from pydantic import BaseModel

class Entity(BaseModel):
    name: str
    entity_type: str
    confidence: float

class ExtractionResult(BaseModel):
    entities: list[Entity]

# 自动解析 + 自动重试
result = client.chat.completions.create(
    model="gpt-4o",
    response_model=ExtractionResult,  # 自动保证输出格式
    messages=[{"role": "user", "content": f"提取实体: {text}"}]
)

# 方案3: 用正则做防御性解析
def safe_json_parse(text):
    """从 LLM 输出中安全提取 JSON"""
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试提取 ```json ... ``` 代码块
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # 尝试找第一个 { ... } 或 [ ... ]
    match = re.search(r'[\[{][\s\S]*[\]}]', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"无法从 LLM 输出中解析 JSON: {text[:200]}")
```

---

### 🔥 实战题 3：Agent 调用工具时，参数格式总是出错

**问题描述：** 给 LLM 定义了 Function Calling 的工具，但经常传错参数——类型不对（字符串传成数字）、嵌套结构错误、缺少必填参数。

**解决方案：**

```python
# 问题根因：工具描述不够清晰，LLM 不理解参数格式

# ❌ 不好的工具描述
bad_tool = {
    "name": "search_db",
    "description": "搜索数据库",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "filters": {"type": "object"}
        }
    }
}

# ✅ 好的工具描述
good_tool = {
    "name": "search_products",
    "description": "搜索商品数据库。返回匹配的商品列表。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，如 '红色连衣裙' 或 'iPhone 15'"
            },
            "filters": {
                "type": "object",
                "description": "过滤条件",
                "properties": {
                    "min_price": {
                        "type": "number",
                        "description": "最低价格，单位为元，如 100"
                    },
                    "max_price": {
                        "type": "number",
                        "description": "最高价格，单位为元，如 1000"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "food"],
                        "description": "商品类别"
                    }
                }
            },
            "limit": {
                "type": "integer",
                "description": "返回结果数量，默认 10，最大 50",
                "default": 10,
                "minimum": 1,
                "maximum": 50
            }
        },
        "required": ["query"]
    }
}

# 防御性代码：工具调用前做参数校验
def safe_tool_call(tool_name, arguments, tool_schema):
    try:
        # 自动类型转换
        for key, prop in tool_schema["parameters"]["properties"].items():
            if key in arguments:
                if prop["type"] == "number" and isinstance(arguments[key], str):
                    arguments[key] = float(arguments[key])
                elif prop["type"] == "integer" and isinstance(arguments[key], str):
                    arguments[key] = int(arguments[key])
        # 执行工具
        return TOOLS[tool_name](**arguments)
    except Exception as e:
        # 工具调用失败时，把错误信息反馈给 LLM 重试
        return f"工具调用失败: {e}. 请检查参数格式后重试。"
```

---

### 🔥 实战题 4：多轮对话中模型"忘记"之前的信息

**问题描述：** 一个客服 Agent 对话到第 10 轮时，开始忘记用户前面提到的订单号、问题描述等关键信息。

**解决方案：**

```python
# 根因：对话历史超出了上下文窗口，或者关键信息被"稀释"在大量对话中

# 方案1: 对话摘要压缩
def summarize_conversation(history, llm):
    """当对话超过 N 轮时，对历史做摘要"""
    summary_prompt = f"""请将以下对话历史压缩为关键信息摘要，保留：
1. 用户的核心诉求
2. 提到的关键数据（订单号、金额等）
3. 已达成的结论
4. 待解决的问题

对话历史：
{format_history(history)}

关键信息摘要："""
    return llm.generate(summary_prompt)

def manage_context(history, llm, max_turns=10):
    """智能上下文管理"""
    if len(history) > max_turns:
        # 保留最近 5 轮 + 前面的历史做摘要
        recent = history[-10:]  # 最近 5 轮（user + assistant）
        old_history = history[:-10]
        summary = summarize_conversation(old_history, llm)
        # 重新构建 context
        new_history = [
            {"role": "system", "content": f"之前的对话摘要：\n{summary}"}
        ] + recent
        return new_history
    return history

# 方案2: 关键信息提取（Slot Filling）
class ConversationState:
    def __init__(self):
        self.slots = {}  # 存储关键信息

    def update(self, history, llm):
        """每轮对话后更新关键信息槽"""
        extract_prompt = f"""从最新对话中提取关键信息，以 JSON 格式输出。
已有信息：{json.dumps(self.slots, ensure_ascii=False)}

最新对话：
User: {history[-2]['content']}
Assistant: {history[-1]['content']}

提取或更新的关键信息（只输出变化的部分）："""
        new_info = json.loads(llm.generate(extract_prompt))
        self.slots.update(new_info)

    def get_context(self):
        return f"已知信息：{json.dumps(self.slots, ensure_ascii=False)}"
```

---

### 🔥 实战题 5：LLM 生成的代码在安全沙箱中执行时频繁超时或内存溢出

**问题描述：** 一个代码执行 Agent，让 LLM 写 Python 代码然后在沙箱中运行。经常出现死循环、无限递归、或内存爆炸（如生成了 10GB 的列表）。

**解决方案：**

```python
import subprocess
import resource
import signal
import sys

# 方案: 多层防护
def safe_code_execution(code: str, timeout: int = 10, max_memory_mb: int = 256):
    """安全执行 LLM 生成的代码"""
    
    # 第1层: 静态检查（快速过滤明显问题）
    dangerous_patterns = [
        r'while\s+True\s*:',           # while True（可能死循环）
        r'while\s+1\s*:',
        r'recursion|sys\.setrecursionlimit',  # 递归相关
        r'open\(.*/etc/',               # 敏感文件访问
        r'__import__\s*\(\s*["\']os["\']\)',  # 动态导入 os
        r'subprocess|os\.system',       # 子进程
    ]
    import re
    for pattern in dangerous_patterns:
        if re.search(pattern, code):
            return {"error": f"代码包含危险模式: {pattern}", "status": "blocked"}

    # 第2层: 包装代码，添加防护
    wrapped_code = f"""
import sys
import resource

# 限制内存
resource.setrlimit(resource.RLIMIT_AS, ({max_memory_mb * 1024 * 1024}, {max_memory_mb * 1024 * 1024}))

# 限制递归深度
sys.setrecursionlimit(50)

# 限制大对象创建
class SizeGuard:
    def __init__(self, max_elements=1_000_000):
        self.max_elements = max_elements
    def check(self, obj):
        if hasattr(obj, '__len__') and len(obj) > self.max_elements:
            raise MemoryError(f"对象太大: {{len(obj)}} 元素")

guard = SizeGuard()

# 用户代码开始
try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
except MemoryError as e:
    print(f"内存溢出: {{e}}", file=sys.stderr)
except RecursionError as e:
    print(f"递归过深: {{e}}", file=sys.stderr)
except Exception as e:
    print(f"执行错误: {{type(e).__name__}}: {{e}}", file=sys.stderr)
"""

    # 第3层: 进程级超时
    try:
        result = subprocess.run(
            [sys.executable, "-c", wrapped_code],
            capture_output=True,
            text=True,
            timeout=timeout,  # 硬超时
            # 可选: 使用 Docker 容器进一步隔离
            # docker run --rm --memory=256m --cpus=1 python:3.11 python -c code
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "status": "success" if result.returncode == 0 else "error"
        }
    except subprocess.TimeoutExpired:
        return {"error": f"执行超时 ({timeout}s)", "status": "timeout"}
```

---

## 附录：推荐阅读

| 资源 | 链接 | 说明 |
|------|------|------|
| Attention Is All You Need | arxiv 1706.03762 | Transformer 原论文 |
| Scaling Laws for Neural LMs | arxiv 2001.08361 | Scaling Law |
| Training Compute-Optimal LLMs (Chinchilla) | arxiv 2203.15556 | 最优训练比例 |
| Flash Attention | arxiv 2205.14135 | Flash Attention |
| Direct Preference Optimization | arxiv 2305.18290 | DPO |
| LLaMA / LLaMA 2 / LLaMA 3 | Meta AI | 主流开源模型 |
| The Illustrated Transformer | Jay Alammar 博客 | 最佳可视化教程 |
| 跟李沐学 AI | Bilibili | 论文精读系列 |

---

---

## 25. ⭐⭐⭐ Q: 什么是 MoE（Mixture of Experts）架构？DeepSeek-V2/V3 为什么用 MoE？

> **路由机制、专家网络、负载均衡、MoE 的优缺点**

### 答案

MoE（Mixture of Experts，混合专家）是一种**稀疏激活**架构：模型包含多个"专家"子网络，但每次前向传播只激活其中一小部分。核心思想是**用参数量换性能，用稀疏性控制计算成本**。

**1. 架构组成**

- **Router（门控网络）**：对每个 token 计算一个概率分布，决定将 token 分配给哪些专家
- **Expert（专家网络）**：通常是 FFN 层，每个专家独立学习不同的知识
- **Top-K 选择**：每个 token 只选 K 个专家（如 DeepSeek-V3 用 Top-8）

**2. DeepSeek-V2/V3 的 MoE 设计**

DeepSeek 的 MoE 有三个关键创新：

- **DeepSeekMoE（细粒度专家）**：将传统专家进一步拆分为更小的专家，增加路由灵活性
- **Shared Expert + Routed Expert**：设置若干共享专家（每个 token 都经过），其余为路由专家（按需激活）。共享专家捕获通用知识，路由专家捕获领域特定知识
- **无辅助损失负载均衡**：传统 MoE 需要辅助损失（auxiliary loss）来防止专家坍缩，DeepSeek-V3 改用 bias 项动态调整，避免了辅助损失对模型质量的损害

**3. 为什么用 MoE**

| 维度 | Dense 模型 | MoE 模型 |
|------|-----------|----------|
| 参数量 | 全部激活 | 仅激活部分 |
| 计算量 | 与参数量成正比 | 远小于总参数量 |
| 典型比例 | — | 671B 总参数，37B 激活参数 |

MoE 让模型拥有**海量知识容量**（总参数大），同时保持**可控的推理成本**（激活参数小）。

**4. MoE 的优缺点**

**优点：**
- 总参数大 → 知识容量大，性能上限高
- 激活参数少 → 推理 FLOPs 与小模型相当
- 训练效率高 → 同等计算预算下性能更好

**缺点：**
- 显存占用仍是总参数量级别（需要存放所有专家权重）
- 负载均衡困难，容易出现"专家坍缩"（少数专家被过度使用）
- 通信开销大（分布式部署时需要 All-to-All 通信）
- Batch size 受限（每个专家处理的 token 不均匀）

### 代码示例：简化的 MoE Router 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMoELayer(nn.Module):
    """简化版 MoE 层"""
    
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router: 将 token 映射到专家概率
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # 多个专家网络 (每个都是 FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # (B*S, D)
        
        # 1. Router 计算每个 token 对各专家的分数
        router_logits = self.router(x_flat)        # (B*S, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (B*S, num_experts)
        
        # 2. Top-K 选择
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # 重新归一化 Top-K 的概率
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 3. 将 token 路由到对应专家并加权求和
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]       # (B*S,)
            expert_weight = top_k_probs[:, k]       # (B*S,)
            
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_weight[mask].unsqueeze(-1) * expert_output
        
        return output.view(B, S, D)

# 使用示例
moe = SimpleMoELayer(d_model=512, d_ff=2048, num_experts=8, top_k=2)
x = torch.randn(2, 10, 512)
out = moe(x)
print(f"总参数: {sum(p.numel() for p in moe.parameters()):,}")
print(f"输出形状: {out.shape}")
```

### 追问

1. **为什么 MoE 需要负载均衡？如果某些专家几乎不被选中会怎样？**（专家坍缩 → 死亡专家 → 容量浪费，形成恶性循环）
2. **DeepSeek-V3 的无辅助损失负载均衡是怎么实现的？**（为每个专家维护一个 bias 项，推理时 router_logits + bias，训练时用 bias 的移动平均动态调整）
3. **MoE 模型如何做分布式部署？Expert Parallelism 的通信瓶颈在哪？**（All-to-All 通信，token 需要发送到对应专家所在的 GPU）
4. **Switch Transformer 和 DeepSeekMoE 的路由策略有什么区别？**（Switch 用 Top-1，DeepSeek 用 Top-K + 共享专家）

---

## 26. ⭐⭐ Q: Scaling Law 的核心结论是什么？对实际工作有什么指导意义？

> **Chinchilla 定律、计算最优配置、实际应用指导**

### 答案

Scaling Law（缩放定律）描述了**模型性能（loss）与模型参数量 N、数据量 D、计算量 C 之间的幂律关系**。它回答了一个核心问题：给定有限预算，应该训练更大的模型还是用更多数据？

**1. Kaplan Scaling Law（OpenAI, 2020）**

核心发现：Loss 与 N、D、C 呈幂律关系：

$$L(N) \propto N^{-\alpha_N}, \quad L(D) \propto D^{-\alpha_D}, \quad L(C) \propto C^{-\alpha_C}$$

Kaplan 的结论：**应优先增大模型参数量**，数据量和计算量的增加不如增大模型有效。这直接影响了 GPT-3 的设计（175B 参数，但只用 300B token 训练）。

**2. Chinchilla Scaling Law（DeepMind, 2022）**

推翻了 Kaplan 的结论。核心发现：**模型参数量和训练数据量应该等比例扩展**。

- 对于给定的计算预算 C，存在一个最优的 N 和 D 组合
- 最优比例大约是：**每个参数应该对应约 20 个训练 token**
- GPT-3（175B 参数，300B token）严重**训练不足**——按 Chinchilla 定律应该用 3.5T token
- Chinchilla（70B 参数，1.4T token）用更少的参数但更多的数据，性能超过了 Gopher（280B）

**3. 关键公式**

```
计算最优关系: N_opt ∝ C^0.5, D_opt ∝ C^0.5
即: 计算预算翻 10 倍 → 模型参数和数据各增约 3.16 倍

实际经验公式:
- 每个参数至少 20 个 token（Chinchilla 最优）
- LLaMA 论文进一步证明：用远超 20x 的数据继续训练，
  虽然不满足 Chinchilla 最优，但推理时更划算
```

**4. 对实际工作的指导意义**

| 场景 | 指导 |
|------|------|
| **预算有限** | 用 Chinchilla 定律算出最优 N 和 D，不要盲目追求大模型 |
| **推理成本敏感** | 训练小模型 + 更多数据（LLaMA 策略）——小模型推理便宜 |
| **数据充足** | 适当过度训练小模型，如 7B 用 2T+ token |
| **评估模型潜力** | 用 Scaling Law 预估不同规模模型的 loss，决定是否值得继续扩大 |
| **Early Stopping** | 根据 Scaling Law 预测最终 loss，提前判断训练是否值得继续 |

**5. Scaling Law 的局限**

- 只描述了 **pre-training loss**，不直接等于下游任务表现
- 涌现能力（emergence）可能在特定规模突然出现，无法用平滑曲线预测
- 数据质量、数据配比等因素未被纳入原始公式
- 后训练（SFT/RLHF）的 scaling 行为不同于 pre-training

### 代码示例：用 Scaling Law 估算最优配置

```python
import numpy as np

def chinchilla_optimal(compute_budget_flops: float):
    """
    根据 Chinchilla Scaling Law 估算计算最优的模型参数和训练 token 数
    
    关系: C ≈ 6 * N * D (每个 token 前向+反向传播约 6N FLOPs)
    最优: N_opt ∝ C^0.5, D_opt ∝ C^0.5
    比例: D ≈ 20 * N
    """
    # 由 C = 6 * N * D 且 D = 20N 得:
    # C = 6 * N * 20N = 120 * N^2
    # N = sqrt(C / 120)
    N_opt = np.sqrt(compute_budget_flops / 120)
    D_opt = 20 * N_opt
    
    return {
        "optimal_params_B": N_opt / 1e9,       # 参数量 (B = 十亿)
        "optimal_tokens_T": D_opt / 1e12,       # 训练 token 数 (T = 万亿)
        "compute_flops": compute_budget_flops,
        "param_to_token_ratio": D_opt / N_opt   # 应该 ≈ 20
    }

# 不同计算预算下的最优配置
budgets = {
    "小实验 (1e20 FLOPs)": 1e20,
    "中等 (1e22 FLOPs)": 1e22,
    "GPT-3 级别 (1e24 FLOPs)": 1e24,
    "LLaMA-70B 级别 (1e25 FLOPs)": 1e25,
}

print("=" * 65)
print(f"{'预算':<25} {'参数量(B)':<12} {'Token(T)':<12} {'比例'}")
print("=" * 65)

for name, budget in budgets.items():
    result = chinchilla_optimal(budget)
    print(f"{name:<25} {result['optimal_params_B']:<12.1f} "
          f"{result['optimal_tokens_T']:<12.2f} {result['param_to_token_ratio']:.1f}")

# 对比已知模型的实际配置
print("\n" + "=" * 65)
print("已知模型的实际配置对比:")
print("=" * 65)
models = [
    ("GPT-3",     175, 300),
    ("Chinchilla", 70, 1400),
    ("LLaMA-7B",    7, 1000),
    ("LLaMA-70B",  70, 2000),
    ("DeepSeek-V3", 37, 14800),  # 37B 激活参数
]

for name, params_b, tokens_b in models:
    ratio = tokens_b / params_b
    status = "✓ 训练充足" if ratio >= 20 else "⚠ 训练不足"
    print(f"{name:<15} {params_b:>6}B params, {tokens_b:>6}B tokens, "
          f"比例={ratio:>5.1f} {status}")
```

输出示例：
```
已知模型的实际配置对比:
GPT-3            175B params,    300B tokens, 比例=  1.7 ⚠ 训练不足
Chinchilla        70B params,   1400B tokens, 比例= 20.0 ✓ 训练充足
LLaMA-7B           7B params,   1000B tokens, 比例=142.9 ✓ 训练充足
LLaMA-70B         70B params,   2000B tokens, 比例= 28.6 ✓ 训练充足
DeepSeek-V3       37B params,  14800B tokens, 比例=400.0 ✓ 训练充足
```

### 追问

1. **为什么 LLaMA 选择过度训练小模型而不是按 Chinchilla 最优配置训练？**（推理成本：7B 模型推理比 70B 便宜 10 倍，多花的训练成本很快被推理节省覆盖）
2. **Scaling Law 能预测涌现能力吗？**（不能，涌现能力往往是"相变"式的，平滑的幂律曲线无法预测）
3. **数据质量对 Scaling Law 有什么影响？**（高质量数据相当于增加了有效数据量，可以降低所需的 token 数；FineWeb 等高质量数据集让小数据量也能达到好效果）
4. **你所在团队如何利用 Scaling Law 指导模型选型？**（根据推理 QPS 和延迟要求选择模型大小，反推训练预算和数据需求）

---

> 📝 最后更新：2025 年 6 月 | 作者：LLM 面试题库项目
