# 03 - RAG 系统设计面试题

> RAG（Retrieval-Augmented Generation）是大模型应用中最核心的技术之一。本文档覆盖从基础原理到高级优化的全流程面试题，适合大模型应用工程师 / Agent 开发工程师面试准备。

---

## 一、架构演进

---

### Q1: RAG 的基本原理是什么？为什么需要 RAG？⭐

**答：**

RAG 的核心思想可以用一句话概括：**先查资料，再回答问题**。就像一个开卷考试——模型不需要把所有知识都记在"脑子里"（参数中），而是先从外部知识库中检索相关内容，然后基于这些内容生成答案。

**为什么需要 RAG？** 因为纯 LLM 有三个致命缺陷：

1. **知识过时**：模型的训练数据有截止日期，无法获知最新信息。比如问"今天的股价是多少"，纯 LLM 无法回答。
2. **幻觉问题**：模型会"一本正经地胡说八道"，编造不存在的事实。RAG 通过提供真实文档作为依据，大幅降低幻觉率。
3. **缺乏私有知识**：企业内部文档、个人笔记等私有数据，模型从未见过。RAG 可以将这些数据接入系统。

**RAG 的基本流程：**

```
用户提问 → Query 处理 → 向量检索 → 拼接上下文 → LLM 生成答案
```

用代码表示核心流程：

```python
from openai import OpenAI

client = OpenAI()

def naive_rag(query: str, knowledge_base: list[str], top_k: int = 3) -> str:
    """最基础的 RAG 流程"""
    # 1. 检索：从知识库中找到最相关的文档片段
    # 这里简化为关键词匹配，实际用向量检索
    relevant_docs = retrieve(query, knowledge_base, top_k)

    # 2. 构造 Prompt：把检索到的内容和用户问题拼在一起
    context = "\n\n".join(relevant_docs)
    prompt = f"""基于以下参考资料回答用户问题。如果资料中没有相关信息，请说明。

参考资料：
{context}

用户问题：{query}"""

    # 3. 生成：让 LLM 基于上下文生成答案
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

**类比理解：** 纯 LLM 像一个闭卷考试的学生，凭记忆答题；RAG 像一个开卷考试的学生，可以翻书找答案，但需要知道"翻哪本书、翻到哪一页"——这就是检索环节的核心挑战。

**追问：**
- RAG 和搜索引擎有什么区别？搜索引擎返回链接，RAG 直接返回答案
- RAG 一定比纯 LLM 好吗？不一定，如果检索质量差，反而会引入噪声，导致"检索到了错误信息并据此回答"的情况

---

### Q2: Naive RAG → Advanced RAG → Modular RAG 的演进？⭐⭐

**答：**

RAG 的发展经历了三个阶段，每个阶段都在解决前一阶段的痛点：

**第一阶段：Naive RAG（朴素 RAG）**

最简单的"检索 + 生成"模式。用户提问 → 向量检索 Top-K → 拼接上下文 → LLM 生成。

问题很多：
- 检索质量不稳定，经常检索到不相关的内容
- 不支持多轮对话，没有记忆
- 无法处理需要多步推理的复杂问题
- 上下文窗口有限，拼太多文档会超出限制

**第二阶段：Advanced RAG（进阶 RAG）**

在 Naive RAG 的基础上，针对每个环节做了优化：

```
优化前检索：  Query → 向量检索 → Top-K → 生成
优化后检索：  Query → Query 改写/扩展 → 混合检索 → Reranking → Top-K → 生成
```

核心改进包括：
- **检索前优化**：Query 改写（HyDE）、查询分解（将复杂问题拆成子问题）
- **检索中优化**：混合检索（Dense + Sparse）、多路召回
- **检索后优化**：Reranking 重排序、上下文压缩、去重

**第三阶段：Modular RAG（模块化 RAG）**

把 RAG 拆解成可插拔的模块，像搭积木一样灵活组合：

```python
class ModularRAG:
    """模块化 RAG 框架示意"""

    def __init__(self):
        self.modules = {
            "query_transform": QueryTransformModule(),   # 查询改写
            "retriever": HybridRetriever(),              # 混合检索
            "reranker": CrossEncoderReranker(),          # 重排序
            "compressor": ContextCompressor(),           # 上下文压缩
            "generator": LLMGenerator(),                 # 生成
            "memory": ConversationMemory(),              # 对话记忆
            "router": QueryRouter(),                     # 路由：决定走哪个流程
        }

    def run(self, query: str) -> str:
        # 路由：根据查询类型选择不同的处理流程
        route = self.modules["router"].classify(query)

        if route == "simple":
            # 简单问题：直接检索 + 生成
            docs = self.modules["retriever"].retrieve(query)
            return self.modules["generator"].generate(query, docs)

        elif route == "complex":
            # 复杂问题：先改写查询，多路检索，再 Reranking
            expanded_queries = self.modules["query_transform"].expand(query)
            docs = self.modules["retriever"].multi_retrieve(expanded_queries)
            docs = self.modules["reranker"].rerank(query, docs)
            docs = self.modules["compressor"].compress(docs)
            return self.modules["generator"].generate(query, docs)

        elif route == "conversational":
            # 对话问题：结合历史上下文
            history = self.modules["memory"].get_history()
            reformulated = self.modules["query_transform"].rewrite(query, history)
            docs = self.modules["retriever"].retrieve(reformulated)
            return self.modules["generator"].generate(reformulated, docs)
```

**类比理解：** Naive RAG 像一个刚学会查字典的小学生；Advanced RAG 像一个会用多种参考书、懂得筛选信息的研究员；Modular RAG 像一个配备了完整研究团队的教授，有专门的人负责查资料、整理信息、撰写报告，流程可以灵活调整。

**追问：**
- 你们项目中用的是哪种 RAG 架构？为什么？
- Modular RAG 的路由策略怎么设计？

---

### Q3: RAG 和微调（Fine-tuning）什么时候用哪个？各自的优缺点？⭐⭐

**答：**

这是面试中的高频问题。很多初学者会把两者混淆，其实它们解决的是**不同的问题**。

| 维度 | RAG | 微调（Fine-tuning） |
|------|-----|---------------------|
| **核心作用** | 给模型"查资料"的能力 | 给模型"学知识"的能力 |
| **知识更新** | 实时更新，改文档即可 | 需要重新训练 |
| **成本** | 低，只需向量数据库 | 高，需要 GPU 和训练数据 |
| **适用场景** | 知识密集型问答、文档检索 | 风格迁移、格式输出、领域适配 |
| **幻觉控制** | 较好，有据可循 | 一般，仍可能幻觉 |
| **上下文长度** | 受窗口限制 | 知识内化，无限制 |

**选择策略：**

```python
def choose_rag_or_finetune(scenario: dict) -> str:
    """根据场景选择 RAG 还是微调"""

    # 场景1：知识频繁更新 → RAG
    if scenario["knowledge_updates"] == "frequent":
        return "RAG"

    # 场景2：需要特定输出风格/格式 → 微调
    if scenario["need_custom_style"]:
        return "Fine-tuning"

    # 场景3：有大量私有文档需要查询 → RAG
    if scenario["has_private_documents"]:
        return "RAG"

    # 场景4：需要模型理解特定领域术语 → 微调（或 RAG + 微调）
    if scenario["need_domain_expertise"]:
        return "Fine-tuning (or both)"

    # 场景5：预算有限，快速上线 → RAG
    if scenario["budget"] == "limited":
        return "RAG"

    # 最佳实践：两者结合
    return "RAG + Fine-tuning"
```

**实际经验：** 大部分场景下，**先用 RAG 快速验证**，再根据效果决定是否需要微调。RAG 的迭代成本远低于微调——改几行 Prompt 或换个检索策略就能看到效果，而微调需要收集数据、训练、评估，周期长得多。

**追问：**
- RAG + 微调结合的方案怎么设计？
- 微调后的模型还需要 RAG 吗？

---

### Q4: RAG 的核心挑战有哪些？⭐⭐

**答：**

RAG 系统在实际落地中面临三大核心挑战：

**挑战一：检索质量（Retrieval Quality）**

检索是 RAG 的"地基"，如果检索不到正确的内容，后面做得再好也没用。常见问题：
- **语义鸿沟**：用户的表述和文档的措辞不同，导致匹配失败。比如用户问"怎么退货"，文档写的是"退换货流程"
- **长尾查询**：冷门问题检索效果差
- **多文档冲突**：不同文档给出矛盾信息，模型不知道听谁的

**挑战二：上下文窗口（Context Window）**

检索到了好几段相关内容，但 LLM 的上下文窗口有限，全部塞进去可能：
- 超出 token 限制
- 信息过多导致模型"迷路"（Lost in the Middle 问题——模型倾向于关注上下文的开头和结尾，忽略中间内容）

```python
# Lost in the Middle 问题示意
prompt = """
文档1（相关度最高）: ...
文档2（相关度一般）: ...
文档3（相关度最高）: ...  ← 模型可能忽略这段！
文档4（相关度一般）: ...
文档5（相关度最高）: ...
"""
# 解决方案：把最相关的文档放在开头和结尾
```

**挑战三：幻觉（Hallucination）**

即使检索到了正确的文档，模型仍可能：
- 忽略文档内容，自行编造答案
- 错误理解文档，给出误导性回答
- 过度依赖文档，不敢说"我不知道"

**实战踩坑经验：**
> 我们曾遇到一个问题：用户问"公司的年假政策是什么"，检索到了 HR 文档，但文档中有多个版本（2023 年和 2024 年），模型把两个版本的信息混在一起回答，导致错误。解决方案是给文档加上时间戳元数据，检索时优先返回最新版本。

**追问：**
- 如何量化和监控这些挑战？
- 你们团队是怎么逐步优化 RAG 系统的？

---

## 二、分块策略

---

### Q5: Chunk size 怎么选？过大过小有什么问题？⭐

**答：**

Chunk size（分块大小）是 RAG 系统中第一个需要调的超参数，选不好会直接影响检索质量。

**过小的问题（如 100 token）：**
- 语义不完整，片段缺乏上下文。比如"这个政策从明年开始执行"——哪个政策？从哪一年开始？信息缺失
- 检索到的片段太碎片化，LLM 难以据此生成完整答案
- 向量数据库中向量数量暴增，存储和检索成本上升

**过大的问题（如 2000 token）：**
- 主题混杂，一个 chunk 里包含多个话题，embedding 被"稀释"
- 检索精度下降，因为一个大 chunk 的 embedding 是所有内容的平均，不够聚焦
- 占用宝贵的上下文窗口，留给真正相关信息的空间变少

**怎么选？** 没有万能值，但有经验法则：

```python
def choose_chunk_size(doc_type: str, model_context: int = 128000) -> int:
    """根据文档类型选择分块大小"""

    # 经验法则：chunk size 通常在 256-1024 token 之间
    size_map = {
        "faq": 256,           # FAQ 问答对，每个 QA 独立，可以小一些
        "technical_doc": 512, # 技术文档，中等大小
        "legal_contract": 1024,  # 法律合同，需要保留完整条款
        "code": 1500,         # 代码文件，需要保留完整函数/类
        "news_article": 512,  # 新闻文章，按段落分
    }

    base_size = size_map.get(doc_type, 512)

    # 如果用的模型上下文窗口很大，可以适当增大 chunk
    if model_context >= 128000:
        base_size = int(base_size * 1.5)

    return base_size
```

**最佳实践：** 先用 512 token 作为基准，然后通过评估指标（检索命中率、答案准确率）来微调。建议做 A/B 测试，对比不同 chunk size 的效果。

**追问：**
- Chunk size 和 Embedding 模型有关系吗？
- 有没有自适应的分块方案？

---

### Q6: 有哪些分块策略？⭐⭐

**答：**

分块策略直接决定了检索的粒度和质量，不同场景需要不同的策略：

**1. 固定大小分块（Fixed-size Chunking）**

最简单粗暴的方式，按固定 token 数切分：

```python
def fixed_size_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """固定大小分块"""
    tokens = text.split()  # 简化处理，实际用 tokenizer
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
```

优点：实现简单、速度快。缺点：可能在句子中间切断，破坏语义完整性。

**2. 递归分块（Recursive Chunking）**

LangChain 默认策略，按层级分隔符递归切分：

```python
def recursive_chunk(text: str, chunk_size: int = 512) -> list[str]:
    """递归分块：优先按大分隔符切分，不行再按小分隔符"""
    separators = ["\n\n", "\n", "。", "！", "？", ".", " ", ""]

    def _split(text, separators):
        if len(text) <= chunk_size:
            return [text]

        sep = separators[0]
        if sep == "":
            return [text]

        parts = text.split(sep)
        chunks = []
        current = ""
        for part in parts:
            if len(current) + len(part) <= chunk_size:
                current += part + sep
            else:
                if current:
                    chunks.append(current.strip())
                current = part + sep
        if current:
            chunks.append(current.strip())

        # 如果某个 chunk 还是太大，用下一级分隔符继续切
        result = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                result.extend(_split(chunk, separators[1:]))
            else:
                result.append(chunk)
        return result

    return _split(text, separators)
```

**3. 语义分块（Semantic Chunking）**

基于语义相似度来决定在哪里切分，相似度低的地方就是天然的切分点：

```python
import numpy as np
from sentence_transformers import SentenceTransformer

def semantic_chunk(text: str, threshold: float = 0.5) -> list[str]:
    """语义分块：在语义转折处切分"""
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 先按句子切分
    sentences = text.split("。")
    embeddings = model.encode(sentences)

    # 计算相邻句子的相似度
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i], embeddings[i-1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1])
        )
        if similarity < threshold:
            # 相似度低，说明话题转变，切分
            chunks.append("。".join(current_chunk) + "。")
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append("。".join(current_chunk) + "。")

    return chunks
```

**4. 文档结构分块（Document Structure Chunking）**

利用文档本身的结构（标题、段落、列表）来分块：

```python
def structure_aware_chunk(markdown_text: str) -> list[str]:
    """基于 Markdown 结构分块：每个标题下的内容作为一个 chunk"""
    import re

    sections = []
    current_section = {"title": "", "content": ""}

    for line in markdown_text.split("\n"):
        if re.match(r"^#{1,3}\s", line):  # 遇到标题
            if current_section["content"]:
                sections.append(current_section)
            current_section = {"title": line, "content": ""}
        else:
            current_section["content"] += line + "\n"

    if current_section["content"]:
        sections.append(current_section)

    # 如果某个 section 太大，再用递归分块切分
    chunks = []
    for section in sections:
        full_text = section["title"] + "\n" + section["content"]
        if len(full_text) > 1000:
            chunks.extend(recursive_chunk(full_text))
        else:
            chunks.append(full_text.strip())

    return chunks
```

**策略选择指南：**

| 场景 | 推荐策略 |
|------|---------|
| 通用文档 | 递归分块（性价比最高） |
| FAQ/问答对 | 固定大小，按 QA 对切分 |
| 技术文档/手册 | 文档结构分块 |
| 长篇文章/报告 | 语义分块 |
| 代码文件 | 按函数/类分块（AST 解析） |

**追问：**
- 语义分块的 threshold 怎么调？
- 混合分块策略怎么设计？

---

### Q7: Overlap 的作用是什么？怎么设置？⭐

**答：**

Overlap（重叠）是指相邻两个 chunk 之间重叠的部分。它的作用是**防止上下文断裂**。

想象你把一篇文章切成几段，如果完全不重叠，有些关键信息可能正好落在切分点上——前一段的结尾提到了"该政策适用于……"，但"适用于谁"被切到了下一段，导致两段都不完整。加上 overlap 后，相邻 chunk 有一部分内容重叠，就能保证每段都包含足够的上下文。

```python
# 无 Overlap 的问题示例
text = "人工智能是计算机科学的一个分支。它试图理解智能的本质。这种理解产生了新的软件。"

# 无 overlap 切分：
# chunk1: "人工智能是计算机科学的一个分支。"
# chunk2: "它试图理解智能的本质。这种理解产生了新的软件。"
# 问题：chunk2 中的"它"指代什么？上下文丢失了。

# 有 overlap 切分（overlap = 1 句）：
# chunk1: "人工智能是计算机科学的一个分支。它试图理解智能的本质。"
# chunk2: "它试图理解智能的本质。这种理解产生了新的软件。"
# "它试图理解智能的本质。" 作为重叠部分，保证了语义连贯。
```

**怎么设置？**

```python
def get_overlap_size(chunk_size: int, doc_type: str = "general") -> int:
    """根据 chunk size 计算 overlap 大小"""

    # 经验法则：overlap 为 chunk_size 的 10%-20%
    ratio_map = {
        "general": 0.1,      # 通用文档：10%
        "technical": 0.15,   # 技术文档：15%（术语多，上下文依赖强）
        "legal": 0.2,        # 法律文档：20%（条款之间关联性强）
        "conversation": 0.1, # 对话记录：10%
    }

    ratio = ratio_map.get(doc_type, 0.1)
    overlap = int(chunk_size * ratio)

    # 确保 overlap 至少是 1-2 个句子的长度（约 50-100 token）
    return max(overlap, 50)
```

**踩坑经验：**
> Overlap 不是越大越好。我们曾把 overlap 设为 chunk_size 的 50%，结果向量数据库中存储量翻倍，检索时大量重复内容被召回，反而降低了效果。一般 10%-20% 就够了。

**追问：**
- Overlap 会导致检索到重复内容吗？怎么处理？
- 有没有不用 overlap 也能保持上下文完整的方法？

---

### Q8: 处理表格、图片、代码块等特殊内容有什么技巧？⭐⭐⭐

**答：**

特殊内容是 RAG 系统中的"老大难"问题，处理不好会严重影响效果。

**1. 表格处理**

表格是最棘手的，因为表格的语义依赖行列结构，简单切成文本会丢失结构信息。

```python
def process_table(table_data: dict) -> list[str]:
    """表格处理策略：多模态表示"""

    chunks = []

    # 策略1：转为自然语言描述
    # 表格：| 姓名 | 年龄 | 部门 |
    #       | 张三 | 25  | 技术 |
    # 转为："张三，年龄25岁，属于技术部门。"
    for row in table_data["rows"]:
        desc = "，".join([f"{col}是{val}" for col, val in zip(table_data["columns"], row)])
        chunks.append(desc)

    # 策略2：保留原始 Markdown 表格格式
    markdown_table = "| " + " | ".join(table_data["columns"]) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(table_data["columns"])) + " |\n"
    for row in table_data["rows"]:
        markdown_table += "| " + " | ".join(row) + " |\n"
    chunks.append(markdown_table)

    # 策略3：给表格加上描述性标题
    # "以下是2024年Q1销售数据表：\n" + markdown_table

    return chunks
```

**2. 图片处理**

图片需要多模态能力来处理：

```python
def process_image_with_multimodal(image_path: str, client) -> str:
    """用多模态模型提取图片内容"""
    import base64

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "请详细描述这张图片的内容，包括所有文字、数据、图表信息。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]
        }]
    )
    # 将图片描述作为文本 chunk 存入向量数据库
    return response.choices[0].message.content
```

**3. 代码块处理**

代码需要按函数/类为单位分块，保留完整结构：

```python
import ast

def chunk_python_code(code: str) -> list[str]:
    """基于 AST 解析 Python 代码，按函数/类分块"""
    tree = ast.parse(code)
    chunks = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # 提取完整的函数/类定义
            start_line = node.lineno - 1
            end_line = node.end_lineno
            lines = code.split("\n")
            chunk = "\n".join(lines[start_line:end_line])

            # 加上上下文：所在类名、文件名等
            context = f"# File: extracted\n# Type: {type(node).__name__}\n"
            chunks.append(context + chunk)

    return chunks
```

**通用技巧：**
- 表格：转为 Markdown 格式 + 自然语言描述双份存储
- 图片：用多模态模型生成描述文本，存入向量库
- 代码：用 AST 解析，按函数/类分块，保留签名和注释
- 混合文档（如 PDF）：用专门的文档解析工具（如 Unstructured、LlamaParse）

**追问：**
- PDF 中的复杂排版怎么处理？
- OCR 和文档解析工具怎么选？

---

## 三、Embedding 与向量数据库

---

### Q9: Embedding 模型怎么选？有哪些主流模型？⭐⭐

**答：**

Embedding 模型把文本转为向量，是 RAG 系统的"翻译官"。选错模型，后面的检索全白搭。

**选型考虑维度：**

```python
# Embedding 模型选型 checklist
selection_criteria = {
    "维度（dimension）": "768-1536 维，越高表达能力越强，但存储和计算成本也越高",
    "最大 token 长度": "是否支持长文本，512 vs 8192 token",
    "多语言支持": "中文场景必须关注，很多英文模型中文效果差",
    "推理速度": "在线场景需要低延迟",
    "MTEB 排名": "Massive Text Embedding Benchmark，权威评测榜单",
}
```

**主流模型对比：**

| 模型 | 维度 | 最大长度 | 中文支持 | 特点 |
|------|------|---------|---------|------|
| OpenAI text-embedding-3-large | 3072 | 8191 | 好 | 商业最强之一，支持维度裁剪 |
| OpenAI text-embedding-3-small | 1536 | 8191 | 好 | 性价比高 |
| BGE-large-zh (BAAI) | 1024 | 512 | 优秀 | 中文开源最强之一 |
| BGE-M3 (BAAI) | 1024 | 8192 | 优秀 | 多语言、多粒度、多功能 |
| GTE-large-zh (阿里) | 1024 | 8192 | 优秀 | 阿里开源，中文表现好 |
| E5-mistral-7b | 4096 | 32768 | 一般 | 基于 LLM 的 Embedding，效果强 |
| Jina-embeddings-v3 | 1024 | 8192 | 好 | 多语言，支持 Task LoRA |

```python
# 实际使用示例
from sentence_transformers import SentenceTransformer

# 中文场景推荐
model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
texts = ["RAG 是什么？", "检索增强生成技术介绍"]
embeddings = model.encode(texts)
print(f"维度: {embeddings.shape}")  # (2, 1024)

# 计算相似度
import numpy as np
similarity = np.dot(embeddings[0], embeddings[1]) / (
    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
)
print(f"相似度: {similarity:.4f}")
```

**实战建议：**
- 中文场景优先选 BGE 系列或 GTE 系列
- 需要长文本支持选 BGE-M3
- 追求极致效果且不差钱选 OpenAI text-embedding-3-large
- 选定模型后不要轻易更换，因为换模型意味着**所有已入库的向量都要重新生成**

**追问：**
- Embedding 模型的微调怎么做？
- 不同 Embedding 模型的向量能混用吗？

---

### Q10: 向量数据库的索引原理？⭐⭐⭐

**答：**

向量数据库的核心挑战是**在百万甚至亿级向量中快速找到最相似的 Top-K**。暴力搜索（逐一计算距离）太慢，所以需要索引来加速。

**HNSW（Hierarchical Navigable Small World）— 最常用**

原理类似"社交网络找人"：

```
Layer 2 (稀疏层):  A ———————— D        ← 快速定位到大致区域
                    |          |
Layer 1 (稠密层):  A — B — C — D — E    ← 精细搜索
                    |\  |\  |\  |\  |
Layer 0 (全量层):  A-B-C-D-E-F-G-H-I-J  ← 遍历最近邻
```

搜索过程：从最高层的稀疏图开始，快速跳到目标区域附近，然后逐层下降，在稠密层中精确搜索。

```python
# HNSW 参数调优
import hnswlib

index = hnswlib.Index(space="cosine", dim=1024)

# M: 每个节点的最大连接数，越大越精确，但内存和构建时间增加
# ef_construction: 构建时的搜索范围，越大索引质量越高
# ef_search: 查询时的搜索范围，越大结果越精确但越慢
index.init_index(max_elements=1000000, ef_construction=200, M=16)

# 添加向量
index.add_items(embeddings, ids)

# 搜索
index.set_ef(50)  # ef_search
labels, distances = index.knn_query(query_embedding, k=10)
```

**IVF（Inverted File Index）— 适合大规模数据**

原理类似"图书馆分区"：先把所有向量聚类分成 N 个区域（Voronoi cell），搜索时只在最相关的几个区域中查找。

```python
# IVF 索引示意（用 FAISS）
import faiss

dimension = 1024
nlist = 100  # 聚类中心数量

# 创建 IVF 索引
quantizer = faiss.IndexFlatL2(dimension)  # 暴力搜索作为聚类器
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# 训练（学习聚类中心）
index.train(embeddings)

# 添加向量
index.add(embeddings)

# 搜索：nprobe 控制搜索多少个区域
index.nprobe = 10  # 搜索 10 个最近的区域
distances, indices = index.search(query_embedding, k=10)
```

**PQ（Product Quantization）— 适合内存受限场景**

原理是"压缩存储"：把高维向量分成多段，每段用一个聚类中心的 ID 代替，大幅减少内存占用。

```python
# PQ 索引示意
m = 32          # 子量化器数量（向量被分成 32 段）
nbits = 8       # 每段用 8 bit 编码（256 个聚类中心）

# IVF + PQ 组合：先用 IVF 缩小范围，再用 PQ 压缩存储
index = faiss.IndexIVFPQ(
    faiss.IndexFlatL2(dimension),
    dimension, nlist, m, nbits
)
```

**类比理解：**
- HNSW 像"GPS 导航"：通过层级结构快速定位，精度高但内存大
- IVF 像"图书馆分区"：先去对的楼层找，速度快但可能漏掉跨区的结果
- PQ 像"zip 压缩"：牺牲一点精度换取大幅内存节省

**追问：**
- HNSW 的 M 和 ef 怎么调？
- 亿级向量场景怎么设计索引？

---

### Q11: 什么是 ANN？为什么不用精确搜索？⭐

**答：**

ANN（Approximate Nearest Neighbors，近似最近邻）是一种**用精度换速度**的搜索策略。

**为什么不用精确搜索（Exact Search）？**

精确搜索需要和数据库中的**每一个**向量计算距离。假设数据库有 100 万个 1024 维的向量：

```
精确搜索复杂度：O(N × D) = O(1,000,000 × 1024) ≈ 10 亿次浮点运算
ANN 搜索复杂度：O(log N × D) ≈ O(20 × 1024) ≈ 2 万次浮点运算
```

速度快了约 5 万倍！而精度损失通常在 95%+ 以上（即 ANN 找到的结果和精确搜索的结果有 95% 以上重叠）。

```python
import numpy as np
import time

# 对比精确搜索和 ANN 搜索的速度
np.random.seed(42)
database = np.random.randn(1000000, 1024).astype("float32")
query = np.random.randn(1, 1024).astype("float32")

# 精确搜索（暴力计算）
start = time.time()
distances = np.linalg.norm(database - query, axis=1)
top_k_exact = np.argsort(distances)[:10]
print(f"精确搜索耗时: {time.time() - start:.2f}s")

# ANN 搜索（用 FAISS HNSW）
import faiss
index = faiss.IndexHNSWFlat(1024, 32)
index.add(database)
start = time.time()
distances, indices = index.search(query, 10)
print(f"ANN 搜索耗时: {time.time() - start:.4f}s")  # 通常快 100-10000 倍
```

**为什么 RAG 场景可以接受 ANN？**

因为 RAG 本身就不需要"绝对精确"——检索到的文档最终要送给 LLM 处理，LLM 有理解能力，能从"差不多相关"的文档中提取有用信息。而且 Top-10 中偶尔有一个不太相关的影响不大。

**追问：**
- ANN 的召回率怎么评估？
- 什么场景必须用精确搜索？

---

### Q12: 主流向量数据库对比？⭐⭐

**答：**

| 特性 | Milvus | Pinecone | Weaviate | Chroma | FAISS |
|------|--------|----------|----------|--------|-------|
| **类型** | 分布式数据库 | 云服务 | 数据库 | 嵌入式库 | 算法库 |
| **部署** | 自托管/云 | 纯云 | 自托管/云 | 嵌入式 | 嵌入式 |
| **规模** | 十亿级 | 十亿级 | 百万级 | 十万级 | 十亿级 |
| **索引** | HNSW/IVF/DiskANN | 自研 | HNSW | HNSW | HNSW/IVF/PQ |
| **元数据过滤** | ✅ 强 | ✅ 强 | ✅ 强 | ✅ 基础 | ❌ 不支持 |
| **适用场景** | 生产级大规模 | 快速上线 | 中等规模 | 原型开发 | 算法研究 |
| **学习成本** | 高 | 低 | 中 | 低 | 中 |

```python
# 快速上手示例

# Chroma（最适合原型开发）
import chromadb
client = chromadb.Client()
collection = client.create_collection("my_docs")
collection.add(
    documents=["RAG是检索增强生成", "向量数据库存储embedding"],
    ids=["doc1", "doc2"]
)
results = collection.query(query_texts=["什么是RAG"], n_results=2)

# Milvus（适合生产环境）
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
connections.connect("default", host="localhost", port="19530")

# FAISS（适合算法实验）
import faiss
index = faiss.IndexFlatIP(1024)  # 内积索引
index.add(embeddings)
distances, indices = index.search(query_embedding, k=10)
```

**选择建议：**
- 原型验证 → Chroma（最简单，pip install 就能用）
- 生产环境，数据量大 → Milvus（分布式，功能全面）
- 不想运维，快速上线 → Pinecone（全托管 SaaS）
- 算法实验，追求性能 → FAISS（Meta 开源，性能最强）

**踩坑经验：**
> 我们最初用 Chroma 做原型，效果不错。上线时切到 Milvus，发现元数据过滤的语法完全不同，花了两天迁移。建议从一开始就考虑好生产方案，不要用太简陋的原型方案。

**追问：**
- 向量数据库和传统数据库（如 PostgreSQL + pgvector）怎么选？
- 向量数据库的数据持久化和备份怎么做？

---

## 四、检索优化

---

### Q13: Dense Retrieval vs Sparse Retrieval vs Hybrid Search？⭐⭐

**答：**

这三种检索方式各有优劣，理解它们的区别是优化 RAG 检索质量的基础。

**Dense Retrieval（稠密检索）**

用 Embedding 模型把文本转为稠密向量，通过向量相似度搜索。擅长**语义匹配**。

```python
# Dense Retrieval 示例
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

# "如何退货" 和 "退换货流程" 语义相近，能匹配上
query_emb = model.encode(["如何退货"])
doc_emb = model.encode(["退换货流程：请在7天内申请..."])
similarity = np.dot(query_emb[0], doc_emb[0]) / (
    np.linalg.norm(query_emb[0]) * np.linalg.norm(doc_emb[0])
)
print(f"Dense 相似度: {similarity:.4f}")  # 较高，语义匹配成功
```

**Sparse Retrieval（稀疏检索）**

传统的关键词检索，如 BM25。擅长**精确词汇匹配**。

```python
# Sparse Retrieval (BM25) 示例
from rank_bm25 import BM25Okapi

corpus = [
    "退换货流程：请在7天内申请退货退款",
    "公司年度营收增长20%",
    "Python安装教程"
]

# BM25 基于词频和逆文档频率计算相关性
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

query = "退货退款"
tokenized_query = query.split()
scores = bm25.get_scores(tokenized_query)
print(f"BM25 分数: {scores}")  # 第一个文档分数最高
```

**Hybrid Search（混合检索）**

结合两者的优势，用 RRF（Reciprocal Rank Fusion）等方法融合排序：

```python
def hybrid_search(query: str, dense_index, sparse_index, k: int = 10, alpha: float = 0.7):
    """混合检索：Dense + Sparse，用 RRF 融合排序"""

    # Dense 检索结果
    dense_results = dense_index.search(query, k=k * 2)  # 多召回一些

    # Sparse 检索结果
    sparse_results = sparse_index.search(query, k=k * 2)

    # RRF 融合
    rrf_scores = {}
    for rank, doc_id in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + alpha / (60 + rank + 1)

    for rank, doc_id in enumerate(sparse_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - alpha) / (60 + rank + 1)

    # 按 RRF 分数排序
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:k]]
```

**类比理解：**
- Dense Retrieval 像"意会"——理解你的意思，找到语义相关的
- Sparse Retrieval 像"言传"——只认关键词，你说什么就查什么
- Hybrid Search 像"既听你说的，也理解你的意思"——最全面

**什么时候用哪种？**
- 专业术语多（法律、医学）→ Sparse 更好（精确匹配关键词）
- 用户表述多样化 → Dense 更好（语义理解）
- 通用场景 → Hybrid 最稳

**追问：**
- RRF 中的 alpha 参数怎么调？
- 除了 RRF，还有哪些融合方法？

---

### Q14: 什么是 Reranking？为什么需要两阶段检索？⭐⭐

**答：**

Reranking（重排序）是 RAG 优化中最有效的手段之一。核心思想是：**先粗筛，再精排**。

**为什么需要两阶段？**

第一阶段的向量检索（Bi-encoder）速度快但精度有限——它独立地为 query 和 document 编码，然后计算相似度。这就像两个人分别写自我介绍，然后比较文字来判断是否匹配，效率高但可能忽略细节。

第二阶段的 Reranking（Cross-encoder）精度高但速度慢——它把 query 和 document 拼在一起输入模型，让模型"仔细阅读"后给出相关性分数。

```python
from sentence_transformers import CrossEncoder

# 第一阶段：Bi-encoder 粗筛（快，召回 50 个候选）
# query 和 document 独立编码，用向量相似度排序
# 速度：毫秒级

# 第二阶段：Cross-encoder 精排（慢，但更准）
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

query = "如何实现 RAG 系统"
candidates = [
    "RAG系统由检索和生成两个模块组成...",       # 高度相关
    "RAG是指检索增强生成技术...",               # 高度相关
    "如何搭建一个博客网站...",                  # 不相关
    "生成式AI在RAG中的应用和实现方案...",        # 高度相关
    "机器学习的基本概念介绍...",                 # 不太相关
]

# Cross-encoder 逐对打分
pairs = [[query, doc] for doc in candidates]
scores = reranker.predict(pairs)

# 按分数重排序
ranked_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

for doc, score in ranked_results[:3]:
    print(f"Score: {score:.4f} | {doc[:50]}...")
```

**为什么不直接用 Cross-encoder 检索？**

因为 Cross-encoder 需要对每个 (query, document) 对逐一计算，如果数据库有 100 万个文档，就要算 100 万次，太慢了。所以用两阶段策略：先用 Bi-encoder 快速筛出 Top-50，再用 Cross-encoder 从 50 个中精排。

```
第一阶段（Bi-encoder）：1,000,000 → 50 个候选（毫秒级）
第二阶段（Cross-encoder）：50 → Top-5（几十毫秒）
```

**常用 Reranker 模型：**
- `BAAI/bge-reranker-v2-m3`：多语言，中文效果好
- `Cohere/rerank-multilingual-v3.0`：商业 API，效果强
- `ms-marco-MiniLM-L-12-v2`：轻量级，速度快

**追问：**
- Reranker 的延迟怎么优化？
- Reranking 和 Lost in the Middle 问题有什么关系？

---

### Q15: 什么是 HyDE（Hypothetical Document Embeddings）？⭐⭐

**答：**

HyDE 是一种巧妙的查询优化技术，核心思想是：**先让 LLM 生成一个"假设性答案"，用这个答案去检索，比用原始问题检索效果更好**。

**为什么有效？** 因为用户的提问和文档的表述之间往往存在"语义鸿沟"。用户问"苹果公司的创始人是谁"，但文档可能写的是"乔布斯于1976年创立了苹果电脑公司"——两者用词差异大，直接用问题去检索可能匹配不好。但如果 LLM 先生成一个假设性答案"苹果公司由史蒂夫·乔布斯创立"，这个答案和文档的语义就更接近了。

```python
from openai import OpenAI

client = OpenAI()

def hyde_retrieval(query: str, retriever, llm_model: str = "gpt-4o-mini") -> list:
    """HyDE 检索流程"""

    # 第一步：让 LLM 生成假设性答案
    hypothetical_answer = client.chat.completions.create(
        model=llm_model,
        messages=[{
            "role": "user",
            "content": f"请回答以下问题（即使你不确定也要尝试回答）：\n{query}"
        }],
        max_tokens=300,
        temperature=0.7
    ).choices[0].message.content

    # 第二步：用假设性答案去检索（而不是用原始问题）
    # 假设性答案通常比原始问题更接近文档的表述
    results = retriever.search(hypothetical_answer, top_k=10)

    return results

# 示例
query = "RAG系统怎么解决幻觉问题"
# 假设性答案可能是："RAG系统通过检索外部知识库中的真实文档，将相关上下文
# 注入到LLM的prompt中，从而减少模型编造信息的可能性..."

# 用这个假设性答案去检索，比直接用问题检索更精准
```

**HyDE 的变体：多假设检索**

```python
def multi_hyde_retrieval(query: str, retriever, n_hypotheses: int = 3) -> list:
    """生成多个假设性答案，合并检索结果"""
    hypotheses = []
    for i in range(n_hypotheses):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"请回答：{query}"}],
            temperature=0.7 + i * 0.1,  # 不同温度产生不同答案
            max_tokens=200
        )
        hypotheses.append(response.choices[0].message.content)

    # 用所有假设性答案检索，合并去重
    all_results = []
    for hyp in hypotheses:
        results = retriever.search(hyp, top_k=5)
        all_results.extend(results)

    # 去重并按出现频率/相关度排序
    return deduplicate_and_rank(all_results)
```

**注意事项：** HyDE 会增加一次 LLM 调用，增加延迟和成本。在对延迟敏感的场景中，可以用小模型（如 GPT-4o-mini）生成假设性答案。另外，如果 LLM 对问题完全不了解，生成的假设性答案可能是错误的，反而会误导检索。

**追问：**
- HyDE 在什么场景下效果不好？
- 有没有不用 LLM 的 HyDE 替代方案？

---

### Q16: 多路召回是什么？怎么实现？⭐⭐

**答：**

多路召回（Multi-Route Recall）是指**同时使用多种检索策略，从不同维度召回候选文档，然后合并排序**。单一检索策略总有盲区，多路召回能互补。

```python
from typing import List, Dict

class MultiRouteRecall:
    """多路召回系统"""

    def __init__(self):
        self.dense_retriever = DenseRetriever()      # 向量检索
        self.sparse_retriever = BM25Retriever()      # 关键词检索
        self.entity_retriever = EntityRetriever()     # 实体检索
        self.reranker = CrossEncoderReranker()        # 重排序

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        candidates = {}

        # 路线1：Dense Retrieval — 语义匹配
        dense_results = self.dense_retriever.search(query, k=20)
        for rank, doc in enumerate(dense_results):
            candidates.setdefault(doc["id"], {"doc": doc, "scores": {}})
            candidates[doc["id"]]["scores"]["dense"] = 1.0 / (rank + 1)

        # 路线2：Sparse Retrieval (BM25) — 关键词匹配
        sparse_results = self.sparse_retriever.search(query, k=20)
        for rank, doc in enumerate(sparse_results):
            candidates.setdefault(doc["id"], {"doc": doc, "scores": {}})
            candidates[doc["id"]]["scores"]["sparse"] = 1.0 / (rank + 1)

        # 路线3：Entity-based Retrieval — 实体匹配
        entities = self._extract_entities(query)  # 提取实体
        for entity in entities:
            entity_results = self.entity_retriever.search(entity, k=10)
            for rank, doc in enumerate(entity_results):
                candidates.setdefault(doc["id"], {"doc": doc, "scores": {}})
                candidates[doc["id"]]["scores"]["entity"] = 1.0 / (rank + 1)

        # 融合分数（加权 RRF）
        weights = {"dense": 0.5, "sparse": 0.3, "entity": 0.2}
        final_scores = {}
        for doc_id, info in candidates.items():
            score = sum(
                weights.get(route, 0) * s
                for route, s in info["scores"].items()
            )
            final_scores[doc_id] = score

        # 取 Top-K 做 Reranking
        top_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:30]
        top_docs = [candidates[doc_id]["doc"] for doc_id, _ in top_candidates]

        # Reranking 精排
        reranked = self.reranker.rerank(query, top_docs)
        return reranked[:top_k]
```

**多路召回的典型路线：**
1. **Dense 路线**：语义相关的内容
2. **Sparse 路线**：包含关键词的内容
3. **Entity 路线**：包含特定实体（人名、公司名、产品名）的内容
4. **Metadata 路线**：按时间、来源、类型等元数据过滤

**追问：**
- 各路线的权重怎么调？
- 多路召回的延迟怎么控制？

---

### Q17: 如何处理多跳推理（Multi-hop Reasoning）？⭐⭐⭐

**答：**

多跳推理是指**需要综合多个文档的信息才能回答的复杂问题**。比如："爱因斯坦出生城市的现任市长是谁？"——需要先找到爱因斯坦出生在乌尔姆，再查乌尔姆的现任市长。

**方法一：Query Decomposition（查询分解）**

把复杂问题拆成多个子问题，逐步检索：

```python
def multi_hop_qa(query: str, client, retriever) -> str:
    """多跳问答：查询分解 + 逐步检索"""

    # 第一步：将复杂问题分解为子问题
    decomposition = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""请将以下复杂问题分解为需要依次回答的子问题：

问题：{query}

输出格式：
1. [子问题1]
2. [子问题2]
..."""
        }]
    ).choices[0].message.content

    sub_questions = parse_sub_questions(decomposition)

    # 第二步：逐步检索和回答
    context_so_far = ""
    for sub_q in sub_questions:
        # 结合之前的答案来改写当前子问题
        enriched_query = f"{sub_q}\n已知信息：{context_so_far}" if context_so_far else sub_q

        # 检索
        docs = retriever.search(enriched_query, top_k=5)

        # 回答当前子问题
        answer = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""基于以下文档回答问题。

文档：{chr(10).join(docs)}
问题：{sub_q}"""
            }]
        ).choices[0].message.content

        context_so_far += f"\n{sub_q} → {answer}"

    # 第三步：综合所有子问题的答案回答原始问题
    final_answer = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""基于以下推理过程，回答原始问题。

推理过程：{context_so_far}

原始问题：{query}"""
        }]
    )

    return final_answer.choices[0].message.content
```

**方法二：Graph RAG（后面详细讲）**

用知识图谱存储实体之间的关系，直接在图上做多跳查询。

**方法三：Iterative Retrieval（迭代检索）**

Agent 自主决定是否需要继续检索，直到信息足够：

```python
def iterative_retrieval(query: str, client, retriever, max_iterations: int = 5):
    """迭代检索：Agent 自主判断是否需要继续检索"""
    context = ""
    for i in range(max_iterations):
        # LLM 判断是否已有足够信息
        judgment = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""问题：{query}
已收集的信息：{context}
请判断：已有信息是否足以回答问题？
如果不足，请说明还需要什么信息。
格式：STATUS: sufficient/insufficient
NEED: [还需要什么信息]"""
            }]
        ).choices[0].message.content

        if "STATUS: sufficient" in judgment:
            break

        # 提取需要进一步检索的内容
        need = extract_need(judgment)
        new_docs = retriever.search(need, top_k=3)
        context += "\n" + "\n".join(new_docs)

    return generate_answer(query, context)
```

**追问：**
- 多跳推理的错误会级联放大吗？怎么处理？
- Query Decomposition 和 Iterative Retrieval 哪个更好？

---

## 五、高级 RAG

---

### Q18: 什么是 Graph RAG？和传统 RAG 有什么区别？⭐⭐⭐

**答：**

传统 RAG 把文档切成片段，用向量相似度检索——它擅长找"哪段话和问题相关"，但不擅长理解**实体之间的关系**。

Graph RAG 的核心思想是：**先从文档中提取实体和关系，构建知识图谱，然后利用图的结构信息来增强检索和推理**。

```
传统 RAG：
文档 → 分块 → Embedding → 向量检索 → LLM 生成

Graph RAG：
文档 → 实体/关系提取 → 知识图谱 → 图查询 + 向量检索 → LLM 生成
```

```python
# Graph RAG 核心流程示意
class GraphRAG:
    def __init__(self):
        self.graph = KnowledgeGraph()  # 知识图谱
        self.vector_store = VectorStore()  # 向量存储

    def build_graph(self, documents: list[str]):
        """从文档中提取实体和关系，构建知识图谱"""
        for doc in documents:
            # 用 LLM 提取实体和关系
            entities_relations = self._extract_entities_and_relations(doc)

            for entity in entities_relations["entities"]:
                self.graph.add_node(entity["name"], entity["type"], entity["properties"])

            for rel in entities_relations["relations"]:
                self.graph.add_edge(rel["source"], rel["target"], rel["type"], rel["description"])

    def query(self, question: str) -> str:
        """Graph RAG 查询：结合图结构和向量检索"""

        # 1. 从问题中提取实体
        entities = self._extract_entities(question)

        # 2. 在图中查找相关子图
        subgraph = self.graph.get_subgraph(entities, hops=2)

        # 3. 向量检索补充
        vector_results = self.vector_store.search(question, top_k=5)

        # 4. 融合图信息和向量检索结果
        context = self._format_context(subgraph, vector_results)

        # 5. LLM 生成答案
        return self._generate(question, context)
```

**Graph RAG 的优势：**
- **多跳推理**：通过图的边直接查询关系链，不需要多次检索
- **全局理解**：可以对整个知识库做社区摘要，回答"这份文档的主要主题是什么"这类全局性问题
- **可解释性**：答案可以追溯到具体的实体和关系路径

**类比理解：** 传统 RAG 像在图书馆里按关键词找书，Graph RAG 像有一张人物关系图谱——你问"A 和 B 是什么关系"，不需要翻遍所有书，直接在图上就能找到路径。

**追问：**
- Graph RAG 的构建成本高吗？
- 什么时候用 Graph RAG，什么时候用传统 RAG？

---

### Q19: 什么是 Agentic RAG？Agent 如何优化 RAG 流程？⭐⭐⭐

**答：**

传统 RAG 是一个固定的流水线：检索 → 生成，不管什么问题都走同一条路。Agentic RAG 把 Agent 的决策能力引入 RAG，让系统能**根据问题动态调整检索策略**。

```python
class AgenticRAG:
    """Agentic RAG：Agent 驱动的自适应检索"""

    def __init__(self):
        self.tools = {
            "dense_search": DenseRetriever(),       # 语义检索
            "keyword_search": BM25Retriever(),       # 关键词检索
            "sql_query": SQLExecutor(),              # 结构化数据查询
            "web_search": WebSearchTool(),           # 联网搜索
            "calculator": CalculatorTool(),          # 计算器
            "knowledge_graph": KGQueryTool(),        # 知识图谱查询
        }

    def run(self, query: str) -> str:
        """Agent 自主决定使用什么工具、检索几次"""

        messages = [
            {"role": "system", "content": """你是一个 RAG 助手。你可以使用以下工具来回答问题：
- dense_search: 语义搜索，适合模糊查询
- keyword_search: 关键词搜索，适合精确查询
- sql_query: 查询结构化数据（如数据库表格）
- web_search: 搜索互联网获取最新信息
- knowledge_graph: 查询实体关系

请根据问题类型选择合适的工具。如果一次检索不够，可以多次使用不同工具。
当信息足够时，生成最终答案。"""},
            {"role": "user", "content": query}
        ]

        # Agent 循环：自主决策 → 执行工具 → 观察结果 → 继续决策
        for _ in range(5):  # 最多 5 轮
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self._get_tool_definitions()
            )

            if response.choices[0].finish_reason == "stop":
                return response.choices[0].message.content

            # 执行工具调用
            for tool_call in response.choices[0].message.tool_calls:
                result = self._execute_tool(tool_call)
                messages.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
```

**Agentic RAG 的核心能力：**

1. **自适应路由**：根据问题类型选择不同的检索策略
2. **查询改写**：如果第一次检索结果不好，自动改写查询重试
3. **多步推理**：自动分解复杂问题，逐步检索
4. **自我反思**：评估检索结果质量，决定是否需要补充检索
5. **工具选择**：除了向量检索，还能调用 SQL、API、计算器等工具

**追问：**
- Agentic RAG 的延迟怎么控制？Agent 循环太多轮怎么办？
- 如何防止 Agent 陷入死循环？

---

### Q20: 如何处理长文档的 RAG？⭐⭐⭐

**答：**

长文档（如 100 页的报告、一本电子书）的 RAG 面临独特挑战：分块太多导致检索精度下降，分块太大导致信息丢失。

**策略一：层次化索引（Hierarchical Indexing）**

```python
class HierarchicalIndex:
    """层次化索引：摘要 → 章节 → 段落"""

    def index_document(self, full_text: str):
        # 第一层：文档级摘要
        doc_summary = self._generate_summary(full_text, level="document")

        # 第二层：章节级摘要
        sections = self._split_by_sections(full_text)
        section_summaries = []
        for section in sections:
            summary = self._generate_summary(section, level="section")
            section_summaries.append({
                "title": section["title"],
                "summary": summary,
                "chunks": self._chunk_section(section)
            })

        # 检索时：先检索摘要找到相关章节，再在章节内检索具体段落
        # 这样可以大幅减少搜索空间

    def search(self, query: str) -> list[str]:
        # 第一步：在摘要层检索，找到最相关的 2-3 个章节
        relevant_sections = self._search_summaries(query, top_k=3)

        # 第二步：在这些章节内做细粒度检索
        fine_results = []
        for section in relevant_sections:
            results = self._search_chunks(query, section["chunks"], top_k=3)
            fine_results.extend(results)

        return fine_results
```

**策略二：Map-Reduce 模式**

```python
def map_reduce_rag(query: str, long_document: str, client) -> str:
    """Map-Reduce 处理长文档"""

    # Map 阶段：对每个 chunk 独立提取相关信息
    chunks = split_document(long_document, chunk_size=2000)
    partial_answers = []

    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"从以下文本中提取与问题相关的信息，如果没有相关信息就说'无相关信息'。\n\n"
                          f"问题：{query}\n\n文本：{chunk}"
            }]
        )
        partial_answers.append(response.choices[0].message.content)

    # Reduce 阶段：合并所有部分答案
    combined = "\n---\n".join([a for a in partial_answers if "无相关信息" not in a])

    final = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"基于以下信息，全面回答问题。\n\n信息：{combined}\n\n问题：{query}"
        }]
    )

    return final.choices[0].message.content
```

**策略三：利用长上下文模型**

随着模型上下文窗口的增大（如 128K、200K token），可以直接将长文档塞入上下文：

```python
def long_context_rag(query: str, full_document: str, client) -> str:
    """直接利用长上下文窗口"""
    # 如果文档在模型上下文窗口内，可以直接塞入
    # 但要注意：长上下文模型也有 Lost in the Middle 问题
    # 建议把最重要的信息放在开头和结尾

    response = client.chat.completions.create(
        model="gpt-4o",  # 支持 128K 上下文
        messages=[{
            "role": "user",
            "content": f"""以下是完整文档：

{full_document}

请基于文档内容回答：{query}"""
        }]
    )
    return response.choices[0].message.content
```

**追问：**
- 长上下文模型能完全替代 RAG 吗？
- Lost in the Middle 问题怎么缓解？

---

### Q21: 什么是 Contextual Retrieval？Anthropic 的方案是什么？⭐⭐⭐

**答：**

Contextual Retrieval 是 Anthropic 提出的一种优化方案，核心解决的问题是：**分块后，每个 chunk 失去了原始文档的上下文信息**。

比如一个 chunk 内容是"该政策从2024年1月1日起执行"——哪个政策？在哪个文件里？chunk 本身不知道。Contextual Retrieval 的做法是：**在分块时，用 LLM 为每个 chunk 生成一段上下文前缀，说明这段内容出自哪里、讲的是什么**。

```python
def contextual_retrieval_indexing(documents: list[dict], client) -> list[dict]:
    """Contextual Retrieval 索引流程"""

    enriched_chunks = []

    for doc in documents:
        full_text = doc["content"]
        chunks = split_document(full_text, chunk_size=512)

        for chunk in chunks:
            # 用 LLM 为每个 chunk 生成上下文前缀
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # 用小模型节省成本
                messages=[{
                    "role": "user",
                    "content": f"""<document>
{full_text[:5000]}  # 取文档前部分作为参考
</document>

<chunk>
{chunk}
</chunk>

请用 1-2 句话简要说明这段内容在原文中的上下文（它讲的是什么主题，在哪个章节/部分）。"""
                }],
                max_tokens=100
            )

            context_prefix = response.choices[0].message.content

            # 将上下文前缀加到 chunk 前面
            enriched_chunk = f"{context_prefix}\n\n{chunk}"
            enriched_chunks.append({
                "original_chunk": chunk,
                "context": context_prefix,
                "enriched_chunk": enriched_chunk
            })

    return enriched_chunks

# 优化前的 chunk:
# "该政策从2024年1月1日起执行。员工需在系统中提交申请。"

# 优化后的 chunk:
# "以下内容来自《2024年公司年假管理制度》第三章'执行细则'部分：
# 该政策从2024年1月1日起执行。员工需在系统中提交申请。"
```

**Anthropic 方案的完整流程：**

1. **Contextual Embedding**：为每个 chunk 添加上下文前缀，再做 Embedding
2. **Contextual BM25**：用带上下文的 chunk 构建 BM25 索引
3. **Hybrid Search**：结合 Contextual Embedding 和 Contextual BM25
4. **Reranking**：对结果做重排序

Anthropic 的评测显示，这套方案将检索失败率降低了 67%（相比普通 RAG）。

**追问：**
- Contextual Retrieval 的成本高吗？每个 chunk 都要调 LLM
- 有没有不用 LLM 的上下文增强方案？

---

## 六、评估体系

---

### Q22: RAG 系统怎么评估？有哪些指标？⭐⭐

**答：**

RAG 系统的评估需要从**检索质量**和**生成质量**两个维度来衡量。

**检索质量指标：**

```python
# 检索质量评估
def evaluate_retrieval(queries, ground_truth_docs, retrieved_docs, k=5):
    """评估检索质量"""

    metrics = {}

    # 1. Hit Rate（命中率）：Top-K 中是否包含正确文档
    hits = 0
    for q_idx in range(len(queries)):
        gt_set = set(ground_truth_docs[q_idx])
        ret_set = set(retrieved_docs[q_idx][:k])
        if gt_set & ret_set:
            hits += 1
    metrics["hit_rate"] = hits / len(queries)

    # 2. MRR（Mean Reciprocal Rank）：第一个正确结果的排名倒数的均值
    mrr_sum = 0
    for q_idx in range(len(queries)):
        gt_set = set(ground_truth_docs[q_idx])
        for rank, doc_id in enumerate(retrieved_docs[q_idx][:k]):
            if doc_id in gt_set:
                mrr_sum += 1.0 / (rank + 1)
                break
    metrics["mrr"] = mrr_sum / len(queries)

    # 3. NDCG（Normalized Discounted Cumulative Gain）：考虑排名位置的评分
    # 排名越靠前的正确结果贡献越大

    # 4. Recall@K：Top-K 中正确文档占所有正确文档的比例

    return metrics
```

**生成质量指标：**

```python
def evaluate_generation(queries, answers, reference_answers, contexts):
    """评估生成质量"""

    metrics = {}

    # 1. Faithfulness（忠实度）：答案是否忠于检索到的上下文
    # 用 LLM 评估：答案中的每个声明是否都能在上下文中找到依据

    # 2. Answer Relevance（答案相关性）：答案是否回答了用户的问题
    # 用 LLM 评估：答案和问题的语义匹配度

    # 3. Answer Correctness（答案正确性）：答案是否正确
    # 和标准答案对比

    # 4. Context Relevance（上下文相关性）：检索到的上下文是否和问题相关
    # 用 LLM 评估：上下文中有多少内容是真正有用的

    return metrics
```

**评估工具推荐：**
- **RAGAS**：最流行的 RAG 评估框架（下一道题详解）
- **DeepEval**：支持多种评估指标
- **TruLens**：提供可视化评估面板
- **LangSmith**：LangChain 官方的追踪和评估平台

**追问：**
- 人工评估和自动评估怎么结合？
- 没有标准答案（ground truth）时怎么评估？

---

### Q23: RAGAS 框架的评估维度？⭐⭐

**答：**

RAGAS（Retrieval Augmented Generation Assessment）是最常用的 RAG 评估框架，它通过 LLM 自动评估，不需要人工标注。

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,           # 忠实度
    answer_relevancy,       # 答案相关性
    context_precision,      # 上下文精确度
    context_recall,         # 上下文召回率
)

# 准备评估数据
eval_dataset = {
    "question": ["RAG是什么？"],
    "answer": ["RAG是检索增强生成技术，通过检索外部知识来增强LLM的回答。"],
    "contexts": [["RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术..."]],
    "ground_truth": ["RAG是一种通过检索外部文档来增强大语言模型生成质量的技术"]
}

# 运行评估
result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(result)
```

**四大核心维度：**

| 维度 | 含义 | 评估方式 |
|------|------|---------|
| **Faithfulness（忠实度）** | 答案中的每句话是否都能在上下文中找到依据 | LLM 检查每个声明 |
| **Answer Relevancy（答案相关性）** | 答案是否回答了用户的问题 | 用答案反向生成问题，看和原问题的匹配度 |
| **Context Precision（上下文精确度）** | 检索到的上下文中有多少是真正相关的 | 评估相关文档的排名位置 |
| **Context Recall（上下文召回率）** | 所有相关信息是否都被检索到了 | 对比标准答案，看覆盖了多少 |

**类比理解：**
- Faithfulness 像"查重"——答案有没有"抄袭"上下文（而不是自己编的）
- Answer Relevancy 像"偏题检查"——有没有答非所问
- Context Precision 像"精准投递"——投递的内容是否都是有用的
- Context Recall 像"遗漏检查"——有没有漏掉重要信息

**追问：**
- RAGAS 的评估结果可靠吗？有什么局限？
- 如何在 CI/CD 中集成 RAGAS？

---

### Q24: 如何评估检索质量和生成质量？⭐⭐

**答：**

在实际项目中，需要建立一套完整的评估 pipeline，分别监控检索和生成。

```python
class RAGEvaluator:
    """RAG 系统完整评估 Pipeline"""

    def __init__(self, rag_system):
        self.rag = rag_system
        self.eval_llm = ChatOpenAI(model="gpt-4o")  # 用 LLM 做评估

    def full_evaluation(self, test_cases: list[dict]) -> dict:
        """完整评估流程"""
        results = {
            "retrieval_metrics": [],
            "generation_metrics": [],
            "end_to_end_metrics": []
        }

        for case in test_cases:
            query = case["question"]
            ground_truth = case["ground_truth_answer"]

            # 运行 RAG 系统
            rag_output = self.rag.run(query)

            # === 检索质量评估 ===
            retrieval_score = {
                "context_relevance": self._eval_context_relevance(query, rag_output["contexts"]),
                "context_coverage": self._eval_context_coverage(ground_truth, rag_output["contexts"]),
                "retrieval_latency": rag_output["retrieval_time"],
            }
            results["retrieval_metrics"].append(retrieval_score)

            # === 生成质量评估 ===
            generation_score = {
                "faithfulness": self._eval_faithfulness(rag_output["answer"], rag_output["contexts"]),
                "answer_relevance": self._eval_answer_relevance(query, rag_output["answer"]),
                "answer_correctness": self._eval_correctness(ground_truth, rag_output["answer"]),
                "generation_latency": rag_output["generation_time"],
            }
            results["generation_metrics"].append(generation_score)

        # 汇总统计
        return self._aggregate_metrics(results)

    def _eval_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """评估忠实度：答案中的每个声明是否能在上下文中找到依据"""
        response = self.eval_llm.invoke(f"""请评估以下答案的忠实度。

上下文：{''.join(contexts)}

答案：{answer}

请将答案拆解为独立的声明，然后判断每个声明是否能在上下文中找到支持。
输出格式：
- 声明1: [内容] → 支持/不支持
- 声明2: [内容] → 支持/不支持
...
最后给出忠实度分数（0-1）。""")

        return parse_score(response.content)

    def _eval_answer_relevance(self, question: str, answer: str) -> float:
        """评估答案相关性：答案是否回答了问题"""
        # 用答案反向生成问题，然后计算和原始问题的相似度
        response = self.eval_llm.invoke(f"""基于以下答案，生成 3 个这个答案可能在回答的问题：

答案：{answer}""")

        generated_questions = parse_questions(response.content)
        similarities = [compute_similarity(question, q) for q in generated_questions]
        return sum(similarities) / len(similarities)
```

**实战评估建议：**

1. **建立 Golden Dataset**：准备 100-200 个带标准答案的测试用例，覆盖简单/复杂/边缘场景
2. **自动化评估**：在 CI/CD 中集成 RAGAS，每次改动后自动跑评估
3. **分层监控**：
   - 检索层：监控 Hit Rate、MRR，低于阈值告警
   - 生成层：监控 Faithfulness，低于 0.8 要调查
   - 端到端：监控用户满意度（点赞/点踩比例）
4. **A/B 测试**：重要改动先在小流量上验证

**追问：**
- 评估数据集怎么构建？
- 评估结果不好时，怎么定位是检索还是生成的问题？

---

## 七、实战难题

---

### 难题1: 检索到了正确文档，但 LLM 就是不基于它回答

**现象：** 检索到的 Top-3 中明明有正确答案，但 LLM 输出的是自己的"知识"，而不是基于检索结果回答。

**原因分析：**
- Prompt 设计问题：没有明确要求"基于提供的资料回答"
- LLM 的训练偏差：模型倾向于用自己的知识回答
- 上下文太长：正确信息被"淹没"在大量无关上下文中

**解决方案：**

```python
# 优化 Prompt，明确指令
prompt = """你是一个严格的问答助手。你必须且只能基于以下参考资料来回答问题。

规则：
1. 只能使用参考资料中的信息来回答
2. 如果参考资料中没有相关信息，必须回答"根据现有资料，无法回答此问题"
3. 回答时请引用来源，格式：[来源: 文档名]
4. 不要使用你自己的知识补充

参考资料：
{context}

问题：{query}"""

# 如果还是不行，可以加 Self-Check
def self_check(answer: str, context: str, client) -> str:
    """自我检查：答案是否基于上下文"""
    check = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""请检查以下答案是否完全基于参考资料。如果不是，请修正。

参考资料：{context}
答案：{answer}

请输出修正后的答案。"""
        }]
    )
    return check.choices[0].message.content
```

---

### 难题2: 用户的 Query 表述和文档用语差距大，检索不到

**现象：** 用户问"怎么退钱"，但文档写的是"退款申请流程"，语义匹配失败。

**解决方案：**

```python
# Query 改写 Pipeline
class QueryRewriter:
    def __init__(self, client):
        self.client = client

    def rewrite(self, query: str) -> list[str]:
        """生成多个改写版本，提高检索召回率"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""请将以下用户问题改写为 3 个不同的表述方式，
使其更可能匹配到相关文档。保持语义相同，使用更正式/专业的措辞。

原始问题：{query}

输出格式：
1. [改写1]
2. [改写2]
3. [改写3]"""
            }]
        )
        return parse_list(response.choices[0].message.content)

# 还可以维护一个同义词表
SYNONYM_MAP = {
    "退钱": ["退款", "退费", "退货退款"],
    "登录": ["登入", "登陆", "登录系统"],
    "密码忘了": ["忘记密码", "密码重置", "找回密码"],
}
```

---

### 难题3: 知识库更新后，旧的向量没有删除，导致矛盾信息

**现象：** 公司年假政策更新了，但旧版本的文档还在向量库中，导致有时候检索到新政策，有时候检索到旧政策。

**解决方案：**

```python
# 文档版本管理
class DocumentVersionManager:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def update_document(self, doc_id: str, new_content: str, metadata: dict):
        """更新文档：先删除旧版本，再插入新版本"""

        # 方案1：基于 doc_id 删除旧向量
        self.vector_store.delete(where={"doc_id": doc_id})

        # 方案2：基于时间戳过滤（检索时只返回最新版本）
        metadata["version"] = datetime.now().isoformat()
        metadata["is_latest"] = True

        # 将旧版本标记为非最新
        self.vector_store.update(
            where={"doc_id": doc_id, "is_latest": True},
            update={"is_latest": False}
        )

        # 插入新版本
        chunks = split_and_embed(new_content)
        for chunk in chunks:
            chunk.metadata.update(metadata)
        self.vector_store.add(chunks)

    def search_with_version_filter(self, query: str):
        """检索时只返回最新版本"""
        return self.vector_store.search(
            query=query,
            where={"is_latest": True},  # 只检索最新版本
            top_k=5
        )
```

---

### 难题4: 生产环境中 RAG 延迟太高，用户等不及

**现象：** 完整 RAG 流程（Query 改写 → 多路检索 → Reranking → LLM 生成）要 5-8 秒，用户体验差。

**解决方案：**

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedRAGPipeline:
    """优化延迟的 RAG Pipeline"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def run(self, query: str) -> str:
        # 优化1：并行执行多路检索
        dense_task = asyncio.create_task(self._dense_search(query))
        sparse_task = asyncio.create_task(self._sparse_search(query))

        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        # 优化2：用轻量级 Reranker（如 FlashRank）
        # 而不是重量级 Cross-encoder
        results = self._fast_rerank(query, dense_results + sparse_results, top_k=5)

        # 优化3：流式输出
        answer = ""
        async for chunk in self._stream_generate(query, results):
            answer += chunk
            yield chunk  # 流式返回给前端

    # 优化4：缓存常见问题
    async def cached_run(self, query: str) -> str:
        cache_key = self._normalize_query(query)
        cached = await self.redis.get(cache_key)
        if cached:
            return cached

        answer = await self.run(query)
        await self.redis.set(cache_key, answer, ex=3600)  # 缓存 1 小时
        return answer

    # 优化5：Embedding 预计算 + 向量量化
    # 使用 PQ 或 SQ 压缩向量，减少检索时间
```

**延迟优化清单：**

| 优化手段 | 预期收益 | 实现难度 |
|---------|---------|---------|
| 并行检索 | -50% 检索延迟 | 低 |
| 轻量级 Reranker | -80% Reranking 延迟 | 低 |
| 流式输出 | 感知延迟大幅降低 | 低 |
| 结果缓存 | 命中时延迟趋近于零 | 中 |
| 向量量化 (PQ/SQ) | -30% 检索延迟，-70% 内存 | 中 |
| 减少检索范围 | -40% 检索延迟 | 中 |

---

### 难题5: 多语言/多模态混合文档的 RAG

**现象：** 知识库中同时有中文文档、英文文档、图表、代码，用户可能用任何语言提问。

**解决方案：**

```python
class MultilingualMultimodalRAG:
    """多语言多模态 RAG"""

    def __init__(self):
        # 使用多语言 Embedding 模型
        self.embedder = SentenceTransformer("BAAI/bge-m3")  # 支持 100+ 语言
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def index_document(self, doc: dict):
        """统一索引流程"""
        doc_type = doc["type"]

        if doc_type == "text":
            chunks = self._chunk_text(doc["content"])
        elif doc_type == "table":
            # 表格：同时存 Markdown 格式和自然语言描述
            chunks = self._process_table(doc["content"])
        elif doc_type == "image":
            # 图片：用多模态模型生成描述
            chunks = self._process_image(doc["content"])
        elif doc_type == "code":
            # 代码：按函数分块，保留注释
            chunks = self._process_code(doc["content"])

        # 统一 Embedding
        for chunk in chunks:
            embedding = self.embedder.encode(chunk["text"])
            self.vector_store.add(chunk["text"], embedding, chunk["metadata"])

    def search(self, query: str, source_lang: str = "auto") -> list:
        """跨语言检索"""

        # 如果用户用中文提问，也检索英文文档
        # bge-m3 等多语言模型可以跨语言匹配
        results = self.vector_store.search(query, top_k=20)

        # Reranking 时也支持跨语言
        reranked = self.reranker.rerank(query, results)
        return reranked[:10]
```

---

## 总结：RAG 系统设计 Checklist

```
□ 数据准备
  □ 文档解析（PDF、Word、HTML → 纯文本）
  □ 分块策略选择（固定/递归/语义/结构）
  □ Chunk size 和 Overlap 调优
  □ 特殊内容处理（表格、图片、代码）

□ 索引构建
  □ Embedding 模型选型（中文/多语言/长文本）
  □ 向量数据库选型（Chroma/Milvus/FAISS）
  □ 元数据设计（时间戳、来源、类型等）
  □ 索引参数调优（HNSW M/ef, IVF nlist）

□ 检索优化
  □ 混合检索（Dense + Sparse）
  □ Query 改写/扩展
  □ Reranking 重排序
  □ 多路召回融合

□ 生成优化
  □ Prompt 工程（指令明确、角色设定）
  □ 上下文压缩（去除冗余信息）
  □ Lost in the Middle 缓解
  □ 流式输出

□ 评估监控
  □ 建立 Golden Dataset
  □ RAGAS 自动评估
  □ A/B 测试框架
  □ 用户反馈收集

□ 生产化
  □ 延迟优化（并行/缓存/量化）
  □ 文档版本管理
  □ 增量索引更新
  □ 异常监控和告警
```

---

> 📌 **面试提示：** RAG 系统设计题的关键不是背概念，而是展示**你实际踩过什么坑、怎么解决的**。建议准备 2-3 个真实的优化案例，用数据说话（比如"优化后检索命中率从 72% 提升到 91%"）。
