# 19. LLM 应用数据工程

> 本章聚焦 LLM 应用开发中的数据工程问题，包括数据采集、清洗、标注、合成数据生成、数据质量评估、数据飞轮和文档处理流水线。

---

### Q: LLM 应用需要处理哪些类型的数据？⭐⭐

**答：**

LLM 应用需要处理的数据类型非常多样，主要包括以下几类：

**1. 训练数据**：包括预训练语料（网页文本、书籍、代码等）和微调数据（指令-响应对、对话数据等）。这些数据决定了模型的基础能力。

**2. 知识库数据**：用于 RAG 系统的文档数据，包括 PDF、Word、HTML、Markdown 等格式的企业内部文档、产品手册等。

**3. 用户交互数据**：用户与系统交互产生的查询日志、反馈数据（点赞/点踩）、对话历史等，是数据飞轮的核心燃料。

**4. 结构化数据**：数据库表格、API 返回的 JSON、知识图谱三元组等，需要转换为 LLM 可理解的自然语言描述。

**5. 多模态数据**：图片、音频、视频等，多模态 LLM 应用需要处理这些非文本数据。

```python
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class DataType(Enum):
    """LLM 应用数据类型枚举"""
    PRETRAIN_CORPUS = "pretrain_corpus"        # 预训练语料
    SFT_DATASET = "sft_dataset"               # 微调数据集
    RAG_DOCUMENT = "rag_document"             # RAG 知识库文档
    USER_INTERACTION = "user_interaction"      # 用户交互数据
    STRUCTURED_DATA = "structured_data"        # 结构化数据
    MULTIMODAL = "multimodal"                  # 多模态数据

@dataclass
class DataItem:
    """通用数据条目"""
    content: str
    data_type: DataType
    metadata: dict = field(default_factory=dict)
    source: Optional[str] = None
    quality_score: Optional[float] = None

class DataCatalog:
    """数据目录管理器 - 管理 LLM 应用中的各类数据"""

    def __init__(self):
        self.catalog: dict[DataType, List[DataItem]] = {dt: [] for dt in DataType}

    def register(self, item: DataItem):
        """注册数据条目到目录"""
        self.catalog[item.data_type].append(item)

    def get_stats(self) -> dict:
        """获取各类型数据统计"""
        return {
            dt.value: {
                "count": len(items),
                "avg_quality": sum(
                    i.quality_score for i in items if i.quality_score
                ) / max(len([i for i in items if i.quality_score]), 1)
            }
            for dt, items in self.catalog.items()
            if items
        }

    def load_directory(self, dir_path: str, data_type: DataType):
        """从目录批量加载文件"""
        path = Path(dir_path)
        for file_path in path.rglob("*"):
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    self.register(DataItem(
                        content=content,
                        data_type=data_type,
                        source=str(file_path),
                        metadata={"format": file_path.suffix, "size": file_path.stat().st_size}
                    ))
                except Exception as e:
                    print(f"跳过文件 {file_path}: {e}")

# 使用示例
catalog = DataCatalog()
catalog.register(DataItem(
    content="请帮我写一个 Python 函数来排序列表",
    data_type=DataType.SFT_DATASET,
    metadata={"intent": "code_generation"}
))
stats = catalog.get_stats()
print(stats)
```

不同类型的数据需要不同的处理流程。训练数据重在质量和多样性，知识库数据重在准确性和时效性，用户交互数据重在隐私保护和有效利用。

---

### Q: 如何评估训练数据的质量？⭐⭐⭐

**答：**

数据质量评估是 LLM 应用成功的关键环节。评估维度包括：**准确性**（信息是否正确）、**一致性**（相同问题答案是否一致）、**完整性**（是否缺少关键信息）、**相关性**（与任务是否相关）和**多样性**（是否覆盖足够多的场景）。

自动化评估方法包括：使用规则检测（重复、过短、乱码）、统计特征分析（长度分布、词汇多样性）和 LLM-as-Judge 模式（用强模型评估弱模型的训练数据）。实践中通常结合多种方法。

```python
import re
import hashlib
from collections import Counter
from typing import List, Dict
import numpy as np

class DataQualityEvaluator:
    """训练数据质量评估器"""

    def __init__(self, min_length: int = 10, max_length: int = 4096):
        self.min_length = min_length
        self.max_length = max_length
        self._seen_hashes: set = set()

    def evaluate(self, samples: List[Dict[str, str]]) -> Dict:
        """全面评估数据集质量"""
        results = {
            "total": len(samples),
            "issues": [],
            "scores": {},
            "details": []
        }

        for idx, sample in enumerate(samples):
            issues = []
            text = sample.get("instruction", "") + sample.get("output", "")

            # 1. 长度检测
            if len(text.strip()) < self.min_length:
                issues.append("too_short")
            if len(text) > self.max_length:
                issues.append("too_long")

            # 2. 重复检测（基于哈希）
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self._seen_hashes:
                issues.append("duplicate")
            self._seen_hashes.add(text_hash)

            # 3. 语言一致性检测
            if sample.get("instruction") and sample.get("output"):
                inst_lang = self._detect_lang(sample["instruction"])
                out_lang = self._detect_lang(sample["output"])
                if inst_lang != out_lang and inst_lang and out_lang:
                    issues.append("language_mismatch")

            # 4. 格式异常检测
            if self._has_format_issues(text):
                issues.append("format_issue")

            # 5. 信息密度评估
            diversity = self._lexical_diversity(text)
            if diversity < 0.3:
                issues.append("low_diversity")

            results["details"].append({
                "index": idx,
                "issues": issues,
                "length": len(text),
                "lexical_diversity": round(diversity, 3)
            })

        # 汇总统计
        issue_counts = Counter()
        for d in results["details"]:
            for issue in d["issues"]:
                issue_counts[issue] += 1
        results["issues"] = dict(issue_counts)
        results["scores"]["overall"] = round(
            1 - sum(len(d["issues"]) for d in results["details"]) / max(len(samples) * 5, 1), 3
        )
        return results

    def _detect_lang(self, text: str) -> str:
        """简单语言检测"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        if chinese_chars > english_chars:
            return "zh"
        elif english_chars > 0:
            return "en"
        return ""

    def _has_format_issues(self, text: str) -> bool:
        """检测格式异常"""
        patterns = [
            r'<\|.*?\|>',       # 残留特殊标记
            r'[\x00-\x08\x0b\x0c\x0e-\x1f]',  # 控制字符
            r'(.)\1{20,}',      # 过长重复字符
        ]
        return any(re.search(p, text) for p in patterns)

    def _lexical_diversity(self, text: str) -> float:
        """计算词汇多样性（类型-标记比）"""
        tokens = list(text)
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

# 使用示例
evaluator = DataQualityEvaluator()
samples = [
    {"instruction": "什么是机器学习？", "output": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习规律。"},
    {"instruction": "Hello", "output": "Hi"},  # 太短
    {"instruction": "什么是深度学习？", "output": "深度学习是机器学习的一个分支，使用多层神经网络。深度学习是机器学习的一个分支。"},
]
report = evaluator.evaluate(samples)
print(f"总样本数: {report['total']}")
print(f"问题分布: {report['issues']}")
print(f"综合质量分: {report['scores']['overall']}")
```

此外，还可以引入 LLM-as-Judge 进行更精细的质量评分，如检查答案的事实准确性、逻辑连贯性等。关键是要建立数据质量基线，持续监控数据质量变化趋势。

---

### Q: 什么是数据飞轮？如何设计？⭐⭐⭐

**答：**

数据飞轮（Data Flywheel）是指 LLM 应用在生产环境中形成"数据收集 → 模型优化 → 用户体验提升 → 更多用户使用 → 更多数据"的正反馈循环。它是 LLM 应用持续进化的核心机制。

数据飞轮的关键环节包括：（1）**数据收集**：通过用户反馈（显式点赞/点踩、隐式行为）收集高质量数据；（2）**数据筛选**：自动过滤低质量数据，保留有价值的样本；（3）**数据标注**：利用人工或 LLM 自动标注；（4）**模型迭代**：用新数据微调或更新 RAG 知识库；（5）**效果验证**：通过 A/B 测试评估改进效果。

```python
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3

class FeedbackType(Enum):
    POSITIVE = "positive"       # 用户点赞
    NEGATIVE = "negative"       # 用户点踩
    IMPLICIT_GOOD = "implicit_good"   # 用户复制/采纳回答
    IMPLICIT_BAD = "implicit_bad"     # 用户重新提问/离开

@dataclass
class UserFeedback:
    feedback_type: FeedbackType
    query: str
    response: str
    timestamp: float
    user_id: str
    session_id: str
    metadata: dict = field(default_factory=dict)

class DataFlywheel:
    """数据飞轮管理器"""

    def __init__(self, db_path: str = ":memory:"):
        self.db = sqlite3.connect(db_path)
        self._init_db()
        self.collection_threshold = 0.7   # 低质量阈值，触发数据收集
        self.min_samples_for_update = 100  # 最少多少样本才触发更新

    def _init_db(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_type TEXT,
                query TEXT,
                response TEXT,
                timestamp REAL,
                user_id TEXT,
                session_id TEXT,
                metadata TEXT,
                used_for_training INTEGER DEFAULT 0
            )
        """)
        self.db.commit()

    def collect_feedback(self, feedback: UserFeedback):
        """收集用户反馈"""
        self.db.execute(
            "INSERT INTO feedback (feedback_type, query, response, timestamp, user_id, session_id, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (feedback.feedback_type.value, feedback.query, feedback.response,
             feedback.timestamp, feedback.user_id, feedback.session_id,
             json.dumps(feedback.metadata))
        )
        self.db.commit()

    def get_training_candidates(self, limit: int = 1000) -> List[Dict]:
        """获取可用于训练的数据候选"""
        cursor = self.db.execute(
            "SELECT query, response, feedback_type, metadata FROM feedback "
            "WHERE used_for_training = 0 "
            "ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        candidates = []
        for row in cursor:
            query, response, fb_type, meta = row
            # 计算样本价值得分
            score = self._score_candidate(fb_type, json.loads(meta) if meta else {})
            candidates.append({
                "instruction": query,
                "output": response,
                "score": score,
                "feedback_type": fb_type
            })
        return sorted(candidates, key=lambda x: x["score"], reverse=True)

    def _score_candidate(self, fb_type: str, metadata: dict) -> float:
        """评估候选样本的训练价值"""
        base_scores = {
            "positive": 1.0,
            "implicit_good": 0.8,
            "negative": 0.3,
            "implicit_bad": 0.2,
        }
        score = base_scores.get(fb_type, 0.5)
        # 用户后续追问说明该问答有价值
        if metadata.get("follow_up_questions", 0) > 0:
            score += 0.1
        # 回答被编辑过，说明有改进空间但方向对
        if metadata.get("was_edited", False):
            score += 0.15
        return min(score, 1.0)

    def should_trigger_update(self) -> bool:
        """判断是否应该触发模型更新"""
        cursor = self.db.execute(
            "SELECT COUNT(*) FROM feedback WHERE used_for_training = 0"
        )
        new_count = cursor.fetchone()[0]
        return new_count >= self.min_samples_for_update

    def get_flywheel_metrics(self) -> Dict:
        """获取飞轮运转指标"""
        cursor = self.db.execute("SELECT feedback_type, COUNT(*) FROM feedback GROUP BY feedback_type")
        distribution = dict(cursor.fetchall())

        total = sum(distribution.values())
        positive = distribution.get("positive", 0) + distribution.get("implicit_good", 0)
        negative = distribution.get("negative", 0) + distribution.get("implicit_bad", 0)

        return {
            "total_feedback": total,
            "satisfaction_rate": round(positive / max(total, 1), 3),
            "needs_improvement_count": negative,
            "should_update": self.should_trigger_update(),
            "feedback_distribution": distribution
        }

# 使用示例
flywheel = DataFlywheel()
# 模拟收集反馈
flywheel.collect_feedback(UserFeedback(
    feedback_type=FeedbackType.POSITIVE,
    query="如何用 Python 实现快速排序？",
    response="def quicksort(arr): ...",
    timestamp=time.time(),
    user_id="user_001",
    session_id="session_abc"
))
metrics = flywheel.get_flywheel_metrics()
print(f"满意度: {metrics['satisfaction_rate']}")
print(f"是否需要更新: {metrics['should_update']}")
```

设计数据飞轮时需注意：建立明确的数据质量准入标准，保护用户隐私（脱敏处理），平衡数据收集的广度和深度。

---

### Q: 如何用 LLM 生成合成训练数据？⭐⭐⭐

**答：**

合成数据生成是解决高质量训练数据不足的有效方法。核心思路是利用强大的 LLM（如 GPT-4）作为"教师模型"来生成训练数据，再用这些数据微调较小的"学生模型"。常用方法包括：（1）**Self-Instruct**：从种子任务出发，让 LLM 生成新的指令-响应对；（2）**Evol-Instruct**：对已有指令进行复杂化演化；（3）**知识蒸馏**：用强模型回答问题，生成弱模型的训练数据；（4）**场景模拟**：生成特定领域的对话场景。

```python
import json
import random
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class SynthConfig:
    """合成数据配置"""
    num_samples: int = 100
    difficulty_levels: List[str] = None
    domains: List[str] = None
    languages: List[str] = None

    def __post_init__(self):
        self.difficulty_levels = self.difficulty_levels or ["easy", "medium", "hard"]
        self.domains = self.domains or ["general"]
        self.languages = self.languages or ["zh"]

class SyntheticDataGenerator:
    """合成数据生成器（演示用，实际需接入 LLM API）"""

    def __init__(self, api_client=None):
        self.api_client = api_client  # OpenAI 或其他 LLM 客户端

    def self_instruct(self, seed_tasks: List[Dict], num_new: int = 10) -> List[Dict]:
        """Self-Instruct 方法：从种子任务生成新任务"""
        system_prompt = """你是一个任务生成器。根据以下示例任务，生成全新的、不同但类似的任务。
每个任务包含 instruction（指令）和 output（期望输出）。
要求：任务多样，覆盖不同场景，避免重复。"""

        examples_text = "\n".join([
            f"任务{i+1}: {t['instruction']}\n回答: {t['output'][:100]}..."
            for i, t in enumerate(seed_tasks[:5])
        ])

        prompt = f"""参考以下示例任务：
{examples_text}

请生成 {num_new} 个新的、不同类型的任务。以 JSON 数组格式返回，每个元素包含 instruction 和 output 字段。"""

        # 实际调用 LLM API
        # response = self.api_client.chat(system_prompt, prompt)
        # return json.loads(response)
        return []  # 占位

    def evol_instruct(self, instruction: str, evolution_type: str = "deepening") -> str:
        """Evol-Instruct 方法：对指令进行演化"""
        prompts = {
            "deepening": f"请将以下问题深化，增加更多约束条件和细节要求：\n{instruction}",
            "widening": f"请将以下问题扩展，增加相关的子问题和维度：\n{instruction}",
            "reasoning": f"请将以下问题改写，要求回答者展示详细的推理过程：\n{instruction}",
            "constrained": f"请为以下问题添加具体的格式约束（如必须用表格/代码/步骤展示）：\n{instruction}",
        }
        prompt = prompts.get(evolution_type, prompts["deepening"])
        # response = self.api_client.chat("你是一个指令演化器。", prompt)
        return prompt

    def generate_domain_data(self, domain: str, config: SynthConfig) -> List[Dict]:
        """生成特定领域的合成数据"""
        templates = {
            "code": [
                ("用{lang}实现{algorithm}", "以下是{lang}实现的{algorithm}代码：\n```{lang}\n...\n```"),
                ("解释以下代码的作用：{code}", "这段代码的作用是..."),
                ("调试以下{lang}代码中的错误：{code}", "代码中的问题是...修复方案如下..."),
            ],
            "qa": [
                ("{topic}是什么？", "{topic}是指..."),
                ("{topic}和{topic2}有什么区别？", "主要区别在于..."),
                ("如何解决{problem}？", "可以通过以下步骤解决..."),
            ],
            "summarization": [
                ("请总结以下文本的要点：{text}", "主要要点包括：1. ... 2. ... 3. ..."),
            ],
        }
        templates_for_domain = templates.get(domain, templates["qa"])
        generated = []
        for i in range(config.num_samples):
            template = random.choice(templates_for_fields := templates_for_domain)
            # 实际使用中用 LLM 填充模板
            generated.append({
                "instruction": template[0],
                "output": template[1],
                "domain": domain,
                "difficulty": random.choice(config.difficulty_levels),
                "synthetic": True
            })
        return generated

    def quality_filter(self, samples: List[Dict], threshold: float = 0.7) -> List[Dict]:
        """对合成数据进行质量过滤"""
        filtered = []
        for sample in samples:
            # 规则过滤
            if len(sample.get("output", "")) < 20:
                continue
            if sample.get("instruction", "") == sample.get("output", ""):
                continue
            # 可以用 LLM 进行更精细的质量评估
            # score = self._llm_judge(sample)
            # if score >= threshold:
            #     filtered.append(sample)
            filtered.append(sample)
        return filtered

    def deduplicate(self, samples: List[Dict], threshold: float = 0.9) -> List[Dict]:
        """基于语义相似度去重"""
        unique = []
        seen_instructions = set()
        for sample in samples:
            inst = sample.get("instruction", "").strip().lower()
            if inst not in seen_instructions:
                seen_instructions.add(inst)
                unique.append(sample)
        return unique

# 使用示例
generator = SyntheticDataGenerator()
config = SynthConfig(num_samples=50, domains=["code", "qa"])
# code_data = generator.generate_domain_data("code", config)
# filtered = generator.quality_filter(code_data)

# Evol-Instruct 示例
simple_question = "什么是机器学习？"
complex_question = generator.evol_instruct(simple_question, "deepening")
print(f"原始: {simple_question}")
print(f"演化后: {complex_question}")
```

关键实践建议：（1）生成后必须进行质量过滤，剔除低质量和重复数据；（2）控制合成数据与真实数据的比例，避免分布偏移；（3）使用多样性采样确保覆盖面。

---

### Q: 数据标注有哪些策略？⭐⭐

**答：**

数据标注是 LLM 应用开发中的重要环节，主要策略包括：

**人工标注**：最可靠但成本最高。适合小规模高质量数据集。需要设计清晰的标注指南（Annotation Guidelines），进行标注者培训和一致性检查（Inter-Annotator Agreement）。

**LLM 辅助标注**：用 LLM 预标注后人工审核修正，效率提升 3-5 倍。适合中等规模标注项目。

**全自动标注**：完全由 LLM 完成，成本最低但质量需要严格验证。适合对质量要求不极高的场景或初始版本数据。

**主动学习标注**：模型识别出最不确定的样本优先标注，最大化标注价值。

```python
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum

class AnnotationLabel(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"

@dataclass
class AnnotationTask:
    task_id: str
    text: str
    labels: List[AnnotationLabel] = field(default_factory=list)
    annotators: List[str] = field(default_factory=list)
    llm_label: Optional[AnnotationLabel] = None
    confidence: float = 0.0
    is_reviewed: bool = False

class AnnotationPipeline:
    """数据标注流水线"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.tasks: Dict[str, AnnotationTask] = {}
        self.annotation_guidelines = ""

    def add_tasks(self, texts: List[str]):
        """添加待标注任务"""
        for i, text in enumerate(texts):
            task_id = f"task_{i:04d}"
            self.tasks[task_id] = AnnotationTask(task_id=task_id, text=text)

    def llm_auto_annotate(self, batch_size: int = 10) -> Dict:
        """LLM 自动标注"""
        unannotated = [t for t in self.tasks.values() if not t.llm_label]
        annotated_count = 0

        for task in unannotated[:batch_size]:
            # 模拟 LLM 标注
            prompt = f"""请对以下文本进行情感分类，选择：positive, negative, neutral。
文本：{task.text}
请只返回标签名称。"""
            # label_str = self.llm_client.complete(prompt)
            # 模拟结果
            label_str = random.choice(["positive", "negative", "neutral"])
            task.llm_label = AnnotationLabel(label_str)
            task.confidence = round(random.uniform(0.6, 0.99), 2)
            annotated_count += 1

        return {
            "annotated": annotated_count,
            "remaining": len(unannotated) - annotated_count
        }

    def get_uncertain_samples(self, top_k: int = 20) -> List[AnnotationTask]:
        """主动学习：获取最不确定的样本优先人工标注"""
        uncertain = sorted(
            [t for t in self.tasks.values() if t.llm_label and not t.is_reviewed],
            key=lambda t: abs(t.confidence - 0.5)  # 置信度接近 0.5 最不确定
        )
        return uncertain[:top_k]

    def human_review(self, task_id: str, annotator: str, label: AnnotationLabel) -> bool:
        """人工审核标注"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        task.labels.append(label)
        task.annotators.append(annotator)
        task.is_reviewed = True
        return True

    def compute_agreement(self) -> Dict:
        """计算标注一致性"""
        reviewed = [t for t in self.tasks.values() if t.is_reviewed]
        if not reviewed:
            return {"agreement_rate": 0, "count": 0}

        agree_with_llm = sum(
            1 for t in reviewed
            if t.llm_label in t.labels
        )
        return {
            "total_reviewed": len(reviewed),
            "agreement_rate": round(agree_with_llm / len(reviewed), 3),
            "correction_rate": round(1 - agree_with_llm / len(reviewed), 3)
        }

    def get_stats(self) -> Dict:
        total = len(self.tasks)
        auto_annotated = sum(1 for t in self.tasks.values() if t.llm_label)
        human_reviewed = sum(1 for t in self.tasks.values() if t.is_reviewed)
        return {
            "total": total,
            "auto_annotated": auto_annotated,
            "human_reviewed": human_reviewed,
            "coverage": round(auto_annotated / max(total, 1), 3)
        }

# 使用示例
pipeline = AnnotationPipeline()
texts = [
    "这个产品非常好用，强烈推荐！",
    "服务态度太差了，再也不来了。",
    "今天天气不错。",
] * 10
pipeline.add_tasks(texts)
auto_result = pipeline.llm_auto_annotate(batch_size=30)
print(f"自动标注: {auto_result}")

uncertain = pipeline.get_uncertain_samples(top_k=5)
print(f"最不确定的样本: {[t.task_id for t in uncertain]}")
```

最佳实践：建立完善的标注规范文档，使用多标注者 + 投票机制保证质量，定期校准标注标准。

---

### Q: 如何构建文档处理流水线？⭐⭐⭐

**答：**

文档处理流水线是 RAG 系统的基础。典型的流水线包括：**文档解析**（提取文本和结构信息）→ **文本分块**（按语义或固定大小切分）→ **元数据提取**（标题、作者、时间等）→ **向量化**（Embedding）→ **索引存储**（写入向量数据库）。

关键挑战包括：处理多种文档格式（PDF、Word、HTML）、保留文档结构（标题层级、表格、图片）、合理分块（避免切断语义完整性）、增量更新（只处理变化的文档）。

```python
import hashlib
import re
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DocumentChunk:
    """文档分块"""
    chunk_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if not self.chunk_id:
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]
            self.chunk_id = f"chunk_{content_hash}"

class DocumentProcessor:
    """文档处理流水线"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_file(self, file_path: str) -> List[DocumentChunk]:
        """处理单个文件"""
        path = Path(file_path)
        text = self._extract_text(path)
        metadata = self._extract_metadata(path, text)
        chunks = self._split_text(text, metadata)
        return chunks

    def _extract_text(self, path: Path) -> str:
        """提取文本内容"""
        suffix = path.suffix.lower()
        if suffix == ".txt" or suffix == ".md":
            return path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            return self._extract_pdf(path)
        elif suffix == ".html":
            return self._extract_html(path)
        else:
            try:
                return path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return ""

    def _extract_pdf(self, path: Path) -> str:
        """提取 PDF 文本（示例，实际用 PyPDF2 或 pdfplumber）"""
        # import pdfplumber
        # with pdfplumber.open(path) as pdf:
        #     return "\n".join(page.extract_text() or "" for page in pdf.pages)
        return f"[PDF 内容: {path.name}]"

    def _extract_html(self, path: Path) -> str:
        """提取 HTML 文本"""
        # from bs4 import BeautifulSoup
        # soup = BeautifulSoup(path.read_text(), "html.parser")
        # return soup.get_text(separator="\n", strip=True)
        return f"[HTML 内容: {path.name}]"

    def _extract_metadata(self, path: Path, text: str) -> dict:
        """提取文档元数据"""
        metadata = {
            "source": str(path),
            "filename": path.name,
            "format": path.suffix,
            "size": path.stat().st_size,
        }
        # 尝试从 Markdown 提取标题
        title_match = re.match(r'^#\s+(.+)', text)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        # 提取可能的章节结构
        headings = re.findall(r'^(#{1,6})\s+(.+)', text, re.MULTILINE)
        if headings:
            metadata["sections"] = [h[1] for h in headings]
        return metadata

    def _split_text(self, text: str, base_metadata: dict) -> List[DocumentChunk]:
        """智能文本分块"""
        chunks = []
        # 优先按段落/章节分割
        sections = re.split(r'\n{2,}', text)
        current_chunk = ""
        chunk_index = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # 如果单个 section 就超长，按句子进一步分割
            if len(section) > self.chunk_size:
                sentences = re.split(r'(?<=[。！？.!?])\s*', section)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > self.chunk_size:
                        if current_chunk:
                            chunks.append(DocumentChunk(
                                chunk_id="",
                                content=current_chunk.strip(),
                                metadata={**base_metadata, "chunk_index": chunk_index}
                            ))
                            chunk_index += 1
                            # 保留 overlap
                            current_chunk = current_chunk[-self.chunk_overlap:] + sentence
                        else:
                            current_chunk = sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
            else:
                if len(current_chunk) + len(section) + 1 > self.chunk_size:
                    if current_chunk:
                        chunks.append(DocumentChunk(
                            chunk_id="",
                            content=current_chunk.strip(),
                            metadata={**base_metadata, "chunk_index": chunk_index}
                        ))
                        chunk_index += 1
                    current_chunk = section
                else:
                    current_chunk += "\n\n" + section if current_chunk else section

        # 处理最后一个 chunk
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                chunk_id="",
                content=current_chunk.strip(),
                metadata={**base_metadata, "chunk_index": chunk_index}
            ))
        return chunks

class PipelineManager:
    """流水线管理器 - 支持批量处理和增量更新"""

    def __init__(self, processor: DocumentProcessor):
        self.processor = processor
        self.processed_hashes: Dict[str, str] = {}  # path -> content_hash

    def process_directory(self, dir_path: str) -> List[DocumentChunk]:
        """批量处理目录下所有文档"""
        all_chunks = []
        for file_path in Path(dir_path).rglob("*"):
            if file_path.is_file() and file_path.suffix in {".txt", ".md", ".pdf", ".html"}:
                content_hash = self._file_hash(file_path)
                # 增量处理：跳过未变化的文件
                if self.processed_hashes.get(str(file_path)) == content_hash:
                    continue
                chunks = self.processor.process_file(str(file_path))
                all_chunks.extend(chunks)
                self.processed_hashes[str(file_path)] = content_hash
        return all_chunks

    def _file_hash(self, path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()

# 使用示例
processor = DocumentProcessor(chunk_size=512, chunk_overlap=64)
manager = PipelineManager(processor)
# chunks = manager.process_directory("/path/to/documents")
# print(f"生成 {len(chunks)} 个分块")

# 单文件处理演示
demo_chunks = processor._split_text(
    "# 机器学习基础\n\n机器学习是人工智能的重要分支。\n\n## 监督学习\n\n监督学习使用标注数据训练模型。\n\n## 无监督学习\n\n无监督学习不需要标注数据。",
    {"source": "demo.md", "format": ".md"}
)
for chunk in demo_chunks:
    print(f"[{chunk.chunk_id}] {chunk.metadata.get('chunk_index')}: {chunk.content[:50]}...")
```

建议使用 LangChain 或 LlamaIndex 等框架的文档加载器和分块器，它们内置了丰富的文档格式支持。

---

### Q: 如何检测和处理数据中的偏见？⭐⭐

**答：**

数据偏见会严重影响 LLM 应用的公平性和可靠性。常见偏见类型包括：**性别偏见**（职业与性别刻板关联）、**文化偏见**（西方中心视角）、**选择偏见**（训练数据分布不均）、**确认偏见**（倾向于证实已有观点）。

检测方法包括：统计分析（关键词频率、分布不均衡检测）、探针测试（用特定提示词测试模型偏见）和反事实评估（替换敏感属性后看输出是否变化）。处理方法包括：数据平衡采样、偏见标注与过滤、对抗训练和后处理去偏。

```python
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass

@dataclass
class BiasReport:
    """偏见检测报告"""
    bias_type: str
    severity: str       # low, medium, high
    examples: List[str]
    description: str
    suggestion: str

class BiasDetector:
    """数据偏见检测器"""

    def __init__(self):
        # 敏感属性词典
        self.gender_terms = {
            "male": {"他", "先生", "男人", "父亲", "儿子", "he", "him", "his", "man", "father"},
            "female": {"她", "女士", "女人", "母亲", "女儿", "she", "her", "hers", "woman", "mother"},
        }
        self.stereotypical_associations = {
            "male": {"工程师", "程序员", "司机", "CEO", "科学家", "engineer", "developer", "CEO"},
            "female": {"护士", "教师", "秘书", "保姆", "nurse", "teacher", "secretary"},
        }
        # 文化偏见关键词
        self.cultural_markers = {
            "western": {"Christmas", "Thanksgiving", "Halloween", "英语"},
            "chinese": {"春节", "中秋节", "端午节", "中文"},
        }

    def detect_gender_bias(self, texts: List[str]) -> BiasReport:
        """检测性别偏见"""
        gender_counts = Counter()
        stereotype_examples = []

        for text in texts:
            text_lower = text.lower()
            for gender, terms in self.gender_terms.items():
                for term in terms:
                    if term in text_lower:
                        gender_counts[gender] += 1

            # 检测刻板印象关联
            for gender, stereotypes in self.stereotypical_associations.items():
                for stereotype in stereotypes:
                    opposite_gender = "female" if gender == "male" else "male"
                    for term in self.gender_terms.get(gender, set()):
                        if term in text_lower and stereotype in text_lower:
                            stereotype_examples.append(
                                f"发现关联: '{term}' ↔ '{stereotype}'"
                            )

        total = sum(gender_counts.values()) or 1
        imbalance = abs(gender_counts["male"] - gender_counts["female"]) / total
        severity = "high" if imbalance > 0.5 else "medium" if imbalance > 0.3 else "low"

        return BiasReport(
            bias_type="gender_bias",
            severity=severity,
            examples=stereotype_examples[:10],
            description=f"性别提及分布: 男性{gender_counts['male']}次, 女性{gender_counts['female']}次, "
                        f"不平衡度: {imbalance:.2%}",
            suggestion="建议平衡不同性别的数据比例，检查职业-性别关联的刻板印象"
        )

    def detect_distribution_bias(self, labels: List[str]) -> BiasReport:
        """检测标签分布偏见"""
        counter = Counter(labels)
        total = len(labels)
        distribution = {k: v / total for k, v in counter.most_common()}

        # 计算基尼系数作为不均衡度量
        sorted_freqs = sorted(distribution.values())
        n = len(sorted_freqs)
        gini = sum((2 * (i + 1) - n - 1) * freq for i, freq in enumerate(sorted_freqs)) / (n * sum(sorted_freqs))

        severity = "high" if gini > 0.6 else "medium" if gini > 0.3 else "low"
        minority_classes = [k for k, v in distribution.items() if v < 0.05]

        return BiasReport(
            bias_type="distribution_bias",
            severity=severity,
            examples=[f"类 '{k}' 占比仅 {v:.2%}" for k, v in distribution.items() if v < 0.05],
            description=f"标签分布不均匀，基尼系数: {gini:.3f}",
            suggestion=f"建议对少数类进行过采样或对多数类进行欠采样。少数类: {minority_classes}"
        )

    def counterfactual_test(self, text: str, replacements: Dict[str, str]) -> List[Dict]:
        """反事实测试：替换敏感属性后对比"""
        results = [{"text": text, "variant": "original"}]
        for original, replacement in replacements.items():
            modified = text.replace(original, replacement)
            if modified != text:
                results.append({
                    "text": modified,
                    "variant": f"replaced '{original}' → '{replacement}'"
                })
        return results

class BiasMitigator:
    """偏见缓解器"""

    @staticmethod
    def balance_sampling(data: List[Dict], label_key: str = "label") -> List[Dict]:
        """平衡采样：对少数类过采样"""
        label_groups = defaultdict(list)
        for item in data:
            label_groups[item[label_key]].append(item)

        max_count = max(len(group) for group in label_groups.values())
        balanced = []
        for label, group in label_groups.items():
            # 循环采样到最大类数量
            times = max_count // len(group)
            remainder = max_count % len(group)
            balanced.extend(group * times + group[:remainder])

        return balanced

    @staticmethod
    def add_diversity_markers(data: List[Dict]) -> List[Dict]:
        """为数据添加多样性标记，提示模型注意公平"""
        enhanced = []
        for item in data:
            enhanced_item = item.copy()
            enhanced_item["instructions_prefix"] = "请确保回答客观、公正，避免刻板印象。"
            enhanced.append(enhanced_item)
        return enhanced

# 使用示例
detector = BiasDetector()
texts = [
    "他是一名优秀的工程师，她是一名温柔的护士",
    "这位先生是公司的CEO，那位女士是他的秘书",
    "男程序员写代码，女教师教书",
]
report = detector.detect_gender_bias(texts)
print(f"偏见类型: {report.bias_type}")
print(f"严重程度: {report.severity}")
print(f"描述: {report.description}")
print(f"建议: {report.suggestion}")

# 分布偏见检测
labels = ["positive"] * 80 + ["negative"] * 15 + ["neutral"] * 5
dist_report = detector.detect_distribution_bias(labels)
print(f"\n分布偏见: {dist_report.description}")
print(f"少数类: {dist_report.examples}")
```

建议在数据管道中集成偏见检测作为质量门禁，定期审查模型输出中的偏见表现。

---

### Q: 知识库构建的最佳实践？⭐⭐⭐

**答：**

高质量知识库是 RAG 系统成功的基础。构建最佳实践涵盖以下方面：

**1. 数据准备阶段**：选择权威、可靠的数据源；建立数据更新机制，确保时效性；去除冗余和矛盾内容。

**2. 文档处理阶段**：使用适合的分块策略（语义分块优于固定大小分块）；保留文档层级结构和元数据信息；对表格、代码块特殊处理。

**3. 索引构建阶段**：选择合适的 Embedding 模型（考虑语言、领域适配）；建立多级索引（向量索引 + 关键词索引 + 知识图谱）；设置合理的相似度检索参数。

**4. 检索优化阶段**：实现混合检索（语义 + 关键词）；支持查询改写和扩展；使用重排序（Reranker）提升结果质量。

```python
import hashlib
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class KBDocument:
    """知识库文档"""
    doc_id: str
    title: str
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_count: int = 0
    last_updated: Optional[str] = None

@dataclass
class KBChunk:
    """知识库分块"""
    chunk_id: str
    doc_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: dict = field(default_factory=dict)

class KnowledgeBaseBuilder:
    """知识库构建器"""

    def __init__(self, embedding_model=None, chunk_size: int = 512):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.documents: Dict[str, KBDocument] = {}
        self.chunks: Dict[str, KBChunk] = {}
        self.index: Dict[str, List[str]] = {}  # 简易倒排索引

    def add_document(self, title: str, content: str, metadata: dict = None) -> str:
        """添加文档到知识库"""
        doc_id = hashlib.md5(content.encode()).hexdigest()[:16]
        if doc_id in self.documents:
            return doc_id

        doc = KBDocument(
            doc_id=doc_id,
            title=title,
            content=content,
            metadata=metadata or {},
            last_updated=self._current_time()
        )
        self.documents[doc_id] = doc

        # 分块
        chunks = self._intelligent_chunk(content, doc_id, metadata)
        doc.chunk_count = len(chunks)

        # 建立索引
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            self._update_index(chunk)

        return doc_id

    def _intelligent_chunk(self, content: str, doc_id: str, metadata: dict) -> List[KBChunk]:
        """智能分块：保留语义完整性"""
        chunks = []
        # 按 Markdown 标题结构分割
        sections = self._split_by_headers(content)

        for section in sections:
            if len(section["content"]) <= self.chunk_size:
                chunk = self._create_chunk(section["content"], doc_id, {
                    **(metadata or {}),
                    "section_title": section.get("title", ""),
                    "level": section.get("level", 0)
                })
                chunks.append(chunk)
            else:
                # 长段落按句子边界进一步分割
                sub_chunks = self._split_by_sentences(section["content"])
                for i, text in enumerate(sub_chunks):
                    chunk = self._create_chunk(text, doc_id, {
                        **(metadata or {}),
                        "section_title": section.get("title", ""),
                        "part": i
                    })
                    chunks.append(chunk)
        return chunks

    def _split_by_headers(self, content: str) -> List[Dict]:
        """按 Markdown 标题分割"""
        import re
        parts = re.split(r'(^#{1,6}\s+.+$)', content, flags=re.MULTILINE)
        sections = []
        current_title = ""
        current_level = 0

        for part in parts:
            header_match = re.match(r'^(#{1,6})\s+(.+)', part)
            if header_match:
                current_level = len(header_match.group(1))
                current_title = header_match.group(2).strip()
            elif part.strip():
                sections.append({
                    "title": current_title,
                    "level": current_level,
                    "content": part.strip()
                })
        return sections

    def _split_by_sentences(self, text: str) -> List[str]:
        """按句子边界分割，保持 chunk_size 内"""
        import re
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > self.chunk_size and current:
                chunks.append(current.strip())
                current = sent
            else:
                current += " " + sent if current else sent
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _create_chunk(self, content: str, doc_id: str, metadata: dict) -> KBChunk:
        chunk_id = hashlib.md5(content.encode()).hexdigest()[:16]
        # 实际中调用 embedding 模型
        embedding = None
        # if self.embedding_model:
        #     embedding = self.embedding_model.encode(content)
        return KBChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata
        )

    def _update_index(self, chunk: KBChunk):
        """更新倒排索引"""
        tokens = set(chunk.content.lower().split())
        for token in tokens:
            if token not in self.index:
                self.index[token] = []
            self.index[token].append(chunk.chunk_id)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[KBChunk, float]]:
        """混合检索：关键词 + 向量"""
        # 关键词检索
        query_tokens = set(query.lower().split())
        candidate_ids: Dict[str, int] = {}
        for token in query_tokens:
            for chunk_id in self.index.get(token, []):
                candidate_ids[chunk_id] = candidate_ids.get(chunk_id, 0) + 1

        # 按匹配词数排序
        scored = [
            (self.chunks[cid], count / max(len(query_tokens), 1))
            for cid, count in candidate_ids.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _current_time(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()

    def get_stats(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "avg_chunks_per_doc": round(
                len(self.chunks) / max(len(self.documents), 1), 1
            ),
            "index_terms": len(self.index)
        }

# 使用示例
kb = KnowledgeBaseBuilder(chunk_size=256)

# 添加文档
kb.add_document(
    title="Python 基础教程",
    content="# Python 入门\n\nPython 是一门简洁优雅的编程语言。\n\n## 安装\n\n从 python.org 下载安装包。\n\n## 基本语法\n\n使用缩进表示代码块，变量无需声明类型。Python 支持多种数据类型包括列表、字典和集合。",
    metadata={"category": "tutorial", "author": "team"}
)

stats = kb.get_stats()
print(f"知识库统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")

results = kb.search("Python 编程语言")
for chunk, score in results:
    print(f"[{score:.2f}] {chunk.content[:60]}...")
```

知识库维护建议：定期更新过期内容，监控检索命中率，收集用户对检索结果的反馈来持续优化分块和索引策略。

---

### Q: 如何实现数据去重和近似去重？⭐⭐

**答：**

数据去重是保证训练数据质量的重要步骤。完全相同的重复数据可以通过哈希精确去重，但更多情况下需要**近似去重**（Near-Duplicate Detection），因为两个文档可能只在个别词句上有差异。

常用方法包括：（1）**MinHash + LSH**：高效计算文档集合的 Jaccard 相似度，适用于大规模数据；（2）**SimHash**：将文档映射为固定长度的指纹，通过汉明距离判断相似度；（3）**基于 Embedding 的语义去重**：计算语义向量的余弦相似度。

```python
import hashlib
import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict

class MinHashDeduplicator:
    """基于 MinHash + LSH 的近似去重器"""

    def __init__(self, num_hashes: int = 128, threshold: float = 0.8, num_bands: int = 16):
        self.num_hashes = num_hashes
        self.threshold = threshold
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.hash_functions = self._generate_hash_functions()

    def _generate_hash_functions(self) -> List:
        """生成 MinHash 哈希函数族（模拟）"""
        import random
        funcs = []
        for _ in range(self.num_hashes):
            a = random.randint(1, 2**31 - 1)
            b = random.randint(0, 2**31 - 1)
            funcs.append(lambda x, a=a, b=b: (a * hash(x) + b) % (2**31 - 1))
        return funcs

    def _get_shingles(self, text: str, k: int = 5) -> Set[str]:
        """将文本转为 k-gram 集合"""
        text = re.sub(r'\s+', ' ', text.strip().lower())
        return {text[i:i+k] for i in range(max(len(text) - k + 1, 1))}

    def _compute_minhash(self, shingles: Set[str]) -> List[int]:
        """计算 MinHash 签名"""
        signature = []
        for hf in self.hash_functions:
            min_val = min(hf(s) for s in shingles) if shingles else 0
            signature.append(min_val)
        return signature

    def _lsh_buckets(self, signature: List[int]) -> List[str]:
        """LSH 分桶"""
        buckets = []
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = tuple(signature[start:end])
            bucket_id = f"band_{band_idx}_{hash(band)}"
            buckets.append(bucket_id)
        return buckets

    def deduplicate(self, documents: List[Dict[str, str]]) -> Tuple[List[Dict], List[Tuple[int, int]]]:
        """执行去重，返回去重结果和重复对"""
        signatures = []
        bucket_to_docs = defaultdict(list)
        duplicates = set()

        # 计算签名并分桶
        for idx, doc in enumerate(documents):
            shingles = self._get_shingles(doc.get("text", ""))
            sig = self._compute_minhash(shingles)
            signatures.append(sig)

            buckets = self._lsh_buckets(sig)
            for bucket_id in buckets:
                bucket_to_docs[bucket_id].append(idx)

        # 查找候选重复对
        candidate_pairs = set()
        for bucket_id, doc_ids in bucket_to_docs.items():
            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    candidate_pairs.add((min(doc_ids[i], doc_ids[j]),
                                         max(doc_ids[i], doc_ids[j])))

        # 验证候选对
        duplicate_pairs = []
        for i, j in candidate_pairs:
            similarity = self._jaccard_similarity(signatures[i], signatures[j])
            if similarity >= self.threshold:
                duplicates.add(j)  # 保留较早的，标记较晚的为重复
                duplicate_pairs.append((i, j, round(similarity, 3)))

        # 构建结果
        unique_docs = [doc for idx, doc in enumerate(documents) if idx not in duplicates]
        return unique_docs, duplicate_pairs

    def _jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """估算 Jaccard 相似度"""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

class ExactDeduplicator:
    """精确去重器（基于内容哈希）"""

    def deduplicate(self, documents: List[Dict[str, str]]) -> Tuple[List[Dict], int]:
        seen: Set[str] = set()
        unique = []
        dup_count = 0

        for doc in documents:
            # 归一化后计算哈希
            normalized = re.sub(r'\s+', ' ', doc.get("text", "").strip().lower())
            content_hash = hashlib.md5(normalized.encode()).hexdigest()

            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(doc)
            else:
                dup_count += 1

        return unique, dup_count

# 使用示例
docs = [
    {"text": "机器学习是人工智能的一个分支，通过算法从数据中学习。"},
    {"text": "机器学习是人工智能的一个分支，通过算法从数据中学习规律。"},  # 近似重复
    {"text": "深度学习使用多层神经网络进行特征学习和表示学习。"},
    {"text": "机器学习是人工智能的一个分支，通过算法从数据中学习。"},  # 完全重复
]

# 精确去重
exact = ExactDeduplicator()
unique_exact, dup_count = exact.deduplicate(docs)
print(f"精确去重: {len(docs)} → {len(unique_exact)} (去除 {dup_count} 个)")

# 近似去重
minhash = MinHashDeduplicator(num_hashes=64, threshold=0.7, num_bands=8)
unique_approx, dup_pairs = minhash.deduplicate(docs)
print(f"近似去重: {len(docs)} → {len(unique_approx)}")
print(f"发现重复对: {dup_pairs}")
```

对于大规模数据（百万级文档），推荐使用 MinHash + LSH，时间复杂度接近 O(n)。对于小规模数据，直接使用 Embedding 余弦相似度即可。去重后建议人工抽检确认效果。

---

> **本章小结**：LLM 应用的数据工程覆盖从数据采集、清洗、标注到质量评估的全流程。数据飞轮是应用持续进化的核心驱动力，合成数据和智能标注是解决数据瓶颈的关键手段。构建高质量知识库需要在文档处理、分块策略和索引优化上下功夫。偏见检测和数据去重则是保障数据质量的重要环节。
