# 27. 知识图谱与GraphRAG

> 知识图谱(Knowledge Graph)以图结构组织实体与关系，GraphRAG结合图谱与大语言模型实现知识增强的检索生成，是RAG系统从"文本片段匹配"向"语义关系推理"演进的关键方向。
---

## 1. 什么是知识图谱？和传统数据库有什么区别？⭐⭐

### 知识图谱定义

知识图谱是一种以**图结构**表示知识的语义网络，核心由**三元组(Subject-Predicate-Object)**组成：

```
(爱因斯坦, 出生于, 乌尔姆)
(爱因斯坦, 提出了, 相对论)
(相对论, 属于, 物理学)
```

### 与传统数据库的核心区别

| 维度 | 关系型数据库 | 知识图谱 |
|------|------------|---------|
| 数据模型 | 二维表(行/列) | 图(节点/边/属性) |
| Schema | 严格预定义 | 灵活可扩展 |
| 查询方式 | SQL JOIN | 图遍历/模式匹配 |
| 关系处理 | 通过外键关联 | 关系是一等公民 |
| 语义能力 | 无 | 支持推理和本体 |
| 适用场景 | 结构化事务 | 复杂关联关系 |

### 代码示例：从关系型DB到知识图谱

```python
# === 关系型数据库视角 ===
import sqlite3

conn = sqlite3.connect(":memory:")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE persons (
    id INT PRIMARY KEY, name TEXT, birth_place TEXT
)
""")
cursor.execute("""
CREATE TABLE relations (
    subject_id INT, predicate TEXT, object_id INT,
    FOREIGN KEY(subject_id) REFERENCES persons(id),
    FOREIGN KEY(object_id) REFERENCES persons(id)
)""")

# 查询: "谁和爱因斯坦出生于同一个城市？" 需要复杂JOIN
cursor.execute("""
SELECT p2.name FROM persons p1
JOIN persons p2 ON p1.birth_place = p2.birth_place
WHERE p1.name = '爱因斯坦' AND p2.name != '爱因斯坦'
""")

# === 知识图谱视角 ===
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def create_kg(tx):
    tx.run("""
    CREATE (e:Person {name: '爱因斯坦', birth_place: '乌尔姆'})
    CREATE (m:Person {name: '闵可夫斯基', birth_place: '乌尔姆'})
    CREATE (r:Theory {name: '相对论'})
    CREATE (e)-[:BORN_IN]->(bp:City {name: '乌尔姆'})
    CREATE (m)-[:BORN_IN]->(bp)
    CREATE (e)-[:PROPOSED]->(r)
    """)

with driver.session() as session:
    session.execute_write(create_kg)

    # 图遍历: 直观且高效
    result = session.run("""
    MATCH (e:Person {name: '爱因斯坦'})-[:BORN_IN]->(city)<-[:BORN_IN]-(other)
    RETURN other.name
    """)
    for record in result:
        print(record["other.name"])  # 闵可夫斯基
```

### 知识图谱的技术栈

```python
# 知识图谱核心组件
class KnowledgeGraph:
    """知识图谱核心架构"""

    def __init__(self):
        self.components = {
            "本体层(Ontology)": {
                "作用": "定义概念、属性、关系的schema",
                "工具": "OWL, RDF Schema, SKOS",
                "示例": "定义Person类有name属性, 可以PROPOSED Theory"
            },
            "数据层(Data)": {
                "作用": "存储具体的实体和三元组",
                "工具": "Neo4j, NebulaGraph, JanusGraph",
                "示例": "(爱因斯坦)-[:PROPOSED]->(相对论)"
            },
            "推理层(Reasoning)": {
                "作用": "基于规则推导隐含知识",
                "工具": "OWL推理机, 自定义规则引擎",
                "示例": "A是B的父亲, B是C的父亲 → A是C的祖父"
            }
        }

    def rdf_triple_example(self):
        """RDF三元组的标准表示"""
        # Turtle格式
        turtle = """
        @prefix ex: <http://example.org/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

        ex:Einstein rdf:type ex:Person ;
                    ex:name "爱因斯坦" ;
                    ex:birthPlace ex:Ulm ;
                    ex:proposed ex:Relativity .

        ex:Relativity rdf:type ex:Theory ;
                      ex:field ex:Physics .
        """
        return turtle
```

---

## 2. 如何用LLM自动构建知识图谱？⭐⭐⭐

### LLM构建知识图谱的流程

```
原始文档 → 文本分块 → 实体/关系抽取 → 实体消歧 → 图谱入库
```

### Microsoft GraphRAG的构建方法

```python
import openai
from typing import List, Dict, Tuple
import json
import re

class LLMKnowledgeGraphBuilder:
    """使用LLM从文本自动构建知识图谱"""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = openai.OpenAI()

    def extract_entities_relations(self, text: str) -> Dict:
        """从文本中抽取实体和关系"""

        prompt = """从以下文本中提取实体和关系。

要求：
1. 实体: 提取人物、组织、地点、概念、事件等
2. 关系: 提取实体间的语义关系
3. 每个实体包含: name, type, description
4. 每个关系包含: source, target, relation_type, description

输出JSON格式:
{
  "entities": [
    {"name": "...", "type": "PERSON|ORG|LOCATION|CONCEPT|EVENT", "description": "..."}
  ],
  "relations": [
    {"source": "...", "target": "...", "relation_type": "...", "description": "..."}
  ]
}

文本:
{text}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt.format(text=text)}],
            response_format={"type": "json_object"},
            temperature=0
        )

        return json.loads(response.choices[0].message.content)

    def summarize_entity(self, entity_name: str, descriptions: List[str]) -> str:
        """对同一实体的多次描述进行汇总"""
        prompt = f"""将以下关于"{entity_name}"的多段描述合并为一段简洁的摘要(不超过100字):

{chr(10).join(f'- {d}' for d in descriptions)}

摘要:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def process_document(self, text: str, chunk_size: int = 1500) -> Dict:
        """处理完整文档, 分块抽取并合并"""
        # 分块
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        all_entities = {}  # name -> {type, descriptions}
        all_relations = []  # (source, target, type, desc)

        for i, chunk in enumerate(chunks):
            print(f"处理第 {i+1}/{len(chunks)} 块...")
            result = self.extract_entities_relations(chunk)

            # 合并实体
            for entity in result.get("entities", []):
                name = entity["name"]
                if name in all_entities:
                    all_entities[name]["descriptions"].append(entity["description"])
                else:
                    all_entities[name] = {
                        "type": entity["type"],
                        "descriptions": [entity["description"]]
                    }

            # 收集关系
            for rel in result.get("relations", []):
                all_relations.append(rel)

        # 为每个实体生成统一描述
        for name, info in all_entities.items():
            if len(info["descriptions"]) > 1:
                info["summary"] = self.summarize_entity(name, info["descriptions"])
            else:
                info["summary"] = info["descriptions"][0]

        return {
            "entities": {k: {"type": v["type"], "summary": v["summary"]}
                        for k, v in all_entities.items()},
            "relations": all_relations
        }


class KGNeo4jInserter:
    """将抽取结果写入Neo4j"""

    def __init__(self, uri, user, password):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def insert_graph(self, graph_data: Dict):
        entities = graph_data["entities"]
        relations = graph_data["relations"]

        with self.driver.session() as session:
            # 创建实体节点
            for name, info in entities.items():
                session.run("""
                    MERGE (e:{type} {{name: $name}})
                    SET e.description = $description
                """.format(type=info["type"]),
                    name=name,
                    description=info["summary"]
                )

            # 创建关系
            for rel in relations:
                session.run("""
                    MATCH (a {{name: $source}})
                    MATCH (b {{name: $target}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r.description = $description
                """.format(rel_type=rel["relation_type"].upper().replace(" ", "_")),
                    source=rel["source"],
                    target=rel["target"],
                    description=rel.get("description", "")
                )

    def create_vector_index(self):
        """创建实体描述的向量索引(用于混合检索)"""
        with self.driver.session() as session:
            session.run("""
                CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                FOR (e:Entity) ON (e.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }}
            """)

    def close(self):
        self.driver.close()


# 完整使用流程
def build_knowledge_graph():
    # 1. 抽取
    builder = LLMKnowledgeGraphBuilder(model="gpt-4o-mini")

    doc = """
    爱因斯坦出生于德国乌尔姆, 后移居瑞士苏黎世。
    他在苏黎世联邦理工学院学习物理, 1905年发表了狭义相对论。
    普朗克是量子力学的先驱, 他非常欣赏爱因斯坦的光电效应理论。
    玻尔提出了原子的量子模型, 与爱因斯坦就量子力学的完备性进行了著名的辩论。
    """

    graph_data = builder.process_document(doc)
    print(f"抽取到 {len(graph_data['entities'])} 个实体, "
          f"{len(graph_data['relations'])} 条关系")

    # 2. 入库
    inserter = KGNeo4jInserter("bolt://localhost:7687", "neo4j", "password")
    inserter.insert_graph(graph_data)
    inserter.close()
```

### 使用LLM进行实体类型标准化

```python
def standardize_entity_types(entities: Dict, existing_ontology: List[str]) -> Dict:
    """将LLM抽取的实体类型映射到已有本体"""
    prompt = f"""将以下实体类型映射到标准本体中的类型。

标准本体类型: {', '.join(existing_ontology)}

需要映射的实体:
{json.dumps(list(set(e['type'] for e in entities.values())), ensure_ascii=False)}

返回JSON映射表: {{"原始类型": "标准类型"}}"""

    response = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    mapping = json.loads(response.choices[0].message.content)

    for name, info in entities.items():
        info["type"] = mapping.get(info["type"], "CONCEPT")

    return entities
```

---

## 3. 什么是GraphRAG？和传统RAG有什么区别？⭐⭐⭐

### 传统RAG vs GraphRAG

```
传统RAG:  问题 → 向量检索 → 文本片段 → LLM生成答案
GraphRAG: 问题 → 图检索+社区摘要 → 结构化知识 → LLM生成答案
```

### 核心区别

| 维度 | 传统RAG | GraphRAG |
|------|---------|----------|
| 检索单元 | 文本片段(chunk) | 实体、关系、社区摘要 |
| 索引方式 | 向量相似度 | 图结构+社区检测 |
| 全局问题 | ❌ 无法回答 | ✅ 通过社区摘要回答 |
| 推理能力 | 弱(仅语义匹配) | 强(图路径推理) |
| 成本 | 低 | 高(需要预处理构建图) |

### Microsoft GraphRAG 概念模型

```python
"""
Microsoft GraphRAG 核心概念:
1. Source Documents → Text Chunks → Entity/Relation Extraction
2. Entity/Relation → Graph → Community Detection (Leiden)
3. Community → Community Summaries (hierarchical)
4. Query: Local Search / Global Search
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Entity:
    name: str
    type: str
    description: str
    description_embedding: Optional[List[float]] = None
    community_ids: List[int] = field(default_factory=list)

@dataclass
class Relationship:
    source: str
    target: str
    description: str
    weight: float = 1.0
    description_embedding: Optional[List[float]] = None

@dataclass
class Community:
    community_id: int
    level: int  # 层级, 0为最细粒度
    entities: List[str]
    summary: str
    summary_embedding: Optional[List[float]] = None
    rating: float = 0.0  # 社区重要性评分
    rating_explanation: str = ""

@dataclass
class CommunityReport:
    """社区报告 - GraphRAG的核心产出"""
    community_id: int
    title: str
    summary: str
    findings: List[str]  # 关键发现列表
    rating: float  # 重要性 1-10
    rating_explanation: str
```

### 与传统RAG的对比实验

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.schema import Document

class TraditionalRAG:
    """传统向量RAG"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.vectorstore = None

    def index(self, documents: List[str]):
        docs = [Document(page_content=doc) for doc in documents]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def query(self, question: str, k: int = 3) -> str:
        docs = self.vectorstore.similarity_search(question, k=k)
        context = "\n---\n".join(d.page_content for d in docs)
        prompt = f"根据以下信息回答问题:\n\n{context}\n\n问题: {question}"
        return self.llm.invoke(prompt).content


class GraphRAGSystem:
    """GraphRAG系统 - 结合图谱与社区摘要"""

    def __init__(self, neo4j_uri, neo4j_auth):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def local_search(self, question: str) -> str:
        """局部搜索: 找到相关实体, 沿图遍历获取上下文"""
        # 1. 识别问题中的实体
        entities = self._extract_entities_from_query(question)

        # 2. 从图中检索相关子图
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.name IN $entities
                MATCH path = (e)-[*1..2]-(related)
                RETURN e.name AS entity,
                       collect(DISTINCT {
                           related: related.name,
                           rel_type: type(relationships(path)[0])
                       }) AS connections
            """, entities=entities)
            graph_context = [dict(r) for r in result]

        # 3. 构建上下文
        context_parts = []
        for item in graph_context:
            for conn in item["connections"]:
                context_parts.append(
                    f"{item['entity']} --[{conn['rel_type']}]--> {conn['related']}"
                )

        context = "\n".join(context_parts)
        prompt = f"基于以下知识图谱信息回答:\n\n{context}\n\n问题: {question}"
        return self.llm.invoke(prompt).content

    def global_search(self, question: str) -> str:
        """全局搜索: 利用社区摘要回答全局性问题"""
        with self.driver.session() as session:
            # 获取所有社区摘要
            result = session.run("""
                MATCH (c:Community)
                RETURN c.community_id AS id,
                       c.summary AS summary,
                       c.rating AS rating
                ORDER BY c.rating DESC
            """)
            communities = [dict(r) for r in result]

        # 让LLM选择相关社区并综合回答
        summaries = "\n".join(
            f"[社区{c['id']}] (重要性:{c['rating']}): {c['summary']}"
            for c in communities[:20]  # 取top20
        )

        prompt = f"""基于以下多个主题社区的摘要, 回答全局性问题。

社区摘要:
{summaries}

问题: {question}

请综合多个社区的信息, 给出全面的回答。"""

        return self.llm.invoke(prompt).content

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """从查询中提取实体名"""
        response = self.llm.invoke(
            f"从以下问题中提取人名、地名、组织名等实体, 返回JSON数组:\n{query}"
        )
        try:
            return json.loads(response.content)
        except:
            return [query]  # fallback
```

---

## 4. GraphRAG的索引和查询流程是什么？⭐⭐⭐

### 索引流程(Indexing Pipeline)

```python
"""
Microsoft GraphRAG 索引流程:

1. 文档分块 (Chunking)
2. 实体/关系抽取 (Entity Extraction)
3. 实体解析 (Entity Resolution)
4. 图构建 (Graph Construction)
5. 社区检测 (Community Detection - Leiden算法)
6. 社区摘要生成 (Community Summarization)
7. 向量嵌入 (Embedding)
"""

import networkx as nx
import leidenalg
import igraph as ig
from typing import List, Dict

class GraphRAGIndexer:
    """GraphRAG索引构建器"""

    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=llm_model)
        self.graph = nx.Graph()
        self.entities = {}
        self.relationships = []

    def step1_chunk_documents(self, documents: List[str],
                               max_tokens: int = 600,
                               overlap: int = 100) -> List[str]:
        """步骤1: 文档分块(带重叠)"""
        chunks = []
        for doc in documents:
            # 简化: 按字符分块(实际应用按token)
            words = doc.split()
            for i in range(0, len(words), max_tokens - overlap):
                chunk = " ".join(words[i:i + max_tokens])
                if len(chunk.strip()) > 50:
                    chunks.append(chunk)
        return chunks

    def step2_extract_entities(self, chunks: List[str]) -> Dict:
        """步骤2: 使用LLM抽取实体和关系(带默认提示词)"""
        all_entities = {}
        all_relations = []

        entity_prompt = """-目标-
给定以下文本文档, 识别所有实体和关系。

-实体-
entity(name, type, description)
type: PERSON, ORGANIZATION, GEO, EVENT, CONCEPT, TECHNOLOGY

-关系-
relationship(source, target, type, description, weight)
weight: 1-10, 表示关系强度

-输出格式-
("entity"{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description)
("relationship"{tuple_delimiter}source{tuple_delimiter}target{tuple_delimiter}relation_type{tuple_delimiter}description{tuple_delimiter}weight)

文本:
{input_text}"""

        for i, chunk in enumerate(chunks):
            response = self.llm.invoke(entity_prompt.format(
                input_text=chunk,
                tuple_delimiter="<|>"
            ))
            parsed = self._parse_extraction(response.content)
            all_entities.update(parsed["entities"])
            all_relations.extend(parsed["relations"])

        return {"entities": all_entities, "relations": all_relations}

    def step3_resolve_entities(self, entities: Dict) -> Dict:
        """步骤3: 实体消歧和合并"""
        # 简化版: 基于名称相似度合并
        resolved = {}
        name_map = {}  # alias -> canonical_name

        entity_names = list(entities.keys())

        for name in entity_names:
            if name in name_map:
                continue
            # 查找相似实体
            similar = [n for n in entity_names
                      if n != name and self._name_similarity(name, n) > 0.8]

            canonical = name
            resolved[canonical] = entities[canonical]
            name_map[canonical] = canonical

            for alias in similar:
                name_map[alias] = canonical
                # 合并描述
                resolved[canonical]["description"] += f"; {entities[alias]['description']}"

        return resolved, name_map

    def step4_build_graph(self, entities: Dict, relations: List,
                          name_map: Dict) -> nx.Graph:
        """步骤4: 构建NetworkX图"""
        G = nx.Graph()

        # 添加节点
        for name, info in entities.items():
            G.add_node(name, **info)

        # 添加边(权重)
        for rel in relations:
            src = name_map.get(rel["source"], rel["source"])
            tgt = name_map.get(rel["target"], rel["target"])
            if src in G and tgt in G:
                if G.has_edge(src, tgt):
                    G[src][tgt]["weight"] += rel.get("weight", 1)
                else:
                    G.add_edge(src, tgt,
                              relation=rel["type"],
                              description=rel["description"],
                              weight=rel.get("weight", 1))

        self.graph = G
        return G

    def step5_community_detection(self, G: nx.Graph) -> Dict:
        """步骤5: 使用Leiden算法进行社区检测"""
        # 转换为igraph
        ig_graph = ig.Graph.from_networkx(G)

        # 多层级社区检测
        communities = {}
        for resolution in [0.5, 1.0, 2.0]:
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution
            )

            level = {0: 0, 1: 1, 2: 2}[resolution]  # 简化映射
            for idx, community_id in enumerate(partition.membership):
                node_name = ig_graph.vs[idx]["_nx_name"]
                if node_name not in communities:
                    communities[node_name] = []
                communities[node_name].append({
                    "level": level,
                    "community_id": community_id
                })

        return communities

    def step6_generate_community_reports(self, G: nx.Graph,
                                          communities: Dict) -> List[Dict]:
        """步骤6: 为每个社区生成摘要报告"""
        # 按社区分组
        community_groups = {}
        for node, comms in communities.items():
            for comm in comms:
                key = (comm["level"], comm["community_id"])
                if key not in community_groups:
                    community_groups[key] = []
                community_groups[key].append(node)

        reports = []
        for (level, comm_id), members in community_groups.items():
            # 收集社区内实体和关系信息
            subgraph = G.subgraph(members)
            entity_descs = []
            for node in subgraph.nodes():
                desc = subgraph.nodes[node].get("description", "")
                entity_descs.append(f"- {node}: {desc}")

            rel_descs = []
            for u, v, data in subgraph.edges(data=True):
                rel_descs.append(f"- {u} --[{data.get('relation', '')}]--> {v}")

            # 用LLM生成社区摘要
            prompt = f"""为以下社区生成摘要报告:

实体:
{chr(10).join(entity_descs[:20])}

关系:
{chr(10).join(rel_descs[:20])}

请生成:
1. 标题
2. 一句话总结
3. 2-3个关键发现
4. 重要性评分(1-10)"""

            response = self.llm.invoke(prompt)
            reports.append({
                "level": level,
                "community_id": comm_id,
                "members": members,
                "report": response.content
            })

        return reports

    def _parse_extraction(self, text: str) -> Dict:
        """解析LLM抽取结果"""
        entities = {}
        relations = []
        for line in text.split("\n"):
            if '"entity"' in line:
                parts = line.split("<|>")
                if len(parts) >= 4:
                    name = parts[1].strip().strip('"')
                    entities[name] = {
                        "type": parts[2].strip(),
                        "description": parts[3].strip().strip('"').rstrip(")")
                    }
            elif '"relationship"' in line:
                parts = line.split("<|>")
                if len(parts) >= 6:
                    relations.append({
                        "source": parts[1].strip().strip('"'),
                        "target": parts[2].strip().strip('"'),
                        "type": parts[3].strip(),
                        "description": parts[4].strip(),
                        "weight": int(parts[5].strip().rstrip(")") or "1")
                    })
        return {"entities": entities, "relations": relations}

    def _name_similarity(self, a: str, b: str) -> float:
        """简单名称相似度"""
        a_set = set(a)
        b_set = set(b)
        return len(a_set & b_set) / len(a_set | b_set)
```

### 查询流程

```python
class GraphRAGQueryEngine:
    """GraphRAG查询引擎"""

    def __init__(self, indexer: GraphRAGIndexer):
        self.indexer = indexer
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def query(self, question: str, method: str = "auto") -> str:
        """
        method:
        - "local": 局部搜索, 适合具体实体问题
        - "global": 全局搜索, 适合总结性/全局性问题
        - "auto": 自动选择
        """
        if method == "auto":
            method = self._classify_query(question)

        if method == "global":
            return self._global_search(question)
        else:
            return self._local_search(question)

    def _classify_query(self, question: str) -> str:
        """自动判断查询类型"""
        response = self.llm.invoke(
            f"""判断以下问题适合"local"(具体实体查询)还是"global"(全局总结查询):

问题: {question}

只回答 local 或 global:"""
        )
        return "global" if "global" in response.content.lower() else "local"

    def _local_search(self, question: str) -> str:
        """局部搜索: 实体→子图→上下文→LLM"""
        G = self.indexer.graph

        # 1. 提取查询中的实体
        entities = self._extract_entities(question)

        # 2. 扩展子图(2跳)
        context_nodes = set()
        for entity in entities:
            if entity in G:
                context_nodes.add(entity)
                for neighbor in G.neighbors(entity):
                    context_nodes.add(neighbor)
                    for n2 in G.neighbors(neighbor):
                        context_nodes.add(n2)

        # 3. 构建上下文
        subgraph = G.subgraph(context_nodes)
        context = self._subgraph_to_text(subgraph)

        # 4. LLM回答
        prompt = f"""基于以下知识图谱信息回答问题。

图谱上下文:
{context}

问题: {question}"""
        return self.llm.invoke(prompt).content

    def _global_search(self, question: str) -> str:
        """全局搜索: 社区摘要→Map-Reduce→LLM"""
        # 1. 获取所有社区报告
        reports = self.indexer.step6_generate_community_reports(
            self.indexer.graph, {}
        )

        # 2. Map: 每个社区独立回答
        map_results = []
        for report in reports[:15]:  # 限制数量
            prompt = f"""基于以下社区信息, 回答问题(如果无关返回"N/A"):

社区: {report['report']}

问题: {question}
相关回答:"""
            response = self.llm.invoke(prompt)
            if "N/A" not in response.content:
                map_results.append(response.content)

        # 3. Reduce: 综合所有回答
        reduce_prompt = f"""综合以下多个视角的回答, 生成一个全面、一致的最终答案:

{chr(10).join(f'视角{i+1}: {r}' for i, r in enumerate(map_results))}

问题: {question}
最终答案:"""
        return self.llm.invoke(reduce_prompt).content

    def _subgraph_to_text(self, subgraph: nx.Graph) -> str:
        """将子图转换为文本描述"""
        lines = []
        for node in subgraph.nodes():
            desc = subgraph.nodes[node].get("description", "")
            lines.append(f"[实体] {node}: {desc}")
        for u, v, data in subgraph.edges(data=True):
            rel = data.get("relation", "RELATED_TO")
            desc = data.get("description", "")
            lines.append(f"[关系] {u} --[{rel}]--> {v}: {desc}")
        return "\n".join(lines)

    def _extract_entities(self, question: str) -> List[str]:
        """从问题中提取实体"""
        entities_in_graph = list(self.indexer.graph.nodes())
        # 简化: 检查问题中是否包含图谱中的实体名
        found = [e for e in entities_in_graph if e in question]
        return found if found else [question]
```

---

## 5. Neo4j vs NebulaGraph 如何选型？⭐⭐

### 对比分析

```python
comparison = {
    "Neo4j": {
        "数据模型": "属性图(Property Graph), 原生图存储",
        "查询语言": "Cypher (声明式)",
        "存储引擎": "原生图存储, 节点/关系直接物理邻接",
        "事务支持": "完整ACID",
        "生态": "最成熟, 大量教程和工具",
        "社区版限制": "社区版免费, 企业版收费(集群/因果一致性)",
        "适用场景": "中小规模(亿级), 复杂图遍历, 快速原型",
        "Python驱动": "neo4j (官方)",
        "特色": "APOC插件, GDS图算法库, Bloom可视化",
    },
    "NebulaGraph": {
        "数据模型": "属性图, 分布式存储",
        "查询语言": "nGQL (类SQL, 声明式+过程式混合)",
        "存储引擎": "分布式架构, 存算分离",
        "事务支持": "支持(有限制)",
        "生态": "国内社区活跃, 企业级功能免费",
        "社区版限制": "Apache 2.0 开源, 无功能限制",
        "适用场景": "大规模(千亿级), 高并发, 多数据中心",
        "Python驱动": "nebula3-python",
        "特色": "原生分布式, 多副本, 支持多种后端存储",
    }
}
```

### 代码对比

```python
# ============ Neo4j Cypher ============

# 创建节点
"""
CREATE (e:Person {name: '爱因斯坦', born: 1879})
CREATE (t:Theory {name: '相对论', year: 1905})
CREATE (e)-[:PROPOSED {context: '奇迹年'}]->(t)
"""

# 查询: 找出与爱因斯坦相关的所有2跳内实体
"""
MATCH (e:Person {name: '爱因斯坦'})-[*1..2]-(related)
RETURN DISTINCT related.name, labels(related)
"""

# 聚合: 每个人的关系数量
"""
MATCH (p:Person)-[r]->()
RETURN p.name, count(r) AS rel_count
ORDER BY rel_count DESC LIMIT 10
"""

# 全文搜索
"""
CALL db.index.fulltext.queryNodes('entity_name', '爱因斯坦')
YIELD node, score
RETURN node.name, score
"""


# ============ NebulaGraph nGQL ============

# 创建Space(类似数据库)
"""
CREATE SPACE IF NOT EXISTS knowledge_graph (
    vid_type = FIXED_STRING(128),
    partition_num = 10,
    replica_factor = 1
);
"""

# 创建Schema
"""
CREATE TAG IF NOT EXISTS Person(name string, born int);
CREATE TAG IF NOT EXISTS Theory(name string, year int);
CREATE EDGE IF NOT EXISTS PROPOSED(context string);
"""

# 插入数据
"""
INSERT VERTEX Person(name, born) VALUES '爱因斯坦':('爱因斯坦', 1879);
INSERT VERTEX Theory(name, year) VALUES '相对论':('相对论', 1905);
INSERT EDGE PROPOSED(context) VALUES '爱因斯坦'->'相对论':('奇迹年');
"""

# 2跳查询
"""
GO 2 STEPS FROM '爱因斯坦' OVER * YIELD dst(edge) AS related_id
| FETCH PROP ON * $-.related_id YIELD vertex AS related;
"""

# 聚合查询
"""
MATCH (v:Person)-[e]->()
RETURN v.Person.name AS name, count(e) AS rel_count
ORDER BY rel_count DESC LIMIT 10;
"""
```

### 选型建议

```python
def choose_graph_db(requirements: dict) -> str:
    """根据需求选择图数据库"""
    score_neo4j = 0
    score_nebula = 0

    # 数据规模
    if requirements.get("data_scale", "") == "large":  # 百亿+
        score_nebula += 3
    else:
        score_neo4j += 2

    # 是否需要分布式
    if requirements.get("distributed", False):
        score_nebula += 3
    else:
        score_neo4j += 1

    # 团队经验
    if requirements.get("team_experience", "") == "neo4j":
        score_neo4j += 2
    elif requirements.get("team_experience", "") == "nebula":
        score_nebula += 2

    # 预算
    if requirements.get("budget", "") == "zero":
        score_nebula += 2  # NebulaGraph完全开源

    # 生态需求
    if requirements.get("need_rich_ecosystem", True):
        score_neo4j += 2

    return "Neo4j" if score_neo4j >= score_nebula else "NebulaGraph"

# 测试
print(choose_graph_db({"data_scale": "small", "distributed": False}))
# Neo4j
print(choose_graph_db({"data_scale": "large", "distributed": True, "budget": "zero"}))
# NebulaGraph
```

---

## 6. 知识图谱的实体消歧和融合？⭐⭐

### 实体消歧(Entity Resolution)

```python
import numpy as np
from difflib import SequenceMatcher

class EntityResolver:
    """实体消歧与融合"""

    def __init__(self, embeddings_model=None):
        self.embeddings = embeddings_model

    def resolve(self, entities: Dict[str, Dict]) -> Tuple[Dict, Dict]:
        """
        输入: {entity_name: {type, description, ...}}
        输出: (resolved_entities, name_mapping)
        """
        names = list(entities.keys())
        n = len(names)

        # 计算相似度矩阵
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                sim = self._compute_similarity(
                    names[i], names[j],
                    entities[names[i]], entities[names[j]]
                )
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim

        # 贪心合并
        merged = set()
        name_map = {}
        resolved = {}

        for i in range(n):
            if names[i] in merged:
                continue

            canonical = names[i]
            group = [canonical]

            for j in range(i+1, n):
                if names[j] in merged:
                    continue
                if sim_matrix[i][j] > 0.85:  # 阈值
                    group.append(names[j])
                    merged.add(names[j])
                    name_map[names[j]] = canonical

            # 融合实体信息
            resolved[canonical] = self._merge_entity_info(
                [entities[name] for name in group]
            )
            name_map[canonical] = canonical

        return resolved, name_map

    def _compute_similarity(self, name1: str, name2: str,
                            info1: Dict, info2: Dict) -> float:
        """综合相似度计算"""
        # 1. 名称相似度
        name_sim = SequenceMatcher(None, name1, name2).ratio()

        # 2. 类型匹配
        type_sim = 1.0 if info1.get("type") == info2.get("type") else 0.0

        # 3. 描述语义相似度(如果有embedding)
        desc_sim = 0.5  # 默认
        if self.embeddings and info1.get("description") and info2.get("description"):
            emb1 = self.embeddings.embed_query(info1["description"])
            emb2 = self.embeddings.embed_query(info2["description"])
            desc_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # 加权综合
        return 0.4 * name_sim + 0.3 * type_sim + 0.3 * desc_sim

    def _merge_entity_info(self, infos: List[Dict]) -> Dict:
        """融合多个实体的信息"""
        merged = {
            "type": infos[0].get("type", "UNKNOWN"),
            "descriptions": [],
            "aliases": set()
        }

        for info in infos:
            if info.get("description"):
                merged["descriptions"].append(info["description"])
            if info.get("name"):
                merged["aliases"].add(info["name"])

        # 合并描述(取最长或拼接)
        merged["description"] = max(merged["descriptions"], key=len, default="")
        merged["aliases"] = list(merged["aliases"])

        return merged


# 在图数据库中执行实体融合
def merge_entities_in_neo4j(driver, canonical: str, aliases: List[str]):
    """在Neo4j中合并重复实体"""
    with driver.session() as session:
        # 1. 将所有关系重定向到canonical实体
        for alias in aliases:
            if alias == canonical:
                continue
            session.run("""
                MATCH (old {name: $alias})
                MATCH (canonical {name: $canonical})
                // 复制入边
                MATCH (other)-[r]->(old)
                CALL apoc.merge.node(labels(canonical), {name: $canonical})
                YIELD node AS target
                WITH other, r, type(r) AS rel_type, properties(r) AS props, target
                CALL apoc.create.relationship(other, rel_type, props, target)
                YIELD rel
                RETURN count(rel)
            """, alias=alias, canonical=canonical)

            # 2. 删除旧节点
            session.run("""
                MATCH (n {name: $alias})
                DETACH DELETE n
            """, alias=alias)
```

---

## 7. 如何实现图谱增强的问答系统？⭐⭐⭐

### 完整的GraphQA系统

```python
from enum import Enum
from typing import List, Dict, Optional

class QueryType(Enum):
    ENTITY_FACT = "entity_fact"         # 实体事实查询
    RELATION_PATH = "relation_path"     # 关系路径查询
    AGGREGATION = "aggregation"         # 聚合统计查询
    GLOBAL_SUMMARY = "global_summary"   # 全局总结查询

class GraphEnhancedQA:
    """图谱增强的问答系统"""

    def __init__(self, neo4j_uri: str, auth: tuple):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(neo4j_uri, auth=auth)
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def answer(self, question: str) -> Dict:
        """主入口: 分类→检索→生成"""

        # 1. 问题分类
        query_type = self._classify_question(question)

        # 2. 选择策略检索
        if query_type == QueryType.ENTITY_FACT:
            context = self._retrieve_entity_context(question)
        elif query_type == QueryType.RELATION_PATH:
            context = self._retrieve_path_context(question)
        elif query_type == QueryType.AGGREGATION:
            context = self._retrieve_aggregation_context(question)
        else:
            context = self._retrieve_community_context(question)

        # 3. 生成答案
        answer = self._generate_answer(question, context, query_type)

        return {
            "question": question,
            "type": query_type.value,
            "answer": answer,
            "context_used": context
        }

    def _classify_question(self, question: str) -> QueryType:
        """问题分类"""
        response = self.llm.invoke(f"""将问题分类为以下类型之一:
- entity_fact: 关于某个实体的具体事实
- relation_path: 需要追踪实体间关系路径
- aggregation: 需要统计/聚合多个实体
- global_summary: 需要全局性总结

问题: {question}
只回答类型名:""")
        try:
            return QueryType(response.content.strip().split()[-1])
        except:
            return QueryType.ENTITY_FACT

    def _retrieve_entity_context(self, question: str) -> str:
        """实体事实查询: 精确匹配实体属性"""
        # 提取实体名
        entity = self._extract_main_entity(question)

        with self.driver.session() as session:
            result = session.run("""
                MATCH (e {name: $name})
                OPTIONAL MATCH (e)-[r]-(related)
                RETURN e AS entity,
                       collect({
                           relation: type(r),
                           direction: CASE
                               WHEN startNode(r) = e THEN 'outgoing'
                               ELSE 'incoming'
                           END,
                           other: related.name,
                           other_labels: labels(related)
                       }) AS connections
            """, name=entity)

            records = [dict(r) for r in result]

        if not records:
            return f"未找到实体 '{entity}' 的信息"

        # 格式化上下文
        entity_node = records[0]["entity"]
        lines = [f"实体: {entity_node['name']}"]
        for key, val in entity_node.items():
            if key != "name":
                lines.append(f"  {key}: {val}")

        for conn in records[0]["connections"][:10]:
            if conn["direction"] == "outgoing":
                lines.append(f"  --[{conn['relation']}]--> {conn['other']}")
            else:
                lines.append(f"  <--[{conn['relation']}]-- {conn['other']}")

        return "\n".join(lines)

    def _retrieve_path_context(self, question: str) -> str:
        """关系路径查询: 找最短路径"""
        entities = self._extract_two_entities(question)
        if len(entities) < 2:
            return self._retrieve_entity_context(question)

        with self.driver.session() as session:
            result = session.run("""
                MATCH (a {name: $name1}), (b {name: $name2})
                MATCH path = shortestPath((a)-[*..5]-(b))
                RETURN [n IN nodes(path) | n.name] AS node_names,
                       [r IN relationships(path) | type(r)] AS rel_types
            """, name1=entities[0], name2=entities[1])

            paths = [dict(r) for r in result]

        if not paths:
            return f"未找到 {entities[0]} 和 {entities[1]} 之间的路径"

        lines = []
        for path in paths:
            nodes = path["node_names"]
            rels = path["rel_types"]
            desc = nodes[0]
            for i, rel in enumerate(rels):
                desc += f" --[{rel}]--> {nodes[i+1]}"
            lines.append(desc)

        return "\n".join(lines)

    def _retrieve_aggregation_context(self, question: str) -> str:
        """聚合查询"""
        with self.driver.session() as session:
            # 获取图谱统计信息
            stats = session.run("""
                MATCH (n)
                WITH labels(n) AS lbls, count(n) AS cnt
                RETURN lbls, cnt ORDER BY cnt DESC LIMIT 10
            """)
            node_stats = [dict(r) for r in stats]

            rel_stats = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS rel_type, count(r) AS cnt
                ORDER BY cnt DESC LIMIT 10
            """)
            rel_counts = [dict(r) for r in rel_stats]

        context = "图谱统计:\n"
        for s in node_stats:
            context += f"  {s['lbls']}: {s['cnt']}个\n"
        for r in rel_counts:
            context += f"  关系[{r['rel_type']}]: {r['cnt']}条\n"

        return context

    def _retrieve_community_context(self, question: str) -> str:
        """全局搜索: 社区摘要"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Community)
                WHERE c.summary IS NOT NULL
                RETURN c.community_id AS id,
                       c.summary AS summary,
                       c.rating AS rating
                ORDER BY c.rating DESC LIMIT 10
            """)
            communities = [dict(r) for r in result]

        return "\n".join(
            f"[社区{c['id']}] {c['summary']}" for c in communities
        )

    def _generate_answer(self, question: str, context: str,
                         query_type: QueryType) -> str:
        """生成答案"""
        prompt = f"""你是一个知识图谱问答助手。基于以下图谱信息回答问题。

查询类型: {query_type.value}
图谱上下文:
{context}

问题: {question}

请给出准确、简洁的回答。如果图谱信息不足以回答, 请说明。"""

        return self.llm.invoke(prompt).content

    def _extract_main_entity(self, question: str) -> str:
        """提取问题中的主要实体"""
        # 简化实现: 用LLM提取
        response = self.llm.invoke(
            f"从问题中提取最主要的一个实体名称, 只返回实体名:\n{question}"
        )
        return response.content.strip()

    def _extract_two_entities(self, question: str) -> List[str]:
        """提取两个实体"""
        response = self.llm.invoke(
            f"从问题中提取两个实体名称, 返回JSON数组:\n{question}"
        )
        try:
            return json.loads(response.content)
        except:
            return []
```

### Cypher查询生成( Text2Cypher)

```python
class Text2Cypher:
    """自然语言转Cypher查询"""

    def __init__(self, driver, schema_info: str):
        self.driver = driver
        self.schema = schema_info
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def generate_and_execute(self, question: str) -> str:
        # 1. 生成Cypher
        prompt = f"""基于以下Neo4j schema, 将自然语言问题转换为Cypher查询。

Schema:
{self.schema}

规则:
- 只返回Cypher语句, 不要解释
- 使用参数化查询防止注入
- 限制结果数量(LIMIT 20)

问题: {question}
Cypher:"""

        cypher = self.llm.invoke(prompt).content.strip()
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()

        # 2. 执行
        with self.driver.session() as session:
            try:
                result = session.run(cypher)
                records = [dict(r) for r in result]
                return json.dumps(records, ensure_ascii=False, default=str)
            except Exception as e:
                return f"查询执行失败: {e}\n生成的Cypher: {cypher}"
```

---

## 8. 社区检测在GraphRAG中的作用？⭐⭐

### 社区检测算法

```python
import networkx as nx
import leidenalg
import igraph as ig
from collections import defaultdict

class CommunityDetector:
    """知识图谱社区检测"""

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def leiden_detection(self, resolution: float = 1.0) -> Dict:
        """
        Leiden算法 - GraphRAG推荐的社区检测算法
        优势: 速度快, 质量高, 支持层次化
        """
        # NetworkX → iGraph
        ig_graph = ig.Graph.from_networkx(self.graph)

        # 执行Leiden
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            n_iterations=10
        )

        # 解析结果
        communities = defaultdict(list)
        for idx, comm_id in enumerate(partition.membership):
            node_name = ig_graph.vs[idx]["_nx_name"]
            communities[comm_id].append(node_name)

        return dict(communities)

    def hierarchical_leiden(self) -> Dict:
        """
        层次化Leiden检测 - GraphRAG的核心
        生成多个粒度级别的社区
        """
        levels = {}
        resolutions = [0.3, 0.7, 1.5, 3.0]  # 从粗到细

        for level, res in enumerate(resolutions):
            communities = self.leiden_detection(resolution=res)
            levels[level] = communities
            print(f"Level {level} (resolution={res}): {len(communities)} communities")

        return levels

    def compute_community_statistics(self, communities: Dict) -> List[Dict]:
        """计算社区统计信息"""
        stats = []
        G = self.graph

        for comm_id, members in communities.items():
            subgraph = G.subgraph(members)
            internal_edges = subgraph.number_of_edges()
            total_degree = sum(G.degree(n) for n in members)
            external_edges = total_degree - 2 * internal_edges

            stats.append({
                "community_id": comm_id,
                "size": len(members),
                "internal_edges": internal_edges,
                "external_edges": external_edges,
                "density": nx.density(subgraph) if len(members) > 1 else 0,
                "modularity_contribution": internal_edges / max(total_degree, 1)
            })

        return sorted(stats, key=lambda x: x["size"], reverse=True)

    def generate_community_summary(self, community_id: int,
                                    members: List[str],
                                    entity_info: Dict) -> str:
        """生成社区摘要(供GraphRAG全局搜索使用)"""
        G = self.graph
        subgraph = G.subgraph(members)

        # 收集社区内的关键信息
        entities_desc = []
        for node in members[:30]:  # 限制长度
            info = entity_info.get(node, {})
            entities_desc.append(
                f"- {node} ({info.get('type', '未知')}): "
                f"{info.get('description', '无描述')}"
            )

        # 收集社区内的关键关系
        key_relations = []
        for u, v, data in subgraph.edges(data=True):
            rel = data.get("relation", "RELATED_TO")
            weight = data.get("weight", 1)
            if weight >= 2:  # 只保留强关系
                key_relations.append(f"- {u} --[{rel}]--> {v} (强度:{weight})")

        return {
            "community_id": community_id,
            "members": members,
            "entity_descriptions": entities_desc,
            "key_relations": key_relations[:20],
            "size": len(members)
        }


# 社区检测在GraphRAG中的作用示例
def demonstrate_community_role():
    """展示社区检测在GraphRAG中的核心作用"""

    # 1. 构建示例图
    G = nx.Graph()

    # 物理学社区
    physics = ["爱因斯坦", "玻尔", "海森堡", "薛定谔", "狄拉克", "普朗克"]
    for i, p1 in enumerate(physics):
        for p2 in physics[i+1:]:
            G.add_edge(p1, p2, relation="colleague", weight=3)

    # AI社区
    ai = ["Hinton", "LeCun", "Bengio", "Goodfellow", "Vaswani"]
    for i, a1 in enumerate(ai):
        for a2 in ai[i+1:]:
            G.add_edge(a1, a2, relation="colleague", weight=3)

    # 跨社区弱连接
    G.add_edge("爱因斯坦", "Hinton", relation="influenced", weight=1)

    # 2. 社区检测
    detector = CommunityDetector(G)
    communities = detector.leiden_detection(resolution=1.0)

    print("检测到的社区:")
    for comm_id, members in communities.items():
        print(f"  社区{comm_id}: {members}")

    # 3. 社区摘要 → 全局搜索的基础
    # 用户问: "物理学和AI领域有什么联系？"
    # → 通过社区摘要可以跨社区综合回答
    # 而不需要遍历整个图

demonstrate_community_role()
```

---

## 9. 知识图谱的更新和维护策略？⭐⭐

### 增量更新策略

```python
class KnowledgeGraphMaintenance:
    """知识图谱维护和更新"""

    def __init__(self, driver):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(**driver)
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    # === 1. 增量实体/关系更新 ===

    def incremental_update(self, new_documents: List[str]):
        """增量更新流程"""
        builder = LLMKnowledgeGraphBuilder()

        for doc in new_documents:
            # 抽取新知识
            new_kg = builder.process_document(doc)

            with self.driver.session() as session:
                for name, info in new_kg["entities"].items():
                    # MERGE: 存在则更新, 不存在则创建
                    session.run("""
                        MERGE (e {name: $name})
                        ON CREATE SET e.type = $type,
                                      e.description = $desc,
                                      e.created_at = datetime(),
                                      e.updated_at = datetime()
                        ON MATCH SET e.description =
                            CASE WHEN e.description CONTAINS $desc
                                 THEN e.description
                                 ELSE e.description + ' | ' + $desc
                            END,
                            e.updated_at = datetime()
                    """, name=name, type=info["type"], desc=info["summary"])

                for rel in new_kg["relations"]:
                    session.run("""
                        MATCH (a {name: $src})
                        MATCH (b {name: $tgt})
                        MERGE (a)-[r:{rel_type}]->(b)
                        ON CREATE SET r.description = $desc,
                                      r.created_at = datetime(),
                                      r.weight = 1
                        ON MATCH SET r.weight = r.weight + 1,
                                     r.updated_at = datetime()
                    """.format(rel_type=rel["relation_type"].upper().replace(" ", "_")),
                        src=rel["source"], tgt=rel["target"],
                        desc=rel.get("description", "")
                    )

    # === 2. 陈旧知识检测 ===

    def detect_stale_knowledge(self, days_threshold: int = 90) -> List[Dict]:
        """检测长时间未更新的知识"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e)
                WHERE e.updated_at < datetime() - duration({days: $days})
                AND e.updated_at IS NOT NULL
                RETURN e.name AS name,
                       e.type AS type,
                       e.description AS description,
                       e.updated_at AS last_updated
                ORDER BY e.updated_at
                LIMIT 50
            """, days=days_threshold)
            return [dict(r) for r in result]

    # === 3. 质量检查 ===

    def quality_check(self) -> Dict:
        """图谱质量检查"""
        checks = {}

        with self.driver.session() as session:
            # 孤立节点
            result = session.run("""
                MATCH (e)
                WHERE NOT (e)--()
                RETURN count(e) AS orphan_count
            """)
            checks["orphan_nodes"] = result.single()["orphan_count"]

            # 无描述实体
            result = session.run("""
                MATCH (e)
                WHERE e.description IS NULL OR e.description = ''
                RETURN count(e) AS no_desc_count
            """)
            checks["no_description"] = result.single()["no_desc_count"]

            # 重复实体(名称高度相似)
            result = session.run("""
                MATCH (a), (b)
                WHERE a.name < b.name
                AND a.name CONTAINS b.name
                RETURN a.name AS name1, b.name AS name2
                LIMIT 20
            """)
            checks["potential_duplicates"] = [dict(r) for r in result]

            # 图统计
            result = session.run("""
                MATCH (n) RETURN count(n) AS total_nodes
            """)
            checks["total_nodes"] = result.single()["total_nodes"]

            result = session.run("""
                MATCH ()-[r]->() RETURN count(r) AS total_edges
            """)
            checks["total_edges"] = result.single()["total_edges"]

        return checks

    # === 4. 版本管理 ===

    def create_snapshot(self, snapshot_name: str):
        """创建图谱快照(用于回滚)"""
        with self.driver.session() as session:
            # 使用APOC导出
            session.run("""
                CALL apoc.export.cypher.all($filename, {
                    format: 'cypher-shell',
                    useOptimizations: {type: 'UNWIND_BATCH', unwindBatchSize: 100}
                })
            """, filename=f"snapshots/{snapshot_name}.cypher")

    # === 5. 定时维护任务 ===

    def scheduled_maintenance(self):
        """定时维护任务"""
        tasks = {
            "rebuild_community_detection": self._rebuild_communities,
            "refresh_embeddings": self._refresh_embeddings,
            "cleanup_temp_nodes": self._cleanup_temp,
            "update_statistics": self._update_stats,
        }

        for task_name, task_func in tasks.items():
            try:
                task_func()
                print(f"✅ {task_name} 完成")
            except Exception as e:
                print(f"❌ {task_name} 失败: {e}")

    def _rebuild_communities(self):
        """重建社区检测"""
        with self.driver.session() as session:
            session.run("MATCH (e) REMOVE e.community_id, e.community_level")
        # 重新运行Leiden...
        print("社区重建完成")

    def _refresh_embeddings(self):
        """刷新向量嵌入"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e)
                WHERE e.embedding IS NULL AND e.description IS NOT NULL
                RETURN e.name AS name, e.description AS desc
                LIMIT 100
            """)
            entities = [dict(r) for r in result]

        # 重新生成embedding
        embeddings = OpenAIEmbeddings()
        for entity in entities:
            emb = embeddings.embed_query(entity["desc"])
            with self.driver.session() as session:
                session.run("""
                    MATCH (e {name: $name})
                    SET e.embedding = $embedding
                """, name=entity["name"], embedding=emb)

    def _cleanup_temp(self):
        """清理临时节点"""
        with self.driver.session() as session:
            session.run("""
                MATCH (n:Temporary)
                WHERE n.expires_at < datetime()
                DETACH DELETE n
            """)

    def _update_stats(self):
        """更新统计信息"""
        with self.driver.session() as session:
            # 计算节点度数
            session.run("""
                MATCH (e)
                OPTIONAL MATCH (e)-[r]-()
                SET e.degree = count(r)
            """)
```

---

## 10. GraphRAG的成本和性能优化？⭐⭐

### 成本分析

```python
"""
GraphRAG成本构成:

1. 索引阶段(一次性, 高成本)
   - LLM调用: 实体抽取 + 社区摘要
   - 假设100万token文档:
     - 实体抽取: ~500次API调用 × $0.003/次 ≈ $1.5
     - 社区摘要: ~50次API调用 × $0.005/次 ≈ $0.25
     - 嵌入生成: ~10000实体 × $0.0001 ≈ $1.0
     - 总计: ~$3-5 (gpt-4o-mini), ~$30-50 (gpt-4o)

2. 查询阶段(持续, 中等成本)
   - 每次查询: 1-3次LLM调用
   - 局部搜索: ~$0.002/次
   - 全局搜索: ~$0.01/次 (Map-Reduce)

3. 存储成本
   - Neo4j: 内存需求 ≈ 图大小 × 2-3倍
   - 向量索引: 10000实体 × 1536维 × 4字节 ≈ 60MB
"""

class CostEstimator:
    """GraphRAG成本估算器"""

    def __init__(self):
        self.pricing = {
            "gpt-4o-mini": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
            "gpt-4o": {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
            "text-embedding-3-small": {"input": 0.02 / 1e6},
        }

    def estimate_indexing_cost(self, total_tokens: int,
                                model: str = "gpt-4o-mini") -> Dict:
        """估算索引成本"""
        p = self.pricing[model]

        # 实体抽取: 输入≈chunk_size, 输出≈3倍
        num_chunks = total_tokens // 1200
        extract_input = num_chunks * 1500
        extract_output = num_chunks * 500

        # 社区摘要: 约num_chunks/10个社区
        num_communities = max(num_chunks // 10, 5)
        comm_input = num_communities * 2000
        comm_output = num_communities * 500

        # 实体描述合并
        merge_input = num_chunks * 1000
        merge_output = num_chunks * 200

        total_cost = (
            (extract_input + comm_input + merge_input) * p["input"] +
            (extract_output + comm_output + merge_output) * p["output"]
        )

        return {
            "model": model,
            "total_input_tokens": extract_input + comm_input + merge_input,
            "total_output_tokens": extract_output + comm_output + merge_output,
            "estimated_cost_usd": round(total_cost, 4),
            "num_chunks": num_chunks,
            "num_communities": num_communities
        }

    def estimate_query_cost(self, queries_per_day: int,
                             avg_local: float = 0.7,
                             model: str = "gpt-4o-mini") -> Dict:
        """估算每日查询成本"""
        p = self.pricing[model]

        # 局部搜索: ~3000 tokens
        local_cost = queries_per_day * avg_local * 3000 * (p["input"] + p["output"])

        # 全局搜索: ~15000 tokens (Map-Reduce)
        global_cost = queries_per_day * (1 - avg_local) * 15000 * (p["input"] + p["output"])

        return {
            "daily_cost_usd": round(local_cost + global_cost, 4),
            "monthly_cost_usd": round((local_cost + global_cost) * 30, 2)
        }


# 成本优化策略
class GraphRAGOptimizer:
    """GraphRAG性能和成本优化"""

    @staticmethod
    def optimization_strategies():
        return {
            "索引阶段优化": {
                "1. 使用更便宜的模型": {
                    "策略": "实体抽取用gpt-4o-mini, 社区摘要再用gpt-4o",
                    "节省": "60-70%",
                    "风险": "实体质量可能下降"
                },
                "2. 减少抽取轮次": {
                    "策略": "增大chunk_size(1200→2000), 减少总调用次数",
                    "节省": "30-40%",
                    "风险": "长文本抽取质量可能下降"
                },
                "3. 并行化处理": {
                    "策略": "并发调用LLM, 利用asyncio",
                    "节省": "时间成本50-70%",
                    "实现": "见下方代码"
                },
                "4. 缓存LLM结果": {
                    "策略": "对相同输入缓存LLM响应",
                    "节省": "增量更新时70-90%",
                },
                "5. 分层索引": {
                    "策略": "先粗粒度索引, 按需细化",
                    "节省": "初始成本40-50%",
                }
            },
            "查询阶段优化": {
                "1. 查询缓存": {
                    "策略": "缓存相似查询的结果",
                    "节省": "重复查询80%+",
                },
                "2. 限制图遍历深度": {
                    "策略": "局部搜索限制2跳, 关键实体3跳",
                    "节省": "每查询减少30% token",
                },
                "3. 社区过滤": {
                    "策略": "全局搜索只处理相关社区(向量预筛选)",
                    "节省": "Map步骤减少50-70%",
                },
                "4. 使用更小模型做分类": {
                    "策略": "查询分类用小模型, 答案生成用大模型",
                    "节省": "每查询减少20%",
                }
            },
            "存储优化": {
                "1. 向量量化": {
                    "策略": "float32→int8量化, 减少4倍内存",
                    "精度损失": "1-3%",
                },
                "2. 图裁剪": {
                    "策略": "移除低权重边和孤立节点",
                    "效果": "减少20-40%存储",
                }
            }
        }

    @staticmethod
    def implement_caching():
        """实现查询缓存"""
        import hashlib
        from functools import lru_cache

        class CachedGraphRAG:
            def __init__(self, graph_rag):
                self.rag = graph_rag
                self.cache = {}  # query_hash -> response

            def query(self, question: str, method: str = "auto") -> str:
                # 生成缓存key
                cache_key = hashlib.md5(
                    f"{question}:{method}".encode()
                ).hexdigest()

                if cache_key in self.cache:
                    return self.cache[cache_key]

                result = self.rag.query(question, method)
                self.cache[cache_key] = result
                return result

    @staticmethod
    async def parallel_extraction(chunks: List[str],
                                   builder: 'LLMKnowledgeGraphBuilder',
                                   max_concurrent: int = 10):
        """并行实体抽取"""
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(chunk: str):
            async with semaphore:
                # 假设builder有异步版本
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, builder.extract_entities_relations, chunk
                )

        tasks = [process_one(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        # 合并结果
        all_entities = {}
        all_relations = []
        for result in results:
            for entity in result.get("entities", []):
                name = entity["name"]
                if name in all_entities:
                    all_entities[name]["descriptions"].append(entity["description"])
                else:
                    all_entities[name] = {
                        "type": entity["type"],
                        "descriptions": [entity["description"]]
                    }
            all_relations.extend(result.get("relations", []))

        return {"entities": all_entities, "relations": all_relations}


# 展示优化策略
optimizer = GraphRAGOptimizer()
strategies = optimizer.optimization_strategies()
for category, items in strategies.items():
    print(f"\n{category}:")
    for name, detail in items.items():
        print(f"  {name}")
        for k, v in detail.items():
            print(f"    {k}: {v}")
```

### 性能基准测试

```python
import time

def benchmark_graphrag_vs_rag():
    """对比GraphRAG与传统RAG的性能"""
    results = {
        "传统RAG": {
            "索引时间": "~1分钟/100文档",
            "索引成本": "仅embedding费用",
            "查询延迟": "0.5-2秒",
            "全局问题质量": "差(无法综合多文档)",
            "实体问题质量": "中(依赖chunk质量)",
            "维护成本": "低",
        },
        "GraphRAG": {
            "索引时间": "~10分钟/100文档",
            "索引成本": "$3-50/100文档(取决于模型)",
            "查询延迟": "1-5秒(局部), 3-10秒(全局)",
            "全局问题质量": "优秀(社区摘要)",
            "实体问题质量": "优秀(结构化知识)",
            "维护成本": "中高(需要图谱维护)",
        }
    }

    # 适用场景建议
    recommendations = {
        "适合传统RAG的场景": [
            "文档数量少(<1000)",
            "主要回答局部/具体问题",
            "预算有限",
            "快速原型验证",
        ],
        "适合GraphRAG的场景": [
            "需要回答全局性/总结性问题",
            "实体关系复杂, 需要多跳推理",
            "知识需要频繁更新和扩展",
            "对答案质量要求高",
        ],
        "混合方案(推荐)": [
            "简单查询走传统RAG(快速低成本)",
            "复杂查询走GraphRAG(高质量)",
            "用查询分类器自动路由",
        ]
    }

    return results, recommendations
```

---

> **总结**: 知识图谱为LLM提供了结构化的知识表示, GraphRAG通过图检索+社区摘要解决了传统RAG无法回答全局性问题的缺陷。核心难点在于: ①LLM自动构图的质量和成本控制; ②社区检测和摘要的层次化管理; ③索引成本与查询质量的平衡。实践中建议采用"传统RAG + GraphRAG混合"方案, 根据查询复杂度动态路由。
