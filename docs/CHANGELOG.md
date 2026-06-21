# 更新日志

本文档记录项目的主要变更。

---

## 2026-06-22

### 文档维护
- 同步所有顶层 `.md` 文件与 `docs/工具箱/` 内容一致
- 同步所有扁平 `docs/*.md` 文件与分章节目录版本一致（修复 `×` 与 `&times;` 差异）
- 更新 README.md 统计数据：27 个专题、450+ 道题、56000+ 行、9 个工具
- 新增顶层 `TOP50-高频题.md`
- 生成 `docs/CHANGELOG.md`

---

## 2026-06-21

### Bug 修复
- 删除临时备份文件
- 转义行内 `{{}}` 防止 Vue 模板编译失败，代码块加 `v-pre`
- 修复 SVG 文件末尾多余内容
- 缩小主标题字号防止文字溢出

---

## 2026-06-20

### 功能新增
- 统一 README 科技风格 + 添加 MIT License + SVG Banner
- 项目大重构 — 合并/新增/工具箱
- 全面扩充 — 新增 6 章 + 9 工具 + 扩充 5 章
- 优化项目结构，分 6 个阶段 27 个专题

### 新增章节
- 25 - 多模态应用开发
- 26 - Embedding与向量数据库
- 27 - 知识图谱与GraphRAG

### 新增工具箱
- 速查手册 (CHEATSHEET)
- Top 50 高频题 (TOP50-高频题)
- 术语表 (GLOSSARY)
- 公司面试风格 (COMPANY-MAP)
- Debug 场景 (DEBUG-SCENARIOS)
- 系统设计模板 (SYSTEM-DESIGN-TEMPLATES)
- 项目展示 (PROJECT-SHOWCASE)
- 成本计算器 (COST-CALCULATOR)
- 持续学习资源 (STAYING-CURRENT)

---

## 2026-06-19

### 功能新增
- 新增第 22 章 - 第三方 API 调用与集成
- 新增 21 - 大模型前沿方向
- 新增 17 - CICD与MLOps
- 新增 16 - 大模型发展脉络
- 新增 15 - Agent工程实践
- 新增 MCP 章节 + 扩充推理模型/Context Caching/Agentic RAG
- 新增 50+ 道面试题，覆盖前沿技术和深度考点

### Bug 修复
- 修复内容错误（章节编号/代码损坏/变量名）

### 重构
- 按学习路径重新排序章节（6 个阶段 20 章）

---

## 2026-06-18

### 功能新增
- 新增 Python/FastAPI/LangChain 三章
- 扩充微调和部署实战指南
- 新增 VitePress 文档站点
- GitHub Pages 部署
