<div align="center">

<img src="assets/banner.svg" width="100%" alt="LLM & Agent 面试题全集">

<br>

### 🎯 LLM & Agent 面试题全集

[![Stars](https://img.shields.io/github/stars/dirjaker/llm_agent_interview?style=flat-square&label=Stars&color=FFD700)](https://github.com/dirjaker/llm_agent_interview/stargazers)
[![Forks](https://img.shields.io/github/forks/dirjaker/llm_agent_interview?style=flat-square&label=Forks&color=4A90D9)](https://github.com/dirjaker/llm_agent_interview/network/members)
[![Contributors](https://img.shields.io/github/contributors/dirjaker/llm_agent_interview?style=flat-square&label=Contributors&color=8B4513)](https://github.com/dirjaker/llm_agent_interview/graphs/contributors)
[![License](https://img.shields.io/github/license/dirjaker/llm_agent_interview?style=flat-square&label=License&color=20B2AA)](https://github.com/dirjaker/llm_agent_interview/blob/dev/LICENSE)

</div>

---

## ✨ 功能特性

| 功能 | 描述 |
|------|------|
| 📚 **47 个专题** | Python → LLM原理 → Agent → 模型优化 → 工程化 → 生产实战 → 前沿 → 网关运维 |
| 📝 **600+ 道题** | 精选高频面试题，含详细解答、追问场景、代码示例 |
| 📖 **82,000+ 行** | 深度内容，每章平均 1,600 行，不是浅尝辄止的面试宝典 |
| 🔧 **9 个工具** | 速查表、Top50 高频题、术语表、公司面试风格、调试场景集等 |
| 🎯 **真实面试题** | 来自大厂 AI 工程师岗位的真实面试题和实战难题 |
| 💡 **追问场景** | 每道题附带面试官可能的追问和回答策略 |
| 📊 **难度分级** | ⭐ 基础 / ⭐⭐ 进阶 / ⭐⭐⭐ 高级三级难度标注 |

## 🗺️ 学习路径

> 从零基础到面试冲刺，按照 Phase 顺序逐步学习

| 阶段 | 重点 | 适合人群 |
|------|------|---------|
| 🟢 **Phase 1** 基础知识 | Python 高级编程、LLM 原理 | 入门必读 |
| 🔵 **Phase 2** 核心技能 | Prompt/RAG/Agent/LangChain | 核心能力 |
| 🟣 **Phase 3** 模型优化 | 微调训练、部署推理 | 进阶提升 |
| 🟠 **Phase 4** 工程化 | FastAPI/评估/安全/CICD/数据工程 | 工程落地 |
| 🔴 **Phase 5** 生产实战 | 生产环境、系统设计、算法编码 | 实战检验 |
| 🟡 **Phase 6** 面试冲刺 | 前沿方向、行为面试、多模态等 | 面试前最后冲刺 |
| 🟢 **Phase 7** 前沿专题 | 前沿架构、Agent Loop、Context Engineering | 加分项 |
| 🔵 **Phase 8** ML/DL 基础 | 经典机器学习、深度学习、NLP | 理论根基 |
| 🟣 **Phase 9** 实战工程 | Agent 开发实战、LLM 应用工程 | 项目经验 |
| 🟠 **Phase 10** 进阶专题 | LangChain/RAG/Agent 进阶 | 深度拓展 |
| ⚪ **Phase 11** 网关运维 | LLM 网关、Token 治理、可观测性 | 架构师方向 |

## 📖 在线文档

<div align="center">

**📚 [点击访问在线文档](https://dirjaker.github.io/llm_agent_interview/)**

</div>

文档链接：
| 文档 | 说明 |
|------|------|
| [面试速查表](https://dirjaker.github.io/llm_agent_interview/工具箱/CHEATSHEET) | 面试前 30 分钟快速复习 |
| [Top50 高频题](https://dirjaker.github.io/llm_agent_interview/工具箱/TOP50-高频题) | 最常考 50 题浓缩答案 |
| [术语表](https://dirjaker.github.io/llm_agent_interview/工具箱/GLOSSARY) | 120+ 术语中英对照 |
| [公司面试风格](https://dirjaker.github.io/llm_agent_interview/工具箱/COMPANY-MAP) | 7 家公司面试特点 |
| [调试场景集](https://dirjaker.github.io/llm_agent_interview/工具箱/DEBUG-SCENARIOS) | 10 个生产故障排查 |
| [系统设计模板](https://dirjaker.github.io/llm_agent_interview/工具箱/SYSTEM-DESIGN-TEMPLATES) | 5 类系统设计答题框架 |
| [更新日志](https://dirjaker.github.io/llm_agent_interview/CHANGELOG) | 版本更新记录 |

## 📂 项目结构

```
llm_agent_interview/
├── README.md                      # 项目说明
├── LICENSE                        # MIT 许可证
├── docs/                          # VitePress 文档站点
│   ├── index.md                   # 首页
│   ├── .vitepress/                # VitePress 配置
│   ├── 01-基础知识/               # Phase 1: Python + LLM 基础
│   │   ├── 01-Python高级编程.md
│   │   ├── 02-大模型发展脉络.md
│   │   └── 03-LLM基础原理.md
│   ├── 02-核心技能/               # Phase 2: Prompt/RAG/Agent
│   │   ├── 04-Prompt工程.md
│   │   ├── 05-RAG系统设计.md
│   │   ├── 06-Agent架构与实现.md
│   │   ├── 07-记忆系统.md
│   │   ├── 08-多智能体系统.md
│   │   ├── 09-MCP与工具生态.md
│   │   ├── 10-Agent工程实践.md
│   │   └── 11-LangChain与LangGraph.md
│   ├── 03-模型优化/               # Phase 3: 微调 + 部署
│   ├── 04-工程化/                 # Phase 4: FastAPI/安全/CICD
│   ├── 05-生产实战/               # Phase 5: 生产/系统设计/编码
│   ├── 06-面试冲刺/               # Phase 6: 冲刺 + 软技能
│   ├── 07-前沿专题/               # Phase 7: 前沿架构
│   ├── 08-进阶专题/               # Phase 10: 进阶深度
│   ├── 09-大模型网关与运维/        # Phase 11: 网关运维
│   ├── 34-40 号章节               # Phase 8-9: ML/DL + 实战
│   └── 工具箱/                    # 面试工具箱 (9 个工具)
└── .github/workflows/             # CI: GitHub Pages 自动部署
```

## 🚀 快速开始

```bash
# 克隆项目
git clone https://github.com/dirjaker/llm_agent_interview.git
cd llm_agent_interview

# 安装文档依赖
npm install

# 本地预览
npx vitepress dev docs
```

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| **文档引擎** | VitePress 1.6 |
| **搜索** | 内置本地搜索 |
| **部署** | GitHub Pages + Actions |
| **内容** | Markdown (中文) |

## 📝 开发日志

- [x] 47 个专题全部完成
- [x] 600+ 道面试题含解答与追问
- [x] 9 个面试工具箱
- [x] 11 阶段学习路径
- [x] VitePress 文档站点 + GitHub Pages
- [x] Phase 7 前沿专题 (2025-2026)
- [x] Phase 8 ML/DL 基础
- [x] Phase 9 实战工程
- [x] Phase 10 进阶专题
- [x] Phase 11 大模型网关与运维
- [ ] 视频讲解
- [ ] 模拟面试功能

## 📄 许可证

[MIT License](LICENSE)

---

<div align="center">

🔗 **GitHub**: [dirjaker/llm_agent_interview](https://github.com/dirjaker/llm_agent_interview)

⭐ 如果这个项目对你有帮助，请给一个 Star 支持一下！

</div>
