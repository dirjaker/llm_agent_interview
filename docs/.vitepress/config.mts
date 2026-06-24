import { defineConfig } from 'vitepress'

export default defineConfig({
  base: '/llm_agent_interview/',
  title: 'LLM Agent Interview',
  description: 'LLM 应用 & Agent 开发 · 面试题与实战难题全集',
  lang: 'zh-CN',
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '基础知识', link: '/01-基础知识/01-Python高级编程' },
      { text: '核心技能', link: '/02-核心技能/04-Prompt工程' },
      { text: '生产实战', link: '/05-生产实战/20-生产环境实战' },
      { text: '工具箱', link: '/工具箱/CHEATSHEET' },
      { text: 'GitHub', link: 'https://github.com/dirjaker/llm_agent_interview' }
    ],
    sidebar: [
      {
        text: '🟢 Phase 1 · 基础知识',
        collapsed: false,
        items: [
          { text: '01 - Python高级编程', link: '/01-基础知识/01-Python高级编程' },
          { text: '02 - 大模型发展脉络', link: '/01-基础知识/02-大模型发展脉络' },
          { text: '03 - LLM基础原理', link: '/01-基础知识/03-LLM基础原理' }
        ]
      },
      {
        text: '🔵 Phase 2 · 核心技能',
        collapsed: false,
        items: [
          { text: '04 - Prompt工程', link: '/02-核心技能/04-Prompt工程' },
          { text: '05 - RAG系统设计', link: '/02-核心技能/05-RAG系统设计' },
          { text: '06 - Agent架构与实现', link: '/02-核心技能/06-Agent架构与实现' },
          { text: '07 - 记忆系统', link: '/02-核心技能/07-记忆系统' },
          { text: '08 - 多智能体系统', link: '/02-核心技能/08-多智能体系统' },
          { text: '09 - MCP与工具生态', link: '/02-核心技能/09-MCP与工具生态' },
          { text: '10 - Agent工程实践', link: '/02-核心技能/10-Agent工程实践' },
          { text: '11 - LangChain与LangGraph', link: '/02-核心技能/11-LangChain与LangGraph' }
        ]
      },
      {
        text: '🟣 Phase 3 · 模型优化',
        collapsed: false,
        items: [
          { text: '12 - 模型微调与训练', link: '/03-模型优化/12-模型微调与训练' },
          { text: '13 - 模型部署与推理优化', link: '/03-模型优化/13-模型部署与推理优化' }
        ]
      },
      {
        text: '🟠 Phase 4 · 工程化',
        collapsed: false,
        items: [
          { text: '14 - FastAPI与服务开发', link: '/04-工程化/14-FastAPI与服务开发' },
          { text: '15 - 评估与测试', link: '/04-工程化/15-评估与测试' },
          { text: '16 - 安全与防护', link: '/04-工程化/16-安全与防护' },
          { text: '17 - CICD与MLOps', link: '/04-工程化/17-CICD与MLOps' },
          { text: '18 - 语音与实时交互', link: '/04-工程化/18-语音与实时交互' },
          { text: '19 - LLM应用数据工程', link: '/04-工程化/19-LLM应用数据工程' }
        ]
      },
      {
        text: '🔴 Phase 5 · 生产实战',
        collapsed: false,
        items: [
          { text: '20 - 生产环境实战', link: '/05-生产实战/20-生产环境实战' },
          { text: '21 - 系统设计题', link: '/05-生产实战/21-系统设计题' },
          { text: '22 - 算法与编码题', link: '/05-生产实战/22-算法与编码题' }
        ]
      },
      {
        text: '🔴 Phase 6 · 面试冲刺',
        collapsed: false,
        items: [
          { text: '23 - 大模型前沿方向', link: '/06-面试冲刺/23-大模型前沿方向' },
          { text: '24 - 行为面试与软技能', link: '/06-面试冲刺/24-行为面试与软技能' },
          { text: '25 - 多模态应用开发', link: '/06-面试冲刺/25-多模态应用开发' },
          { text: '26 - Embedding与向量数据库', link: '/06-面试冲刺/26-Embedding与向量数据库' },
          { text: '27 - 知识图谱与GraphRAG', link: '/06-面试冲刺/27-知识图谱与GraphRAG' }
        ]
      },
      {
        text: '🚀 Phase 7 · 前沿专题 (2025-2026)',
        collapsed: false,
        items: [
          { text: '28 - 前沿大模型架构', link: '/07-前沿专题/28-前沿大模型架构' },
          { text: '29 - Agent Loop Engineering', link: '/07-前沿专题/29-Agent-Loop-Engineering' },
          { text: '30 - Hermes Engineering', link: '/07-前沿专题/30-Hermes-Engineering' },
          { text: '31 - Context Engineering', link: '/07-前沿专题/31-Context-Engineering' },
          { text: '32 - Agentic Coding', link: '/07-前沿专题/32-Agentic-Coding' },
          { text: '33 - Compound AI Systems', link: '/07-前沿专题/33-Compound-AI-Systems' }
        ]
      },
      {
        text: '🧮 Phase 8 · ML/DL 基础',
        collapsed: false,
        items: [
          { text: '34 - 经典机器学习基础', link: '/34-经典机器学习基础' },
          { text: '35 - 深度学习核心机制', link: '/35-深度学习核心机制' },
          { text: '36 - 优化器与训练技巧', link: '/36-优化器与训练技巧' },
          { text: '37 - NLP基础与技术演进', link: '/37-NLP基础与技术演进' }
        ]
      },
      {
        text: '📎 面试工具箱',
        collapsed: false,
        items: [
          { text: '速查手册', link: '/工具箱/CHEATSHEET' },
          { text: 'Top 50 高频题', link: '/工具箱/TOP50-高频题' },
          { text: '术语表', link: '/工具箱/GLOSSARY' },
          { text: '公司面试风格', link: '/工具箱/COMPANY-MAP' },
          { text: 'Debug 场景', link: '/工具箱/DEBUG-SCENARIOS' },
          { text: '系统设计模板', link: '/工具箱/SYSTEM-DESIGN-TEMPLATES' },
          { text: '项目展示', link: '/工具箱/PROJECT-SHOWCASE' },
          { text: '成本计算器', link: '/工具箱/COST-CALCULATOR' },
          { text: '持续学习资源', link: '/工具箱/STAYING-CURRENT' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/dirjaker/llm_agent_interview' }
    ],
    search: {
      provider: 'local'
    },
    footer: {
      message: 'LLM 应用 & Agent 开发面试准备',
      copyright: '© 2026 dirjaker'
    }
  }
})
