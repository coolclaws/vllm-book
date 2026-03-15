import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'vLLM 源码解析',
  description: 'PagedAttention 与高吞吐量 LLM 推理引擎深度剖析',
  lang: 'zh-CN',

  base: '/',

  head: [
    ['link', { rel: 'icon', type: 'image/png', href: '/logo.png' }],
    ['meta', { name: 'theme-color', content: '#06b6d4' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'vLLM 源码解析' }],
    ['meta', { property: 'og:description', content: 'PagedAttention 与高吞吐量 LLM 推理引擎深度剖析' }],
  ],

  themeConfig: {
    logo: { src: '/logo.png', alt: 'vLLM' },

    nav: [
      { text: '开始阅读', link: '/chapters/01-overview' },
      { text: '目录', link: '/contents' },
      { text: 'GitHub', link: 'https://github.com/coolclaws/vllm-book' },
    ],

    sidebar: [
      {
        text: '前言',
        items: [
          { text: '关于本书', link: '/' },
          { text: '完整目录', link: '/contents' },
        ],
      },
      {
        text: '第一部分：宏观认知',
        collapsed: false,
        items: [
          { text: '第 1 章　项目概览与设计哲学', link: '/chapters/01-overview' },
          { text: '第 2 章　Repo 结构与模块依赖', link: '/chapters/02-repo-structure' },
        ],
      },
      {
        text: '第二部分：PagedAttention 核心',
        collapsed: false,
        items: [
          { text: '第 3 章　KV Cache 与 PagedAttention 设计', link: '/chapters/03-kv-cache-paged-attention' },
          { text: '第 4 章　BlockManager 实现', link: '/chapters/04-block-manager' },
          { text: '第 5 章　Attention 算子实现', link: '/chapters/05-attention-kernel' },
        ],
      },
      {
        text: '第三部分：调度系统',
        collapsed: false,
        items: [
          { text: '第 6 章　Scheduler 架构', link: '/chapters/06-scheduler' },
          { text: '第 7 章　序列管理', link: '/chapters/07-sequence' },
          { text: '第 8 章　Preemption 与 Swap', link: '/chapters/08-preemption-swap' },
        ],
      },
      {
        text: '第四部分：执行引擎',
        collapsed: false,
        items: [
          { text: '第 9 章　LLMEngine 总览', link: '/chapters/09-llm-engine' },
          { text: '第 10 章　Worker 与 Executor', link: '/chapters/10-worker-executor' },
          { text: '第 11 章　模型加载与权重', link: '/chapters/11-model-loading' },
          { text: '第 12 章　采样器', link: '/chapters/12-sampler' },
        ],
      },
      {
        text: '第五部分：连续批处理',
        collapsed: false,
        items: [
          { text: '第 13 章　Continuous Batching 原理', link: '/chapters/13-continuous-batching' },
          { text: '第 14 章　Chunked Prefill', link: '/chapters/14-chunked-prefill' },
          { text: '第 15 章　Speculative Decoding', link: '/chapters/15-speculative-decoding' },
        ],
      },
      {
        text: '第六部分：分布式推理',
        collapsed: false,
        items: [
          { text: '第 16 章　Tensor Parallelism', link: '/chapters/16-tensor-parallelism' },
          { text: '第 17 章　Pipeline Parallelism', link: '/chapters/17-pipeline-parallelism' },
          { text: '第 18 章　分布式 KV Cache 与 P/D 分离', link: '/chapters/18-distributed-kv-cache' },
        ],
      },
      {
        text: '第七部分：API 服务层',
        collapsed: false,
        items: [
          { text: '第 19 章　OpenAI 兼容 API', link: '/chapters/19-openai-api' },
          { text: '第 20 章　LoRA 与多 Adapter 服务', link: '/chapters/20-lora-adapter' },
        ],
      },
      {
        text: '第八部分：量化与优化',
        collapsed: false,
        items: [
          { text: '第 21 章　量化支持', link: '/chapters/21-quantization' },
          { text: '第 22 章　性能调优与 Profiling', link: '/chapters/22-performance-tuning' },
        ],
      },
      {
        text: '附录',
        collapsed: true,
        items: [
          { text: '附录 A：推荐阅读路径', link: '/chapters/appendix-a-reading-path' },
          { text: '附录 B：核心类型速查', link: '/chapters/appendix-b-type-reference' },
          { text: '附录 C：名词解释', link: '/chapters/appendix-c-glossary' },
        ],
      },
    ],

    outline: {
      level: [2, 3],
      label: '本页目录',
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/coolclaws/vllm-book' },
    ],

    footer: {
      message: '基于 MIT 协议发布',
      copyright: 'Copyright © 2025-present',
    },

    search: {
      provider: 'local',
    },
  },

  markdown: {
    lineNumbers: true,
  },
})
