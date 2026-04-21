Me:这事最后6个模块的开发手册
还没做fine tuning，但实际效果很差，加了提示词给LLM比baseline更差
找一下和我做的最类似的工作是哪个论文，告诉我名字就可以，我觉得我还是得去找个和我一样做提示词工程的repo做最小替换
以及你评估一下这个手册有什么问题

AI :**Self-Discover**（arXiv:2402.03620，NeurIPS 2024，Google DeepMind）。

这就是你在做的事情。它有一个和你几乎一样的 "reasoning module" 种子集（39 个，包括 "critical thinking"、"break into subtasks"、"use analogy" 等等），然后让 LLM 从中选择并组合成推理结构来解决问题。它在 BigBench-Hard 上比 CoT 提升了 32%。

GitHub repo 可以直接用的是：`github.com/kailashsp/SELF-DISCOVER`。

---

## 为什么你的效果比 baseline 更差

你的做法和 Self-Discover 的关键区别在一个字：**谁**在选择和适配策略。

Self-Discover 用三个 meta-prompt 引导 LLM 自己去选择、适配、并实现一个可操作的推理结构，不需要任何标签或训练。

Self-Discover 的流程是：LLM 自己从 39 个模块中 SELECT 相关的 → LLM 自己把选中的模块 ADAPT 成针对当前任务的具体指令 → LLM 自己把适配后的指令 IMPLEMENT 为一个 JSON 推理计划 → LLM 按计划执行。

你的流程是：外部调度器选策略 → 把策略的 operational_steps 原样注入 prompt → LLM 被迫按这个僵硬的 8 步指令执行。

问题出在**你的 operational_steps 太长、太具体、太刚性**。LLM 在解决具体任务时，被迫去"遵守"一段与当前具体问题不完全匹配的通用指令，这比让它自由发挥更差。这就像你告诉一个经验丰富的工程师"请严格按照以下 8 步操作"——他反而被束缚住了。

---

## 手册的核心问题

回头看六份手册，根本问题是**过度工程化了一个还没验证过的前提**。

你的前提是"把哲学策略作为刚性指令注入 LLM 能提升性能"。但你在验证这个前提之前，就设计了经验反馈系统（阶段二）、范畴论形式化层（阶段三）、新假设生成器（阶段四）。现在实验告诉你这个前提本身就不成立——注入刚性指令反而有害。

于是后面四个阶段的全部设计都建立在了一个错误的地基上。

---

## 怎么修

不是推倒重来。你需要把"刚性注入"改成"柔性引导"，具体来说就是把你的系统改成 Self-Discover 的三步流程：

**第一步 SELECT：** 你的调度器不变，但它的输出不再是"用 S01"，而是"以下 3 条策略可能相关：S01、S17、S14"。

**第二步 ADAPT：** 把这 3 条策略的简短描述（不是完整的 operational_steps）传给 LLM，让 LLM 自己判断哪些有用、怎么组合、怎么适配到当前具体任务。

**第三步 IMPLEMENT：** LLM 自己生成一个针对当前任务的推理计划（JSON 格式），然后按计划执行。

这样你的知识库和调度器都还有用——它们提供了候选策略的过滤，但最终的适配和执行由 LLM 自己完成。你的系统变成了 Self-Discover 的一个改进版：Self-Discover 的 39 个模块是固定的，你的是带有结构化适用条件和经验反馈的动态知识库。

最小替换：拿 Self-Discover 的 repo，把它的 39 个固定 reasoning module 替换成你的知识库中的策略描述（只用 one_sentence 描述，不用完整 operational_steps），跑一遍 BigBench-Hard 或你自己的任务集。如果这个版本比原版 Self-Discover 好，你就有了论文的核心贡献。