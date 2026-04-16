"""
Phase 0: Seed definitions for 23 meta-strategies.
Each seed contains the minimum info needed; build_kb.py will use Claude API
to expand each seed into the full schema (descriptions, cases, conditions, etc.)
"""

STRATEGY_SEEDS = [
    # === CAT_A: 问题分解与简化 ===
    {"id": "S01", "name_zh": "控制变量法", "name_en": "Controlled Variable Method",
     "one_sentence": "固定其他条件，每次只改变一个因素",
     "category": "empirical_testing", "sources": ["Mill 1843", "Bacon 1620"]},
    {"id": "S02", "name_zh": "分而治之", "name_en": "Divide and Conquer",
     "one_sentence": "将复杂问题分解为可独立解决的子问题",
     "category": "decomposition", "sources": ["Polya 1945", "Knuth"]},
    {"id": "S09", "name_zh": "降维/简化", "name_en": "Dimension Reduction / Simplification",
     "one_sentence": "去掉不影响核心的复杂因素，先解决简化版",
     "category": "decomposition", "sources": ["Polya 1945", "Simon 1969"]},
    {"id": "S06", "name_zh": "先特殊后一般", "name_en": "Specialization Before Generalization",
     "one_sentence": "先解决最简单的特殊情况，再逐步推广",
     "category": "decomposition", "sources": ["Polya 1945", "Lakatos 1976"]},

    # === CAT_B: 搜索与探索 ===
    {"id": "S07", "name_zh": "反向推理", "name_en": "Backward Reasoning",
     "one_sentence": "从目标状态出发，反推需要什么前提条件",
     "category": "search", "sources": ["Polya 1945"]},
    {"id": "S08", "name_zh": "试错法/猜测-检验", "name_en": "Trial and Error / Guess-and-Check",
     "one_sentence": "提出一个猜测，测试，根据结果修正",
     "category": "search", "sources": ["Popper 1959", "Polya 1945"]},
    {"id": "S19", "name_zh": "约束松弛", "name_en": "Constraint Relaxation",
     "one_sentence": "暂时放宽某些约束条件，看问题是否变得可解",
     "category": "search", "sources": ["运筹学传统", "Polya 1945"]},

    # === CAT_C: 推理与证明 ===
    {"id": "S03", "name_zh": "类比推理", "name_en": "Analogical Reasoning",
     "one_sentence": "将当前问题映射到一个已解决的相似问题",
     "category": "reasoning", "sources": ["Polya 1945", "Aristotle"]},
    {"id": "S04", "name_zh": "反证法/归谬法", "name_en": "Proof by Contradiction / Reductio ad Absurdum",
     "one_sentence": "假设结论不成立，推导出矛盾",
     "category": "reasoning", "sources": ["Aristotle", "Polya 1945"]},
    {"id": "S18", "name_zh": "抽象化/泛化", "name_en": "Abstraction / Generalization",
     "one_sentence": "去掉具体细节，提取问题的抽象结构",
     "category": "reasoning", "sources": ["Polya 1945", "范畴论传统"]},
    {"id": "S20", "name_zh": "对偶/互补视角", "name_en": "Dual / Complementary Perspective",
     "one_sentence": "从问题的对立面或互补角度重新审视",
     "category": "reasoning", "sources": ["物理学传统", "哲学辩证法"]},

    # === CAT_D: 实证与检验 ===
    {"id": "S13", "name_zh": "证伪优先", "name_en": "Falsification First",
     "one_sentence": "优先寻找能推翻当前假设的证据，而非确认它",
     "category": "empirical_testing", "sources": ["Popper 1959"]},
    {"id": "S14", "name_zh": "边界条件分析", "name_en": "Boundary Condition Analysis",
     "one_sentence": "检查极端情况和边界值来测试假设的稳健性",
     "category": "empirical_testing", "sources": ["工程传统", "Polya 1945"]},
    {"id": "S16", "name_zh": "求同法", "name_en": "Method of Agreement",
     "one_sentence": "找出所有成功案例的共同因素",
     "category": "empirical_testing", "sources": ["Mill 1843"]},
    {"id": "S17", "name_zh": "求异法", "name_en": "Method of Difference",
     "one_sentence": "找出成功和失败案例之间的关键差异",
     "category": "empirical_testing", "sources": ["Mill 1843"]},

    # === CAT_E: 评估与选择 ===
    {"id": "S05", "name_zh": "奥卡姆剃刀", "name_en": "Occam's Razor",
     "one_sentence": "在同等解释力下选择更简单的假设",
     "category": "evaluation", "sources": ["William of Ockham"]},
    {"id": "S10", "name_zh": "对称性利用", "name_en": "Symmetry Exploitation",
     "one_sentence": "识别问题中的对称结构以减少搜索空间",
     "category": "evaluation", "sources": ["Polya 1945", "物理学传统"]},
    {"id": "S11", "name_zh": "满意化", "name_en": "Satisficing",
     "one_sentence": "不追求最优解，找到第一个足够好的解即停止",
     "category": "evaluation", "sources": ["Simon 1969"]},
    {"id": "S12", "name_zh": "贝叶斯更新", "name_en": "Bayesian Updating",
     "one_sentence": "用新证据持续更新对假设的信念强度",
     "category": "evaluation", "sources": ["Bayes", "Kahneman 2011"]},

    # === CAT_F: 构建与迭代 ===
    {"id": "S15", "name_zh": "增量构建", "name_en": "Incremental Building",
     "one_sentence": "从最小可工作版本开始，逐步添加功能",
     "category": "construction", "sources": ["敏捷开发", "Deming 1986"]},

    # === CAT_G: 元决策与终止 ===
    {"id": "S21", "name_zh": "死胡同识别/及时止损", "name_en": "Dead End Recognition / Cut Losses",
     "one_sentence": "识别当前路径已不可能成功，放弃并回溯到更高层决策点",
     "category": "meta_decision", "sources": ["Kahneman 2011 (沉没成本谬误)", "Simon 1969"]},
    {"id": "S22", "name_zh": "问题重构", "name_en": "Problem Reframing",
     "one_sentence": "不在当前框架内继续尝试，而是重新定义问题本身",
     "category": "meta_decision", "sources": ["Kuhn 1962 (范式转换)", "Polya 1945"]},
    {"id": "S23", "name_zh": "资源约束下的近似接受", "name_en": "Approximate Acceptance Under Resource Constraints",
     "one_sentence": "资源即将耗尽时，接受当前最优近似解并停止",
     "category": "meta_decision", "sources": ["Simon 1969 (满意化极端版)", "工程传统"]},

    # === CAT_H: 系统结构与涌现 ===
    {"id": "S24", "name_zh": "关键节点/瓶颈识别", "name_en": "Critical Node / Bottleneck Identification",
     "one_sentence": "在分析系统变革可行性时，先识别关键决策节点及其利益方向",
     "category": "system_structure", "sources": ["政治学（制度分析）", "运筹学（TOC 约束理论）", "网络科学"]},
    {"id": "S25", "name_zh": "涌现性检测", "name_en": "Emergence Detection",
     "one_sentence": "部分的属性不能线性外推到整体——组合后需要重新评估涌现属性",
     "category": "system_structure", "sources": ["复杂系统科学", "化学（非加性效应）", "语言学（组合语义）"]},
    {"id": "S26", "name_zh": "路径依赖分析", "name_en": "Path Dependency Analysis",
     "one_sentence": "识别当前状态是由历史路径决定的，而非由当前条件唯一确定",
     "category": "system_structure", "sources": ["演化生物学", "经济史 (Arthur 1994)", "软件工程（技术债）"]},
    {"id": "S27", "name_zh": "激励结构分析", "name_en": "Incentive Structure Analysis",
     "one_sentence": "在多方参与的系统中，分析每个参与者的激励方向是否与目标对齐",
     "category": "system_structure", "sources": ["博弈论（纳什均衡）", "制度经济学", "组织行为学"]},
]

COMPOSITION_SEEDS = [
    {"id": "COMP_001", "name_zh": "先简化再排查", "name_en": "Simplify-then-Isolate",
     "sequence": ["S06", "S01"],
     "transition_condition": "S06 成功将问题简化为可控规模后，切换到 S01 逐一检查"},
    {"id": "COMP_002", "name_zh": "分解后增量", "name_en": "Decompose-then-Build",
     "sequence": ["S02", "S15"],
     "transition_condition": "S02 成功分解为独立子问题后，用 S15 逐步构建各子问题的解"},
    {"id": "COMP_003", "name_zh": "类比后验证", "name_en": "Analogize-then-Falsify",
     "sequence": ["S03", "S13"],
     "transition_condition": "S03 找到类似问题的解后，用 S13 证伪法验证是否真正适用"},
    {"id": "COMP_004", "name_zh": "边界分析后反证", "name_en": "Boundary-then-Contradict",
     "sequence": ["S14", "S04"],
     "transition_condition": "S14 找到边界条件后，用 S04 反证法证明一般情况"},
    {"id": "COMP_005", "name_zh": "估计后精炼", "name_en": "Estimate-then-Refine",
     "sequence": ["S12", "S01"],
     "transition_condition": "S12 用贝叶斯估计缩小搜索范围后，S01 逐一精确排查"},
]

CATEGORY_DEFINITIONS = {
    "CAT_A": {"name": "问题分解与简化", "description": "将问题变小或变简单", "strategies": ["S02", "S09", "S06"]},
    "CAT_B": {"name": "搜索与探索", "description": "在解空间中寻找答案", "strategies": ["S07", "S08", "S19"]},
    "CAT_C": {"name": "推理与证明", "description": "逻辑推导和论证", "strategies": ["S03", "S04", "S18", "S20"]},
    "CAT_D": {"name": "实证与检验", "description": "通过实验或观察验证假设", "strategies": ["S01", "S13", "S14", "S16", "S17"]},
    "CAT_E": {"name": "评估与选择", "description": "在多个候选方案中做选择", "strategies": ["S05", "S10", "S11", "S12"]},
    "CAT_F": {"name": "构建与迭代", "description": "逐步构造解", "strategies": ["S15"]},
    "CAT_G": {"name": "元决策与终止", "description": "决定是否继续当前路径", "strategies": ["S21", "S22", "S23"]},
    "CAT_H": {"name": "系统结构与涌现", "description": "分析多方/多部分系统的非线性行为", "strategies": ["S24", "S25", "S26", "S27"]},
}
