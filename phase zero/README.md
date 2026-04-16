# Phase Zero: 哲学方法论知识库

## 快速开始

### 1. 配置 API Key

```bash
cp .env.template .env
# 编辑 .env，填入你的 ANTHROPIC_API_KEY
```

### 2. 构建知识库（23 条策略）

```bash
cd scripts

# 先试一条看效果
python build_kb.py --strategy S01

# 验证生成结果
python validate_kb.py

# 构建全部（约 23 次 API 调用，~$0.50）
python build_kb.py --skip-existing
```

### 3. 生成标注问题集（150-200 道）

```bash
# 先生成一个领域试试
python generate_problems.py --domain software_engineering --count 5

# 全部生成（6 次 API 调用，~$0.30）
python generate_problems.py --skip-existing
```

### 4. 自动标注（多标注者一致性分析）

```bash
# 每道题 5 个独立标注（约 1000 次 API 调用，~$3）
python annotate_problems.py --annotators 5 --skip-existing

# 或先小规模测试
python annotate_problems.py --domain software_engineering --annotators 3
```

## 目录结构

```
phase zero/
├── kb/
│   ├── strategies/       # S01-S23 的完整 JSON 文件
│   ├── compositions/     # COMP_001-005 策略组合
│   ├── categories.json   # 策略类别定义
│   └── schema.json       # JSON Schema（TODO）
├── experience_log/       # 阶段二将使用
├── change_history/       # 变更历史 JSONL
├── benchmark/
│   ├── problems/         # 按领域分的问题 JSON
│   ├── annotations/      # 每道题的多标注者标注
│   └── analysis/         # 一致性分析报告
├── scripts/
│   ├── strategy_seeds.py      # 23 条策略的种子定义
│   ├── build_kb.py            # 用 Claude API 扩展为完整 schema
│   ├── generate_problems.py   # 生成标注问题集
│   ├── annotate_problems.py   # 多标注者自动标注 + 一致性分析
│   └── validate_kb.py         # 知识库格式验证
├── .env.template
└── README.md
```

## 预估成本

| 步骤 | API 调用次数 | 预估成本 |
|------|------------|---------|
| build_kb.py (23 策略) | 23 | ~$0.50 |
| generate_problems.py (6 领域) | 6 | ~$0.30 |
| annotate_problems.py (170题 × 5标注) | 850 | ~$2.50 |
| **总计** | **~879** | **~$3.30** |
