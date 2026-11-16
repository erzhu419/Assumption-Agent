# Hegel Machine a66cd82b 审阅包

## 阅读顺序

1. `01_系统审计与下一阶段方案.md`
2. `02_下一施工合同.md`
3. `04_差距与责任矩阵.md`
4. `03_来源与核查范围.md`

机器可读文件：`findings.json`、`source_manifest.json`、`audit_scope.json`。

## 已运行的小型检查

```bash
python probe_certificate_domain.py --output result.json
```

默认模式只检查从固定源码读出的精确代数条件，**不声称运行了仓库验证器**。本轮结果已保存在`certificate_domain_reference_result.json`。

## 在本地固定提交上直接检查

```bash
python probe_certificate_domain.py \
  --repo '/path/to/Assumption-Agent' \
  --output actual_repo_probe.json
```

该模式需要仓库模块的NumPy/SciPy/SymPy依赖已经安装。它只导入指定文件、调用验证器和小型实验室，不调用API、不改仓库或历史回执，并关闭字节码写入。结果包含实际模块的Git blob SHA1，便于确认是否为a66代码。

此模式本轮未执行。修复后再次运行时，不应再在同一合法语义下同时接受可达和无条件不可达；若将域解释为真实合法域，就必须一致拒绝离域证人；若域只是分析域，就应拒绝/限定无覆盖证明的不可达结论。

## 重要边界

审阅没有执行13套科学研究、完整测试、付费LLM、Lean编译或外部COW磁盘检查。没有把已记录的高通过率解释为跨所有世界的普适能力。当前建议是修复可信语义与归因，并做一条有实际内容的持久研究链，不是扩大物理家族或新建形式化基础设施。
