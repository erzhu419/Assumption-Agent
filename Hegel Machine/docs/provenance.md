# Provenance and reuse

## Source documents

`markdown/` 中五份用户指定文档是逐字复制；另外复制
`category_morphism_dialogue_20260602.md`，因为《黑格尔机》明确引用它作为
diagram/morphism 前史。`source_checksums.sha256` 固定内容。

## v1 snapshot

`legacy/v1_assumption_os/` 保存 v1 的 FrameworkNode/Branch、保守泛化义务、
生命周期、形式证书和模拟器路由的最小源码与定向测试。它只作设计迁移和
回归参照，不是活动依赖。尤其：

- `conservative_generalization_gate_v2.py` 的所谓 real suite 指标仍由固定
  公式/fixture 构成；
- lifecycle 的旧实现以记录为主，不能替代 v3 的强制状态迁移；
- 旧 PASS 和 growth score 不进入 v3 receipts。

## v2 snapshot

`legacy/v2_gscl/` 保存 UAO/meta-assumption、GSCL exact residual、extractor、
controlled corpus 和测试。复制时其中部分文件尚未被原工作树 Git 跟踪，
因此 `source_manifest.tsv` 同时记录原路径、原 tracked 状态和 SHA-256。

v2 提供：

```text
evidence → StructuralEpisode → LawBinding
→ executable residual → partial StructuralCorrespondence → HypothesisClaim
```

v3 提供新的独立 package，把这条结构识别链与 v1 的 theory-growth 治理相接。
活动代码不 import 两个 legacy 目录。

## Reuse rule

复制不是继承证据。旧模块只有在以下条件满足后才可进入活动路径：

1. 在 `src/hegel_machine/` 中以明确合同重写或通过 adapter 隔离；
2. source hash 和语义差异可审计；
3. 新测试使用测得的 receipts，而非旧生成式得分；
4. claim 重新从当前 benchmark 产生。
