# Release Status

Status date: 2026-06-13

Active branch: `reconstruction-v2`

Current claim level: L3.5 bounded recursive self-evolution prototype.

## What Is Implemented

- Assumption Graph memory with typed assumption nodes, evidence, trials, and lifecycle journals.
- Recursive hypothesis generation, falsification gates, selective retention, and gated graph apply.
- Bounded dialectical framework growth: branch, candidate framework, active scoped framework, demotion, rollback, and negative evidence retention.
- Category-inspired structural morphism layer with finite certificates and Lean-checked theorem fragments.
- Simulator-guided routing as a cheap gate/router, not a replacement for live validation or judges.
- Paper-facing evidence pack with frozen benchmark evidence, 720-call repaired broad-generator evidence, claim-frontier ledger, and external-review packet.
- Hegel framework-evolution R1-R9 artifacts: object model, prior library, residual-to-framework generator, conservative generalization gate v2, lifecycle ledger, simulator-guided search, formal certificates, multi-generation benchmark, and external evaluation pack.

## Claim Boundaries

- This branch does not claim an unbounded 24/7 autonomous self-evolution OS.
- The world model is a router and budget gate, not a production simulator that replaces live ablation or judge calls.
- The formal layer is a bounded finite proof/certificate stack, not a full category-theory theorem prover.
- Human expert evaluation is packaged and proxy-preflighted, but a completed human panel is not fabricated.
- Fresh live LLM-generated framework candidates require API credentials in environment variables and an explicit `--execute-live` run.
- Main graph mutation remains gated and canary-scoped.

## Reproduce

Run the main performance validation:

```bash
python3 -m assumption_os.performance_validation \
  --root . \
  --graph-dir "phase four/assumption_graph" \
  --eval-id hegel_release_validation_20260613 \
  --summary-out "phase four/assumption_graph/paper_readiness_20260604/performance_validation_hegel_20260613.json" \
  --report-out "reconstruction/md/performance_validation_hegel_20260613.md"
```

Run the Hegel coverage audit:

```bash
python3 -m assumption_os.hegel_assumption_coverage_audit \
  --root . \
  --out "phase four/assumption_graph/paper_readiness_20260604/hegel_assumption_coverage_audit_20260613.json" \
  --md-out "reconstruction/md/hegel_assumption_coverage_audit_20260613.md"
```

Run the LLM-framework-candidate preflight:

```bash
python3 -m assumption_os.llm_framework_candidate_experiment \
  --root . \
  --out "phase four/assumption_graph/paper_readiness_20260604/llm_framework_candidate_experiment_20260613.json" \
  --md-out "reconstruction/md/llm_framework_candidate_experiment_20260613.md"
```

For a real live LLM synthesis run, set credentials outside the repository:

```bash
export RUOLI_GPT_KEY="<set-outside-code>"
export RUOLI_BASE_URL="https://ruoli.dev"
python3 -m assumption_os.llm_framework_candidate_experiment --root . --execute-live --model gpt-5.4-mini
```
