# Self-Evolution Paper Evidence Pack

- pass: `True`
- roadmap closure: `19/19` items, bounded UGSE `0.923`
- frozen main: `1768` problems, margin over best baseline `0.0417`
- fresh repaired broad generator: `720` calls, trigger utility `0.5462`, delta `0.1301`
- simulator: leakage audit `True`, production router `True`
- autonomy: supervised production candidate `True`
- formal: bounded `True`, full prover `False`

## Main Tables

| Table | Purpose | Key Metric |
| --- | --- | --- |
| Table 1 | Same-batch frozen benchmark against hard baselines | 1768 problems; margin 0.0417 |
| Table 2 | Fresh repaired broad-generator rerun | 720 calls; trigger 0.5462; delta 0.1301 |
| Table 3 | Framework-growth ablation and open-ended self-evolution | ablation margin 0.1604; open-run score 0.7374 |
| Table 4 | Safety and claim-boundary evidence | simulator leakage pass=True; formal full prover allowed=False |

## Manuscript Skeleton

### Abstract

Present Assumption-Agent as a bounded recursive self-evolution system that treats agent decisions as falsifiable assumptions, with explicit graph memory, residual-derived framework growth, simulator routing, finite formal gates, and supervised graph maintenance.

### Introduction

Motivate self-evolution as conservative generalization: new assumptions must explain residuals, preserve validated old successes, reduce to parent assumptions under old scope conditions, and add testable consequences.

### Related Work

Position against RAG/memory systems, self-reflection agents, AI-scientist-style loops, world models for agents, and category-inspired structural transfer.  State claim boundaries early.

### Assumption Lifecycle Kernel

Define assumption nodes, overlays, verifier stack, trial manifests, residual taxonomy, and gated retention as the core state machine.

### Dialectical Framework Growth

Describe residual-to-framework generation, conservative extension gates, branch ledgers, framework promotion ladders, and selective retention across generations.

### Simulator-Guided Verification

Use the graph-action simulator only for proposal triage and verifier routing.  Include the leakage audit and preserve the block on simulator-as-judge claims.

### Finite Formal Transfer Gates

Present bounded finite diagrams, finite theorem fragments, external Lean checks, and negative controls.  Avoid claiming a full category-theory theorem prover.

### Supervised Autonomy and Main-Graph Maintenance

Report the restricted 30-day-equivalent supervised autonomy run and canary-scope controlled main-graph apply with rollback/readback monitoring.

### Experiments

Report the frozen same-batch table over 1768 problems and the fresh repaired 720-call broad-generator run with delta 0.1301.

### Negative Results and Claim Boundaries

Keep failures and blocked claims: raw unfiltered broad generation failed, simulator cannot replace live validation, autonomy is supervised/bounded, and formal reasoning is finite and scoped.

### Reproducibility

List exact commands, artifact hashes, environment variable names only, redaction policy, and the one-command evidence-pack generation path.

## Claim Boundaries

- `bounded_recursive_self_evolution`: allowed=`True`; Roadmap coverage is closed at the bounded research-prototype level.
- `fresh_repaired_broad_generator`: allowed=`True`; A repaired evidence-calibrated frontier passed a 720-call fresh rerun.
- `unbounded_autonomous_os`: allowed=`False`; Autonomy evidence is supervised and restricted to low-risk graph maintenance.
- `simulator_replaces_live_validation`: allowed=`False`; Simulator evidence is leakage-audited for routing, not judge/live replacement.
- `full_category_theory_theorem_prover`: allowed=`False`; Formal evidence is a bounded finite fragment with external Lean checks.
- `ungated_main_graph_or_policy_mutation`: allowed=`False`; Main-graph changes are canary-scoped; policy/default changes remain gated.

## Repro Commands

- `self_evo_roadmap_coverage`: `python3 -m assumption_os.self_evo_roadmap_coverage_audit --root . --out 'phase four/assumption_graph/paper_readiness_20260604/self_evo_roadmap_coverage_audit_20260612.json'`
- `fresh_broad_generator_repair_integration`: `python3 -m assumption_os.paper_broad_generator_repair_integration --root . --out 'phase four/assumption_graph/paper_readiness_20260604/paper_broad_generator_repair_integration_20260612.json'`
- `paper_evidence_pack`: `python3 -m assumption_os.self_evo_paper_evidence_pack --root . --out 'phase four/assumption_graph/paper_readiness_20260604/self_evo_paper_evidence_pack_20260612.json'`
- `performance_validation`: `python3 -m assumption_os.performance_validation --eval-id performance_validation_self_evo_paper_pack_20260612`
