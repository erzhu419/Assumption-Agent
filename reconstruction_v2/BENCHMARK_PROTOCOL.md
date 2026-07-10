# Benchmark Protocol

## Why SkillLearnBench First

HLE is primarily a one-shot, broad-knowledge exam. It can test external transfer, but it gives the system little repeated feedback from which to learn reusable assumptions.

SkillLearnBench is the primary environment because it has executable task verifiers, skill-quality metrics, and trajectory keypoints. The upstream inventory has 20 task families and 100 verified instances. The frozen paper subset has 19 families and 95 instances: it excludes the complete `github-repo-analytics` family because all five tasks declare the private external credential `GH_TOKEN`. These correspond directly to:

- hypothesis quality;
- hypothesis application fidelity;
- final task outcome.

## Protocol A: Instance Holdout

Purpose: test whether a hypothesis learned from earlier instances of one task family improves unseen instances of that family.

- Train: 42 instances.
- Validation: 18 instances.
- Sealed test: 35 instances.
- Test instructions are inaccessible until the archive is frozen.
- Hypothesis generation and recursive repair use train only.
- Policy and evaluator selection use validation only.

This is the main test of executable assumption learning.

## Protocol B: Family Out

Purpose: test whether policy and evaluator hypotheses generalize beyond task families used for evolution.

- Train: 11 families, 54 instances.
- Validation: 3 families, 14 instances.
- Sealed test: 5 families, 27 instances.

Task-specific hypotheses are not expected to transfer here. Policy hypotheses, evaluator hypotheses, and the proposal procedure are.

## Controls

Every reported comparison must include:

1. `raw_no_skill`: no-skill agent, the primary baseline;
2. `static_generic_v2`: one frozen generic requirement/audit skill, the no-learning control;
3. `v2_no_recursive_repair`: the same evolution budget with recursive proposal repair disabled;
4. `promoted_v2`: the frozen incumbent from the full recursive loop, the primary candidate;
5. `skilllearn_b1_sonnet`: the upstream one-shot static skill reference;
6. `human_authored`: the upstream human-authored upper reference.

Model, tool access, trial count, retries, and inference budget must be matched. Report task success, skill quality, trajectory quality, model calls, wall time, and failure rate.

The full and no-recursive evolution arms each have a predeclared maximum of three generations and stop after two consecutive non-promotions, no residuals, or duplicate behavior. Later proposals may receive only aggregate prior hypothesis status and promotion summaries; validation instructions and per-item outcomes are not returned to the proposal model.

The two arms branch from one immutable first-generation checkpoint: identical raw train observations, identical mined residuals, and identical proposed root programs. Training is not rerun for the no-recursive arm. After that branch, each arm evolves only from its own archive and aggregate feedback. This makes recursive repair, rather than model-sampling drift or a different training replay, the causal ablation.

All proposed roots receive train-only schema, runtime-feature-vocabulary, trigger-support, executable-action, and evaluator-epoch checks. Instruction text may inform the action graph but is forbidden as a trigger key because it is not a runtime feature. If more than one root passes, one candidate is selected before any validation run by maximum residual support, then minimum anti-support, predicate count, action count, and finally payload hash. Other valid roots remain shadow hypotheses. Validation outcomes never choose among same-generation proposals.

This primary runtime experiment admits only task and policy hypotheses. An evaluator hypothesis cannot be compiled into an agent skill, because that would relabel agent guidance as evaluator evolution. Evaluator changes require the separate fixed-anchor epoch-challenger protocol and are outside the primary performance claim until that path is run.

Reports separate aggregate static-tree recursion across every proposed root from recursion depth in the ultimately selected tree. This prevents a repaired but non-selected candidate from disappearing from the recursive/no-recursive mechanism audit.

Policy-off and policy-on trials use one fairness fingerprint derived from backend, provider, agent, model, step budget, verifier isolation, runner-local agent-registry isolation, image-cache policy, and the actual shared agent-runtime key. Their order is deterministically balanced by pair ID. A provider, budget, or runtime mismatch invalidates the pair. Parallel backends receive deep-copied upstream agent registries so one subscription context cannot clear or restore another backend's agent definition.

Execution caches only the exact non-oracle task environment. Oracle `skills/` content is excluded from the environment hash and build context. Before development, all train/validation images must pass a bounded prewarm gate without invoking the model; the signed receipt becomes part of the development report and archive-freeze checks. Node is pinned by image digest, Codex CLI is pinned to `0.144.1`, and one read-only runtime volume is shared by all item images. Parallelism is across benchmark items only. Every variant for one item runs sequentially in a protocol-derived balanced order, so raw and candidate do not contend against each other inside one pair.

Compiled hypotheses use a content-hashed per-item routing manifest. A skill is injected only when that exact item's structured features satisfy its trigger; sharing a task family with a matched item is insufficient. Missing routes are explicit abstentions. Upstream human and B1 controls retain their native family-level layout because they do not declare V2 triggers.

Proposal-provider failover is allowed only before a candidate program is fixed, under the run's declared provider chain. The selected provider and chain hash become part of candidate provenance. Trial-side policy-off/policy-on pairs still require identical providers; one variant cannot receive a healthier endpoint or a different subscription route.

For Codex subscription trials, both variants mount independently materialized copies of the same local auth source under the same provider policy. Tokens are never serialized into requests, reports, or trial artifacts. Auth materialization failure invalidates the pair before task scoring.

Verifier code is not visible to the agent. The v2 subprocess proxy removes the upstream `/tests` bind from `docker run`; after the agent exits, it creates `/tests`, copies the frozen verifier files, and then invokes `test.sh`. The isolation version is part of provider and fairness fingerprints. Any trial produced by the earlier visible-verifier path is explicitly invalid.

Task success is the primary externally executable outcome. Skill-quality and trajectory-quality scores are separate upstream post-processing analyses, not fields returned by `eval_runner.py`; they are reported as secondary metrics only when their frozen post-processor completes successfully.

## Promotion

Promotion uses same-item policy-off/policy-on validation outcomes under one frozen evaluator epoch. Failure frequency cannot stand in for utility. A program must:

- activate on enough validation rows;
- declare and obey pre-execution abstention/trigger behavior;
- have positive paired net gain;
- satisfy its harm and cost contracts;
- clear a one-sided paired-effect lower confidence bound.

Every pair must also contain two valid external evaluations. Authentication errors, endpoint failures, Docker failures, and verifier infrastructure failures block the entire candidate promotion; they are never silently scored as task failures.

Before any model call, manifest-scoped preflight enumerates `required_env` declarations from `task.toml`. A selected task with an unavailable variable blocks the run. Reports persist variable names and affected counts only; credential values are never serialized. The credential-independent subset is generated from metadata before split assignment, so no observed model outcome influences exclusion.

SkillLearnBench cannot reveal verifier success to the agent before completion, so no method may perform a post-verifier oracle rollback. On non-activated rows, the baseline is used directly. On activated rows, conservatism is enforced by the frozen harm-rate and lower-bound promotion gates. The `preserve_baseline` program field is therefore a policy contract, not a claim that a failed candidate output was retroactively replaced.

The final report uses benchmark item as the analysis unit, majority vote across three sealed repeats, Wilson intervals, an item-clustered bootstrap for paired deltas, exact McNemar tests, and Holm multiplicity correction. A result is not claim-eligible unless all expected item/control/repeat keys are present with matching provider, budget, pair, protocol, manifest, and evaluator-epoch identities.

Train failures may provide the proposal model with the train instruction, structured metadata, sanitized evaluator metrics, and generic failure taxonomy. Verifier implementation, solution files, gold fields, validation content, and test content are forbidden proposal inputs. The instruction context may shape actions but cannot become a runtime trigger.

## HLE Transfer

After an archive is frozen on SkillLearnBench, run a never-inspected HLE family-out cohort. Compare raw, budget-matched raw, HippoRAG, fixed v2, and evolving v2. HLE results are transfer evidence, not evolution data.
