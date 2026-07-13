# Benchmark Protocol

## Why SkillLearnBench First

HLE is primarily a one-shot, broad-knowledge exam. It can test external transfer, but it gives the system little repeated feedback from which to learn reusable assumptions.

SkillLearnBench is the primary environment because it has executable task verifiers, skill-quality metrics, and trajectory keypoints. The upstream inventory has 20 task families and 100 instances. The frozen paper subset has 16 families and 86 instances with complete offline verifier support. It preserves the earlier split assignment while excluding the credential-bound `github-repo-analytics` family, three infrastructure-blocked families, and `weighted-gdp-calculation-2`, whose upstream verifier payload is incomplete. This scope is frozen before model calls and is independent of model outcomes. These tasks correspond directly to:

- hypothesis quality;
- hypothesis application fidelity;
- final task outcome.

## Protocol A: Instance Holdout

Purpose: test whether a hypothesis learned from earlier instances of one task family improves unseen instances of that family.

- Train: 38 instances.
- Validation: 16 instances.
- Sealed test: 32 instances.
- Test instructions are inaccessible until the archive is frozen.
- Hypothesis generation and recursive repair use train only.
- Policy promotion uses validation only; evaluator replacement is outside this primary experiment.

This is the main test of executable assumption learning.

## Protocol B: Family Out

Purpose: test whether learned policies and the proposal procedure generalize beyond task families used for evolution.

- Train: 9 families, 48 instances.
- Validation: 2 families, 11 instances.
- Sealed test: 5 families, 27 instances.

Task-specific hypotheses are not expected to transfer here. Policy hypotheses and the proposal procedure are. Evaluator hypotheses require the separate fixed-anchor epoch-challenger protocol.

## Controls

Every reported comparison must include:

1. `raw_no_skill`: no-skill agent, the primary baseline;
2. `static_generic_v2`: one frozen generic requirement/audit skill, the no-learning control;
3. `v2_no_recursive_repair`: the same evolution budget with recursive proposal repair disabled;
4. `promoted_v2`: the frozen incumbent from the full recursive loop, the primary candidate;
5. `skilllearn_b1_sonnet`: the upstream one-shot static skill reference;
6. `human_authored`: the upstream human-authored upper reference.

Model, tool access, trial count, retries, and inference budget must be matched. Report task success, skill quality, trajectory quality, model calls, wall time, and failure rate.

The active protocol fixes every primary arm to `gpt-5.4-mini` through the same ruoli OpenAI-compatible route. Proposal, raw, static controls, recursive ablations, and promoted-agent trials may not mix Spark, subscription auth, or a second provider into that run. Proposal/repair JSON is sent directly through the frozen OpenAI-compatible endpoint; task-trial Codex commands use the frozen `codex_custom_responses_provider_v1` configuration, including `/v1` normalization, Responses wire API, disabled WebSockets, disabled OpenAI-login requirements, ignored user config, and ephemeral execution. Model, provider mode, endpoint identity, custom-provider version, and the single-route policy enter the protocol and fairness fingerprints. Historical Spark results are diagnostics and are not pooled with v3.

The full and no-recursive evolution arms each have a predeclared maximum of three generations and stop after two consecutive scientific non-promotions, no failed training rows, duplicate behavior, proposal failure, or exhausted invalid counterfactual evidence. Invalid evidence is terminal non-claim and does not consume the consecutive-non-promotion counter. Later proposals may receive only aggregate prior hypothesis status and promotion summaries; validation instructions and per-item outcomes are not returned to the proposal model.

The two arms branch from one immutable first-generation checkpoint: identical raw train observations, identical labels for every valid train row, and identical proposed root programs. Failed rows retain sanitized proposal context; successful controls contain only runtime features and the success label, never instruction, evaluator feedback, or execution context. A run-scoped train cache also reuses those exact observations in later generations whenever the incumbent executable behavior, task set, evaluator epoch, model, and runtime are unchanged. Thus a non-promotion cannot silently resample the same 38 raw tasks. Root proposals are likewise replayed when the complete structured proposal request is identical, including labeled train evidence, evaluator epoch, runtime capabilities, prior hypotheses, and prior promotion feedback. Validation evidence is replayed when candidate and incumbent executable-behavior hashes, task features, evaluator epoch, and runtime version are identical. Each replay emits source/target hashes and makes zero new executions. A promoted incumbent, changed search history, or recursive repair changes the corresponding key and receives new evidence.

All proposed roots receive train-only schema, runtime-feature-vocabulary, failed-row trigger-support, executable-action, and evaluator-epoch checks. Instruction text may inform the action graph but is forbidden as a trigger key because it is not a runtime feature. V3.6-v3.9 select one candidate by exact failure activations divided by all failure-plus-success-control activations before support and complexity. V3.10 introduced exact-three roots and family-coverage-first selection. V3.11 keeps exact-three cardinality and atomic parsing but audits activation-signature diversity instead of rejecting equal-signature action variants. It selects one before validation by capped deficit to `ceil(minimum_activation_rate × distinct train families)`, then exact activation precision, success false positives, failed-row support, predicate/action complexity, and payload hash. Prompt-directive values must be complete imperative TRAIN-grounded sentences, and compiler lowering retains target plus readable value. The target is derived from the existing promotion contract and is not a new gate; validation features/outcomes never choose a same-generation candidate. Other valid roots remain shadow hypotheses. This family-coverage proxy is scoped to primary instance-holdout development and is not an unseen-family transfer claim.

The protocol freezes `candidate_local_repair_and_generation_terminal_root_failure_v1`. Its typed boundary covers transport/JSON failures and successful model calls with malformed request-specific envelopes or unparseable canonical programs. All consumed root rows stage successfully before any proposal event or replay record. A root-proposal failure ends both shared-checkpoint arms with preserved, explicitly non-claim reports. A recursive-repair failure is local to that branch during static audit, but the affected generation cannot execute held-out counterfactuals or promote an archive node. Request/response hashes, response-shape hashes/types/counts, error phase, and failure count are logged without raw responses, errors, or secrets. Execution reports derive claim eligibility from generation rows; paper freeze independently recomputes and rejects any failure or inconsistent top-level claim field.

This primary runtime experiment admits only task and policy hypotheses. An evaluator hypothesis cannot be compiled into an agent skill, because that would relabel agent guidance as evaluator evolution. Evaluator changes require the separate fixed-anchor epoch-challenger protocol and are outside the primary performance claim until that path is run.

Reports separate aggregate static-tree recursion across every proposed root from recursion depth in the ultimately selected tree. This prevents a repaired but non-selected candidate from disappearing from the recursive/no-recursive mechanism audit.

Repair branch identity uses `parent_content_scoped_repair_id_v1`: the parent ID, a status-independent canonical parent-content hash, repair depth, and canonical candidate program without its model-declared ID determine a canonical SHA-256 ID. Model-declared IDs and statuses never control the archive key or lifecycle state; every repair enters as harness-owned `candidate`. Replaying the same branch is deterministic; distinct parent/depth/content branches cannot alias except through a hash collision, while the archive's same-ID/different-content check remains fail closed.

Policy-off and policy-on trials use one fairness fingerprint derived from backend, provider, agent, model, action budget, verifier isolation, runner-local agent-registry isolation, trial-timeout policy, provider-route policy, image-cache policy, and the actual shared agent-runtime key. Their order is deterministically balanced by pair ID. An evaluator-invalid row or provider, budget, or runtime mismatch invalidates the pair bundle. Such bundles are neither cached nor replayed. The protocol additionally freezes `behavior_identical_validation_replay_v1`; replay is validation-only and cannot consume or populate sealed-test evidence. Parallel backends receive deep-copied upstream agent registries so one provider context cannot clear or restore another backend's agent definition.

Execution caches only the exact non-oracle task environment. Oracle `skills/` content is excluded from the environment hash and build context. Before development, all images and offline-verifier runtimes referenced by train, validation, and sealed-test IDs must pass a bounded cache-only prewarm without invoking an agent or sealed scoring. This infrastructure phase does inspect and hash test files; the v4 receipt therefore records `test_infrastructure_inspected=true`, `sealed_test_scoring_performed=false`, and `sealed_test_bytes_exposed_to_model=false`, then binds the manifest, item set, exact shared-runtime key/version, supervisor policy/hash, image IDs, and verifier runtimes. Missing dependencies may be downloaded only in an explicit preparation phase that cannot produce paper evidence. This freezes infrastructure availability without exposing sealed semantics to the model. Node is pinned by image digest, Codex CLI is pinned to `0.144.1`, and one read-only runtime volume is shared by all item images. V3.11 permits six item pipelines but only one online agent stage concurrently; local setup and offline verification bypass the model slot. Every variant for one item runs sequentially in a protocol-derived balanced order, so raw and candidate do not contend against each other inside one pair.

Before development, freeze, validation controls, or sealed controls can perform model work, one unified validator rechecks the lock content hash, protocol/evolution/promotion mappings, offline-readiness hash, current model/provider/origin, egress and network budget, provider readiness, selected benchmark payload fingerprint, code fingerprint, clean scoped Git state, and locked commit. Agent ID, action budget, parallelism, retry policy, trigger support, generation budget, and proposal count are derived from the protocol; the CLI is not a second owner.

Benchmark task payloads and verifier trees are local and require no Hugging Face or online leaderboard access. Model inference is online through the protocol-frozen endpoint. Trial containers use a provider-only endpoint allowlist, pinned host mapping with no external DNS, fail-closed prebuilt dependency caches, and the active v3.11 64 MiB Docker network fuse. The protocol owns the exact Codex execution-policy, proposal response budget, and model-slot mappings and binds them into plans, locks, fairness fingerprints, and reports. Top-level `web_search="disabled"` is verified on the actual Responses wire. A content-hashed supervisor owns and truncates the current attempt's trace, emits a random nonce, counts every `item.started`, terminates at the limit, and scans `/proc/<tgid>/task/<tid>` to remove all live tasks born after the dedicated container baseline before verifier materialization. This includes `setsid` descendants and worker threads whose leader is already a zombie; an incomplete scan fails closed. Receipt and trace are cross-audited. Natural completion requires valid token usage, while truncated rows report missing token usage and all v3.11 arms use action starts for promotion cost. V3.1-v3.10 contracts remain immutable prior evidence. Verifier and runtime verification run under `--network none`; any networked dependency preparation is explicit and outside evaluation. V3.7 failed with 6/6 first-wave 429 at six model calls; v3.8 reached 16 valid rows before two simultaneous 503s; v3.9 completed cleanly with one shared model slot but no incumbent; v3.10 increased activation without gain and exposed action-lowering and semantic-diversity failures. V3.11 keeps the scheduler and changes only pre-validation proposal/action treatment.

Compiled hypotheses use a content-hashed per-item routing manifest. A skill is injected only when that exact item's structured features satisfy its trigger; sharing a task family with a matched item is insufficient. Missing routes are explicit abstentions. Upstream human and B1 controls retain their native family-level layout because they do not declare V2 triggers.

The framework can represent a provider chain, but the active v3.11 protocol freezes a single ruoli route, so no active failover occurs. Trial-side policy-off/policy-on pairs require identical providers; one variant cannot receive a healthier endpoint or a different route.

Historical subscription-trial code paths are outside the active protocol. Active v3 credentials are read from environment for request/container authorization and are never serialized into reports or trial artifacts; missing authorization invalidates evidence before task scoring.

Verifier code is not visible to the agent. The v2 subprocess proxy removes the upstream `/tests` bind from `docker run`; after the agent exits, it creates `/tests`, copies the frozen verifier files, and then invokes `test.sh`. The isolation version is part of provider and fairness fingerprints. Any trial produced by the earlier visible-verifier path is explicitly invalid.

The upstream runner's fixed 1800-second wall timeout is removed from the agent and verifier subprocesses. Its separate `sleep 3600` trial-container lifetime is replaced by a signal-terminable keepalive loop, so a long agent cannot lose its container immediately before post-agent verifier materialization. Long but active tasks therefore run to completion; image construction, dependency installation, and prewarm still use bounded retry and timeout policies. Legacy rows with `agent_timed_out=true`, `verifier_exit=-1`, or container-expiry infrastructure failures are invalid and cannot enter promotion or performance statistics.

Task success is the primary externally executable outcome. Skill-quality and trajectory-quality scores are separate upstream post-processing analyses, not fields returned by `eval_runner.py`; they are reported as secondary metrics only when their frozen post-processor completes successfully.

## Promotion

Promotion uses same-item policy-off/policy-on validation outcomes under one frozen evaluator epoch. Failure frequency cannot stand in for utility. A program must:

- activate on enough validation rows;
- use the frozen prospective trigger, leaving non-activated rows on the baseline;
- have positive paired net gain;
- satisfy evaluator-owned harm and cost limits;
- clear a one-sided paired-effect lower confidence bound.

Pairs, confidence, net gain, activation, minimum effect lower bound, maximum
harm, and maximum cost all come from one protocol-owned `PromotionGateSpec`.
Candidate-declared limits are combined only in the stricter direction; a
candidate cannot lower its own promotion bar. The protocol, candidate, and
effective thresholds are all persisted in the promotion decision.

Every pair must also contain two valid external evaluations. Authentication errors, endpoint failures, Docker failures, and verifier infrastructure failures block the entire candidate promotion; they are never silently scored as task failures.

V3.6-v3.11 additionally report `valid_activation_count`, activated gain/harm counts, activation precision, activated harm rate, abstention count/rate, and explicit `defined` flags. The valid-activation denominator excludes the union of evaluator-invalid, provider-mismatched, and budget-mismatched pairs. With zero valid activations, precision and activated harm rate are `null` and their flags are false. These are application-fidelity diagnostics only; the evaluator-owned promotion blockers and thresholds above are unchanged. For the current SkillLearn projection, `selection_change_count` compares the pass/fail label rather than raw trajectory identity.

Training evidence has the stronger all-valid contract: one invalid train observation blocks residual mining, hypothesis proposal, and both recursive-ablation arms. Before blocking, only invalid observations receive up to the protocol-frozen number of same-request retries through a one-worker retry queue; a valid observation is never rerun. Validation applies the same clean-replacement rule independently to each arm. Codex JSONL `error` and `turn.failed` events are classified before verifier execution. Authentication, rate-limit, model-availability, and other fatal provider failures open one shared circuit so queued rows become explicit circuit-open invalid observations without further model calls.

Before any model call, manifest-scoped preflight enumerates `required_env` declarations and offline-verifier coverage. A selected task with an unavailable variable, missing authoritative verifier payload, inactive profile, or missing profile blocks the run. Reports persist variable names and affected hashes only; credential values are never serialized. The active offline-ready subset is a stable filter of the earlier manifests, so no item changes split and no observed model outcome influences exclusion.

SkillLearnBench cannot reveal verifier success to the agent before completion, so no method may perform a post-verifier oracle rollback. On non-activated rows, the baseline is used directly. On activated rows, conservatism is enforced by the frozen harm-rate and lower-bound promotion gates. The `preserve_baseline` program field is therefore a policy contract, not a claim that a failed candidate output was retroactively replaced.

The external SkillLearn compiler accepts only `execute_step`, `produce_artifact`, and `request_evidence` as prompt directives plus `check_condition` as an agent-local self-check. Lane mutation, parameter mutation, `require_verifier`, and `abstain` have no SkillLearn backend lowering and fail closed. External-verifier checks, policy-off/on evidence, and expected-effect thresholds are not rendered into `SKILL.md`. Compile receipts bind the lowering version and record that the external verifier was not exposed to the agent.

The final report uses benchmark item as the analysis unit, majority vote across three sealed repeats, Wilson intervals, an item-clustered bootstrap for paired deltas, exact McNemar tests, and Holm multiplicity correction. A result is not claim-eligible unless all expected item/control/repeat keys are present with matching provider, budget, pair, protocol, manifest, and evaluator-epoch identities.

Train failures may provide the proposal model with the train instruction, structured metadata, sanitized evaluator metrics, and generic failure taxonomy. Successful negative controls provide only runtime features and their label; they cannot leak instruction, feedback, or execution context. Verifier implementation, solution files, gold fields, validation content, and test content are forbidden proposal inputs. Failure instruction context may shape actions but cannot become a runtime trigger.

## HLE Transfer

After an archive is frozen on SkillLearnBench, run a never-inspected HLE family-out cohort. Compare raw, budget-matched raw, HippoRAG, fixed v2, and evolving v2. HLE results are transfer evidence, not evolution data.
