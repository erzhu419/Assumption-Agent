# Assumption Agent Reconstruction V2

Reconstruction v2 starts from a small, explicit optimization surface. It does not import the legacy HLE inference monolith.

## Research Claim

The system should improve by proposing falsifiable hypotheses, compiling them into backend-auditable treatments, recursively validating failed proposals, and promoting only policies with paired held-out evidence under a frozen evaluator epoch. The internal lane runtime retains typed executable actions; the external SkillLearn path uses only the explicitly documented prompt-directive and self-check lowering.

The architecture distinguishes:

1. `TaskHypothesis`: a relation or procedure useful inside one task family.
2. `PolicyHypothesis`: when to enable an operator, source, solver, or abstention route.
3. `EvaluatorHypothesis`: how to judge intermediate behavior while remaining anchored to external outcomes.

Every hypothesis is a `HypothesisProgram` with structured triggers, an action DAG, an expected effect, a verifier contract, a baseline-preserving fallback, lineage, and evaluator-epoch provenance.

The primary SkillLearn experiment currently promotes only task and policy hypotheses into the agent runtime. Evaluator hypotheses require the separate anchored epoch-challenger path and cannot be compiled as agent skills. This avoids claiming evaluator co-evolution before that ablation is implemented.

## Closed Loop

```text
training residuals
  -> structured proposal model
  -> recursive validation / revision tree
  -> same-item policy-off versus policy-on shadow run
  -> frozen validation evaluator
  -> paired-effect lower confidence bound
  -> archive promotion or rejection
  -> promoted program changes the next runtime plan
```

Test outcomes never create or select hypotheses. Evaluator replacement is allowed only at epoch boundaries against a fixed anchor; records scored by a displaced evaluator are invalidated.

## Benchmark Strategy

Primary: SkillLearnBench instance holdout. Its task-success, skill-quality, and trajectory-quality measurements directly test proposal quality, application fidelity, and final outcome.

Secondary: SkillLearnBench family-out and LifelongAgentBench, testing whether learned policies and the proposal procedure transfer beyond a learned task family. Evaluator hypotheses belong to a separate anchored epoch-challenger experiment.

External transfer: sealed HLE operator-bearing family-out cohorts. HLE remains useful, but it is not the continual-learning environment.

Synthetic tests in this package prove only that the learning loop is causally wired. They are not performance evidence.

## Commands

```bash
cd reconstruction_v2
python3 -m pip install --user -e '.[skilllearn]'
TMPDIR=/tmp TMP=/tmp TEMP=/tmp python3 -m pytest
python3 -m assumption_agent.benchmarks.preflight \
  --env-file ../.env \
  --manifest manifests/skilllearnbench_instance_holdout_offline_ready_v1.json \
  --trial-provider-mode openai_compatible \
  --root reference/self_evo_continual_20260707/repos/SkillLearnBench

# Build a no-network, no-test-access execution plan.
# The pipeline normally exports these public route controls; a direct CLI call
# must do the same. The API key is still loaded from the local env file.
export ASSUMPTION_V2_MODEL=gpt-5.4-mini
export ASSUMPTION_V2_SKILLLEARN_PROVIDER_MODE=openai_compatible
export ASSUMPTION_V2_PROVIDER_CHAIN=openai_compatible
export ASSUMPTION_V2_API_BASE=https://ruoli.dev
export ASSUMPTION_V2_API_ALLOWED_IPV4S=45.78.76.197
export ASSUMPTION_V2_SKILLLEARN_CACHE_ONLY=1
export ASSUMPTION_V2_TRIAL_NETWORK_BYTE_LIMIT=67108864
python3 -m assumption_agent.benchmarks.skilllearn_experiment \
  --root reference/self_evo_continual_20260707/repos/SkillLearnBench \
  --protocol manifests/skilllearn_paper_protocol_v3_14_ruoli_gpt54mini.json \
  --manifest manifests/skilllearnbench_instance_holdout_offline_ready_v1.json \
  --env-file ../.env \
  --out artifacts/skilllearn_plan.json \
  --events artifacts/skilllearn.events.jsonl \
  --work-dir artifacts/skilllearn_run \
  --train-limit 4 \
  --validation-limit 2

# Publication pipeline. This runs preflight, lock, all-manifest image/runtime
# prewarm, smoke, and development. It stops successfully when development has no
# promoted recursive candidate; only a real incumbent proceeds to freeze/validation.
./scripts/run_paper_pipeline.sh all-development

# Run family-out only after primary development has a real frozen incumbent;
# it uses an independent archive/run root and never substitutes for that prerequisite.
MANIFEST=manifests/skilllearnbench_family_out_offline_ready_v1.json \
RUN_ROOT=artifacts/paper_family_out_v3_9_offline86_ruoli_gpt54mini_outer6_model1 \
./scripts/run_paper_pipeline.sh all-development

# Run only after reviewing the frozen validation report.
./scripts/run_paper_pipeline.sh sealed-test
```

The direct command above is a dry run. Claim-bearing execution should use the paper pipeline; a manual `--execute` invocation must additionally supply a claim-eligible `--protocol-lock` and a passing `--prewarm-receipt`. Execution first collects every valid frozen train row. Failures provide proposal evidence; successes become label-only negative controls with no instruction, evaluator feedback, or execution context. The system then proposes and recursively checks hypotheses, compiles one selected candidate to `SKILL.md`, and runs same-item validation policy-off/policy-on trials. It never opens the sealed test split.

The local WSL environment now has Docker Engine 29.1.3 running under systemd. `scripts/bootstrap_docker_wsl.sh` remains available for a fresh machine. If the invoking process predates docker-group membership, the paper pipeline automatically uses `sg docker` for that command.

The active v3.14 paper runner fixes the agent, `gpt-5.4-mini`, the `ruoli.dev` OpenAI-compatible route, Codex custom-provider configuration, action budget, provider fingerprint, verifier-isolation version, and split manifest for every raw and agent arm. Each task command explicitly selects a temporary Responses-wire provider, sets authoritative top-level `web_search="disabled"`, disables image generation and standalone web search, ignores user configuration, and runs ephemerally. A container-local supervisor owns the JSONL trace, counts every `item.started` as `codex_action_start_v1`, and stops the dedicated trial-container process scope at the frozen limit before the verifier is materialized. It audits `/proc/<tgid>/task/<tid>`, so `setsid` descendants and live workers behind zombie leaders cannot hide from the post-agent cleanup; an incomplete scan invalidates the receipt. The API key is passed only through the named container environment variable; it is never embedded in the command or materialized as `auth.json`. Historical Spark/subscription and v3.1-v3.13 artifacts are not mixed into v3.14 evidence. The upstream `/tests` mount is removed before container start and verifier files are copied in only after all agent-stage tasks exit. Infrastructure errors, provider changes, budget mismatches, or isolation mismatches invalidate evidence instead of counting as wrong answers.

V3.5 changes exactly one execution resource from v3.4: every online phase uses one benchmark worker instead of four. The v3.4 action-budget canary passed, but its full development run produced 17 valid local evaluations followed by four concurrent 429s and 17 circuit skips. Those 17 rows are not reused. V3.5 retains the same model, subset, action and network budgets, offline evaluator, retry/circuit semantics, search, promotion, controls, and sealed policy; the serial schedule is protocol-hashed rather than supplied as a CLI override.

The final v3.5 fresh-root run completed 38/38 valid train rows and the full proposal/repair/paired-validation/report/archive mechanism, but promoted no incumbent. Its clean first generation showed the key selection failure: the recursive maximum-support repair produced 0 gains and 2 harms, while the narrower no-recursive root produced 1 gain and 0 harms but missed the frozen effect lower bound. A later no-recursive generation was contaminated by one Ruoli 503, circuit skips, 9 invalid pairs, and 8 budget mismatches. That arm is now rejected by freeze even under its historical schema. No empty-incumbent controls, family-out run, or sealed test were started.

V3.6 changes the learning evidence and lifecycle classification once, without changing the model, provider, single-worker schedule, split, action/network budgets, offline evaluator, promotion thresholds, controls, or sealed policy. Candidate selection uses exact train-side trigger precision, then fewer success false positives, greater failure support, lower complexity, and payload hash. Exhausted invalid counterfactual evidence terminates the generation as non-claim without consuming the scientific consecutive-non-promotion counter. Reports add evidence-valid activation, activated gain/harm, precision, harm-rate, and abstention diagnostics; these fields diagnose behavior and do not add a promotion gate.

The serial v3.6 live run passed its lock and 86/86 prewarm, completed 38/38 valid train rows (7 successes, 31 residuals), and selected one root with 26/27 train activation precision and one success-control false positive. It completed only 2/16 validation pairs before being intentionally stopped for throughput; it produced no report, archive, promotion, freeze, family-out, or sealed evidence. V3.7 then changed only all five online phase worker counts from one to six. Its first six simultaneous train requests all received provider 429, the circuit locally skipped the remaining 30, and no reusable training bundle or report was produced. V3.8 reduced workers to two; it completed 16 valid train rows before two simultaneous provider 503s opened the circuit and skipped 20. V3.9 restores six outer item workers but binds one shared online-agent slot: Docker preparation and offline verifier work may overlap, while `docker exec ... codex exec` is globally sequential inside the process. Variants of one item remain sequential. The evaluator, model, route, retry/circuit behavior, learning semantics, promotion contract, splits, and budgets are unchanged, and no prior row is reused.

V3.9 implementation and 289/289 offline tests pass. After two earlier roots and ten probes were blocked by HTTP 503, the lower-cost Ruoli credential recovered on 2026-07-13. A new root passed its claim lock and 86/86 cache-only prewarm and completed 56/56 actual external trials: maximum online-agent concurrency was one, every slot was released, and provider, circuit, infrastructure, budget, and pair-mismatch failures were all zero. The shared train bundle was 38/38 valid with 10 successes and 28 residuals. Recursive generations one and two each activated only 1/16 validation items and tied raw at 3/16 with 0 gain / 0 harm; both were rejected by the unchanged promotion contract. Both no-recursive roots failed train-only static audit before held-out execution. The four report/archive artifacts are complete and both archives have `incumbent_id=null`; sealed/test access is false. This clean negative result localizes the next problem to candidate search and prospective coverage, not transport or evaluation. No freeze, controls, family-out, or sealed run is permitted without a real incumbent.

V3.10 completed a fresh 38/38-valid train and 16/16-valid paired development run with no provider, infrastructure, budget, or mismatch failure. Exact-three search raised prospective activation from v3.9's 1/16 to 2/16, but candidate and raw still tied at 3/16 with 0 gain / 0 harm. Generation two then received two valid exact-three JSON batches whose three roots each collapsed to one train activation signature; the old response contract rejected both batches and made both reports non-claim. The run has no incumbent and cannot proceed to freeze, controls, family-out, HippoRAG transfer, or sealed test.

V3.11 was the bounded response to that result, still entirely before the unchanged promotion gate. Exact cardinality and atomic parsing remained strict, but activation-signature diversity became an audited search preference: equal-signature candidates with distinct action treatments were retained, with no retry. The compiler preserved `execute_step` and `check_condition` targets beside their values and rendered mappings as readable directives. Its fresh root produced three distinct activation signatures and an actually executed court-form treatment: the activated candidate used 16 action starts versus raw's 66 and changed the output PDF, but both failed, so candidate/raw remained 4/16 with 0 gain / 0 harm. The no-recursive report was non-claim because an unrelated policy-off poster trial exceeded the unchanged 64 MiB hard cap. The recursive arm stopped earlier when a repair response used batch field `hypotheses` instead of singular `hypothesis`. A post-run call-path audit found that the repair payload already omitted `proposal_batch_contract`; the remaining defect was that the generic system contract had no protocol-bound, repair-specific singular response instruction, so the model could still reuse the root response shape despite the nested output schema. Neither arm produced an incumbent, and no freeze, controls, family-out, HippoRAG transfer, or sealed run followed.

V3.12 changes only that request scope. Root proposals remain exact three, while recursive repair carries the train-only coverage objective plus a top-level `single_candidate_excludes_root_batch_contract_v1` contract that explicitly requires exactly one object under `hypothesis`; the model system contract applies this rule only when that versioned policy is present. The proposer defensively removes a root batch contract if a caller supplies one, but this is hardening rather than the diagnosed v3.11 cause. It does not change the evaluator, promotion gate, 64 MiB fuse, model, provider, split, scheduler, retries, controls, or sealed rules. V3.11 rows are immutable diagnostics and were not reused.

The fresh v3.12 root passed a clean lock and 86/86 cache-only prewarm, then completed 56/56 valid external trials with zero provider, infrastructure, action-budget, network-cap, or pair-mismatch failures. Train was 38/38 valid (8 successes, 30 residuals). Both generations returned exact three roots with distinct activation signatures and 3/3 static acceptance, but every root covered only one train family. Coverage-first selected a three-failure court policy each time; it activated only `court-form-filling-5` (1/16), changed raw's 9 actions to 32 and then 43 candidate actions, and remained 0→0. Both arms consequently stayed 4/16 with zero gain/harm and ended after two non-promotions with byte-identical `incumbent_id=null` archives. All 12 static nodes passed, so the full run made no repair request: the singular contract is bound and did not break normal development, but its live repair path remains untested.

The same invocation exposed and quarantined an orchestration defect after development: the old runner wrote an empty-candidate freeze receipt and began validation controls despite `selected_candidate_available=false`. It was stopped immediately. Only 2/96 control records exist, both interruption-invalid, with no report or sealed/test access; their machine-readable marker forbids row reuse and performance claims. The runner now exits after development when there is no promoted recursive candidate, `paper_freeze` refuses an empty recursive incumbent, and `paper_controls` independently rejects legacy empty-candidate receipts before backend/model work. These checks enforce one phase prerequisite and do not alter any scoring threshold.

V3.13 addresses the information loss revealed by v3.12: the proposal already supplied three mutually complementary, perfect-precision, zero-false-positive single-family roots, but the old selector discarded two. V3.13 enumerates the exact-three nonempty subsets using TRAIN evidence only and fixes one program set before validation. Union precision ranks before capped family coverage, so coverage cannot force in a lower-precision member; overlap, bundle size, support, complexity, and canonical set hash resolve later ties. The runner binds delta/full/matched program-set identities, executes at most one policy-on trial per item when any new member matches, and strictly aliases baseline otherwise. Replay, report, archive, and freeze use the canonical set treatment hash and selected IDs. Bundle rejection remains a node-level result and retains members as shadow rather than falsely rejecting them individually. Promotion remains `evaluator_owned_paired_validation_v2`, with the same numeric contract and conservative aggregation of member-declared limits. The 375/375 offline suite passes; no v3.12 row was reused in the completed live evidence described below.

The completed v3.13 development is a clean negative result. The original partial development invocation is isolated as diagnostic-only; the unchanged lock, 86/86 prewarm, and smoke evidence were retained because the Plus/Pro credential tier is not part of the frozen model/provider identity, and a new formal development event/work tree was used. That invocation completed 76/76 valid attempt-one trials: 38 train policy-off rows, 32 validation policy-off rows, and six activated policy-on rows, with zero provider, infrastructure, action-budget, network-cap, or pair-mismatch failure. Train was 8 successes and 30 residuals. Each selected two-member set joined a poster policy with a court-form policy, had 5/5 failure activation across two train families and 0/8 success-control false positives, and activated exactly `anthropic-poster-design-2` plus `court-form-filling-5` on validation. All six policy-on executions failed, as did their paired policy-off executions, so all four generation decisions had zero gain and zero harm; both arms stopped after two non-promotions with `incumbent_id=null`. Recursive live repair was exercised without model/shape failure. The largest trial used 62,200,000 of 67,108,864 network bytes and the largest action receipt used 69/100 starts. No freeze, controls, family-out, HippoRAG transfer, or sealed/test access followed.

The bundle mechanism therefore worked, but TRAIN trigger precision predicted known failure membership rather than treatment utility. In no-recursive generation two, the precise three-member subset covered 7/7 failures across three families but ranked fourth because family deficit is capped at two and `bundle_size_asc` precedes `failure_support_desc`. A separate replay defect also became visible: recursive generation two reused the shared G1 raw cohort, whereas no-recursive generation two resampled 16 raw rows; each arm's paired evidence is valid, but the 2/16 versus 4/16 raw difference is not a recursion-only effect.

V3.14 is the bounded response and changes exactly those two mechanisms. `behavior_identical_shared_validation_baseline_arm_replay_v2` records each valid validation policy-off observation once in an immutable cohort shared by recursive and no-recursive runners across generations; challenger-specific pair metadata is excluded from its evidence identity, invalid rows are not recorded, and conflicting evidence cannot overwrite the cohort. `train_contrastive_complementary_family_support_bundle_precision_first_v2` preserves precision-first selection and the success-false-positive safeguard, then—among otherwise tied precise candidates—places actual family count and failure support before bundle size. It still fixes exactly one TRAIN-selected set before one held-out paired validation. The current 411/411 offline suite passes. No evaluator threshold, promotion blocker, model, route, split, budget, control, or sealed policy changed. This is not a claim that the broader set will improve performance: v3.14 live development is pending, and another zero-gain result ends selector iteration in favor of a separate executable-action-quality workstream.

The Plus and Pro Ruoli credentials both serve the frozen `gpt-5.4-mini` route. Switching credential tier inside an otherwise unchanged protocol does not by itself require restarting from lock, prewarm, or smoke; only the failed development event/work tree must remain isolated. V3.14 is nevertheless a new protocol/code identity, so its first live run requires a fresh lock, cache-only prewarm, smoke, and run root. Existing model/runtime images and dependencies remain reusable and need not be downloaded again unless the cache-only prewarm reports a genuinely missing artifact.

Repair hypotheses do not trust model-generated IDs or lifecycle statuses. `parent_content_scoped_repair_id_v1` derives a deterministic branch ID from the parent ID, a status-independent canonical parent-content hash, repair depth, and canonical candidate child content without its declared ID. The harness forces a repair to enter as `candidate`; only the existing lifecycle may later promote or reject it. This prevents different repairs that reuse a model ID from colliding while preserving the archive's strict rejection of one canonical ID mapped to different content. The event ledger records the identity policy/hash and a hash—not the value—of the discarded model ID.

Codex JSONL terminal events are parsed directly from the complete trace, so a late `turn.failed` usage-limit, authentication, rate-limit, or model-availability error cannot be hidden by a clipped upstream result or an earlier generic stream error. Subscription and third-party provider failures retain distinct labels. The first global provider failure opens a run-scoped circuit and suppresses queued model calls. Any invalid train observation blocks residual mining and proposal generation; the experiment must be resumed under the same frozen protocol only after provider health returns.

Transient invalid observations first enter a protocol-bounded, one-worker retry queue. Only the same frozen request key may clean-replace an invalid attempt; valid rows are never retried. A run-scoped training cache additionally reuses exact observations whenever the incumbent executable behavior is unchanged, so a rejected generation does not spend another 38 online task calls merely to resample raw behavior.

Proposal and recursive-repair calls have a separate conservative failure boundary. It covers transport/JSON failures and successful calls whose request-specific envelope or canonical program cannot be parsed. Root rows are parsed atomically before any proposal event or replay record; a malformed root preserves terminal non-claim reports, while a malformed repair abandons only that candidate branch for static diagnostics and blocks counterfactual validation and promotion for the generation. Events persist hashes and shape metadata, never the raw response. Reports derive failure count, claim eligibility, and blockers from generation rows, and paper freeze independently recomputes and rejects failed or tampered reports.

An exact-request root proposal cache removes another ablation confound. If recursive and no-recursive arms enter a later generation with identical residuals, capabilities, archive context, and promotion feedback, they reuse the same proposed roots and make no second model call. A changed state necessarily changes the request hash and receives a fresh proposal.

Docker execution uses content-addressed per-item base images with oracle skill directories excluded. Before any development model call, every image and offline-verifier runtime referenced by the frozen train, validation, or sealed-test manifest must pass a three-attempt cache-only prewarm. This is infrastructure verification only: it inspects and hashes test infrastructure but launches neither an agent nor sealed scoring and exposes no test bytes to the model. Its v4 receipt records those distinctions and binds the manifest, full item set, image IDs, verifier runtimes, shared runtime key, exact Codex version, and supervisor policy/hash. Missing runtime dependencies are prepared only by an explicit `scripts/prepare_codex_agent_runtime.py --allow-network-download` step; they are never downloaded by paper execution. A single read-only agent runtime is shared across those images and is locked to `node@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0` plus `@openai/codex@0.144.1` and the content-hashed supervisor. The runner still deep-copies the upstream agent registry before mutating provider fields, preventing setup/restore leakage. Fixed wall-clock timeouts are removed from the active agent and external-verifier stages, and the upstream one-hour trial-container lifetime is replaced by a signal-terminable keepalive; build, setup, and prewarm limits remain bounded. The timeout policy, registry-isolation version, runtime key, exact CLI version, supervisor receipt, base image ID, cache key, reuse state, shared-slot policy, and slot count are logged and bound into fairness provenance. V2 skills use a hashed per-item route, so a trigger match never spills across a whole family. The active v3.14 schedule runs at most six benchmark-item pipelines concurrently but admits only one online Codex agent stage at a time; all controls for one item and policy-off/policy-on within one pair remain sequential.

Task payloads, local databases, task images, and verifier trees come from the frozen local SkillLearnBench checkout; there is no Hugging Face dataset request or online leaderboard scoring. Model inference still uses the online ruoli endpoint. Trial containers use the provider-only restricted network and the unchanged 64 MiB network fuse; verifier probes and dependency preparation run outside scoring, with verifier/runtime verification under Docker `--network none`. V3.14 inherits v3.13's program-set treatment identity, v3.12's singular repair scope, v3.11's actionable lowering, v3.10's exact-three search, v3.9's two-level scheduler, v3.6's contrastive/invalid-evidence semantics, and v3.4's model-only executable-action-budget boundary. Promotion cost is uniformly measured in action starts for all v3.14 arms; incomplete token usage is reported as missing rather than as zero. A triggered local history checkpoint still uses the same online model route; it is not an online evaluator. V3.1-v3.13 runs remain immutable prior evidence. The active paper subset contains the 86 credential-free items with complete offline verifier support. The remaining nine credential-independent items stay in a separate coverage-extension workstream rather than silently fetching dependencies or switching evaluator.

The paper manifests are `manifests/skilllearnbench_instance_holdout_offline_ready_v1.json` and `manifests/skilllearnbench_family_out_offline_ready_v1.json`. They preserve the earlier split assignments while filtering the three infrastructure-blocked families and `weighted-gdp-calculation-2`; the credential-requiring `github-repo-analytics` family remains excluded. The resulting instance split is 38/16/32 and the family-out split is 48/11/27. This exclusion is frozen before model calls and is independent of model outcomes. The protocol binds `manifests/skilllearn_offline_readiness_receipt_v1.json`, which summarizes 7/7 profile contracts, 15/15 train-family probes, and all-selected-item static preflight. It is distinct from the runtime prewarm receipt: v3.5 requires a fresh v4 receipt for all 86 selected images/runtimes, bound to the current supervisor/runtime and produced without invoking a model or sealed scoring.

`manifests/skilllearn_paper_protocol_v3_14_ruoli_gpt54mini.json` freezes the active paper model and single-provider route, six outer item workers, one shared online-agent slot, offline-ready subset, agent runtime and supervisor, canonical model-only tool boundary, executable action budget, 64 MiB per-trial model-network budget, three-generation search budget, exact-three root proposal cardinality with audit-only activation diversity, singular repair-request scope, support-aware train-only complementary program-set selection, program-set validation replay v2, shared immutable baseline-arm replay v2, actionable prompt-directive lowering v2, recursive/no-recursive ablation, invalid-evidence lifecycle policy, proposal-failure isolation, exact-request root replay, same-key invalid retry budget, training replay, complete evaluator-owned promotion contract, six final controls, repeats, paired statistics, and instance/family-out sample counts. Candidate-declared effect limits are conservatively aggregated and may only tighten the promotion contract. SkillLearn programs may lower only to explicit prompt directives or agent-local self-checks; the post-agent external verifier is never rendered as a callable action, and `baseline_preserved` is observed only on non-activation aliases. Historical Spark and v3.1-v3.13 evidence remain non-active for v3.14. The two evolution arms share the exact first-generation labeled train observations, proposed roots, and validation policy-off cohort; recursive repair is the only intended search difference at that checkpoint. Run-scoped proposal, train, baseline-arm, and counterfactual caches include the complete relevant protocol/fairness identity, so state-identical evidence is exact replay rather than a new sample and invalid/mismatched rows never populate replay. Every proposed root is checked against runtime vocabulary and failed-row support before one TRAIN-selected set may consume validation outcomes. Only a nonempty promoted recursive incumbent may enter `paper_freeze`; controls additionally require a content-bound receipt with `selected_candidate_available=true`. A sealed journal permits only same-key infrastructure retries.

One execution contract owns the paper run. Before development, freeze, validation controls, or sealed controls can perform model work, the code revalidates the lock content hash, protocol/evolution/promotion mappings, readiness and v4 prewarm receipt hashes, model/provider/origin, egress and network budget, provider readiness, benchmark payload fingerprint, code fingerprint, clean scoped Git state, and locked commit. Agent ID, action budget, workers, retries, trigger support, generation budget, and candidate count are derived from the protocol rather than accepted from a second CLI source.

## Proposal Providers

Proposal and recursive-repair calls use one fixed provider chain for an entire run. The active v3.14 protocol is:

```text
openai_compatible (ruoli.dev) / gpt-5.4-mini
```

Raw, static controls, recursive ablations, and the evolving agent all use this same route. Trial commands use the protocol-bound `codex_custom_responses_provider_v1` configuration rather than relying on `OPENAI_BASE_URL` alone. A provider failure opens the run-scoped circuit; it does not switch only one arm to a healthier backend. Provider choice, route/config hash, response hash, elapsed time, and tool-use rejection are logged without endpoint credentials or response text. Because ruoli is a third-party OpenAI-compatible service, the protocol identifies it as the provider and does not describe the result as an OpenAI-direct run.

Proposal and repair requests use `OpenAICompatibleProposalModel`, which sends schema-constrained JSON directly to the frozen ruoli endpoint. The API key is read from the process environment only when constructing the Authorization header; it is not written to requests, events, reports, or workspace files. This proposal transport does not spawn a Codex child process and does not expose runtime tools.

The pipeline reads model, trial-provider mode, and proposal-provider chain from the protocol and exports them before loading `.env`, so local defaults cannot silently change an arm. The active protocol is a single ruoli/OpenAI-compatible route; there is no active Spark/subscription fallback.

## Local References

The Red Queen paper, the full self-evolution bundle, and the two local provider-bridge reference clones live under `reference/`. They are intentionally ignored by Git because the cloned repositories and PDFs are large local research inputs.

The architecture assessment that motivated this reconstruction is preserved in
[`docs/red_queen_architecture_diagnosis_20260711.md`](docs/red_queen_architecture_diagnosis_20260711.md).
