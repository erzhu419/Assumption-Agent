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
  --protocol manifests/skilllearn_paper_protocol_v3_4_ruoli_gpt54mini.json \
  --manifest manifests/skilllearnbench_instance_holdout_offline_ready_v1.json \
  --env-file ../.env \
  --out artifacts/skilllearn_plan.json \
  --events artifacts/skilllearn.events.jsonl \
  --work-dir artifacts/skilllearn_run \
  --train-limit 4 \
  --validation-limit 2

# Publication pipeline. This runs preflight, lock, all-manifest image/runtime
# prewarm, smoke, development, freeze, and validation. It stops before sealed test.
./scripts/run_paper_pipeline.sh all-development

# Family-out development and validation use an independent archive/run root.
MANIFEST=manifests/skilllearnbench_family_out_offline_ready_v1.json \
RUN_ROOT=artifacts/paper_family_out_v3_4_offline86_ruoli_gpt54mini \
./scripts/run_paper_pipeline.sh all-development

# Run only after reviewing the frozen validation report.
./scripts/run_paper_pipeline.sh sealed-test
```

The direct command above is a dry run. Claim-bearing execution should use the paper pipeline; a manual `--execute` invocation must additionally supply a claim-eligible `--protocol-lock` and a passing `--prewarm-receipt`. Execution first collects no-skill failures from the frozen train IDs, proposes and recursively checks hypotheses, compiles one selected candidate to `SKILL.md`, and runs same-item validation policy-off/policy-on trials. It never opens the sealed test split.

The local WSL environment now has Docker Engine 29.1.3 running under systemd. `scripts/bootstrap_docker_wsl.sh` remains available for a fresh machine. If the invoking process predates docker-group membership, the paper pipeline automatically uses `sg docker` for that command.

The active v3.4 paper runner fixes the agent, `gpt-5.4-mini`, the `ruoli.dev` OpenAI-compatible route, Codex custom-provider configuration, action budget, provider fingerprint, verifier-isolation version, and split manifest for every raw and agent arm. Each task command explicitly selects a temporary Responses-wire provider, sets authoritative top-level `web_search="disabled"`, disables image generation and standalone web search, ignores user configuration, and runs ephemerally. A container-local supervisor owns the JSONL trace, counts every `item.started` as `codex_action_start_v1`, and stops the dedicated trial-container process scope at the frozen limit before the verifier is materialized. It audits `/proc/<tgid>/task/<tid>`, so `setsid` descendants and live workers behind zombie leaders cannot hide from the post-agent cleanup; an incomplete scan invalidates the receipt. The API key is passed only through the named container environment variable; it is never embedded in the command or materialized as `auth.json`. Historical Spark/subscription and v3.1-v3.3 artifacts are diagnostics and are not mixed into v3.4 evidence. The upstream `/tests` mount is removed before container start and verifier files are copied in only after all agent-stage tasks exit. Infrastructure errors, provider changes, budget mismatches, or isolation mismatches invalidate evidence instead of counting as wrong answers.

Codex JSONL terminal events are parsed directly from the complete trace, so a late `turn.failed` usage-limit, authentication, rate-limit, or model-availability error cannot be hidden by a clipped upstream result or an earlier generic stream error. Subscription and third-party provider failures retain distinct labels. The first global provider failure opens a run-scoped circuit and suppresses queued model calls. Any invalid train observation blocks residual mining and proposal generation; the experiment must be resumed under the same frozen protocol only after provider health returns.

Transient invalid observations first enter a protocol-bounded, one-worker retry queue. Only the same frozen request key may clean-replace an invalid attempt; valid rows are never retried. A run-scoped training cache additionally reuses exact observations whenever the incumbent executable behavior is unchanged, so a rejected generation does not spend another 38 online task calls merely to resample raw behavior.

Proposal and recursive-repair calls have a separate conservative failure boundary. A root proposal outage produces a terminal generation report instead of losing the run; a repair outage abandons only that candidate branch for diagnostics, then blocks counterfactual validation and promotion for the whole generation. Reports expose the failure count, so degraded search can never be presented as clean performance evidence.

An exact-request root proposal cache removes another ablation confound. If recursive and no-recursive arms enter a later generation with identical residuals, capabilities, archive context, and promotion feedback, they reuse the same proposed roots and make no second model call. A changed state necessarily changes the request hash and receives a fresh proposal.

Docker execution uses content-addressed per-item base images with oracle skill directories excluded. Before any development model call, every image and offline-verifier runtime referenced by the frozen train, validation, or sealed-test manifest must pass a three-attempt cache-only prewarm. This is infrastructure verification only: it inspects and hashes test infrastructure but launches neither an agent nor sealed scoring and exposes no test bytes to the model. Its v4 receipt records those distinctions and binds the manifest, full item set, image IDs, verifier runtimes, shared runtime key, exact Codex version, and supervisor policy/hash. Missing runtime dependencies are prepared only by an explicit `scripts/prepare_codex_agent_runtime.py --allow-network-download` step; they are never downloaded by paper execution. A single read-only agent runtime is shared across those images and is locked to `node@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0` plus `@openai/codex@0.144.1` and the content-hashed supervisor. Each parallel backend deep-copies the upstream agent registry into its own loaded runner before mutating provider fields, preventing cross-thread setup/restore races. Fixed wall-clock timeouts are removed from the active agent and external-verifier stages, and the upstream one-hour trial-container lifetime is replaced by a signal-terminable keepalive; build, setup, and prewarm limits remain bounded. The timeout policy, registry-isolation version, runtime key, exact CLI version, supervisor receipt, base image ID, cache key, and reuse state are logged for every trial and enter the fairness audit. V2 skills use a hashed per-item route, so a trigger match never spills across a whole family. Different benchmark items may run concurrently; all controls for one item, and policy-off/policy-on within one counterfactual pair, remain sequential in a deterministically balanced order.

Task payloads, local databases, task images, and verifier trees come from the frozen local SkillLearnBench checkout; there is no Hugging Face dataset request or online leaderboard scoring. Model inference still uses the online ruoli endpoint. Trial containers use the provider-only restricted network and the unchanged 64 MiB network fuse; verifier probes and dependency preparation run outside scoring, with verifier/runtime verification under Docker `--network none`. V3.4 retains v3.3's low reasoning, low verbosity, 32,768-token `body_after_prefix` compaction threshold, 10,000-token tool-output limit, request compression, and disabled remote compaction, while adding canonical hosted-web disable and executable action-budget/process-containment receipts. Promotion cost is uniformly measured in action starts for all v3.4 arms; incomplete token usage is reported as missing rather than as zero. A triggered local history checkpoint still uses the same online model route; it is not an online evaluator. V3.1-v3.3 runs remain immutable execution diagnostics. The active paper subset contains the 86 credential-free items with complete offline verifier support. The remaining nine credential-independent items stay in a separate coverage-extension workstream rather than silently fetching dependencies or switching evaluator.

The paper manifests are `manifests/skilllearnbench_instance_holdout_offline_ready_v1.json` and `manifests/skilllearnbench_family_out_offline_ready_v1.json`. They preserve the earlier split assignments while filtering the three infrastructure-blocked families and `weighted-gdp-calculation-2`; the credential-requiring `github-repo-analytics` family remains excluded. The resulting instance split is 38/16/32 and the family-out split is 48/11/27. This exclusion is frozen before model calls and is independent of model outcomes. The protocol binds `manifests/skilllearn_offline_readiness_receipt_v1.json`, which summarizes 7/7 profile contracts, 15/15 train-family probes, and all-selected-item static preflight. It is distinct from the runtime prewarm receipt: active v3.4 now passes all 86 selected images/runtimes with 47 unique images and seven offline verifier runtimes, bound to its current supervisor/runtime and without invoking a model or sealed scoring.

`manifests/skilllearn_paper_protocol_v3_4_ruoli_gpt54mini.json` freezes the active paper model and single-provider route, offline-ready subset, agent runtime and supervisor, canonical model-only tool boundary, executable action budget, 64 MiB per-trial model-network budget, three-generation search budget, recursive/no-recursive ablation, train-only candidate selection, proposal-failure isolation, exact-request root replay, same-key invalid retry budget, train/validation replay policies, complete evaluator-owned promotion contract, prompt-action lowering/fallback semantics, six final controls, repeats, paired statistics, and instance/family-out sample counts. Candidate-declared effect limits may only tighten the promotion contract. SkillLearn programs may lower only to explicit prompt directives or agent-local self-checks; the post-agent external verifier is never rendered as a callable action, and `baseline_preserved` is observed only on non-activation aliases. Historical Spark runs and v3.1-v3.3 execution diagnostics remain non-active evidence. The two evolution arms share the exact first-generation train observations, residuals, and proposed roots; recursive repair is the only intended difference at that checkpoint. Run-scoped proposal, train, and validation caches include the Codex execution-policy hash, so state-identical work is exact replay rather than a new sample and evidence cannot cross policy revisions. Every proposed root is checked against runtime vocabulary and train support before one candidate may consume validation outcomes. `paper_freeze` compiles content-hashed validation/test controls, and a sealed journal permits only same-key infrastructure retries.

One execution contract owns the paper run. Before development, freeze, validation controls, or sealed controls can perform model work, the code revalidates the lock content hash, protocol/evolution/promotion mappings, readiness and v4 prewarm receipt hashes, model/provider/origin, egress and network budget, provider readiness, benchmark payload fingerprint, code fingerprint, clean scoped Git state, and locked commit. Agent ID, action budget, workers, retries, trigger support, generation budget, and candidate count are derived from the protocol rather than accepted from a second CLI source.

## Proposal Providers

Proposal and recursive-repair calls use one fixed provider chain for an entire run. The active v3.4 protocol is:

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
