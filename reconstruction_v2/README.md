# Assumption Agent Reconstruction V2

Reconstruction v2 starts from a small, explicit optimization surface. It does not import the legacy HLE inference monolith.

## Research Claim

The system should improve by proposing falsifiable hypotheses, compiling them into executable programs, recursively validating failed proposals, and promoting only policies with paired held-out evidence under a frozen evaluator epoch.

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

Secondary: SkillLearnBench family-out and LifelongAgentBench, testing whether policy and evaluator hypotheses transfer beyond a learned task family.

External transfer: sealed HLE operator-bearing family-out cohorts. HLE remains useful, but it is not the continual-learning environment.

Synthetic tests in this package prove only that the learning loop is causally wired. They are not performance evidence.

## Commands

```bash
cd reconstruction_v2
python3 -m pip install --user -e '.[skilllearn]'
TMPDIR=/tmp TMP=/tmp TEMP=/tmp python3 -m pytest
python3 -m assumption_agent.benchmarks.preflight \
  --env-file ../.env \
  --manifest manifests/skilllearnbench_instance_holdout_credential_independent_v1.json \
  --trial-provider-mode openai_compatible \
  --root reference/self_evo_continual_20260707/repos/SkillLearnBench

# Build a no-network, no-test-access execution plan.
python3 -m assumption_agent.benchmarks.skilllearn_experiment \
  --root reference/self_evo_continual_20260707/repos/SkillLearnBench \
  --manifest manifests/skilllearnbench_instance_holdout_credential_independent_v1.json \
  --env-file ../.env \
  --out artifacts/skilllearn_plan.json \
  --events artifacts/skilllearn.events.jsonl \
  --work-dir artifacts/skilllearn_run \
  --train-limit 4 \
  --validation-limit 2 \
  --minimum-pairs 2 \
  --parallel-workers 4 \
  --trial-provider-mode openai_compatible \
  --model gpt-5.4-mini

# Publication pipeline. This runs preflight, lock, train/validation image
# prewarm, smoke, development, freeze, and validation. It stops before sealed test.
./scripts/run_paper_pipeline.sh all-development

# Family-out development and validation use an independent archive/run root.
MANIFEST=manifests/skilllearnbench_family_out_credential_independent_v1.json \
RUN_ROOT=artifacts/paper_family_out_v3_ruoli_gpt54mini \
./scripts/run_paper_pipeline.sh all-development

# Run only after reviewing the frozen validation report.
./scripts/run_paper_pipeline.sh sealed-test
```

Add `--execute` only after preflight has no blockers and the model health probe succeeds. Execution first collects no-skill failures from the frozen train IDs, proposes and recursively checks one hypothesis, compiles it to candidate `SKILL.md` files, and runs same-item validation policy-off/policy-on trials. It never opens the sealed test split.

The local WSL environment now has Docker Engine 29.1.3 running under systemd. `scripts/bootstrap_docker_wsl.sh` remains available for a fresh machine. If the invoking process predates docker-group membership, the paper pipeline automatically uses `sg docker` for that command.

The active v3 paper runner fixes the agent, `gpt-5.4-mini`, the `ruoli.dev` OpenAI-compatible route, Codex custom-provider configuration, step budget, provider fingerprint, verifier-isolation version, and split manifest for every raw and agent arm. Each task command explicitly selects a temporary Responses-wire provider, disables WebSockets and OpenAI-login requirements, ignores user configuration, and runs ephemerally. The API key is passed only through the named container environment variable; it is never embedded in the command or materialized as `auth.json`. The historical v2 protocol remains fixed to Spark plus ChatGPT subscription auth and is not mixed into v3 evidence. The upstream `/tests` mount is removed before container start and verifier files are copied in only after the agent process exits. Infrastructure errors, provider changes, budget mismatches, or isolation mismatches invalidate evidence instead of counting as wrong answers.

Codex JSONL terminal events are parsed directly, so a `turn.failed` usage-limit, authentication, rate-limit, or model-availability error cannot be hidden by the upstream `codex | tee` pipeline's zero exit code. Subscription and third-party provider failures retain distinct labels. The first global provider failure opens a run-scoped circuit and suppresses queued model calls. Any invalid train observation blocks residual mining and proposal generation; the experiment must be resumed under the same frozen protocol only after provider health returns.

Docker execution uses content-addressed per-item base images with oracle skill directories excluded. Before any development model call, every train/validation image must pass a three-attempt prewarm gate whose receipt binds the manifest, item set, image IDs, and shared runtime. A single read-only agent runtime is shared across those images and is locked to `node@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0` plus `@openai/codex@0.144.1`. Each parallel backend deep-copies the upstream agent registry into its own loaded runner before mutating subscription auth fields, preventing cross-thread setup/restore races. Fixed wall-clock timeouts are removed from the active agent and external-verifier stages, and the upstream one-hour trial-container lifetime is replaced by a signal-terminable keepalive; build, setup, and prewarm limits remain bounded. The timeout policy, registry-isolation version, runtime key, exact CLI version, base image ID, cache key, and reuse state are logged for every trial and enter the fairness audit. V2 skills use a hashed per-item route, so a trigger match never spills across a whole family. Different benchmark items may run concurrently; all controls for one item, and policy-off/policy-on within one counterfactual pair, remain sequential in a deterministically balanced order.

The paper manifests are `manifests/skilllearnbench_instance_holdout_credential_independent_v1.json` and `manifests/skilllearnbench_family_out_credential_independent_v1.json`. They freeze a 95-item subset and exclude the complete five-item `github-repo-analytics` family because its upstream `task.toml` requires a personal `GH_TOKEN`. This exclusion is metadata-only, precedes all model calls, and keeps the benchmark independent of private credentials. Manifest-scoped preflight blocks before execution if any selected task still has an unavailable `required_env`.

`manifests/skilllearn_paper_protocol_v3_ruoli_gpt54mini.json` freezes the active paper model and single-provider route, credential-independent subset, agent runtime, three-generation search budget, recursive/no-recursive ablation, train-only candidate selection, six final controls, repeats, invalid-row policy, paired statistics, and instance/family-out sample counts. `manifests/skilllearn_paper_protocol_v2.json` is retained as the historical Spark protocol. The two evolution arms share the exact first-generation train observations, residuals, and proposed roots; recursive repair is the only intended difference at that checkpoint. A run-scoped validation cache keys external evidence by candidate behavior, incumbent behavior, task features, evaluator epoch, and runtime version. If both arms compile to identical behavior, the second arm replays the exact first-arm counterfactual pairs with zero new model calls; a real repair changes the key and receives an independent evaluation. Every proposed root is checked against the runtime feature vocabulary and train residual support. Only one statically accepted candidate, chosen by frozen train-only support and complexity ordering, may consume validation outcomes in a generation. `paper_freeze` compiles content-hashed validation/test control directories from the two selected archives. A sealed journal binds the test record file and permits only same-key infrastructure retries.

## Proposal Providers

Proposal and recursive-repair calls use one fixed provider chain for an entire run. The active v3 protocol is:

```text
openai_compatible (ruoli.dev) / gpt-5.4-mini
```

Raw, static controls, recursive ablations, and the evolving agent all use this same route. Trial commands use the protocol-bound `codex_custom_responses_provider_v1` configuration rather than relying on `OPENAI_BASE_URL` alone. A provider failure opens the run-scoped circuit; it does not switch only one arm to a healthier backend. Provider choice, route/config hash, response hash, elapsed time, and tool-use rejection are logged without endpoint credentials or response text. Because ruoli is a third-party OpenAI-compatible service, the protocol identifies it as the provider and does not describe the result as an OpenAI-direct run.

The Codex proposal turn is ephemeral and runs in an empty temporary working directory with read-only sandboxing, no dynamic tools, no environment capabilities, and no approval path. Any observed tool item or server-side runtime request invalidates the response. The child process receives only the minimal Codex/auth/runtime environment; API keys are deliberately excluded.

The pipeline reads model, trial-provider mode, and proposal-provider chain from the protocol and exports them before loading `.env`, so local defaults cannot silently change an arm. The legacy v2 Spark protocol can still be selected explicitly with `PROTOCOL=manifests/skilllearn_paper_protocol_v2.json` after its subscription quota is healthy.

## Local References

The Red Queen paper, the full self-evolution bundle, and the two local provider-bridge reference clones live under `reference/`. They are intentionally ignored by Git because the cloned repositories and PDFs are large local research inputs.
