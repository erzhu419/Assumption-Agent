# Reconstruction V2 Status

## Implemented

- Three typed hypothesis kinds: task, policy, and evaluator.
- Structured `HypothesisProgram` with trigger, anti-trigger, action DAG, expected effect, verifier, fallback, lineage, and evaluator epoch.
- Runtime actions that actually enable, disable, and prioritize lanes or inject executable operator steps.
- Baseline preservation in the internal lane runtime; external SkillLearn trials use pre-execution abstention plus a paired harm gate rather than post-verifier rollback.
- JSON-only model proposal adapters with protocol-approved Spark and `gpt-5.4-mini` routes.
- Active v3 single-provider routing through ruoli `gpt-5.4-mini`, with a per-run circuit breaker and provider provenance.
- A route policy embedded in experiment plans, provider fingerprints, and fairness fingerprints so every raw/agent arm uses one model and provider mode.
- Protocol-bound Codex custom Responses-provider commands for v3, with `/v1` normalization, ignored user config, ephemeral turns, and no credential in command/event provenance.
- ChatGPT subscription reuse through the local official Codex app-server transport; legacy direct OAuth is not used.
- Isolated ephemeral proposal turns with strict JSON Schemas, empty workspace roots, no accepted tools, and a child-environment secret allowlist.
- SkillLearn Codex trials use a separate subscription-auth mode: an ephemeral Codex home is mounted into the task container, API endpoint variables are omitted, and the upstream runner is restored after every call.
- Recursive validation that asks the proposal model for a child repair after failed checks.
- Same-task policy-off/policy-on counterfactual execution.
- Promotion from paired validation effect, harm, activation, and cost rather than failure frequency.
- Archive nodes representing complete active-program configurations.
- Frozen evaluator epochs, anchor lower-bound challenger comparison, and selective score invalidation.
- Train/validation/sealed-test access guard.
- SkillLearnBench inventory, instance-holdout and family-out manifests.
- Credential-independent 95-item paper subset plus manifest-scoped `required_env` hard preflight.
- Compiler from promoted hypothesis programs to SkillLearnBench-compatible `SKILL.md` files.
- SkillLearnBench trial contracts and a sanitized adapter around the upstream Docker verifier runner.
- Failed train-only trials converted into semantic residuals with task instructions available to proposal but absent from persistent logs and runtime triggers.
- Validation and sealed-test compiler targeting, including family-out transfer to unseen validation families.
- Deterministically balanced policy-off/policy-on execution order with provider and budget fingerprints.
- Promotion blockers for invalid endpoint/container rows, provider mismatch, and budget mismatch.
- A guarded `skilllearn_experiment` dry-run/execute CLI with a model health probe before expensive trials.
- Bounded multi-generation evolution with aggregate promotion feedback, behavior deduplication, rejection status, and consecutive-non-promotion early stopping.
- Paper protocol lock, recursive/no-recursive archive freeze, content-hashed compiled controls, resumable matched controls, sealed-access journal, and item-clustered paper statistics.
- Post-agent verifier injection: `/tests` is absent while the model acts and copied into the container only after agent exit.
- Content-addressed per-item base images with oracle skills excluded, plus one digest-pinned, read-only `codex-cli 0.144.1` runtime volume.
- Zero-model-cost train/validation image prewarm with bounded retries and a manifest-bound provenance receipt.
- Bounded four-worker train/counterfactual/control execution across items, with every same-item variant sequence kept serial.
- Runner-local deep copies of the upstream agent registry, eliminating parallel subscription setup/restore races.
- No-fixed-timeout policy for active agent/verifier stages and the trial-container lifetime, with legacy timeout results classified as invalid evidence.
- Structured Codex terminal-error detection, a shared provider-failure circuit, and all-valid training evidence before proposal.
- Bounded, audited cleanup retries for ephemeral subscription-auth homes.
- Shared first-generation evidence and root checkpoint for the recursive versus no-recursive causal ablation.
- Behavior-identical validation replay across paired ablation arms, preventing model resampling variance from appearing as a recursive-repair effect.
- Hard runtime trigger vocabulary with context-only instruction fields rejected programmatically.
- Train-only evaluation of every proposed root and deterministic support/complexity selection of the sole validation candidate.
- Separate aggregate and selected-candidate recursion counts, depths, and repair-candidate provenance.
- Runtime-kind guard: evaluator hypotheses cannot masquerade as compiled agent skills and require a separate anchored epoch challenger.
- Per-item compiled skill routing, preventing a trigger match on one item from leaking to sibling items in the same family.
- Trial provenance for base image key/ID, runtime key/version, cache reuse, withheld verifier mount, and post-agent verifier materialization.
- Per-attempt model audit events; non-retryable HTTP authentication errors stop after one request.
- Structured in-memory or JSONL events for proposal, validation, runtime, counterfactual, promotion, archive, evaluator transition, and benchmark compilation.

## Evidence Scope

The 55 offline tests prove that the learning loop is connected, that a promoted hypothesis changes future runtime behavior, and that the same lifecycle works against SkillLearnBench's real inventory with a fake external trial backend. They also cover invalid-pair blocking, all-valid training evidence before proposal, provider circuit breaking, structured Codex terminal errors, split isolation, provider failover, strict Codex schemas, runtime tool rejection, trigger-vocabulary isolation, all-root train-only selection, credential-independent manifests, required-env blocking, prewarm receipt binding, ephemeral subscription auth and cleanup retry, runner-local registry isolation, post-agent verifier injection, active-stage timeout removal, timeout-row invalidation, content-addressed environment caching, shared-checkpoint ablation, behavior-identical validation replay and behavior-different cache misses, ordered item parallelism, per-item skill routing, runtime mismatch blocking, content-bound freeze receipts, invalid-only retries, paper statistics, the ruoli model route, endpoint-origin normalization, custom-provider command sanitization, provider-scoped error labels, and route/config drift rejection. They do not prove real benchmark improvement.

Docker Engine 29.1.3 is installed and the real upstream runner, historical ChatGPT subscription auth, Spark model canary, and container canary have passed. The pinned shared-runtime raw canary completed in 43.7 seconds with `agent_exit=0`, `verifier_exit=0`, no timeout, and matching Spark/runtime/image provenance. Its ordered audit events prove that the verifier mount was withheld during agent execution and materialized only afterward. The task result was wrong, so this is infrastructure evidence only. Dataclaw is absent but optional for task-success execution; it is needed only for parts of the separate trajectory/skill post-processing pipeline.

A prior Spark subscription canary returned successfully through local Codex login, but the subscription is currently quota-exhausted. The replaced ruoli credential passed sanitized `gpt-5.4-mini` probes against both chat-completions and Responses on 2026-07-10. A host Codex custom-provider canary then completed over the Responses route with user config ignored, WebSockets disabled, and no OpenAI login. These checks prove endpoint and Codex transport compatibility, not benchmark quality or independent verification of the provider's internal model mapping.

A real exhausted-quota canary now returns `subscription_usage_limit`, opens the shared circuit, skips verifier materialization, completes ephemeral-auth cleanup, and records the container-lifetime rewrite. This is failure-semantics evidence only; the canary is not a benchmark row and makes no performance claim.

The v2 proposal API can be injected directly with `ASSUMPTION_V2_*` variables. For local continuity, `--env-file ../.env` also maps the existing `GPT5_*` / `RUOLI_*` aliases into `ASSUMPTION_V2_API_BASE` and `ASSUMPTION_V2_API_KEY` without persisting secrets. `ASSUMPTION_V2_PROVIDER_CHAIN` controls the fixed provider order, and `ASSUMPTION_V2_CODEX_PATH` optionally selects the local Codex binary. The active v3 protocol locks `gpt-5.4-mini` to `openai_compatible`; the retained v2 protocol locks Spark to subscription mode. Other models require the explicit diagnostic-only override.

No accuracy or superiority claim exists yet. Five historical Spark development attempts exposed credential, image-build, registry-race, fixed-timeout, container-lifetime, and quota/error-classification defects; all are invalid diagnostics and sealed test remained untouched. The first v3 smoke also produced four invalid train rows because Codex ignored a bare `OPENAI_BASE_URL` and contacted the official OpenAI endpoint with the third-party credential. It generated no proposal and touched no sealed item; its sanitized artifacts are retained under `artifacts/paper_primary_v3_ruoli_gpt54mini/diagnostics/f2a7a9b0_pre_custom_codex_provider`, with the four upstream `auth.json` credential files deleted. A full literal scan of tracked V2 files and the v3 artifact tree reports zero remaining configured-key values. V3 now injects an explicit custom provider instead of relying on that environment variable.

The first post-fix v3 smoke at commit `60352f15` completed all 12 external trials with 12 custom-provider preparation events, zero invalid rows, zero provider/budget mismatches, zero infrastructure errors, and no sealed access. It proved the ruoli/Codex transport and full proposal-to-counterfactual lifecycle, but not performance: recursive validation was a 1/2 versus 1/2 tie and no-recursive was 0/2 versus raw 1/2. Both arms selected the same hypothesis and archive-node behavior with no recursive repair, so the disagreement came from independently resampling identical policies. That smoke is retained under `artifacts/paper_primary_v3_ruoli_gpt54mini/diagnostics/60352f15_pre_counterfactual_replay` as diagnostic evidence for the new behavior-identical replay policy and is not an admissible recursive-ablation result. Protocol v3 does not inherit performance evidence from Spark or either diagnostic smoke. The first admissible v3 result must come from a clean run under the replay-locked protocol; sealed test remains untouched until an archive is frozen.

One initial real smoke attempt using the previous model was interrupted and marked invalid after audit found that upstream SkillLearnBench mounted `/tests` during agent execution. No hypothesis was generated and no sealed item was accessed. The adapter now removes that mount; a real container inspection confirmed only `/logs`, the read-only agent runtime, and the ephemeral Codex home are mounted during the agent stage.

## Legacy Isolation

No module in reconstruction v2 imports `assumption_os.hle_smoke_eval` or its policy branches. HLE integration will be a separate adapter after the v2 lifecycle passes an external continual-learning benchmark.
