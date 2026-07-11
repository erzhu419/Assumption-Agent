# Architecture Contract

## Non-Negotiable Invariants

- The slow baseline is always executable and cannot be disabled by a hypothesis.
- Proposal uses training outcomes only; promotion uses validation outcomes only.
- Sealed test items cannot be accessed during proposal, recursive repair, or promotion.
- Utility is measured from paired policy-off/policy-on outcomes, never inferred from failure frequency.
- A policy action must alter the runtime plan or be recorded as non-activated.
- Evaluators are frozen within an epoch.
- Evaluator challengers are compared on the same fixed anchor.
- Replacing an evaluator invalidates only records that depend on its prior epoch.
- Every proposal, validation check, runtime action, evaluation, and promotion decision emits a structured event.
- One protocol-frozen provider route and one model ID serve every proposal, raw, ablation, and agent call in a run.
- Raw, ablations, and the evolving agent use the same digest-pinned container runtime, model, provider policy, and step budget.
- Benchmark task payloads and verifier trees come from the frozen local inventory; online model transport is logged separately from local evaluation evidence.
- Subscription-backed proposal turns may not use tools. Any tool item or server runtime request invalidates the model response.

## Proposal Boundary

The proposal model is behind a transport-neutral JSON contract. The active v3 protocol uses only the ruoli OpenAI-compatible endpoint with `gpt-5.4-mini`; the historical v2 protocol used local Codex app-server before an OpenAI-compatible fallback. Each configured provider receives the same system contract, payload, model ID, and strict output schema.

Codex app-server runs one ephemeral turn in a fresh empty directory. The thread is read-only, has no dynamic tools, environment capabilities, selected capability roots, or approval route, and receives no API-key environment variables. The transport rejects any observed tool event. Logs retain only provider/config/request/response hashes, timing, status, and error classes.

A provider failure opens its circuit for the rest of the run. V3 has no within-run provider fallback, so a failure invalidates pending evidence instead of routing only one arm elsewhere.

The external SkillLearn task agent supports both a historical `codex_subscription` boundary and an `openai_compatible` boundary. Active v3 trials compile a protocol-versioned Codex custom provider into every raw/agent command: the normalized `/v1` endpoint uses the Responses wire API with WebSockets and OpenAI-login requirements disabled. User config is ignored and the turn is ephemeral. The key and base route enter the container only through named environment variables, while upstream `auth_json` setup is disabled; command/event provenance contains hashes and public endpoint identity but no credential value. In subscription mode, the host auth file is copied into a fresh secret temporary directory, mounted as `/root/.codex` for one trial, and destroyed on context exit. The upstream agent registry and subprocess module are restored after every call.

Task containers are built from an exact non-oracle environment hash. The environment image never contains benchmark-provided skills. A single read-only Node/Codex runtime volume, pinned by builder digest and package version, is mounted into every variant. Image ID, runtime key, CLI version, and cache reuse are trial provenance. Different items may execute concurrently, but variants of one item are sequential to prevent within-pair provider contention.

SkillLearn task data and verifier code are local: task payloads enter through a content-addressed image and the verifier enters through a post-agent `docker cp`. The model call remains online. Trial containers enforce a provider-only endpoint allowlist, pinned host mapping without external DNS, fail-closed prebuilt dependency caches, and the active v3.3 64 MiB network fuse; verifier/runtime preparation uses `--network none`. V3.3 additionally freezes low reasoning and verbosity, a 32,768-token `body_after_prefix` local-history checkpoint, the effective 10,000-token tool-output limit, request compression, and disabled remote compaction. A local-history checkpoint still invokes the same model endpoint and is not an evaluator. The previous v3.1/v3.2 contracts are retained as execution-infeasibility diagnostics. This distinguishes offline benchmark evaluation from online model inference without making a false fully-offline-inference claim.

The experiment shares two run-scoped evidence caches. Training evidence is keyed by incumbent executable behavior, train task features, manifest, evaluator epoch, model, and runtime; an unchanged incumbent receives the exact prior observations with zero new task calls. Counterfactual evidence is keyed by candidate and incumbent behavior plus validation task features, evaluator epoch, split, and runtime. Identical behavior reuses exact evidence, while changed behavior misses the cache. Invalid train observations and invalid pair bundles are never cached. Replay is forbidden for sealed-test evidence.

Transient invalid trials use a protocol-bounded retry queue. A replacement is admissible only when split, item, variant, model, provider policy, pair ID, and request hash are unchanged. Valid rows are never rerun by this mechanism; exhausted retries remain invalid and block proposal or promotion.

Proposal transport failures are isolated from experiment persistence. A failed root proposal terminates that generation with a sanitized request hash and a non-claim-eligible report. A failed recursive repair terminates only that candidate branch so the remaining roots can still be audited, but any repair failure blocks validation execution and archive promotion for the generation. Earlier valid train or counterfactual evidence is retained; no model error is silently converted into a rejected scientific hypothesis.

Root proposal evidence is cached only by the complete structured request hash. This binds train residuals, evaluator epoch, runtime capabilities, feature catalog, prior hypotheses, and prior promotion feedback. Paired arms with identical state therefore receive the exact same roots with zero new proposal-model executions; any state change forces a cache miss. Recursive repair responses are not conflated with root proposals.

V2-compiled skills are routed by hashed item ID after evaluating triggers against that item's structured features. The compiler never promotes a family-level match from one item to all sibling items. A missing route means abstain and execute the raw path.

## Benchmark Boundary

SkillLearnBench is connected through an explicit external-trial boundary rather than imported into the policy runtime:

```text
frozen train IDs
  -> no-skill Docker trials + external verifier
  -> sanitized failures + train instruction context
  -> HypothesisProgram proposal / recursive repair
  -> compile matching validation families to SKILL.md
  -> paired no-skill versus generated-skill trials
  -> validity/fairness audit
  -> promotion gate and archive
```

Instruction text is available only as ephemeral train proposal context. It is not a trigger feature and is not persisted in JSONL. Validation instructions are consumed only inside the external trial runner. Test execution is rejected before archive freeze.

The proposer receives an explicit catalog built from frozen train-split runtime features. Both trigger and anti-trigger predicates are rejected if they use keys outside that catalog. All roots are recursively checked on train evidence; a frozen support/anti-support/complexity ordering selects exactly one candidate before validation, preventing held-out outcomes from acting as a same-generation proposal router.

Task and policy hypotheses may control the primary agent runtime. Evaluator hypotheses may not pass through the SkillLearn skill compiler; they require the evaluator-epoch controller, fixed anchor, and dependency invalidation path. Until that separate experiment is executed, no primary result is described as evaluator co-evolution.

An external trial has a stable request hash, pair ID, split, variant, model, step budget, manifest hash, provider fingerprint, fairness fingerprint, metrics, cost, latency, and sanitized error type. Endpoint or container failure is invalid evidence, not a negative task outcome.

The verifier is a delayed capability. Its bind mount is removed before `docker run`; the proxy records that withholding event, waits for the agent command to exit, copies a content-hashed verifier tree into `/tests`, records materialization, and only then invokes the test script. A trace without this event order is not admissible evidence.

External task fallback is prospective, not oracle-assisted. A policy may abstain before execution and leave the raw route unchanged. Once a candidate acts, its output is judged as produced; the system cannot inspect verifier success and retroactively substitute raw. Candidate harm is controlled by paired validation and promotion thresholds.

For the SkillLearn backend, the prospective abstention is the frozen trigger itself: a trigger miss aliases the baseline and is recorded as observed baseline preservation. A trigger hit injects an independent candidate skill and records no same-arm baseline preservation. The compiler exposes only prompt directives and agent-local self-checks; typed lane mutations and the post-agent external verifier are not claimed as lowered actions.

## Archive Unit

An archive node is a complete behavior configuration:

- active hypothesis program IDs;
- runtime and selector version;
- evaluator epoch ID;
- parent node and lineage;
- paired validation records;
- promotion status.

Atomic assumptions remain reusable programs. The archive evaluates interactions between programs rather than pretending their utilities are additive.

## Evaluation Layers

1. Hypothesis quality: schema, falsifiability, scope, executable actions, and fallback.
2. Application fidelity: trigger fit, action activation, lane-plan change, and verifier execution.
3. Outcome: externally judged success, paired gain/harm, cost, latency, and stability.

Only layer 3 may promote a runtime policy. Layers 1 and 2 diagnose why it failed.
