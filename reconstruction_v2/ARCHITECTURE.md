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
- Freeze and control execution require a nonempty promoted recursive incumbent; an empty archive is a completed negative development result, not a control treatment.
- Every proposal, validation check, runtime action, evaluation, and promotion decision emits a structured event.
- A proposal-only feasibility diagnostic may block future trial spend, but its acceptance state is never a promotion score, benchmark outcome, or additional promotion gate.
- One protocol-frozen provider route and one model ID serve every proposal, raw, ablation, and agent call in a run.
- Raw, ablations, and the evolving agent use the same digest-pinned container runtime, model, provider policy, and action budget.
- Benchmark task payloads and verifier trees come from the frozen local inventory; online model transport is logged separately from local evaluation evidence.
- Subscription-backed proposal turns may not use tools. Any tool item or server runtime request invalidates the model response.

## Proposal Boundary

The proposal model is behind a transport-neutral JSON contract. The last completed benchmark protocol, v3.15, uses only the ruoli OpenAI-compatible endpoint with `gpt-5.4-mini`; the historical v2 protocol used local Codex app-server before an OpenAI-compatible fallback. Each configured provider receives the same model ID and strict output schema. Root proposal calls keep exact-three cardinality and atomic parse as response invariants, while train-failure activation-signature diversity is a search audit. Equal-signature candidates may continue when their action/backend treatments differ; no candidate-completion retry is added. Recursive repair remains explicitly singular: v3.12's `single_candidate_excludes_root_batch_contract_v1` adds a top-level one-object/`hypothesis` response contract, retains the train-only coverage objective, and defensively removes root batch semantics if a caller supplies them. V3.15 conditionally adds the request-local `train_only_material_action_delta_prompt_audit_v1` contract; legacy protocol prompts remain unchanged when that contract is absent. The v3.13 live development exercised the singular repair path without a response/model-shape failure, but did not establish repair benefit because no candidate was promoted.

V3.16/V3.17 add an isolated proposal-feasibility branch before the benchmark boundary:

```text
frozen V3.15 TRAIN receipt
  -> local 38-row / 31-profile reconstruction, zero agent re-execution
  -> three family proposal slots
  -> production proposer kernel, one logical model call per slot
  -> local trigger/action/profile audit
  -> pass: only makes a separately frozen typed-selection integration diagnostic eligible
  -> pass does not authorize a development protocol or task execution
  -> fail: stop before backend, evaluator, validation, promotion, or task trial
```

V3.16's V1 formation assigned singular family slots and profile grounding. V3.17's V2 formation additionally makes the trigger exactly `family == target`, fixes the anti-trigger to empty, selects one reusable artifact deterministically, and supplies a read → parse → update → serialize → write-back blueprint. The failed primitive values remain local to the production audit; only a count and set hash enter the model request. Reports and the provider/model/proposer ledger persist hashes and bounded metadata, never raw responses, source instructions, or credentials.

The replacement representation is now implemented and preregistered for one formal offline decision, which has not yet run. It reconstructs 38 complete receipt-bound V3.15 TRAIN traces as 429 chronological allowlisted command occurrences, including 70 failed occurrences, while separately counting 208 discarded commands. The completeness claim is therefore limited to the allowlisted chronology; it does not claim full raw-command coverage. Each family graph registers task-local artifacts, capabilities, typed operators, and recipes, while the proposal output is closed to one opaque registered `recipe_id`. Primitive values, artifact locators, and free-text actions are outside that selection grammar. This boundary closes proposal generation, not runtime execution: the selected recipe still becomes harness-owned prompt directives/self-checks for the general SkillLearn agent, capability implementation is unverified, and no restricted executor or restricted runtime-argument surface is claimed. Even a PASS only permits a separately frozen typed-selection integration diagnostic; development remains unauthorized.

Codex app-server runs one ephemeral turn in a fresh empty directory. The thread is read-only, has no dynamic tools, environment capabilities, selected capability roots, or approval route, and receives no API-key environment variables. The transport rejects any observed tool event. Logs retain only provider/config/request/response hashes, timing, status, and error classes.

A provider failure opens its circuit for the rest of the run. V3 has no within-run provider fallback, so a failure invalidates pending evidence instead of routing only one arm elsewhere.

The external SkillLearn task agent supports both a historical `codex_subscription` boundary and an `openai_compatible` boundary. Claim-bearing v3.15 trials compile a protocol-versioned Codex custom provider into every raw/agent command: the normalized `/v1` endpoint uses the Responses wire API with WebSockets and OpenAI-login requirements disabled, while authoritative top-level `web_search="disabled"` removes hosted web search from the actual request. User config is ignored and the turn is ephemeral. The key and base route enter the container only through named environment variables, while upstream `auth_json` setup is disabled; command/event provenance contains hashes and public endpoint identity but no credential value. In subscription mode, the host auth file is copied into a fresh secret temporary directory, mounted as `/root/.codex` for one trial, and destroyed on context exit. The upstream agent registry and subprocess module are restored after every call.

Task containers are built from an exact non-oracle environment hash. The environment image never contains benchmark-provided skills. A single read-only Node/Codex runtime volume, pinned by builder digest and package version, is mounted into every variant. Image ID, runtime key, CLI version, and cache reuse are trial provenance. Different items may execute concurrently, but variants of one item are sequential to prevent within-pair provider contention.

SkillLearn task data and verifier code are local: task payloads enter through a content-addressed image and the verifier enters through a post-agent `docker cp`. The model call remains online. Trial containers enforce a provider-only endpoint allowlist, pinned host mapping without external DNS, fail-closed prebuilt dependency caches, and the active v3.15 64 MiB network fuse; verifier/runtime verification uses `--network none`. Six item pipelines may prepare containers and execute offline verifiers concurrently, but one shared semaphore surrounds only the `docker exec ... codex exec` agent stage. This keeps Ruoli inference concurrency at one after v3.7/v3.8 failed at six/two simultaneous model calls. The container-local supervisor owns the JSONL trace, counts all `item.started` events, binds a per-attempt nonce and trace hash, and removes every live task created after the dedicated trial-container baseline before verifier injection. The `/proc/<tgid>/task/<tid>` audit covers new sessions created with `setsid` and a live worker whose thread-group leader is already a zombie; an incomplete task scan fails closed. Natural completion requires one `turn.completed` with valid token usage; a budget truncation may omit usage but all arms use action starts for promotion cost. A local-history checkpoint still invokes the same model endpoint and is not an evaluator. The previous v3.1-v3.14 contracts are retained as prior evidence. This distinguishes offline benchmark evaluation from online model inference without making a false fully-offline-inference claim.

The experiment shares distinct run-scoped training, baseline-arm, and paired-counterfactual evidence caches. Training evidence is keyed by incumbent executable behavior, train task features, manifest, evaluator epoch, model, and runtime; an unchanged incumbent receives the exact prior observations with zero new task calls. V3.14 adds an immutable validation policy-off cohort shared by recursive and no-recursive runners across generations. Its identity binds the baseline behavior/treatment and frozen task/runtime/fairness context but excludes challenger-specific pair metadata, so changing a candidate cannot resample the raw arm. One per-key lock admits the first valid observation, and a conflicting observation cannot mutate the cohort. V3.15 additionally memoizes an exhausted same-request terminal-invalid baseline outcome. Later consumers replay that tombstone with zero new baseline execution, but it remains `promotion_evidence=false`, cannot become a valid row or score, and only propagates non-claim. Paired counterfactual evidence remains keyed by candidate and incumbent behavior plus validation task features, evaluator epoch, split, and runtime. Identical valid behavior reuses exact evidence, while changed behavior misses the paired cache. Invalid train observations and invalid pair bundles are never cached. Replay is forbidden for sealed-test evidence.

The Plus and Pro Ruoli credentials both authorize the same frozen `gpt-5.4-mini` route. Credential-tier substitution alone is not a model/provider-treatment change and does not invalidate or require repeating already passed lock, prewarm, or smoke phases; a failed phase uses an isolated event/work tree. A protocol/code revision such as v3.15 does require a new lock, cache-only prewarm, smoke, and run root; it can reuse existing content-addressed model/runtime images and dependencies rather than downloading them again.

Recursive repair identity and lifecycle status are harness-owned rather than model-owned. The proposer forces each repair to enter as `candidate`, then derives its branch ID from a versioned policy, the parent program ID, a status-independent canonical parent-content hash, repair depth, and canonical candidate child content with the model-declared ID removed. This makes same-content replay deterministic, distinguishes sibling or deeper branches, and prevents accidental self-parent IDs. The archive still fails closed if one canonical ID is ever presented with different payload content.

Transient invalid trials use a protocol-bounded retry queue. A replacement is admissible only when split, item, variant, model, provider policy, pair ID, and request hash are unchanged. Valid rows are never rerun by this mechanism; exhausted training retries block proposal. Exhausted validation evidence terminates a v3.6-v3.15 generation as non-claim without incrementing the scientific consecutive-non-promotion counter. In v3.15 only, an exhausted baseline outcome may populate the non-promotional terminal-invalid memo described above. Evaluator-invalid, provider-mismatched, and budget-mismatched pair bundles never enter counterfactual replay.

Proposal failures are isolated from experiment persistence at a typed boundary covering transport, JSON, request-specific envelope, and canonical-program parsing. Root responses stage every consumed row before emitting proposal events or populating replay, so a mixed malformed response is atomic. A failed root proposal terminates that generation with sanitized request/response hashes and a non-claim report. A failed recursive repair terminates only that candidate branch so the remaining roots can still be statically audited, but any repair failure blocks validation execution and archive promotion for the generation. Reports derive claim state from generation failure rows, and paper freeze independently revalidates it. Parseable scientific candidates still face the existing static checks; archive and harness invariants remain outside this catch boundary.

Root proposal evidence is cached only by the complete structured request hash. This binds train residuals, evaluator epoch, runtime capabilities, feature catalog, prior hypotheses, and prior promotion feedback. Paired arms with identical state therefore receive the exact same roots with zero new proposal-model executions; any state change forces a cache miss. Recursive repair responses are not conflated with root proposals.

V2-compiled skills are routed by hashed item ID after evaluating triggers against that item's structured features. The compiler never promotes a family-level match from one item to all sibling items. A missing route means abstain and execute the raw path.

## Benchmark Boundary

SkillLearnBench is connected through an explicit external-trial boundary rather than imported into the policy runtime:

```text
frozen train IDs
  -> no-skill Docker trials + external verifier
  -> labeled failures + instruction/context-free success controls
  -> HypothesisProgram proposal / recursive repair
  -> compile matching validation families to SKILL.md
  -> paired no-skill versus generated-skill trials
  -> validity/fairness audit
  -> promotion gate and archive
  -> nonempty recursive incumbent
  -> freeze and controls
```

Instruction text is available only as ephemeral failed-train proposal context. A successful negative control contains the runtime feature label but no instruction, feedback, or execution context. Instruction text is not a trigger feature and is not persisted in JSONL. Validation instructions are consumed only inside the external trial runner. Test execution is rejected before archive freeze.

V3.15 supplements failed-TRAIN proposal context with request-local action profiles. The environment component reads only a safe allowlist of non-oracle task-environment metadata; the baseline-action component retains bounded normalized command signatures, status/exit metadata, and `/root` file-change summaries. Profiles are hashed and shared in the first-generation checkpoint, while raw logs, command output, model prose, credentials, solutions, tests, verifiers, validation, and sealed content are excluded. The proposal-side action-delta audit is observability only: it does not reject or retry a response, alter candidate ordering, or participate in promotion.

The proposer receives an explicit catalog built from frozen train-split runtime features. Both trigger and anti-trigger predicates are rejected if they use keys outside that catalog. All roots are recursively checked on train evidence. V3.6-v3.9 select one candidate by exact failure-activation precision before support and complexity. V3.10-v3.12 rank individual roots with a train-only family-coverage objective. V3.13 introduced enumeration of the at-most-seven nonempty subsets of the exact-three static-valid roots and fixed exactly one canonical delta set before held-out access. Its order was union precision, capped family-target deficit, success false positives, overlap, bundle size, failure support, complexity, and set hash. V3.14-v3.15 retain precision, capped deficit, success false positives, and overlap as the leading terms, but place actual family count and failure support before bundle size. Thus lower-precision or higher-false-positive members cannot be forced in for breadth, while otherwise tied precise sets no longer lose solely for being larger despite covering more observed TRAIN families/failures. The compiler routes every matching member independently and the policy-on arm runs once per item when at least one new member matches. Delta/full/matched set hashes distinguish causal treatment from incumbent behavior and per-item activation. Program-set replay is order-invariant but never reuses `{A}` evidence for `{A,B}`. Bundle rejection belongs to the archive node; component hypotheses remain shadow. None of this reads validation features/outcomes or adds a promotion blocker.

Task and policy hypotheses may control the primary agent runtime. Evaluator hypotheses may not pass through the SkillLearn skill compiler; they require the evaluator-epoch controller, fixed anchor, and dependency invalidation path. Until that separate experiment is executed, no primary result is described as evaluator co-evolution.

An external trial has a stable request hash, pair ID, split, variant, model, action budget, manifest hash, provider fingerprint, fairness fingerprint, metrics, cost, latency, and sanitized error type. Endpoint or container failure is invalid evidence, not a negative task outcome.

The verifier is a delayed capability. Its bind mount is removed before `docker run`; the proxy records that withholding event, waits for the agent command to exit, copies a content-hashed verifier tree into `/tests`, records materialization, and only then invokes the test script. A trace without this event order is not admissible evidence.

External task fallback is prospective, not oracle-assisted. A policy may abstain before execution and leave the raw route unchanged. Once a candidate acts, its output is judged as produced; the system cannot inspect verifier success and retroactively substitute raw. Candidate harm is controlled by paired validation and promotion thresholds.

For the SkillLearn backend, the prospective abstention is the frozen trigger itself: a trigger miss aliases the baseline and is recorded as observed baseline preservation. A trigger hit injects an independent candidate skill and records no same-arm baseline preservation. The compiler exposes only prompt directives and agent-local self-checks; typed lane mutations and the post-agent external verifier are not claimed as lowered actions. Lowering v2 always retains each action target, renders mapping values as deterministic readable phrases, and keeps the top-level fallback separate from already-activated action nodes.

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
2. Application fidelity: trigger fit, action activation, lane-plan change, verifier execution, evidence-valid activation precision/harm, and abstention.
3. Outcome: externally judged success, paired gain/harm, cost, latency, and stability.

Only layer 3 may promote a runtime policy. Layers 1 and 2 diagnose why it failed. V3.6-v3.15 activation precision, activated harm rate, abstention fields, and the v3.15 material-action audit are diagnostics only and do not add promotion blockers. V3.14-v3.15 program sets still use `evaluator_owned_paired_validation_v2`; member-declared effect, harm, and cost limits are conservatively aggregated before the unchanged protocol thresholds are applied. On SkillLearn, `selection_change_count` currently compares the boolean success projection, not trace identity; command/answer changes can therefore exist when this field is zero.

The V3.17 proposal-only diagnostic passed eight of nine layer-1 feasibility checks: three distinct single-family signatures, target support 2/2/3, valid schema, no target self-block, profile binding, concrete executable deltas, and no restatement. It failed failed-primitive avoidance because the third free-text action bound two primitives extracted from failed TRAIN commands. No layer-2 application or layer-3 outcome evaluation was therefore authorized.

V3.12 completed clean development but produced no layer-3 improvement: 56/56 external trials were valid, both generations activated 1/16 and tied raw at 4/16 with zero gain/harm, and both archives remained empty. A subsequent empty-freeze/partial-control admission error is quarantined and cannot count as control evidence. The runner, freeze producer, and control consumer now enforce the same nonempty-incumbent phase prerequisite before any later model work; no promotion threshold changed.

V3.13 then completed a clean 76/76-trial development: all six activated policy-on trials failed, all generation decisions remained at 2/16 activation with zero gain/harm, and both archives retained `incumbent_id=null`. It established program-set routing and live singular-repair mechanics, not performance. V3.14 selected the previously missed 7/7 three-family set and raised held-out activation to 3/16, proving the new tied-set ordering works, but all seven executed policy-on trials across its completed arms still failed with zero gain/harm. One recursive policy-off trial exceeded the unchanged 64 MiB fuse, so the primary report is non-claim; the no-recursive report completed two mechanically claim-eligible non-promotions. Valid baseline evidence produced 31 cross-consumer replays, while the invalid key was re-executed once because invalid state is not memoized. Both archives remain empty. The result closes selector iteration and localizes the next problem to executable action content, not another promotion gate; it makes no improvement, family-out, HippoRAG, or sealed-test claim.

V3.15 implements the bounded action-quality and terminal-invalid corrections at commit `696a2954`; 453/453 offline tests pass. Its live root passed a clean claim lock, 86/86 cache-only prewarm, and smoke, then completed 57/57 valid actual trials: 38 TRAIN policy-off observations, one 16-item shared validation baseline cohort, and three activated policy-on executions. All 8/8 proposal/repair model calls completed, maximum online model concurrency was one, and no provider, infrastructure, action-budget, network-cap, or pair-mismatch failure occurred. TRAIN contained 6 successes and 32 residuals. Both reports are claim-eligible. Recursive G1/G2 each activated 1/16 and tied raw at 4/16 with zero gain/harm; no-recursive G1 was statically rejected and G2 activated 1/16 with zero gain/harm. The baseline cohort produced 32 zero-execution replays. Both arms terminated at `consecutive_non_promotion_limit`, both archives remain empty, and sealed/test access and every downstream phase remain false.

V3.16/V3.17 show that structural family allocation, exact triggers, artifact blueprints, and stronger prose can remove family collapse, self-blocking, and restatement without making hard negative constraints reliable. Their failed primitive check also conflated failed-command co-occurrence with causal inadmissibility. The new causal span extractor and closed typed graph move the proposal constraint into the representation: the model can select only a registered recipe, and forbidden primitive/action fields are not expressible. The implementation and preregistration are ready, but the one formal offline feasibility decision is pending. This is still a representation boundary rather than a trustworthy restricted treatment runtime. PASS means only `typed-selection integration diagnostic freeze-eligible`; PASS does not authorize development.
