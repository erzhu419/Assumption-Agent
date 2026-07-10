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
- One frozen provider chain and one model ID serve all proposal/repair calls in a run; failover changes transport, not model or evidence access.
- Raw, ablations, and the evolving agent use the same digest-pinned container runtime, model, provider policy, and step budget.
- Subscription-backed proposal turns may not use tools. Any tool item or server runtime request invalidates the model response.

## Proposal Boundary

The proposal model is behind a transport-neutral JSON contract. The default provider order is local Codex app-server followed by an OpenAI-compatible endpoint. Each provider receives the same system contract, payload, model ID, and strict output schema.

Codex app-server runs one ephemeral turn in a fresh empty directory. The thread is read-only, has no dynamic tools, environment capabilities, selected capability roots, or approval route, and receives no API-key environment variables. The transport rejects any observed tool event. Logs retain only provider/config/request/response hashes, timing, status, and error classes.

A provider failure opens its circuit for the rest of the run. The next provider may continue the exact request, but a provider change is retained in provenance and must be accounted for by benchmark fairness fingerprints.

The external SkillLearn task agent uses a distinct `codex_subscription` boundary. The host auth file is copied into a fresh secret temporary directory, that directory is bind-mounted as `/root/.codex` for one Docker trial, and the temporary copy is destroyed on context exit. The upstream agent registry and subprocess module are restored after the call. API keys and alternate base URLs are not passed into subscription-mode containers.

Task containers are built from an exact non-oracle environment hash. The environment image never contains benchmark-provided skills. A single read-only Node/Codex runtime volume, pinned by builder digest and package version, is mounted into every variant. Image ID, runtime key, CLI version, and cache reuse are trial provenance. Different items may execute concurrently, but variants of one item are sequential to prevent within-pair provider contention.

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

An external trial has a stable request hash, pair ID, split, variant, model, step budget, manifest hash, provider fingerprint, fairness fingerprint, metrics, cost, latency, and sanitized error type. Endpoint or container failure is invalid evidence, not a negative task outcome.

The verifier is a delayed capability. Its bind mount is removed before `docker run`; the proxy records that withholding event, waits for the agent command to exit, copies a content-hashed verifier tree into `/tests`, records materialization, and only then invokes the test script. A trace without this event order is not admissible evidence.

External task fallback is prospective, not oracle-assisted. A policy may abstain before execution and leave the raw route unchanged. Once a candidate acts, its output is judged as produced; the system cannot inspect verifier success and retroactively substitute raw. Candidate harm is controlled by paired validation and promotion thresholds.

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
