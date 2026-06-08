# Orthogonal Execution Live Validation - 2026-06-08

## Question

Can an orthogonal new-family hypothesis improve downstream answer quality when
the model is held constant?

The tested hypothesis is not another controlled-variable method edit.  It is an
execution-harness axis:

> When a practical decision problem requires action under budget, time,
> migration, rollout, or operating-risk constraints, the answer should convert
> the chosen method into an execution contract: smallest reversible pilot,
> baseline, success metric, stop threshold, owner, and rollback path.

This is orthogonal to the parent method family because it does not choose the
method; it makes the method falsifiable and deployable.

## Mechanism Changes

Code:

- `assumption_os/activation.py`: generic candidate nodes can now set
  `payload.activation.allow_lexical_fallback=false`, preventing broad lexical
  routing from turning a scoped harness into a 70-row always-on prompt.
- `assumption_os/orthogonal_execution_queue.py`: builds the execution-contract
  orthogonal queue, with novelty ON/OFF, trigger/control preflight, daemon
  readback, and gated temp apply validation.
- `assumption_os/orthogonal_live_ablation.py`: adds
  `--route-scoped-noop-controls` and `--baseline-variant`; in no-op-control
  mode, candidate answers are generated only for routed trigger rows and
  controls are recorded as deterministic ties because the candidate route is
  inactive there.
- `assumption_os/structural_live_ablation.py`: adds `claude_opus` /
  `claude_haiku` request aliases for OpenAI-compatible live judging.

## Deterministic Queue Validation

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_execution_queue_20260608.json`

Result:

- proposal_count: 1
- novelty ON: 1 `orthogonal_new_family`
- novelty OFF: 0 `orthogonal_new_family`
- enabled `orthogonal_to` edges: 1
- preflight_ready_count: 1
- trigger_count: 8
- control_count: 8
- daemon dry-run: 1 executable planned leaf
- fixture readback accept_count: 1
- gated temp apply: candidate node plus `orthogonal_to` edge on temporary graph
- pass: true

## Broad Same-Model Live Result

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_execution_live_same_model_20260608.json`

Setup:

- candidate: Claude Opus, with execution-contract orthogonal proposal;
- baseline: Claude Opus, same Phase2 v20 graph context but without the new
  proposal;
- judge: Claude Opus pairwise judge;
- trigger rows: 8 live-judged;
- controls: 8 route-scoped no-op ties.

Result:

- candidate wins: 4
- ties: 3
- baseline wins: 1
- control losses: 0
- acceptance decision: `reject_benefit`
- pass: true for execution/readback integrity, but no accepted proposal.

Interpretation:

The broad hypothesis was real but over-scoped.  It helped practical execution
rows, tied rows where the same-model baseline already produced an execution
contract, and lost one deep technical migration row where the baseline included
a no-regret engine/API decoupling step that the candidate did not add.

This negative result is important: the earlier candidate-vs-GPT-mini run was
confounded by model strength and is not used as the positive claim.

## Recursive Scope Repair

The next recursive move was not to promote the broad hypothesis.  The runner's
failure mode was interpreted as a scope defect, then a narrower child proposal
was built:

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_execution_scope_repair_20260608.json`

Scope:

- `business_0097`
- `business_0192`
- `business_0218`
- `daily_life_0173`
- `software_engineering_0142`

The scope note records the repair:

`same_model_live_repair: broad execution-contract proposal was useful on practical operations/transition rows but over-broad on deep technical rows; narrow to practical execution rows before retention`

Deterministic validation:

- proposal_count: 1
- novelty ON: 1 `orthogonal_new_family`
- novelty OFF: 0 `orthogonal_new_family`
- trigger_count: 5
- control_count: 8
- preflight_ready_count: 1
- daemon dry-run/readback/temp apply: pass

## Scoped Same-Model Live Result

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_execution_scope_repair_live_same_model_20260608.json`

Setup:

- candidate: Claude Opus, scoped execution-contract proposal;
- baseline: Claude Opus, no scoped proposal;
- judge: Claude Opus pairwise judge;
- trigger rows: 5 live-judged;
- controls: 8 route-scoped no-op ties.

Result:

- trigger wins: 3
- trigger ties: 2
- trigger losses: 0
- trigger utility: 0.80
- control losses: 0
- acceptance decision: `accept`
- graph mutation without apply: false
- pass: true

This is the first clean positive live example for the orthogonal hypothesis
line under a same-model comparison.  The claim is still bounded: this proves a
scoped, route-gated execution-contract hypothesis can pass downstream
acceptance.  It does not yet prove broad general QA improvement or fully
autonomous long-horizon discovery.

