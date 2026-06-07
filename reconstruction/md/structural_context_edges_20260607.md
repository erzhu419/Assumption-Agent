# Structural Context Edges - 2026-06-07

## Why

The previous QA probes showed that ordinary word-level retrieval and direct morphism routing solve different problems. The useful idea from HippoRAG is not only synonym expansion, but context spreading: local terms activate neighboring context, and the graph retrieves a larger meaningful neighborhood.

For Assumption OS, the analogous layer should not stop at word-level context edges. It should lift context edges into generalized assumption contexts:

`phrase -> structural role -> generalized assumption context -> pattern -> classic realization`

Example:

`growth / increase -> opposing response -> constraint / equilibrium -> pat_negative_feedback -> Lenz + Le Chatelier`

This lets the agent treat Le Chatelier and Lenz as structural priors for a family of cases, not as an answer template.

## Mechanism

New module:

- `assumption_os/structural_context_edges.py`

It implements:

- role-level structural synonym edges;
- generalized context edges from roles to `pat_negative_feedback`;
- context-to-realization edges for `real_lenz_negative_feedback` and `real_le_chatelier_shift`;
- negative-control edges for positive feedback, runaway amplification, random response, and missing opposition/constraint.

The accepted prior is deliberately bounded:

> When growth or perturbation induces a compensating response under a constraint, do not extrapolate monotonic growth blindly; expect dampening, plateau, or convergence toward a constrained equilibrium unless negative controls indicate runaway amplification.

This is the key difference from overfitting. It does not say “everything converges”; it says to check perturbation, induced opposition, preserved constraint, lag/overshoot, and positive-feedback controls.

## Validation

Artifact:

- `phase four/assumption_graph/paper_readiness_20260604/structural_context_edges_20260607.json`

Results:

| metric | value |
|---|---:|
| positive cases | 7 |
| negative controls | 4 |
| structural-context positive recall | 1.0000 |
| word-context baseline positive recall | 0.0000 |
| negative-control block/abstain rate | 1.0000 |
| classic realization expansion rate | 1.0000 |

Positive examples include:

- platform traffic growth triggering rate limits / moderation queues;
- prey population growth triggering predators / resource competition;
- Chinese market demand growth triggering price/supply counter-response;
- API retry growth triggering backoff / circuit breakers;
- direct Lenz and Le Chatelier cases.

Negative controls include:

- runaway positive feedback;
- monotone ad-spend forecast with no opposing mechanism;
- random jitter without induced compensation;
- plain bottleneck capacity without opposing-response structure.

## Interpretation

This directly addresses the concern that the agent should know when to consult a same-category classic result. The context edge is no longer “this word is near that word”; it is:

`this problem preserves the same structural assumption context as a known family`

For paper language, the safe claim is:

> We generalize HippoRAG-style synonym/context spreading from lexical neighborhoods to bounded assumption-context neighborhoods. This lets structural roles activate reusable priors such as negative-feedback equilibrium restoration, while negative controls prevent treating all growth processes as convergent.

Follow-up `assumption_family_discovery_20260607` generalizes this from one hand-built context to open-set assumption-family induction over multiple scientific, mathematical, engineering, and philosophical theory cards.

## Reproduction

```bash
python3 -m assumption_os.structural_context_edges \
  --out 'phase four/assumption_graph/paper_readiness_20260604/structural_context_edges_20260607.json'

python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_structural_context_edges_generalize_hipporag_context
```
