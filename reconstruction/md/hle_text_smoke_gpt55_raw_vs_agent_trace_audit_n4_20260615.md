# HLE Assumption Agent Trace Audit

- pass: `True`
- sample count: `4`
- raw accuracy: `0.5`
- assumption_agent accuracy: `0.5`
- agent minus raw: `0.0`
- transition counts: `{'both_correct': 2, 'both_wrong': 2}`
- context decisions: `{'use_context': 2, 'abstain_to_raw_prompt': 2}`
- failed gates: `[]`

## Diagnosis

- The full Assumption Agent wrapper tied raw gpt-5.5: no regression, but no rescue cases.
- assumption_graph_retrieval activated on 4/4 agent rows.
- structural_morphism_transfer activated on 4/4 agent rows.
- world_model_router activated on 4/4 agent rows.
- recursive_assumption_runner activated on 4/4 agent rows.
- There were no agent_rescue transitions; every raw-wrong problem remained wrong.
- There were no agent_regression transitions; the gated wrapper did not harm this slice.
- World model injected context on 2 rows.
- World model abstained to raw prompt on 2 rows.
- Structural morphism hits were weak/repair-level on 2 rows, so morphism transfer was active but not strong answer-bearing evidence.
- Formal mapping produced zero hits on this HLE slice; the agent relied on generic graph/structural context.
- At least one exact-match question received injected context; this can add latency/structure without supplying missing factual knowledge.
- Recursive runner produced applicability frames, but did not execute child answer-generation/judge loops inside this single-call HLE wrapper.

## Problem-Level Trace

| problem hash | answer type | category | raw | agent | transition | decision | context | formal hits | structural hits |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `5c858f912a242f30` | `multipleChoice` | `Humanities/Social Science` | `True` | `True` | `both_correct` | `use_context` | `True` | `0` | `1` |
| `74ba73f3019cfdcf` | `exactMatch` | `Other` | `False` | `False` | `both_wrong` | `use_context` | `True` | `0` | `1` |
| `bd15ca4889ebb98f` | `exactMatch` | `Math` | `False` | `False` | `both_wrong` | `abstain_to_raw_prompt` | `False` | `0` | `0` |
| `e3a69b88d62cadcc` | `exactMatch` | `Math` | `True` | `True` | `both_correct` | `abstain_to_raw_prompt` | `False` | `0` | `0` |

## Choice Assessment

| problem hash | verdict | flags | rationale |
| --- | --- | --- | --- |
| `5c858f912a242f30` | `questionable_choice` | `weak_retrieval, weak_morphism_evidence, risky_low_score_context_injection, recursive_planning_only` | retrieval top score 0.152 is weak. structural decisions were weak: ['repair_under_specified']. context was injected despite low retrieval score. recursive runner opened applicability checks but did not execute child validation. |
| `74ba73f3019cfdcf` | `questionable_choice` | `weak_morphism_evidence, risky_exact_match_context_injection, recursive_planning_only, no_rescue` | retrieval top score 0.322 is usable. structural decisions were weak: ['repair_under_specified']. context was injected into an exact-match item without strong morphism/formal evidence. recursive runner opened applicability checks but did not execute child validation. |
| `bd15ca4889ebb98f` | `safe_but_unhelpful` | `borderline_retrieval, no_morphism_evidence, recursive_planning_only, no_rescue` | retrieval top score 0.238 is borderline. no formal or structural morphism evidence was found. recursive runner opened applicability checks but did not execute child validation. |
| `e3a69b88d62cadcc` | `reasonable_no_regression` | `borderline_retrieval, no_morphism_evidence, recursive_planning_only` | retrieval top score 0.207 is borderline. no formal or structural morphism evidence was found. recursive runner opened applicability checks but did not execute child validation. |

## Module Activation

| module | activated count |
| --- | ---: |
| `answer_format_verifier` | `4` |
| `answer_type_router` | `4` |
| `assumption_graph_retrieval` | `4` |
| `domain_router` | `4` |
| `multi_candidate_self_verifier` | `0` |
| `prompt_builder` | `4` |
| `recursive_assumption_runner` | `4` |
| `residual_writeback` | `0` |
| `structural_morphism_transfer` | `4` |
| `world_model_router` | `4` |

## Claim Boundary

This is a 4-item text-only HLE diagnostic, not a full HLE benchmark.  It verifies module activation and explains why this particular Assumption Agent wrapper tied raw gpt-5.5; it does not establish leaderboard-level HLE performance.
