# HLE Assumption Agent Trace Audit

- pass: `True`
- sample count: `8`
- raw accuracy: `0.25`
- assumption_agent accuracy: `0.5`
- agent minus raw: `0.25`
- transition counts: `{'both_wrong': 4, 'agent_rescue': 2, 'both_correct': 2}`
- context decisions: `{'abstain_to_raw_prompt': 7, 'use_context': 1}`
- failed gates: `[]`

## Diagnosis

- The full Assumption Agent wrapper beat raw gpt-5.5 on this diagnostic slice.
- assumption_graph_retrieval activated on 8/8 agent rows.
- structural_morphism_transfer activated on 8/8 agent rows.
- world_model_router activated on 8/8 agent rows.
- recursive_assumption_runner activated on 8/8 agent rows.
- There were no agent_regression transitions; the gated wrapper did not harm this slice.
- World model injected context on 1 rows.
- World model abstained to raw prompt on 7 rows.
- Structural morphism hits were weak/repair-level on 2 rows, so morphism transfer was active but not strong answer-bearing evidence.
- Formal mapping produced zero hits on this HLE slice; the agent relied on generic graph/structural context.
- Recursive runner produced applicability frames, but did not execute child answer-generation/judge loops inside this single-call HLE wrapper.

## Problem-Level Trace

| problem hash | answer type | category | raw | agent | transition | decision | context | formal hits | structural hits |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `65a5d5e072234b72` | `exactMatch` | `Math` | `False` | `False` | `both_wrong` | `abstain_to_raw_prompt` | `False` | `0` | `0` |
| `6bb535bfdd0d333f` | `exactMatch` | `Physics` | `False` | `False` | `both_wrong` | `abstain_to_raw_prompt` | `False` | `0` | `1` |
| `7bc43696807ccd83` | `exactMatch` | `Math` | `False` | `True` | `agent_rescue` | `abstain_to_raw_prompt` | `False` | `0` | `0` |
| `9c68db92af55c1df` | `multipleChoice` | `Computer Science/AI` | `True` | `True` | `both_correct` | `use_context` | `True` | `0` | `0` |
| `b45e436bc3f6c7b9` | `exactMatch` | `Computer Science/AI` | `True` | `True` | `both_correct` | `abstain_to_raw_prompt` | `False` | `0` | `0` |
| `d32e7460eb8a3b50` | `exactMatch` | `Humanities/Social Science` | `False` | `True` | `agent_rescue` | `abstain_to_raw_prompt` | `False` | `0` | `0` |
| `de4812f3eea217b6` | `exactMatch` | `Physics` | `False` | `False` | `both_wrong` | `abstain_to_raw_prompt` | `False` | `0` | `1` |
| `edc6b41df4b483f8` | `exactMatch` | `Math` | `False` | `False` | `both_wrong` | `abstain_to_raw_prompt` | `False` | `0` | `0` |

## Rescue Attribution

| problem hash | transition | attribution |
| --- | --- | --- |
| `65a5d5e072234b72` | `both_wrong` | `not_rescue` |
| `6bb535bfdd0d333f` | `both_wrong` | `not_rescue` |
| `7bc43696807ccd83` | `agent_rescue` | `not_module_attributable_raw_prompt_repeat_variance` |
| `9c68db92af55c1df` | `both_correct` | `not_rescue` |
| `b45e436bc3f6c7b9` | `both_correct` | `not_rescue` |
| `d32e7460eb8a3b50` | `agent_rescue` | `not_module_attributable_raw_prompt_repeat_variance` |
| `de4812f3eea217b6` | `both_wrong` | `not_rescue` |
| `edc6b41df4b483f8` | `both_wrong` | `not_rescue` |

## Choice Assessment

| problem hash | verdict | flags | rationale |
| --- | --- | --- | --- |
| `65a5d5e072234b72` | `safe_but_unhelpful` | `no_morphism_evidence, recursive_planning_only, no_rescue` | retrieval top score 0.266 is usable. no formal or structural morphism evidence was found. recursive runner opened applicability checks but did not execute child validation. |
| `6bb535bfdd0d333f` | `safe_but_unhelpful` | `borderline_retrieval, weak_morphism_evidence, recursive_planning_only, no_rescue` | retrieval top score 0.191 is borderline. structural decisions were weak: ['repair_under_specified']. recursive runner opened applicability checks but did not execute child validation. |
| `7bc43696807ccd83` | `lucky_or_repeat_variance` | `borderline_retrieval, no_morphism_evidence, recursive_planning_only, rescue, abstain_rescue_not_module_attributable` | retrieval top score 0.204 is borderline. no formal or structural morphism evidence was found. recursive runner opened applicability checks but did not execute child validation. agent rescued while abstaining to the raw prompt, so this is likely repeat-call variance. |
| `9c68db92af55c1df` | `reasonable_no_regression` | `no_morphism_evidence, recursive_planning_only` | retrieval top score 0.644 is usable. no formal or structural morphism evidence was found. recursive runner opened applicability checks but did not execute child validation. |
| `b45e436bc3f6c7b9` | `reasonable_no_regression` | `no_morphism_evidence, recursive_planning_only` | retrieval top score 0.658 is usable. no formal or structural morphism evidence was found. recursive runner opened applicability checks but did not execute child validation. |
| `d32e7460eb8a3b50` | `lucky_or_repeat_variance` | `no_morphism_evidence, recursive_planning_only, rescue, abstain_rescue_not_module_attributable` | retrieval top score 0.308 is usable. no formal or structural morphism evidence was found. recursive runner opened applicability checks but did not execute child validation. agent rescued while abstaining to the raw prompt, so this is likely repeat-call variance. |
| `de4812f3eea217b6` | `safe_but_unhelpful` | `weak_morphism_evidence, recursive_planning_only, no_rescue` | retrieval top score 0.262 is usable. structural decisions were weak: ['repair_under_specified']. recursive runner opened applicability checks but did not execute child validation. |
| `edc6b41df4b483f8` | `safe_but_unhelpful` | `no_morphism_evidence, recursive_planning_only, no_rescue` | retrieval top score 0.243 is usable. no formal or structural morphism evidence was found. recursive runner opened applicability checks but did not execute child validation. |

## Module Activation

| module | activated count |
| --- | ---: |
| `answer_format_verifier` | `8` |
| `answer_type_router` | `8` |
| `assumption_graph_retrieval` | `8` |
| `domain_router` | `8` |
| `multi_candidate_self_verifier` | `0` |
| `prompt_builder` | `8` |
| `recursive_assumption_runner` | `8` |
| `residual_writeback` | `0` |
| `structural_morphism_transfer` | `8` |
| `world_model_router` | `8` |

## Claim Boundary

This is a 8-item text-only HLE diagnostic, not a full HLE benchmark.  It verifies module activation and audits whether the wrapper's choices are attributable to graph/morphism/world-model/recursive modules; it does not establish leaderboard-level HLE performance.
