# LLM Framework Candidate Experiment

- pass: `True`
- live LLM API executed: `False`
- LLM-contract candidates: `10`
- real residual sources: `10`
- top2 validation count: `2`
- top2 min old-success preservation: `0.97`
- top2 min residual explanation: `0.92`
- accepted/candidate validations: `2`
- negative-control validations: `1`

## Top Candidates

| Candidate | Trajectory | Source residual | Validation decision | Growth score |
| --- | --- | --- | --- | --- |
| `r3_candidate_219_66ecf184` | `framework_combination_branch` | `trace_residual_cluster_01_4ec319d4` | `candidate_framework` | `0.8179` |
| `r3_candidate_218_a2666e00` | `parent_generalization_branch` | `trace_residual_cluster_01_4ec319d4` | `candidate_framework` | `0.8169` |

## Claim Boundaries

- `fresh_live_llm_candidate_generation_completed`: blocked=`True`; Blocked unless --execute-live succeeds with API credentials in environment; default artifact is a deterministic LLM-contract replay over real residual clusters.
- `unfiltered_llm_framework_generator_is_reliable`: blocked=`True`; The experiment validates top candidates and a negative control; it does not promote all generated candidates.
- `llm_candidates_can_skip_conservative_gate`: blocked=`True`; Every candidate remains subject to old-success, residual, limiting-case, unseen, and control obligations.
