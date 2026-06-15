# HLE Recursive Verify Log Analysis - 2026-06-15

## What changed

The HLE Assumption Agent variant now has a real recursive verification path instead of only opening
`verify_applicability` frames.

- `assumption_agent_recursive_verify` runs graph retrieval, morphism search, world-model routing, and the recursive runner.
- It then executes child answer attempts and selects by normalized majority or verifier choice.
- It records child/verifier logs with hashes, prompt kinds, latency, and stage status only. It does not persist raw HLE questions, gold answers, or prediction text.
- It uses two-vote majority early stop to avoid fixed four-way serial cost.
- It triggers strict exactMatch repair for suspicious single-letter/empty exact answers.

## Performance validation

Artifacts:

- `phase four/assumption_graph/paper_readiness_20260604/hle_text_smoke_gpt55_raw_vs_recursive_verify_n4_seed13_20260615.json`
- `phase four/assumption_graph/paper_readiness_20260604/hle_text_smoke_gpt55_recursive_verify_exact_repair_n4_seed13_20260615.json`
- `phase four/assumption_graph/paper_readiness_20260604/hle_text_smoke_gpt55_exact_repair_graph_context_seed17_20260615.json`

Same 4-problem HLE slice, `gpt-5.5`:

| run | raw accuracy | agent accuracy | agent MCQ accuracy | agent exact accuracy | underlying calls | missing expected modules |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed child verifier | 0.50 | 0.75 | 1.00 | 0.00 | 19 | none |
| early-stop + exact repair | 0.50 | 0.75 | 1.00 | 0.00 | 14 | none |

Focused exactMatch rerun, `seed_offset=17`, `sample_size=1`:

| run | raw accuracy | agent accuracy | repair activated | repair context used | result |
| --- | ---: | ---: | --- | --- | --- |
| exact repair + graph context | 0.00 | 0.00 | yes | yes, 3119 chars | repair changed the candidate and made it non-suspicious, but still did not match gold |

Unit validation:

- `python3 -m py_compile assumption_os/hle_smoke_eval.py`
- `python3 -m unittest tests.test_hle_smoke_eval` -> 16 tests OK

## Module behavior

Latest 4-problem run:

- graph retrieval: 4/4 activated.
- structural morphism transfer: 4/4 activated, 3 structural hits total, 0 strong supported transfers.
- formal mapping: 0 formal hits.
- world model router: 3 `use_context`, 1 `abstain_to_raw_prompt`.
- recursive runner: 4/4 activated.
- recursive child validation: 4/4 activated.
- multi-candidate self verifier: 4/4 activated.
- exact answer repair: 1/4 activated.
- expected missing modules: none.

## Diagnosis

The recursive verification mechanism now works on HLE: it executes real child calls and produces final module traces with child validation and self-verifier activated. It also improves the tested MCQ slice relative to raw in two same-slice runs.

The remaining weakness is exactMatch answer-bearing evidence. The focused exactMatch item failed for both raw and agent. The repair stage changed a suspicious single-letter candidate into a non-suspicious candidate when prompted to output option text, but it still missed gold. That points to a knowledge/evidence gap rather than a pure answer-format bug.

The graph retrieval context is still mostly assumption/policy context, not HLE answer-bearing evidence. For HLE exactMatch, the next meaningful improvement is not another prompt wrapper. It is a retrieval/evidence layer that can surface concrete entity, option text, or factual support while preserving the no-raw-HLE-persistence constraint.

## Next fixes

1. Add parallel child execution or per-child timeout for HLE recursive verification. Early-stop reduced calls from 19 to 14 on the same 4-problem slice, but wall time is still dominated by slow child calls.
2. Build an HLE evidence bridge that stores only hashes/metadata in artifacts but can use allowed transient evidence during model calls.
3. Add an exactMatch-specific evidence-needed gate: if all child attempts collapse to a suspicious exact answer, route to evidence retrieval instead of only prompt repair.
4. Keep the current claim boundary: this supports gated recursive verification on a small text-only HLE slice, not a full HLE benchmark claim.
