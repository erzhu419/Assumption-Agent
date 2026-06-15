# HLE Parallel Child + Evidence Bridge Analysis - 2026-06-15

## Change

This iteration targets the two remaining HLE bottlenecks from the previous audit:

- child attempts were serial and expensive;
- exactMatch repair lacked concrete answer-bearing evidence.

Implemented changes:

- `agent_child_mode=parallel_quorum`:
  - first two child attempts run concurrently;
  - if they form a safe two-vote majority, remaining child attempts are skipped;
  - exactMatch suspicious single-letter answers cannot trigger early stop.
- `agent_child_timeout`:
  - per-child timeout is recorded in every child start event.
- transient HLE evidence bridge:
  - exactMatch suspicious-collapse cases query Wikipedia search transiently;
  - artifacts persist only query/source hashes, result counts, and evidence char counts;
  - raw HLE question, raw evidence snippets, and model predictions are not persisted.
- evidence-grounded child:
  - if all exactMatch child attempts collapse to suspicious answers, an `evidence_grounded_answer` child is added before final selection.
- exactMatch selection rule:
  - non-suspicious exact candidates are preferred over suspicious single-letter majorities.

## Performance validation

Same 4-problem HLE slice, `gpt-5.5`, seed offset 13:

| run | agent correct | agent accuracy | MCQ accuracy | exact accuracy | agent latency sum | underlying calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed child verifier | 3/4 | 0.75 | 1.00 | 0.00 | 534.924s | 19 |
| serial early-stop + exact repair | 3/4 | 0.75 | 1.00 | 0.00 | 347.517s | 14 |
| parallel quorum + evidence bridge | 3/4 | 0.75 | 1.00 | 0.00 | 245.189s | 15 |

The new path keeps the same 3/4 score on this slice while reducing agent wall-clock latency by:

- 54.2% vs fixed child verifier;
- 29.4% vs serial early-stop + exact repair.

Focused exactMatch validation, seed offset 17:

| run | raw accuracy | agent accuracy | evidence bridge | evidence child | outcome |
| --- | ---: | ---: | --- | --- | --- |
| parallel evidence exact focus | 0.00 | 0.00 | activated, 5 sources, 1162 chars | activated | all child attempts including evidence-grounded answer still collapsed to the same suspicious answer hash |

## Interpretation

The parallel child executor is a real performance improvement. It reduces wall-clock cost without losing the MCQ gains from recursive verification.

The evidence bridge is now mechanically active and audited, but it did not solve the focused exactMatch item. The failure mode is informative:

- Wikipedia search returned evidence rows.
- The evidence-grounded child ran.
- The model still produced the same suspicious answer hash.

So the remaining exactMatch weakness is not a missing runner stage. It is insufficient answer-bearing retrieval or poor query targeting for this HLE item.

## Verification

- `python3 -m py_compile assumption_os/hle_smoke_eval.py`
- `python3 -m unittest tests.test_hle_smoke_eval` -> 18 tests OK

## Next

The next useful improvement is not more prompt repair. It is a stronger evidence bridge:

1. generate multiple hashed evidence queries from the question with a dedicated query planner;
2. retrieve from more sources than Wikipedia, or use a controlled local corpus;
3. rank evidence snippets before the evidence-grounded child;
4. log retrieval quality with only hashes/counts.
