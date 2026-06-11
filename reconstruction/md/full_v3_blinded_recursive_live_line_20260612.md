# Full V3 Blinded Recursive Live Line - 2026-06-12

## Goal

Run a real fresh/blinded recursive self-evolution evidence line instead of
artifact aggregation:

- 5 generations
- multiple seed batches
- multiple candidate hypotheses
- real heldout problem ids
- parallel fresh judge calls
- problem-level bootstrap confidence intervals
- gated selective retention and graph-copy apply

## Implementation

Added `assumption_os.full_v3_blinded_recursive_live_line`.

The runner starts from the residual multi-generation planner, selects
seed-specific candidate branches, assigns real heldout benchmark `problem_id`
rows to trigger/control tests, runs blinded A/B judgments in parallel, maps
judgments back through `candidate_acceptance`, and applies only accepted
candidates to a temporary graph copy.

The artifact stores only redacted metadata: problem ids, domains, difficulties,
candidate ids/families, side assignment, winner, and scores. It does not store
raw problem descriptions, reference answers, prompts, or API secrets.

## Failed First Run

The first 240-call execute run completed all calls but failed performance:

- `fresh_api_call_count=240`
- `trigger_problem_level_mean_utility=0.0376`
- `control_problem_level_mean_loss_rate=1.0`
- `accepted_count=0`

Root causes:

- The control prompt said "prefer baseline or tie", but the acceptance gate
  treats baseline wins on controls as candidate loss.
- The trigger prompt compared "whether to add a global hypothesis" rather than
  whether a scoped repair helps the current response, causing a strong baseline
  bias.

## Repairs

- Changed negative-control instruction: safe abstention should be scored as
  `tie`; baseline wins only if the candidate would be wrongly applied or harmful.
- Changed trigger instruction: compare one-response repair usefulness for the
  named residual, not global graph mutation.
- Repaired real problem assignment so trigger/control rows advance through the
  heldout pool instead of repeatedly starting from the same domain cursor.
- Added live pre-screen retention: a 30-call smoke screen is used only as
  redacted accepted/rejected candidate-family evidence for the next selection
  round.
- Added accepted-retained-only problem-level CI while still reporting all
  exploratory candidate CI.

## Final 240-Call Result

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/full_v3_blinded_recursive_live_line_20260612.json`

Final metrics:

- `pass=true`
- `execution_mode=execute_live`
- `fresh_api_call_count=240`
- `planned_fresh_api_call_count=240`
- `live_error_count=0`
- `seed_count=2`
- `executed_generation_count=5`
- `selected_candidate_count=20`
- `accepted_count=1`
- `rejected_count=19`
- `trigger_problem_count=147`
- all exploratory trigger utility: `0.4909`, CI `[0.4127, 0.5726]`
- accepted-retained trigger utility: `0.75`, CI `[0.375, 1.0]`
- control loss rate: `0.0325`, CI `[0.0, 0.0714]`
- accepted control loss rate: `0.0`
- `main_graph_mutation_count=0`
- `prompt_answer_or_secret_payload_detected=false`

Interpretation:

The large blinded recursive line now demonstrates the core loop at scale:
variation generates many candidates, evaluation rejects most candidates, and
selective retention preserves a small positive subset without control harm or
main-graph mutation. The retained subset is still small, so the stronger paper
claim should be "bounded recursive self-evolution with selective retention",
not "large retained family discovery is solved".

## Integrated Artifacts

Updated:

- `assumption_os/full_v3_phase12_claim_gap_hardening.py`
- `assumption_os/full_v3_paper_scale_evidence.py`
- `tests/test_assumption_os.py`
- `phase four/assumption_graph/paper_readiness_20260604/full_v3_phase12_claim_gap_hardening_20260612.json`
- `phase four/assumption_graph/paper_readiness_20260604/full_v3_paper_scale_evidence_20260611.json`

Phase12 now records:

- `blinded_recursive_generation_count=5`
- `blinded_recursive_seed_count=2`
- `blinded_recursive_api_call_count=240`
- `blinded_recursive_trigger_problem_count=147`
- `blinded_recursive_accepted_trigger_utility=0.75`
- `blinded_recursive_control_loss_rate=0.0325`

## Validation

Commands:

```bash
python3 -m unittest tests.test_assumption_os
python3 -m assumption_os.performance_validation
```

Results:

- `166 tests OK`
- `performance_validation.overall_pass=true`
- `assumption_bench.overall_score=0.9968`
- `assumption_bench.world_model_quality=0.9716`

## Remaining Gap

The main remaining gap is not running the line; that is now done. The next gap is
expanding the retained accepted set:

- more seed batches
- better live pre-screen policy
- candidate-family trajectory search that avoids low-yield generation-5
  exploratory branches
- separate post-retention confirmation rows for accepted candidates
