# Existing budget and tail test results

Saved from the completed tool outputs; tests were not rerun to create this note. No JUnit file was requested for these commands.

Candidate command:
```sh
PYTHONPATH=src python3 -m pytest -q tests/test_cognition_complete_confirmation_budget.py tests/test_cognition_unified_tail_calibration.py
```
Result: 12 cases, 10 passed, 2 failed. All 5 new complete-confirmation-budget cases passed; the existing tail file had 5 passed, 2 failed.

The 2 failures were:
- `test_unknown_noisy_task_calibrates_paid_validation_before_new_confirmation`: line 100 expected the most recent tail proposal to precede the comparison; actual observation cost 240, expected 72 + 48. The compared proposal is a risk-actions scale revision.
- `test_supported_short_trial_compares_both_frozen_tails_on_its_new_confirmation`: line 155 expected `validate_short_learned_trial`; actual previous acquisition was `compare_on_new_trajectory`.

Frozen 0.156 reproduction command:
```sh
PYTHONPATH=artifacts/mechanism_noise_research_20261001/runtime/src python3 -m pytest -q tests/test_cognition_unified_tail_calibration.py::test_unknown_noisy_task_calibrates_paid_validation_before_new_confirmation tests/test_cognition_unified_tail_calibration.py::test_supported_short_trial_compares_both_frozen_tails_on_its_new_confirmation
```
Result: the same 2 failures, same assertion values. Stack traces confirm the frozen runtime path `artifacts/mechanism_noise_research_20261001/runtime/src/hegel_machine/cognition/unified_research.py:4481`. Tests and assertions were retained unchanged.

An earlier run of the first 4 new budget cases also passed; this is duplicate verification, not 4 additional distinct passing tests.
