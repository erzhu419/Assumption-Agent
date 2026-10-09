"""Author edits buy one actual practice, with ordinary costs and source coverage."""

from copy import deepcopy
import json

import pytest

from scripts import run_python_method_reopening_study as runner
from scripts.python_method_edit_practice import EditPracticeMethodLearner


def learner_with_real_practice(tmp_path, case):
    sources = json.loads(runner.SOURCE_OUTCOMES.read_text())["sources"]
    plan = runner.study_protocol(sources)
    source = sources[plan["author_source_index"]]
    memory = runner.archive_memory(source)
    original = (runner.BRIDGE / "historical_bridge.py").read_text()
    memory["files"][runner.METHOD_PATH] = original
    learner = EditPracticeMethodLearner(None, tmp_path, knowledge=memory,
        counterexample_case=case, cases=plan["author_replicas"][0]["practice_cases"],
        lab_factory=runner.laboratory, config=runner.InquiryLLMConfig(**plan["author_limits"]),
        prepare_trial=runner.trial_preparer(source, plan))
    return learner, original


def change(learner, action, source):
    arguments = ({"path": runner.METHOD_PATH, "code": source} if action == "write" else
        {"code": "from pathlib import Path\n"
                 f"Path({runner.METHOD_PATH!r}).write_text({source!r})\n"})
    return learner._execute(action, arguments, learner.interface)


@pytest.mark.parametrize("action", ["write", "run"])
def test_real_edit_practices_once_but_identical_write_and_read_do_not_buy_again(tmp_path, action):
    learner, original = learner_with_real_practice(tmp_path, "case_5")
    edited = original + "\n# Same behavior; the known late failure must remain visible.\n"
    feedback = change(learner, action, edited)
    assert len(learner.practice_trials) == 1
    trial = learner.practice_trials[0]
    assert trial["method_source"] == edited
    assert trial["case"] == "case_5" and trial["status"] == "method_stopped"
    assert feedback["edit_practice"]["fresh_confirmation"]["passed"] is False
    assert feedback["current_source_coverage"]["confirmed_cases"] == []
    assert learner.published_source is learner.model is None
    for resource in ("observation_cost", "measurement_cost", "compute_cost"):
        assert learner.costs[resource] == trial["costs"][resource] > 0
    assert learner.files[runner.METHOD_PATH] == edited
    paid = deepcopy(learner.costs)
    identical = change(learner, "write", edited)
    read = learner._execute("run", {"code": "from pathlib import Path\n"
        f"print(len(Path({runner.METHOD_PATH!r}).read_text()))\n"}, learner.interface)
    assert "edit_practice" not in identical and "edit_practice" not in read
    assert len(learner.practice_trials) == 1
    for resource in ("observation_cost", "measurement_cost", "compute_cost"):
        assert learner.costs[resource] == paid[resource]
    assert learner.costs["model_calls"] == 0


def test_new_source_loses_old_coverage_and_restoring_confirmed_source_does_not_repay(tmp_path):
    learner, original = learner_with_real_practice(tmp_path, "case_0")
    successful = original + "\n# Exercise the real unchanged-prior practice.\n"
    first = change(learner, "write", successful)
    assert first["edit_practice"]["fresh_confirmation"]["passed"] is True
    assert first["current_source_coverage"]["confirmed_cases"] == ["case_0"]
    assert len(first["current_source_coverage"]["missing_cases"]) == 5
    assert learner.published_source is learner.model is None

    failed = "def select_actions(public_spec, records, memory):\n    return {'stop': 'fixture stops before confirmation'}\n"
    second = change(learner, "write", failed)
    assert len(learner.practice_trials) == 2
    assert second["edit_practice"]["status"] == "method_stopped"
    assert second["current_source_coverage"]["confirmed_cases"] == []
    assert len(second["current_source_coverage"]["missing_cases"]) == 6
    assert learner.published_source is learner.model is None
    paid = deepcopy(learner.costs)

    restored = change(learner, "run", successful)
    assert "edit_practice" not in restored
    assert restored["current_source_coverage"]["confirmed_cases"] == ["case_0"]
    assert len(learner.practice_trials) == 2
    for resource in ("observation_cost", "measurement_cost", "compute_cost"):
        assert learner.costs[resource] == paid[resource]
        assert learner.costs[resource] == sum(trial["costs"][resource] for trial in learner.practice_trials)
    assert learner.published_source is learner.model is None
