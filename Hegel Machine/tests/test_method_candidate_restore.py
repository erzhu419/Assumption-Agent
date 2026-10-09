"""Keep experimentally supported drafts recoverable after later edits fail."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from test_python_method_edit_practice import change, learner_with_real_practice
from scripts.run_python_method_reopening_study import METHOD_PATH


def candidates(learner):
    saved = json.loads((learner.workspace / "method_candidates.json").read_text())
    assert saved == json.loads(learner.files["method_candidates.json"])
    return saved


def restore(learner, trial):
    return learner._execute("run", {"operator": "restore_method", "trial": trial}, learner.interface)


def test_paid_success_survives_a_failed_edit_and_restores_only_its_own_coverage(tmp_path):
    learner, original = learner_with_real_practice(tmp_path, "case_0")
    successful = original + "\n# Candidate with actual unchanged and changed response practices.\n"
    first = change(learner, "write", successful)
    second = learner._execute("run", {"operator": "practice", "case": "case_1"}, learner.interface)
    assert first["edit_practice"]["status"] == second["status"] == "completed"
    failed = "def select_actions(public_spec, records, memory):\n    return {'stop': 'unconfirmed replacement'}\n"
    assert change(learner, "write", failed)["edit_practice"]["status"] == "method_stopped"
    book = candidates(learner)
    assert len(book) == 2
    supported = next(entry for entry in book if entry["confirmed_cases"])
    rejected = next(entry for entry in book if entry["current"])
    assert supported["confirmed_cases"] == ["case_0", "case_1"]
    assert supported["failed_cases"] == [] and len(supported["trials"]) == 2
    assert not supported["current"]
    assert rejected["confirmed_cases"] == [] and rejected["failed_cases"] == ["case_0"]
    assert (learner.workspace / supported["source_path"]).read_text() == successful
    assert (learner.workspace / rejected["source_path"]).read_text() == failed

    paid = deepcopy(learner.costs)
    # Both trial directories point to the same measured source candidate.
    trial = supported["trials"][1]["directory"]
    recovered = restore(learner, trial)
    assert recovered["restored_method"] == trial
    assert "edit_practice" not in recovered
    assert recovered["current_source_coverage"]["confirmed_cases"] == ["case_0", "case_1"]
    assert recovered["current_source_coverage"]["missing_cases"] == ["case_2", "case_3", "case_4", "case_5"]
    assert learner.files[METHOD_PATH] == (learner.workspace / METHOD_PATH).read_text() == successful
    assert learner.costs == paid and len(learner.practice_trials) == 3
    assert learner.published_source is learner.model is None
    assert next(entry for entry in candidates(learner) if entry["current"])["trial"] == supported["trial"]

    # Retrieval does not grant the four missing confirmations or bypass finish.
    learner.config = replace(learner.config, observation_budget=learner.costs["observation_cost"])
    submitted = learner._execute("finish", {"path": METHOD_PATH}, learner.interface)
    assert "published" not in submitted
    assert submitted["unconfirmed_cases"] == ["case_2", "case_3", "case_4", "case_5"]
    assert learner.published_source is learner.model is None
    assert learner.costs == paid


def test_real_failed_author_can_retrieve_its_successful_draft_after_resume(tmp_path):
    origin = Path(__file__).resolve().parents[1] / (
        "artifacts/python_method_edit_practice_luna_20261006/author_2/learning/result.json")
    saved = origin.read_bytes()
    previous = json.loads(saved)
    assert previous["status"] == "token_budget_exhausted"
    successful = previous["practice_trials"][3]
    failed = previous["practice_trials"][4]
    assert successful["status"] == "completed" and failed["status"] == "method_stopped"
    assert successful["method_source"] != failed["method_source"]

    learner, _ = learner_with_real_practice(tmp_path / "resumed", "case_5")
    learner.restore_practice(previous)
    context_text = (learner.workspace / "resume_context.json").read_text()
    assert len(context_text) < 8000
    context = json.loads(context_text)
    assert context["previous_status"] == "token_budget_exhausted"
    assert context["current_source_coverage"]["confirmed_cases"] == []
    assert len(context["previous_practice"]) == 5
    old_action = next(event for event in reversed(previous["events"])
                      if event["action"] != "provider_error")
    assert len(old_action["feedback"]["stdout"]) > 8000
    assert len(context["last_action"]["feedback"]["stdout"]) < 2000
    full_event = learner.workspace / context["last_action"]["full_action_file"]
    assert json.loads(full_event.read_text()) == old_action
    # The actual ordinary Python reader previously returned only the end of
    # this JSON, hiding its status and coverage while the author kept re-reading.
    read = learner._execute("run", {"code": "from pathlib import Path\n"
        "print(Path('resume_context.json').read_text())\n"}, learner.interface)
    assert json.loads(read["stdout"]) == context
    book = candidates(learner)
    assert len(book) == len({trial["method_source"] for trial in previous["practice_trials"]}) == 5
    assert next(entry for entry in book if entry["current"])["confirmed_cases"] == []
    entry = next(entry for entry in book if entry["trial"] == successful["directory"])
    assert entry["confirmed_cases"] == ["case_5"]
    assert (learner.workspace / entry["source_path"]).read_text() == successful["method_source"]
    paid = deepcopy(learner.costs)
    outcome = restore(learner, entry["trial"])
    assert "edit_practice" not in outcome
    assert outcome["current_source_coverage"]["confirmed_cases"] == ["case_5"]
    assert outcome["current_source_coverage"]["missing_cases"] == [f"case_{i}" for i in range(5)]
    assert learner.costs == paid and len(learner.practice_trials) == 5
    assert learner.published_source is learner.model is None

    # The catalogue is rebuilt from retained trials, not an in-memory-only selection.
    resumed, _ = learner_with_real_practice(tmp_path / "resumed_again", "case_5")
    resumed.restore_practice(learner.result("model_call_budget_exhausted"))
    assert next(entry for entry in candidates(resumed) if entry["current"])["confirmed_cases"] == ["case_5"]
    before = deepcopy(resumed.costs)
    assert restore(resumed, successful["directory"])["current_source_coverage"]["confirmed_cases"] == ["case_5"]
    assert resumed.costs == before and resumed.published_source is None
    assert origin.read_bytes() == saved
