"""Historical physical tasks and matched interfaces for the live-method pilot.

These checks use no external model provider; they test the experiment rather than claim
evidence of autonomous research.
"""

from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.temporal_inquiry import _control_observation
from hegel_machine.cognition.temporal_method_lab import TemporalMethodLab
from hegel_machine.cognition.temporal_operator_agent import INSTRUMENT_METHODS_PATH
from scripts import run_temporal_legacy_method_llm as runner


def task(case="window", stage=0):
    return {"case": case, "stage": stage, "seed": 151 if stage == 0 else 373,
            "center": .21 if stage == 0 else -.31,
            "half_width": 1e-4, "response_gain": 2.}


def config():
    return InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
        compute_budget=300000, execution_seconds=60.)


@pytest.mark.parametrize("case", ["window", "event"])
@pytest.mark.parametrize("stage", [0, 4])
def test_terminal_preserves_historical_activation_physics_without_meter_leakage(case, stage):
    settings = task(case, stage)
    lab = runner.LegacyMethodInquiryLab(**settings)
    original = TemporalMethodLab(**settings)
    before = deepcopy(lab.costs)
    actual = lab.evaluation_rows(seed=811, length=128)
    expected = [_control_observation(row) for row in original.evaluation_rows(
        seed=811, trajectories=4, length=32)]
    assert actual == expected
    assert lab.costs == before
    assert len(actual) == 128
    assert all(set(row.features) == {"u"} for row in actual)
    assert all(set(point) == {"u"} for row in actual for point in row.history)
    assert any(row.target > 0. for row in actual)
    if case == "event":
        assert any(row.target < 0. for row in actual)
    # Random-only confirmation can miss this task's tiny activation window.
    # Terminal evaluation must retain the historical activation/exit probes.
    assert sum(row.target ** 2 for row in actual) / len(actual) > 1e-10


@pytest.mark.parametrize("case", ["window", "event"])
def test_preparation_charges_calibration_without_choosing_response_or_learning_a_method(tmp_path, case):
    agent, lab = runner.prepare_episode(tmp_path / case, task(case), config())
    assert agent.costs["observation_cost"] == lab.costs["observation_cost"] == 72
    assert agent.costs["measurement_cost"] == lab.costs["measurement_cost"] == 216
    assert agent.costs["compute_cost"] > 0
    assert agent.costs["model_calls"] == 0
    assert len(agent.observations) == 3
    assert all(len(record["rows"]) == len(record["measurements"]) == 24
               for record in agent.observations)
    assert all(row["target"] == 0. for record in agent.observations for row in record["rows"])
    assert agent.model is None and agent.confirmations == []
    assert not (agent.workspace / INSTRUMENT_METHODS_PATH).exists()
    assert not (agent.workspace / "response_experiments").exists()
    assert all(item["operator"] not in {"response_disagreement", "assess_response"}
               for item in agent.operator_results)


@pytest.mark.parametrize("case", ["window", "event"])
def test_supplied_candidates_are_executable_exports_of_the_same_measured_laws(tmp_path, case):
    settings = task(case)
    agent, lab = runner.prepare_episode(tmp_path / case, settings, config())
    hypotheses = json.loads((agent.workspace / "hypotheses.json").read_text())
    assert len(hypotheses) == (3 if case == "window" else 5)
    # The historical endpoint-targeted evaluator supplies active rows as well as
    # random rows. All-zero preparation data would conceal mismatched exports.
    measured = TemporalMethodLab(**settings).evaluation_rows(seed=811, trajectories=4, length=32)
    controlled = [_control_observation(row) for row in measured]
    losses = []
    for hypothesis in hypotheses:
        assert hypothesis["hypothesis_id"] == hypothesis["path"]
        assert hypothesis["path"].startswith("models/")
        namespace = {}
        source = (agent.workspace / hypothesis["path"]).read_text()
        exec(compile(source, hypothesis["path"], "exec"), namespace)
        program = Program.from_dict(hypothesis["program"])
        expected = [program.predict(row) for row in measured]
        actual = [namespace["predict"](asdict(row)) for row in controlled]
        assert actual == pytest.approx(expected, abs=1e-10)
        losses.append(sum((guess - row.target) ** 2 for guess, row in zip(actual, controlled)))
    assert sum(loss < 1e-10 for loss in losses) == 1


def test_transferred_method_book_is_the_only_initial_difference_between_matched_targets(tmp_path):
    source, source_lab = runner.prepare_episode(tmp_path / "source", task("event"), config())
    proposal = source._execute("run", {"operator": "response_disagreement",
        "hypotheses": "hypotheses.json", "instrument": "instrument.json"}, source_lab)
    acquired = source._execute("query", {"actions": [proposal["action"]],
        "after_observation": [{"operator": "assess_response", "experiment": proposal["experiment"],
                               "records": "acquired"}]}, source_lab)
    assert acquired["after_observation_results"][0]["status"] == "distinguished"
    book = json.loads((source.workspace / INSTRUMENT_METHODS_PATH).read_text())
    assert book and book[0]["support"]
    lesson = "Compare current candidates using newly purchased evidence."
    retained, retained_lab = runner.prepare_episode(tmp_path / "retained", task("event", 4),
        config(), book=book, lesson=lesson)
    deleted, deleted_lab = runner.prepare_episode(tmp_path / "deleted", task("event", 4),
        config(), lesson=lesson)
    assert retained.arm == deleted.arm == "code_react"
    assert retained.common_interface == deleted.common_interface
    assert retained.operator_interface == deleted.operator_interface
    assert "Code/ReAct receives no previous-task knowledge" not in retained.common_interface
    assert retained_lab.public_spec() == deleted_lab.public_spec()
    assert retained.observations == deleted.observations
    assert retained.knowledge.get("lessons") == deleted.knowledge.get("lessons")
    assert set(retained.files) - set(deleted.files) == {INSTRUMENT_METHODS_PATH}
    assert set(deleted.files) - set(retained.files) == set()
    for path in deleted.files:
        assert retained.files[path] == deleted.files[path]
    assert json.loads((retained.workspace / INSTRUMENT_METHODS_PATH).read_text()) == book
    assert not (deleted.workspace / INSTRUMENT_METHODS_PATH).exists()
    assert retained.costs["observation_cost"] == deleted.costs["observation_cost"] == 72
    assert retained.costs["measurement_cost"] == deleted.costs["measurement_cost"] == 216


@pytest.mark.parametrize("confirmed,mse,correct,false_confirmation", [
    (True, .0625, False, True),
    (True, 0., True, False),
    (False, 0., False, False),
])
def test_summary_distinguishes_random_confirmation_from_terminal_prediction(
        confirmed, mse, correct, false_confirmation):
    result = {"status": "completed" if confirmed else "budget_exhausted",
              "model": "models/candidate_00.py" if confirmed else None,
              "fresh_confirmation": {"passed": confirmed}, "events": [], "costs": {}}
    terminal = {"returncode": 0, "mse": mse, "rows": 128,
                "nonzero_targets": 2, "negative_targets": 0}
    summary = runner.episode_summary(result, terminal)
    assert summary["fresh_confirmed"] is confirmed
    assert summary["confirmed_and_correct"] is correct
    assert summary["false_confirmation"] is false_confirmation
    assert not summary["identified"]
    assert not summary["selected_identified_candidate"]


def test_summary_records_measured_identification_separately_from_final_candidate_selection():
    result = {"status": "completed", "model": "models/candidate_00.py",
        "fresh_confirmation": {"passed": True}, "costs": {}, "events": [
            {"feedback": {"operator": "response_disagreement"}},
            {"feedback": {"after_observation_results": [{"operator": "assess_response",
                "status": "distinguished", "hypotheses": [
                    {"hypothesis_id": "models/candidate_00.py", "status": "counterexample", "rows": 8},
                    {"hypothesis_id": "models/candidate_01.py", "status": "supported_on_probe", "rows": 8}],
                "method_feedback": {"action": "use_program_experiment_method", "cross_task": True}}]}}]}
    summary = runner.episode_summary(result, {"returncode": 0, "mse": .125})
    assert summary["identified"]
    assert not summary["selected_identified_candidate"]
    assert summary["false_confirmation"]
    assert summary["response_plans"] == summary["measured_response_experiments"] == 1
    assert summary["method_events"] == [{"action": "use_program_experiment_method", "cross_task": True}]


def test_historical_mode_uses_existing_measured_books_in_four_matched_targets(tmp_path):
    historical = runner.ROOT / (
        "artifacts/temporal_method_reuse_20260913/development_v1/runs/151_temporal_methods.json")
    original = json.loads(historical.read_text())
    books = {stream["case"]: stream["knowledge"]["program_experiment_methods"]
             for stream in original["streams"]}
    output = tmp_path / "historical_transfer"
    requests = {}

    class ExistingCandidateProvider:
        def __init__(self, workspace):
            self.key = tuple(workspace.relative_to(output).parts)

        def next_action(self, request):
            requests.setdefault(self.key, deepcopy(request))
            # A stubbed decision exercises orchestration and real confirmation,
            # not the model's ability to select the correct historical law.
            return {"action": "finish", "arguments": {"model": "models/candidate_00.py"},
                    "usage": {"total_tokens": 0, "input_tokens": 0, "cached_input_tokens": 0}}

    summary = runner.run_study(output, provider_factory=ExistingCandidateProvider,
                              historical_run=historical)
    expected = {(case, role) for case in ("window", "event") for role in ("retained", "removed")}
    assert set(requests) == expected
    assert {(episode["case"], episode["role"]) for episode in summary["episodes"]} == expected
    assert summary["registered_episodes"] == summary["attempted_episodes"] == 4
    assert json.loads((output / "historical_run.json").read_text()) == original
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["historical_training_costs"]["observation_cost"] == 648
    assert protocol["historical_training_costs"]["measurement_cost"] == 1944
    assert protocol["historical_training_costs"]["compute_cost"] == 36607
    for metric in ("observation_cost", "measurement_cost", "compute_cost"):
        assert summary["costs"][metric] == sum(episode["costs"][metric] for episode in summary["episodes"])
    for case in ("window", "event"):
        assert not (output / case / "source").exists()
        assert json.loads((output / case / "frozen_methods.json").read_text()) == books[case]
        assert any(method["counterexamples"] for method in books[case])
        retained, removed = requests[(case, "retained")], requests[(case, "removed")]
        for key in ("task_interface", "knowledge_condition", "capabilities", "limits",
                    "observations", "laboratory_observations"):
            assert retained[key] == removed[key]
        assert retained["session_knowledge"]["lessons"] == removed["session_knowledge"]["lessons"] == []
        assert removed["session_knowledge"]["files"] == {}
        assert set(retained["session_knowledge"]["files"]) == {INSTRUMENT_METHODS_PATH}
        assert json.loads(retained["session_knowledge"]["files"][INSTRUMENT_METHODS_PATH]) == books[case]
        assert set(retained["files"]) - set(removed["files"]) == {INSTRUMENT_METHODS_PATH}
        for path in removed["files"]:
            assert retained["files"][path] == removed["files"][path]

