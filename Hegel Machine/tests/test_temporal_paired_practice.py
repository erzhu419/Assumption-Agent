"""Scripted paired-practice orchestration; not evidence of method learning."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import temporal_paired_practice as practice
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts import run_temporal_paired_method as runner


@pytest.mark.parametrize("unpublished_seed", [False, True])
def test_final_source_requires_both_fresh_cases_and_restores_each_prior(tmp_path, monkeypatch, unpublished_seed):
    def donor(instrument, predictor, calibration):
        return {"status": "completed", "model": predictor,
                "fresh_confirmation": {"passed": True},
                "outgoing_knowledge": {
                    "files": {instrument: calibration, predictor: "def predict(row): return 0.\n"},
                    "manifest": {"hypotheses": [{"id": "prior", "path": predictor}],
                                 "research_programs": []}, "lessons": []}}

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    seed = tmp_path / "seed.json"
    first.write_text(json.dumps({"tasks": [{"status": "wrong_source"},
        donor("shared/meter.json", "shared/prior.py", "first paid calibration")]}))
    second.write_text(json.dumps({"tasks": [donor("shared/current_instrument.json",
        "shared/second_prior.py", "second paid calibration")]}))
    source_a = "def select_actions(spec, records, memory): return {'actions': []}\n"
    source_b = "def select_actions(spec, records, memory): return {'model': memory['research_roles']['predictor']}\n"
    candidate = tmp_path / "candidate.py"
    candidate.write_text(source_b)
    seed.write_text(json.dumps({"status": "completed", "method_path": METHOD_PATH,
        "method_source": source_a, "method_entry": {"id": "seed", "path": METHOD_PATH},
        "lesson": "Use current paid evidence."}))
    for name, path in (("DONOR", first), ("SECOND_DONOR", second), ("SEED", seed)):
        monkeypatch.setattr(runner, name, path)

    output = tmp_path / "study"
    snapshots, executions, donor_reads = [], [], []
    paid = {"observation_cost": 60, "measurement_cost": 36, "compute_cost": 120,
            "python_execution_seconds": .25, "model_calls": 0, "reported_tokens": 0}

    class FrozenTrial:
        def __init__(self, workspace, *, knowledge, method_source, config, confirmation_seed):
            self.workspace = Path(workspace)
            self.workspace.mkdir(parents=True, exist_ok=True)
            self.knowledge = deepcopy(knowledge)
            self.source = method_source
            snapshots.append(deepcopy(knowledge))
            for name, source in self.knowledge["files"].items():
                path = self.workspace / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source)
            assert config.observation_budget >= 60 and config.confirmation_rows == 48
            role = self.knowledge["research_roles"]["instrument"]
            self.knowledge["files"][role] = "trial-local refitted calibration"
            (self.workspace / "trial_calibration.json").write_text(self.knowledge["files"][role])

        def execute_method(self, lab, max_steps):
            assert max_steps == 8 and lab.public_spec()["max_probe_length"] == 24
            executions.append((self.source, deepcopy(lab._settings)))
            model = self.knowledge["research_roles"]["predictor"]
            if lab._settings["task_index"] == 2:
                model = "shared/revised.py"
                (self.workspace / model).write_text("def predict(row): return 1.\n")
            return {"status": "completed", "model": model,
                    "costs": deepcopy(paid), "events": [], "method_results": [],
                    "fresh_confirmation": {"passed": True, "mse": 0., "rows": 48}}

    monkeypatch.setattr(practice, "FrozenMethodAgent", FrozenTrial)
    monkeypatch.setattr(runner, "FrozenMethodAgent", FrozenTrial)
    original_donor_memory = runner.donor_memory

    def load_donor(path, position, instrument):
        if path == second:
            learned = json.loads((output / "learning/result.json").read_text())
            assert learned["status"] == "completed" and learned["method_source"] == source_b
            assert len(executions) == 4
        donor_reads.append((path, position, instrument))
        return original_donor_memory(path, position, instrument)

    monkeypatch.setattr(runner, "donor_memory", load_donor)
    finish = {"path": METHOD_PATH, "lesson": "A paired tested method."}
    actions = [
        ("run", {"operator": "practice", "case": "unsupported"}),
        ("write", {"path": METHOD_PATH, "code": source_a}),
        ("run", {"operator": "practice", "case": "case_a"}),
        ("run", {"operator": "practice", "case": "case_b"}),
        ("write", {"path": METHOD_PATH, "code": source_b}),
        ("finish", finish),
        ("run", {"operator": "practice", "case": "case_a"}),
        ("finish", finish),
        ("run", {"operator": "practice", "case": "case_b"}),
        ("finish", finish),
    ]

    class Author:
        def __init__(self):
            self.calls = 0

        def next_action(self, request):
            step = self.calls
            self.calls += 1
            if step == 0:
                assert request["files"][METHOD_PATH] == (source_b if unpublished_seed else source_a)
            if step == 1:
                assert "choose case_a or case_b" in request["events"][-1]["feedback"]["error"]
                assert not executions and request["costs"]["observation_cost"] == 0
            if step in (6, 8):
                expected = ["case_a", "case_b"] if step == 6 else ["case_b"]
                assert request["events"][-1]["feedback"]["unconfirmed_cases"] == expected
            if step in (2, 5):
                assert request["events"][-1]["feedback"]["current_source_coverage"]["missing_cases"] == ["case_a", "case_b"]
            if step in (3, 7):
                coverage = request["events"][-1]["feedback"]["current_source_coverage"]
                assert coverage["confirmed_cases"] == coverage["revised_cases"] == ["case_a"]
                assert coverage["missing_cases"] == ["case_b"]
            if step == 9:
                coverage = request["events"][-1]["feedback"]["current_source_coverage"]
                assert coverage["missing_cases"] == [] and coverage["prior_reuse_cases"] == ["case_b"]
            assert [item[0] for item in donor_reads] == [first]
            assert "second paid calibration" not in json.dumps(request)
            action, arguments = actions[step]
            return {"action": action, "arguments": arguments,
                    "usage": {"backend": "scripted_interface_test", "total_tokens": 1}}

    author = Author()

    def evaluate_after_all_four(chain, tasks, **kwargs):
        assert len(executions) == 8
        assert len(list(output.glob("frozen/*/training.json"))) == 4
        chain["evaluation"] = [{"status": "completed", "mse": 0., "rows": 256}]

    monkeypatch.setattr(runner, "evaluate_chain", evaluate_after_all_four)
    summary = runner.run_study(output, provider_factory=lambda path: author,
                              seed_source=candidate if unpublished_seed else None)
    learned = json.loads((output / "learning/result.json").read_text())
    assert author.calls == 10 and summary["status"] == "completed"
    assert learned["method_source"] == source_b
    assert learned["lesson"] == finish["lesson"]
    assert [trial["method_source"] for trial in learned["practice_trials"]] == [source_a, source_a, source_b, source_b]
    assert [trial["case"] for trial in learned["practice_trials"]] == ["case_a", "case_b"] * 2
    assert [trial["selected_prior_predictor"] for trial in learned["practice_trials"]] == [False, True] * 2
    for key in ("observation_cost", "measurement_cost", "compute_cost", "python_execution_seconds"):
        assert learned["costs"][key] == 4 * paid[key]
        assert summary["frozen_costs"][key] == 4 * paid[key]
    assert learned["costs"]["reported_tokens"] == 10
    assert donor_reads == [(first, 1, "shared/meter.json"), (second, 0, "shared/current_instrument.json")]
    for memory in snapshots[:4]:
        assert memory["research_roles"] == {"instrument": "shared/meter.json", "predictor": "shared/prior.py"}
        assert memory["files"]["shared/meter.json"] == "first paid calibration"
    assert [settings["task_index"] for _, settings in executions[:4]] == [2, 1, 2, 1]
    assert all(settings["apparatus_seed"] == 98106007 for _, settings in executions[:4])
    for memory, (source, settings), case in zip(snapshots[4:], executions[4:], runner.protocol()["frozen_cases"]):
        assert source == source_b and settings["task_index"] == case["task"]["task_index"]
        assert memory["lessons"][-1] == finish["lesson"]
        role = memory["research_roles"]["instrument"]
        expected = "second paid calibration" if case["prior"] == "second" else "first paid calibration"
        assert memory["files"][role] == expected
    assert all(case["confirmed_and_correct"] for case in summary["frozen_cases"])


@pytest.mark.parametrize("overwrite_prior_path", [False, True])
def test_two_confirmed_refits_do_not_publish_and_overwritten_prior_is_not_reuse(
        tmp_path, monkeypatch, overwrite_prior_path):
    original_predictor = "def predict(row): return 0.\n"
    source = "def select_actions(spec, records, memory): return {'model':'shared/revised.py'}\n"
    knowledge = {"files": {"shared/prior.py": original_predictor,
                            "shared/instrument.json": "{}", METHOD_PATH: source},
                 "research_roles": {"predictor": "shared/prior.py", "instrument": "shared/instrument.json"},
                 "manifest": {"hypotheses": [{"id": "prior", "path": "shared/prior.py"}],
                              "research_programs": [{"id": "method", "path": METHOD_PATH}]}}

    class ConfirmedRefit:
        def __init__(self, workspace, *, knowledge, method_source, config, confirmation_seed):
            self.workspace = Path(workspace)
            self.workspace.mkdir(parents=True, exist_ok=True)
            self.knowledge = deepcopy(knowledge)
            for name, contents in self.knowledge["files"].items():
                path = self.workspace / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents)

        def execute_method(self, lab, max_steps):
            model = self.knowledge["research_roles"]["predictor"] if overwrite_prior_path else "shared/revised.py"
            (self.workspace / model).write_text("def predict(row): return 1.\n")
            return {"status": "completed", "model": model,
                    "fresh_confirmation": {"passed": True, "mse": 0., "rows": 48},
                    "events": [], "method_results": [],
                    "costs": {"observation_cost": 60, "measurement_cost": 36, "compute_cost": 120,
                              "python_execution_seconds": .25, "model_calls": 0, "reported_tokens": 0}}

    monkeypatch.setattr(practice, "FrozenMethodAgent", ConfirmedRefit)
    learner = practice.PairedMethodLearner(None, tmp_path / "learner", knowledge=knowledge,
        cases=runner.protocol()["practice_cases"], lab_factory=runner.laboratory,
        config=InquiryLLMConfig(observation_budget=960, confirmation_rows=0,
                               execution_seconds=20., compute_budget=10000))
    for case in ("case_a", "case_b"):
        outcome = learner._execute("run", {"operator": "practice", "case": case}, learner.interface)
        assert outcome["status"] == "completed" and outcome["fresh_confirmation"]["passed"]
        assert outcome["selected_prior_predictor"] is False
        if overwrite_prior_path:
            assert outcome["model"] == knowledge["research_roles"]["predictor"]
    refused = learner._execute("finish", {"path": METHOD_PATH}, learner.interface)
    assert refused["error"]
    assert refused["confirmed_prior_reuse_cases"] == []
    assert refused["confirmed_revision_cases"] == ["case_a", "case_b"]
    assert learner.model is None
    assert learner.knowledge["files"]["shared/prior.py"] == original_predictor

    learner._execute("write", {"path": METHOD_PATH, "code": source + "# Untested source revision.\n"},
                     learner.interface)
    untested = learner._execute("finish", {"path": METHOD_PATH}, learner.interface)
    assert untested["unconfirmed_cases"] == ["case_a", "case_b"]
