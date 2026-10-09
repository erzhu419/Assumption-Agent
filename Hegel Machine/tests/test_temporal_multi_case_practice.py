"""Scripted publication plumbing for three cases; not method-learning evidence."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import temporal_paired_practice as practice
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH


@pytest.mark.parametrize("labels", [("case_a", "case_b", "case_c"),
                                    ("trial_x", "trial_y", "trial_z")])
def test_all_actual_cases_require_current_source_confirmation_without_exposing_settings(
        tmp_path, monkeypatch, labels):
    source_a = "def select_actions(spec, records, memory): return {'actions': []}\n"
    source_b = source_a + "# A newly authored revision invalidates earlier coverage.\n"
    knowledge = {"files": {"shared/prior.py": "def predict(row): return 0.\n",
                            "shared/instrument.json": "{}", METHOD_PATH: source_a},
                 "research_roles": {"predictor": "shared/prior.py",
                                    "instrument": "shared/instrument.json"},
                 "manifest": {"hypotheses": [{"id": "prior", "path": "shared/prior.py"}],
                              "research_programs": [{"id": "method", "path": METHOD_PATH}]}}
    cases = {label: {"confirmation_seed": index + 10,
                     "private_mechanism": "hidden-original" if index == 0 else "hidden-changed"}
             for index, label in enumerate(labels)}
    trial_settings = []

    class Laboratory:
        def __init__(self, settings):
            self.settings = settings

        def public_spec(self):
            return {"max_probe_length": 24}

    class ConfirmedTrial:
        def __init__(self, workspace, *, knowledge, method_source, config, confirmation_seed):
            self.workspace = Path(workspace)
            self.workspace.mkdir(parents=True)
            self.knowledge = deepcopy(knowledge)
            for name, contents in knowledge["files"].items():
                path = self.workspace / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents)

        def execute_method(self, lab, max_steps):
            trial_settings.append(deepcopy(lab.settings))
            model = self.knowledge["research_roles"]["predictor"]
            if lab.settings["private_mechanism"] == "hidden-changed":
                model = "shared/revised.py"
                (self.workspace / model).write_text("def predict(row): return 1.\n")
            return {"status": "completed", "model": model, "events": [],
                    "fresh_confirmation": {"passed": True, "mse": 0., "rows": 48},
                    "costs": {"observation_cost": 60, "measurement_cost": 36,
                              "compute_cost": 120, "python_execution_seconds": .25,
                              "model_calls": 0, "reported_tokens": 0}}

    monkeypatch.setattr(practice, "FrozenMethodAgent", ConfirmedTrial)
    actions = [
        ("run", {"operator": "practice", "case": "absent_case"}),
        ("run", {"operator": "practice", "case": labels[0]}),
        ("run", {"operator": "practice", "case": labels[1]}),
        ("finish", {"path": METHOD_PATH}),
        ("write", {"path": METHOD_PATH, "code": source_b}),
        ("run", {"operator": "practice", "case": labels[2]}),
        ("finish", {"path": METHOD_PATH}),
        ("run", {"operator": "practice", "case": labels[0]}),
        ("run", {"operator": "practice", "case": labels[1]}),
        ("finish", {"path": METHOD_PATH}),
    ]

    class Author:
        def __init__(self):
            self.calls = 0

        def next_action(self, request):
            step = self.calls
            self.calls += 1
            public = json.dumps(request)
            for label in labels:
                assert label in request["task_interface"]
            assert "private_mechanism" not in public and "hidden-" not in public
            assert request["capabilities"] == {"max_probe_length": 24}
            if labels[0] != "case_a":
                assert "case_a" not in request["task_interface"]
                assert "case_b" not in request["task_interface"]
            if step == 1:
                error = request["events"][-1]["feedback"]["error"]
                assert all(label in error for label in labels)
                assert not trial_settings
            if step == 4:
                feedback = request["events"][-1]["feedback"]
                assert feedback["unconfirmed_cases"] == [labels[2]]
                assert labels[2] in feedback["error"]
            if step == 5:
                coverage = request["events"][-1]["feedback"]["current_source_coverage"]
                assert coverage["confirmed_cases"] == []
                assert coverage["missing_cases"] == sorted(labels)
            if step == 7:
                assert request["events"][-1]["feedback"]["unconfirmed_cases"] == sorted(labels[:2])
            action, arguments = actions[step]
            return {"action": action, "arguments": arguments,
                    "usage": {"backend": "scripted_interface_test", "total_tokens": 1}}

    author = Author()
    learner = practice.PairedMethodLearner(author, tmp_path / "learner", knowledge=knowledge,
        cases=cases, lab_factory=Laboratory,
        config=InquiryLLMConfig(observation_budget=960, confirmation_rows=0))
    result = learner.run()
    assert result["status"] == "completed" and author.calls == len(actions)
    assert result["method_source"] == source_b
    assert result["events"][-1]["feedback"]["confirmed_cases"] == sorted(labels)
    assert [trial["case"] for trial in result["practice_trials"]] == [
        labels[0], labels[1], labels[2], labels[0], labels[1]]
    coverage = learner._current_source_coverage()
    assert coverage["prior_reuse_cases"] == [labels[0]]
    assert coverage["revised_cases"] == sorted(labels[1:])
    assert trial_settings == [cases[label] for label in
                              (labels[0], labels[1], labels[2], labels[0], labels[1])]
    assert result["costs"]["observation_cost"] == 300
