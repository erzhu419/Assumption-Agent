"""Scripted method-repair orchestration checks, not evidence of LLM learning."""

from copy import deepcopy
import json
from pathlib import Path
from random import Random
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import temporal_paired_practice as practice
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH, method_memory_pair
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent
from scripts import run_temporal_calibration_transfer as transfer
from scripts import run_temporal_counterexample_learning as runner
from test_temporal_paired_retention import LESSON, METHOD, original_memory


def failure_file(path):
    lab = runner.laboratory(runner.protocol()["practice_cases"]["case_c"])
    record = lab.execute({"kind": "trajectory", "sequence": [{"u": .2}, {"u": -.3}],
                          "channels": ["meter"]})
    costs = {"observation_cost": 2, "measurement_cost": 6, "compute_cost": 2,
             "python_execution_seconds": .01, "model_calls": 0, "reported_tokens": 0}
    failed = {"status": "budget_exhausted", "costs": costs, "paid_observations": [record],
              "events": [{"action": "query", "arguments": {"path": METHOD_PATH},
                  "feedback": {"method_submission": "shared/prior.py",
                      "fresh_confirmation": {"passed": False, "mse": 1.25,
                                             "rows": 2, "observations": [record]},
                      "after_observation_results": [{"operator": "assess_instrument",
                                                     "supported": True, "mse": 0.}]}}],
              "hidden_settings": {"answer": "hidden-answer-must-not-transfer"},
              "evaluation": {"mse": 77, "note": "terminal-answer-must-not-transfer"}}
    path.write_text(json.dumps({"tasks": [failed], "evaluation": [failed["evaluation"]]}))
    return json.loads(path.read_text())["tasks"][0]


def test_counterexample_supplies_paid_failure_without_changing_prior_or_confirmation(tmp_path):
    path = tmp_path / "failure.json"
    failed = failure_file(path)
    memory = original_memory(1.25, "shared/meter.json")
    memory["files"][METHOD_PATH] = METHOD
    learner = practice.PairedMethodLearner(None, tmp_path / "author", knowledge=memory,
        cases=runner.protocol()["practice_cases"], lab_factory=runner.laboratory,
        config=InquiryLLMConfig())
    original = deepcopy(learner.knowledge)

    assert runner.supply_counterexample(learner, path) == failed["costs"]
    summary = json.loads(learner.files["counterexample.json"])
    feedback = summary["steps"][0]["feedback"]
    assert feedback["method_submission"] == "shared/prior.py"
    assert feedback["fresh_confirmation"] == {"passed": False, "mse": 1.25, "rows": 2}
    assert feedback["operators"][0]["operator"] == "assess_instrument"
    records = json.loads((learner.workspace / "counterexample_observations.json").read_text())
    assert records == failed["paid_observations"]
    trace = json.loads((learner.workspace / "counterexample_trace.json").read_text())
    assert trace[0]["feedback"]["fresh_confirmation"]["passed"] is False
    assert trace[0]["feedback"]["after_observation_results"][0]["supported"] is True
    supplied = "".join(p.read_text() for p in learner.workspace.glob("counterexample*.json"))
    assert "hidden-answer-must-not-transfer" not in supplied
    assert "terminal-answer-must-not-transfer" not in supplied
    assert learner.knowledge == original and learner.practice_trials == []
    assert learner.observations == learner.rows == []
    assert learner.costs["observation_cost"] == 0
    assert learner._current_source_coverage()["missing_cases"] == ["case_a", "case_b", "case_c"]


def test_counterexample_ledger_exposes_actual_mixed_record_scopes_and_source(tmp_path):
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061, response_seed=910061,
        calibration_change={"after_rows": 72, "apparatus_seed": 910061, "response_seed": 99106007})
    agent = TemporalOperatorAgent(None, tmp_path / "failed_trial", arm="code_react",
        config=InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
                               execution_seconds=60., compute_budget=200000))
    events = []

    def act(action, **arguments):
        feedback = agent._execute(action, arguments, lab)
        events.append({"step": len(events), "action": action,
                       "arguments": arguments, "feedback": deepcopy(feedback)})
        return feedback

    rng = Random(106007)

    def purchase():
        return act("query", actions=[{"kind": "trajectory", "channels": ["meter"],
            "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)]} for _ in range(3)])

    purchase()
    act("run", operator="instrument", path="old_meter.json")
    old = act("run", operator="response", instrument="old_meter.json", output_dir="old")
    selected = min(old["candidates"], key=lambda item: item["validation_mse"])
    first = act("finish", model=selected["path"])
    assert first["fresh_confirmation"]["passed"] is False
    purchased = purchase()
    assert purchased["record_indices"] == [5, 6, 7]
    act("run", operator="instrument", path="current_meter.json", records=[5, 6, 7])
    mixed = act("run", operator="response", instrument="current_meter.json", output_dir="mixed")
    selected = min(mixed["candidates"], key=lambda item: item["validation_mse"])
    last = act("finish", model=selected["path"])
    assert last["fresh_confirmation"]["passed"] is False
    source = "def select_actions(spec, records, memory):\n    return {'model':'mixed/response_00.py'}\n"
    failed = {"status": "confirmation_budget_exhausted", "costs": agent.costs,
              "events": events, "paid_observations": agent.observations,
              "outgoing_knowledge": {"files": {METHOD_PATH: source}}}
    path = tmp_path / "failure.json"
    path.write_text(json.dumps({"tasks": [failed]}))
    author = tmp_path / "author"
    author.mkdir()
    learner = SimpleNamespace(workspace=author, files={}, common_interface="")
    runner.supply_counterexample(learner, path, case_label="case_b")
    summary = json.loads(learner.files["counterexample.json"])
    assert summary["practice_case"] == "case_b"
    assert summary["executed_method"]["source"] == source
    assert summary["executed_method"]["path"] == METHOD_PATH
    assert summary["paid_records"][0] == {"record": 0, "row_interval": [0, 23],
                                          "rows": 24, "meter_rows": 24}
    assert summary["paid_records"][3] == {"record": 3, "row_interval": [72, 95],
                                          "rows": 24, "meter_rows": 0}
    assert summary["paid_records"][5] == {"record": 5, "row_interval": [120, 143],
                                          "rows": 24, "meter_rows": 24}
    assert summary["steps"][3]["feedback"]["fresh_confirmation"]["row_interval"] == [72, 119]
    assert summary["steps"][7]["feedback"]["fresh_confirmation"]["row_interval"] == [192, 239]
    instrument = summary["steps"][5]["feedback"]["operators"][0]
    assert instrument["records"] == [5, 6, 7] and instrument["support_rows"] == 72
    response = summary["steps"][6]["feedback"]["operators"][0]
    assert response["records"] == response["final_fit_records"] == list(range(8))
    assert response["training_records"] == [0, 1, 2, 3, 4, 5]
    assert response["validation_records"] == [6, 7]
    assert json.loads((author / "counterexample_observations.json").read_text()) == json.loads(
        json.dumps(agent.observations))
    assert "calibration_change" not in learner.files["counterexample.json"]
    assert len(learner.files["counterexample.json"]) < 12000  # full provider preview, not a truncated file


def test_frozen_plan_separates_new_response_and_instrument_changes_from_author_cases():
    plan = runner.protocol()
    practices, cases = plan["practice_cases"], plan["frozen_cases"]
    assert set(practices) == {"case_a", "case_b", "case_c"} and len(cases) == 18
    assert {task["apparatus_seed"] for task in practices.values()} == {transfer.CALIBRATIONS["first"][0]}
    assert {case["prior"] for case in cases} == {"first", "second"}
    author_responses = {task.get("response_seed", task["apparatus_seed"])
                        for task in practices.values()}
    for prior in transfer.CALIBRATIONS:
        response_cases = [case for case in cases if case["prior"] == prior
                          and case["change"] == "response_only"]
        assert len(response_cases) == 4
        assert len({case["task"]["response_seed"] for case in response_cases}) == 4
        assert all(case["task"]["response_seed"] not in author_responses for case in response_cases)
    for case in cases:
        task, old_seed = case["task"], transfer.CALIBRATIONS[case["prior"]][0]
        if case["change"] in ("stable", "response_only"):
            assert task["apparatus_seed"] == old_seed
        else:
            assert task["apparatus_seed"] not in {value[0] for value in transfer.CALIBRATIONS.values()}
        if case["change"] == "instrument_only":
            assert task["response_seed"] == old_seed
        elif case["change"] in ("stable", "both"):
            assert "response_seed" not in task
        assert task["confirmation_seed"] not in {p["confirmation_seed"] for p in practices.values()}
        assert task["evaluation_seed"] != task["confirmation_seed"]


@pytest.mark.parametrize("publish,separate_comparator", [(False, False), (True, False), (True, True)])
def test_publication_gates_equal_frozen_cohorts_and_transfers_only_method(
        tmp_path, monkeypatch, publish, separate_comparator):
    failure = tmp_path / "failure.json"
    history = failure_file(failure)
    seed = tmp_path / "published.json"
    seed.write_text(json.dumps({"status": "completed", "method_path": METHOD_PATH,
        "method_source": METHOD, "method_entry": {"id": "prior_method", "path": METHOD_PATH},
        "outgoing_knowledge": {"lessons": [LESSON]}}))
    monkeypatch.setattr(runner, "SEED", seed)
    monkeypatch.setattr(runner, "COUNTEREXAMPLE", failure)
    output = tmp_path / "study"
    originals = {"first": original_memory(1.25, "shared/meter.json"),
                 "second": original_memory(2.5, "shared/current_instrument.json")}
    donor_reads, trials, evaluations = [], [], []

    def donor(path, position, instrument):
        prior = "first" if position == 1 else "second"
        if prior == "second":
            assert json.loads((output / "learning/result.json").read_text())["status"] == "completed"
        donor_reads.append(prior)
        return deepcopy(originals[prior])

    monkeypatch.setattr(runner, "donor_memory", donor)
    monkeypatch.setattr(transfer, "donor_memory", donor)
    revised = METHOD + "\n# Scripted authored revision, not a learned performance result.\n"
    paid = {"observation_cost": 48, "measurement_cost": 0, "compute_cost": 48,
            "python_execution_seconds": .01, "model_calls": 0, "reported_tokens": 0}

    class FrozenTrial:
        def __init__(self, workspace, *, knowledge, method_source, config, confirmation_seed):
            self.workspace = Path(workspace)
            self.workspace.mkdir(parents=True, exist_ok=True)
            self.knowledge, self.source = deepcopy(knowledge), method_source
            for name, source in knowledge["files"].items():
                path = self.workspace / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source)
            if "trials" not in self.workspace.parts:
                assert json.loads((output / "learning/result.json").read_text())["status"] == "completed"
            # A fitted practice artifact is inspectable locally, but must never
            # be promoted into the next trial's numerical prior.
            assert "shared/practice_fit.py" not in knowledge["files"]
            (self.workspace / "shared/practice_fit.py").write_text("def predict(row): return 9.\n")

        def execute_method(self, lab, max_steps):
            assert max_steps == 8
            trials.append((self.workspace, self.source, deepcopy(self.knowledge), deepcopy(lab._settings)))
            changed = lab._settings["task_index"] == 2 or lab._settings.get("response_seed") is not None
            model = "shared/practice_fit.py" if changed else self.knowledge["research_roles"]["predictor"]
            return {"workspace": str(self.workspace), "status": "completed", "model": model,
                    "events": [], "method_results": [], "costs": deepcopy(paid),
                    "fresh_confirmation": {"passed": True, "mse": 0., "rows": 48}}

    monkeypatch.setattr(practice, "FrozenMethodAgent", FrozenTrial)
    monkeypatch.setattr(transfer, "FrozenMethodAgent", FrozenTrial)

    def evaluate(chain, tasks, **kwargs):
        cohort = Path(chain["tasks"][0]["workspace"]).parent
        assert len(list(cohort.glob("*/training.json"))) == 18
        evaluations.append(deepcopy(tasks))
        chain["evaluation"] = [{"mse": 0., "rows": 256, "status": "completed"}]

    monkeypatch.setattr(transfer, "evaluate_chain", evaluate)
    actions = [("finish", {"path": METHOD_PATH})]
    if publish:
        actions += [("write", {"path": METHOD_PATH, "code": revised})]
        actions += [("run", {"operator": "practice", "case": case})
                    for case in ("case_a", "case_b", "case_c")]
        actions += [("finish", {"path": METHOD_PATH, "lesson": "The tested repair."})]
    else:
        actions += [("provider_error", {})]

    class Author:
        calls = 0

        def next_action(self, request):
            assert "second" not in donor_reads
            assert request["session_knowledge"] == method_memory_pair(originals["first"], {
                "status": "completed", "method_path": METHOD_PATH, "method_source": METHOD,
                "method_entry": {"id": "prior_method", "path": METHOD_PATH}, "lesson": LESSON})[0]
            if self.calls == 1:
                assert request["events"][-1]["feedback"]["unconfirmed_cases"] == ["case_a", "case_b", "case_c"]
            action, arguments = actions[self.calls]
            self.calls += 1
            return {"action": action, "arguments": arguments,
                    "error": {"message": "end scripted test"},
                    "usage": {"backend": "scripted_interface_test", "total_tokens": 1}}

    plan = runner.protocol()
    plan["counterexample_case"] = "case_b"
    comparator_source = METHOD
    if separate_comparator:
        comparator = json.loads(seed.read_text())
        comparator_source = METHOD + "\n# Earlier successful comparator, distinct from author seed.\n"
        comparator["method_source"] = comparator_source
        comparator_path = tmp_path / "comparator.json"
        comparator_path.write_text(json.dumps(comparator))
        plan["comparison_method"] = str(comparator_path)
    summary = runner.run_study(output, provider_factory=lambda path: Author(), study_protocol=plan)
    assert json.loads((output / "learning/counterexample.json").read_text())["practice_case"] == "case_b"
    assert summary["reused_counterexample_costs"] == history["costs"]
    if not publish:
        assert summary["status"] == "method_not_published"
        assert trials == evaluations == [] and donor_reads == ["first"]
        assert summary["new_learning_costs"]["observation_cost"] == 0
        return
    assert summary["status"] == "completed" and len(trials) == 39
    assert summary["new_learning_costs"]["observation_cost"] == 3 * paid["observation_cost"]
    assert evaluations[:18] == evaluations[18:]
    old_trials, new_trials = trials[3:21], trials[21:]
    for old, new, case in zip(old_trials, new_trials, runner.protocol()["frozen_cases"]):
        assert old[1] == comparator_source and new[1] == revised
        assert old[3] == new[3]
        for memory in (old[2], new[2]):
            original = originals[case["prior"]]
            assert memory["research_roles"] == original["research_roles"]
            for role in original["research_roles"].values():
                assert memory["files"][role] == original["files"][role]
            assert "counterexample.json" not in memory["files"]
            assert "shared/practice_fit.py" not in memory["files"]
