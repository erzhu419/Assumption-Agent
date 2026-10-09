"""Shared research operators remain explicit choices over paid observations."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from random import Random

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import ARMS, InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import (
    TemporalOperatorAgent, export_temporal_program,
)


def make_agent(tmp_path, arm="code_react", *, source=False, mode="recurrent"):
    lab = TemporalInquiryLab(task_index=0 if source else 1, apparatus_seed=910061,
                             instrument_mode=mode)
    agent = TemporalOperatorAgent(None, tmp_path, arm=arm, source_task=source,
        config=InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
                               execution_seconds=30., compute_budget=100000))
    return agent, lab


def random_action(rng, length, *, meter=False):
    return {"kind": "trajectory", "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(length)],
            "channels": ["meter"] if meter else []}


def buy_target_prefix(agent, lab):
    rng = Random(71)
    actions = [random_action(rng, 24) for _ in range(4)]
    actions += [{"kind": "trajectory", "sequence": [{"u": 1.}, {"u": 0.}, {"u": 0.}],
                 "channels": ["meter"]}]
    actions += [random_action(rng, 3, meter=True) for _ in range(4)]
    acquired = agent._execute("query", {"actions": actions}, lab)
    assert "error" not in acquired
    assert len(agent.rows) == 111


def run_operator(agent, lab, operator, **arguments):
    result = agent._execute("run", {"operator": operator, **arguments}, lab)
    assert "error" not in result, result
    return result


def test_long_trajectories_support_both_fit_and_validation_after_short_plan_reuse(tmp_path):
    agent, lab = make_agent(tmp_path, mode="finite2")
    rng = Random(79)
    agent._execute("query", {"actions": [random_action(rng, length) for length in (3, 3, 24, 24)]}, lab)
    candidate = run_operator(agent, lab, "history")
    split = candidate["split"]
    for group in ("training_records", "validation_records"):
        assert any(len(agent.observations[index]["rows"]) == 24 for index in split[group])
    assert candidate["status"] == "revised" and candidate["selected_lags"] == 3
    assert candidate["revision"]["heldout_mse"] < 1e-10
    score = agent._execute("finish", {"model": candidate["path"]}, lab)
    assert score["fresh_confirmation"]["passed"]


@pytest.mark.parametrize("mode", ["finite2", "recurrent"])
def test_short_scope_history_exports_valid_base_when_extension_unsupported(tmp_path, mode):
    agent, lab = make_agent(tmp_path, source=True, mode=mode)
    rng = Random(72)
    agent._execute("query", {"actions": [random_action(rng, 3) for _ in range(12)]}, lab)
    before = deepcopy(lab.costs)
    candidate = run_operator(agent, lab, "history")
    assert candidate["status"] == "unsupported"
    assert candidate["selected_lags"] == 2
    assert lab.costs == before
    assert agent.model is None
    score = agent._execute("evaluate", {"model": candidate["path"]}, lab)
    assert score["returncode"] == 0
    assert score["mse"] < 1e-10


def test_all_arms_receive_identical_fits_and_ambiguous_response_candidates(tmp_path):
    outputs = []
    for arm in ARMS:
        agent, lab = make_agent(tmp_path / arm, arm)
        buy_target_prefix(agent, lab)
        before = deepcopy(lab.costs)
        instruments = run_operator(agent, lab, "instrument")
        assert len(instruments["hypotheses"]) >= 2
        assert all(item["status"] == "training_candidate" for item in instruments["hypotheses"])
        stored_instrument = (agent.workspace / instruments["path"]).read_text()
        response = run_operator(agent, lab, "response", instrument=instruments["path"],
                                validation_records=[2, 3])
        assert response["instrument"] == instruments["path"]
        assert "evidence saved in this instrument file" in response["instrument_status_note"]
        assert "does not test current meter readings" in response["instrument_status_note"]
        assert (agent.workspace / instruments["path"]).read_text() == stored_instrument
        assert len(response["candidates"]) == len(instruments["hypotheses"])
        assert all(candidate["stored_instrument_status"] == "training_candidate"
                   and "instrument_status" not in candidate for candidate in response["candidates"])
        losses = [item["validation_mse"] for item in response["candidates"]]
        assert min(losses) < 1e-10
        assert max(losses) > 1e-6
        assert lab.costs == before
        assert agent.model is None
        # Ordinary Code has the same executable candidate artifacts and fits as
        # Hegel; the operator itself makes no submission or acquisition choice.
        outputs.append({
            "hypotheses": instruments["hypotheses"],
            "scores": response["candidates"],
            "sources": [(agent.workspace / candidate["path"]).read_text()
                        for candidate in response["candidates"]],
        })
    assert outputs[0] == outputs[1] == outputs[2]


def test_disagreement_proposes_without_acquisition_then_assesses_paid_record(tmp_path):
    agent, lab = make_agent(tmp_path)
    buy_target_prefix(agent, lab)
    instruments = run_operator(agent, lab, "instrument")
    before = deepcopy(lab.costs)
    evidence = deepcopy(agent.observations)
    proposal = run_operator(agent, lab, "disagreement", instrument=instruments["path"])
    assert proposal["action"] is not None
    assert proposal["action"]["channels"] == ["meter"]
    assert len(proposal["action"]["sequence"]) > 3
    assert lab.costs == before
    assert agent.observations == evidence

    first_new_record = len(agent.observations)
    acquired = agent._execute("query", {"actions": [proposal["action"]]}, lab)
    assert "error" not in acquired
    paid_costs = deepcopy(lab.costs)
    assessed = run_operator(agent, lab, "assess_instrument", instrument=instruments["path"],
                            records=[first_new_record])
    survivors = [item for item in assessed["hypotheses"]
                 if item["status"] not in {"refuted", "counterexample"}]
    assert len(survivors) == 1
    assert survivors[0]["status"] == "supported"
    assert lab.costs == paid_costs
    assert agent.model is None


def test_response_validation_labels_do_not_enter_candidate_training(tmp_path):
    class ShiftedValidationLab(TemporalInquiryLab):
        def execute(self, action):
            record = super().execute(action)
            if record["observation_index"] in {48, 72}:
                for row in record["rows"]:
                    row["target"] += 10.
            return record

    comparisons = []
    for shifted in (False, True):
        agent, lab = make_agent(tmp_path / str(shifted))
        if shifted:
            lab = ShiftedValidationLab(task_index=1, apparatus_seed=910061,
                                       instrument_mode="recurrent")
        buy_target_prefix(agent, lab)
        instruments = run_operator(agent, lab, "instrument")
        response = run_operator(agent, lab, "response", instrument=instruments["path"],
                                validation_records=[2, 3])
        comparisons.append({candidate["hypothesis_id"]: candidate
                            for candidate in response["candidates"]})
    assert comparisons[0].keys() == comparisons[1].keys()
    for identifier, original in comparisons[0].items():
        changed = comparisons[1][identifier]
        assert changed["train_mse"] == pytest.approx(original["train_mse"], abs=1e-20)
        assert changed["validation_mse"] > 90.


def test_exported_operator_candidate_uses_existing_evaluate_and_fresh_finish(tmp_path):
    agent, lab = make_agent(tmp_path)
    buy_target_prefix(agent, lab)
    instruments = run_operator(agent, lab, "instrument")
    response = run_operator(agent, lab, "response", instrument=instruments["path"],
                            validation_records=[2, 3])
    candidate = min(response["candidates"], key=lambda item: item["validation_mse"])
    frozen = (agent.workspace / candidate["path"]).read_text()
    assert "hegel_machine" not in frozen
    evaluated = agent._execute("evaluate", {"model": candidate["path"]}, lab)
    assert evaluated["returncode"] == 0 and evaluated["mse"] < 1e-10
    assert agent.model is None
    finished = agent._execute("finish", {"model": candidate["path"]}, lab)
    assert finished["fresh_confirmation"]["passed"]
    assert finished["fresh_confirmation"]["rows"] == 48
    assert agent.model == candidate["path"]
    assert lab.costs["observation_cost"] == 159
    assert (agent.workspace / candidate["path"]).read_text() == frozen


@pytest.mark.parametrize("recurrent", [False, True])
def test_export_matches_program_for_long_histories_and_resets(recurrent):
    control = Program("feature", name="u")
    previous_control = Program("if", children=(Program("available", lag=1),
        Program("lag", name="u", lag=1), Program("const", value=0.)))
    instrument = (Program("accumulate", value=.63, children=(control, Program("const", value=.2)))
                  if recurrent else Program("add", children=(control, previous_control)))
    previous_instrument = Program("if", children=(Program("available", lag=1),
        Program("sub", children=(instrument, Program("event_delta", lag=1, children=(instrument,)))),
        Program("const", value=0.)))
    predictor = Program("add", children=(Program("mul", children=(instrument, instrument)),
                                          previous_instrument))
    source = export_temporal_program(predictor)
    assert "hegel_machine" not in source
    namespace = {}
    exec(compile(source, "exported_model.py", "exec"), namespace)
    predict = namespace["predict"]
    rng = Random(73)
    examples = []
    for length in (3, 24, 47, 1):
        history = []
        for _ in range(length):
            features = {"u": rng.uniform(-1., 1.)}
            row = {"features": features, "history": deepcopy(history)}
            expected = predictor.predict(Observation(features, 0., tuple(history)))
            examples.append((row, expected))
            history.append(features)
    for row, expected in [*examples, *reversed(examples)]:
        # No target or measured state is passed to the exported predictor, and
        # reversing calls exposes accidental state retained across trajectories.
        assert predict(row) == pytest.approx(expected, rel=1e-11, abs=1e-11)


def buy_short_prefix(agent, lab):
    rng = Random(74)
    actions = [random_action(rng, 3) for _ in range(12)]
    actions += [{"kind": "trajectory", "sequence": [{"u": 1.}, {"u": 0.}, {"u": 0.}],
                 "channels": ["meter"]}]
    actions += [random_action(rng, 3, meter=True) for _ in range(4)]
    feedback = agent._execute("query", {"actions": actions}, lab)
    assert "error" not in feedback
    assert len(agent.rows) == 51


def continuation(output_dir="shared/revised"):
    return [
        {"operator": "assess_instrument", "instrument": "instrument.json", "records": "acquired"},
        {"operator": "response", "instrument": "instrument.json", "output_dir": output_dir},
    ]


def test_new_measured_evidence_selects_different_child_and_repairs_parent(tmp_path):
    paired = []
    for mode in ("finite2", "recurrent"):
        agent, lab = make_agent(tmp_path / mode, mode=mode)
        buy_short_prefix(agent, lab)
        if not paired:
            instruments = run_operator(agent, lab, "instrument")
            frozen = json.loads((agent.workspace / instruments["path"]).read_text())
        else:
            # Transfer the identical frozen hypotheses learned from the common
            # short-scope evidence, not two independently rounded fits.
            frozen = deepcopy(paired[0][3])
            agent._execute("write", {"path": "instrument.json", "code": json.dumps(frozen)}, lab)
        proposal = run_operator(agent, lab, "disagreement", instrument="instrument.json")
        paired.append((agent, lab, deepcopy(agent.observations), frozen, proposal["action"]))

    # The short observations agree to floating-point precision. The frozen
    # competitors and proposed action are exactly the same in both worlds.
    for left, right in zip(paired[0][2], paired[1][2]):
        assert left["action"] == right["action"]
        assert [row["target"] for row in left["rows"]] == pytest.approx(
            [row["target"] for row in right["rows"]], abs=1e-14)
        assert [item["values"]["meter"] for item in left["measurements"]] == pytest.approx(
            [item["values"]["meter"] for item in right["measurements"]], abs=1e-14)
    assert paired[0][3:] == paired[1][3:]
    selected, revised_sources = [], []
    for agent, lab, old_records, frozen, action in paired:
        before_count = len(agent.observations)
        feedback = agent._execute("query", {"actions": [action],
            "after_observation": continuation()}, lab)
        assert "error" not in feedback and "after_observation_error" not in feedback
        assert feedback["record_indices"] == [before_count]
        assessment, response = feedback["after_observation_results"]
        assert assessment["records"] == [before_count]
        assert assessment["operator"] == "assess_instrument"
        assert response["operator"] == "response"
        assert len(response["candidates"]) == 1
        candidate = response["candidates"][0]
        assert candidate["stored_instrument_status"] == "supported"
        assert candidate["validation_mse"] < 1e-10
        assert agent.model is None  # continuation never submits its own choice
        revised = json.loads((agent.workspace / "instrument.json").read_text())
        assert [item["program"] for item in revised["fields"]["meter"]["hypotheses"]] == [
            item["program"] for item in frozen["fields"]["meter"]["hypotheses"]]
        selected.append(candidate["hypothesis_id"])
        revised_sources.append((agent.workspace / candidate["path"]).read_text())
        finished = agent._execute("finish", {"model": candidate["path"]}, lab)
        assert finished["fresh_confirmation"]["passed"]
    assert selected[0] != selected[1]
    assert revised_sources[0] != revised_sources[1]


def test_reused_executable_plan_chooses_and_runs_continuations_in_every_arm(tmp_path):
    source = """import json
def select_actions(spec, observations, memory):
    long_seen = any(len(record['action']['sequence']) > 3 for record in observations)
    if not long_seen:
        action = json.load(open('probe.json'))['action']
        steps = [{'operator':'assess_instrument','instrument':'instrument.json','records':'acquired'},
                 {'operator':'response','instrument':'instrument.json','output_dir':'shared/first'}]
    else:
        action = {'kind':'trajectory','sequence':[{'u':.6 if i%2 else -.4}
                  for i in range(spec['max_probe_length'])],'channels':[]}
        steps = [{'operator':'response','instrument':'instrument.json','output_dir':'shared/second'}]
    return {'actions':[action], 'after_observation':steps,
            'question':{'id':'long2' if long_seen else 'long1',
                        'question':'Which learned coordinate predicts the extended response?',
                        'parent_id':'response','depends_on':[]}}
"""
    outcomes = []
    for arm in ARMS:
        agent, lab = make_agent(tmp_path / arm, arm)
        buy_short_prefix(agent, lab)
        instruments = run_operator(agent, lab, "instrument")
        proposal = run_operator(agent, lab, "disagreement", instrument=instruments["path"])
        agent._execute("write", {"path": "probe.json", "code": json.dumps(proposal)}, lab)
        agent._execute("write", {"path": "shared/plan.py", "code": source}, lab)
        first = agent._execute("query", {"path": "shared/plan.py"}, lab)
        assert first["continuation_origin"] == "shared/plan.py"
        assert [item["operator"] for item in first["after_observation_results"]] == [
            "assess_instrument", "response"]
        assert first["observations"][0]["action"] == proposal["action"]
        assert first["after_observation_results"][0]["records"] == first["record_indices"]
        second = agent._execute("query", {"path": "shared/plan.py"}, lab)
        assert second["continuation_origin"] == "shared/plan.py"
        assert [item["operator"] for item in second["after_observation_results"]] == ["response"]
        assert second["observations"][0]["action"] != first["observations"][0]["action"]
        assert len(second["observations"][0]["rows"]) == lab.public_spec()["max_probe_length"]
        assert first["executed_path"] == second["executed_path"] == "shared/plan.py"
        assert [question["id"] for question in agent.questions] == ["long1", "long2"]
        assert agent.questions[0]["after_observation_results"] == first["after_observation_results"]
        assert agent.model is None
        candidates = second["after_observation_results"][0]["candidates"]
        assert len(candidates) == 1 and candidates[0]["validation_mse"] < 1e-10
        assert candidates[0]["path"].startswith("shared/second/")
        outcomes.append(candidates)
        before_compute = agent.costs["compute_cost"]
        overridden = agent._execute("query", {"path": "shared/plan.py", "after_observation": []}, lab)
        assert overridden["continuation_origin"] == "arguments"
        assert overridden["after_observation"] == []
        assert not overridden.get("after_observation_results")
        assert overridden["observations"]  # explicit empty suppresses only the continuation
        assert agent.costs["compute_cost"] == before_compute
    assert outcomes[0] == outcomes[1] == outcomes[2]


def test_failed_continuation_keeps_paid_evidence_and_query_failure_skips_continuation(tmp_path):
    agent, lab = make_agent(tmp_path)
    buy_short_prefix(agent, lab)
    instruments = run_operator(agent, lab, "instrument")
    proposal = run_operator(agent, lab, "disagreement", instrument=instruments["path"])
    before_count = len(agent.observations)
    steps = [continuation()[0],
             {"operator": "response", "instrument": "missing.json"},
             {"operator": "history", "path": "should_not_run.py"}]
    feedback = agent._execute("query", {"actions": [proposal["action"]],
        "after_observation": steps}, lab)
    assert "error" not in feedback
    assert feedback["after_observation_error"]
    assert len(agent.observations) == before_count + 1
    assert feedback["observations"] == agent.observations[-1:]
    assert feedback["record_indices"] == [before_count]
    assert len(feedback["after_observation_results"]) == 2
    completed, failed = feedback["after_observation_results"]
    assert completed["operator"] == "assess_instrument" and "error" not in completed
    assert any(item["status"] == "supported" for item in completed["hypotheses"])
    assert failed["operator"] == "response" and failed["error"]
    assert not (agent.workspace / "should_not_run.py").exists()
    assert json.loads((agent.workspace / "laboratory_observations.json").read_text()) == json.loads(
        json.dumps(agent.observations))

    costs = deepcopy(agent.costs)
    evidence = deepcopy(agent.observations)
    denied = agent._execute("query", {
        "actions": [random_action(Random(75), 24)] * 10,
        "after_observation": [{"operator": "history", "path": "should_not_run.py"}]}, lab)
    assert denied["error"]
    assert not denied.get("after_observation_results")
    assert {key: value for key, value in agent.costs.items() if key != "python_execution_seconds"} == {
        key: value for key, value in costs.items() if key != "python_execution_seconds"}
    assert agent.costs["python_execution_seconds"] >= costs["python_execution_seconds"]
    assert agent.observations == evidence
    assert not (agent.workspace / "should_not_run.py").exists()


@pytest.fixture
def supported_old_instrument(tmp_path):
    agent, lab = make_agent(tmp_path / "old_apparatus")
    buy_target_prefix(agent, lab)
    instrument = run_operator(agent, lab, "instrument")
    proposal = run_operator(agent, lab, "disagreement", instrument=instrument["path"])
    record_index = len(agent.observations)
    agent._execute("query", {"actions": [proposal["action"]]}, lab)
    assessed = run_operator(agent, lab, "assess_instrument", instrument=instrument["path"],
                            records=[record_index])
    survivors = [item for item in assessed["hypotheses"] if item["status"] == "supported"]
    assert len(survivors) == 1
    return (agent.workspace / instrument["path"]).read_text()


def test_response_reports_current_meter_contradiction_without_changing_stored_hypothesis(
        tmp_path, supported_old_instrument):
    class ShiftedResponseLab(TemporalInquiryLab):
        def __init__(self, shift):
            super().__init__(task_index=2, apparatus_seed=99106007, instrument_mode="recurrent")
            self.shift = shift

        def execute(self, action):
            record = super().execute(action)
            for row in record["rows"]:
                row["target"] += self.shift
            return record

    saved = json.loads(supported_old_instrument)
    supported = [hypothesis for hypothesis in saved["fields"]["meter"]["hypotheses"]
                 if hypothesis["status"] == "supported"][0]
    program = Program.from_dict(supported["program"])
    diagnostics = []
    for shift in (0., 10.):
        agent, _ = make_agent(tmp_path / f"current_{shift}")
        lab = ShiftedResponseLab(shift)
        buy_target_prefix(agent, lab)
        agent._execute("write", {"path": "inherited.json", "code": supported_old_instrument}, lab)
        before_compute = agent.costs["compute_cost"]
        response = run_operator(agent, lab, "response", instrument="inherited.json", validation_records=[2, 3])
        diagnostic = response["paid_meter_consistency"]
        diagnostics.append(diagnostic)
        assert diagnostic["status"] == "available" and diagnostic["reason"] is None
        assert diagnostic["records"] == [4, 5, 6, 7, 8]
        assert diagnostic["measured_rows"] == diagnostic["scored_rows"] == 15
        assert diagnostic["compute_cost"] == 15
        assert len(diagnostic["hypotheses"]) == 1
        comparison = diagnostic["hypotheses"][0]
        assert comparison["hypothesis_id"] == supported["hypothesis_id"]
        expected_errors = []
        for record in agent.observations[4:]:
            measured = {item["observation_index"]: item["values"]["meter"]
                        for item in record["measurements"]}
            for index, row in zip(record["row_indices"], record["rows"]):
                predicted = program.predict(Observation(row["features"], 0., tuple(row["history"])))
                expected_errors.append((predicted - measured[index]) ** 2)
        assert comparison["mse"] == pytest.approx(sum(expected_errors) / len(expected_errors))
        assert comparison["max_squared_error"] == pytest.approx(max(expected_errors))
        assert comparison["mse"] > 1e-4
        assert comparison["scored_rows"] == 15
        # Existing response fits still return the stored supported candidate.
        # Reporting contradictory measurements neither refutes nor replaces it.
        assert len(response["candidates"]) == 1
        assert response["candidates"][0]["hypothesis_id"] == supported["hypothesis_id"]
        assert response["candidates"][0]["stored_instrument_status"] == "supported"
        assert (agent.workspace / "inherited.json").read_text() == supported_old_instrument
        response_fit_cost = 12 * 63 + 48 + 12 * 111
        assert response["compute_cost"] == response_fit_cost + 15
        assert agent.costs["compute_cost"] - before_compute == response["compute_cost"]
    assert diagnostics[0] == diagnostics[1]  # response targets cannot affect meter residuals
    # Exhaust only the allowance for this additional diagnostic. The existing
    # response fit remains affordable and must still return the same candidates.
    agent.config = replace(agent.config, compute_budget=agent.costs["compute_cost"] + response_fit_cost)
    limited = run_operator(agent, lab, "response", instrument="inherited.json", validation_records=[2, 3])
    assert limited["candidates"] == response["candidates"]
    assert limited["paid_meter_consistency"]["status"] == "unavailable"
    assert limited["paid_meter_consistency"]["reason"] == "insufficient_compute_budget"
    assert limited["paid_meter_consistency"]["scored_rows"] == 0
    assert limited["paid_meter_consistency"]["compute_cost"] == 0
    assert agent.costs["compute_cost"] == agent.config.compute_budget
    assert (agent.workspace / "inherited.json").read_text() == supported_old_instrument


def test_response_with_no_paid_meter_reports_unavailable_instead_of_support(
        tmp_path, supported_old_instrument):
    agent, lab = make_agent(tmp_path / "unmeasured")
    rng = Random(76)
    agent._execute("query", {"actions": [random_action(rng, 24) for _ in range(4)]}, lab)
    agent._execute("write", {"path": "inherited.json", "code": supported_old_instrument}, lab)
    response = run_operator(agent, lab, "response", instrument="inherited.json", validation_records=[2, 3])
    diagnostic = response["paid_meter_consistency"]
    assert diagnostic["status"] == "unavailable"
    assert diagnostic["reason"] == "no_paid_meter_readings"
    assert diagnostic["records"] == []
    assert diagnostic["measured_rows"] == diagnostic["scored_rows"] == diagnostic["compute_cost"] == 0
    assert diagnostic["hypotheses"] == []
    assert "mse" not in diagnostic
    assert response["compute_cost"] == 12 * 48 + 48 + 12 * 96
    assert response["candidates"][0]["stored_instrument_status"] == "supported"
    assert (agent.workspace / "inherited.json").read_text() == supported_old_instrument


def test_same_procedure_uses_current_results_to_assess_refit_and_fit_without_new_acquisitions(
        tmp_path, supported_old_instrument):
    agent, _ = make_agent(tmp_path)
    lab = TemporalInquiryLab(task_index=2, apparatus_seed=99106007, instrument_mode="recurrent")

    def forbidden_terminal(**kwargs):
        raise AssertionError("research state cannot request terminal observations")

    lab.evaluation_rows = forbidden_terminal
    buy_target_prefix(agent, lab)
    agent._execute("write", {"path": "instrument.json", "code": supported_old_instrument}, lab)
    source = """import json
def select_actions(spec, observations, memory):
    state = json.load(open('research_state.json'))
    assert state['costs']['observation_cost'] == sum(len(record['rows']) for record in observations)
    assert state['limits']['confirmation_rows'] == 48
    outcomes = state['operator_results']
    if not outcomes:
        step = {'operator':'assess_instrument','instrument':'instrument.json',
                'records':[i for i,record in enumerate(observations) if record['measurements']]}
    elif outcomes[-1]['operator'] == 'assess_instrument':
        hypotheses = outcomes[-1]['hypotheses']
        if any(item['status'] == 'supported' for item in hypotheses):
            step = {'operator':'response','instrument':'instrument.json'}
        else:
            step = {'operator':'instrument','path':'instrument.json'}
    else:
        assert outcomes[-1]['operator'] in ('instrument','response')
        step = {'operator':'response','instrument':'instrument.json'}
    return {'actions':[], 'after_observation':[step],
            'question':{'id':'adapt','question':'Which analysis does the latest evidence require?',
                        'parent_id':'response','depends_on':[]}}
"""
    agent._execute("write", {"path": "shared/adapt.py", "code": source}, lab)
    records_before, physical_costs = deepcopy(agent.observations), deepcopy(lab.costs)
    operations = ("assess_instrument", "instrument", "response", "response")
    previous_outcomes = []
    for index, expected in enumerate(operations):
        before_costs = deepcopy(agent.costs)
        before_compute = agent.costs["compute_cost"]
        result = agent._execute("query", {"path": "shared/adapt.py"}, lab)
        assert not result.get("error") and not result.get("after_observation_error"), result
        assert not result.get("returncode"), result
        assert result["record_indices"] == result["observations"] == []
        assert result["observation_cost"] == 0
        outcome = result["after_observation_results"][0]
        assert outcome["operator"] == expected
        assert outcome["compute_cost"] > 0
        assert agent.costs["compute_cost"] - before_compute == outcome["compute_cost"]
        assert agent.observations == records_before and lab.costs == physical_costs
        assert (agent.workspace / "shared/adapt.py").read_text() == source
        state = json.loads((agent.workspace / "research_state.json").read_text())
        required_keys = {"operator_results", "limits", "costs",
                         "exported_candidate_paths", "evaluated_model"}
        assert set(state) in (required_keys, required_keys | {"research_advice"})
        if "research_advice" in state:
            assert state["research_advice"]["paid_rows"] == len(agent.rows)
            assert state["research_advice"]["evaluated_model"] == state["evaluated_model"]
        assert state["evaluated_model"] is None
        assert state["exported_candidate_paths"] == (
            [candidate["path"] for candidate in previous_outcomes[-1]["candidates"]]
            if index == 3 else [])
        assert state["limits"] == asdict(agent.config)
        # The state is the snapshot supplied to this invocation: prior results
        # and costs are current before this call's Python and operator work.
        assert state["costs"] == before_costs
        assert len(state["operator_results"]) == index
        assert [{key: value for key, value in entry.items() if key != "index"}
                for entry in state["operator_results"]] == previous_outcomes
        if index:
            assert state["operator_results"][-1]["arguments"]["operator"] == operations[index - 1]
            assert state["operator_results"][-1]["operator"] == operations[index - 1]
        if expected == "assess_instrument":
            assert all(item["status"] == "refuted" for item in outcome["hypotheses"])
        previous_outcomes.append(deepcopy({key: value for key, value in outcome.items()
                                           if key != "research_advice"}))
    assert min(candidate["validation_mse"] for candidate in outcome["candidates"]) < 1e-10
    assert agent.model is None


def test_analysis_only_acquired_sentinel_cannot_reuse_earlier_measurements(
        tmp_path, supported_old_instrument):
    agent, lab = make_agent(tmp_path)
    buy_target_prefix(agent, lab)
    agent._execute("write", {"path": "instrument.json", "code": supported_old_instrument}, lab)
    payload = {"actions": [], "after_observation": [
        {"operator": "assess_instrument", "instrument": "instrument.json", "records": "acquired"}]}
    source = "def select_actions(spec, observations, memory):\n    return " + repr(payload) + "\n"
    agent._execute("write", {"path": "analysis.py", "code": source}, lab)
    before, spent = deepcopy(lab.costs), agent.costs["compute_cost"]
    result = agent._execute("query", {"path": "analysis.py"}, lab)
    assert result["record_indices"] == result["observations"] == []
    assert result["after_observation_error"]
    assert result["after_observation_results"][0]["arguments"]["records"] == []
    assert result["after_observation_results"][0]["error"]
    assert lab.costs == before and agent.costs["compute_cost"] == spent
    assert (agent.workspace / "instrument.json").read_text() == supported_old_instrument
    empty = "def select_actions(spec, observations, memory):\n    return {'actions':[], 'after_observation':[]}\n"
    agent._execute("write", {"path": "analysis.py", "code": empty}, lab)
    with pytest.raises(ValueError):
        agent._execute("query", {"path": "analysis.py"}, lab)
    assert lab.costs == before and agent.costs["compute_cost"] == spent
    state = json.loads((agent.workspace / "research_state.json").read_text())
    assert state["operator_results"][-1]["arguments"]["records"] == []
    assert state["operator_results"][-1]["error"]
    assert {key: value for key, value in state["operator_results"][-1].items()
            if key != "index"} == result["after_observation_results"][0]
