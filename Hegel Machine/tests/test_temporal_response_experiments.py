"""The shared interface reuses paid response-experiment methods from v0.76."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from random import Random

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_method_lab import TemporalMethodLab
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent


BOOK = "shared/instrument_methods.json"
INSTRUMENT = "instrument.json"
HYPOTHESES = "response_hypotheses.json"


def agent_for(workspace, knowledge=None):
    return TemporalOperatorAgent(None, workspace, arm="hegel", source_task=False,
        knowledge=deepcopy(knowledge), config=InquiryLLMConfig(
            observation_budget=240, compute_budget=300000, execution_seconds=30.))


def operator(agent, lab, name, **arguments):
    return agent._execute("run", {"operator": name, **arguments}, lab)


def book(agent):
    path = agent.workspace / BOOK
    return json.loads(path.read_text()) if path.exists() else []


def acquire_random(agent, lab, *, length=24, batches=3, seed=817):
    rng = Random(seed)
    return agent._execute("query", {"actions": [
        {"kind": "trajectory", "sequence": [
            {"u": rng.uniform(-1., 1.)} for _ in range(length)],
         "channels": ["meter"]} for _ in range(batches)]}, lab)


def calibrate(agent, lab):
    acquired = acquire_random(agent, lab, batches=2)
    records = acquired["record_indices"]
    fitted = operator(agent, lab, "instrument", path=INSTRUMENT, records=records[:2])
    assert fitted["status"] == "requires_fresh_validation"
    fresh = acquire_random(agent, lab, batches=1, seed=981)
    assessed = operator(agent, lab, "assess_instrument", instrument=INSTRUMENT,
                        records=fresh["record_indices"])
    assert assessed["status"] == "supported_on_measured_trajectory"


def original_fixture(case, stage, seed, center):
    # Reuse the historical benchmark physics; only its public action adapter is
    # new. Stages 0 and 4 already use the supported control/meter names.
    lab = TemporalInquiryLab(task_index=1)
    lab._laboratory = TemporalMethodLab(case, stage, seed, center=center,
                                      half_width=1e-4, response_gain=2.)
    return lab


def supplied_hypotheses(agent, lab, center):
    # These are the original benchmark's supplied candidates, not discoveries
    # attributed to the agent. The experiment must identify them by buying data.
    script = Path(__file__).resolve().parents[1] / "scripts/run_temporal_method_study.py"
    spec = importlib.util.spec_from_file_location("temporal_method_study", script)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    study.Program = Program
    programs = study.candidate_programs(lab._laboratory, center,
        {"half_width": 1e-4, "candidate_gains": [1., 2.]})
    payload = [{"hypothesis_id": f"response_{index}", "program": program.to_dict()}
               for index, program in enumerate(programs)]
    (agent.workspace / HYPOTHESES).write_text(json.dumps(payload))
    return payload


def propose(agent, lab):
    return operator(agent, lab, "response_disagreement", hypotheses=HYPOTHESES,
                    instrument=INSTRUMENT)


def acquire_and_assess(agent, lab, proposal):
    acquired = agent._execute("query", {"actions": [proposal["action"]],
        "after_observation": [{"operator": "assess_response",
            "experiment": proposal["experiment"], "records": "acquired"}]}, lab)
    return acquired, acquired["after_observation_results"][0]


@pytest.mark.parametrize("case", ["window", "event"])
def test_original_narrow_predicate_experiment_is_learned_and_rebound(tmp_path, case):
    first, lab = agent_for(tmp_path / "first"), original_fixture(case, 0, 151, .21)
    calibrate(first, lab)
    supplied_hypotheses(first, lab, .21)
    assert all(row["target"] == 0. for record in first.observations for row in record["rows"])
    before = deepcopy(lab.costs)
    proposal = propose(first, lab)
    assert proposal["forecast_status"] == "predicted_distinction"
    assert proposal["proposal_source"] == "program_disagreement"
    assert proposal["action"]["channels"] == ["meter"]
    assert lab.costs == before and book(first) == []
    first_plan = json.loads((first.workspace / proposal["experiment"]).read_text())
    acquired, assessed = acquire_and_assess(first, lab, proposal)
    assert assessed["status"] == "distinguished"
    assert assessed["calibration_status"] == "supported_on_probe"
    assert sum(item["status"] == "supported_on_probe" for item in assessed["hypotheses"]) == 1
    assert assessed["method_feedback"]["action"] == "learn_program_experiment_method"
    assert any(row["target"] != 0. for row in first.observations[acquired["record_indices"][0]]["rows"])
    original = book(first)[0]
    assert len(original["support"]) == 1
    outgoing = first.result("researching")["outgoing_knowledge"]
    assert BOOK in outgoing["files"]

    second, lab = agent_for(tmp_path / "second", outgoing), original_fixture(case, 4, 373, -.31)
    calibrate(second, lab)
    supplied_hypotheses(second, lab, -.31)
    before_book, before = deepcopy(book(second)), deepcopy(lab.costs)
    reused = propose(second, lab)
    assert reused["proposal_source"] == "learned_method"
    assert reused["proposal"]["method_id"] == original["method_id"]
    assert book(second) == before_book and lab.costs == before
    second_plan = json.loads((second.workspace / reused["experiment"]).read_text())
    assert second_plan["fresh_synthesis_skipped"]
    # Recalled steps solve the new calibration and threshold, rather than
    # replaying controls that activated the old apparatus's narrow window.
    assert reused["action"] != proposal["action"]
    assert second_plan["predicted_fields"] != first_plan["predicted_fields"]
    _, assessed = acquire_and_assess(second, lab, reused)
    assert assessed["status"] == "distinguished"
    assert assessed["calibration_status"] == "supported_on_probe"
    assert assessed["method_feedback"]["action"] == "use_program_experiment_method"
    assert assessed["method_feedback"]["cross_task"]
    assert len(book(second)) == 1 and len(book(second)[0]["support"]) == 2
    assert first.costs["model_calls"] == second.costs["model_calls"] == 0


def test_fitted_response_programs_match_exports_and_drive_paid_distinction(tmp_path):
    agent = agent_for(tmp_path)
    lab = TemporalInquiryLab(task_index=1, seed=113, apparatus_seed=157)
    acquire_random(agent, lab, length=3, batches=8)
    operator(agent, lab, "instrument", path=INSTRUMENT, records=list(range(6)))
    assessed = operator(agent, lab, "assess_instrument", instrument=INSTRUMENT, records=[6, 7])
    assert assessed["status"] == "ambiguous_on_measured_trajectory"
    response = operator(agent, lab, "response", instrument=INSTRUMENT)
    hypotheses = json.loads((agent.workspace / response["hypotheses_path"]).read_text())
    assert len(hypotheses) == len(response["candidates"]) >= 2
    fresh = lab.evaluation_rows(length=48, seed=937)
    for hypothesis, candidate in zip(hypotheses, response["candidates"]):
        assert hypothesis["hypothesis_id"] == candidate["path"]
        assert hypothesis["path"] == candidate["path"]
        program = Program.from_dict(hypothesis["program"])
        namespace = {}
        exec(compile((agent.workspace / candidate["path"]).read_text(), candidate["path"], "exec"), namespace)
        assert [program.predict(row) for row in fresh] == pytest.approx(
            [namespace["predict"](row) for row in fresh], abs=1e-12)
    before = deepcopy(lab.costs)
    proposal = operator(agent, lab, "response_disagreement", hypotheses=response["hypotheses_path"])
    assert proposal["forecast_status"] == "predicted_distinction"
    assert proposal["action"] and lab.costs == before
    acquired, assessed = acquire_and_assess(agent, lab, proposal)
    assert acquired["record_indices"] == [8]
    assert assessed["status"] == "distinguished"
    assert any(item["status"] == "supported_on_probe" for item in assessed["hypotheses"])
    assert any(item["status"] == "counterexample" for item in assessed["hypotheses"])


def test_changed_meter_refutes_forecast_but_preserves_measured_event_law(tmp_path):
    first, lab = agent_for(tmp_path / "first"), original_fixture("event", 0, 151, .21)
    calibrate(first, lab)
    supplied_hypotheses(first, lab, .21)
    _, initial = acquire_and_assess(first, lab, propose(first, lab))
    assert initial["method_feedback"]["action"] == "learn_program_experiment_method"

    second = agent_for(tmp_path / "second", first.result("researching")["outgoing_knowledge"])
    lab = original_fixture("event", 4, 373, -.31)
    calibrate(second, lab)
    supplied_hypotheses(second, lab, -.31)
    proposal = propose(second, lab)
    assert proposal["proposal_source"] == "learned_method"
    before = deepcopy(book(second))

    # The unchanged response law now receives different real meter values.
    # Switch only the old fixture's calibration between planning and purchase,
    # retaining all already paid global row indices and costs.
    lab._past_costs = deepcopy(lab.costs)
    lab._laboratory = TemporalMethodLab("event", 4, 739, center=-.31,
                                       half_width=1e-4, response_gain=2.)
    _, assessed = acquire_and_assess(second, lab, proposal)
    assert assessed["calibration_status"] == "counterexample"
    assert assessed["field_losses"]["meter"] > 1e-10
    correct = next(item for item in assessed["hypotheses"]
                   if item["hypothesis_id"] == "response_3")
    assert correct["status"] == "supported_on_probe"
    assert correct["measured_loss"] < 1e-10 < correct["forecast_loss"]
    feedback = assessed["method_feedback"]
    assert feedback["action"] == "revise_program_experiment_method"
    assert feedback["reason"] == "calibration_counterexample"
    assert feedback["status"] == "condition_restricted"
    after = book(second)
    assert len(after) == len(before) == 1
    assert after[0]["support"] == before[0]["support"]
    assert len(after[0]["counterexamples"]) == len(before[0]["counterexamples"]) + 1
