"""Scripted real-laboratory checks of the evaluator, not method-learning evidence."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.temporal_selective_practice import (
    SelectiveMethodLearner, selective_revision_evidence,
)
from test_temporal_paired_method import calibrated_knowledge, make_frozen


SCRIPTED_SELECTIVE = """import json
from random import Random
def select_actions(spec, records, memory):
    state = json.load(open('research_state.json'))
    roles = memory['research_roles']
    failed = [entry for entry in state['method_results']
              if entry['feedback'].get('fresh_confirmation', {}).get('passed') is False]
    assessed = [entry for entry in state['operator_results'] if entry['operator']=='assess_instrument']
    if not records or (failed and len(assessed)==1):
        rng=Random(281+len(records))
        actions=[{'kind':'trajectory', 'channels':['meter'],
                  'sequence':[{'u':rng.uniform(-1.,1.)} for _ in range(24)]} for _ in range(3)]
        return {'actions':actions,'after_observation':[
            {'operator':'assess_instrument','instrument':roles['instrument'],'records':'acquired'}]}
    latest=state['operator_results'][-1]
    if latest['operator']=='assess_instrument':
        supported='meter' in latest['supported_fields']
        if supported and not failed:
            return {'model':roles['predictor']}
        steps=[]
        instrument=roles['instrument']
        if not supported:
            instrument='shared/current_instrument.json'
            steps.append({'operator':'instrument','path':instrument,'records':latest['records']})
        steps.append({'operator':'response','instrument':instrument,'records':latest['records'],
                      'output_dir':'shared/selective_response'})
        return {'actions':[],'after_observation':steps}
    return {'model':min(latest['candidates'],key=lambda c:c['validation_mse'])['path']}
"""


def settings(change):
    return {"task_index": 1, "apparatus_seed": 99106007, "instrument_mode": "recurrent",
            "calibration_change": {"after_rows": 72,
                "apparatus_seed": 99306197 if change == "instrument" else 99106007,
                "response_seed": 99306197 if change == "response" else 99106007}}


@pytest.fixture
def real_repairs(tmp_path, calibrated_knowledge):
    runs = {}
    for change in ("response", "instrument"):
        agent = make_frozen(tmp_path / change, calibrated_knowledge, SCRIPTED_SELECTIVE)
        result = agent.execute_method(TemporalInquiryLab(**settings(change)))
        assert result["status"] == "completed" and result["fresh_confirmation"]["passed"]
        runs[change] = result
    return runs


def test_paid_support_and_refutation_produce_distinct_confirmed_paths(real_repairs):
    retained = selective_revision_evidence(real_repairs["response"], "instrument.json")
    reopened = selective_revision_evidence(real_repairs["instrument"], "instrument.json")
    assert retained["retained_instrument"] and not retained["reopened_instrument"]
    assert reopened["reopened_instrument"] and not reopened["retained_instrument"]
    for result in (retained, reopened):
        assert result["parent_failure_steps"] == [1]
        assert result["post_failure_assessments"][0]["records"] == [5, 6, 7]
        assert result["reason"] == "confirmed_selective_repair"


def test_old_support_refitting_and_unconfirmed_or_unselected_candidates_do_not_count(real_repairs):
    result = real_repairs["response"]
    # Failed-confirmation records have responses but no meter readings. Selecting
    # them alongside current meter records does not make meter evidence stale.
    mixed = deepcopy(result)
    mixed["method_results"][2]["feedback"]["after_observation_results"][0]["records"] = [3, 4, 5, 6, 7]
    assert selective_revision_evidence(mixed, "instrument.json")["retained_instrument"]
    stale = deepcopy(result)
    assessment = stale["method_results"][2]["feedback"]["after_observation_results"][0]
    assessment["records"] = [0, 1, 2]
    assert not selective_revision_evidence(stale, "instrument.json")["retained_instrument"]
    stale = deepcopy(result)
    # This is an instrument-fitting operation on the ordinary inherited path,
    # a reachable path overwrite, not merely assessment metadata changing.
    stale["method_results"][2]["feedback"]["after_observation_results"].insert(0,
        {"operator": "instrument", "path": "instrument.json", "records": [5, 6, 7]})
    assert not selective_revision_evidence(stale, "instrument.json")["retained_instrument"]
    failed = deepcopy(real_repairs["instrument"])
    failed["status"] = "confirmation_budget_exhausted"
    failed["fresh_confirmation"]["passed"] = False
    assert not selective_revision_evidence(failed, "instrument.json")["reopened_instrument"]
    unrelated = deepcopy(result)
    unrelated["model"] = "shared/other.py"
    assert not selective_revision_evidence(unrelated, "instrument.json")["retained_instrument"]


def test_precomputed_response_can_be_selected_after_assessment_and_intermediate_failure(real_repairs):
    result = deepcopy(real_repairs["response"])
    # An assessment and a response fit may be composed in one linear continuation;
    # the program chooses the exported candidate on the next invocation.
    result["method_results"][2]["feedback"]["after_observation_results"].extend(
        result["method_results"].pop(3)["feedback"]["after_observation_results"])
    assert selective_revision_evidence(result, "instrument.json")["retained_instrument"]
    # A rejected response candidate does not un-settle the same physical apparatus.
    failed = deepcopy(result["method_results"][1])
    failed["feedback"]["fresh_confirmation"]["observation_indices"] = list(range(192, 216))
    result["method_results"].insert(3, failed)
    assert selective_revision_evidence(result, "instrument.json")["retained_instrument"]


def test_current_source_requires_selective_paths_and_prior_prediction_reuse(tmp_path, calibrated_knowledge):
    knowledge = deepcopy(calibrated_knowledge)
    knowledge["files"][METHOD_PATH] = SCRIPTED_SELECTIVE
    cases = {"alpha": {**settings("response"), "confirmation_seed": 7001},
             "beta": {**settings("instrument"), "confirmation_seed": 7002},
             "gamma": {**settings("stable"), "confirmation_seed": 7003}}
    learner = SelectiveMethodLearner(None, tmp_path / "author", knowledge=knowledge, cases=cases,
        lab_factory=lambda case: TemporalInquiryLab(**{key: value for key, value in case.items()
                                                       if key != "confirmation_seed"}),
        config=InquiryLLMConfig(observation_budget=1200, confirmation_rows=0,
                               execution_seconds=60, compute_budget=300000))
    for case in cases:
        feedback = learner._practice(case)
        assert feedback["status"] == "completed"
    coverage = feedback["current_source_coverage"]
    assert coverage["retained_instrument_cases"] == ["alpha"]
    assert coverage["reopened_instrument_cases"] == ["beta"]
    assert coverage["prior_reuse_cases"] == ["gamma"]
    assert not coverage["missing_cases"]
    # Predictive success alone cannot replace evidence of instrument retention.
    saved = learner.practice_trials[0]["selective_revision"]
    learner.practice_trials[0]["selective_revision"] = {}
    rejected = learner._execute("finish", {"path": METHOD_PATH}, learner.interface)
    assert "selective repair" in rejected["error"] and learner.model is None
    learner.practice_trials[0]["selective_revision"] = saved
    published = learner._execute("finish", {"path": METHOD_PATH}, learner.interface)
    assert published["published"] == METHOD_PATH
    learner._execute("write", {"path": METHOD_PATH, "code": SCRIPTED_SELECTIVE + "\n# new edit\n"},
                     learner.interface)
    coverage = learner._current_source_coverage()
    assert coverage["missing_cases"] == sorted(cases)
    assert not coverage["retained_instrument_cases"] and not coverage["reopened_instrument_cases"]


def test_preceding_published_refit_both_program_does_not_count_as_selective():
    path = Path(__file__).resolve().parents[1] / (
        "artifacts/temporal_evidence_scope_learning_luna_20261006/study/revised/episode_02/training.json")
    if not path.exists():
        pytest.skip("historical research artifact is not included in this checkout")
    result = json.loads(path.read_text())["tasks"][0]
    assert result["status"] == "completed" and result["fresh_confirmation"]["passed"]
    evidence = selective_revision_evidence(result, "shared/meter.json")
    assert evidence["parent_failure_steps"]
    assert not evidence["retained_instrument"] and not evidence["reopened_instrument"]
    assert not evidence["post_failure_assessments"]
