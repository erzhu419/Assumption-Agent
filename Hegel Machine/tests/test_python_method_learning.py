"""Archive-based method practice uses real trials and exact current-source coverage."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.code_agent import archive_confirmed_python_model
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.python_method_learning import (
    PreparedPythonMethodAgent, PythonArchiveMethodLearner, SEED_PYTHON_METHOD,
)
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.temporal_paired_method import ProcedureConsumerAgent
from scripts.run_candidate_retention_cycle import task_for
from scripts.run_candidate_retention_cycle_llm import BANK_PATH, prepare_episode
from test_python_research_procedure import configuration, purchase


ADAPTIVE_METHOD = '''import json
def select_actions(spec, records, memory):
    state = json.load(open('research_state.json'))
    if not state['method_results']:
        return {'action': 'evaluate', 'arguments': {'model': memory['research_roles']['predictor']}}
    return state.get('research_advice', {}).get('next_action') or {'stop': 'unresolved'}
'''

DEPENDENCY_EDIT_METHOD = '''import json
from pathlib import Path
def select_actions(spec, records, memory):
    state = json.load(open('research_state.json'))
    steps = state['method_results']
    prior = memory['research_roles']['predictor']
    if not steps:
        return {'action': 'write', 'arguments': {
            'path': str(Path(prior).parent / 'basis_config.py'), 'code': 'FACTOR = 2.0\\n'}}
    if len(steps) == 1:
        return {'action': 'evaluate', 'arguments': {'model': prior}}
    return state.get('research_advice', {}).get('next_action') or {'stop': 'unresolved'}
'''


@pytest.fixture
def archived_predictor(tmp_path):
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061,
        response_seed=910061, instrument_mode="finite2")
    agent = ProcedureConsumerAgent(None, tmp_path / "source", arm="code_react",
        source_task=False, config=configuration())
    purchase(agent, lab)
    learned = agent._execute("run", {"operator": "history", "base_lags": 3,
        "path": "model.py"}, lab)
    source = (agent.workspace / learned["path"]).read_text()
    # This imported factor really participates in both fitting and prediction.
    # Changing the feature units preserves the family, but changes its context.
    agent._save("basis_config.py", "FACTOR = 1.0\n")
    agent._save("model.py", source + "\nimport basis_config\n"
        "_unscaled_basis = _basis\n"
        "def _basis(row):\n return basis_config.FACTOR * _unscaled_basis(row)\n")
    submitted = agent._execute("finish", {"model": "model.py"}, lab)
    assert submitted["fresh_confirmation"]["passed"]
    archive = archive_confirmed_python_model(agent, tmp_path / "archive" / "_evaluated_model")
    return {**archive, "candidate_id": "prior", "model":
        str(Path(BANK_PATH) / "prior" / "_evaluated_model" / archive["inner_model"])}


def make_learner(tmp_path, candidate, *, source=SEED_PYTHON_METHOD, config=None):
    base = {"seed": 71, "apparatus_seed": 910061, "changed_response_seed": 910079}
    cases = {"case_a": {**base, "position": 0}, "case_b": {**base, "position": 1}}
    for case in cases.values():
        case["confirmation_seed"] = task_for(case, case["position"])["seed"] + 7919
    prepared = []

    def laboratory(case):
        return TemporalInquiryLab(**task_for(case, case["position"]))

    def prepare(case, path, method_source, limits):
        plan = {"limits": asdict(limits),
                "prefix": {"trajectories": 3, "length": 24, "channels": []}}
        payload = prepare_episode(case, case["position"], path, [candidate], plan,
            laboratory=laboratory(case), agent_class=PreparedPythonMethodAgent)
        payload["agent"].knowledge["research_roles"] = {"predictor": candidate["model"]}
        payload["agent"].bind_method(method_source)
        prepared.append(payload)
        return payload

    learner = PythonArchiveMethodLearner(None, tmp_path / "learner", cases=cases,
        knowledge={"files": {METHOD_PATH: source},
            "research_roles": {"predictor": candidate["model"]}},
        lab_factory=laboratory, prepare_trial=prepare,
        config=config or InquiryLLMConfig(observation_budget=768, compute_budget=1000000,
            execution_seconds=120., confirmation_rows=24, max_model_calls=12))
    return learner, prepared


def practice(learner, case):
    return learner._execute("run", {"operator": "practice", "case": case}, learner.interface)


def publish(learner):
    return learner._execute("finish", {"path": METHOD_PATH}, learner.interface)


def test_seed_reuses_original_archive_but_changed_case_fails_and_keeps_all_paid_costs(tmp_path, archived_predictor):
    learner, prepared = make_learner(tmp_path, archived_predictor)
    stable = practice(learner, "case_a")
    changed = practice(learner, "case_b")
    assert stable["status"] == "completed" and stable["selected_prior_predictor"]
    assert stable["fresh_confirmation"]["passed"]
    assert changed["status"] == "method_no_progress" and changed["fresh_confirmation"] is None
    failed_submissions = [step["feedback"] for step in changed["steps"][1:]]
    assert len(failed_submissions) == 2
    assert all(step["mse"] > 1e-4 and "paid evidence contradicts this model" in step["error"]
               and "fresh_confirmation" not in step for step in failed_submissions)
    assert changed["no_progress"]["kind"] == "repeated_execution_error"
    assert changed["current_source_coverage"] == {"confirmed_cases": ["case_a"],
        "missing_cases": ["case_b"], "prior_reuse_cases": ["case_a"], "revised_cases": []}
    for result, payload, observation_cost, compute_cost in zip(
            (stable, changed), prepared, (96, 72), (168, 216)):
        assert payload["prefix_costs"]["observation_cost"] == 72
        assert result["costs"]["observation_cost"] == observation_cost
        assert result["costs"]["compute_cost"] == compute_cost
        assert result["costs"]["observation_cost"] == payload["laboratory"].costs["observation_cost"]
        assert result["costs"]["model_calls"] == result["costs"]["reported_tokens"] == 0
        assert len(payload["agent"].rows) == observation_cost
    assert not prepared[1]["agent"].confirmations
    for resource in ("observation_cost", "measurement_cost", "compute_cost", "python_execution_seconds"):
        assert learner.costs[resource] == stable["costs"][resource] + changed["costs"][resource]
    refused = publish(learner)
    assert refused["unconfirmed_cases"] == ["case_b"] and learner.model is None
    assert [item["case"] for item in refused["publication_practices"]] == ["case_b"]
    retried = refused["publication_practices"][0]
    assert retried["fresh_confirmation"] is None and retried["status"] == "method_no_progress"
    assert retried["costs"]["observation_cost"] == 72 and retried["costs"]["compute_cost"] == 216
    assert len(prepared) == 3  # Explicit submission repays only the unsupported case.
    assert learner.costs["observation_cost"] == 240 and learner.costs["compute_cost"] == 600
    for resource in ("observation_cost", "measurement_cost", "compute_cost", "python_execution_seconds"):
        assert learner.costs[resource] == sum(trial["costs"][resource] for trial in learner.practice_trials)


def test_repeated_case_keeps_the_same_prefix_but_buys_distinct_fresh_confirmations(tmp_path, archived_predictor):
    learner, prepared = make_learner(tmp_path, archived_predictor)
    first = practice(learner, "case_a")
    before = deepcopy(learner.costs)
    second = practice(learner, "case_a")
    assert first["status"] == second["status"] == "completed"
    assert first["fresh_confirmation"]["passed"] and second["fresh_confirmation"]["passed"]
    original, repeated = [item["agent"] for item in prepared]
    assert original.observations[:3] == repeated.observations[:3]
    first_fresh = [record["action"]["sequence"]
                   for record in original.confirmations[-1]["observations"]]
    second_fresh = [record["action"]["sequence"]
                    for record in repeated.confirmations[-1]["observations"]]
    assert first_fresh != second_fresh
    assert first["costs"]["observation_cost"] == second["costs"]["observation_cost"] == 96
    for resource in ("observation_cost", "measurement_cost", "compute_cost", "python_execution_seconds"):
        assert learner.costs[resource] == before[resource] + second["costs"][resource]
    assert len(learner.practice_trials) == 2


def test_current_method_coverage_requires_actual_reuse_revision_and_invalidates_an_unexecuted_edit(
        tmp_path, archived_predictor):
    learner, prepared = make_learner(tmp_path, archived_predictor, source=ADAPTIVE_METHOD)
    results = [practice(learner, case) for case in ("case_a", "case_b")]
    assert all(result["status"] == "completed" and result["fresh_confirmation"]["passed"]
               for result in results)
    assert results[0]["model"].startswith("_evaluated_model/")
    assert results[0]["selected_prior_predictor"] is True  # A retained snapshot is still the old context.
    assert results[1]["selected_prior_predictor"] is False
    assert learner._current_source_coverage() == {"confirmed_cases": ["case_a", "case_b"],
        "missing_cases": [], "prior_reuse_cases": ["case_a"], "revised_cases": ["case_b"]}
    revised_source = ADAPTIVE_METHOD + "\n# An initially unexecuted edit.\n"
    edited = learner._execute("write", {"path": METHOD_PATH,
        "code": revised_source}, learner.interface)
    assert edited["current_source_coverage"]["confirmed_cases"] == []
    assert len(prepared) == 2
    submitted = publish(learner)
    assert submitted["published"] == METHOD_PATH
    assert [item["case"] for item in submitted["publication_practices"]] == ["case_a", "case_b"]
    assert len(prepared) == 4 and learner.published_source == revised_source
    before = deepcopy(learner.costs)
    assert publish(learner)["publication_practices"] == []
    assert learner.costs == before


def test_failed_publication_returns_first_actual_trace_then_explicit_edit_can_publish(
        tmp_path, archived_predictor):
    learner, prepared = make_learner(tmp_path, archived_predictor)
    learner.cases = {"case_b": learner.cases["case_b"], "case_a": learner.cases["case_a"]}
    refused = publish(learner)
    assert len(prepared) == 1 and learner.model is learner.published_source is None
    assert [item["case"] for item in refused["publication_practices"]] == ["case_b"]
    failed = refused["publication_practices"][0]
    assert failed["status"] == "method_no_progress" and failed["fresh_confirmation"] is None
    assert failed["costs"]["observation_cost"] == 72 and failed["costs"]["compute_cost"] == 216
    assert len(failed["steps"]) == 3
    for step in failed["steps"][1:]:
        assert step["feedback"]["mse"] > 1e-4
        assert "paid evidence contradicts this model" in step["feedback"]["error"]
        assert "fresh_confirmation" not in step["feedback"]
    assert not prepared[0]["agent"].confirmations
    assert refused["unconfirmed_cases"] == ["case_a", "case_b"]
    old_source = (learner.workspace / METHOD_PATH).read_text()
    assert old_source == SEED_PYTHON_METHOD
    learner._execute("write", {"path": METHOD_PATH, "code": ADAPTIVE_METHOD}, learner.interface)
    assert len(prepared) == 1  # Only the author-selected submit triggers this workflow.
    completed = publish(learner)
    assert completed["published"] == METHOD_PATH
    assert [item["case"] for item in completed["publication_practices"]] == ["case_b", "case_a"]
    assert [trial["status"] for trial in learner.practice_trials] == ["method_no_progress", "completed", "completed"]
    assert completed["current_source_coverage"]["prior_reuse_cases"] == ["case_a"]
    assert completed["current_source_coverage"]["revised_cases"] == ["case_b"]
    for resource in ("observation_cost", "measurement_cost", "compute_cost", "python_execution_seconds"):
        assert learner.costs[resource] == sum(trial["costs"][resource] for trial in learner.practice_trials)
    assert learner.costs["model_calls"] == learner.costs["reported_tokens"] == 0


def test_publication_retains_paid_success_when_next_case_cannot_be_afforded(tmp_path, archived_predictor):
    limits = InquiryLLMConfig(observation_budget=96, compute_budget=1000000,
        execution_seconds=120., confirmation_rows=24)
    learner, prepared = make_learner(tmp_path, archived_predictor, source=ADAPTIVE_METHOD, config=limits)
    with pytest.raises(ValueError, match="finish needs path"):
        learner._execute("finish", {}, learner.interface)
    assert not prepared and learner.costs["observation_cost"] == 0
    result = publish(learner)
    assert learner.model is learner.published_source is None and len(prepared) == 1
    assert result["current_source_coverage"]["confirmed_cases"] == ["case_a"]
    assert result["unconfirmed_cases"] == ["case_b"]
    assert result["publication_practices"][0]["fresh_confirmation"]["passed"]
    assert "remaining resources" in result["publication_practices"][1]["error"]
    assert learner.costs["observation_cost"] == 96


def test_successful_practices_still_require_actual_unchanged_prior_reuse(tmp_path, archived_predictor):
    learner, prepared = make_learner(tmp_path, archived_predictor, source=DEPENDENCY_EDIT_METHOD)
    result = publish(learner)
    assert len(prepared) == 2
    assert all(item["fresh_confirmation"]["passed"] for item in result["publication_practices"])
    assert result["confirmed_prior_reuse_cases"] == []
    assert result["confirmed_revision_cases"] == ["case_a", "case_b"]
    assert learner.model is learner.published_source is None


def test_changed_imported_dependency_is_not_unchanged_prior_reuse_even_when_predictions_pass(
        tmp_path, archived_predictor):
    learner, prepared = make_learner(tmp_path, archived_predictor, source=DEPENDENCY_EDIT_METHOD)
    result = practice(learner, "case_a")
    assert result["status"] == "completed" and result["fresh_confirmation"]["passed"]
    assert result["selected_prior_predictor"] is False
    agent = prepared[0]["agent"]
    original = Path(archived_predictor["snapshot_workspace"])
    assert (agent.workspace / result["model"]).read_bytes() == (original / archived_predictor["inner_model"]).read_bytes()
    assert (agent.workspace / "_evaluated_model" / "basis_config.py").read_text() == "FACTOR = 2.0\n"
    assert (original / "basis_config.py").read_text() == "FACTOR = 1.0\n"
    coverage = learner._current_source_coverage()
    assert coverage["prior_reuse_cases"] == [] and coverage["revised_cases"] == ["case_a"]


def test_practice_trace_matches_program_state_and_preserves_failed_confirmation_events(
        tmp_path, archived_predictor):
    source = '''import json
from pathlib import Path
def select_actions(spec, records, memory):
    steps = json.load(open('research_state.json'))['method_results']
    Path('seen_method_results.json').write_text(json.dumps(steps))
    if not steps:
        return {'model': memory['research_roles']['predictor']}
    return {'stop': 'inspect the actual failed confirmation'}
'''
    learner, prepared = make_learner(tmp_path, archived_predictor, source=source)
    feedback = practice(learner, "case_b")
    agent = prepared[0]["agent"]
    assert feedback["status"] == "method_stopped"
    assert feedback["fresh_confirmation"] is None and not agent.confirmations
    assert feedback["costs"]["observation_cost"] == 72 and feedback["costs"]["compute_cost"] == 72
    assert len(agent.rows) == 72 and len(agent.observations) == 3
    assert len(feedback["steps"]) == len(agent.method_results) == 2
    observed = json.loads((agent.workspace / "seen_method_results.json").read_text())
    state = json.loads((agent.workspace / "research_state.json").read_text())
    recorded_steps = json.loads(json.dumps(agent.method_results))
    assert observed == state["method_results"] == recorded_steps[:-1]
    step = feedback["steps"][0]
    assert set(step) == {"step", "path", "proposal", "feedback", "evidence_before", "evidence_after"}
    assert step["proposal"] == {"model": archived_predictor["model"]}
    assert step["evidence_before"] == step["evidence_after"] == {"records": 3, "rows": 72}
    assert step["feedback"]["mse"] > 1e-4
    assert "paid evidence contradicts this model" in step["feedback"]["error"]
    assert "fresh_confirmation" not in step["feedback"]
    assert step["feedback"]["paid_residuals"]["evaluated_rows"] == 72

    persisted = json.loads((agent.workspace / "result.json").read_text())
    assert persisted["method_results"] == recorded_steps
    assert learner.workspace / feedback["full_trace_file"] == agent.workspace / "result.json"
    for compact, original in zip(feedback["steps"], persisted["method_results"]):
        for key in ("step", "path", "proposal", "evidence_before", "evidence_after"):
            assert compact[key] == original[key]
        for key in ("error", "mse", "status", "reason"):
            assert compact["feedback"].get(key) == original["feedback"].get(key)
    assert persisted["method_results"][0]["feedback"]["paid_residuals"]["examples"]
    assert persisted["events"] == json.loads(json.dumps(agent.events))
    event = persisted["events"][0]
    assert event["action"] == "query" and event["arguments"] == {"path": METHOD_PATH}
    assert event["feedback"]["error"] == step["feedback"]["error"]
    assert "fresh_confirmation" not in event["feedback"]
    assert event["usage"] == {"backend": "frozen_program_execution", "total_tokens": 0}


@pytest.mark.parametrize("resource", ["observation_budget", "compute_budget"])
def test_insufficient_prefix_and_confirmation_budget_does_not_prepare_a_trial(tmp_path, archived_predictor, resource):
    limits = replace(InquiryLLMConfig(observation_budget=768, compute_budget=1000000,
        execution_seconds=120., confirmation_rows=24), **{resource: 95})
    learner, prepared = make_learner(tmp_path, archived_predictor, config=limits)
    before = deepcopy(learner.costs)
    result = practice(learner, "case_a")
    assert result["error"] and not prepared and not learner.practice_trials
    assert learner.costs == before
    assert not (learner.workspace / "trials").exists()
