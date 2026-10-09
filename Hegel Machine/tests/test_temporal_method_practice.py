"""Scripted paid practice tests; no external-model performance evidence."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.history_revision import revise_history
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_instrument_research import fit_temporal_instrument
from hegel_machine.cognition.temporal_conditional_method import ConditionalMethodPracticeAgent
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH, method_memory_pair
from hegel_machine.cognition.temporal_method_practice import MethodPracticeAgent


def procedure(*, meter=False, callbacks=None, revision="seed"):
    return "\n".join([
        "from random import Random",
        "def select_actions(spec, observations, memory):",
        "    rng = Random(71 + len(observations))",
        "    actions = [{'kind':'trajectory','sequence':[{'u':rng.uniform(-1.,1.)}",
        "                for _ in range(spec['max_probe_length'])],",
        f"                'channels':{['meter'] if meter else []!r}}} for _ in range(2)]",
        f"    return {{'actions':actions,'after_observation':{callbacks or []!r},",
        f"            'question':{{'id':{revision!r},'question':'Which measured state predicts the response?',",
        "                        'parent_id':'response','depends_on':[]}}",
    ]) + "\n"


SEED_SOURCE = procedure()
REVISED_SOURCE = procedure(meter=True, callbacks=[
    {"operator": "instrument", "path": "instrument.json"},
    {"operator": "response", "instrument": "instrument.json", "output_dir": "shared/practiced"},
], revision="revised")


@pytest.fixture
def original_knowledge():
    lab = TemporalInquiryLab(task_index=0, apparatus_seed=910061, instrument_mode="recurrent")
    rng = Random(73)
    records = [lab.execute({"kind": "trajectory", "sequence": [{"u": rng.uniform(-1., 1.)}
        for _ in range(3)], "channels": ["meter"]}) for _ in range(12)]
    instrument = fit_temporal_instrument(records, lab.public_spec())
    model = revise_history([row for record in records for row in record["rows"]]).to_code()
    manifest = {"hypotheses": [{"id": "old_response", "path": "shared/model.py"}],
                "research_programs": [{"id": "old_plan", "path": "shared/old_plan.py"}],
                "support_questions": []}
    return {"files": {"shared/model.py": model, "instrument.json": json.dumps(instrument),
                       "shared/old_plan.py": SEED_SOURCE,
                       "shared/knowledge.json": json.dumps(manifest)},
            "manifest": manifest, "lessons": ["Old short-scope fit."]}


def seed_memory(original):
    memory = deepcopy(original)
    memory["files"][METHOD_PATH] = SEED_SOURCE
    memory["manifest"]["research_programs"].append({"id": "practice_method", "path": METHOD_PATH})
    memory["files"]["shared/knowledge.json"] = json.dumps(memory["manifest"])
    return memory


def make_agent(tmp_path, original, provider=None):
    agent = MethodPracticeAgent(provider, tmp_path, knowledge=seed_memory(original),
        config=InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
            compute_budget=100000, execution_seconds=30., max_model_calls=8, token_budget=1000),
        confirmation_seed=9173)
    lab = TemporalInquiryLab(task_index=2, apparatus_seed=99106007, instrument_mode="recurrent")
    return agent, lab


def write_method(agent, lab, source):
    feedback = agent._execute("write", {"path": METHOD_PATH, "code": source}, lab)
    assert "error" not in feedback


def paid_trial(agent, lab):
    feedback = agent._execute("query", {"path": METHOD_PATH}, lab)
    assert "error" not in feedback
    assert not feedback.get("returncode"), feedback
    return feedback


def candidate_from(feedback):
    assert not feedback.get("after_observation_error")
    candidates = feedback["after_observation_results"][-1]["candidates"]
    candidate = min(candidates, key=lambda item: item["validation_mse"])
    assert candidate["validation_mse"] < 1e-10
    return candidate["path"]


def manifest_for(memory, model):
    manifest = deepcopy(memory["manifest"])
    manifest["hypotheses"].append({"id": "practiced_response", "path": model})
    return manifest


def finish_candidate(agent, lab, candidate):
    manifest = manifest_for(agent.knowledge, candidate)
    written = agent._execute("write", {"path": "shared/knowledge.json", "code": json.dumps(manifest)}, lab)
    assert "error" not in written
    return agent._execute("finish", {"model": candidate, "manifest": "shared/knowledge.json",
                                      "lesson": "Recheck the instrument with long measured probes."}, lab)


class PracticingProvider:
    candidate = None

    def next_action(self, request):
        step = len(request["events"])
        if step in (0, 3):
            action, arguments = "query", {"path": METHOD_PATH}
        elif step == 1:
            action, arguments = "finish", {"model": "shared/model.py"}
        elif step == 2:
            assert request["events"][-1]["feedback"]["error"]
            action, arguments = "write", {"path": METHOD_PATH, "code": REVISED_SOURCE}
        elif step == 4:
            self.candidate = candidate_from(request["events"][-1]["feedback"])
            action, arguments = "write", {"path": "shared/knowledge.json", "code": json.dumps(
                manifest_for(request["session_knowledge"], self.candidate))}
        else:
            action, arguments = "finish", {"model": self.candidate, "lesson": "Use measured long probes."}
        return {"action": action, "arguments": arguments,
                "usage": {"backend": "scripted_interface_test", "total_tokens": 1}}


def test_paid_seed_then_revised_method_confirms_and_exports_only_method(original_knowledge, tmp_path):
    agent, lab = make_agent(tmp_path, original_knowledge, PracticingProvider())
    result = agent.run(lab)
    assert result["status"] == "completed"
    assert result["fresh_confirmation"]["passed"]
    assert result["costs"]["observation_cost"] == 144  # 48 seed + 48 revision + 48 fresh
    assert result["costs"]["measurement_cost"] == 144
    assert result["method_source"] == REVISED_SOURCE
    assert result["method_path"] == METHOD_PATH
    assert len(result["method_trials"]) == 2
    assert not result["method_trials"][0]["exported_candidates"]
    assert result["method_trials"][1]["exported_candidates"]

    retained, deleted = method_memory_pair(original_knowledge, result)
    for memory in (retained, deleted):
        assert memory["files"]["instrument.json"] == original_knowledge["files"]["instrument.json"]
        assert memory["files"]["shared/model.py"] == original_knowledge["files"]["shared/model.py"]
        assert not any(path.startswith("shared/practiced/") for path in memory["files"])
    assert retained["files"][METHOD_PATH] == REVISED_SOURCE
    assert METHOD_PATH not in deleted["files"]
    assert retained["lessons"] == deleted["lessons"]


def test_failed_callback_keeps_paid_trial_and_can_be_revised(original_knowledge, tmp_path):
    agent, lab = make_agent(tmp_path, original_knowledge)
    paid_trial(agent, lab)
    broken = procedure(meter=True, callbacks=[
        {"operator": "instrument", "path": "instrument.json"},
        {"operator": "response", "instrument": "missing.json"},
    ], revision="broken")
    write_method(agent, lab, broken)
    failed = paid_trial(agent, lab)
    assert failed["after_observation_error"]
    assert failed["after_observation_results"][0]["operator"] == "instrument"
    assert not failed["after_observation_results"][0].get("error")
    assert lab.costs["observation_cost"] == 96
    assert len(agent.rows) == 96
    assert len(agent.result("researching")["method_trials"]) == 2
    assert agent._execute("finish", {"model": "shared/model.py"}, lab)["error"]
    assert lab.costs["observation_cost"] == 96

    write_method(agent, lab, REVISED_SOURCE)
    candidate = candidate_from(paid_trial(agent, lab))
    finished = finish_candidate(agent, lab, candidate)
    assert finished["fresh_confirmation"]["passed"]
    assert lab.costs["observation_cost"] == 192


def test_manual_fits_and_query_argument_callbacks_cannot_replace_method_trial(original_knowledge, tmp_path):
    agent, lab = make_agent(tmp_path, original_knowledge)
    paid_trial(agent, lab)
    write_method(agent, lab, procedure(meter=True, revision="acquisition_only"))
    before = deepcopy(lab.costs)
    denied = agent._execute("query", {"path": METHOD_PATH,
        "after_observation": [{"operator": "history"}]}, lab)
    assert denied["error"] and lab.costs == before
    paid_trial(agent, lab)
    instrument = agent._execute("run", {"operator": "instrument", "path": "instrument.json"}, lab)
    assert "error" not in instrument
    response = agent._execute("run", {"operator": "response", "instrument": "instrument.json"}, lab)
    candidate = min(response["candidates"], key=lambda item: item["validation_mse"])
    assert candidate["validation_mse"] < 1e-10
    assert not agent.result("researching")["method_trials"][-1]["exported_candidates"]
    before = deepcopy(lab.costs)
    refused = finish_candidate(agent, lab, candidate["path"])
    assert refused["error"] and lab.costs == before
    assert agent.result("researching")["method_path"] is None


def test_unexecuted_final_edit_requires_new_paid_trial(original_knowledge, tmp_path):
    agent, lab = make_agent(tmp_path, original_knowledge)
    paid_trial(agent, lab)
    write_method(agent, lab, REVISED_SOURCE)
    candidate = candidate_from(paid_trial(agent, lab))
    final_source = REVISED_SOURCE + "# Record this revision for the next research task.\n"
    write_method(agent, lab, final_source)
    before = deepcopy(lab.costs)
    refused = finish_candidate(agent, lab, candidate)
    assert refused["error"] and lab.costs == before
    assert agent.result("researching")["method_path"] is None

    candidate = candidate_from(paid_trial(agent, lab))
    finished = finish_candidate(agent, lab, candidate)
    assert finished["fresh_confirmation"]["passed"]
    assert agent.result("completed")["method_source"] == final_source
    assert lab.costs["observation_cost"] == 192


def test_practice_study_restores_predictive_memory_before_four_independent_transfers(
        original_knowledge, tmp_path, monkeypatch):
    from scripts import run_temporal_method_learning as transfers
    from scripts.run_temporal_method_practice import run_practice_study

    costs = {"observation_cost": 36, "measurement_cost": 108, "compute_cost": 100,
             "python_execution_seconds": .1, "model_calls": 0, "reported_tokens": 0}
    experience = {"agent_arm": "hegel", "status": "completed", "task_position": 1,
                  "paid_observations": [], "events": [], "outgoing_knowledge": original_knowledge,
                  "costs": costs}
    seed = {"status": "completed", "method_path": METHOD_PATH, "method_source": SEED_SOURCE,
            "method_entry": {"id": "practice_method", "path": METHOD_PATH},
            "lesson": "Acquire longer trajectories.",
            "costs": {**dict.fromkeys(costs, 0), "model_calls": 2, "reported_tokens": 2}}
    called, evaluated = [], []

    class StoppedConsumer:
        def __init__(self, path):
            self.path = path

        def next_action(self, request):
            memory = request["session_knowledge"]
            for name in ("instrument.json", "shared/model.py", "shared/old_plan.py"):
                assert memory["files"][name] == original_knowledge["files"][name]
            assert not any(name.startswith("shared/practiced/") for name in memory["files"])
            assert (METHOD_PATH in memory["files"]) == (self.path.name == "method_retained")
            called.append(str(self.path))
            return {"action": "provider_error", "arguments": {},
                    "error": {"status": 503, "message": "scripted provider unavailable"},
                    "usage": {"backend": "scripted_interface_test", "total_tokens": 2}}

    def provider(path):
        return PracticingProvider() if path.name == "practice" else StoppedConsumer(path)

    actual_evaluate = transfers.evaluate_chain

    def after_all_transfers(chain, tasks, **kwargs):
        assert len(called) == 4
        assert len(list(tmp_path.glob("changed_*/*/training.json"))) == 4
        evaluated.append((chain["case_id"], chain["condition"]))
        return actual_evaluate(chain, tasks, **kwargs)

    monkeypatch.setattr(transfers, "evaluate_chain", after_all_transfers)
    summary = run_practice_study(experience, seed, tmp_path, provider_factory=provider,
                                 practice_seed=99106007, transfer_seed=99406007, workers=2)
    assert summary["status"] == "completed"
    assert summary["evidence_kind"] == "interface_test_not_llm_performance"
    assert len(evaluated) == 4
    assert summary["common_learning_costs"]["practice"]["observation_cost"] == 144
    for condition in transfers.CONDITIONS:
        result = summary["conditions"][condition]
        assert result["statuses"] == ["provider_failed", "provider_failed"]
        assert result["registered_tasks"] == result["attempted_tasks"] == 2
        assert result["confirmed_and_correct"] == 0
        assert result["costs_including_common_learning"]["observation_cost"] == 180


def test_trial_feedback_separates_program_edits_from_new_and_repeated_actions(
        original_knowledge, tmp_path):
    first_action = {"kind": "trajectory", "sequence": [{"u": .8}, {"u": 0.}, {"u": -.3}],
                    "channels": []}
    second_action = {"kind": "trajectory", "sequence": [{"u": -.5}, {"u": .2}, {"u": .9}],
                     "channels": []}

    def fixed_program(actions):
        payload = {"actions": actions, "question": {"id": "fixed",
            "question": "What did this experiment reveal?", "parent_id": None, "depends_on": []}}
        return ("def select_actions(spec, observations, memory):\n"
                f"    return {payload!r}\n")

    initial_source = fixed_program([first_action, first_action])
    memory = seed_memory(original_knowledge)
    memory["files"][METHOD_PATH] = initial_source
    agent = MethodPracticeAgent(None, tmp_path, knowledge=memory,
        config=InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
                               execution_seconds=20., compute_budget=100000))
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=99106007, instrument_mode="recurrent")
    initial = paid_trial(agent, lab)["method_trial_feedback"]
    assert initial["program_changed_since_previous_trial"] is None
    assert initial["new_action_count"] == initial["repeated_action_count"] == 1

    repeated = paid_trial(agent, lab)["method_trial_feedback"]
    assert repeated["program_changed_since_previous_trial"] is False
    assert repeated["new_action_count"] == 0 and repeated["repeated_action_count"] == 2

    write_method(agent, lab, initial_source + "# The procedure text changed; its actions did not.\n")
    text_only = paid_trial(agent, lab)["method_trial_feedback"]
    assert text_only["program_changed_since_previous_trial"] is True
    assert text_only["new_action_count"] == 0 and text_only["repeated_action_count"] == 2

    write_method(agent, lab, fixed_program([second_action, first_action, second_action]))
    changed = paid_trial(agent, lab)["method_trial_feedback"]
    assert changed["program_changed_since_previous_trial"] is True
    assert changed["new_action_count"] == 1 and changed["repeated_action_count"] == 2
    assert lab.costs["observation_cost"] == 27  # repetitions remain paid, permitted experiments
    assert len(agent.method_trials) == 4
    for trial, feedback in zip(agent.method_trials, [initial, repeated, text_only, changed]):
        for key in ("program_changed_since_previous_trial", "new_action_count", "repeated_action_count"):
            assert trial[key] == feedback[key]


def test_analysis_only_revision_exports_and_confirms_from_existing_paid_evidence(
        original_knowledge, tmp_path):
    agent, _ = make_agent(tmp_path, original_knowledge)
    lab = TemporalInquiryLab(task_index=2, apparatus_seed=99106007, instrument_mode="finite2")
    paid_trial(agent, lab)
    payload = {"actions": [], "after_observation": [
        {"operator": "history", "path": "shared/analyzed.py"}],
        "question": {"id": "history_analysis", "question": "Does the paid data support a longer history?",
                     "parent_id": "response", "depends_on": []}}
    analysis = "def select_actions(spec, observations, memory):\n    return " + repr(payload) + "\n"
    write_method(agent, lab, analysis)
    before, compute = deepcopy(lab.costs), agent.costs["compute_cost"]
    result = paid_trial(agent, lab)
    assert result["record_indices"] == [] and lab.costs == before
    assert agent.costs["compute_cost"] > compute
    assert len(agent.method_trials) == 2
    assert agent.method_trials[-1]["observation_cost"] == 0
    assert agent.method_trials[-1]["exported_candidates"][0]["path"] == "shared/analyzed.py"
    finished = finish_candidate(agent, lab, "shared/analyzed.py")
    assert finished["fresh_confirmation"]["passed"]
    assert lab.costs["observation_cost"] == 96
    assert agent.result("completed")["method_source"] == analysis


def conditional_practice(tmp_path, original):
    memory = seed_memory(original)
    memory["files"][METHOD_PATH] = procedure(meter=True)
    agent = ConditionalMethodPracticeAgent(None, tmp_path, knowledge=memory,
        config=InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
                               compute_budget=100000, execution_seconds=30.))
    lab = TemporalInquiryLab(task_index=2, apparatus_seed=99106007, instrument_mode="recurrent")
    paid_trial(agent, lab)
    return agent, lab


def test_conditional_method_changes_stages_from_actual_feedback_without_source_rewrites(
        original_knowledge, tmp_path):
    agent, lab = conditional_practice(tmp_path, original_knowledge)
    conditional_source = """import json
def select_actions(spec, observations, memory):
    outcomes = json.load(open('research_state.json'))['operator_results']
    if not outcomes:
        step = {'operator':'assess_instrument','instrument':'instrument.json',
                'records':[index for index,record in enumerate(observations) if record['measurements']]}
    elif outcomes[-1]['operator'] == 'assess_instrument':
        if any(item['status'] == 'supported' for item in outcomes[-1]['hypotheses']):
            step = {'operator':'response','instrument':'instrument.json'}
        else:
            step = {'operator':'instrument','path':'instrument.json'}
    else:
        assert outcomes[-1]['operator'] == 'instrument'
        step = {'operator':'response','instrument':'instrument.json'}
    return {'actions':[], 'after_observation':[step],
            'question':{'id':'conditional','question':'Which next step follows from the measured evidence?',
                        'parent_id':'response','depends_on':[]}}
"""
    write_method(agent, lab, conditional_source)
    for expected in ("assess_instrument", "instrument", "response"):
        result = paid_trial(agent, lab)
        assert not result.get("after_observation_error"), result
        outcome = result["after_observation_results"][0]
        assert outcome["operator"] == expected
        assert agent.method_trials[-1]["operator_sequence"] == [expected]
        assert agent.method_trials[-1]["observation_cost"] == 0
        assert (agent.workspace / METHOD_PATH).read_text() == conditional_source
        if expected == "assess_instrument":
            assert all(item["status"] == "refuted" for item in outcome["hypotheses"])
    assert lab.costs["observation_cost"] == 48
    assert agent.conditional_trials() == [1, 2, 3]
    candidate = candidate_from(result)
    finished = finish_candidate(agent, lab, candidate)
    assert finished["fresh_confirmation"]["passed"]
    assert agent.result("completed")["conditional_method_trials"] == [1, 2, 3]
    assert lab.costs["observation_cost"] == 96


def test_repeating_same_analysis_sequence_does_not_publish_as_conditional_method(
        original_knowledge, tmp_path):
    agent, lab = conditional_practice(tmp_path, original_knowledge)
    payload = {"actions": [], "after_observation": [
        {"operator": "instrument", "path": "instrument.json"},
        {"operator": "response", "instrument": "instrument.json"}],
        "question": {"id": "fixed_analysis", "question": "Fit the instrument and response again.",
                     "parent_id": "response", "depends_on": []}}
    fixed_source = "def select_actions(spec, observations, memory):\n    return " + repr(payload) + "\n"
    write_method(agent, lab, fixed_source)
    candidate_from(paid_trial(agent, lab))
    candidate = candidate_from(paid_trial(agent, lab))
    assert agent.method_trials[-1]["operator_count_before"] > agent.method_trials[-2]["operator_count_before"]
    assert agent.method_trials[-1]["operator_sequence"] == agent.method_trials[-2]["operator_sequence"]
    assert agent.conditional_trials() == []
    before = deepcopy(lab.costs)
    refused = finish_candidate(agent, lab, candidate)
    assert refused["error"] and "substantive" in refused["error"]
    assert lab.costs == before and lab.costs["observation_cost"] == 48
    assert agent.model is None


def test_failed_stage_then_successful_same_sequence_does_not_qualify_as_conditional(
        original_knowledge, tmp_path):
    agent, lab = conditional_practice(tmp_path, original_knowledge)
    payload = {"actions": [], "after_observation": [
        {"operator": "assess_instrument", "instrument": "late.json", "records": [0, 1]},
        {"operator": "response", "instrument": "late.json"}],
        "question": {"id": "fixed_dependency", "question": "Assess and use the instrument model.",
                     "parent_id": "response", "depends_on": []}}
    source = "def select_actions(spec, observations, memory):\n    return " + repr(payload) + "\n"
    write_method(agent, lab, source)
    failed = paid_trial(agent, lab)
    assert failed["after_observation_error"]
    assert agent.method_trials[-1]["operator_sequence"] == ["assess_instrument"]

    # Creating the missing dependency lets the same authored sequence complete;
    # it does not add a new decision or branch to that research method.
    created = agent._execute("run", {"operator": "instrument", "path": "late.json"}, lab)
    assert "error" not in created
    successful = paid_trial(agent, lab)
    assert not successful.get("after_observation_error")
    assert agent.method_trials[-1]["operator_sequence"] == ["assess_instrument", "response"]
    assert agent.method_trials[-1]["method_source"] == agent.method_trials[-2]["method_source"] == source
    assert agent.method_trials[-2]["after_observation_error"]  # keep the failed trial as evidence
    assert agent.conditional_trials() == []
    before = deepcopy(lab.costs)
    rejected = finish_candidate(agent, lab, candidate_from(successful))
    assert rejected["error"] and "substantive" in rejected["error"]
    assert lab.costs == before and lab.costs["observation_cost"] == 48
