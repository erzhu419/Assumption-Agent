"""The shared online path executes native research without an LLM or new labels."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import code_agent, structural_inquiry
from hegel_machine.cognition.anonymous_method_learning import AnonymousPreparedMethodAgent
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.code_provider_context import load_code_context, prepare_code_context
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from test_common_archive_online_research import ScriptedProvider
from test_native_condition_research import observed, paid_fixture, public_spec, random_sequence, record


ROOT = {"id": "task", "question": "Predict the public apparatus response",
        "parent_id": None, "depends_on": []}
COMPOSITION = {"id": "compositions", "question": "Which measured condition composition survives?",
               "parent_id": "task", "depends_on": []}


def execute(agent, lab, selected, arguments):
    result = agent._execute(selected, deepcopy(arguments), lab)
    agent.events.append({"step": len(agent.events), "action": selected,
        "arguments": deepcopy(arguments), "feedback": result, "usage": {"total_tokens": 0}})
    return result


def full_round(agent, summary):
    path = summary['json_path']
    assert summary['full_content_file'] == 'research_state.json'
    assert path[0] == 'questions' and path[2] == 'native_condition_research'
    return agent.questions[path[1]]['native_condition_research'][path[3]]


class PaidConditionalLab:
    task_interface = "Supply public control trajectories and predict from controls and history."

    def __init__(self):
        self.paid, self.actions, self.changed = 0, [], False

    def public_spec(self):
        return public_spec()

    @property
    def costs(self):
        return {"observation_cost": self.paid, "measurement_cost": 0, "queries": len(self.actions)}

    def action_cost(self, action):
        assert action["kind"] == "trajectory" and not action.get("channels")
        assert 1 <= len(action["sequence"]) <= 16
        assert all(set(point) == {"drive", "switch"} for point in action["sequence"])
        return len(action["sequence"])

    def execute(self, action):
        count = self.action_cost(action)
        purchased = record(action["sequence"], self.paid, changed=self.changed)
        self.paid += count
        self.actions.append(deepcopy(action))
        return purchased

    def observation_rows(self, purchased):
        return observed(purchased["rows"])

    def prediction_rows(self, action):
        self.action_cost(action)
        return [Observation(dict(point), 0., tuple(deepcopy(action["sequence"][:index])))
                for index, point in enumerate(action["sequence"])]

    def restore_paid_account(self, observations, costs):
        assert self.paid == 0 and self.actions == []
        self.paid = costs["observation_cost"]

    def confirmation_actions(self, count, rng):
        raise AssertionError("Native research and evaluation must not buy confirmation")


def prepared(path, arm="hegel"):
    return CommonArchiveInquiryAgent(ScriptedProvider([]), path, arm=arm,
        source_task=False, confirmation_seed=1709,
        config=InquiryLLMConfig(observation_budget=300, confirmation_rows=16,
            compute_budget=2_000_000, execution_seconds=60., max_model_calls=2),
        knowledge={"files": {"shared/knowledge.json": json.dumps({
            "hypotheses": [], "research_programs": [], "support_questions": []})}})


def buy_fixture(agent, lab):
    feedback = execute(agent, lab, "query", {"question": ROOT,
        "actions": [item["action"] for item in paid_fixture()]})
    assert "error" not in feedback


def test_boundary_query_resumes_old_matched_evidence_and_updates_new_paid_row(tmp_path, monkeypatch):
    agent, lab = prepared(tmp_path), PaidConditionalLab()
    buy_fixture(agent, lab)
    first = research(agent, lab)["native_condition_research"]
    boundary = next(item for item in first["proposed_experiments"]
                    if item.get("purpose") == "conditional_boundary")
    selected = boundary["query_action"]
    assert first["research_decision"]["next_experiment"] == selected
    continuation, = selected["arguments"]["after_observation"]
    assert continuation["matched_pairs"] == [
        {"reference_record_index": 0, "record_index": 1}]
    assert continuation["boundary_update"] == {
        "round_index": 0, "proposal_index": first["proposed_experiments"].index(boundary)}

    def repeat_search(*args, **kwargs):
        raise AssertionError("A boundary measurement must not repeat full structure search")

    monkeypatch.setattr(structural_inquiry, "research_native_conditions", repeat_search)
    result = execute(agent, lab, selected["action"], selected["arguments"])
    assert "error" not in result
    assert "after_observation_error" not in result
    second = agent.questions[-1]["native_condition_research"][-1]
    assert len(second["fit_evidence"]) == 81
    assert second["evidence"][0]["reference_record_index"] == 0
    assert second["evidence"][0]["record_index"] == 1
    after = next(item for item in second["proposed_experiments"]
                 if item.get("purpose") == "conditional_boundary")
    assert after["threshold_interval"] == [0., .25]
    assert boundary["threshold_interval"] == [0., .5]
    assert after["query_action"]["arguments"]["actions"] != selected["arguments"]["actions"]
    assert second["research_decision"]["next_experiment"] == after["query_action"]
    assert min(item["fit_mse"] for item in second["exported_models"]) < 1e-8
    assert second["compute_cost"] < first["compute_cost"] / 4
    assert second["completion_reserve"]["completion_funded"]
    assert second["research_decision"]
    assert second["remaining_boundary"]
    choice = next(item for item in decision_context(agent, lab)["method_decision"]["research_options"]
                  if item["id"] == "native_condition_research")
    assert choice["current_executions"][-1]["research_decision"] == second["research_decision"]
    assert choice["current_executions"][-1]["remaining_boundary"] == second["remaining_boundary"]
    assert result["after_observation_results"][0]["returned_answer"]["answer"][
        "research_decision"] == second["research_decision"]
    assert lab.paid == 81 and agent.costs["observation_cost"] == 81
    assert agent.costs["model_calls"] == 0 and agent.confirmations == []


def test_repeated_boundary_updates_retain_recipes_and_original_candidates(tmp_path, monkeypatch):
    agent, lab = prepared(tmp_path), PaidConditionalLab()
    buy_fixture(agent, lab)
    first = research(agent, lab)["native_condition_research"]
    original = deepcopy(full_round(agent, first))
    original_sources = {item["model"]: (tmp_path / item["model"]).read_text()
                        for item in first["exported_models"]}

    def repeat_search(*args, **kwargs):
        raise AssertionError("Interval updates must reuse the retained candidate's structure")

    monkeypatch.setattr(structural_inquiry, "research_native_conditions", repeat_search)
    current, interval_widths = first, []
    for round_index in range(3):
        proposal = next(item for item in current["proposed_experiments"]
                        if item.get("purpose") == "conditional_boundary")
        left, right = proposal["threshold_interval"]
        interval_widths.append(right - left)
        selected = proposal["query_action"]
        assert selected["arguments"]["after_observation"][0]["boundary_update"][
            "round_index"] == round_index
        feedback = execute(agent, lab, selected["action"], selected["arguments"])
        assert "after_observation_error" not in feedback
        current = feedback["after_observation_results"][0]["native_condition_research"]
        best = min(current["candidates"], key=lambda item: item["fit_mse"])
        assert best["fit_mse"] < 1e-8 and best["research_program"]
        assert len(current["fit_evidence"]) == 81 + round_index
        assert current["compute_cost"] < first["compute_cost"] / 4
    assert interval_widths == [.5, .25, .125]
    question = next(q for q in agent.questions if q["id"] == COMPOSITION["id"])
    assert question["native_condition_research"][0] == original
    assert all((tmp_path / path).read_text() == source
               for path, source in original_sources.items())
    assert lab.paid == 83 and agent.costs["model_calls"] == 0


def test_boundary_update_does_not_spend_old_candidate_completion_budget(tmp_path):
    class ConfirmingLab(PaidConditionalLab):
        def confirmation_actions(self, count, rng):
            return [{"kind": "trajectory", "channels": [],
                     "sequence": random_sequence(rng.randrange(1_000_000), count)}]

    agent, lab = prepared(tmp_path), ConfirmingLab()
    buy_fixture(agent, lab)
    first = research(agent, lab)["native_condition_research"]
    saved = deepcopy(full_round(agent, first))
    candidate = min(first["exported_models"], key=lambda item: item["fit_mse"])
    source = (tmp_path / candidate["model"]).read_bytes()
    boundary = next(item for item in first["proposed_experiments"]
                    if item.get("purpose") == "conditional_boundary")
    selected = boundary["query_action"]
    rows_after_purchase = len(agent.rows) + boundary["observation_cost"]
    completion = 2 * rows_after_purchase + agent.config.confirmation_rows
    before = agent.costs["compute_cost"]
    agent.config = replace(agent.config, compute_budget=before + completion)
    feedback = execute(agent, lab, selected["action"], selected["arguments"])
    assert "after_observation_error" not in feedback
    latest = feedback["after_observation_results"][0]["native_condition_research"]
    assert latest["compute_cost"] == 0
    assert latest["completion_reserve"]["completion_funded"]
    assert agent.costs["compute_cost"] == before
    choice = next(item for item in decision_context(agent, lab)["method_decision"]["research_options"]
                  if item["id"] == "native_condition_research")
    assert candidate in choice["current_executions"][0]["models"]
    selected = candidate["evaluate_action"]
    evaluated = execute(agent, lab, selected["action"], selected["arguments"])
    assert evaluated["returncode"] == 0 and evaluated["mse"] < 1e-8
    confirmed = execute(agent, lab, "finish", {"model": candidate["model"]})
    assert confirmed["fresh_confirmation"]["passed"]
    assert agent.costs["compute_cost"] == agent.config.compute_budget
    assert agent.costs["model_calls"] == 0
    question = next(q for q in agent.questions if q["id"] == COMPOSITION["id"])
    assert question["native_condition_research"][0] == saved
    assert (tmp_path / candidate["model"]).read_bytes() == source


def test_boundary_counterexample_returns_structure_choice_without_automatic_search(tmp_path, monkeypatch):
    agent, lab = prepared(tmp_path), PaidConditionalLab()
    buy_fixture(agent, lab)
    first = research(agent, lab)["native_condition_research"]
    selected = first["research_decision"]["next_experiment"]

    def unchosen_search(*args, **kwargs):
        raise AssertionError("Only the researcher may choose a new structural search")

    monkeypatch.setattr(structural_inquiry, "research_native_conditions", unchosen_search)
    lab.changed = True
    feedback = execute(agent, lab, selected["action"], selected["arguments"])
    assert "after_observation_error" not in feedback
    latest = feedback["after_observation_results"][0]["native_condition_research"]
    assert latest["research_decision"]["kind"] == "structural_mismatch"
    assert latest["exported_models"]
    assert latest["proposed_experiments"] == []
    assert lab.paid == 81 and agent.costs["model_calls"] == 0
    choice = next(item for item in decision_context(agent, lab)["method_decision"]["research_options"]
                  if item["id"] == "native_condition_research")
    assert choice["current_executions"][-1]["research_decision"] == latest["research_decision"]
    assert "boundary_update" not in choice["execute_action"]["arguments"]


def research(agent, lab):
    return execute(agent, lab, "run", {"operator": "native_condition_research",
        "question": COMPOSITION, "max_compute": 500_000,
        "matched_pairs": [{"reference_record_index": 0, "record_index": 1}]})


def test_incomplete_joint_search_returns_optional_same_bank_compute_choice(tmp_path):
    agent, lab = prepared(tmp_path), PaidConditionalLab()
    buy_fixture(agent, lab)
    arguments = {'operator': 'native_condition_research', 'question': COMPOSITION,
        'max_compute': 2600, 'programs': [{'op': 'const', 'value': 0.}],
        'matched_pairs': [{'reference_record_index': 0, 'record_index': 1}],
        'record_indices': list(range(5)), 'max_candidates': 4,
        'notes': 'An earlier unverified interpretation is not an execution argument.'}
    returned = execute(agent, lab, 'run', arguments)
    result = returned['native_condition_research']
    assert result['status'] == 'search_incomplete'
    assert returned['returned_answer']['answer']['research_decision']['kind'] == 'computation_incomplete'
    assert result['research_decision']['completed_joint_fits'] == 0
    choice = result['compute_continuation']
    selected = choice['action']
    retry = selected['arguments']
    assert retry['question'] == {'id': COMPOSITION['id']}
    assert 'notes' not in retry
    assert 2600 < retry['max_compute'] <= choice['available_compute_after_completion_reserve']
    for name in ('record_indices', 'matched_pairs', 'programs', 'max_candidates'):
        assert retry[name] == arguments[name]
    record = next(item for item in decision_context(agent, lab)['method_decision']['research_options']
                  if item['id'] == 'native_condition_research')['current_executions'][-1]
    assert record['compute_continuation'] == choice
    old = deepcopy(full_round(agent, result))
    old_source = (agent.workspace / result['exported_models'][0]['model']).read_text()
    assert lab.paid == 80 and agent.costs['model_calls'] == 0
    assert len(agent.questions[-1]['native_condition_research']) == 1
    resumed = execute(agent, lab, selected['action'], retry)['native_condition_research']
    assert resumed['status'] == 'candidates_require_evaluation'
    assert resumed['searched_candidates'] > 0
    assert min(item['fit_mse'] for item in resumed['candidates']) < 1e-8
    assert full_round(agent, result) == old
    assert (agent.workspace / result['exported_models'][0]['model']).read_text() == old_source
    assert lab.paid == 80 and agent.costs['model_calls'] == 0
    assert resumed['completion_reserve']['completion_funded']


def test_native_feedback_keeps_decisions_and_references_full_candidate_recipes(tmp_path):
    agent, lab = prepared(tmp_path), PaidConditionalLab()
    buy_fixture(agent, lab)
    feedback = research(agent, lab)['native_condition_research']
    complete = full_round(agent, feedback)
    frozen = deepcopy(complete)
    for field in ('status', 'finite_prior', 'round_index', 'exported_models',
                  'proposed_experiments', 'fit_evidence', 'research_decision',
                  'remaining_boundary', 'completion_reserve'):
        assert feedback[field] == complete[field]
    state = json.loads((tmp_path / 'research_state.json').read_text())
    for candidate, original in zip(feedback['candidates'], complete['candidates']):
        assert {key: value for key, value in candidate.items() if key != 'research_program'} == {
            key: value for key, value in original.items() if key != 'research_program'}
        if 'research_program' not in original:
            continue
        recipe = candidate['research_program']
        resolved = state
        for part in recipe['json_path']:
            resolved = resolved[part]
        assert resolved == json.loads(json.dumps(original['research_program']))
        assert recipe['full_content_file'] == 'research_state.json'
        assert [item['operation'] for item in recipe['selection_steps']] == [
            item['operation'] for item in original['research_program']['selection_steps']]
        assert all('parameter_trials' not in step and 'program' not in step
                   for step in recipe['selection_steps'])
    # The real fixture contains search trials and graph bodies. Those remain
    # in one restorable place rather than following each returned candidate.
    assert len(json.dumps(feedback)) < len(json.dumps(complete)) / 2
    assert complete == frozen


def register_retained_method(agent, prior_execution=None):
    path = "shared/retained_method.py"
    source = "def select_actions(*args): return {'stop': 'Retained research finding'}\n"
    entry = {"id": "retained_method", "path": path, "max_steps": 4,
             "prior_execution": prior_execution or {"scope": "Earlier paid practice",
                "cases": [{"case": "previous", "fresh_confirmation": {"passed": True}}]}}
    manifest = {"hypotheses": [], "research_programs": [entry], "support_questions": []}
    agent.knowledge["manifest"] = deepcopy(manifest)
    agent.knowledge["files"][path] = source
    agent.knowledge["files"]["shared/knowledge.json"] = json.dumps(manifest)
    agent._save(path, source)
    agent._save("shared/knowledge.json", json.dumps(manifest))
    return entry


def decision_context(agent, lab):
    return prepare_code_context(agent.workspace, agent._request(lab),
                                schema={}, instructions="Interface test only")


def test_prior_bank_resume_uses_current_paid_rows_and_original_search_options(tmp_path):
    agent, lab = prepared(tmp_path), PaidConditionalLab()
    buy_fixture(agent, lab)
    options = {'basis': [{'op': 'feature', 'name': 'drive'}],
        'programs': [{'op': 'const', 'value': 99.}], 'max_candidates': 3, 'max_compute': 100_000}
    first = execute(agent, lab, 'run', {'operator': 'native_condition_research',
        'question': COMPOSITION, 'matched_pairs': [
            {'reference_record_index': 0, 'record_index': 1}], **options})['native_condition_research']
    original = deepcopy(full_round(agent, first))
    execute(agent, lab, 'query', {'question': {'id': ROOT['id']}, 'actions': [
        {'kind': 'trajectory', 'channels': [], 'sequence': random_sequence(79)}]})
    before = deepcopy(agent.costs)
    choice, = decision_context(agent, lab)['method_decision']['research_options']
    bank, = choice['current_executions']
    assert choice['execute_action']['arguments']['research_question']['id'] == ROOT['id']
    assert bank['paid_rows'] == 80 and bank['current_paid_rows'] == 96
    assert bank['original_evidence_selection']['matched_pairs'] == [
        {'reference_record_index': 0, 'record_index': 1}]
    action = bank['resume_action']
    assert action['arguments'] == {'operator': 'native_condition_research',
        'research_question': {'id': COMPOSITION['id']}, **options}
    assert agent.costs == before

    # Actually choosing the exposed action rescored the retained bank on the
    # new rows. It did not create a separate root bank or buy more observations.
    resumed = execute(agent, lab, action['action'], action['arguments'])['native_condition_research']
    assert resumed['round_index'] == 1 and len(resumed['fit_evidence']) == 96
    assert any(item['source'] == 'retained_program' and item['program'] == original['candidates'][0]['program']
               and len(item['fit_evidence']) == 96 for item in resumed['candidates'])
    root = next(q for q in agent.questions if q['id'] == ROOT['id'])
    assert not root.get('native_condition_research')
    assert full_round(agent, first) == original
    assert lab.paid == 96 and agent.costs['model_calls'] == 0


def test_explicit_native_fit_scope_is_not_silently_expanded_by_resume(tmp_path):
    agent, lab = prepared(tmp_path), PaidConditionalLab()
    buy_fixture(agent, lab)
    execute(agent, lab, 'run', {'operator': 'native_condition_research',
        'question': COMPOSITION, 'record_indices': [0, 1], 'max_compute': 20_000})
    choice, = decision_context(agent, lab)['method_decision']['research_options']
    bank, = choice['current_executions']
    assert bank['fit_scope'] == 'selected_records'
    assert bank['original_evidence_selection']['record_indices'] == [0, 1]
    assert 'resume_action' not in bank


@pytest.mark.parametrize("arm", ["hegel", "code_react"])
def test_native_choice_is_visible_and_callable_beside_retained_methods(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm), PaidConditionalLab()
    retained = register_retained_method(agent)
    buy_fixture(agent, lab)
    before = deepcopy(agent.costs)
    first = decision_context(agent, lab)
    assert list(first).index("method_decision") < list(first).index("task_interface")
    view = first["method_decision"]
    method, = view["methods"]
    assert method["path"] == retained["path"]
    assert method["prior_execution"] == retained["prior_execution"]
    assert method["execute_action"] == {"action": "run", "arguments": {
        "operator": "procedure", "path": retained["path"], "max_steps": 4, "confirm": True}}
    choice = next(item for item in view["research_options"]
                  if item["id"] == "native_condition_research")
    assert choice["execute_action"] == {"action": "run", "arguments": {
        "operator": "native_condition_research", "research_question": {"id": ROOT["id"]}}}
    assert choice["finite_prior"] and choice["current_paid_rows"] == 80
    assert choice["current_executions"] == []
    assert agent.costs == before and agent.method_results == []

    # Choosing another existing question changes the advertised binding, rather
    # than requiring an invented composition question or a host-selected answer.
    execute(agent, lab, "write", {"path": "shared/question_marker.py", "code": "# local inquiry\n",
        "research_question": COMPOSITION})
    selected = next(item for item in decision_context(agent, lab)["method_decision"]["research_options"]
                    if item["id"] == "native_condition_research")["execute_action"]
    assert selected["arguments"]["research_question"] == {"id": COMPOSITION["id"]}
    outcome = execute(agent, lab, selected["action"], selected["arguments"])
    actual = outcome["native_condition_research"]
    assert actual["exported_models"]
    followup = decision_context(agent, lab)["method_decision"]
    choice = next(item for item in followup["research_options"]
                  if item["id"] == "native_condition_research")
    execution, = choice["current_executions"]
    assert execution["question_id"] == COMPOSITION["id"]
    assert execution["round_index"] == 0 and execution["paid_rows"] == 80
    assert execution["status"] == actual["status"]
    assert execution["models"] == actual["exported_models"]
    assert followup["methods"][0]["execute_action"] == method["execute_action"]
    best = min(execution["models"], key=lambda item: item["fit_mse"])
    action = best["evaluate_action"]
    scored = execute(agent, lab, action["action"], action["arguments"])
    assert scored["returncode"] == 0 and scored["mse"] < 1e-8
    assert lab.paid == 80 and agent.costs["model_calls"] == 0
    assert agent.costs["observation_cost"] == 80 and agent.method_results == []
    assert agent.model is None and agent.confirmations == []


def test_large_retained_history_keeps_real_method_choices_in_provider_context(tmp_path):
    agent, lab = prepared(tmp_path), PaidConditionalLab()
    retained = register_retained_method(agent)
    buy_fixture(agent, lab)
    executions = [{"step": index, "path": retained["path"],
        "proposal": {"action": "run", "arguments": {"path": "shared/analysis.py"}},
        "feedback": {"status": "method_step_complete", "returncode": 0,
            "stdout": "paid execution feedback " * (18_000 if index < 3 else 1)}}
        for index in range(59)]
    for entry in executions[-2:]:
        entry["feedback"] = {"status": "method_review_required", "review": {
            "reason": "Acquired evidence awaits interpretation", "method_step": entry["step"],
            "current_paid_rows": 80}}
    agent.method_results = deepcopy(executions)
    before = deepcopy(agent.costs)
    context = decision_context(agent, lab)
    assert len(json.dumps(context)) <= 1_000_000
    decision = context["method_decision"]
    assert "research_options" in decision and "methods" in decision
    choice = next(item for item in decision["research_options"]
                  if item["id"] == "native_condition_research")
    assert choice["execute_action"]["arguments"] == {
        "operator": "native_condition_research", "research_question": {"id": ROOT["id"]}}
    method = next(item for item in decision["methods"] if item["path"] == retained["path"])
    assert method["execute_action"] == {"action": "run", "arguments": {
        "operator": "procedure", "path": retained["path"], "max_steps": 4, "confirm": True}}
    history = method["current_executions"]
    assert history["count"] == 59 and history["preview_start"] == 57
    assert history["latest"] == executions[-2:]
    assert history["full_content_file"] == "method_decision.json"
    assert history["json_path"] == ["methods", 0, "current_executions"]
    assert method["continuation"]["review"] == executions[-1]["feedback"]["review"]
    complete = json.loads((agent.workspace / "method_decision.json").read_text())
    saved_executions = complete["methods"][0]["current_executions"]
    assert len(json.dumps(saved_executions)) < len(json.dumps(executions)) / 2
    reference = saved_executions[0]['feedback']['stdout']
    assert reference['full_content_file'] == 'research_state.json'
    full = json.loads((agent.workspace / reference['full_content_file']).read_text())
    assert full['method_results'] == executions
    for part in reference['json_path']:
        full = full[part]
    assert full == executions[0]['feedback']['stdout']
    assert complete["methods"][0]["prior_execution"] == retained["prior_execution"]
    assert agent.costs == before and agent.method_results == executions
    assert lab.paid == 80 and agent.model is None and agent.confirmations == []


def test_final_provider_request_retains_callable_methods_after_global_compaction(tmp_path, monkeypatch):
    agent, lab = prepared(tmp_path), PaidConditionalLab()
    retained = register_retained_method(agent)
    buy_fixture(agent, lab)
    executions = [{"step": index, "path": retained["path"],
        "proposal": {"action": "run", "arguments": {"path": "shared/analysis.py"}},
        "feedback": {"returncode": 0, "status": "method_review_required",
            "stdout": "measured residual pending interpretation\n" * (10_000 if index < 3 else 190),
            "review": {"reason": "Acquired evidence awaits interpretation",
                       "method_step": index, "current_paid_rows": 80}}}
        for index in range(59)]
    agent.method_results = deepcopy(executions)
    request = agent._request(lab)
    # The earlier regression only enlarged method history, which local preview
    # already shrinks. Real cumulative notes and runtime facts also exceed the
    # final provider bound after those local previews have been constructed.
    request["research_notes"] = "Previously measured result needs interpretation.\n" * 16_000
    request["evidence_notebook"]["past_observation_summary"] = (
        "Paid evidence remains in the recorded trajectories.\n" * 6_000)
    original = deepcopy(request)
    before = deepcopy(agent.costs)
    transmitted = []

    def completion(command, payload, timeout, cwd):
        context = json.loads(payload)
        transmitted.append(context)
        assert len(payload) <= 1_000_000
        assert context["research_notes"]["json_path"] == ["research_notes"]
        assert context["limits"] == original["limits"]
        assert context["costs"] == original["costs"]
        assert context["online_research"]["active_question"] == ROOT["id"]
        decision = context["method_decision"]
        assert len(json.dumps(decision)) > 12_000
        method = next(item for item in decision["methods"] if item["path"] == retained["path"])
        assert method["execute_action"] == {"action": "run", "arguments": {
            "operator": "procedure", "path": retained["path"], "max_steps": 4, "confirm": True}}
        history = method["current_executions"]
        assert history["count"] == 59 and history["preview_start"] == 57
        assert history["latest"] == executions[-2:]
        assert method["continuation"]["review"] == executions[-1]["feedback"]["review"]
        choice = next(item for item in decision["research_options"]
                      if item["id"] == "native_condition_research")
        selected = choice["execute_action"]
        assert selected == {"action": "run", "arguments": {
            "operator": "native_condition_research", "research_question": {"id": ROOT["id"]}}}
        stored = load_code_context(cwd)
        assert stored == json.loads(json.dumps(original))
        complete = json.loads((Path(cwd) / history["full_content_file"]).read_text())
        assert len(json.dumps(complete["methods"][0]["current_executions"])) < len(json.dumps(executions)) / 2
        canonical = json.loads((Path(cwd) / 'research_state.json').read_text())
        assert canonical['method_results'] == executions
        destination = Path(command[command.index("--output-last-message") + 1])
        destination.write_text(json.dumps({"action": selected["action"],
            "arguments_json": json.dumps(selected["arguments"]),
            "research_notes": "Try the available finite method on the paid evidence."}))
        return 0, json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 10, "output_tokens": 2}}), "", False

    monkeypatch.setattr(code_agent, "_run", completion)
    provider = code_agent.CodexCodeProvider(agent.workspace, executable="/test/codex")
    selected = provider.next_action(request)
    assert len(transmitted) == 1 and request == original and agent.costs == before
    outcome = execute(agent, lab, selected["action"], selected["arguments"])
    assert outcome["native_condition_research"]["exported_models"]
    assert lab.paid == 80 and agent.costs["model_calls"] == 0
    assert agent.model is None and agent.confirmations == []


@pytest.mark.parametrize("arm", ["hegel", "code_react"])
def test_restored_paid_scene_research_and_evaluation_share_both_arms(tmp_path, arm):
    original, old_lab = prepared(tmp_path / "old", arm), PaidConditionalLab()
    buy_fixture(original, old_lab)
    (original.workspace / "research_context.json").write_text(json.dumps(original._request(old_lab)))
    previous = original.result("researching")
    resumed, lab = prepared(tmp_path / "resumed", arm), PaidConditionalLab()
    restored = resumed.restore_research(previous, original.workspace, lab, confirmation_seed=1709)
    assert restored["new_observations"] == 0 and lab.actions == []
    before_paid = lab.paid
    outcome = research(resumed, lab)
    result = outcome["native_condition_research"]
    assert "native_condition_research" in resumed._request(lab)["task_interface"]
    assert lab.paid == before_paid == 80 and lab.actions == []
    assert result["exported_models"] and result["round_index"] == 0
    best = min(result["exported_models"], key=lambda item: item["fit_mse"])
    action = best["evaluate_action"]
    scored = execute(resumed, lab, action["action"], action["arguments"])
    assert scored["returncode"] == 0 and scored["mse"] < 1e-8
    assert resumed.evaluated_model["evaluated_rows"] == 80
    assert resumed.costs["model_calls"] == 0 and resumed.costs["observation_cost"] == 80
    assert lab.actions == [] and resumed.model is None and resumed.confirmations == []


def test_selected_query_feeds_purchased_pairs_back_to_the_same_question(tmp_path):
    agent, lab = prepared(tmp_path), PaidConditionalLab()
    buy_fixture(agent, lab)
    first = research(agent, lab)["native_condition_research"]
    state = json.loads((tmp_path / "research_state.json").read_text())
    assert state["operator_results"] == json.loads(json.dumps(agent.operator_results))
    assert state["operator_results"][-1]["native_condition_research"]["round_index"] == 0
    proposal = next(item for item in first["proposed_experiments"] if item["matched_action_pairs"])
    selected = proposal["query_action"]
    before = len(agent.observations)
    assert selected["arguments"]["question"]["id"] == COMPOSITION["id"]
    assert selected["arguments"]["after_observation"][0]["research_question"]["id"] == COMPOSITION["id"]
    bought = execute(agent, lab, selected["action"], selected["arguments"])
    assert "after_observation_error" not in bought
    repeated = bought["after_observation_results"][0]
    assert repeated["operator"] == "native_condition_research"
    assert repeated["research_question_id"] == COMPOSITION["id"]
    second = repeated["native_condition_research"]
    state = json.loads((tmp_path / "research_state.json").read_text())
    assert state["operator_results"] == json.loads(json.dumps(agent.operator_results))
    assert state["operator_results"][-1]["native_condition_research"]["round_index"] == 1
    assert second["round_index"] == 1
    assert all(item["reference_record_index"] >= before and item["record_index"] >= before
               for item in second["evidence"])
    assert lab.paid == 80 + proposal["observation_cost"]
    assert agent.model is None and not agent.confirmations


@pytest.mark.parametrize("continuation", [False, True])
def test_compiled_method_consumes_native_feedback_from_the_shared_operator_ledger(tmp_path, continuation):
    agent = AnonymousPreparedMethodAgent(None, tmp_path, arm="hegel", source_task=False,
        config=InquiryLLMConfig(observation_budget=300, confirmation_rows=16,
            compute_budget=2_000_000, execution_seconds=60., max_model_calls=2))
    lab = PaidConditionalLab()
    buy_fixture(agent, lab)
    # Execute the actual select_actions subprocess boundary. The method reads
    # only the documented flat ledger, including analysis run inside a query.
    method = '''import json
def select_actions(spec, observations, memory):
    state = json.load(open('research_state.json'))
    entries = [entry for entry in state['operator_results']
               if entry['operator'] == 'native_condition_research']
    if not entries:
        return {'action':'run', 'arguments':{
            'operator':'native_condition_research',
            'question':{'id':'compositions', 'question':'Compare measured conditions',
                        'parent_id':'task', 'depends_on':[]},
            'matched_pairs':[{'reference_record_index':0, 'record_index':1}],
            'max_compute':500000}}
    result = entries[-1]['native_condition_research']
    if CONTINUATION and len(entries) == 1:
        return result['proposed_experiments'][0]['query_action']
    selected = result['exported_models'][0]
    if not state['evaluated_model']:
        return selected['evaluate_action']
    return {'stop':'Evaluated the declared candidate from the actual native result'}
'''.replace("CONTINUATION", repr(continuation))
    agent.bind_method(method)
    outcome = agent.execute_method(lab, max_steps=4 if continuation else 3)
    assert outcome["status"] == "method_stopped"
    ledger = agent.operator_results
    assert len(ledger) == (2 if continuation else 1)
    assert [item["index"] for item in ledger] == list(range(len(ledger)))
    assert all(item["research_question_id"] == COMPOSITION["id"] for item in ledger)
    latest = ledger[-1]["native_condition_research"]
    assert len(latest["fit_evidence"]) == len(agent.rows)
    assert latest["round_index"] == int(continuation)
    evaluated = next(item for item in outcome["method_results"]
                     if item["proposal"].get("action") == "evaluate")
    assert evaluated["proposal"] == latest["exported_models"][0]["evaluate_action"]
    assert evaluated["feedback"]["evaluated_model"]["evaluated_rows"] == len(agent.rows)
    saved = json.loads((tmp_path / "research_state.json").read_text())
    assert saved["operator_results"] == json.loads(json.dumps(ledger))
    if continuation:
        query = next(item for item in outcome["method_results"]
                     if item["proposal"].get("action") == "query")
        returned = query["feedback"]["after_observation_results"][0]
        assert {key: value for key, value in ledger[-1].items() if key != "index"} == returned
        assert len(lab.actions) == len(paid_fixture()) + len(query["proposal"]["arguments"]["actions"])
    else:
        assert len(agent.rows) == 80
    assert outcome["costs"]["model_calls"] == 0 and not agent.confirmations


def test_failed_native_operator_is_persisted_without_losing_paid_evidence(tmp_path):
    agent, lab = prepared(tmp_path), PaidConditionalLab()
    buy_fixture(agent, lab)
    before = deepcopy(agent.observations)
    arguments = {"operator": "native_condition_research", "question": COMPOSITION,
                 "record_indices": [len(agent.observations)]}
    with pytest.raises(ValueError):
        execute(agent, lab, "run", arguments)
    entry, = agent.operator_results
    assert entry["operator"] == "native_condition_research" and entry["error"]
    assert entry["arguments"] == arguments and entry["index"] == 0
    state = json.loads((tmp_path / "research_state.json").read_text())
    assert state["operator_results"] == [entry]
    assert agent.observations == before and lab.paid == 80
    assert agent.model is None and not agent.confirmations


def test_invalid_pair_returns_same_question_automatic_retry_without_research_answer(tmp_path):
    agent, lab = prepared(tmp_path), PaidConditionalLab()
    buy_fixture(agent, lab)
    sequence = deepcopy(agent.observations[0]['action']['sequence'])
    sequence[0]['drive'] = .9
    sequence[2]['switch'] = .5
    execute(agent, lab, 'query', {'question': ROOT, 'actions': [
        {'kind': 'trajectory', 'channels': [], 'sequence': sequence}]})
    before = deepcopy(agent.observations)
    arguments = {'operator': 'native_condition_research', 'question': COMPOSITION,
                 'matched_pairs': [{'reference_record_index': 0, 'record_index': 5}],
                 'record_indices': [0, 1, 2, 3, 4, 5], 'max_candidates': 8,
                 'max_compute': 500_000, 'programs': []}
    feedback = execute(agent, lab, 'run', arguments)
    assert feedback['status'] == 'research_not_executed' and feedback['error']
    assert 'returned_answer' not in feedback and 'native_condition_research' not in feedback
    question = next(item for item in agent.questions if item['id'] == COMPOSITION['id'])
    assert question['native_condition_research'] == []
    assert not question.get('returned_answer')
    assert not (tmp_path / '_experiment_models').exists()
    assert len(agent.operator_results) == 1
    assert agent.operator_results[0]['status'] == 'research_not_executed'
    retry = feedback['automatic_matching_action']
    assert retry == {'action': 'run', 'arguments': {
        **{key: value for key, value in arguments.items() if key not in {'matched_pairs', 'question'}},
        'question': {'id': COMPOSITION['id']}}}
    assert feedback['completion_reserve']['search_allowance'] == arguments['max_compute']
    assert feedback['compute_cost'] == 96
    result = execute(agent, lab, retry['action'], retry['arguments'])
    assert 'error' not in result and result['research_question_id'] == COMPOSITION['id']
    assert result['native_condition_research']['candidates']
    assert len(question['native_condition_research']) == 1
    assert min(item['fit_mse'] for item in result['native_condition_research']['candidates']) < 1e-8
    assert agent.observations == before and lab.paid == 96
    assert agent.costs['model_calls'] == 0 and not agent.confirmations


def test_research_round_rescores_old_models_and_preserves_the_original_artifacts(tmp_path):
    agent, lab = prepared(tmp_path), PaidConditionalLab()
    buy_fixture(agent, lab)
    original = research(agent, lab)["native_condition_research"]
    saved = deepcopy(full_round(agent, original))
    files = {item["model"]: (agent.workspace / item["model"]).read_text()
             for item in original["exported_models"]}
    lab.changed = True
    execute(agent, lab, "query", {"actions": [{"kind": "trajectory", "channels": [],
        "sequence": random_sequence(79)}], "question": {"id": COMPOSITION["id"]}})
    second = research(agent, lab)["native_condition_research"]
    retained = [item for item in second["candidates"] if item["source"] == "retained_program"]
    assert retained and min(item["fit_mse"] for item in retained) > .001
    previous = next(item for item in full_round(agent, original)["candidates"]
                    if item.get("research_program"))
    preserved = next(item for item in full_round(agent, second)["candidates"]
                     if item["program"] == previous["program"])
    assert preserved["research_program"] == previous["research_program"]
    question = next(item for item in agent.questions if item["id"] == COMPOSITION["id"])
    assert question["native_condition_research"][0] == saved
    assert second["round_index"] == 1
    assert set(files).isdisjoint(item["model"] for item in second["exported_models"])
    assert all((agent.workspace / name).read_text() == source for name, source in files.items())
    assert lab.paid == 96 and agent.model is None and agent.confirmations == []
    # Native rounds live on the actual question, and their nested frozen model
    # roots must survive the same checkpoint restoration used by online research.
    (agent.workspace / "research_context.json").write_text(json.dumps(agent._request(lab)))
    resumed, fresh = prepared(tmp_path / "restored"), PaidConditionalLab()
    receipt = resumed.restore_research(agent.result("researching"), agent.workspace, fresh,
                                      confirmation_seed=1709)
    restored_question = next(item for item in resumed.questions if item["id"] == COMPOSITION["id"])
    assert restored_question["native_condition_research"] == question["native_condition_research"]
    assert resumed.operator_results == json.loads(json.dumps(agent.operator_results))
    assert len(resumed.operator_results) == 2
    assert all((resumed.workspace / name).read_text() == source for name, source in files.items())
    assert receipt["new_observations"] == 0 and fresh.actions == [] and fresh.paid == 96


@pytest.mark.parametrize("depth,stale", [(0, False), (1, True), (2, False)])
def test_search_preserves_actual_candidate_delivery_and_fresh_confirmation(tmp_path, depth, stale):
    class ConfirmingLab(PaidConditionalLab):
        def confirmation_actions(self, count, rng):
            return [{"kind": "trajectory", "channels": [],
                     "sequence": random_sequence(rng.randrange(1_000_000), count)}]

    agent, lab = prepared(tmp_path), ConfirmingLab()
    agent.config = replace(agent.config, max_model_calls=20)
    buy_fixture(agent, lab)
    if stale:
        execute(agent, lab, "write", {"path": "shared/baseline.py",
            "code": "def predict(row): return 0.0\n"})
        execute(agent, lab, "evaluate", {"model": "shared/baseline.py"})
        execute(agent, lab, "query", {"actions": [{"kind": "trajectory", "channels": [],
            "sequence": random_sequence(79)}]})
    paid, old_events = deepcopy(agent.observations), deepcopy(agent.events)
    for level in range(depth):
        execute(agent, lab, "run", {"operator": "inquire", "max_model_calls": 10 - level,
            "question": {**COMPOSITION, "id": f"composition_{level}"}})
    question_id = agent._inquiry_stack[-1]["question_id"] if depth else COMPOSITION["id"]
    arguments = {"operator": "native_condition_research",
        "question": {**COMPOSITION, "id": question_id}, "max_compute": 500_000,
        "matched_pairs": [{"reference_record_index": 0, "record_index": 1}]}
    learned = execute(agent, lab, "run", arguments)["native_condition_research"]
    candidate = min(learned["exported_models"], key=lambda item: item["fit_mse"])
    saved = deepcopy(full_round(agent, learned))
    source = (agent.workspace / candidate["model"]).read_bytes()
    reserve = learned["completion_reserve"]
    assert reserve["local_deliveries"] == depth and reserve["retained_comparisons"] == int(stale)
    assert reserve["completion_funded"] is True
    # Reproduce the observed end-of-search state: only delivery/confirmation
    # funding remains. Another unrestricted native call must not consume it.
    agent.config = replace(agent.config,
        compute_budget=agent.costs["compute_cost"] + reserve["compute_cost"])
    before = deepcopy(agent.costs)
    repeated = execute(agent, lab, "run", arguments)["native_condition_research"]
    assert repeated["status"] == "compute_budget_exhausted"
    assert repeated["compute_cost"] == repeated["completion_reserve"]["search_allowance"] == 0
    assert repeated["completion_reserve"]["completion_funded"] is True
    assert agent.costs["compute_cost"] == before["compute_cost"]
    choice = next(item for item in decision_context(agent, lab)["method_decision"]["research_options"]
                  if item["id"] == "native_condition_research")
    assert choice["current_executions"][-1]["completion_reserve"] == repeated["completion_reserve"]
    selected = candidate["evaluate_action"]
    scored = execute(agent, lab, selected["action"], selected["arguments"])
    assert scored["returncode"] == 0 and scored["mse"] < 1e-8
    if stale:
        assert scored["retained_comparison"]["returncode"] == 0
    for _ in range(depth):
        returned = execute(agent, lab, "finish", {"model": candidate["model"]})
        assert returned["status"] == "inquiry_finished" and not agent.confirmations
        agent._settle_inquiries()
    confirmed = execute(agent, lab, "finish", {"model": candidate["model"]})
    assert confirmed["fresh_confirmation"]["passed"] is True
    assert agent.costs["compute_cost"] == agent.config.compute_budget
    assert agent.costs["model_calls"] == 0
    assert agent.observations[:len(paid)] == paid and agent.events[:len(old_events)] == old_events
    question = next(item for item in agent.questions if item["id"] == question_id)
    assert question["native_condition_research"][0] == saved
    assert (agent.workspace / candidate["model"]).read_bytes() == source
