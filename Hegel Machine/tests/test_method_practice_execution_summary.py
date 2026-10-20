"""Keep actual research outcomes visible while the author repairs its program."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.anonymous_method_learning import AnonymousResearchMethodLearner
from hegel_machine.cognition.code_provider_context import _bound_provider_context, _compact_practice_feedback
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.temporal_paired_practice import (
    candidate_execution_summary, practice_execution_summary,
)
from test_python_method_learning import ADAPTIVE_METHOD, archived_predictor, make_learner, practice


def test_stop_after_computation_is_not_counted_as_acquisition_and_keeps_execution_feedback():
    execution = {'path': METHOD_PATH, 'returncode': 0, 'timeout': False,
        'python_execution_seconds': 17., 'stderr': 'fit=0.4 validation=0.7\n'}
    feedback = {'status': 'method_stopped', 'reason': 'revise model',
                'program_execution': execution}
    result = {'status': 'method_stopped', 'practice_start': {'method_results': 1, 'events': 1},
        'continuation_cost': {'python_execution_seconds': 17., 'compute_cost': 0,
                             'observation_cost': 0},
        'method_results': [{'proposal': {'action': 'query'}, 'feedback': {}},
            {'step': 1, 'proposal': {'stop': 'revise model'}, 'feedback': feedback}],
        'events': [{'feedback': {}}, {'feedback': feedback}]}
    summary = practice_execution_summary(result, 'trials/trial_006')
    assert summary['action_counts'] == {'stop': 1}
    assert summary['latest_program_execution'] == execution
    assert summary['research_steps'][0]['program_execution'] == execution
    assert summary['last_execution_message']['program_execution'] == execution
    assert summary['practice_execution_cost'] == result['continuation_cost']
    bounded = _bound_provider_context({'method_decision': {'latest_practice': {
        'execution_summary': candidate_execution_summary(summary)}}, 'events': [],
        'files': {'history.json': 'x' * 1500000}})
    view = bounded['method_decision']['latest_practice']['execution_summary']
    assert view['latest_program_execution'] == execution
    assert view['practice_execution_cost']['python_execution_seconds'] == 17.


def test_legacy_stopped_practice_reports_recorded_cost_without_inventing_fit_scores():
    summary = practice_execution_summary({'status': 'method_stopped',
        'continuation_cost': {'python_execution_seconds': 506., 'compute_cost': 0,
                             'observation_cost': 0},
        'method_results': [{'proposal': {'stop': 'unresolved'}, 'feedback': {
            'status': 'method_stopped', 'reason': 'unresolved'}}]}, 'trials/trial_006')
    assert summary['action_counts'] == {'stop': 1}
    assert summary['practice_execution_cost']['python_execution_seconds'] == 506.
    assert 'latest_program_execution' not in summary
    assert 'latest_evaluation' not in summary
    assert summary['confirmation_count'] == 0


def native_query_steps():
    candidates = [{"model": f"_experiment_models/round_1/candidate_{i:04d}.py",
        "fit_mse": 5.59e-20 + i,
        "evaluate_action": {"action": "evaluate", "arguments": {
            "model": f"_experiment_models/round_1/candidate_{i:04d}.py"}},
        "program": {"source": "large retained expression" * 10000}}
        for i in range(12)]
    return [{"step": 0, "proposal": {"action": "run", "arguments": {
        "operator": "native_condition_research"}}, "feedback": {
        "native_condition_research": {"status": "needs_matched_evidence", "candidates": []}}},
        {"step": 1, "proposal": {"action": "query", "arguments": {"actions": []}},
         "feedback": {"after_observation_results": [{"operator": "native_condition_research",
             "native_condition_research": {"status": "candidates_require_evaluation",
                 "candidates": candidates}}]}}]


@pytest.mark.parametrize('failure', [
    {'returncode': 1, 'timeout': False, 'stderr': (
        "Traceback (most recent call last):\n  select_actions failed\n"
        "AttributeError: 'NoneType' object has no attribute 'get'\n")},
    {'error': 'ValueError: invalid program result'},
    {'returncode': 124, 'timeout': True, 'stderr': 'Program execution timed out\n'},
])
def test_failed_program_without_proposal_is_not_an_acquisition_and_exposes_error(failure):
    result = {'status': 'method_no_progress', 'method_results': [
        {'step': 0, 'proposal': {'actions': [{'kind': 'trajectory',
            'sequence': [{'u': 1}, {'u': 2}]}]}, 'feedback': {'observation_cost': 2}},
        {'step': 1, 'proposal': None, 'feedback': deepcopy(failure)}],
        'events': [{'feedback': {'reason': 'Earlier invocation stopped'}},
                   {'feedback': deepcopy(failure)}]}
    original = deepcopy(result)

    summary = practice_execution_summary(result, 'trials/trial_001')

    assert summary['action_counts'] == {'query': 1, 'program_error': 1}
    assert summary['research_steps'][0]['observation_cost'] == 2
    assert summary['research_steps'][0]['trajectory_lengths'] == [2]
    error = summary['research_steps'][1]
    assert error['action'] == summary['last_action']['action'] == 'program_error'
    assert 'observation_cost' not in error and 'trajectory_lengths' not in error
    message = summary['last_execution_message']
    for key in ('error', 'returncode', 'timeout'):
        if key in failure:
            assert error[key] == message[key] == failure[key]
    if failure.get('stderr'):
        assert error['stderr'] == message['stderr'] == failure['stderr'].strip().splitlines()[-1]
    assert 'reason' not in message
    assert summary['submission_count'] == summary['confirmation_count'] == 0
    assert candidate_execution_summary(summary)['last_execution_message'] == message
    bounded = _bound_provider_context({'method_decision': {'latest_practice': {
        'execution_summary': summary}}, 'events': [], 'files': {'history.json': 'x' * 1500000}})
    assert bounded['method_decision']['latest_practice']['execution_summary'][
        'last_execution_message'] == message
    assert result == original


def test_query_continuation_candidates_survive_without_copying_the_expression_pool():
    result = {"status": "method_stopped", "method_results": native_query_steps()}
    summary = practice_execution_summary(result, "trials/trial_105")
    native = summary["latest_native_result"]
    assert native["status"] == "candidates_require_evaluation"
    assert native["candidate_count"] == 12
    assert native["json_path"] == ["method_results", 1, "feedback",
        "after_observation_results", 0, "native_condition_research"]
    assert native["candidate_preview"][0]["evaluate_action"]["arguments"]["model"] == (
        native["candidate_preview"][0]["model"])
    assert summary["submission_count"] == summary["confirmation_count"] == 0
    assert "latest_evaluation" not in summary
    assert len(json.dumps(summary)) < 4000
    assert summary["result_steps_field"] == "method_results"
    assert summary["author_feedback_steps_field"] == "steps"


def test_successful_paid_evaluation_is_not_a_submission_or_confirmation():
    steps = native_query_steps()
    candidate = steps[1]["feedback"]["after_observation_results"][0][
        "native_condition_research"]["candidates"][0]
    for index in range(2, 31):
        steps.append({"step": index, "proposal": candidate["evaluate_action"],
            "feedback": {"mse": 5.59e-20, "returncode": 0}})
    summary = practice_execution_summary({"status": "method_step_limit",
        "method_results": steps}, "trials/trial_100")
    assert summary["action_counts"]["evaluate"] == 29
    assert summary["latest_evaluation"]["mse"] == 5.59e-20
    assert summary["latest_evaluation"]["model"] == candidate["model"]
    assert summary["submission_count"] == summary["confirmation_count"] == 0


def test_same_row_incumbent_comparison_keeps_distinct_fitting_views_and_actual_choice():
    decision = {'status': 'candidate_retained', 'selected_model': '_evaluated_model/model.py',
        'same_paid_rows': True, 'scope': 'Paid scoring and retention only; no independent confirmation.',
        'candidate': {'model': 'model.py', 'mse': 1e-28, 'evaluated_rows': 1603,
            'evaluation_scope': 'paid_fit', 'fitting_view': {'rows': 1603}},
        'incumbent': {'model': '_evaluated_model/model.py', 'mse': .32,
            'evaluated_rows': 1603, 'evaluation_scope': 'paid_prediction',
            'fitting_view': {'rows': 1411}}}
    feedback = {'returncode': 0, 'mse': 1e-28, 'evaluation_scope': 'paid_fit',
        'fitting_view': {'scope': 'current_paid_prefix', 'file': 'model_observations.json',
                         'rows': 1603, 'response_row_range': [0, 1602]},
        'paid_residuals': {'evaluated_rows': 1603, 'examples': ['large'] * 10000},
        'retained_comparison': {'returncode': 0, 'mse': .32, 'evaluation_scope': 'paid_prediction',
            'fitting_view': {'rows': 1411, 'response_row_range': [0, 1410]},
            'paid_residuals': {'evaluated_rows': 1603, 'examples': ['large'] * 10000},
            'later_readings': {'rows': 192, 'mse': .152, 'response_row_range': [1411, 1602]}},
        'retention_decision': decision,
        'evaluated_model': {'model': '_evaluated_model/model.py', 'source_model': 'model.py',
            'mse': 1e-28, 'evaluated_rows': 1603, 'evaluation_scope': 'paid_fit'}}
    result = {'status': 'method_stopped', 'method_results': [{'step': 0,
        'proposal': {'action': 'evaluate', 'arguments': {'model': 'model.py'}}, 'feedback': feedback}]}
    summary = practice_execution_summary(result, 'trials/trial_001')
    score = summary['latest_evaluation']
    assert score['evaluated_rows'] == score['retained_comparison']['evaluated_rows'] == 1603
    assert score['fitting_view']['rows'] == 1603
    assert score['retained_comparison']['fitting_view']['rows'] == 1411
    assert score['retained_comparison']['later_readings']['mse'] == .152
    assert score['retention_decision'] == decision
    assert summary['latest_retention_decision'] == decision
    assert candidate_execution_summary(summary)['latest_evaluation'] == score
    assert summary['confirmation_count'] == summary['submission_count'] == 0
    assert 'large' not in json.dumps(summary)
    assert len(json.dumps(summary)) < 5000


def test_budget_incomplete_stage_and_optional_continuation_reach_practice_summary():
    native = {'status': 'search_incomplete', 'candidates': [],
        'search_stages': [{'stage': 'joint_windows', 'status': 'computation_incomplete',
            'compute_cost': 0, 'completed_fits': 0,
            'blocked_work': {'required_compute': 14724, 'remaining_compute': 12070,
                             'additional_compute': 2654}}],
        'compute_continuation': {'status': 'available',
            'available_compute_after_completion_reserve': 2000000,
            'action': {'action': 'run', 'arguments': {'operator': 'native_condition_research',
                'question': {'id': 'condition'}, 'max_compute': 1885371}}}}
    summary = practice_execution_summary({'method_results': [{
        'proposal': {'action': 'run', 'arguments': {'operator': 'native_condition_research'}},
        'feedback': {'native_condition_research': native}}]}, 'trials/trial_002')
    returned = summary['latest_native_result']
    assert returned['search_stages'] == native['search_stages']
    assert returned['compute_continuation'] == native['compute_continuation']
    assert returned['status'] == 'search_incomplete'
    assert 'latest_evaluation' not in summary
    assert summary['submission_count'] == summary['confirmation_count'] == 0


def test_candidate_boundary_proposal_remains_visible_separate_from_actual_purchase():
    steps = native_query_steps()
    native = steps[1]["feedback"]["after_observation_results"][0]["native_condition_research"]
    query = {"action": "query", "arguments": {
        "actions": [{"kind": "trajectory", "sequence": [{"control": 0.4}]}],
        "after_observation": [{"operator": "native_condition_research"}]}}
    native["proposed_experiments"] = [{
        "purpose": "conditional_boundary", "control": "control",
        "predicted_separation": 0.718, "observation_cost": 1,
        "status": "proposed_not_purchased", "query_action": query,
        "threshold_interval": [0.3, 0.5], "interval_width": 0.2}]
    summary = practice_execution_summary({"method_results": steps}, "trials/trial_151")
    returned = summary["latest_native_result"]
    preview = returned["proposed_experiment_preview"][0]
    assert returned["candidate_count"] == 12
    assert preview["purpose"] == "conditional_boundary"
    assert preview["predicted_separation"] == 0.718
    assert preview["observation_cost"] == 1
    assert preview["after_observation_operators"] == ["native_condition_research"]
    actual = {"method_results": steps}
    for key in preview["query_action_path"]:
        actual = actual[key]
    assert actual == query
    assert preview["threshold_interval"] == [0.3, 0.5]
    assert preview["interval_width"] == 0.2
    assert "query_action" not in preview
    assert len(json.dumps(summary)) < 5000

    steps.append({"step": 2, "proposal": deepcopy(query),
                  "feedback": {"observation_cost": 1}})
    summary = practice_execution_summary({"method_results": steps}, "trials/trial_151")
    assert summary["research_steps"][-1]["observation_cost"] == 1
    assert summary["research_steps"][-1]["trajectory_lengths"] == [1]
    assert summary["latest_native_result"]["proposed_experiment_preview"][0][
        "status_when_returned"] == "proposed_not_purchased"
    assert summary["submission_count"] == summary["confirmation_count"] == 0


@pytest.mark.parametrize("decision_kind", ["remaining_parameter_ambiguity", "structural_mismatch"])
def test_native_research_decision_reaches_author_after_paid_view_and_context_compaction(
        tmp_path, archived_predictor, decision_kind):
    steps = native_query_steps()
    native = steps[-1]["feedback"]["after_observation_results"][0]["native_condition_research"]
    next_experiment = {"action": "query", "arguments": {
        "actions": [{"kind": "trajectory", "channels": [], "sequence": [{"c": 0.183}]}],
        "question": {"id": "condition"}, "after_observation": [{
            "operator": "native_condition_research", "research_question": {"id": "condition"},
            "boundary_update": {"round_index": 2, "proposal_index": 0}}]}}
    native.update(
        research_decision={"kind": decision_kind,
            **({"next_experiment": next_experiment} if decision_kind == "remaining_parameter_ambiguity"
               else {"branch_errors": [0.2, 0.3], "observation_index": 6})},
        remaining_boundary={"control": "c", "threshold_interval": [0.173, 0.193],
            "interval_width": 0.02, "branch_predictions": [0.681, -0.485],
            "predicted_separation": 1.36},
        boundary_update={"status": "boundary_refined" if decision_kind == "remaining_parameter_ambiguity"
            else "structure_mismatch", "before_interval": [0.153, 0.193],
            "after_interval": [0.173, 0.193], "observed_target": -0.485,
            "branch_errors": [1.36, 2e-21], "matched_branch": 1},
        completion_reserve={"compute_cost": 338, "remaining_compute": 620935,
            "confirmation_rows": 96, "completion_funded": True},
        compute_cost=1103, round_index=2)
    # The omitted candidate expression pool is several megabytes; only the
    # executed decision and compact boundary evidence should reach the author.
    result = {"status": "method_stopped", "method_results": steps}
    summary = practice_execution_summary(result, "trials/trial_239")
    author, _ = make_learner(tmp_path / "author", archived_predictor)
    trial = {"case": "case_a", "directory": "trials/trial_239", "status": "method_stopped",
        "method_source": (author.workspace / METHOD_PATH).read_text(), "execution_summary": summary}
    paid_view = author._paid_practice_view(trial)
    bounded = _bound_provider_context({
        "method_decision": {"latest_practice": paid_view},
        "files": {"large_saved_history.json": "x" * 1500000}, "events": []})
    returned = bounded["method_decision"]["latest_practice"]["execution_summary"]["latest_native_result"]
    for key in ("research_decision", "remaining_boundary", "boundary_update", "completion_reserve",
                "compute_cost", "round_index"):
        assert returned[key] == native[key]
    if decision_kind == "remaining_parameter_ambiguity":
        assert returned["research_decision"]["next_experiment"] == next_experiment
    assert len(json.dumps(summary)) < 6000
    assert len(json.dumps(bounded)) < 12000
    assert "large retained expression" not in json.dumps(bounded)
    assert bounded["files"]["large_saved_history.json"]["full_content_file"] == "research_context.json"


def test_confirmation_keeps_actual_submitted_native_structure_not_first_candidate():
    steps = native_query_steps()
    native = steps[1]["feedback"]["after_observation_results"][0]["native_condition_research"]
    selected = native["candidates"][4]
    selected["source"] = "conditional_experiment:4"
    selected["program"] = Program("window_sum", lag=5, children=(
        Program("if", children=(Program("ge", children=(
            Program("feature", name="c"), Program("const", value=.225))),
            Program("feature", name="x"), Program("const", value=0.))),)).to_dict()
    selected["research_program"] = {"selection_steps": [{
        "operation": "discover_conditional_coordinates", "width": 5,
        "placement": "inside_window", "threshold_interval": [.15, .30],
        "program": selected["program"], "parameter_trials": ["large search ledger"] * 1000}]}
    native["proposed_experiments"] = [{"candidate_source": selected["source"],
        "purpose": "conditional_boundary", "observation_cost": 1,
        "predicted_separation": .718, "status": "proposed_not_purchased",
        "query_action": {"action": "query", "arguments": {
            "actions": [{"kind": "trajectory", "sequence": [{"c": .225}]}],
            "after_observation": [{"operator": "native_condition_research"}]}}}]
    frozen = "_submitted_models/confirmation_0000/candidate_0004.py"
    steps.append({"proposal": {"model": selected["model"]}, "feedback": {
        "submitted_candidate": {"model": frozen, "source_model": selected["model"],
            "paid_rows": 120, "mse": 1e-20},
        "fresh_confirmation": {"passed": False, "mse": .01, "rows": 96,
                               "residuals": {"examples": [{"row_index": 160}]}}}})
    original = deepcopy(steps)
    summary = practice_execution_summary({"method_results": steps}, "trials/trial_169")
    actual = summary["latest_submitted_predictor"]
    assert actual["source_model"] == selected["model"]
    assert actual["model_file"] == "trials/trial_169/" + frozen
    assert actual["native_candidate"]["expression"].startswith("window_sum(5, if(ge(c,")
    assert actual["native_candidate"]["fitted_structure"][0]["placement"] == "inside_window"
    assert "parameter_trials" not in actual["native_candidate"]["fitted_structure"][0]
    node = {"method_results": steps}
    for key in actual["native_candidate"]["json_path"]:
        node = node[key]
    assert node == selected
    assert actual["experiments_proposed_with_candidate"][0]["observation_cost"] == 1
    assert actual["fresh_confirmation"]["passed"] is False
    assert actual["confirmation_residuals_path"][-1] == "residuals"
    assert len(json.dumps(actual)) < 2000
    assert candidate_execution_summary(summary)["latest_submitted_predictor"][
        "native_candidate"] == actual["native_candidate"]
    assert steps == original


@pytest.mark.parametrize("selected_index", [0, 1])
def test_retained_candidate_proposal_uses_actual_program_not_shared_source(selected_index):
    candidates = [{"model": f"candidate_{index}.py", "source": "retained_program",
        "fit_mse": 0., "program": Program("const", value=float(index)).to_dict()}
        for index in range(2)]
    proposed = {"candidate_source": "retained_program",
        "candidate_program": deepcopy(candidates[1]["program"]),
        "purpose": "conditional_boundary", "observation_cost": 1,
        "query_action": {"action": "query", "arguments": {"actions": []}}}
    selected = candidates[selected_index]
    steps = [{"proposal": {"action": "run", "arguments": {
        "operator": "native_condition_research"}}, "feedback": {
        "native_condition_research": {"status": "candidates_require_evaluation",
            "candidates": candidates, "proposed_experiments": [proposed]}}},
        {"proposal": {"model": selected["model"]}, "feedback": {
            "submitted_candidate": {"source_model": selected["model"],
                "model": f"frozen/{selected['model']}"},
            "fresh_confirmation": {"passed": True, "mse": 0., "rows": 96}}}]
    submitted = practice_execution_summary({"method_results": steps}, "trials/trial_240")[
        "latest_submitted_predictor"]
    associated = submitted["experiments_proposed_with_candidate"]
    assert len(associated) == selected_index
    if selected_index:
        assert associated[0]["purpose"] == "conditional_boundary"


def test_ordinary_submission_preserves_snapshot_without_inventing_native_structure():
    steps = [{"proposal": {"model": "model.py"}, "feedback": {
        "submitted_candidate": {"model": "_submitted_models/confirmation_0000/model.py",
                                "source_model": "model.py", "paid_rows": 120},
        "fresh_confirmation": {"passed": True, "mse": 0., "rows": 96}}}]
    actual = practice_execution_summary({"method_results": steps}, "trials/trial_001")[
        "latest_submitted_predictor"]
    assert actual["source_model"] == "model.py"
    assert actual["model_file"].endswith("_submitted_models/confirmation_0000/model.py")
    assert "native_candidate" not in actual


def test_python_submission_preview_uses_frozen_source_after_working_copy_changes(tmp_path):
    frozen = '_submitted_models/confirmation_0003/_evaluated_model/model.py'
    source_file = tmp_path / frozen
    source_file.parent.mkdir(parents=True)
    source = 'from sklearn.ensemble import ExtraTreesRegressor\n' + '# retained source\n' * 90
    source_file.write_text(source)
    (tmp_path / 'model.py').write_text('def predict(row): return 0.\n')
    steps = native_query_steps()
    steps.append({'proposal': {'model': 'model.py'}, 'feedback': {
        'submitted_candidate': {'model': frozen, 'source_model': 'model.py', 'paid_rows': 1941},
        'fresh_confirmation': {'passed': False, 'mse': 4., 'rows': 96}}})
    result = {'workspace': str(tmp_path), 'method_results': steps}

    summary = practice_execution_summary(result, 'trials/trial_002')
    actual = summary['latest_submitted_predictor']
    preview = actual['source_preview']

    assert preview == {'file': str(source_file), 'characters': len(source), 'preview': source[:800]}
    assert 'native_candidate' not in actual
    assert 'return 0.' not in preview['preview']
    compact = candidate_execution_summary(summary)
    assert compact['latest_submitted_predictor']['source_preview'] == preview
    feedback = _compact_practice_feedback({'execution_summary': summary}, ['feedback'])
    bounded = _bound_provider_context({'method_decision': {
        'latest_practice': feedback, 'candidates': [compact]},
        'events': [], 'files': {'history.json': 'x' * 1500000}})
    assert bounded['method_decision']['latest_practice']['execution_summary'][
        'latest_submitted_predictor']['source_preview'] == preview
    assert bounded['method_decision']['candidates'][0]['latest_submitted_predictor'][
        'source_preview'] == preview
    assert source_file.read_text() == source


def test_missing_submission_source_does_not_fall_back_to_working_model(tmp_path):
    (tmp_path / 'model.py').write_text('def predict(row): return 0.\n')
    result = {'workspace': str(tmp_path), 'method_results': [{
        'proposal': {'model': 'model.py'}, 'feedback': {
            'submitted_candidate': {'model': '_submitted_models/missing/model.py',
                                    'source_model': 'model.py'},
            'fresh_confirmation': {'passed': False, 'mse': 4., 'rows': 96}}}]}

    summary = practice_execution_summary(result, 'trials/trial_002')

    assert 'source_preview' not in summary['latest_submitted_predictor']
    assert 'source_preview' not in candidate_execution_summary(summary)['latest_submitted_predictor']


def test_submission_coverage_distinguishes_training_and_confirmation_trajectories():
    records, offset = [], 0
    for length in [64, 8, 12, 12, 12, 12, 1, 64, 32, 16]:
        records.append({"row_indices": list(range(offset, offset + length)),
            "rows": [{"history": [{}] * index} for index in range(length)]})
        offset += length
    result = {"paid_observations": records, "method_results": [{
        "proposal": {"model": "candidate.py"}, "feedback": {
            "submitted_candidate": {"model": "frozen.py", "paid_rows": 121},
            "fresh_confirmation": {"passed": False, "rows": 96}}}]}
    original = deepcopy(result)
    evidence = practice_execution_summary(result, "trials/trial_209")[
        "latest_submitted_predictor"]["submission_evidence"]
    assert evidence == {
        "paid_before_submission": {"row_count": 121,
            "trajectory_row_counts": [64, 8, 12, 12, 12, 12, 1], "max_history_length": 63},
        "fresh_confirmation": {"row_count": 96,
            "trajectory_row_counts": [64, 32], "max_history_length": 63}}
    # The later research record and the confirmation do not enter the old fit view.
    assert result == original


def test_recorded_practice_summary_survives_restore_and_large_author_context(
        tmp_path, archived_predictor):
    author, _ = make_learner(tmp_path / "origin", archived_predictor)
    outcome = practice(author, "case_a")
    summary = outcome["execution_summary"]
    assert summary == author.practice_trials[-1]["execution_summary"]
    assert summary["latest_evaluation"]["mse"] < 1e-10
    assert summary["submission_count"] == summary["confirmation_count"] == 1
    assert summary["latest_confirmation"]["passed"]
    author._execute("write", {"path": METHOD_PATH,
        "code": "def select_actions(spec, records, memory):\n    return {'stop': 'inspect returned candidates'}\n"},
        author.interface)
    stopped = practice(author, "case_a")["execution_summary"]
    assert "latest_evaluation" not in stopped
    previous = author.result("model_call_budget_exhausted")
    resumed, _ = make_learner(tmp_path / "resumed", archived_predictor)
    resumed.restore_practice(previous)
    resumed._prepare_program_state()
    decision = json.loads(resumed.files["method_decision.json"])
    assert decision["latest_practice"]["execution_summary"] == stopped
    assert decision["latest_evaluated_practice"]["execution_summary"] == summary
    context = _bound_provider_context({"method_decision": deepcopy(decision),
        "files": {"large_saved_history.json": "x" * 1500000}, "events": []})
    assert context["method_decision"]["latest_practice"]["execution_summary"] == stopped
    assert context["method_decision"]["latest_evaluated_practice"]["execution_summary"] == summary


@pytest.mark.parametrize("with_checkpoint", [False, True])
def test_last_call_edit_practice_exports_current_source_support_without_another_request(
        tmp_path, archived_predictor, with_checkpoint):
    template, _ = make_learner(tmp_path / "template", archived_predictor)

    class LastCallEdit:
        def next_action(self, request):
            before = json.loads(request["files"]["method_decision.json"])
            assert before["current_source_coverage"]["confirmed_cases"] == ["case_a"]
            return {"action": "write", "arguments": {"path": METHOD_PATH, "code": ADAPTIVE_METHOD},
                    "usage": {"total_tokens": 1, "backend": "scripted_interface_test"}}

    def new_author(directory, provider):
        return AnonymousResearchMethodLearner(provider, directory, knowledge=template.knowledge,
            cases=template.cases, lab_factory=template.lab_factory,
            prepare_trial=template.prepare_trial, practice_limits=template.practice_limits,
            config=replace(template.config, max_model_calls=1))

    author = new_author(tmp_path / "author", LastCallEdit())
    assert practice(author, "case_a")["status"] == "completed"
    assert practice(author, "case_b")["status"] != "completed"
    checkpoints = []

    def save_checkpoint(result):
        decision = json.loads((author.workspace / "method_decision.json").read_text())
        checkpoints.append((result, decision))

    result = author.run(on_checkpoint=save_checkpoint if with_checkpoint else None)
    assert result["status"] == "model_call_budget_exhausted"
    assert len(result["practice_trials"]) == 3
    assert result["practice_trials"][-1]["method_source"] == ADAPTIVE_METHOD
    assert result["practice_trials"][-1]["case"] == "case_b"
    assert result["practice_trials"][-1]["fresh_confirmation"]["passed"]
    assert result["outgoing_knowledge"]["files"][METHOD_PATH] == ADAPTIVE_METHOD
    decision = json.loads((author.workspace / "method_decision.json").read_text())
    assert decision["current_source_coverage"]["confirmed_cases"] == ["case_b"]
    assert decision["latest_practice"]["directory"] == "trials/trial_002"
    if with_checkpoint:
        assert len(checkpoints) == 1
        assert checkpoints[0][1] == decision
    resumed = new_author(tmp_path / "resumed", None)
    resumed.restore_practice(result)
    restored = resumed.result("researching")
    assert restored["practice_trials"] == result["practice_trials"]
    assert restored["outgoing_knowledge"]["files"][METHOD_PATH] == ADAPTIVE_METHOD
    assert json.loads(resumed.files["method_decision.json"])["current_source_coverage"] == (
        decision["current_source_coverage"])
