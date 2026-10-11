"""Condition research stays actionable without resending its executable pool."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.code_provider_context import prepare_code_context


def condition_research():
    coordinate = {"op": "feature", "name": "density"}
    for coefficient in range(24):
        coordinate = {"op": "add", "children": [coordinate,
            {"op": "const", "value": coefficient / 100.}]}
    candidates = [{
        "id": f"condition_{index}", "claim": f"Activation iff residual is below {index + 1}",
        "status": "refuted" if index == 0 else "compatible",
        "checked_response_count": 289, "conflict_count": 1 if index == 0 else 0,
        "conflicting_responses": [{"crystal": 2, "frequency": 19., "activated": False,
            "predicted_activation": True, "evidence": [288]}] if index == 0 else [],
        "program": {"op": "lt", "children": [deepcopy(coordinate),
            {"op": "const", "value": float(index + 1)}]},
        "feature_derivations": {"residual_0": deepcopy(coordinate)},
        "evidence": list(range(289)), "proposed_evidence": list(range(288)),
    } for index in range(16)]
    trial = {"id": "condition_2_19", "number": 2, "prediction": 19.,
        "outcome": "Crystal 2 activates at setting 19",
        "predictions": {candidate["id"]: bool(index % 2)
                        for index, candidate in enumerate(candidates[1:], 1)}}
    return {"status": "discriminating_experiment_available",
        "candidates": candidates, "inputs": {"responses": [
            {"crystal": 2, "frequency": float(index), "activated": False,
             "evidence": [index]} for index in range(289)]},
        "compute_cost": 90000, "confirmation_status": "unconfirmed",
        "scope_is_exhaustive": False, "search_scope": "finite Boolean conditions",
        "selected_experiment": trial, "experiment_action": {"action": "run", "arguments": {
            "operator": "reference_response", "path": "shared/controller.py",
            "experiment": {"number": 2, "setting": 19.}}}}


def read_reference(workspace, descriptor):
    value = json.loads((workspace / descriptor["full_content_file"]).read_text())
    for key in descriptor["json_path"]:
        value = value[key]
    return value


def assert_actionable(compact, original):
    for key in ("status", "compute_cost", "confirmation_status", "scope_is_exhaustive",
                "search_scope", "selected_experiment", "experiment_action"):
        assert compact[key] == original[key]
    assert "inputs" not in compact
    assert len(compact["candidates"]) == len(original["candidates"])
    for summary, full in zip(compact["candidates"], original["candidates"]):
        for key in ("id", "claim", "status", "checked_response_count", "conflict_count"):
            assert summary[key] == full[key]
        assert not {"program", "feature_derivations", "evidence", "proposed_evidence"} & summary.keys()
    conflict = compact["candidates"][0]["conflicting_responses"][0]
    assert {key: conflict[key] for key in ("crystal", "frequency", "activated",
            "predicted_activation")} == {"crystal": 2, "frequency": 19.,
                                          "activated": False, "predicted_activation": True}


def test_condition_results_remain_actionable_in_events_and_decision_view(tmp_path):
    research = condition_research()
    decision = {"questions": [{"id": "premise", "status": "inconclusive",
        "answer": {"condition_research": deepcopy(research)}}]}
    decision_file = tmp_path / "method_decision.json"
    decision_file.write_text(json.dumps(decision))
    original_decision = decision_file.read_bytes()
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
        "files": {"method_decision.json": json.dumps(decision)},
        "events": [{"action": "run", "feedback": {"condition_research": deepcopy(research)}}]}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    event_summary = context["events"][0]["feedback"]["condition_research"]
    decision_summary = context["method_decision"]["questions"][0]["answer"]["condition_research"]
    for summary in (event_summary, decision_summary):
        assert_actionable(summary, research)
        assert read_reference(tmp_path, summary) == research
    assert event_summary["full_content_file"] == "research_context.json"
    assert decision_summary["full_content_file"] == "method_decision.json"
    assert len(json.dumps(context)) < len(json.dumps(original)) / 2
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original
    assert decision_file.read_bytes() == original_decision


def test_large_nested_proposal_keeps_condition_action_instead_of_text_preview(tmp_path):
    research = condition_research()
    proposal = {"path": "shared/premise.py", "status": "proposal_ready",
                "condition_research": deepcopy(research)}
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
        "research": {"llm_inquiries": {"proposal:shared/premise.py": {
            "condition_research": deepcopy(research), "proposal": proposal}}},
        "events": [{"action": "run", "feedback": {"proposal": deepcopy(proposal)}}]}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    inquiry = context["research"]["llm_inquiries"]["proposal:shared/premise.py"]
    summaries = [inquiry["condition_research"], inquiry["proposal"]["condition_research"],
                 context["events"][0]["feedback"]["proposal"]["condition_research"]]
    for summary in summaries:
        assert_actionable(summary, research)
        assert read_reference(tmp_path, summary) == research
    assert inquiry["proposal"]["path"] == proposal["path"]
    assert inquiry["proposal"]["status"] == proposal["status"]
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original


@pytest.mark.parametrize('has_experiment', [True, False])
def test_growing_condition_pool_keeps_verdicts_outcome_examples_and_exact_retrieval(tmp_path, has_experiment):
    research = condition_research()
    candidates = research['candidates']
    research['candidates'] = [
        {**deepcopy(candidates[index % len(candidates)]), 'id': f'condition_{index}'}
        for index in range(96)]
    # An authored alternative in the middle must remain visible beside the
    # native pool, even when it has the same status and forecast.
    research['candidates'][45].update(origin='authored', authored_id='conditional_rule')
    if has_experiment:
        research['selected_experiment']['predictions'] = {
            candidate['id']: bool(index % 2) for index, candidate in enumerate(research['candidates'])}
    else:
        research.update(status='no_discriminating_experiment', selected_experiment=None,
                        experiment_action=None)
    revision = {'action': 'run', 'arguments': {
        'operator': 'propose_explanations', 'path': 'shared/binding.py'}}
    research['research_obligation'] = {'status': 'condition_language_extension_needed',
                                      'targets': [5], 'revision_action': revision}
    research['research_action'] = deepcopy(revision)
    research['target_consequences'] = {'targets': [
        {'number': 5, 'status': 'empty_support', 'scope_complete': True},
        {'number': 4, 'status': 'unresolved', 'scope_complete': False}]}
    request = {'evidence': {'observations_file': 'laboratory_observations.json'},
               'events': [{'feedback': {'condition_research': research}}]}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')

    compact = context['events'][0]['feedback']['condition_research']
    for key in ('status', 'selected_experiment', 'experiment_action', 'target_consequences',
                'research_obligation', 'research_action'):
        assert compact[key] == research[key]
    pool = compact['candidates']
    assert pool['count'] == 96
    assert pool['status_counts'] == {'refuted': 6, 'compatible': 90}
    assert read_reference(tmp_path, pool) == research['candidates']
    predictions = (research['selected_experiment'] or {}).get('predictions', {})
    expected_groups = {(row.get('origin'), row['status'], predictions.get(row['id']))
                       for row in research['candidates']}
    shown_groups = {(row.get('origin'), row['status'], predictions.get(row['id']))
                    for row in pool['preview']}
    assert shown_groups == expected_groups
    assert any(row.get('authored_id') == 'conditional_rule' for row in pool['preview'])
    for row in pool['preview']:
        actual = read_reference(tmp_path, row)
        assert {key: value for key, value in row.items()
                if key not in ('full_content_file', 'json_path')} == {
                    key: actual[key] for key in row if key not in ('full_content_file', 'json_path')}
    assert len(json.dumps(pool)) < len(json.dumps([
        {key: value for key, value in row.items() if key not in (
            'program', 'feature_derivations', 'evidence', 'proposed_evidence')}
        for row in research['candidates']])) / 3
    assert request == original


def test_previous_condition_dispatch_inputs_remain_retrievable_as_snapshot(tmp_path):
    snapshot = {'responses': [{'crystal': 5, 'frequency': float(index), 'activated': False}
                              for index in range(300)],
                'controller_path': 'shared/controller.py', 'targets': [4, 5],
                'bound_rows': {'5': {'signal': 8.}}, 'residual_sources': {'r': {'op': 'const', 'value': 3.}}}
    current = condition_research()
    request = {'evidence': {'observations_file': 'laboratory_observations.json'},
        'research': {'llm_inquiries': {'extension': {'condition_extension_context': snapshot,
                                                    'condition_research': current}}}}
    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')
    inquiry = context['research']['llm_inquiries']['extension']
    descriptor = inquiry['condition_extension_context']
    assert descriptor['response_count'] == 300
    assert descriptor['controller_path'] == snapshot['controller_path']
    assert descriptor['targets'] == snapshot['targets']
    assert read_reference(tmp_path, descriptor) == snapshot
    assert_actionable(inquiry['condition_research'], current)


def test_large_identical_pools_share_examples_but_new_test_outcomes_get_current_examples(tmp_path):
    research = condition_research()
    research['candidates'] = [{**deepcopy(research['candidates'][index % 16]), 'id': f'condition_{index}'}
                              for index in range(96)]
    changed = deepcopy(research)
    changed['selected_experiment']['predictions'] = {
        candidate['id']: index > 32 for index, candidate in enumerate(changed['candidates'])}
    changed['selected_experiment']['prediction'] = 25.
    request = {'evidence': {'observations_file': 'laboratory_observations.json'},
        'research': {'llm_inquiries': {'author': {'condition_research': research}}},
        'events': [{'feedback': {'condition_research': deepcopy(research)}},
                   {'feedback': {'condition_research': changed}}]}
    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')
    first = context['research']['llm_inquiries']['author']['condition_research']
    repeated, updated = [event['feedback']['condition_research'] for event in context['events']]
    assert 'preview' in first['candidates']
    assert 'preview' not in repeated['candidates']
    assert read_reference(tmp_path, repeated['candidates']) == research['candidates']
    assert 'preview' in updated['candidates']
    assert updated['selected_experiment'] == changed['selected_experiment']
    assert read_reference(tmp_path, updated['candidates']) == changed['candidates']


def test_repeated_source_comparison_shares_full_body_but_keeps_actions_and_new_verdicts(tmp_path):
    comparison = {'status': 'family_research_needed', 'comparison_status': 'inconclusive',
        'binding_path': 'shared/binding.py', 'response_count': 315,
        'candidates': [{**deepcopy(condition_research()['candidates'][0]), 'id': f'fit_{index}'}
                       for index in range(30)],
        'revision_action': {'action': 'run', 'arguments': {'operator': 'propose_explanations',
                                                        'path': 'shared/binding.py'}},
        'new_responses': [{'crystal': 5, 'frequency': 12., 'activated': False, 'evidence': [315]}]}
    changed = deepcopy(comparison)
    changed['comparison_status'] = 'refuted'
    changed['candidates'][0]['conflict_count'] = 2
    request = {'evidence': {'observations_file': 'laboratory_observations.json'},
        'research': {'llm_inquiries': {'first': {'comparison_result': comparison},
            'mirror': {'comparison_result': deepcopy(comparison)},
            'revised': {'comparison_result': changed}}}}
    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')
    first, repeated, updated = [inquiry['comparison_result']
        for inquiry in context['research']['llm_inquiries'].values()]
    assert 'candidates' in first
    assert 'candidates' not in repeated
    assert repeated['revision_action'] == comparison['revision_action']
    assert repeated['new_responses'] == comparison['new_responses']
    assert read_reference(tmp_path, repeated) == comparison
    assert updated['comparison_status'] == 'refuted'
    assert updated['candidates']['preview'][0]['status'] == 'refuted'
