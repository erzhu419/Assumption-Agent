"""Keep paid proposal work while funding a complete next research stage."""

from copy import deepcopy

import pytest

from hegel_machine.cognition import representation_inquiry
from hegel_machine.cognition.checkpoint_storage import write_json_checkpoint
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, TaskSpec
from hegel_machine.cognition.research_programs import _Fitter
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_representation_counterexample_coordinates import setup


def _spec():
    return TaskSpec('paid_reuse_fixture', '', ('u',), (), (), feature_bounds={'u': (-1., 1.)})


def _paid_scene(tmp_path, monkeypatch):
    agent, parent, arguments, _ = setup(tmp_path, monkeypatch)
    calls = {'coordinates': 0, 'retained_composition': 0, 'witness_coordinates': 0}
    u = Program('feature', name='u')

    def propose(direction, training, budget, term):
        calls[direction] += 1
        fitted = _Fitter(training, budget, .5).fit((term,))
        assert fitted is not None
        return [{'candidate': Candidate(direction, fitted['program'], direction),
                 'training_loss': fitted['loss']}]

    def coordinates(self, spec, training, budget, actions):
        return propose('coordinates', training, budget, u)

    def composition(self, spec, training, budget, parents, actions, **kwargs):
        return propose('retained_composition', training, budget,
                       Program('window_sum', lag=2, children=(u,)))

    def witness(*args, **kwargs):
        calls['witness_coordinates'] += 1
        return []

    monkeypatch.setattr(UnifiedResearchAgent, '_coordinate_claims', coordinates)
    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', composition)
    monkeypatch.setattr(representation_inquiry, '_witness_coordinate_claims', witness)
    question = deepcopy(arguments['research_question'])
    original = representation_inquiry.execute_native_representation_inquiry(
        agent, question, spec=_spec(), parent_programs={'old': parent},
        validation_records=[4], compute_limit=12000, max_candidates=4)
    assert len(original['candidates']) == 2
    payload = tmp_path / 'paid_scene.json'
    write_json_checkpoint(payload, {'observations': agent.observations,
                                   'parents': {'old': parent.to_dict()}})
    reference = {'file': str(payload.resolve())}
    reuse = {
        'source_ref': question['native_representation_research'][-1]['source_ref'],
        'paid_observations_ref': {**reference, 'json_path': ['observations']},
        'parent_program_refs': {'old': {**reference, 'json_path': ['parents', 'old']}},
        'public_spec': {'feature_names': ['u'], 'risk_quantile': .5, 'max_probe_length': 12,
                        'feature_bounds': {'u': [-1., 1.]}, 'context': {}},
        'directions': {'coordinates': [0], 'retained_composition': [1]},
    }
    return agent, parent, arguments, original, reuse, calls


def test_paid_old_directions_are_reused_while_new_relation_work_runs(tmp_path, monkeypatch):
    agent, parent, arguments, original, reuse, calls = _paid_scene(tmp_path, monkeypatch)
    source_path = agent.workspace / reuse['source_ref']['file']
    original_bytes = source_path.read_bytes()
    before = dict(agent.costs)
    result = representation_inquiry.execute_native_representation_inquiry(
        agent, deepcopy(arguments['research_question']), spec=_spec(),
        parent_programs={'old': parent}, validation_records=[4],
        compute_limit=12000, max_candidates=4, paid_direction_reuse=reuse)

    assert calls == {'coordinates': 1, 'retained_composition': 1, 'witness_coordinates': 2}
    assert [item['program'] for item in result['candidates']] == [
        item['program'] for item in original['candidates']]
    assert [item['validation_mse'] for item in result['candidates']] == [
        item['validation_mse'] for item in original['candidates']]
    for direction in ('coordinates', 'retained_composition'):
        retained = next(item for item in result['search_directions'] if item['direction'] == direction)
        paid = next(item for item in original['search_directions'] if item['direction'] == direction)
        assert retained['status'] == 'paid_candidates_reused'
        assert retained['compute_cost'] == retained['completed_fits'] == 0
        assert retained['source_compute_cost'] == paid['compute_cost'] > 0
        assert retained['candidate_refs']
    assert 0 < result['compute_cost'] < original['compute_cost']
    assert agent.costs['compute_cost'] - before['compute_cost'] == result['compute_cost']
    assert agent.costs['model_calls'] == agent.costs['observation_cost'] == 0
    assert source_path.read_bytes() == original_bytes


@pytest.mark.parametrize('changed', ['target', 'history', 'split', 'parent_parameters',
                                    'public_spec', 'replay_scope'])
def test_paid_direction_scope_change_stops_before_new_research(tmp_path, monkeypatch, changed):
    agent, parent, arguments, original, reuse, calls = _paid_scene(tmp_path, monkeypatch)
    spec, validation_records, replay_scope = _spec(), [4], 'procedure'
    if changed == 'target':
        agent.observations[0]['rows'][0]['target'] += .125
    elif changed == 'history':
        agent.observations[0]['rows'][1]['history'][0]['u'] += .125
    elif changed == 'split':
        validation_records = [3]
    elif changed == 'parent_parameters':
        parent = Program('mul', children=(Program('const', value=2.), parent))
    elif changed == 'public_spec':
        spec = TaskSpec('paid_reuse_fixture', '', ('u',), (), (), max_probe_length=13)
    else:
        replay_scope = 'retained_coordinates'
    before, call_counts = dict(agent.costs), dict(calls)
    with pytest.raises(ValueError):
        representation_inquiry.execute_native_representation_inquiry(
            agent, deepcopy(arguments['research_question']), spec=spec,
            parent_programs={'old': parent}, validation_records=validation_records,
            compute_limit=12000, max_candidates=4, retained_replay_scope=replay_scope,
            paid_direction_reuse=reuse)
    assert agent.costs == before
    assert calls == call_counts


def test_unused_earlier_direction_budget_reaches_witness_stage(tmp_path, monkeypatch):
    agent, parent, arguments, _ = setup(tmp_path, monkeypatch)
    observed = []

    def witness(spec, training, parents, groups, budget, actions, **kwargs):
        observed.append(budget.remaining)
        # This complete fixture stage exceeds the former isolated one-third cap.
        assert budget.spend(7000)
        return []

    monkeypatch.setattr(representation_inquiry, '_witness_coordinate_claims', witness)
    result = representation_inquiry.execute_native_representation_inquiry(
        agent, deepcopy(arguments['research_question']), spec=_spec(),
        parent_programs={'old': parent}, validation_records=[4],
        compute_limit=12000, max_candidates=4)
    assert len(observed) == 1 and observed[0] > 7000
    direction = next(item for item in result['search_directions']
                     if item['direction'] == 'witness_coordinates')
    assert direction['compute_cost'] == 7000
    assert direction['last_denied_request'] is None
    assert result['comparison_completed']
    assert result['selected']['retained']
    assert result['compute_cost'] <= 12000


def test_nested_search_exhaustion_survives_successful_candidate_comparison(tmp_path, monkeypatch):
    from hegel_machine.cognition import conditional_coordinates, research_programs

    agent, parent, arguments, _ = setup(tmp_path, monkeypatch)
    u = Program('feature', name='u')

    def finish_one_fit_then_exhaust(training, budget, term, progress):
        fitted = _Fitter(training, budget, .5).fit((term,))
        assert fitted is not None
        progress.update(last_finished_stage='joint_fit', delivered_candidates=1)
        assert not budget.can_spend(budget.remaining + 1)
        return [{'candidate': Candidate('finished_joint_fit', fitted['program'], 'fixture'),
                 'training_loss': fitted['loss']}]

    def residual(spec, training, budget, calls, *, progress, **kwargs):
        return finish_one_fit_then_exhaust(training, budget,
            Program('window_sum', lag=2, children=(u,)), progress)

    def conditional(spec, training, budget, *, receipt, **kwargs):
        return finish_one_fit_then_exhaust(training, budget, u, receipt)

    monkeypatch.setattr(research_programs, 'propose_research_programs', residual)
    monkeypatch.setattr(conditional_coordinates, 'propose_conditional_coordinates', conditional)
    result = representation_inquiry.execute_native_representation_inquiry(
        agent, deepcopy(arguments['research_question']), spec=_spec(),
        parent_programs={'old': parent}, validation_records=[4],
        compute_limit=90000, max_candidates=4)

    assert result['comparison_completed'] and result['candidates']
    assert result['last_denied_request'] is None
    assert result['research_completion'] == 'bounded_search_incomplete'
    branches = next(action['branches'] for action in result['actions']
                    if action['action'] == 'compose_witness_relations')
    assert {branch['direction'] for branch in branches} == {
        'residual_composition', 'conditional_response'}
    for branch in branches:
        assert branch['completion'] == 'incomplete_compute'
        assert branch['proposed_candidates'] == branch['completed_fits'] == 1
        assert branch['compute_cost'] > 0
        propagated = next(item for item in result['incomplete_search_stages']
                          if item['stage'] == branch['direction'])
        assert propagated['last_denied_request'] == branch['last_denied_request']
        assert propagated['progress']['last_finished_stage'] == 'joint_fit'
    assert result['compute_cost'] == agent.costs['compute_cost'] <= 90000


def test_common_route_forwards_declared_paid_directions(tmp_path, monkeypatch):
    agent, parent, arguments, original, reuse, calls = _paid_scene(tmp_path, monkeypatch)
    from test_native_representation_routing import PublicOnlyLab

    arguments['native_representation_plan']['paid_direction_reuse'] = reuse
    routed = agent._execute('run', arguments, PublicOnlyLab())
    result = representation_inquiry.load_native_representation_result(
        routed['native_representation_ref'], agent.workspace)

    assert not routed['cache_hit']
    assert calls == {'coordinates': 1, 'retained_composition': 1, 'witness_coordinates': 2}
    assert len(routed['exported_models']) == len(original['candidates']) == 2
    assert {item['direction'] for item in result['search_directions']
            if item.get('status') == 'paid_candidates_reused'} == {
                'coordinates', 'retained_composition'}


def _paid_prediction_scene(tmp_path):
    from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
    from test_python_paid_score_reuse import choose, scene

    source, lab = scene(tmp_path / 'source')
    program = Program('feature', name='u')
    text = export_temporal_program(program, feature_names=('u',))
    source._save('source_candidate.py', text)
    paid = choose(source, lab, ['source_candidate.py'])
    reference = paid['evaluations'][0]['paid_score_reuse']['execution_basis_ref']
    destination, destination_lab = scene(tmp_path / 'destination')
    destination._save('new_native_export.py', text)
    return source, reference, paid, destination, destination_lab


def test_imported_full_pool_score_reuses_changed_export_path_without_execution(tmp_path, monkeypatch):
    from pathlib import Path
    from hegel_machine.cognition.paid_score_history import retain_paid_assessment_references
    from test_python_paid_score_reuse import choose, no_execution, scoring_costs

    source, reference, paid, destination, lab = _paid_prediction_scene(tmp_path)
    original = Path(reference['file']).read_bytes()
    before = deepcopy(destination.costs)
    first = retain_paid_assessment_references(destination, [reference])
    assert first['imported_assessments'] == first['imported_scopes'] == 1
    assert first['existing_assessments'] == first['compute_cost'] == first['new_scores'] == 0
    assert destination.costs == before
    index = destination.workspace / '_paid_python_scores/_evaluated_model/assessments.json'
    imported_bytes = index.read_bytes()
    second = retain_paid_assessment_references(destination, [reference])
    assert second['existing_assessments'] == 1 and second['imported_assessments'] == 0
    assert index.read_bytes() == imported_bytes

    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    costs = scoring_costs(destination)
    result = choose(destination, lab, ['new_native_export.py'])
    assert result['new_scored_count'] == 0 and result['reused_score_count'] == 1
    assert scoring_costs(destination) == costs
    evaluation = result['evaluations'][0]
    assert evaluation['mse'] == paid['evaluations'][0]['mse']
    reuse = evaluation['paid_score_reuse']
    assert reuse['execution_basis_ref'] == reference
    assert reuse['match_basis'] == 'identical_closed_literal_execution'
    assert reuse['original_costs']['compute_cost'] == len(source.rows)
    assert reuse['additional_scoring_compute'] == 0
    assert Path(reference['file']).read_bytes() == original


def test_imported_assessment_does_not_override_current_row_scope(tmp_path):
    from dataclasses import replace
    from hegel_machine.cognition.paid_score_history import retain_paid_assessment_references
    from test_python_paid_score_reuse import choose

    source, reference, paid, destination, lab = _paid_prediction_scene(tmp_path)
    retain_paid_assessment_references(destination, [reference])
    destination.rows[0] = replace(destination.rows[0], target=17.)
    before = destination.costs['compute_cost']
    result = choose(destination, lab, ['new_native_export.py'])
    assert result['reused_score_count'] == 0 and result['new_scored_count'] == 1
    assert destination.costs['compute_cost'] - before == len(destination.rows)
    assert result['evaluations'][0]['mse'] != paid['evaluations'][0]['mse']
