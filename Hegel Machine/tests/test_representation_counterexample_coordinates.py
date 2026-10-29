"""A paid representation conflict changes proposals, not the quality contract."""

from copy import deepcopy
from dataclasses import replace
from random import Random

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.representation_diagnostics import representation_ambiguity
from hegel_machine.cognition.representation_inquiry import load_native_representation_result
from hegel_machine.cognition.temporal_operator_agent import (
    export_temporal_program, _PROGRAM_REPRESENTATION_RUNNER,
)
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_native_representation_routing import PublicOnlyLab


def setup(tmp_path, monkeypatch):
    # Isolate the newly connected direction from the two unchanged generators.
    monkeypatch.setattr(UnifiedResearchAgent, '_coordinate_claims', lambda *args: [])
    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', lambda *args, **kwargs: [])
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='hegel', source_task=False,
        config=replace(InquiryLLMConfig(), compute_budget=100000, max_model_calls=0))
    rng = Random(31)
    agent.observations, agent.rows = [], []
    for index in range(5):
        sequence = [rng.uniform(-1., 1.) for _ in range(12)]
        if index == 0:
            sequence[:3] = [1., -.8, .2]
        elif index == 1:
            sequence[:3] = [-.8, 1., .2]
        history, rows = [], []
        for value in sequence:
            rows.append({'features': {'u': value}, 'history': list(history),
                'target': history[-1]['u'] if history else 0.})
            history.append({'u': value})
        first = len(agent.rows)
        agent.observations.append({'action': {'kind': 'trajectory', 'channels': [],
            'sequence': [{'u': value} for value in sequence]}, 'rows': rows,
            'row_indices': list(range(first, first + len(rows)))})
        agent.rows.extend(rows)
    parent = Program('window_sum', lag=3, children=(Program('feature', name='u'),))
    agent._save('old.py', export_temporal_program(parent))
    module = {}
    exec(export_temporal_program(parent) + _PROGRAM_REPRESENTATION_RUNNER, module)
    diagnostic = representation_ambiguity(
        [module['represent'](row) for row in agent.rows], agent.rows)
    assert diagnostic['conflicting_groups'] >= 1
    question = {'id': 'representation:old', 'question': 'Recover the lost public distinction',
        'kind': 'representation', 'origin': 'computed_representation_conflict',
        'parent_id': 'task', 'depends_on': [], 'representation_diagnostic': diagnostic,
        'use_sites': [{'kind': 'supply_component', 'model': 'old.py', 'slot': 'representation'}]}
    arguments = {'operator': 'research_route', 'research_question': question,
        'theories': [{'id': 'old', 'program': parent.to_dict()}],
        'native_representation_plan': {'parent_theory_ids': ['old'],
            'validation_records': [4], 'compute_limit': 30000, 'max_candidates': 4}}
    return agent, parent, arguments, PublicOnlyLab()


def test_common_conflict_proposes_history_coordinates_used_by_actual_predictor(tmp_path, monkeypatch):
    agent, parent, arguments, lab = setup(tmp_path, monkeypatch)
    original = (agent.workspace / 'old.py').read_bytes()
    result = agent._execute('run', arguments, lab)
    native = load_native_representation_result(result['native_representation_ref'], agent.workspace)
    direction = next(item for item in native['search_directions']
        if item['direction'] == 'witness_coordinates')
    assert direction['completed_fits'] > 0 and 0 < direction['compute_cost'] <= direction['compute_limit']
    assert native['selected']['validation_mse'] < 1e-10
    assert not native['selected']['retained']
    selected = next(item for item in native['candidates'] if item['id'] == native['selected']['id'])
    check = selected['representation_witness_check']
    assert check['status'] == 'distinguishes_inputs_only'
    assert not check['prediction_quality_established']
    namespace = {}
    exec((agent.workspace / result['selected_model']).read_text() + _PROGRAM_REPRESENTATION_RUNNER, namespace)
    left, right = agent.rows[2], agent.rows[14]
    assert namespace['represent'](left) != namespace['represent'](right)
    assert namespace['predict_features'](namespace['represent'](left)) != namespace['predict_features'](
        namespace['represent'](right))
    assert namespace['predict_features'](namespace['represent'](left)) == namespace['predict'](left)
    assert namespace['predict_features'](namespace['represent'](right)) == namespace['predict'](right)
    assert (agent.workspace / 'old.py').read_bytes() == original
    assert not result['representation_child_resolved'] and not result['parent_revision_completed']
    assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0
    assert result['next_action']['arguments']['operator'] == 'research_choice'


def test_witness_separation_alone_does_not_override_full_development_comparison(tmp_path, monkeypatch):
    agent, parent, arguments, lab = setup(tmp_path, monkeypatch)
    first = agent._execute('run', arguments, lab)
    native = load_native_representation_result(first['native_representation_ref'], agent.workspace)
    def fitted_programs(study):
        candidates = list(study['candidates'])
        for action in study['actions']:
            candidates.extend(action.get('unexported_candidates', []))
        return {Program.from_dict(item['program']) for item in candidates}

    programs = fitted_programs(native)
    # The holdout now supports the old model. Its labels must affect selection,
    # never the proposed history coordinate or fitting parameters.
    for row in agent.observations[4]['rows']:
        row['target'] = parent.predict(Observation(row['features'], 0., tuple(row['history'])))
    agent.rows = [row for record in agent.observations for row in record['rows']]
    # This is an independent fixture execution, not cache invalidation research.
    from hegel_machine.cognition.representation_inquiry import execute_native_representation_inquiry
    from hegel_machine.cognition.models import TaskSpec
    second = execute_native_representation_inquiry(agent, deepcopy(arguments['research_question']),
        spec=TaskSpec('fixture', '', ('u',), (), ()), parent_programs={'old': parent},
        validation_records=[4], compute_limit=30000, max_candidates=4)
    # The output cap follows full development loss, so exports may change. All fitted
    # parameters, including the retained non-exported alternatives, must not.
    assert fitted_programs(second) == programs
    assert second['retained_comparison'][0]['validation_mse'] == 0.
    assert second['selected']['development_mse'] <= min(
        item['development_mse'] for item in
        [*second['retained_comparison'], *second['candidates']]) + 1e-10
    assert not second['selected']['selection']['independent_confirmation']
    assert any(item.get('representation_witness_check', {}).get('status') == 'distinguishes_inputs_only'
        for item in second['candidates'])


def test_validation_only_witness_does_not_supply_training_coordinates(tmp_path, monkeypatch):
    agent, parent, arguments, lab = setup(tmp_path, monkeypatch)
    arguments['native_representation_plan']['validation_records'] = [0, 1]
    result = agent._execute('run', arguments, lab)
    native = load_native_representation_result(result['native_representation_ref'], agent.workspace)
    assert [item['direction'] for item in native['search_directions']] == [
        'coordinates', 'retained_composition']
    assert native['selected']['retained']


def test_same_paid_scene_reuses_representation_candidates_without_refitting(tmp_path, monkeypatch):
    agent, parent, arguments, lab = setup(tmp_path, monkeypatch)
    first = agent._execute('run', arguments, lab)
    before = dict(agent.costs)
    second = agent._execute('run', arguments, lab)
    assert second['cache_hit'] and second['new_compute_cost'] == 0
    assert second['exported_models'] == first['exported_models']
    assert agent.costs == before
