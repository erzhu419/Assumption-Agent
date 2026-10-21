"""Paired parameter updates through Common; fixtures are not science evidence."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.theory_consolidation import _save, install_theory_library
from test_question_routing import CONTEXT, future_question, probe, route, theory
from test_theory_consolidation import agent_scene, assert_no_new_science


PARAMETER_SEARCH = {
    # The original refit stream emits constant first, then affine. Keep both.
    'operations': ['refit'], 'max_structures': 2,
    'risk_quantile': .5, 'tolerance': 1e-8,
}


def scaled(coefficient):
    return Program('mul', children=(Program('const', value=float(coefficient)),
                                    Program('feature', name='u')))


def retained_parameter_changes(agent, *, corrupt_held=False):
    """Six executed fixture revisions, three per actual physical source.

    u=2 deliberately makes world responses different from coefficient labels.
    Each source's final record is reserved by the parameter learner, rather
    than selected after inspecting a learned relation.
    """
    episodes, declarations, references, examples, paid = [], [], [], {}, []
    for source_index, coefficients in enumerate(((1., 2., 3.), (4., 5., 6.))):
        physical = f'physical_fixture_{source_index}'
        for revision_index, before_coefficient in enumerate(coefficients):
            index = len(episodes)
            identifier = f'{physical}_revision_{revision_index}'
            after_coefficient = 2. * before_coefficient + 1.
            if corrupt_held and revision_index == 2:
                after_coefficient += 100.
            before, after = scaled(before_coefficient), scaled(after_coefficient)
            row = {'features': {'u': 2.}, 'target': 2. * after_coefficient, 'history': []}
            paid.append(row)
            episodes.append({
                'task_id': identifier, 'program': after.to_dict(),
                'actions': [{'action': 'revise', 'compute_cost': 2, 'details': {
                    'before_program': before.to_dict(), 'after_program': after.to_dict()}}],
                'questions': [], 'status': 'controlled_retained_revision',
            })
            reference = {'file': 'parameter_edits.json',
                         'json_path': ['episodes', index], 'action_cutoff': 0}
            references.append(reference)
            declarations.append(theory(identifier, after,
                context_ref={**deepcopy(CONTEXT), 'source_id': physical},
                context_family_ref='controlled_parameter_revision',
                parameter_paths=[[0]],
                episode_ref={'question_id': identifier, 'meaning_version': 1},
                program_source_ref={'file': reference['file'], 'json_path': [
                    'episodes', index, 'actions', 0, 'details', 'after_program']}))
            examples[identifier] = [{
                'source_ref': {'file': 'source_parameter_paid.json', 'json_path': [index]},
                'observation': row, 'tolerance': 1e-10,
            }]
    agent._save('parameter_edits.json', json.dumps({'episodes': episodes}))
    agent._save('source_parameter_paid.json', json.dumps(paid))
    return declarations, references, examples


def consolidate_parameters(agent, lab, *, corrupt_held=False):
    declarations, references, examples = retained_parameter_changes(
        agent, corrupt_held=corrupt_held)
    return agent._execute('run', {
        'operator': 'theory_consolidation', 'theme': 'controlled_parameter_revision',
        'object_type': 'program_revision_relation', 'theories': declarations,
        'source_theory_ids': [item['id'] for item in declarations],
        'episode_sources': references, 'counterexamples': examples,
        'mission': {'purpose': 'theory_learning',
                    'theory_learning_allocation': {'compute': 35000}},
        'allocation': {'compute': 35000, 'max_groups': 12,
                       'parameter_relations': deepcopy(PARAMETER_SEARCH)},
    }, lab)


def parameter_candidates(agent):
    return [item for item in agent.theory_consolidation.get('candidates', {}).values()
            if item.get('kind') == 'program_revision_relation' and item.get('parameter_relations')]


def transfer_parameter_update(agent, lab, candidate, coefficient, expected_response):
    parent_id = f'actual_parent_{coefficient}'
    register_theories(agent, [theory(parent_id, scaled(coefficient)),
        theory('ordinary_coefficient', Program('const', value=99.), output_unit='dimensionless')])
    row = {'features': {'u': 2.}, 'target': expected_response, 'history': []}
    filename = f'target_parameter_{coefficient}.json'
    agent._save(filename, json.dumps(row))
    arguments = {
        'operator': 'theory_transfer', 'candidate_id': candidate['id'],
        'theory_id': f'parameter_update_{coefficient}', 'parent_theory_id': parent_id,
        'component_theory_ids': ['ordinary_coefficient'],
        'context_ref': deepcopy(CONTEXT), 'output_unit': 'V',
        'allocation': {'compute': 3000},
        'evidence_checks': [{'source_ref': {'file': filename, 'json_path': []},
            'observation': row, 'context_ref': deepcopy(CONTEXT), 'tolerance': 1e-8}],
    }
    before = deepcopy(agent.costs)
    result = agent._execute('run', arguments, lab)
    assert agent.costs['compute_cost'] - before['compute_cost'] == result['new_compute_cost']
    assert agent.theory_index[parent_id]['program'] == scaled(coefficient).to_dict()
    assert agent.config.confirmation_mse_threshold == 1e-10
    return result, parent_id


def derived_updates(result):
    return [item for item in result['generated_theories'] if any(
        binding.get('parameter_relation')
        for binding in item['binding']['components'].values())]


def test_paired_parameter_map_reaches_frozen_transfer_and_PR2_with_real_parent_inputs(tmp_path):
    source, source_lab = agent_scene(tmp_path / 'source')
    before = source.costs['compute_cost']
    built = consolidate_parameters(source, source_lab)
    assert built['new_compute_cost'] == source.costs['compute_cost'] - before
    learned = parameter_candidates(source)
    assert learned, 'All constants changed: a learned numeric link must admit this joint revision.'
    candidate = learned[0]
    assert len(candidate['source_ids']) == 2
    assert all(item['before_exact'] and item['after_exact']
               for item in candidate['source_reconstruction'])
    assert all(item['scientific_support'] is False for item in candidate['parameter_relations'])
    assert candidate['constraint_hypotheses'][0]['id'] == 'observed_fixed'
    # Parent-only coefficient hole and after-only coefficient hole are distinct.
    after_holes = set(candidate['definition']['arguments'])
    assert candidate['parameter_relations'][0]['output_hole'] not in candidate['parent_holes']
    assert set(candidate['parent_holes']) <= after_holes

    target, target_lab = agent_scene(tmp_path / 'target')
    target.theory_consolidation = install_theory_library(target, {
        'file': str(source.workspace / 'shared/theory_library.json'), 'json_path': []})
    _save(target)
    frozen = target.theory_consolidation['candidates'][candidate['id']]
    first, parent_id = transfer_parameter_update(target, target_lab, frozen, 8., 34.)
    derived = derived_updates(first)
    assert derived and any(item['status'] == 'candidate_registered' for item in derived)
    chosen = next(item for item in derived if item['status'] == 'candidate_registered')
    changed = Program.from_dict(target.theory_index[chosen['theory_id']]['program'])
    assert changed.predict(Observation({'u': 1.}, 0.)) == pytest.approx(17., abs=1e-8)
    assert not chosen['binding'].get('fitting'), 'A derived coefficient must not be refitted on world y.'
    assert any(Program.from_dict(target.theory_index[item['theory_id']]['program']) == scaled(99.)
               for item in first['generated_theories']), 'The old ordinary component branch remains.'
    second, _ = transfer_parameter_update(target, target_lab, frozen, 9., 38.)
    changed_again = next(item for item in derived_updates(second) if item['status'] == 'candidate_registered')
    updated = Program.from_dict(target.theory_index[changed_again['theory_id']]['program'])
    assert updated.predict(Observation({'u': 1.}, 0.)) == pytest.approx(19., abs=1e-8)
    assert updated != changed

    question = future_question(target)
    question['candidate_theory_ids'] = [parent_id, chosen['theory_id']]
    routed = route(target, target_lab, question=question, experiments=[probe(target)])
    assert chosen['theory_id'] in {identifier for group in routed['selection']['groups']
                                  for identifier in group['theory_ids']}
    assert routed['parent_use']['status'] == 'parent_query_prepared'
    assert_no_new_science(source, source_lab)
    assert_no_new_science(target, target_lab)


def test_each_physical_source_held_revision_can_reject_the_parameter_map(tmp_path):
    agent, lab = agent_scene(tmp_path)
    result = consolidate_parameters(agent, lab, corrupt_held=True)
    assert not parameter_candidates(agent)
    searches = result['parameter_relation_searches']
    assert searches and any(item['source_record_count'] >= 6 for item in searches)
    attempts = [attempt for search in searches for attempt in search['attempts']]
    fitted = [check for attempt in attempts for check in attempt.get('checks', [])]
    held = [check for fit in fitted for check in fit.get('held', [])]
    assert held and any(check.get('status') in {'failed', 'violated'} for check in held)
    assert all(check.get('source_ref') or check.get('after_ref') for check in held)
    assert agent.config.confirmation_mse_threshold == 1e-10
    assert_no_new_science(agent, lab)


def test_world_response_checks_refute_the_derived_program_without_refitting_its_coefficient(tmp_path):
    agent, lab = agent_scene(tmp_path)
    consolidate_parameters(agent, lab)
    candidate = parameter_candidates(agent)[0]
    reconstruction = deepcopy(candidate['source_reconstruction'])
    result, _ = transfer_parameter_update(agent, lab, candidate, 8., 9999.)
    derived = derived_updates(result)
    assert derived and all(item['status'] == 'refuted' for item in derived)
    for item in derived:
        program = Program.from_dict(agent.theory_index[item['theory_id']]['program'])
        assert program.predict(Observation({'u': 1.}, 0.)) == pytest.approx(17., abs=1e-8)
        assert item['checks'] and any(check['status'] == 'failed' for check in item['checks'])
        assert not item['binding'].get('fitting')
    assert candidate['source_reconstruction'] == reconstruction
    assert_no_new_science(agent, lab)
