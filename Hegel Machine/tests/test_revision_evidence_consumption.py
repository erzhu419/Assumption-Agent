"""Counterexample-conditioned edit maps through Common, using toy revisions only."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.theory_consolidation import _save, install_theory_library
from test_question_routing import theory
from test_revision_parameter_consumption import PARAMETER_SEARCH, derived_updates, scaled
from test_theory_consolidation import agent_scene, assert_no_new_science


ROLES = {'u': {'role': 'public_input', 'unit': 'dimensionless', 'time': 'current'}}
CONTEXT = {'scene': 'evidence_edit_fixture', 'output_unit': 'dimensionless',
           'reset': 'per_trajectory'}


def declaration(identifier, program, **fields):
    return theory(identifier, program, role_signature=deepcopy(ROLES),
                  output_unit='dimensionless', context_ref=deepcopy(CONTEXT), **fields)


def build_evidence_edit_library(agent, lab, *, corrupt_held=False, parameter_search=None):
    """Same before parameters in two sources, different measured residuals.

    The world observation is 2*a+r, the saved after coefficient is a+r.
    These are deliberately different quantities. Both source after Programs
    remain wrong at this counterexample; no scientific support is conferred.
    Five revisions per source leave eight training rows for the unchanged
    joint-affine grammar and one predetermined held revision per source.
    """
    episodes, declarations, sources, examples, paid = [], [], [], {}, []
    for source_index, residual in enumerate((1., 3.)):
        physical = f'evidence_physical_source_{source_index}'
        for revision_index, coefficient in enumerate((1., 2., 3., 4., 5.)):
            index = len(episodes)
            identifier = f'{physical}_revision_{revision_index}'
            after_coefficient = coefficient + residual
            if corrupt_held and revision_index == 4:
                after_coefficient += 100.
            before, after = scaled(coefficient), scaled(after_coefficient)
            observation = {'features': {'u': 2.}, 'target': 2. * coefficient + residual,
                           'history': []}
            paid.append(observation)
            episodes.append({'task_id': identifier, 'program': after.to_dict(),
                'actions': [{'action': 'revise', 'compute_cost': 2, 'details': {
                    'before_program': before.to_dict(), 'after_program': after.to_dict()}}],
                'questions': [], 'status': 'controlled_retained_revision'})
            source = {'file': 'evidence_edits.json', 'json_path': ['episodes', index],
                      'action_cutoff': 0}
            sources.append(source)
            item = declaration(identifier, after,
                context_family_ref='counterexample_conditioned_parameter_edit',
                parameter_paths=[[0]],
                episode_ref={'question_id': identifier, 'meaning_version': 1},
                program_source_ref={'file': source['file'], 'json_path': [
                    'episodes', index, 'actions', 0, 'details', 'after_program']})
            item['context_ref']['source_id'] = physical
            declarations.append(item)
            examples[identifier] = [{
                'source_ref': {'file': 'evidence_edit_paid.json', 'json_path': [index]},
                'observation': observation, 'tolerance': 1e-10}]
    agent._save('evidence_edits.json', json.dumps({'episodes': episodes}))
    agent._save('evidence_edit_paid.json', json.dumps(paid))
    before = agent.costs['compute_cost']
    result = agent._execute('run', {
        'operator': 'theory_consolidation', 'theme': 'counterexample_conditioned_parameter_edit',
        'object_type': 'program_revision_relation', 'theories': declarations,
        'source_theory_ids': [item['id'] for item in declarations],
        'episode_sources': sources, 'counterexamples': examples,
        'mission': {'purpose': 'theory_learning',
                    'theory_learning_allocation': {'compute': 35000}},
        'allocation': {'compute': 35000, 'max_groups': 12,
                       'parameter_relations': deepcopy(
                           PARAMETER_SEARCH if parameter_search is None else parameter_search)},
    }, lab)
    assert result['new_compute_cost'] == agent.costs['compute_cost'] - before
    return result


def evidence_candidates(agent):
    return [candidate for candidate in agent.theory_consolidation['candidates'].values()
            if candidate.get('parameter_relations') and any(
                'evidence:residual' in relation.get('evidence_inputs', [])
                for relation in candidate['parameter_relations'])]


def transfer_with_residual(agent, lab, candidate, residual, *, missing_reference=False,
                           coefficient=8.):
    parent_id = ('unchanged_actual_parent' if coefficient == 8.
                 else f'unchanged_actual_parent_{coefficient}')
    if parent_id not in agent.theory_index:
        register_theories(agent, [declaration(parent_id, scaled(coefficient)),
            declaration('ordinary_coefficient', Program('const', value=99.))])
    row = {'features': {'u': 2.}, 'target': 2. * coefficient + residual, 'history': []}
    filename = f'target_evidence_parent_{coefficient}_residual_{residual}.json'
    agent._save(filename, json.dumps(row))
    check = {'source_ref': {'file': filename, 'json_path': []}, 'observation': row,
             'context_ref': deepcopy(CONTEXT), 'tolerance': 1e-10}
    if missing_reference:
        check.pop('source_ref')
    before = agent.costs['compute_cost']
    result = agent._execute('run', {
        'operator': 'theory_transfer', 'candidate_id': candidate['id'],
        'theory_id': f'evidence_edit_{coefficient}_{residual}_{missing_reference}',
        'parent_theory_id': parent_id, 'component_theory_ids': ['ordinary_coefficient'],
        'context_ref': deepcopy(CONTEXT), 'output_unit': 'dimensionless',
        'allocation': {'compute': 3000}, 'evidence_checks': [check],
    }, lab)
    assert result['new_compute_cost'] == agent.costs['compute_cost'] - before
    assert agent.theory_index[parent_id]['program'] == scaled(coefficient).to_dict()
    assert agent.config.confirmation_mse_threshold == 1e-10
    return result


def test_same_parent_uses_real_counterexample_to_change_derived_after_from_frozen_library(tmp_path):
    source, source_lab = agent_scene(tmp_path / 'source')
    build_evidence_edit_library(source, source_lab)
    candidate, *_ = evidence_candidates(source)
    assert len(candidate['source_ids']) == 2
    assert all(item['before_exact'] and item['after_exact']
               for item in candidate['source_reconstruction'])
    assert all(relation['scientific_support'] is False for relation in candidate['parameter_relations'])
    allowed_evidence = {
        'evidence:feature:u', 'evidence:prediction', 'evidence:residual', 'evidence:history_depth'}
    assert all(set(relation['input_holes']) <= set(candidate['parent_holes'])
               and set(relation.get('evidence_inputs', [])) <= allowed_evidence
               for relation in candidate['parameter_relations'])
    assert any(check['before']['failed'] and check['after']['failed']
               for check in candidate['trigger_checks'])

    target, target_lab = agent_scene(tmp_path / 'target')
    target.theory_consolidation = install_theory_library(target, {
        'file': str(source.workspace / 'shared/theory_library.json'), 'json_path': []})
    _save(target)
    frozen = target.theory_consolidation['candidates'][candidate['id']]
    values = []
    for residual in (1., 3.):
        result = transfer_with_residual(target, target_lab, frozen, residual)
        generated = derived_updates(result)
        assert generated, 'Counterexample context must reach executable parameter components.'
        assert all(item['status'] == 'refuted' for item in generated)
        for item in generated:
            program = Program.from_dict(target.theory_index[item['theory_id']]['program'])
            value = program.predict(Observation({'u': 1.}, 0.))
            assert value == pytest.approx(8. + residual, abs=1e-8)
            values.append(value)
            assert item['checks'] and any(check['status'] == 'failed' for check in item['checks'])
            assignment = next(binding['parameter_relation']
                for binding in item['binding']['components'].values()
                if binding.get('parameter_relation'))
            assert assignment['evidence_inputs']['evidence:residual'] == residual
            assert assignment['evidence_refs']
            assert not item['binding'].get('fitting')
        assert any(Program.from_dict(target.theory_index[item['theory_id']]['program']) == scaled(99.)
                   for item in result['generated_theories']), 'Keep the pre-existing ordinary branch.'
    assert min(values) == pytest.approx(9., abs=1e-8)
    assert max(values) == pytest.approx(11., abs=1e-8)
    assert_no_new_science(source, source_lab)
    assert_no_new_science(target, target_lab)


@pytest.mark.parametrize('missing_reference,residual', [(True, 1.), (False, 0.)])
def test_missing_or_nonfailing_evidence_cannot_fabricate_contextual_revision(
        tmp_path, missing_reference, residual):
    agent, lab = agent_scene(tmp_path)
    build_evidence_edit_library(agent, lab)
    candidate, *_ = evidence_candidates(agent)
    result = transfer_with_residual(agent, lab, candidate, residual,
                                    missing_reference=missing_reference)
    assert not derived_updates(result)
    assert result['status'] in {'UNKNOWN', 'not_applicable'}
    assert agent.config.confirmation_mse_threshold == 1e-10
    assert_no_new_science(agent, lab)


def test_counterexample_conditioned_map_keeps_failed_held_revision_references(tmp_path):
    agent, lab = agent_scene(tmp_path)
    result = build_evidence_edit_library(agent, lab, corrupt_held=True)
    assert not evidence_candidates(agent)
    searches = result['parameter_relation_searches']
    assert any(search['source_record_count'] == 10 for search in searches)
    held = [check for search in searches for attempt in search['attempts']
            for fit in attempt.get('checks', []) for check in fit.get('held', [])]
    assert held and any(check['status'] == 'failed' for check in held)
    assert all(check.get('source_ref') or check.get('after_ref') for check in held)
    assert agent.config.confirmation_mse_threshold == 1e-10
    assert_no_new_science(agent, lab)


def test_joint_search_result_is_fitted_frozen_and_consumed_on_both_input_axes(tmp_path):
    source, source_lab = agent_scene(tmp_path / 'source')
    built = build_evidence_edit_library(source, source_lab,
        parameter_search={**PARAMETER_SEARCH, 'max_structures': 16})
    assert 0 < built['new_compute_cost'] <= 35000
    searches = [search for search in built['parameter_relation_searches']
                if search.get('input_scope') == 'sparse_parent_evidence_pairs']
    assert searches, 'Common consolidation must execute the additional joint search.'
    search = next(search for search in searches if any(
        mapping['input_holes'] and 'evidence:residual' in mapping['evidence_inputs']
        for mapping in search['accepted']))
    assert search['actual_compute_cost'] > 0
    assert any(attempt['input_holes'] and attempt['evidence_inputs'] and attempt.get('fitting')
               and attempt.get('checks') for attempt in search['attempts'])
    mapping = next(mapping for mapping in search['accepted']
                   if mapping['input_holes'] and 'evidence:residual' in mapping['evidence_inputs'])
    assert mapping['training_checks'] and mapping['held_checks']
    assert all(check['status'] == 'passed'
               for check in mapping['training_checks'] + mapping['held_checks'])
    original = source.theory_consolidation['candidates'][search['candidate_id']]
    # Locate this exact fitted map in the actual consolidated object. Earlier
    # stages may have found the same function; their presence cannot substitute
    # for this joint stage's provenance in the target binding.
    preceding = built['parameter_relation_searches'][:built['parameter_relation_searches'].index(search)]
    offset = sum(len(item['accepted']) for item in preceding
                 if item['candidate_id'] == search['candidate_id'])
    assert original['parameter_relations'][offset:offset + len(search['accepted'])] == search['accepted']
    indices = {offset + search['accepted'].index(mapping)}
    target, target_lab = agent_scene(tmp_path / 'target')
    target.theory_consolidation = install_theory_library(target, {
        'file': str(source.workspace / 'shared/theory_library.json'), 'json_path': []})
    _save(target)
    frozen = target.theory_consolidation['candidates'][original['id']]
    executable_fields = ('program', 'input_holes', 'evidence_inputs', 'input_contracts', 'output_hole')
    assert [{key: relation[key] for key in executable_fields}
            for relation in frozen['parameter_relations']] == [
        {key: relation[key] for key in executable_fields} for relation in original['parameter_relations']]
    values = {}
    for coefficient, residual in ((8., 1.), (8., 3.), (9., 1.)):
        result = transfer_with_residual(target, target_lab, frozen, residual,
                                        coefficient=coefficient)
        joint = []
        for generated in derived_updates(result):
            assignments = [binding['parameter_relation']
                for binding in generated['binding']['components'].values()
                if binding.get('parameter_relation')]
            if any(assignment['index'] in indices for assignment in assignments):
                joint.append(generated)
        assert joint, 'The frozen joint-stage map must reach an actual generated Program.'
        assert all(item['status'] == 'refuted' for item in joint)
        for generated in joint:
            program = Program.from_dict(target.theory_index[generated['theory_id']]['program'])
            value = program.predict(Observation({'u': 1.}, 0.))
            assert value == pytest.approx(coefficient + residual, abs=1e-8)
            assert any(check['status'] == 'failed' for check in generated['checks'])
            assert not generated['binding'].get('fitting')
            values[coefficient, residual] = value
        assert any(Program.from_dict(target.theory_index[item['theory_id']]['program']) == scaled(99.)
                   for item in result['generated_theories'])
    assert values[8., 1.] != values[8., 3.]
    assert values[8., 1.] != values[9., 1.]
    assert target.config.confirmation_mse_threshold == 1e-10
    assert_no_new_science(source, source_lab)
    assert_no_new_science(target, target_lab)
