"""PR3 through Common's provider-free operators and the existing PR2 pool.

PublicLab is a local fixture. No provider is installed and its acquisition
method is disabled before consolidation, transfer, or selection is called.
"""
from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.question_routing import resolved_theories
from test_inquiry_revision import PublicLab
from test_question_routing import CONTEXT, ROLES, SQUARE, future_question, probe, route, theory, scene as measured_scene


MISSION = {'purpose': 'theory_learning', 'theory_learning_allocation': {'compute': 20000}}


def agent_scene(tmp_path):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='hegel', source_task=False,
        config=InquiryLLMConfig(observation_budget=50, confirmation_rows=2,
            compute_budget=50000, execution_seconds=30., max_model_calls=0))
    lab = PublicLab()

    def no_acquisition(action):
        raise AssertionError('PR3 must not acquire even fixture observations')

    lab.execute = no_acquisition
    return agent, lab


def affine(slope=2., intercept=1.):
    return Program('add', children=(Program('mul', children=(Program('const', value=slope),
                Program('feature', name='u'))), Program('const', value=intercept)))


def sources(*, same_source=False):
    values = []
    for identifier, program in (('source_a', affine()), ('source_b', affine(4., -3.))):
        context = {**deepcopy(CONTEXT), 'source_id': 'same_source' if same_source else identifier}
        values.append(theory(identifier, program, context_ref=context,
            context_family_ref='declared_drive_theme', parameter_paths=[[0, 0], [1]],
            support={'status': 'candidate', 'evidence': [], 'context_ref': context}))
    return values


def consolidate(agent, lab, **extra):
    return agent._execute('run', {'operator': 'theory_consolidation', 'theme': 'affine_drive',
        'theories': sources(), 'source_theory_ids': ['source_a', 'source_b'],
        'mission': deepcopy(MISSION), 'allocation': {'compute': 20000, 'max_groups': 8}, **extra}, lab)


def library(agent, path='shared/theory_library.json'):
    return json.loads((agent.workspace / path).read_text())


def candidate(agent, identifier=None):
    stored = library(agent)['candidates']
    return stored[identifier] if identifier else next(iter(stored.values()))


def transfer(agent, lab, identifier, **extra):
    return agent._execute('run', {'operator': 'theory_transfer', 'candidate_id': identifier,
        'theory_id': 'transferred_affine', 'mapping': {'role_0': 'u'},
        'role_signature': deepcopy(ROLES), 'output_unit': 'V', 'context_ref': deepcopy(CONTEXT), **extra}, lab)


def assert_no_new_science(agent, lab):
    assert lab.paid == 0 and lab.actions == [] and agent.observations == []
    assert agent.costs['observation_cost'] == 0 and agent.costs['model_calls'] == 0
    assert not agent.confirmations and agent.model is None


def test_frozen_library_reaches_a_new_agent_and_changes_existing_PR2_prediction_and_selection(tmp_path):
    source, source_lab = agent_scene(tmp_path / 'source')
    result = consolidate(source, source_lab)
    assert result['status'] == 'completed'
    frozen, = result['frozen_candidates']
    original = candidate(source, frozen['candidate_id'])
    assert original['independent_source_count'] == 2
    assert all(item['exact'] for item in original['source_reconstruction'])
    target, target_lab = agent_scene(tmp_path / 'target')
    target._save('shared/theory_library.json', (source.workspace / result['library_path']).read_text())
    before = route(target, target_lab, question=future_question(target),
        theories=[theory('square', SQUARE)], experiments=[probe(target)])
    assert before['status'] == 'scoped_unknown' and before['selection']['scores'][0]['G'] == 0.
    applied = transfer(target, target_lab, frozen['candidate_id'])
    assert applied['status'] == 'candidate_registered' and applied['candidate_pool_changed']
    declared = next(item for item in resolved_theories(target) if item['id'] == 'transferred_affine')
    assert Program.from_dict(declared['program']) == affine()
    after = route(target, target_lab, question={'id': 'future_response'})
    assert after['status'] == 'selected' and after['selection']['scores'][0]['G'] == 1.
    assert sorted(group['task_answer'] for group in after['selection']['groups']) == [7., 9.]
    assert after['selection']['support_updated'] is False
    assert candidate(target, frozen['candidate_id'])['source_reconstruction'] == original['source_reconstruction']
    assert_no_new_science(source, source_lab)
    assert_no_new_science(target, target_lab)


def test_completed_consolidation_is_cached_and_library_is_single_source(tmp_path):
    agent, lab = agent_scene(tmp_path)
    first = consolidate(agent, lab)
    before = deepcopy(agent.costs)
    old = library(agent)
    repeated = consolidate(agent, lab)
    assert repeated['cache_hit'] is True and repeated['new_compute_cost'] == 0
    assert agent.costs == before and library(agent)['candidates'] == old['candidates']
    state = agent._program_state()
    encoded = json.dumps(state)
    assert 'theory_consolidation' in state
    assert 'theory_library.json' in encoded
    assert 'source_reconstruction' not in json.dumps(state['theory_consolidation'])
    assert first['frozen_candidates'][0]['candidate_id'] in library(agent)['candidates']
    assert_no_new_science(agent, lab)


def test_PR2_theory_agenda_executes_funded_cross_episode_consolidation(tmp_path):
    agent, lab = agent_scene(tmp_path)
    declarations = sources()
    for index, declaration in enumerate(declarations):
        declaration['support']['status'] = 'supported'
        declaration['support']['evidence'] = [{'file': f'retained_source_{index}.json', 'json_path': ['validation']}]
    result = route(agent, lab, theories=declarations, mission=deepcopy(MISSION))
    assert result['status'] == 'completed' and len(result['frozen_candidates']) == 1
    generated = next(question for question in agent.questions
                     if isinstance(question.get('origin'), dict)
                     and question['origin'].get('kind') == 'multi_episode_commonality')
    assert generated['completed_agenda_basis'] == generated['agenda_basis']
    assert candidate(agent)['theme'] == generated['id']
    assert agent.research_agenda['selected_question_id'] == generated['id']
    assert agent.research_agenda['selected_queue'] == 'theory'
    assert_no_new_science(agent, lab)


def test_repeated_frozen_transfer_reuses_actual_check_and_does_not_add_compute(tmp_path):
    agent, lab = agent_scene(tmp_path)
    compiled = consolidate(agent, lab)
    frozen, = compiled['frozen_candidates']
    check = {'source_ref': {'file': 'retained_target.json', 'json_path': [0]},
             'context_ref': deepcopy(CONTEXT),
             'observation': {'features': {'u': 1.}, 'target': 3., 'history': []}}
    first = transfer(agent, lab, frozen['candidate_id'], evidence_checks=[check])
    assert first['status'] == 'candidate_registered'
    costs, index = deepcopy(agent.costs), deepcopy(agent.theory_index)
    second = transfer(agent, lab, frozen['candidate_id'], evidence_checks=[check])
    assert second['cache_hit'] is True and second['new_compute_cost'] == 0
    assert agent.costs == costs and agent.theory_index == index
    assert_no_new_science(agent, lab)


def test_negative_JSON_encoding_gain_is_kept_as_candidate_and_not_called_compression(tmp_path):
    agent, lab = agent_scene(tmp_path)
    result = consolidate(agent, lab)
    frozen, = result['frozen_candidates']
    assert frozen['encoding_gain_bytes'] < 0 and frozen['active_library'] is False
    assert candidate(agent)['encoding_cost']['formal_mdl'] is False
    applied = transfer(agent, lab, frozen['candidate_id'])
    assert applied['status'] == 'candidate_registered'
    assert applied['support_updated'] is False and applied['independent_confirmation'] is False
    assert_no_new_science(agent, lab)


def test_same_source_has_two_program_versions_but_only_one_independent_origin_T21(tmp_path):
    agent, lab = agent_scene(tmp_path)
    result = consolidate(agent, lab, theories=sources(same_source=True))
    frozen, = result['frozen_candidates']
    compiled = candidate(agent, frozen['candidate_id'])
    assert result['source_count'] == compiled['independent_source_count'] == 1
    assert len(compiled['instances']) == 2
    assert not compiled['active_library'] and not compiled['global_support']
    assert_no_new_science(agent, lab)


def test_retained_new_domain_failure_refutes_only_its_binding_and_cannot_be_omitted_T24(tmp_path):
    agent, lab = agent_scene(tmp_path)
    result = consolidate(agent, lab)
    frozen, = result['frozen_candidates']
    before = deepcopy(candidate(agent, frozen['candidate_id'])['source_reconstruction'])
    check = {'source_ref': {'file': 'retained_target.json', 'json_path': [0]},
             'context_ref': deepcopy(CONTEXT), 'tolerance': 0.,
             'observation': {'features': {'u': 1.}, 'target': 30., 'history': []}}
    failed = transfer(agent, lab, frozen['candidate_id'], evidence_checks=[check])
    assert failed['status'] == 'refuted'
    assert failed['checks'][0]['prediction'] == 3. and failed['checks'][0]['actual'] == 30.
    assert candidate(agent, frozen['candidate_id'])['source_reconstruction'] == before
    refuted = agent.theory_index['transferred_affine']
    assert refuted['compatible'] is False and refuted['support']['status'] == 'refuted'
    repeated = transfer(agent, lab, frozen['candidate_id'], theory_id='renamed_same_failed_binding')
    assert repeated['status'] == 'refuted'
    assert agent.theory_index['renamed_same_failed_binding']['compatible'] is False
    changed = transfer(agent, lab, frozen['candidate_id'], theory_id='changed_parameters', parameters=[29., 1.],
                       evidence_checks=[check])
    assert changed['status'] == 'candidate_registered'
    assert changed['checks'][0]['prediction'] == 30.
    assert changed['support_updated'] is False
    assert_no_new_science(agent, lab)


def test_retained_counterexample_is_rechecked_when_its_actual_value_changes(tmp_path):
    agent, lab = agent_scene(tmp_path)
    example = {'source_ref': {'file': 'retained.json', 'json_path': [0]},
               'observation': {'features': {'u': 1.}, 'target': 3.}}
    first = consolidate(agent, lab, counterexamples={'source_a': [example]})
    assert first['status'] == 'completed'
    example['observation']['target'] = 99.
    changed = consolidate(agent, lab, counterexamples={'source_a': [example]})
    assert changed['cache_hit'] is False
    compiled = candidate(agent, changed['frozen_candidates'][0]['candidate_id'])
    assert compiled['source_counterexample_status'] == 'failed'
    assert compiled['counterexample_checks'][0]['actual'] == 99.
    assert_no_new_science(agent, lab)


def test_shared_fragment_requires_explicit_composition_before_whole_task_transfer(tmp_path):
    agent, lab = agent_scene(tmp_path)
    declarations = sources()
    fragment = Program('abs', children=(Program('feature', name='u'),))
    declarations[0]['program'] = Program('add', children=(fragment, Program('const', value=1.))).to_dict()
    declarations[1]['program'] = Program('mul', children=(fragment, Program('const', value=3.))).to_dict()
    for declaration in declarations:
        declaration['fragment_contracts'] = {'0': {'output_unit': 'V'}}
    result = consolidate(agent, lab, theories=declarations)
    frozen, = result['frozen_candidates']
    assert frozen['kind'] == 'shared_subexpression'
    applied = transfer(agent, lab, frozen['candidate_id'])
    assert applied['status'] == 'author_work_request'
    assert 'transferred_affine' not in agent.theory_index
    assert_no_new_science(agent, lab)


@pytest.mark.parametrize('allocation', [{'compute': 0}, {'compute': 8}, {'compute': 20000, 'max_groups': 0}])
def test_real_budget_cutoff_never_becomes_scientific_failure(tmp_path, allocation):
    agent, lab = agent_scene(tmp_path)
    result = consolidate(agent, lab, allocation=allocation)
    assert result['status'] == 'UNKNOWN' and result['budget_exhausted']
    assert result['frozen_candidates'] == []
    assert result['new_compute_cost'] <= allocation['compute']
    assert all(item['support']['status'] == 'candidate' for item in agent.theory_index.values())
    assert_no_new_science(agent, lab)


def test_missing_theory_mission_retains_deferred_work_without_computation(tmp_path):
    agent, lab = agent_scene(tmp_path)
    result = consolidate(agent, lab, mission={'purpose': 'task'})
    assert result['status'] == 'research_deferred' and result['new_compute_cost'] == 0
    assert agent.costs['compute_cost'] == 0
    assert_no_new_science(agent, lab)


def test_new_duplicate_note_is_classified_once_without_duplicating_frozen_program(tmp_path):
    agent, lab = agent_scene(tmp_path)
    reference = {'file': 'retained_source_notes.json', 'json_path': ['same_record']}
    first = consolidate(agent, lab, materials=[{'id': 'note_a', 'kind': 'external_claim',
        'source_ref': reference, 'claim': 'source statement'}])
    second = consolidate(agent, lab, materials=[{'id': 'note_b', 'kind': 'external_claim',
        'source_ref': reference, 'claim': 'another description of the same record'}])
    assert first['frozen_candidates'][0]['candidate_id'] == second['frozen_candidates'][0]['candidate_id']
    assert len(library(agent)['candidates']) == 1
    materials = library(agent)['materials']
    assert [item['kind'] for item in materials] == ['external_claim', 'duplicate']
    assert materials[1]['duplicate_of'] == 'note_a'
    assert second['support_source_count'] == first['support_source_count']
    consolidate(agent, lab, materials=[{'id': 'note_b', 'kind': 'external_claim',
        'source_ref': reference, 'claim': 'another description of the same record'}])
    assert len(library(agent)['materials']) == 2
    assert_no_new_science(agent, lab)


def test_copying_shared_library_keeps_different_common_sources_with_the_same_local_record_and_question(tmp_path):
    first, first_lab = measured_scene(tmp_path / 'first')
    first._open_question({'id': 'local_q', 'question': 'Read this source response'})
    local_theory = theory('local_theory', episode_ref={'question_id': 'local_q', 'meaning_version': 1})
    material = {'id': 'same_local_note', 'kind': 'measured',
                'source_ref': {'kind': 'observation', 'record_index': 0}, 'claim': 'Actual local source record'}
    arguments = {'theories': [local_theory], 'source_theory_ids': ['local_theory'], 'materials': [material]}
    paid_first = first_lab.paid
    consolidate(first, first_lab, **deepcopy(arguments))
    target = tmp_path / 'second'
    (target / 'shared').mkdir(parents=True)
    (target / 'shared/theory_library.json').write_text((first.workspace / 'shared/theory_library.json').read_text())
    second, second_lab = measured_scene(target)
    second._open_question({'id': 'local_q', 'question': 'Read this source response'})
    paid_second = second_lab.paid
    consolidate(second, second_lab, **deepcopy(arguments))
    stored = library(second)
    notes = [item for item in stored['materials'] if item['id'] == 'same_local_note']
    expected_sources = {str(first.workspace.resolve()), str(second.workspace.resolve())}
    assert len(notes) == 2 and {note['source_id'] for note in notes} == expected_sources
    assert notes[0]['source_ref'] == notes[1]['source_ref']
    assert all(note.get('duplicate_source_id', note['source_id']) == note['source_id'] for note in notes)
    episodes = [item for item in stored['episodes'] if item['episode_id'] == 'local_q@meaning1']
    assert len(episodes) == 2 and {item['source_id'] for item in episodes} == expected_sources
    origins = [item for item in stored['unique_support_sources']
               if item['source_ref'] == {'file': 'laboratory_observations.json', 'json_path': [0]}]
    assert len(origins) == 2 and {item['source_id'] for item in origins} == expected_sources
    consolidate(second, second_lab, **deepcopy(arguments))
    assert len([item for item in library(second)['materials'] if item['id'] == 'same_local_note']) == 2
    assert first_lab.paid == paid_first and second_lab.paid == paid_second
    assert first.costs['model_calls'] == second.costs['model_calls'] == 0


def test_same_global_archive_reference_is_not_an_extra_source_when_the_library_changes_workspace(tmp_path):
    first, first_lab = agent_scene(tmp_path / 'first')
    note = {'id': 'same_global_note', 'kind': 'external_claim', 'claim': 'One cited archived result',
        'source_ref': {'archive': '/retained/history.sqlite3', 'unit': 'source', 'member': 'result.json',
                       'json_path': ['reported_result']}}
    consolidate(first, first_lab, materials=[note])
    target = tmp_path / 'second'
    (target / 'shared').mkdir(parents=True)
    (target / 'shared/theory_library.json').write_text((first.workspace / 'shared/theory_library.json').read_text())
    second, second_lab = agent_scene(target)
    consolidate(second, second_lab, materials=[deepcopy(note)])
    notes = [item for item in library(second)['materials'] if item['id'] == 'same_global_note']
    assert len(notes) == 1 and notes[0]['source_id'] == str(first.workspace.resolve())
    assert_no_new_science(first, first_lab)
    assert_no_new_science(second, second_lab)


def context_relation_scene(tmp_path):
    agent, lab = agent_scene(tmp_path)
    lag = Program('lag', name='u', lag=1)
    gate = Program('available', lag=1)
    zero = Program('const', value=0.)
    variants = [lag, Program('mul', children=(Program('const', value=2.), lag))]
    episodes, declarations, references = [], [], []
    for index, body in enumerate(variants):
        identifier = f'context_source_{index}'
        revised = Program('if', children=(gate, body, zero))
        episodes.append({'task_id': identifier, 'questions': [{'question': 'Repair a measured history failure'}],
            'program': revised.to_dict(), 'actions': [
                {'action': 'observe', 'observation_cost': 3, 'compute_cost': 0, 'details': {'rows': 3}},
                {'action': 'revise', 'observation_cost': 0, 'compute_cost': 7,
                 'details': {'before_program': Program('feature', name='u').to_dict(),
                             'after_program': revised.to_dict(), 'operation': 'representation'}},
                {'action': 'reuse_check', 'details': {'accepted': False, 'loss': 9.}}],
            'observation_cost': 3, 'compute_cost': 7, 'status': 'completed',
            'evaluation': {'loss': 9., 'rows': 4}})
        reference = {'file': 'retained_contexts.json', 'json_path': ['episodes', index], 'action_cutoff': 1}
        references.append(reference)
        declarations.append(theory(identifier, revised,
            context_ref={**deepcopy(CONTEXT), 'source_id': identifier},
            context_family_ref='declared_history_context',
            episode_ref={'question_id': identifier, 'meaning_version': 1},
            program_source_ref={**reference, 'json_path': [*reference['json_path'],
                'actions', 1, 'details', 'after_program']},
            support={'status': 'candidate', 'evidence': []}))
    agent._save('retained_contexts.json', json.dumps({'episodes': episodes}))
    arguments = {'operator': 'theory_consolidation', 'theme': 'actual_history_changes',
        'theories': declarations, 'source_theory_ids': [item['id'] for item in declarations],
        'episode_sources': references, 'mission': deepcopy(MISSION),
        'allocation': {'compute': 20000, 'max_groups': 8}}
    result = agent._execute('run', arguments, lab)
    relation = next(item for item in agent.theory_consolidation['candidates'].values()
                    if item['kind'] == 'program_context_relation')
    return agent, lab, arguments, result, relation


def test_actual_cross_skeleton_changes_compile_to_a_frozen_context_without_future_score(tmp_path):
    agent, lab, arguments, result, relation = context_relation_scene(tmp_path)
    assert result['status'] == 'completed'
    assert all(item['exact'] for item in relation['source_reconstruction'])
    assert all(item['source_cutoff'] == {'action_index_inclusive': 1}
               for item in relation['source_outcomes'])
    assert all('before_program' not in item and 'after_program' not in item
               for item in relation['source_bindings'])
    assert all('loss' not in item['source_outcomes']['independent_results'][0]
               for item in relation['source_outcomes'])
    before = deepcopy(agent.costs)
    repeated = agent._execute('run', deepcopy(arguments), lab)
    assert repeated['cache_hit'] is True and repeated['new_compute_cost'] == 0
    assert agent.costs == before
    assert_no_new_science(agent, lab)


def test_context_relation_transfer_adds_multiple_concrete_programs_to_the_existing_pool(tmp_path):
    from hegel_machine.cognition.question_routing import register_theories
    agent, lab, _, _, relation = context_relation_scene(tmp_path)
    lag = Program('lag', name='u', lag=1)
    register_theories(agent, [theory('lag_component', lag),
        theory('scaled_component', Program('mul', children=(Program('const', value=3.), lag))),
        theory('current_component', Program('feature', name='u'))])
    old_index = deepcopy(agent.theory_index)
    arguments = {'operator': 'theory_transfer', 'candidate_id': relation['id'],
        'theory_id': 'guarded', 'component_theory_ids': [
            'lag_component', 'scaled_component', 'current_component'],
        'context_ref': deepcopy(CONTEXT), 'output_unit': 'V', 'allocation': {'compute': 5000}}
    result = agent._execute('run', deepcopy(arguments), lab)
    assert result['status'] == 'candidates_registered'
    assert len(result['generated_theories']) == 2
    assert all(item['constraint_checks'] for item in result['generated_theories'])
    programs = [Program.from_dict(agent.theory_index[item['theory_id']]['program'])
                for item in result['generated_theories']]
    row = Observation({'u': 7.}, 0., ({'u': 2.},))
    assert sorted(program.predict(row) for program in programs) == [2., 6.]
    assert all(agent.theory_index[identifier] == old for identifier, old in old_index.items())
    routed = route(agent, lab, question=future_question(agent), experiments=[probe(agent)])
    assert routed['status'] == 'scoped_unknown'
    assert routed['selection']['unknown_reason'] == 'incomplete_candidate_predictions'
    assert routed['selection']['scores'][0]['G'] == 1.
    grouped = {identifier for group in routed['selection']['groups'] for identifier in group['theory_ids']}
    assert {item['theory_id'] for item in result['generated_theories']} <= grouped
    before = deepcopy(agent.costs)
    repeated = agent._execute('run', deepcopy(arguments), lab)
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0 and agent.costs == before
    assert_no_new_science(agent, lab)


def test_changed_local_revision_source_and_requested_budget_invalidate_old_consolidation(tmp_path):
    agent, lab, arguments, _, relation = context_relation_scene(tmp_path)
    original = deepcopy(relation['definition'])
    increased = deepcopy(arguments)
    increased['allocation']['compute'] += 1
    first = agent._execute('run', increased, lab)
    assert first['cache_hit'] is False
    assert len([item for item in agent.theory_consolidation['candidates'].values()
                if item['kind'] == 'program_context_relation']) == 1
    assert next(item for item in agent.theory_consolidation['candidates'].values()
                if item['kind'] == 'program_context_relation')['definition'] == original
    source = json.loads((agent.workspace / 'retained_contexts.json').read_text())
    source['episodes'][0]['actions'][0]['compute_cost'] = 1
    agent._save('retained_contexts.json', json.dumps(source))
    changed = agent._execute('run', increased, lab)
    assert changed['cache_hit'] is False
    assert_no_new_science(agent, lab)


def test_exact_context_binding_uses_original_checked_registration_and_keeps_failure_for_alias(tmp_path):
    from hegel_machine.cognition.question_routing import register_theories
    agent, lab, _, _, relation = context_relation_scene(tmp_path)
    register_theories(agent, [theory('lag_component', Program('lag', name='u', lag=1))])
    hole = relation['holes'][0]['name']
    arguments = {'operator': 'theory_transfer', 'candidate_id': relation['id'],
        'theory_id': 'known_bad_guarded', 'component_bindings': {hole: 'lag_component'},
        'context_ref': deepcopy(CONTEXT), 'output_unit': 'V',
        'evidence_checks': [{'source_ref': {'file': 'retained_response.json', 'json_path': [0]},
            'context_ref': deepcopy(CONTEXT), 'tolerance': 0.,
            'observation': {'features': {'u': 7.}, 'history': [{'u': 2.}], 'target': 99.}}]}
    result = agent._execute('run', arguments, lab)
    assert result['status'] == 'refuted'
    generated = result['generated_theories'][0]
    assert generated['theory_id'] == 'known_bad_guarded'
    assert generated['checks'][0]['prediction'] == 2.
    assert generated['checks'][0]['actual'] == 99.
    alias = deepcopy(agent.theory_index['known_bad_guarded'])
    alias.update(id='ordinary_guarded_alias', compatible=True, support={'status': 'candidate', 'evidence': []})
    register_theories(agent, [alias])
    resolved = next(item for item in resolved_theories(agent) if item['id'] == 'ordinary_guarded_alias')
    assert resolved['compatible'] is False and resolved['scope_refutation']['counterexample_refs']
    assert_no_new_science(agent, lab)


def test_refuted_numeric_program_can_be_revised_as_a_component_without_inheriting_support(tmp_path):
    from hegel_machine.cognition.question_routing import register_theories
    agent, lab, _, _, relation = context_relation_scene(tmp_path)
    lag = Program('lag', name='u', lag=1)
    agent._open_question({'id': 'failed_source', 'question': 'Retained failed numeric source'})
    agent.relation_research['failed_source'] = {'round_index': 0, 'candidates': [
        {'program': lag.to_dict(), 'status': 'refuted'}]}
    failed = theory('failed_component', lag)
    failed.pop('program')
    failed['source_ref'] = {'kind': 'relation_candidate', 'question_id': 'failed_source',
        'round_index': 0, 'candidate_index': 0}
    register_theories(agent, [failed])
    assert next(item for item in resolved_theories(agent) if item['id'] == 'failed_component')['compatible'] is False
    transferred = agent._execute('run', {'operator': 'theory_transfer', 'candidate_id': relation['id'],
        'theory_id': 'revised_failed_component', 'component_bindings': {
            relation['holes'][0]['name']: 'failed_component'},
        'context_ref': deepcopy(CONTEXT), 'output_unit': 'V'}, lab)
    assert transferred['status'] == 'candidates_registered'
    generated = agent.theory_index['revised_failed_component']
    assert Program.from_dict(generated['program']) != lag
    assert generated['compatible'] is True and generated['support']['evidence'] == []
    assert generated['support']['status'] == 'candidate'
    assert transferred['support_updated'] is False
    register_theories(agent, [theory('same_failed_numeric_alias', lag)])
    resolved = {item['id']: item for item in resolved_theories(agent)}
    assert resolved['same_failed_numeric_alias']['compatible'] is False
    assert resolved['failed_component']['compatible'] is False
    assert resolved['revised_failed_component']['compatible'] is True
    assert_no_new_science(agent, lab)
