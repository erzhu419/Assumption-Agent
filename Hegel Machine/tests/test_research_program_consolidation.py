"""Saved producer DAGs become optional refitting methods, not world support."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.research_programs import _plan
from hegel_machine.cognition.theory_consolidation import (
    consolidate_research, install_theory_library, recall_frozen_theories,
)
from test_common_source_episodes import source
from test_theory_consolidation import agent_scene


def saved_source(path, source_id, offset, threshold=None):
    agent, identifier, reference = source(path, physical_id=source_id, offset=offset)
    candidate = agent.questions[1]['native_condition_research'][0]['candidates'][0]
    program = Program.from_dict(candidate['program'])
    term = Program('feature', name='drive')
    steps = [{'operation': 'fit_component', 'question': 'Fit the paid measured coordinate',
              'program': program, 'loss': candidate['fit_mse']}]
    if threshold is not None:
        from hegel_machine.cognition.continuous_state import _sum_terms
        term = Program('if', children=(Program('ge', children=(term, Program('const', value=threshold))),
                                       term, Program('const', value=0.)))
        program = _sum_terms([term], [1.], offset)
        candidate['program'] = program.to_dict()
        steps = [{'operation': 'discover_conditional_coordinates',
            'question': 'Retain the condition actually fitted in this source',
            'program': program, 'loss': candidate['fit_mse'],
            'gate_coordinate': Program('feature', name='drive').to_dict(),
            'branch_basis': [Program('feature', name='drive').to_dict()],
            'threshold': threshold, 'width': 1, 'placement': 'inside_window'}]
    candidate['research_program'] = _plan({'fit_terms': (term,),
        'coefficients': [offset, 1.], 'program': program, 'loss': candidate['fit_mse']},
        steps, 16, 32)
    agent._prepare_program_state()
    reference['source_theory_ids'] = [identifier]
    return agent, reference


def compile_sources(tmp_path, *, same_source=False, conditional=False):
    first, a = saved_source(tmp_path / 'source-a', 'source-a', 1., 1.5 if conditional else None)
    second, b = saved_source(tmp_path / 'source-b', 'source-a' if same_source else 'source-b', -2.,
                             4.5 if conditional else None)
    agent, _ = agent_scene(tmp_path / 'learner')
    costs = [deepcopy(first.costs), deepcopy(second.costs)]
    result = consolidate_research(agent, {'object_type': 'research_program',
        'theme': 'measured-coordinate-refit', 'common_source_refs': [a, b],
        'mission': {'purpose': 'theory_learning', 'theory_learning_allocation': {'compute': 2000}},
        'allocation': {'compute': 2000, 'max_groups': 4}})
    assert [first.costs, second.costs] == costs
    return agent, result


def test_two_failed_sources_produce_one_method_without_copying_DAG_or_transferring_support(tmp_path):
    agent, result = compile_sources(tmp_path)
    frozen, = result['frozen_candidates']
    method = agent.theory_consolidation['candidates'][frozen['candidate_id']]
    assert method['kind'] == 'research_program'
    assert method['independent_source_count'] == 2
    assert method['free_parameters'] == ['response_coefficients']
    assert all(item.get('failure_refs') and item.get('cost_refs') for item in method['source_outcomes'])
    assert method['counterexample_refs'] and not method['global_support']
    assert method['trigger']['learned_utility_trigger'] is False
    assert '"graph":' not in json.dumps(method)
    assert not method['independent_confirmation']
    from hegel_machine.cognition.theory_consolidation import _same_frozen_candidate
    first_seed_only = deepcopy(method)
    first_seed_only['research_program_source_refs'] = first_seed_only['research_program_source_refs'][:1]
    assert not _same_frozen_candidate(first_seed_only, method)
    changed_input = deepcopy(method)
    changed_input['source_seed_inputs'] = changed_input['source_seed_inputs'][:1]
    assert not _same_frozen_candidate(changed_input, method)


def test_two_versions_of_one_physical_source_do_not_create_cross_source_method(tmp_path):
    _, result = compile_sources(tmp_path, same_source=True)
    assert not result['frozen_candidates']
    assert any(request['reason'] == 'distinct_physical_research_sources_required'
               for request in result['author_work'])


def test_ordinary_recall_refits_on_paid_training_changes_prediction_and_reuses_completed_fit(tmp_path, monkeypatch):
    from hegel_machine.cognition.finite_native_author import export_revision_candidates, select_actions
    from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
    learner, result = compile_sources(tmp_path)
    target, _ = agent_scene(tmp_path / 'target')
    method = learner.theory_consolidation['candidates'][result['frozen_candidates'][0]['candidate_id']]
    target._save('shared/theory_library.json', (learner.workspace / result['library_path']).read_text())
    target.theory_consolidation = install_theory_library(target)
    target.observations, target.rows = [], []
    for record in range(2):
        rows = [{'features': {'drive': i + record * .1}, 'history': [],
                 'target': (5. * (i + record * .1) - 2.) if record == 0 else 1e6}
                for i in range(8)]
        target.observations.append({'action': {'kind': 'trajectory',
            'sequence': [row['features'] for row in rows]}, 'rows': rows,
            'row_indices': list(range(len(target.rows), len(target.rows) + len(rows)))})
        target.rows.extend(rows)
    roles = {'drive': deepcopy(method['role_signature']['role_0'])}
    question = {'id': 'new-measured-response', 'question': 'Predict the measured response',
        'context_ref': {'source_id': 'target'}, 'parent_id': 'task', 'theory_target': {
        'role_signature': roles, 'output_unit': method['output_unit'],
        'context_ref': {'source_id': 'target'}, 'context_family_ref': 'preregistered-fixture-theme',
        'research_program_fitting': {'validation_records': [1]}}}
    target.questions.append(question)
    applied = recall_frozen_theories(target, question, {'allocation': {'compute': 1000}})
    assert applied['generated_theory_ids']
    theory = target.theory_index[applied['generated_theory_ids'][0]]
    predicted = Program.from_dict(theory['program']).predict(Observation({'drive': 3.}, 0.))
    assert abs(predicted - 13.) < 1e-8  # The held-out million-valued labels were not fitted.
    assert theory['support']['status'] == 'candidate'
    work = next(value['result'] for key, value in target.theory_consolidation['executions'].items()
                if key.startswith('research_program_transfer:'))
    basis = next(value['basis'] for key, value in target.theory_consolidation['executions'].items()
                 if key.startswith('research_program_transfer:'))
    assert basis['binding']['source_seed_inputs'] == method['source_seed_inputs']
    assert basis['binding']['seed_policy'] == method['seed_policy']
    assert basis['binding']['role_signature'] == roles
    fitted = work['generated_theories'][0]
    assert fitted['validation_status'] == 'not_scored_by_method_replay' and fitted['actual_check_refs'] == []
    assert [item['status'] for item in work['seed_replay_work']].count('same_replay_inputs') == 1
    assert not applied['support_updated'] and not target.confirmations
    costs = deepcopy(target.costs)
    replayed = recall_frozen_theories(target, question, {'allocation': {'compute': 1000}})
    assert replayed['cache_hit'] and replayed['new_compute_cost'] == 0
    assert target.costs == costs
    exported = export_revision_candidates(target, question)
    assert exported['new_exports'] == 1
    model = exported['exports'][0]['model']
    assert model in target.candidate_paths and not theory['consolidation_ref'].get('parent')
    target._save('old.py', export_temporal_program(Program('const', value=0.), feature_names=('drive',)))
    target._prepare_program_state()
    monkeypatch.chdir(target.workspace)
    action = select_actions({'controls': ['drive']}, target.observations, {'finite_native_author': {
        'question': question, 'predictor_paths': ['old.py'], 'predictor_catalog_ref': {'file': 'catalog.json'},
        'max_compute': 1000, 'max_candidates': 2, 'boundary_resolution': .1}})
    assert action['arguments']['operator'] == 'research_choice'
    assert action['arguments']['models'] == ['old.py', model]


def test_shared_structure_query_returns_real_method_contract_not_a_role_carrier_program(tmp_path):
    from hegel_machine.cognition.question_object_queries import evaluate_shared_structure
    from test_shared_structure_query import dereference

    agent, _ = compile_sources(tmp_path)
    query = {'object_type': 'research_program', 'typed_program_refs': [
        {'kind': 'theory', 'id': key, 'version': value['version']}
        for key, value in agent.theory_index.items()],
        'common_source_refs': agent.theory_consolidation['executions'][
            'measured-coordinate-refit']['request_basis']['common_source_refs'],
        'mapping_budget': {'compute': 2000, 'max_groups': 4}}
    answer = evaluate_shared_structure(agent, {'id': 'shared-method'}, query, None,
        {'purpose': 'method_learning', 'allocation': {'compute': 2000}})
    assert answer['status'] == 'answered'
    method, = answer['value']['abstractions']
    assert 'abstract_program_ref' not in method and 'inverse_bindings_ref' not in method
    assert dereference(agent, method['research_program_contract_ref'])['operation'] == 'refit_selected_terms'
    assert len(dereference(agent, method['research_program_sources_ref'])) == 2
    assert method['source_reconstruction_scope'] == 'individual_saved_graph_identity_only'
    assert not answer['scientific_support'] and not answer['independent_confirmation']


def test_distinct_retained_conditional_seeds_both_refit_and_export_with_equal_slices(tmp_path):
    from hegel_machine.cognition.finite_native_author import export_revision_candidates

    learner, result = compile_sources(tmp_path, conditional=True)
    method = learner.theory_consolidation['candidates'][result['frozen_candidates'][0]['candidate_id']]
    target, _ = agent_scene(tmp_path / 'target')
    target._save('shared/theory_library.json', (learner.workspace / result['library_path']).read_text())
    target.theory_consolidation = install_theory_library(target)
    target.rows, target.observations = [], []
    for record in range(2):
        rows = [{'features': {'drive': float(i) + record * .1}, 'history': [],
                 'target': float(i ** 2) if record == 0 else 1e6} for i in range(8)]
        target.observations.append({'rows': rows,
            'row_indices': list(range(len(target.rows), len(target.rows) + len(rows)))})
        target.rows.extend(rows)
    question = {'id': 'conditional-successor', 'context_ref': {'source_id': 'target'},
        'parent_id': 'task', 'theory_target': {'role_signature': {
        'drive': deepcopy(method['role_signature']['role_0'])}, 'output_unit': method['output_unit'],
        'context_ref': {'source_id': 'target'}, 'context_family_ref': 'preregistered-fixture-theme',
        'research_program_fitting': {'validation_records': [1]}}}
    target.questions.append(question)
    applied = recall_frozen_theories(target, question, {'allocation': {'compute': 80}})
    work = next(value['result'] for key, value in target.theory_consolidation['executions'].items()
                if key.startswith('research_program_transfer:'))
    assert len(applied['generated_theory_ids']) == 2
    assert len(work['generated_theories']) == len(work['seed_replay_work']) == 2
    assert abs(work['seed_replay_work'][0]['allocated_compute'] -
               work['seed_replay_work'][1]['allocated_compute']) <= 1
    assert all(item['status'] == 'fitted_candidate' for item in work['seed_replay_work'])
    assert not work['search_completed']  # Both bounded fits survive incomplete expanded search.
    exported = export_revision_candidates(target, question)
    assert exported['new_exports'] == 2 and not exported['errors']
    assert target.costs['compute_cost'] <= 80 and not target.confirmations


def method_transfer_scene(tmp_path):
    learner, result = compile_sources(tmp_path, conditional=True)
    method = learner.theory_consolidation['candidates'][result['frozen_candidates'][0]['candidate_id']]
    target, laboratory = agent_scene(tmp_path / 'target')
    target._save('shared/theory_library.json', (learner.workspace / result['library_path']).read_text())
    target.theory_consolidation = install_theory_library(target)
    target.rows, target.observations = [], []
    for record in range(2):
        rows = [{'features': {'drive': i + record * .1, 'force': (i + record * .1) ** 2},
                 'history': [], 'target': float(i ** 2 + record)} for i in range(8)]
        target.observations.append({'rows': rows,
            'row_indices': list(range(len(target.rows), len(target.rows) + len(rows)))})
        target.rows.extend(rows)
    arguments = {'candidate_id': method['id'], 'theory_id': 'method-successor',
        'mapping': {'role_0': 'drive'},
        'role_signature': {'drive': deepcopy(method['role_signature']['role_0'])},
        'output_unit': method['output_unit'], 'context_ref': {'source_id': 'target'},
        'context_family_ref': 'preregistered-fixture-theme',
        'research_program_fitting': {'validation_records': [1]},
        'allocation': {'compute': 10000}}
    return target, laboratory, arguments


def test_new_checks_restore_completed_seed_fits_and_only_pay_for_unfinished_checks(tmp_path, monkeypatch):
    from hegel_machine.cognition import question_queries, research_programs
    from hegel_machine.cognition.theory_consolidation import transfer_theory

    target, laboratory, arguments = method_transfer_scene(tmp_path)
    replay, evaluate = research_programs.replay_research_program, question_queries.evaluate_on_model
    replay_calls, check_calls = [], []

    def counted_replay(*args, **kwargs):
        replay_calls.append(len(args[1]))
        return replay(*args, **kwargs)

    def counted_check(*args, **kwargs):
        def predict(value):
            check_calls.append(args[0])
            return args[1](value)
        return evaluate(args[0], predict, *args[2:], **kwargs)

    monkeypatch.setattr(research_programs, 'replay_research_program', counted_replay)
    monkeypatch.setattr(question_queries, 'evaluate_on_model', counted_check)
    first = transfer_theory(target, arguments)
    assert first['search_completed'] and replay_calls == [8, 8]
    programs = {item['theory_id']: deepcopy(target.theory_index[item['theory_id']]['program'])
                for item in first['generated_theories']}
    assert programs
    checks = [{'source_ref': {'file': str(target.workspace / 'laboratory_observations.json'),
                'json_path': [0, 'rows', index]}, 'context_ref': deepcopy(arguments['context_ref']),
               'observation': deepcopy(target.observations[0]['rows'][index]), 'tolerance': 1e6}
              for index in range(2)]
    check_arguments = {**arguments, 'evidence_checks': checks[:1], 'allocation': {'compute': 0}}
    costs = deepcopy(target.costs)
    unfinished = transfer_theory(target, check_arguments)
    assert unfinished['fit_cache_hit'] and unfinished['new_compute_cost'] == 0
    assert target.costs == costs and replay_calls == [8, 8] and not check_calls
    assert all(item['checks'][0]['status'] == 'UNKNOWN' for item in unfinished['generated_theories'])
    target._prepare_program_state()
    previous = target.result('researching')
    restored, _ = agent_scene(tmp_path / 'restored')
    laboratory.costs = {'observation_cost': 0}
    restored.restore_practice_research(previous, target.workspace, laboratory, confirmation_seed=17)
    assert restored.costs == costs
    check_arguments['allocation'] = arguments['allocation']
    completed = transfer_theory(restored, check_arguments)
    assert completed['fit_cache_hit'] and replay_calls == [8, 8]
    assert len(check_calls) == completed['new_compute_cost'] == len(programs)
    assert restored.costs['compute_cost'] - costs['compute_cost'] == len(programs)
    assert completed['seed_replay_work'] == first['seed_replay_work']
    before = restored.costs['compute_cost']
    extended = transfer_theory(restored, {**check_arguments, 'evidence_checks': checks})
    assert extended['fit_cache_hit'] and replay_calls == [8, 8]
    assert len(check_calls) == 2 * len(programs)
    assert extended['new_compute_cost'] == restored.costs['compute_cost'] - before == len(programs)
    assert all(len(item['checks']) == 2 for item in extended['generated_theories'])
    assert {identifier: restored.theory_index[identifier]['program'] for identifier in programs} == programs
    repeated = transfer_theory(restored, {**check_arguments, 'evidence_checks': checks})
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0
    assert len(check_calls) == 2 * len(programs) and replay_calls == [8, 8]
    assert restored.costs['compute_cost'] == before + len(programs)
    assert first['new_compute_cost'] > extended['new_compute_cost']
    assert extended['seed_replay_work'] == first['seed_replay_work']
    checks[1]['observation']['target'], checks[1]['tolerance'] = 1e6, 0.
    changed = transfer_theory(restored, {**check_arguments, 'evidence_checks': checks})
    assert changed['new_compute_cost'] == len(programs) and replay_calls == [8, 8]
    assert all(item['status'] == 'refuted' for item in changed['generated_theories'])
    omitted = transfer_theory(restored, check_arguments)
    assert omitted['new_compute_cost'] == 0
    assert all(item['status'] == 'refuted' for item in omitted['generated_theories'])
    more_checks = [*checks, {**deepcopy(checks[0]), 'tolerance': .25}]
    stopped = transfer_theory(restored, {**check_arguments, 'evidence_checks': more_checks,
        'allocation': {'compute': 0}})
    assert stopped['new_compute_cost'] == 0
    assert all(item['status'] == 'refuted' and item['checks'][-1]['status'] == 'UNKNOWN'
               for item in stopped['generated_theories'])
    finished = transfer_theory(restored, {**check_arguments, 'evidence_checks': more_checks})
    assert finished['new_compute_cost'] == len(programs) and replay_calls == [8, 8]
    assert all(item['status'] == 'refuted' and item['checks'][-1]['status'] != 'UNKNOWN'
               for item in finished['generated_theories'])


@pytest.mark.parametrize('changed', ['paid_data', 'split', 'binding', 'method_contract'])
def test_real_method_fit_inputs_still_invalidate_seed_replay(tmp_path, monkeypatch, changed):
    from hegel_machine.cognition import research_programs
    from hegel_machine.cognition.theory_consolidation import transfer_theory

    target, _, arguments = method_transfer_scene(tmp_path)
    replay = research_programs.replay_research_program
    calls = []

    def counted(*args, **kwargs):
        calls.append(len(args[1]))
        return replay(*args, **kwargs)

    monkeypatch.setattr(research_programs, 'replay_research_program', counted)
    first = transfer_theory(target, arguments)
    assert first['search_completed'] and calls == [8, 8]
    if changed == 'paid_data':
        rows = deepcopy(target.observations[0]['rows'])
        for row in rows:
            row['features']['drive'] += .2
        target.observations.append({'rows': rows,
            'row_indices': list(range(len(target.rows), len(target.rows) + len(rows)))})
        target.rows.extend(rows)
    elif changed == 'split':
        arguments['research_program_fitting']['validation_records'] = [0]
    elif changed == 'binding':
        arguments['mapping'] = {'role_0': 'force'}
        arguments['role_signature'] = {'force': arguments['role_signature']['drive']}
    else:
        # A new method/parameter-identification contract must not inherit the
        # previous version's fitted output even when this seed is unchanged.
        candidate = target.theory_consolidation['candidates'][arguments['candidate_id']]
        candidate['version'] += 1
        target._save('shared/theory_library.json', json.dumps(target.theory_consolidation))
    before = target.costs['compute_cost']
    second = transfer_theory(target, arguments)
    assert len(calls) == 4 and not second.get('fit_cache_hit')
    assert second['new_compute_cost'] == target.costs['compute_cost'] - before > 0
