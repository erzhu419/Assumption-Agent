"""Deterministic proposals preserve mission, actual sources and unfunded work."""

from copy import deepcopy

from hegel_machine.cognition.question_agenda import build_agenda, generate_questions


def question(identifier='current'):
    return {'id': identifier, 'question': 'Read the paid response in its declared scope',
            'kind': 'value', 'context_ref': {'scene': 'C5'}, 'answer_type': 'number',
            'answer_resolution_ref': {'atol': .01},
            'query_semantics': {'mode': 'model_query', 'query_expression': {
                'op': 'Readout', 'record_index': 0, 'step': 0,
                'binding': {'action': {'reset': True}, 'features': {'u': 1.}, 'history': []}}},
            'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'task', 'path': 'plan.py'}]}


def theory(identifier, context, value=1.):
    return {'id': identifier, 'version': 1, 'program': {'op': 'const', 'value': value},
            'role_signature': {}, 'output_unit': 'V', 'context_ref': context,
            'declared_scope_ref': {'range': [-1., 1.]}, 'measured_coverage_ref': [0, 1],
            'premise_refs': [], 'parameter_paths': [],
            'support': {'status': 'supported', 'evidence': [0, 1], 'context_ref': context}}


def test_failed_prediction_and_stale_binding_reuse_a_real_premise_once():
    premise = question('calibration')
    failed = {'id': 'failure', 'failed_confirmation': {'passed': False, 'returncode': 0,
              'mse': 1., 'observation_indices': [20, 21]},
              'premise_refs': ['calibration']}
    candidates = {'parent.py': {'status': 'needs_rebuild', 'paid_rows': 22,
                               'inputs': {'gain': {'kind': 'parameter_region', 'question_id': 'calibration', 'version': 1}}}}
    original = deepcopy((premise, failed, candidates))
    generated = generate_questions([failed, premise], candidates, [])
    assert len(generated) == 1 and generated[0]['id'] == 'calibration'
    assert generated[0]['query_semantics'] == premise['query_semantics']
    assert {ref['id'] for ref in generated[0]['origin']['source_refs']} == {'failure', 'calibration', 'parent.py'}
    assert {'kind': 'bind_parameter_region', 'model': 'parent.py', 'slot': 'gain'} in generated[0]['use_sites']
    assert (premise, failed, candidates) == original


def test_representation_witness_generates_construction_not_a_hidden_correct_window():
    diagnostic = {'conflicting_groups': 1, 'empirical_mse_floor': .2,
                  'witnesses': [{'rows': [{'row_index': 7, 'history': [{'u': 1.}]}, {'row_index': 9}]}]}
    candidates = {'model.py': {'status': 'representation_conflict', 'paid_rows': 10,
                             'current_evaluation': {'representation_diagnostic': diagnostic}}}
    generated = generate_questions([], candidates, [])
    proposal, = generated
    assert proposal['origin']['kind'] == 'representation_witness'
    assert proposal['agenda_basis']['witness_rows'] == [[7, 9]]
    assert proposal['query_semantics'] == {'mode': 'solution_relation'}
    assert 'author_work_request' in proposal and 'history' not in str(proposal['agenda_basis'])
    agenda = build_agenda([], candidates, [], budget={'compute': 20, 'author_calls': 0})
    assert agenda['selected_question_id'] is None
    assert agenda['deferred'][0]['reason'] == 'author_work_unfunded'


def test_parameter_region_keeps_interval_and_actual_parent_use_without_excluding_family():
    region = question('threshold')
    region['executable_answer'] = {'version': 3, 'status': 'supported', 'evidence_status': 'supported',
                                   'parameter_region': {'lower': .21, 'upper': .29}}
    generated = generate_questions([region], {}, [])
    proposal, = generated
    assert proposal['origin']['kind'] == 'unidentified_parameters'
    assert proposal['agenda_basis']['parameter_region'] == {'lower': .21, 'upper': .29}
    assert proposal['use_sites'] == region['use_sites']
    assert 'query_expression' not in proposal['query_semantics']
    assert 'family' in proposal['author_work_request']['required_output']


def test_old_theory_context_mapping_requests_a_test_not_transferred_confirmation():
    current = question()
    current['theory_target'] = theory('target', {'scene': 'C5', 'unit': 'mV'})
    current.pop('query_semantics')
    matching = {'current': {'status': 'REBIND_THEN_TEST', 'index_version': 8,
                            'matches': [{'source_id': 'old_theory', 'status': 'same_family'}]}}
    proposal, = generate_questions([current], {}, [theory('old_theory', {'scene': 'C4', 'unit': 'V'})], matching=matching)
    assert proposal['origin']['kind'] == 'theory_transfer'
    assert proposal['agenda_basis']['matching_status'] == 'REBIND_THEN_TEST'
    assert {'kind': 'theory', 'id': 'old_theory'} in proposal['origin']['source_refs']
    assert 'author_work_request' in proposal and 'support' not in proposal


def test_T20_funded_offline_theory_mission_is_selected_without_immediate_mse_gain():
    mission = {'mission_id': 'shared', 'purpose': 'theory_learning', 'phase': 'theory', 'allocation': 3}
    theories = [theory('one', {'scene': 'C4'}), theory('two', {'scene': 'C5'})]
    agenda = build_agenda([question()], {}, theories, mission=mission,
                          budget={'compute': 100, 'author_calls': 1})
    assert agenda['selected_queue'] == 'theory'
    assert agenda['selected_question_id'] == 'agenda:shared:shared'
    assert agenda['theory_agenda'][0]['work_kind'] == 'consolidation'
    proposal, = generate_questions([], {}, theories, mission=mission)
    assert proposal['origin']['kind'] == 'multi_episode_commonality'
    assert {ref['id'] for ref in proposal['origin']['source_refs']} == {'one', 'two'}
    assert 'program' not in proposal and 'support' not in proposal


def test_theory_without_allocation_and_without_author_budget_remains_deferred():
    theories = [theory('one', {'scene': 'C4'}), theory('two', {'scene': 'C5'})]
    mission = {'purpose': 'method_learning', 'phase': 'theory', 'theory_learning_allocation': 0}
    agenda = build_agenda([], {}, theories, mission=mission, budget={'compute': 100, 'author_calls': 10})
    assert agenda['selected_question_id'] is None and agenda['theory_agenda']
    assert agenda['deferred'][0]['reason'] == 'theory_allocation_missing'
    mission['theory_learning_allocation'] = {'compute': 100}
    agenda = build_agenda([], {}, theories, mission=mission, budget={'compute': 100, 'author_calls': 0})
    assert agenda['selected_queue'] == 'theory'
    assert agenda['theory_agenda'][0]['work_kind'] == 'consolidation'
    assert not agenda['deferred']
    assert not generate_questions([], {}, theories, mission={'purpose': 'task_prediction'})


def test_theory_conflict_is_an_obligation_not_a_claim_that_programs_disagree():
    mission = {'purpose': 'theory_learning', 'allocation': 1}
    theories = [theory('one', {'scene': 'C5'}, 1.), theory('two', {'scene': 'C5'}, 2.)]
    proposal, = generate_questions([], {}, theories, mission=mission)
    assert proposal['origin']['kind'] == 'theory_conflict_or_redundancy'
    assert proposal['queue'] == 'theory' and 'author_work_request' in proposal
    assert 'query_expression' not in proposal['query_semantics']


def test_supported_task_can_end_while_mechanism_question_remains_unresolved():
    unresolved = question('mechanism')
    unresolved.update(queue='theory', mission={'purpose': 'theory_learning', 'allocation': 5})
    action = {'action': 'finish', 'arguments': {'model': 'retained.py'}}
    mission = {'purpose': 'task_prediction', 'task_supported': True, 'supported_action': action}
    agenda = build_agenda([unresolved], {}, [], mission=mission, budget={'compute': 20})
    assert agenda['status'] == 'task_supported' and agenda['selected_action'] == action
    assert agenda['selected_question_id'] is None
    assert agenda['theory_agenda'][0]['question_id'] == 'mechanism'


def test_T28_completed_equal_basis_is_not_reselected_and_changed_basis_reopens():
    current = question()
    current['agenda_basis'] = {'index_version': 1, 'rows': 20, 'program_version': 3}
    current['completed_agenda_basis'] = deepcopy(current['agenda_basis'])
    agenda = build_agenda([current], {}, [], budget={'compute': 20})
    assert agenda['selected_question_id'] is None
    assert agenda['task_agenda'][0]['status'] == 'completed'
    current['agenda_basis']['rows'] += 1
    agenda = build_agenda([current], {}, [], budget={'compute': 20})
    assert agenda['selected_question_id'] == current['id']
    current.pop('completed_agenda_basis')
    current['semantic_answer'] = {'status': 'scoped_unknown'}
    assert build_agenda([current], {}, [], budget={'compute': 20})['selected_question_id'] == current['id']


def test_existing_paid_query_is_schedulable_with_no_provider_or_author_budget():
    current = question()
    current['origin'] = 'user'
    agenda = build_agenda([current], {}, [], budget={'compute': 1, 'author_calls': 0})
    assert agenda['selected_question_id'] == 'current' and agenda['selected_queue'] == 'task'
    assert agenda['task_agenda'][0]['work_kind'] == 'query'
    assert 'query_semantics' not in agenda['task_agenda'][0]


def test_existing_stale_dependency_precedes_an_unrelated_ready_question():
    current, premise = question(), question('calibration')
    candidates = {'parent.py': {'status': 'needs_rebuild', 'changed_question': 'calibration',
        'inputs': {'gain': {'kind': 'parameter_region', 'question_id': 'calibration', 'version': 1}}}}
    agenda = build_agenda([current, premise], candidates, [], budget={'compute': 20})
    assert agenda['selected_question_id'] == 'calibration'


def test_explicit_theory_purpose_does_not_require_an_mse_gain_or_phase_alias():
    mission = {'mission_id': 'principle', 'purpose': 'method_learning', 'allocation': 1}
    theories = [theory('one', {'scene': 'C4'}), theory('two', {'scene': 'C5'})]
    agenda = build_agenda([question()], {}, theories, mission=mission,
                          budget={'compute': 100, 'author_calls': 1})
    assert agenda['selected_queue'] == 'theory'


def test_unknown_matching_without_a_source_does_not_invent_an_old_theory():
    current = question()
    current['theory_target'] = theory('target', {'scene': 'C5'})
    assert not generate_questions([current], {}, [],
        matching={'current': {'status': 'UNKNOWN_MATCH', 'matches': []}})


def test_execution_failure_does_not_generate_a_scientific_premise_diagnosis():
    premise = question('calibration')
    failure = {'id': 'error', 'failed_confirmation': {'passed': False, 'returncode': 1, 'mse': None},
               'premise_refs': ['calibration']}
    assert not generate_questions([failure, premise], {}, [])


def test_shared_structure_dispatch_without_typed_sources_returns_author_work_not_an_answer():
    from types import SimpleNamespace
    from hegel_machine.cognition.question_object_queries import evaluate_shared_structure

    current = question('shared_structure')
    current['query_semantics'] = {'mode': 'solution_relation',
                                  'query_expression': {'op': 'FindSharedStructure'}}
    agenda = build_agenda([current], {}, [], budget={'compute': 100, 'author_calls': 0})
    assert agenda['selected_question_id'] == current['id']
    assert agenda['task_agenda'][0]['work_kind'] == 'query'
    result = evaluate_shared_structure(SimpleNamespace(), current,
        current['query_semantics']['query_expression'], None, None)
    assert result['status'] == 'author_work_request'
    assert result['reason'] == 'at_least_two_typed_program_refs_required'
    assert 'value' not in result and result['compute_cost'] == 0
