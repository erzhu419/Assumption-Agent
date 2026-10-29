"""Saved fits can plan a distinct history experiment without refitting them."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import structural_inquiry
from hegel_machine.cognition.finite_native_author import (
    publish_research_options, research_segment_options, select_research_segment,
    research_action_semantics,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.research_programs import _plan
from hegel_machine.cognition.native_condition_research import (
    native_structure_planning_contract, plan_native_structure_experiments,
)


def structural_scene():
    public = {'controls': ['u'], 'feature_bounds': {'u': [-1., 1.]},
        'control_levels': {'u': [-1., -.4, 0., .4, .5, 1.]},
        'reset': True, 'action_kinds': ['trajectory'], 'max_probe_length': 12}
    sequence = [{'u': -.4} for _ in range(12)]
    records = [{'action': {'kind': 'trajectory', 'channels': [], 'sequence': sequence},
        'row_indices': list(range(12)), 'rows': [{'features': dict(point), 'target': 0.,
            'history': deepcopy(sequence[:i])} for i, point in enumerate(sequence)]}]
    gate = Program('ge', children=(Program('feature', name='u'), Program('const', value=.45)))
    active = Program('if', children=(gate, Program('const', value=1.), Program('const', value=0.)))
    candidates = []
    for width, placement in [(2, 'outside_window'), (3, 'outside_window'), (2, 'inside_window')]:
        program = (Program('window_sum', lag=width, children=(active,))
            if placement == 'inside_window' else Program('if', children=(gate,
                Program('window_sum', lag=width, children=(Program('const', value=1.),)),
                Program('const', value=0.))))
        step = {'operation': 'discover_conditional_coordinates',
                'question': 'Synthetic saved conditional response',
                'program': program, 'loss': .1, 'width': width, 'placement': placement,
                'gate_coordinate': Program('feature', name='u').to_dict(),
                'branch_basis': [Program('feature', name='u').to_dict()],
                'threshold_interval': [.4, .5]}
        saved = _plan({'program': program, 'loss': .1, 'terms': (program,),
            'fit_terms': (program,), 'coefficients': [0., 1.]}, [step], 12, 100)
        candidates.append({'program': program.to_dict(), 'fit_mse': .1,
            'model': f'candidate_{len(candidates)}.py', 'source': 'synthetic_saved_fit',
            'research_program': saved})
    result = {'round_index': 0, 'status': 'candidates_require_evaluation',
        'observed_record_count': 1, 'fit_evidence': list(range(12)), 'fit_record_indices': [0],
        'finite_prior': 'synthetic control', 'compute_cost': 100,
        'completion_reserve': {'search_allowance': 100}, 'search_stages': [],
        'proposed_experiments': [], 'candidates': candidates,
        'exported_models': [{'model': c['model']} for c in candidates],
        'evidence': [{'status': 'conditional', 'control': 'u', 'background': {'u': -.4},
            'threshold_interval': [.4, .5], 'conditional_loss': 0., 'conditional_gain': .1,
            'width_candidates': [2, 3], 'placement_candidates': ['inside_window', 'outside_window'],
            'reference_record_index': 0}]}
    question = {'id': 'q', 'question': 'Explain the retained public failure', 'meaning_version': 1,
        'context_ref': {'public': 'lab'}, 'native_condition_research': [result],
        'theory_target': {'revision_fitting': {}}}
    rows = [Observation(row['features'], row['target'], tuple(row['history'])) for row in records[0]['rows']]
    agent = SimpleNamespace(rows=rows, observations=records, questions=[question],
        evaluated_model=None, _inquiry_stack=[], _active_question_id='q',
        candidate_paths=[], method_results=[], operator_results=[],
        theory_consolidation={'candidates': {}, 'executions': {}},
        knowledge={'finite_native_author': {'question': question, 'predictor_paths': ['old.py'],
            'max_candidates': 3}},
        config=SimpleNamespace(compute_budget=2_000_000, observation_budget=100,
            execution_seconds=60., confirmation_rows=2),
        costs={'compute_cost': 100, 'observation_cost': 12, 'python_execution_seconds': 0.})
    lab = SimpleNamespace(public_spec=lambda: public)
    source = {'file': 'research_state.json', 'json_path': ['questions', 0, 'native_condition_research', 0]}
    return public, agent, lab, question, result, source


def test_saved_structure_planner_forecasts_pulse_step_without_any_new_fit(monkeypatch):
    from hegel_machine.cognition.research_programs import _Fitter
    public, agent, lab, question, previous, source = structural_scene()
    before = deepcopy((agent.observations, previous))
    monkeypatch.setattr(_Fitter, 'fit', lambda *a, **k: pytest.fail('planning repeated a paid fit'))
    contract = native_structure_planning_contract(public, agent.observations, previous)
    budget = SearchBudget(contract['proposal_compute'])
    result = plan_native_structure_experiments(public, agent.observations, budget,
        previous=previous, requested=contract, source_ref=source)
    assert result['status'] == 'structure_experiments_planned'
    assert result['planning_complete'] and result['fits'] == 0
    assert budget.spent <= contract['proposal_compute']
    proposed, = result['proposed_experiments']
    target = proposed['structural_discrimination']
    assert set(target['forecast_witnesses']) == {'history_width', 'condition_position'}
    assert all(w['delta_predictions'][0] != w['delta_predictions'][1]
               for w in target['forecast_witnesses'].values())
    assert target['mode'] in {'pulse', 'step'}
    assert proposed['query_action']['arguments']['actions'][0] == agent.observations[0]['action']
    assert all(point['u'] in public['control_levels']['u']
        for point in proposed['query_action']['arguments']['actions'][1]['sequence'])
    assert (agent.observations, previous) == before


def test_unfunded_design_does_not_forecast_and_single_representative_is_not_expanded(monkeypatch):
    public, agent, _, _, previous, source = structural_scene()
    contract = native_structure_planning_contract(public, agent.observations, previous)
    monkeypatch.setattr(Program, 'predict', lambda *a: pytest.fail('unfunded forecast'))
    budget = SearchBudget(0)
    outcome = plan_native_structure_experiments(public, agent.observations, budget,
        previous=previous, requested=contract, source_ref=source)
    assert outcome['reason'] == 'complete_structure_planning_unfunded' and budget.spent == 0
    previous['candidates'] = previous['candidates'][:1]
    assert native_structure_planning_contract(public, agent.observations, previous) is None


def test_new_rows_do_not_resell_an_unchanged_bank_forecast(monkeypatch):
    public, agent, _, _, previous, source = structural_scene()
    contract = native_structure_planning_contract(public, agent.observations, previous)
    agent.observations.append(deepcopy(agent.observations[0]))
    monkeypatch.setattr(Program, 'predict', lambda *a: pytest.fail('unchanged saved-bank forecast'))
    budget = SearchBudget(contract['proposal_compute'])
    result = plan_native_structure_experiments(public, agent.observations, budget,
        previous=previous, requested=contract, source_ref=source)
    assert result['reason'] == 'source_fit_scope_changed_requires_native_evidence_update'
    assert budget.spent == 0 and not result['planning_complete']


def test_standalone_author_export_accepts_the_saved_structure_planning_contract(monkeypatch):
    from hegel_machine.cognition.finite_native_author import export_finite_native_author
    public, agent, _, question, previous, _ = structural_scene()
    contract = native_structure_planning_contract(public, agent.observations, previous)
    monkeypatch.setattr(Program, 'predict', lambda *a: pytest.fail('metadata forecast'))
    namespace = {}
    exec(compile(export_finite_native_author(), 'exported_finite_native_author.py', 'exec'), namespace)
    option = {'round_index': previous['round_index'], 'proposal': {
        'action': 'run', 'arguments': {'operator': 'native_condition_research',
            'structure_planning': contract}}}
    assert namespace['_structure_planning_option_matches'](
        option, public, agent.observations, question)


def planned_agent(monkeypatch):
    public, agent, lab, question, previous, source = structural_scene()
    contract = native_structure_planning_contract(public, agent.observations, previous)
    monkeypatch.setattr(structural_inquiry, 'research_native_conditions',
        lambda *a, **k: pytest.fail('planning invoked fitting research'))
    feedback = structural_inquiry.execute_native_condition_research(agent, {
        'structure_planning': contract, 'max_compute': contract['proposal_compute'],
        'max_candidates': 3}, lab, question)
    return public, agent, lab, question, previous, contract, feedback


def test_planning_records_a_separate_result_and_is_reused_without_paid_computation(monkeypatch):
    public, agent, lab, question, previous, contract, feedback = planned_agent(monkeypatch)
    assert feedback['status'] == 'structure_experiments_planned'
    assert len(question['native_condition_research']) == 1
    assert len(question['native_structure_planning']) == 1
    assert feedback['structure_planning']['research_problem']['answer_type'] == 'binding'
    before = deepcopy(agent.costs)
    monkeypatch.setattr(Program, 'predict', lambda *a: pytest.fail('repeated paid planning'))
    reused = structural_inquiry.execute_native_condition_research(agent, {
        'structure_planning': contract, 'max_compute': contract['proposal_compute']}, lab, question)
    assert reused['cache_hit'] and agent.costs == before


def test_new_q_is_published_with_complete_cost_and_bound_to_the_real_planning_return(monkeypatch):
    public, agent, lab, question, previous, contract, feedback = planned_agent(monkeypatch)
    monkeypatch.setattr(Program, 'predict', lambda *a: pytest.fail('metadata publication forecast'))
    published = publish_research_options(agent, public)
    query = next(o for o in published if o.get('purpose') == 'conditional_history_structure')
    assert query['complete_quote']['observation_cost'] == 12
    assert query['complete_quote']['comparison_compute'] > 0
    assert query['proposal']['arguments']['question']['kind'] == 'structural_discrimination'
    callback, = query['proposal']['arguments']['after_observation']
    assert callback['research_question']['id'] == 'q' and callback['matched_pairs'] == [
        {'reference_record_index': 1, 'record_index': 2}]
    state = {'questions': agent.questions, 'operator_results': [], 'method_results': [],
        'exported_candidate_paths': [],
        'limits': {'compute_budget': 2_000_000, 'observation_budget': 100}, 'costs': agent.costs}
    # The declared pool here includes every saved candidate, as in the actual source.
    state['operator_results'] = [{'operator': 'native_condition_research',
        'research_question_id': 'q', 'native_condition_research': previous}]
    memory = {**agent.knowledge, 'research_state': state, 'method_decision': {'research_options': published}}
    view = research_segment_options(public, agent.observations, memory, state=state)
    assert any(o.get('purpose') == 'conditional_history_structure' for o in view['options']), view['excluded']
    altered = deepcopy(published)
    altered_query = next(o for o in altered if o.get('purpose') == 'conditional_history_structure')
    altered_query['proposal']['arguments']['after_observation'][0]['structural_discrimination'] = {}
    altered_memory = {**memory, 'method_decision': {'research_options': altered}}
    rejected = research_segment_options(public, agent.observations, altered_memory, state=state)
    assert not any(o.get('purpose') == 'conditional_history_structure' for o in rejected['options'])
    first = {'action': 'run', 'arguments': {'operator': 'native_condition_research',
        'structure_planning': contract}, 'research_question': question}
    # Include the same current meaning projection the normal compiler binds.
    first_semantics = research_action_semantics(first)
    memory['program_results'] = [{'proposal': {**first, 'research_segment_index': 0}, 'feedback': feedback}]
    ordinary = {'stop': {'status': 'author_work_request'}}
    selected = select_research_segment(public, agent.observations, memory,
        plan={'first_segment': first_semantics,
            'second_segment': {'followup_of_first_segment': True, 'purpose': 'conditional_history_structure'}},
        ordinary_select_actions=lambda *_: ordinary)
    assert selected['action'] == 'query' and selected['research_segment_index'] == 1
    assert selected['arguments']['after_observation'][0]['structural_discrimination'] == query['structural_discrimination']
    memory['program_results'][0]['feedback'] = {**feedback, 'source_ref': {'file': 'wrong.json', 'json_path': ['other']}}
    assert select_research_segment(public, agent.observations, memory,
        plan={'first_segment': first_semantics,
            'second_segment': {'followup_of_first_segment': True, 'purpose': 'conditional_history_structure'}},
        ordinary_select_actions=lambda *_: ordinary) == ordinary


def test_unplanned_saved_bank_is_a_legal_funded_two_segment_method_option(monkeypatch):
    from hegel_machine.cognition.research_programs import propose_research_action_revisions
    public, agent, _, question, previous, _ = structural_scene()
    monkeypatch.setattr(Program, 'predict', lambda *a: pytest.fail('metadata planner ran predictions'))
    published = publish_research_options(agent, public)
    planning = next(o for o in published if o.get('work_reason') == 'unplanned_structural_ambiguity')
    assert planning['complete_quote']['planning_compute'] == 168
    assert planning['proposal']['arguments']['max_compute'] == 168
    assert planning['complete_quote']['following_research_compute'] > 0
    assert planning['complete_quote']['comparison_compute'] > 0
    state = {'questions': agent.questions, 'operator_results': [{
        'operator': 'native_condition_research', 'research_question_id': 'q',
        'native_condition_research': previous}], 'method_results': [],
        'limits': {'compute_budget': 2_000_000, 'observation_budget': 100}, 'costs': agent.costs}
    memory = {**agent.knowledge, 'research_state': state, 'method_decision': {'research_options': published}}
    view = research_segment_options(public, agent.observations, memory, state=state)
    assert any(o.get('work_reason') == 'unplanned_structural_ambiguity' for o in view['options'])
    ordinary = {'action': 'query', 'research_question': question, 'arguments': {'actions': [
        {'kind': 'trajectory', 'sequence': [{'u': .4}], 'channels': []}]}}
    generated = propose_research_action_revisions(ordinary, view)
    body = next(c['research_selection_plan'] for c in generated['candidates']
        if c['research_selection_plan']['first_segment']['arguments'].get('structure_planning'))
    assert body['second_segment'] == {'followup_of_first_segment': True,
                                     'purpose': 'conditional_history_structure'}


def test_selected_structure_query_keeps_actual_pair_callback_and_returns_new_analysis(tmp_path, monkeypatch):
    from hegel_machine.cognition.online_inquiry_llm import InquiryLLMAgent
    from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent
    native = structural_inquiry.research_native_conditions
    public, agent, lab, question, previous, contract, feedback = planned_agent(monkeypatch)
    monkeypatch.setattr(structural_inquiry, 'research_native_conditions', native)
    consumer = object.__new__(TemporalOperatorAgent)
    consumer.__dict__.update(vars(agent))
    agent = consumer
    agent.workspace = tmp_path
    returned = []
    agent._return_structural_answer = lambda q, answer: returned.append((q['id'], answer))
    option = next(o for o in publish_research_options(agent, public)
                  if o.get('purpose') == 'conditional_history_structure')
    arguments = option['proposal']['arguments']

    def purchase_fixture(self, args, laboratory):
        self.questions.append(deepcopy(args['question']))
        sequence = args['actions'][1]['sequence']
        record = {'action': deepcopy(args['actions'][1]), 'row_indices': list(range(12, 24)),
            'rows': [{'features': dict(point), 'history': deepcopy(sequence[:i]), 'target': 0.}
                     for i, point in enumerate(sequence)]}
        self.observations.append(record)
        self.rows.extend(Observation(r['features'], r['target'], tuple(r['history'])) for r in record['rows'])
        self.costs['observation_cost'] += 12
        return {'after_observation': deepcopy(args['after_observation']),
            'record_indices': [0, 1], 'reused_record_indices': [0], 'record_index_map': {'1': 0, '2': 1}}

    def consume(action, args, laboratory):
        assert action == 'run' and args['operator'] == 'native_condition_research'
        return structural_inquiry.execute_native_condition_research(agent, args, laboratory, question)

    agent._execute = consume
    monkeypatch.setattr(InquiryLLMAgent, '_query', purchase_fixture)
    queried = TemporalOperatorAgent._query(agent, arguments, lab)
    actual = queried['after_observation_results'][0]
    assert not actual.get('error') and not queried.get('after_observation_error'), queried
    current = question['native_condition_research'][-1]
    assert len(question['native_condition_research']) == 2 and len(current['fit_evidence']) == 24
    assert current['structural_discrimination']['matched_pairs'] == [
        {'reference_record_index': 0, 'record_index': 1}]
    assert returned[-1][0] == arguments['question']['id']
    assert returned[-1][1]['answer']['target'] == option['structural_discrimination']
    before = deepcopy(agent.costs)
    monkeypatch.setattr(structural_inquiry, 'research_native_conditions',
        lambda *a, **k: pytest.fail('completed structural analysis repeated paid fitting'))
    callback = {**arguments['after_observation'][0], 'matched_pairs': [
        {'reference_record_index': 0, 'record_index': 1}]}
    reused = structural_inquiry.execute_native_condition_research(agent, callback, lab, question)
    assert reused['cache_hit'] and agent.costs == before
    assert len(question['native_condition_research']) == 2


@pytest.mark.parametrize('change', [None, 'rows', 'meaning', 'truncated', 'pool'])
def test_direct_final_comparison_report_is_reused_only_at_its_complete_scope(change):
    from hegel_machine.cognition.finite_native_author import _completed_choice
    report = {'operator': 'research_choice', 'arguments': {'operator': 'research_choice', 'models': ['a.py']},
        'research_question_id': 'q', 'question_meaning_version': 1, 'paid_rows': 12,
        'evaluations': [{'path': 'a.py'}]}
    state = {'questions': [{'id': 'q', 'meaning_version': 1}], 'operator_results': [report]}
    if change == 'rows':
        report['paid_rows'] = 11
    elif change == 'meaning':
        report['question_meaning_version'] = 2
    elif change == 'truncated':
        report['truncated'] = True
    elif change == 'pool':
        report['evaluations'] = [{'path': 'different.py'}]
    result = _completed_choice([], ['a.py'], 12, state['questions'][0], state)
    assert (result is report) if change is None else result is None
