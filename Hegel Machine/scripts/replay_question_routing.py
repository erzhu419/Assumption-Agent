#!/usr/bin/env python3
"""Replay PR2 routing on retained trial_043 records, without acquisition or fitting.

Only the small evidence receipt is retained. Programs, paid rows and their full
public histories are read from the original files; no research_program graph or
paid-trace copy is exported. Units below name the original numeric coordinates,
not inferred physical units or a validated cross-scene mapping.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.question_queries import evaluate_on_model
from hegel_machine.cognition.question_routing import resolved_theories
from hegel_machine.cognition.question_selection import select_question_plan
from hegel_machine.cognition.world_experiment_choice import forecast_equivalence_classes


SOURCE = Path('/mnt/hegel-snapshot-cow/luna_relation_scope_experiment_20261019/'
              'author/trials/trial_043')
OUTPUT = Path(__file__).resolve().parents[1] / 'docs/evidence/question_routing_replay_20261020.json'
UNIT = 'original_public_numeric_coordinate:response'
CONTEXT = {'episode': 'luna_relation_scope_experiment_20261019/trial_043',
           'measurement': 'response', 'coordinate_contract': 'original_public_numeric_coordinates',
           'output_unit': UNIT}
ROLES = {name: {'role': f'public_control:{name}',
                'unit': f'original_public_numeric_coordinate:{name}', 'time': 'current'}
         for name in ('x', 'z', 'c')}


class ReplayLab:
    def __init__(self):
        self.execute_calls = 0

    def execute(self, action):
        self.execute_calls += 1
        raise AssertionError('Offline routing replay must never purchase an observation')

    def public_spec(self):
        return {'controls': [{'name': name} for name in ROLES],
                'replay': 'retained public numeric coordinates; no physical unit inference'}


def readout(record, index, step):
    row = record['rows'][step]
    return {'op': 'Readout', 'record_index': index, 'step': step,
            'target': 'response', 'context_ref': CONTEXT, 'units': UNIT,
            'binding': {'action': record['action'], 'features': row['features'],
                        'history': list(row.get('history', []))}}


def summary(result):
    selection = result.get('selection') or {}
    selected = selection.get('selected_experiment')
    return {'status': result['status'], 'cache_hit': result.get('cache_hit', False),
            'new_compute_cost': result.get('new_compute_cost', 0),
            'matching': {key: result['matching'][key] for key in
                ('status', 'confirmation', 'compute_cost')},
            'matches': [{key: item[key] for key in
                ('source_id', 'status', 'reason', 'confirmation', 'compute_cost')}
                for item in result['matching']['matches']],
            'answer_execution': result['answer_execution'],
            'selection': {key: selection.get(key) for key in
                ('status', 'scope', 'unknown_reason', 'groups', 'pairwise_edges', 'scores',
                 'compute_cost', 'excluded', 'support_updated', 'theory_status', 'tied_experiments')},
            'public_row_validation': selection.get('public_row_validation'),
            'selected_experiment_id': selected['id'] if selected else None,
            'parent_use': result.get('parent_use')}


def replay(source, output):
    state_file, observations_file = source / 'research_state.json', source / 'laboratory_observations.json'
    state = json.loads(state_file.read_text())
    observations = json.loads(observations_file.read_text())
    native = next(q for q in state['questions'] if q['id'] == 'multivariate_native_basis')
    rounds = native['native_condition_research']
    assert len(rounds) == 2 and all(len(r['candidates']) == 1 for r in rounds)
    programs_equal = rounds[0]['candidates'][0]['program'] == rounds[1]['candidates'][0]['program']
    assert programs_equal  # This receipt cannot manufacture a second mechanism.
    bank_id, bank = next(iter(state['relation_research'].items()))
    assert bank['status'] == bank['candidates'][0]['status'] == 'refuted'
    paid_cost = sum(record['observation_cost'] for record in observations)

    # Preserve the actual action, public features and complete captured history.
    # The panel is fixed by public control identity, never selected using targets.
    panel = []
    for index, label in ((2, 'zero_background'), (3, 'x_pulse'), (4, 'z_pulse'), (5, 'c_pulse')):
        record = observations[index]
        step = next((i for i, row in enumerate(record['rows'])
                     if any(row['features'].values())), 0)
        panel.append({'id': f'paid_{index}_{label}', 'actions': [record['action']],
            'query': readout(record, index, step), 'tolerance': 0.,
            'observation_cost': record['observation_cost'],
            'source_ref': {'file': str(observations_file), 'record_index': index, 'step': step}})
    # One additional public zero-background step is not in the paid records.
    # It cannot be answered by accidentally reading an already paid target.
    background = observations[2]
    sequence = background['action']['sequence']
    future_action = {**background['action'], 'sequence': sequence + [sequence[-1]]}
    future = {'op': 'Readout', 'record_index': len(observations), 'step': len(sequence),
        'context_ref': CONTEXT, 'units': UNIT, 'target': 'response',
        'binding': {'action': future_action, 'features': sequence[-1], 'history': sequence}}
    question = {'id': 'replay_future_response', 'parent_id': 'task', 'depends_on': [],
        'question': 'What do the retained alternatives predict for one unmeasured zero-background continuation?',
        'kind': 'observable_relation', 'context_ref': CONTEXT, 'answer_type': 'number',
        'answer_resolution_ref': {'required': 'response at the explicit future public binding'},
        'query_semantics': {'mode': 'model_query', 'query_expression': future},
        'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'task'}]}
    declarations = []
    for index, round_ in enumerate(rounds):
        declarations.append({'id': f'native_round_{index}', 'source_ref': {
            'kind': 'native_candidate', 'question_id': native['id'],
            'round_index': index, 'candidate_index': 0},
            'role_signature': ROLES, 'output_unit': UNIT, 'context_ref': CONTEXT,
            'binding_ref': {'episode': CONTEXT['episode'], 'measurement': 'response'},
            'parameter_paths': [],
            'support': {'status': 'paid_fit', 'context_ref': CONTEXT, 'evidence': [{
                'file': str(state_file), 'question_id': native['id'], 'round_index': index,
                'field': 'candidates[0].fit_evidence'}]},
            'measured_coverage_ref': {'file': str(observations_file),
                'record_indices': round_['fit_record_indices']}})
    declarations.append({'id': 'refuted_relation_round_2', 'source_ref': {
        'kind': 'relation_candidate', 'question_id': bank_id,
        'round_index': bank['round_index'], 'candidate_index': 0},
        'role_signature': ROLES, 'output_unit': UNIT, 'context_ref': CONTEXT,
        'support': {'status': 'refuted', 'evidence': [{
            'file': str(state_file), 'field': f'relation_research.{bank_id}'}]}})
    target = {key: declarations[0][key] for key in
              ('role_signature', 'output_unit', 'context_ref', 'binding_ref', 'parameter_paths')}
    target = {**target, 'id': 'same_coordinate_program', 'program': rounds[0]['candidates'][0]['program']}
    mission = {'mission_id': 'retained_history_routing_replay', 'purpose': 'method_learning',
               'theory_learning_allocation': {'compute': 20000, 'author_calls': 0, 'observations': 0}}
    lab = ReplayLab()
    with tempfile.TemporaryDirectory(prefix='hegel-question-routing-replay-') as temporary:
        agent = CommonArchiveInquiryAgent(None, temporary, arm='hegel', source_task=False,
            config=InquiryLLMConfig(compute_budget=20000,
                observation_budget=paid_cost + 128, execution_seconds=30.))
        # Only the retained executable candidate contract is installed. The
        # large research_program graph and historical model files stay at source.
        agent.questions.append({'id': native['id'], 'question': native['question'],
            'parent_id': 'task', 'depends_on': [], 'status': native['status'],
            'native_condition_research': [{**{key: r[key] for key in
                ('round_index', 'status', 'fit_evidence', 'exported_models')}, 'candidates': [{
                'program': r['candidates'][0]['program'], 'source': r['candidates'][0]['source']}]} for r in rounds]})
        relation_question = next(q for q in state['questions'] if q['id'] == bank_id)
        agent.questions.append({key: relation_question[key] for key in
                                ('id', 'question', 'parent_id', 'depends_on', 'status')})
        agent.relation_research[bank_id] = {'status': bank['status'], 'round_index': bank['round_index'],
            'candidates': [{'program': bank['candidates'][0]['program'], 'status': 'refuted'}]}
        agent.observations = observations
        agent.rows = [Observation(row['features'], row['target'], tuple(row.get('history', [])))
                      for record in observations for row in record['rows']]
        agent.costs['observation_cost'] = paid_cost
        agent.costs['measurement_cost'] = sum(record.get('measurement_cost', 0) for record in observations)
        before = deepcopy(agent.costs)
        arguments = {'operator': 'research_route', 'research_question': question,
            'theories': declarations, 'experiments': panel, 'target': target, 'mission': mission}
        routed = agent._execute('run', deepcopy(arguments), lab)
        pool = resolved_theories(agent)
        assert len(pool) == 3 and all('program' in item for item in pool)
        assert pool[2]['compatible'] is False and pool[2]['source_status'] == 'refuted'
        expanded = routed['selection']
        assert expanded['status'] == 'scoped_unknown'
        assert expanded['unknown_reason'] == 'no_task_answer_disagreement'
        assert len(expanded['groups']) == 1
        assert expanded['groups'][0]['theory_ids'] == ['native_round_0', 'native_round_1']
        assert expanded['selected_experiment'] is None and 'pending_query' not in agent.questions[0]
        after_first = deepcopy(agent.costs)
        repeated = agent._execute('run', deepcopy(arguments), lab)
        assert repeated['cache_hit'] is True and repeated['new_compute_cost'] == 0
        repeat_costs_unchanged = agent.costs == after_first
        assert repeat_costs_unchanged

        compact = select_question_plan(question, pool, panel[:1])
        clones = [dict(pool[0], id=f'native_clone_{index}') for index in range(100)]
        copied = select_question_plan(question, pool + clones, panel)
        assert copied['scores'] == expanded['scores']
        assert copied['status'] == expanded['status']
        assert copied['compute_cost'] == expanded['compute_cost']
        assert copied['selected_experiment'] is None and copied['tied_experiments'] == expanded['tied_experiments']
        assert len(copied['groups']) == 1 and len(copied['groups'][0]['theory_ids']) == 102

        # Baseline uses the same task-and-panel forward predictions with the
        # existing exact grouping helper. It also has no discriminating pair.
        forecasts = []
        for expression in [future, *(plan['query'] for plan in panel)]:
            values = {}
            for item in pool[:2]:
                program = Program.from_dict(item['program'])
                answer = evaluate_on_model(expression, lambda row: program.predict(
                    Observation(row['features'], 0., tuple(row['history']))), context=CONTEXT)
                assert answer['status'] == 'answered' and answer['evidence'] == []
                values[item['id']] = answer['value']
            forecasts.append({'predictions': values})
        baseline_groups = forecast_equivalence_classes(forecasts)
        assert baseline_groups == [['native_round_0', 'native_round_1']]

        # Preserve an actual old question as author work; no exact-conflict
        # witness or new explanatory structure is manufactured from its text.
        original_obligation = next(q for q in state['questions'] if q['id'] == 'source006_order_washout')
        obligation = {key: original_obligation[key] for key in ('id', 'question', 'parent_id', 'depends_on')}
        obligation.update(kind='structural_relation', context_ref=CONTEXT, answer_type='relation',
            answer_resolution_ref={'source_file': str(state_file),
                'field': 'questions[source006_order_washout].evidence_readout_proposal',
                'record_indices': original_obligation['evidence_readout_proposal']['record_indices']},
            query_semantics={'mode': 'solution_relation'},
            use_sites=[{'kind': 'supply_theory_library_candidate', 'mission_id': mission['mission_id']}])
        work = agent._execute('run', {'operator': 'research_route', 'research_question': obligation}, lab)
        assert work['status'] == 'author_work_request'
        assert work['answer_execution']['reason'] == 'missing_executable_query'
        assert work['selection'] is None
        assert lab.execute_calls == 0 and agent.costs['model_calls'] == 0
        assert agent.costs['observation_cost'] == before['observation_cost']
        assert agent.costs['measurement_cost'] == before['measurement_cost']
        receipt = {'status': 'passed', 'scope': 'offline_real_history_routing_replay',
            'source_files': [str(state_file), str(observations_file)],
            'source_counts': {'paid_records': len(observations), 'paid_rows': len(agent.rows),
                              'historical_observation_cost': paid_cost},
            'coordinate_contract': {'context_ref': CONTEXT, 'role_signature': ROLES,
                'output_unit': UNIT, 'physical_unit_inference': False, 'transfer_confirmation': False},
            'native_sources': [{'id': declarations[i]['id'], 'source_ref': pool[i]['source_ref'],
                'source': r['candidates'][0]['source'], 'fit_mse': r['candidates'][0]['fit_mse'],
                'fit_record_count': len(r['fit_record_indices']), 'historical_status': r['status']}
                for i, r in enumerate(rounds)],
            'native_programs_equal': programs_equal,
            'refuted_source': {'source_ref': pool[2]['source_ref'], 'status': pool[2]['source_status'],
                               'compatible': pool[2]['compatible']},
            'future_query': {'record_index': future['record_index'], 'step': future['step'],
                'history_length': len(future['binding']['history']),
                'features': future['binding']['features'], 'not_a_paid_record': True},
            'panel': [{**plan['source_ref'], 'id': plan['id'], 'tolerance': plan['tolerance'],
                'history_length': len(plan['query']['binding']['history']),
                'observation_cost_if_repeated': plan['observation_cost']} for plan in panel],
            'routed': summary(routed),
            'expanded_panel': {'compact_group_count': len(compact['groups']),
                'expanded_group_count': len(expanded['groups']), 'split': False,
                'reason': 'both retained Programs are exactly equal'},
            'clone_check': {'clones_added': 100, 'source_count': 102, 'signature_count': 1,
                'scores_equal': True, 'selection_set_equal': True, 'compute_cost_equal': True,
                'evidence': copied['evidence'], 'support_updated': copied['support_updated']},
            'baseline': {'implementation': 'world_experiment_choice.forecast_equivalence_classes',
                'same_task_and_panel': True, 'groups': baseline_groups,
                'selected_experiment': None, 'tied_experiments': [],
                'conclusion': 'same one-signature ambiguity; no discriminating pair'},
            'repeat_call': {'cache_hit': repeated['cache_hit'],
                'new_compute_cost': repeated['new_compute_cost'], 'costs_unchanged': repeat_costs_unchanged},
            'retained_obligation': {'id': obligation['id'], 'question': obligation['question'],
                'record_indices': obligation['answer_resolution_ref']['record_indices'],
                'status': work['status'], 'reason': work['answer_execution']['reason'],
                'origin': 'existing authored question; not a newly discovered conflict'},
            'no_acquisition': {'execute_calls': lab.execute_calls, 'model_calls': agent.costs['model_calls'],
                'new_observation_cost': agent.costs['observation_cost'] - before['observation_cost'],
                'new_measurement_cost': agent.costs['measurement_cost'] - before['measurement_cost'],
                'pending_query': agent.questions[0].get('pending_query')},
            'limitations': ['The two retained native rounds contain one exact Program, so this history cannot demonstrate panel-induced splitting or positive selection.',
                'Paid fitting is not independent confirmation; native matching remains UNKNOWN and the refuted bank is excluded.',
                'The historical representation diagnostics contain no exact-conflict witnesses; the saved order/washout question remains author work.',
                'This replay tests routing and reuse on a finite retained pool; it establishes neither a new mechanism nor future predictive improvement.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = replay(args.source, args.output)
    print(json.dumps({'status': result['status'], 'output': str(args.output),
                      'routing_status': result['routed']['status'],
                      'no_acquisition': result['no_acquisition']}, ensure_ascii=False))
