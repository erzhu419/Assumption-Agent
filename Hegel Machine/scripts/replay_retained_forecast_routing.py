#!/usr/bin/env python3
"""Route declared endpoint or full-vector questions over saved Python forecasts.

The complete two-panel protocol stays fixed. The question is each panel's
first trajectory endpoint minus its second endpoint, at the original step23,
or an explicitly requested ordered vector of all 48 original panel coordinates.
No predictor, fit or acquisition is executed and no scientific Program is made.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.question_queries import evaluate_on_evidence, numeric_answer_distance
from hegel_machine.cognition.question_routing import resolved_theories
from replay_retained_python_forecasts import SOURCE, ROOT, reference


OUTPUT = ROOT / 'docs/evidence/retained_forecast_routing_20261021.json'
VECTOR_OUTPUT = ROOT / 'docs/evidence/retained_vector_routing_20261021.json'
UNIT = 'original_public_numeric_coordinate:response'
CONTEXT = {'episode': 'luna_window_parameter_transfer_20261017/target_000',
    'measurement': 'response', 'coordinate_contract': 'original_public_numeric_coordinates',
    'output_unit': UNIT}
ROLES = {'u': {'role': 'public_control:u', 'unit': 'original_public_numeric_coordinate:u', 'time': 'current'}}
ENDPOINT_STEP, ANSWER_ATOL, FORECAST_TOLERANCE = 23, 1e-6, 0.


class PublicReplayLab:
    def __init__(self):
        self.execute_calls = 0

    def execute(self, action):
        self.execute_calls += 1
        raise AssertionError('Retained forecast routing must never acquire an observation')

    def public_spec(self):
        return {'controls': [{'name': 'u'}]}

    def prediction_rows(self, action):
        return [Observation(deepcopy(point), 0., tuple(deepcopy(action['sequence'][:index])))
                for index, point in enumerate(action['sequence'])]


def endpoint_difference(actions, first_record):
    if len(actions) != 2 or any(len(action['sequence']) != 24 for action in actions):
        raise ValueError('The original complete two-trajectory panel is required')

    def readout(index):
        action = actions[index]
        return {'op': 'Readout', 'record_index': first_record + index, 'step': ENDPOINT_STEP,
            'target': 'response', 'context_ref': deepcopy(CONTEXT), 'units': UNIT,
            'binding': {'action': deepcopy(action), 'features': deepcopy(action['sequence'][ENDPOINT_STEP]),
                        'history': deepcopy(action['sequence'][:ENDPOINT_STEP])}}

    return {'op': 'Difference', 'left': readout(0), 'right': readout(1)}


def full_panel_readout(actions, first_record):
    """An ordered finite vector, using every original row in both trajectories."""
    return {'op': 'Readout', 'statistic_ref': {'kind': 'vector'}, 'finite_readouts': [
        {'op': 'Readout', 'record_index': first_record + index, 'step': step,
         'target': 'response', 'context_ref': deepcopy(CONTEXT), 'units': UNIT,
         'binding': {'action': deepcopy(action), 'features': deepcopy(point),
                     'history': deepcopy(action['sequence'][:step])}}
        for index, action in enumerate(actions) for step, point in enumerate(action['sequence'])]}


def build_inputs(state, state_file, *, readout='endpoint_difference'):
    if readout not in {'endpoint_difference', 'full_panel_vector'}:
        raise ValueError('unsupported retained readout')
    read = endpoint_difference if readout == 'endpoint_difference' else full_panel_readout
    """Use only original public protocol and pre-acquisition forecast addresses."""
    comparison = state['experiment_comparisons'][0]
    declarations = []
    for index, forecast in enumerate(comparison['forecasts']):
        if forecast['model'] not in comparison['eligible_models']:
            continue
        declarations.append({'id': f'retained_python_forecast_{index}',
            'forecast_ref': {**reference(state_file, 'experiment_comparisons', 0, 'forecasts', index),
                             'model': forecast['model']},
            'role_signature': deepcopy(ROLES), 'output_unit': UNIT, 'context_ref': deepcopy(CONTEXT),
            'parameter_paths': [], 'support': {'status': 'candidate', 'evidence': []}})
    panel = [{'id': experiment['id'], 'actions': deepcopy(experiment['actions']),
        'query': read(experiment['actions'], 24 + 2 * index),
        'observation_cost': experiment['observation_cost'], 'tolerance': FORECAST_TOLERANCE}
        for index, experiment in enumerate(comparison['comparisons'])]
    question = {'id': 'retained_python_endpoint_difference', 'parent_id': 'task', 'depends_on': [],
        'question': 'What is the response difference between the two original trajectory endpoints at step23?',
        'kind': 'observable_relation', 'answer_type': 'number', 'context_ref': deepcopy(CONTEXT),
        'answer_resolution_ref': {'atol': ANSWER_ATOL, 'source': 'operator_supplied_engineering_resolution',
            'scope': 'original two-trajectory endpoint difference; not an MSE or measurement-noise conversion'},
        'public_role_signature': deepcopy(ROLES),
        'theory_target': {'role_signature': deepcopy(ROLES), 'output_unit': UNIT, 'context_ref': deepcopy(CONTEXT)},
        'query_semantics': {'mode': 'model_query', 'query_expression': deepcopy(panel[0]['query'])},
        'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'task'}]}
    if readout == 'full_panel_vector':
        question.update(id='retained_python_full_panel_vector', answer_type='vector',
            question='What is the ordered response vector on every original row of the two trajectories?')
        question['answer_resolution_ref']['scope'] = 'each declared vector coordinate; not an MSE or measurement-noise conversion'
    return declarations, panel, question


def replay(source=SOURCE, output=None, *, readout='endpoint_difference'):
    output = output or (VECTOR_OUTPUT if readout == 'full_panel_vector' else OUTPUT)
    source = Path(source)
    state_file, observations_file = source / 'research_state.json', source / 'laboratory_observations.json'
    state = json.loads(state_file.read_text())
    declarations, panel, question = build_inputs(state, state_file, readout=readout)
    limits = state['limits']
    lab = PublicReplayLab()
    with tempfile.TemporaryDirectory(prefix='hegel-retained-forecast-routing-') as temporary:
        agent = CommonArchiveInquiryAgent(None, temporary, arm='hegel', source_task=False,
            config=InquiryLLMConfig(compute_budget=limits['compute_budget'],
                observation_budget=limits['observation_budget'], max_model_calls=0,
                execution_seconds=limits['execution_seconds'], confirmation_rows=limits['confirmation_rows'],
                confirmation_mse_threshold=limits['confirmation_mse_threshold']))
        # The historical account is already paid. It stays charged while the
        # replay reads no target labels before its public selection is frozen.
        agent.costs.update({key: value for key, value in state['costs'].items() if key in agent.costs})
        before = deepcopy(agent.costs)
        result = agent._execute('run', {'operator': 'research_route', 'research_question': question,
            'theories': declarations, 'experiments': panel,
            'mission': {'purpose': 'task', 'mission_id': 'retained_python_forecast_routing'}}, lab)
        selection = result.get('selection')
        if selection is None:
            raise AssertionError(result)
        costs = deepcopy(agent.costs)
        repeated = agent._execute('run', {'operator': 'research_route',
            'research_question': {'id': question['id']}}, lab)
        repeated_costs_unchanged = agent.costs == costs
        pool = resolved_theories(agent)
        distinguishing = {**deepcopy(question), 'id': 'retained_python_distinguishing_experiment',
            'kind': 'distinguishing_experiment', 'answer_type': 'binding',
            'candidate_theory_ids': [item['id'] for item in declarations],
            'resolution_experiments': deepcopy(panel),
            'query_semantics': {'mode': 'model_query', 'query_expression': {
                'op': 'FindDistinguishingExperiment',
                'target_query': deepcopy(question['query_semantics']['query_expression']),
                'target_answer_resolution_ref': deepcopy(question['answer_resolution_ref']),
                'claim_refs': [{'id': item['id'], 'version': item['version']} for item in pool],
                'legal_constructor_ref': {'kind': 'resolution_experiments'}}}}
        before_distinguishing = agent.costs['compute_cost']
        found = agent._execute('run', {'operator': 'research_step',
            'research_question': distinguishing}, lab)
        distinguishing_compute = agent.costs['compute_cost'] - before_distinguishing
        if any('program' in item for item in agent.theory_index.values()):
            raise AssertionError('Retained scientific Python was incorrectly registered as a Program')
        indexed = json.dumps(agent.theory_index)
        library_file = agent.workspace / 'shared/theory_library.json'
        library = library_file.read_text() if library_file.exists() else '{}'
        if any(key in indexed or key in library for key in ('"prediction_inputs"', '"_retained_forecast"')):
            raise AssertionError('A transient retained forecast table was copied into the index or library')
        if lab.execute_calls or agent.costs['model_calls'] != before['model_calls'] or agent.observations:
            raise AssertionError('The historical route executed acquisition or a model')
        frozen = {key: deepcopy(selection.get(key)) for key in ('status', 'groups', 'scores', 'pairwise_edges',
            'tied_experiments', 'effective_pairs', 'public_row_validation', 'excluded', 'compute_cost', 'tie_diagnostics')}
        frozen['selected_experiment_id'] = (selection.get('selected_experiment') or {}).get('id')
        # Labels are opened only after candidates, complete panels and route freeze.
        observations = json.loads(observations_file.read_text())
        checked = evaluate_on_evidence(question['query_semantics']['query_expression'], observations,
                                       context=CONTEXT, compute_budget=limits['compute_budget'])
        if checked['status'] != 'answered':
            raise AssertionError(checked)
        consequences = [{'theory_ids': deepcopy(group['theory_ids']), 'frozen_value': group['task_answer'],
            'absolute_error': numeric_answer_distance(group['task_answer'], checked['value']),
            'within_declared_resolution': numeric_answer_distance(group['task_answer'], checked['value']) <= ANSWER_ATOL}
            for group in selection['groups']]
        new_compute = costs['compute_cost'] - before['compute_cost']
        receipt = {'status': 'completed', 'scope': 'Common_PR2_saved_Python_forecast_finite_table_historical_replay',
            'source_refs': {'state': reference(state_file), 'paid_records': reference(observations_file),
                'original_comparison': reference(state_file, 'experiment_comparisons', 0),
                'original_revision': reference(state_file, 'revisions', 0)},
            'protocol': {'evidence_cutoff': deepcopy(state['experiment_comparisons'][0]['evidence_cutoff']),
                'endpoint_step': ENDPOINT_STEP,
                'query_contract': {'op': 'Difference', 'left': {'record_index': 24, 'step': ENDPOINT_STEP},
                    'right': {'record_index': 25, 'step': ENDPOINT_STEP}, 'answer_type': 'number',
                    'role_signature': deepcopy(ROLES), 'context_ref': deepcopy(CONTEXT)},
                'panels': [{'id': experiment['id'],
                    'source_ref': reference(state_file, 'experiment_comparisons', 0, 'comparisons', index),
                    'action_count': len(experiment['actions']),
                    'trajectory_lengths': [len(action['sequence']) for action in experiment['actions']],
                    'observation_cost': experiment['observation_cost'], 'endpoint_step': ENDPOINT_STEP}
                    for index, experiment in enumerate(panel)],
                'full_original_panel_retained': True,
                'original_limits': deepcopy(limits), 'original_mse_threshold': state['experiment_comparisons'][0]['mse_threshold'],
                'answer_resolution_atol': ANSWER_ATOL, 'forecast_tolerance': FORECAST_TOLERANCE,
                'resolution_origin': 'operator_supplied_before_replay; independent of labels and original MSE threshold',
                'labels_used_for_question_candidates_or_selection': False,
                'scientific_predictors': 'original arbitrary Python, addressed by frozen forecast_ref',
                'scientific_Program_created': False, 'new_model_executions': 0, 'new_fit_runs': 0,
                'new_observations': 0, 'new_model_calls': 0},
            'registered_candidates': deepcopy(declarations),
            'resolved_candidates': [{key: deepcopy(item.get(key)) for key in
                ('id', 'version', 'forecast_ref', 'source_status', 'compatible', 'support')} for item in pool],
            'route': {'status': result['status'], 'matching': deepcopy(result['matching']),
                'selection': frozen, 'parent_use': deepcopy(result.get('parent_use')),
                'new_compute_cost': new_compute},
            'repeat': {'cache_hit': repeated.get('cache_hit'), 'new_compute_cost': repeated.get('new_compute_cost'),
                'costs_unchanged': repeated_costs_unchanged},
            'find_distinguishing': {'operator': 'research_step', 'status': found['status'],
                'query_answer': deepcopy(found.get('query_answer')),
                'claim_refs': deepcopy(distinguishing['query_semantics']['query_expression']['claim_refs']),
                'selection': {key: deepcopy((found.get('selection') or {}).get(key)) for key in
                    ('status', 'groups', 'scores', 'tied_experiments', 'effective_pairs', 'compute_cost')},
                'new_compute_cost': distinguishing_compute,
                'same_scores_as_research_route': (found.get('selection') or {}).get('scores') == selection.get('scores')},
            'historical_consequence_check': {'query_contract_ref': {'field': 'protocol.query_contract'},
                'readout_source_refs': [reference(observations_file, index, 'rows', ENDPOINT_STEP) for index in (24, 25)],
                'result': checked, 'frozen_consequences': consequences,
                'record_indices': [24, 25], 'observed_scope': 'two terminal responses only',
                'full_48_row_score_ref': {'file': str(ROOT / 'docs/evidence/retained_python_forecasts_20261021.json'),
                                        'json_path': ['forecasts']},
                'unexecuted_panel': {'id': panel[1]['id'], 'status': 'UNKNOWN', 'record_indices': []},
                'support_updated': False, 'independent_confirmation': False},
            'costs': {'original_account': deepcopy(before), 'Common_route': new_compute,
                'Common_FindDistinguishingExperiment': distinguishing_compute,
                'historical_two_readout_check': checked.get('compute_cost', 0),
                'total_new_compute': new_compute + distinguishing_compute + checked.get('compute_cost', 0),
                'original_comparison_compute': state['experiment_comparisons'][0]['compute_cost'],
                'new_acquisition_cost': 0},
            'storage': {'scientific_Programs_in_index': False, 'forecast_columns_copied_to_index_or_library': False,
                'source_file_copied': False, 'source_access': 'absolute_forecast_ref'},
            'failures_preserved': {'all_original_models': deepcopy(state['experiment_comparisons'][0]['models']),
                'original_prior_fit_ref': reference(state_file, 'experiment_comparisons', 0, 'prior_fit'),
                'original_full_successor_failures_ref': {'file': str(ROOT / 'docs/evidence/retained_python_forecasts_20261021.json'),
                                                       'json_path': ['forecasts']}},
            'limitations': ['This is a finite retained forecast table, not Python source induction or a scientific Program.',
                'The endpoint difference and its engineering resolution are operator supplied.',
                'Only the historically executed panel has labels; no new action or independent confirmation is obtained.']}
    if readout == 'full_panel_vector':
        receipt['protocol']['readout'] = readout
        receipt['protocol']['query_contract'] = {'op': 'Readout', 'statistic_ref': {'kind': 'vector'},
            'ordered_records': [24, 25], 'steps_per_record': list(range(24)), 'answer_type': 'vector',
            'role_signature': deepcopy(ROLES), 'context_ref': deepcopy(CONTEXT)}
        receipt['protocol']['resolution_origin'] = 'operator supplied pointwise vector resolution before replay; unrelated to original MSE gate'
        receipt['protocol'].pop('endpoint_step')
        for item in receipt['protocol']['panels']:
            item.pop('endpoint_step')
            item['readout_coordinate_count'] = 48
        check = receipt['historical_consequence_check']
        check['readout_source_refs'] = [reference(observations_file, index, 'rows') for index in (24, 25)]
        check['observed_scope'] = 'all 48 ordered responses in the paid panel'
        receipt['costs']['historical_finite_readout_check'] = receipt['costs'].pop('historical_two_readout_check')
        receipt['limitations'][1] = 'The complete finite vector and pointwise engineering resolution are operator supplied.'
        # Every 48-coordinate result already exists in a retained column or the
        # two paid records. Store references instead of repeating those arrays.
        retained_values = [(checked['value'], {'file': str(observations_file), 'record_indices': [24, 25],
                                             'ordered_field': 'rows[*].target'})]
        for index, forecast in enumerate(state['experiment_comparisons'][0]['forecasts']):
            for start in (0, 48):
                retained_values.append((forecast['predictions'][start:start + 48],
                    {**reference(state_file, 'experiment_comparisons', 0, 'forecasts', index, 'predictions'),
                     'slice': [start, start + 48]}))

        def references(value):
            if isinstance(value, list):
                if len(value) == 48 and all(type(item) in (int, float) for item in value):
                    for original, ref in retained_values:
                        if value == original:
                            return {'retained_values_ref': deepcopy(ref)}
                return [references(item) for item in value]
            if isinstance(value, tuple):
                return [references(item) for item in value]
            if isinstance(value, dict):
                return {key: references(item) for key, item in value.items()}
            return value

        receipt = references(receipt)
        receipt['storage']['finite_vectors_stored_by_reference'] = True
    Path(output).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--readout', choices=('endpoint_difference', 'full_panel_vector'), default='endpoint_difference')
    args = parser.parse_args()
    result = replay(args.source, args.output, readout=args.readout)
    print(json.dumps({'status': result['status'], 'route': result['route']['status'],
        'selection': result['route']['selection']['status'], 'compute': result['costs']['total_new_compute']}, ensure_ascii=False))
