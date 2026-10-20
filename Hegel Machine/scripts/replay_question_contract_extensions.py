#!/usr/bin/env python3
"""Exercise question-contract extensions on the retained historical archive.

The finite-predicate bridge is an operator-supplied integration fixture. It
tests an existing frozen Program on saved labels and stops a pending plan;
it does not supply a revised scientific predictor or acquire an observation.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

from hegel_machine.cognition.evaluation import predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.question_routing import resolved_theories
from replay_cross_skeleton_relations import CONTEXT, new_agent, public_action
from replay_theory_consolidation import (
    POINTER, ROOT, UNIT, declaration, context, roles, load_source, source_ref,
)
from retained_cross_skeleton_evidence import recover_t07_evidence


OUTPUT = ROOT / 'docs/evidence/question_contract_extensions_20261021.json'


class NoAcquisitionLab:
    def __init__(self):
        self.execute_calls = 0

    def public_spec(self):
        return {'controls': [{'name': name} for name in ('x', 'z', 'u', 'v')]}

    def execute(self, action):
        self.execute_calls += 1
        raise AssertionError('Contract replay must use retained evidence only')

    def action_cost(self, action):
        return len(action['sequence'])

    def prediction_rows(self, action):
        return [Observation(deepcopy(point), 0., tuple(deepcopy(action['sequence'][:index])))
                for index, point in enumerate(action['sequence'])]


def finite_application(payload, source):
    evidence = recover_t07_evidence(payload, source)
    rows = evidence['comparison_rows']
    target = payload['episodes'][6]
    program = Program.from_dict(target['actions'][44]['details']['after_program'])
    predictions = [predict(program, row, target['fallback']) for row in rows]
    action = public_action(target, 65)
    cases = [{'op': 'RelationResidual', 'expected': prediction, 'atol': 1e-6,
        'query': {'op': 'Readout', 'record_index': 0, 'step': index, 'context_ref': CONTEXT,
            'binding': {'action': deepcopy(action), 'features': deepcopy(row.features),
                        'history': list(deepcopy(row.history))}}}
        for index, (row, prediction) in enumerate(zip(rows, predictions))]
    lab = NoAcquisitionLab()
    with tempfile.TemporaryDirectory(prefix='hegel-finite-question-replay-') as temporary:
        agent = new_agent(temporary)
        agent.observations = [{'action': deepcopy(action), 'rows': [asdict(row) for row in rows],
                               'row_indices': list(range(len(rows)))}]
        agent.rows = list(rows)
        parent = agent._open_question({'id': 'retained_parent', 'question': 'Review this pending retained plan'})
        parent['pending_query'] = {'actions': [deepcopy(action)]}
        agent._save('shared/finite_parent_use.py', '''import json
def select_actions(spec, observations, memory):
    state=json.load(open('research_state.json'))
    child=next(q for q in state['questions'] if q['id']=='finite_forecast_check')
    answer=child['semantic_answer']
    assert answer['scope']['kind']=='finite_cases'
    if not answer['value']:
        return {'stop':'The frozen forecast is violated on this retained finite set.'}
    parent=next(q for q in state['questions'] if q['id']=='retained_parent')
    return parent['pending_query']
''')
        question = {'id': 'finite_forecast_check', 'question': 'Do all these retained forecasts satisfy their checks?',
            'parent_id': parent['id'], 'depends_on': [], 'kind': 'applicability',
            'answer_type': 'boolean_claim', 'context_ref': deepcopy(CONTEXT),
            'answer_resolution_ref': {'atol': 1e-6, 'scope': 'action65 steps0/1 only'},
            'query_semantics': {'mode': 'model_query', 'query_expression': {
                'op': 'AllOnFiniteSet', 'finite_cases': cases}},
            'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': parent['id'],
                           'path': 'shared/finite_parent_use.py'}]}
        outcome = agent._execute('run', {'operator': 'research_step', 'research_question': question}, lab)
        costs = deepcopy(agent.costs)
        repeated = agent._execute('run', {'operator': 'research_step',
                                         'research_question': {'id': question['id']}}, lab)
        if not outcome.get('parent_revision_completed') or 'pending_query' in parent:
            raise AssertionError('The finite answer was not consumed by its declared parent use')
        if repeated['status'] != 'reused_answer' or agent.costs != costs:
            raise AssertionError('An unchanged finite answer repeated completed work')
        if lab.execute_calls or costs['model_calls'] or costs['observation_cost']:
            raise AssertionError('The historical contract replay acquired external data')
        return {'scope': 'operator_supplied_parent_use_of_a_finite_historical_check',
            'program_ref': source_ref(source, 'episodes[6].actions[44].details.after_program'),
            'case_source_refs': deepcopy(evidence['comparison_refs']),
            'frozen_forecasts': predictions, 'finite_answer': outcome['answer'],
            'parent_use': outcome['uses'], 'parent_revision_completed': outcome['parent_revision_completed'],
            'pending_plan_removed': 'pending_query' not in parent,
            'old_plan_history_count': len(parent.get('pending_query_history', [])),
            'prior_plan_ref': source_ref(source, 'episodes[6].actions[65]'),
            'repeat_status': repeated['status'], 'repeat_compute': 0,
            'frozen_forecast_compute': len(rows), 'common_costs': costs,
            'new_paid_observations': 0, 'new_provider_calls': 0,
            'scientific_scope': 'finite failure of the original frozen instance; no family refutation or new predictor'}


def automatic_frozen_recall(payload, source):
    """Replay t01/t06 construction and t15 recall without choosing an object id."""
    mission = {'purpose': 'theory_learning', 'mission_id': 'historical_ordered_affine_replay',
               'theory_learning_allocation': {'compute': 30000, 'author_calls': 0, 'observations': 0}}
    sources = [(position, payload['episodes'][position]) for position in (0, 5)]
    target_position, target = 14, payload['episodes'][14]
    if [episode['task_id'] for _, episode in sources] != ['t01', 't06'] or target['task_id'] != 't15':
        raise AssertionError('The original source and successor identities changed')
    declarations = [declaration(source, episode, position) for position, episode in sources]
    counterexamples = payload['knowledge']['concepts'][0]['counterexamples']
    counter_index, counter = next((index, item) for index, item in enumerate(counterexamples)
                                  if item['task_id'] == 't13')
    counter_ref = source_ref(source, f'knowledge.concepts[0].counterexamples[{counter_index}]')
    lab = NoAcquisitionLab()
    with tempfile.TemporaryDirectory(prefix='hegel-automatic-frozen-recall-') as temporary:
        source_agent = new_agent(Path(temporary) / 'source')
        built = source_agent._execute('run', {'operator': 'theory_consolidation',
            'theme': 'ordered_affine', 'theories': declarations,
            'source_theory_ids': [item['id'] for item in declarations],
            'episode_sources': [source_ref(source, f'episodes[{position}]') for position, _ in sources],
            'counterexamples': {'history:t01': [{'source_ref': counter_ref,
                'observation': deepcopy(counter['observation']), 'context_ref': context('t13'), 'tolerance': 1e-6}]},
            'mission': deepcopy(mission), 'allocation': {'compute': 30000, 'max_groups': 4}}, lab)
        target_path = Path(temporary) / 'target'
        library_path = built['library_path']
        (target_path / library_path).parent.mkdir(parents=True)
        (target_path / library_path).write_text((source_agent.workspace / library_path).read_text())
        agent = new_agent(target_path)
        revision_index, revision = next((index, action['details']) for index, action in enumerate(target['actions'])
                                        if action['action'] == 'revise')
        incumbent = Program.from_dict(revision['before_program'])
        original = {'id': 'retained:t15:initial_parent', 'program': incumbent.to_dict(),
            'role_signature': roles('u'), 'output_unit': UNIT, 'context_ref': context('t15'),
            'support': {'status': 'candidate', 'evidence': [], 'context_ref': context('t15')}}
        panel_indices = [index for index, action in enumerate(target['actions']) if action['action'] == 'observe'][:4]

        def readout(action):
            return {'op': 'Readout', 'record_index': 0, 'step': 0, 'target': 'response',
                'context_ref': context('t15'), 'units': UNIT,
                'binding': {'action': deepcopy(action), 'features': deepcopy(action['sequence'][0]), 'history': []}}

        panels = [{'id': target['actions'][index]['details']['probe_id'],
            'actions': [public_action(target, index)], 'query': readout(public_action(target, index)),
            'tolerance': 1e-6, 'observation_cost': len(public_action(target, index)['sequence'])}
            for index in panel_indices]
        question = {'id': 't15_automatic_frozen_recall', 'parent_id': 'task', 'depends_on': [],
            'question': 'Which retained candidate applies at this public successor input?',
            'kind': 'value', 'answer_type': 'number', 'context_ref': context('t15'),
            'answer_resolution_ref': {'atol': 1e-6},
            'query_semantics': {'mode': 'model_query', 'query_expression': deepcopy(panels[0]['query'])},
            'theory_target': {'role_signature': roles('u'), 'output_unit': UNIT, 'context_ref': context('t15')},
            'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'task'}]}
        task_mission = {'purpose': 'task', 'mission_id': 'retained_t15_successor'}
        result = agent._execute('run', {'operator': 'research_route', 'research_question': question,
            'theories': [original], 'experiments': panels, 'mission': deepcopy(task_mission)}, lab)
        recalled = result['frozen_recall']
        pool = {item['id']: item for item in resolved_theories(agent)}
        generated = {identifier: Program.from_dict(pool[identifier]['program'])
                     for identifier in recalled['generated_theory_ids']}
        selection = result.get('selection') or {}
        consumed = {identifier for group in selection.get('groups', []) for identifier in group['theory_ids']}
        costs = deepcopy(agent.costs)
        library_writes = []
        save = agent._save

        def record_save(path, content):
            if path == library_path:
                library_writes.append(path)
            return save(path, content)

        agent._save = record_save
        try:
            repeated = agent._execute('run', {'operator': 'research_route',
                'research_question': {'id': question['id']}}, lab)
        finally:
            agent._save = save
        # All source/object choices and public-panel selection are frozen before
        # reading the target's four saved evaluation examples or their labels.
        rows = target['evaluation']['prediction_examples']
        forecasts = {identifier: [program.predict(Observation(row['features'], 0.)) for row in rows]
                     for identifier, program in generated.items()}
        before_predictions = [incumbent.predict(Observation(row['features'], 0.)) for row in rows]
        if lab.execute_calls or agent.costs['model_calls'] or agent.observations:
            raise AssertionError('Automatic historical recall acquired external data')
        return {'status': 'completed' if generated and set(generated) <= consumed else 'UNKNOWN',
            'scope': 'operator_selected_historical_sources_and_typed_successor_contract',
            'protocol': {'source_episode_indices': [0, 5], 'target_episode_index': target_position,
                'candidate_id_supplied_to_successor': False, 'explicit_theory_transfer_calls': 0,
                'target_final_program_or_parameters_used': False, 'target_labels_used_for_recall_or_selection': False,
                'initial_parent_ref': source_ref(source, f'episodes[{target_position}].actions[{revision_index}].details.before_program'),
                'typed_target': deepcopy(question['theory_target']), 'source_construction_mission': mission,
                'mission': task_mission, 'recall_uses_existing_task_budget': True,
                'panel_action_indices': panel_indices, 'new_paid_observations': 0, 'new_provider_calls': 0},
            'source_construction': {'status': built['status'], 'new_compute_cost': built['new_compute_cost'],
                'frozen_candidates': built['frozen_candidates'],
                'counterexample_ref': counter_ref,
                'retained_failed_binding_count': sum(len(candidate['blocked_transfer_bindings'])
                    for candidate in agent.theory_consolidation['candidates'].values()),
                'source_reconstructions_preserved': all(check['exact'] for candidate in
                    agent.theory_consolidation['candidates'].values() for check in candidate['source_reconstruction'])},
            'automatic_recall': recalled, 'original_pool_preserved': original['id'] in pool,
            'selection': {key: deepcopy(selection.get(key)) for key in (
                'status', 'groups', 'scores', 'selected_experiment', 'tied_experiments')},
            'generated_candidates_consumed_by_selector': set(generated) <= consumed,
            'repeat': {'cache_hit': repeated.get('cache_hit'), 'recall_cache_hit': repeated['frozen_recall'].get('cache_hit'),
                'new_compute_cost': repeated['new_compute_cost'], 'costs_unchanged': agent.costs == costs,
                'library_write_calls': len(library_writes)},
            'saved_evaluation_excerpt': {'scope': 'four already saved target examples, not independent confirmation',
                'source_ref': source_ref(source, f'episodes[{target_position}].evaluation.prediction_examples'),
                'row_count': len(rows), 'predictions': forecasts, 'initial_parent_predictions': before_predictions,
                'target_values': [row['target'] for row in rows],
                'initial_parent_mse': sum((value - row['target']) ** 2 for value, row in zip(before_predictions, rows)) / len(rows),
                'recalled_candidate_mse': {identifier: sum((value - row['target']) ** 2
                    for value, row in zip(values, rows)) / len(rows) for identifier, values in forecasts.items()},
                'predictor_adoption_executed': False, 'selected_experiment_executed': False},
            'costs': {'source_construction': deepcopy(source_agent.costs), 'successor': costs,
                      'saved_excerpt_predictions': (1 + len(generated)) * len(rows)},
            'independent_confirmation': False,
            'limitations': ['The source theme and typed successor contract are operator-selected, not autonomous discovery.',
                'The retained initial parent is not the complete old executable-archive baseline; this is an automatic-connection check.',
                't01/t06 share abstraction lineage. Four saved evaluation examples do not establish whole-task or incremental theory-learning gain.']}


def replay(pointer=POINTER, output=OUTPUT):
    payload, source = load_source(pointer)
    result = {'status': 'completed', 'source': source, 'finite_application': finite_application(payload, source),
              'automatic_frozen_recall': automatic_frozen_recall(payload, source)}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pointer', type=Path, default=POINTER)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = replay(arguments.pointer, arguments.output)
    print(json.dumps({'output': str(arguments.output), 'status': result['status'],
        'finite_answer': result['finite_application']['finite_answer'],
        'parent_revision_completed': result['finite_application']['parent_revision_completed']}, ensure_ascii=False))
