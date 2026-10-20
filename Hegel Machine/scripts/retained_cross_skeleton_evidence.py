#!/usr/bin/env python3
"""Recover the small, actually retained t07 evidence used by frozen replays.

No simulator or predictor supplies a target. Repeated memory counterexamples
contribute source references, not extra observations or independent episodes.
"""

from copy import deepcopy

from hegel_machine.cognition.models import Observation
from replay_theory_consolidation import POINTER, load_source, source_ref


EPISODE_INDEX = 6
FREEZE_ACTION_INDEX = 44


def _observation_rows(observation):
    names = tuple(observation['features'])
    rows = [Observation({name: point[name] for name in names}, point['__target__'],
                        tuple(deepcopy(observation['history'][:step])))
            for step, point in enumerate(observation['history'])]
    rows.append(Observation(deepcopy(observation['features']), observation['target'],
                            tuple(deepcopy(observation['history']))))
    return rows


def _match_action(episode, observation):
    names = tuple(observation['features'])
    history = [{name: point[name] for name in names} for point in observation['history']]
    matches = []
    for index, action in enumerate(episode['actions']):
        if action['action'] != 'observe':
            continue
        sequence = action['details']['sequence']
        if len(sequence) > len(history) and sequence[:len(history)] == history \
                and sequence[len(history)] == observation['features']:
            matches.append(index)
    if len(matches) != 1:
        raise ValueError('The retained observation does not identify one actual t07 action prefix')
    return matches[0]


def recover_t07_evidence(payload, source):
    """Return native rows plus exact provenance for the fixed historical split.

    comparison_rows is action65 steps0/1. later_rows is the four stored final
    evaluation examples in writer order, not a reconstruction of all192 rows.
    """
    episode = payload['episodes'][EPISODE_INDEX]
    if episode['task_id'] != 't07':
        raise ValueError('This recovery helper addresses the retained t07 episode at index6')
    recovered = {}
    memory_record_count = 0
    raw_row_reference_count = 0
    for memory_index, event in enumerate(episode['memory_events']):
        observation = event.get('observation')
        if observation is None:
            continue
        memory_record_count += 1
        action_index = _match_action(episode, observation)
        action = episode['actions'][action_index]
        rows = _observation_rows(observation)
        for step, row in enumerate(rows):
            raw_row_reference_count += 1
            field = f'episodes[{EPISODE_INDEX}].memory_events[{memory_index}].observation'
            if step < len(observation['history']):
                field += f'.history[{step}]'
            reference = source_ref(source, field)
            key = (action_index, step)
            if key in recovered:
                if recovered[key]['row'] != row:
                    raise ValueError('Repeated retained observations disagree about an actual t07 row')
                recovered[key]['references']['source_refs'].append(reference)
                continue
            recovered[key] = {'row': row, 'references': {
                'id': f't07:action{action_index}:step{step}', 'action_index': action_index, 'step': step,
                'probe_id': action['details']['probe_id'], 'source_refs': [reference],
                'action_ref': source_ref(source, f'episodes[{EPISODE_INDEX}].actions[{action_index}]'),
                'original_action_observation_cost': action['observation_cost'],
                'original_action_compute_cost': action['compute_cost'],
                'label_used_for_original_replay_fitting': action_index in (14, 40),
                'label_scored_in_prior_continuation_diagnostics': (action_index, step) == (65, 0),
                'new_blind_evidence': False}}
    training_keys = sorted(key for key in recovered if key[0] <= FREEZE_ACTION_INDEX)
    comparison_keys = sorted(key for key in recovered if key[0] > FREEZE_ACTION_INDEX)
    if training_keys != [(14, 0), (14, 1), (14, 2), (40, 0), (40, 1), (40, 2), (40, 3), (40, 4)] \
            or comparison_keys != [(65, 0), (65, 1)]:
        raise ValueError('The retained t07 evidence differs from the fixed chronological split')
    # The native writer stores evaluation[:4] in order. TemporalEnvironment
    # evaluation trajectories contain8 rows, so these four examples are the
    # first four steps of one trajectory. Only their actually saved targets
    # enter the restored histories; no final Program or generator is executed.
    examples = episode['evaluation']['prediction_examples']
    if len(examples) != 4:
        raise ValueError('The retained final evaluation excerpt is not the four writer-ordered examples')
    later_rows, later_refs, history = [], [], []
    for step, example in enumerate(examples):
        later_rows.append(Observation(deepcopy(example['features']), example['target'], tuple(deepcopy(history))))
        later_refs.append({'id': f't07:final_evaluation:step{step}', 'evaluation_example_index': step,
            'trajectory_index': 0, 'step': step, 'action_index': None,
            'source_refs': [source_ref(source, f'episodes[{EPISODE_INDEX}].evaluation.prediction_examples[{step}]')],
            'history_source_refs': [source_ref(source,
                f'episodes[{EPISODE_INDEX}].evaluation.prediction_examples[{i}]') for i in range(step)],
            'original_collection_role': 'post_learning_final_evaluation_excerpt',
            'original_observation_cost': None, 'original_evaluation_cost_logged': False,
            'new_blind_evidence': False})
        history.append({**deepcopy(example['features']), '__target__': example['target']})
    coverage = {}
    for index, action in enumerate(episode['actions']):
        if index <= FREEZE_ACTION_INDEX or action['action'] != 'observe':
            continue
        steps = [step for action_index, step in comparison_keys if action_index == index]
        coverage[str(index)] = {'probe_id': action['details']['probe_id'],
            'recorded_rows': action['details']['rows'], 'recovered_unique_rows': len(steps),
            'recovered_steps': steps,
            'missing_steps': [step for step in range(action['details']['rows']) if step not in steps],
            'original_action_observation_cost': action['observation_cost'],
            'original_action_compute_cost': action['compute_cost'],
            'action_ref': source_ref(source, f'episodes[{EPISODE_INDEX}].actions[{index}]')}
    return {'source': deepcopy(source), 'episode_index': EPISODE_INDEX,
        'freeze_action_index': FREEZE_ACTION_INDEX,
        'training_rows': [recovered[key]['row'] for key in training_keys],
        'training_refs': [recovered[key]['references'] for key in training_keys],
        'comparison_rows': [recovered[key]['row'] for key in comparison_keys],
        'comparison_refs': [recovered[key]['references'] for key in comparison_keys],
        'later_rows': later_rows, 'later_refs': later_refs, 'coverage': coverage,
        'duplicates': {'memory_records_with_observations': memory_record_count,
            'raw_label_references': raw_row_reference_count,
            'unique_recovered_labels': len(recovered),
            'repeated_label_references_merged': raw_row_reference_count - len(recovered),
            'memory_events0_and1_identify_one_action65_prefix': True},
        'historical_costs': {'episode_observation_cost': episode['observation_cost'],
            'episode_compute_cost': episode['compute_cost'],
            'postfreeze_observation_cost': sum(item['original_action_observation_cost'] for item in coverage.values()),
            'new_observation_cost': 0, 'new_provider_calls': 0,
            'recovering_two_labels_does_not_recharge_or_replace_the_original_eight_row_action_cost': True},
        'later_history_order_contract': {'writer_ref': 'src/hegel_machine/cognition/experiments.py:137',
            'writer_operation': 'evaluation[:4]',
            'trajectory_ref': 'src/hegel_machine/cognition/environments.py:237',
            'native_evaluation_trajectory_length': 8,
            'targets_and_features_restored_only_from_stored_examples': True},
        'retained_summary_refs': {
            'comparison': source_ref(source, 'episodes[6].actions[68]'),
            'final_revision': source_ref(source, 'episodes[6].actions[69]'),
            'prediction_check': source_ref(source, 'episodes[6].actions[70]'),
            'final_evaluation': source_ref(source, 'episodes[6].evaluation')},
        'scope': {'comparison_is_partial_historical_postfreeze_replay': True,
            'complete_original_postfreeze_rows_recovered': False,
            'original_final_evaluation_rows': episode['evaluation']['rows'],
            'restored_final_evaluation_examples': len(later_rows),
            'selected_action_executed': False, 'independent_new_confirmation': False,
            'final_programs_and_predictions_used_as_target_sources': False}}


def load_retained_evidence(pointer=POINTER):
    payload, source = load_source(pointer)
    return recover_t07_evidence(payload, source)
