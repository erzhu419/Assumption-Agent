#!/usr/bin/env python3
"""Freeze new question IDs while preserving the same finite historical abilities.

The policy is fixed: one automatic research route under a funded theory mission,
then the same retained successor question. New source questions come from the
original agenda generator; their formulas and mappings come from existing tools.
"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from hegel_machine.cognition.abstractions import extract_templates
from hegel_machine.cognition.evaluation import loss, predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.question_routing import register_theories, resolved_theories
from replay_frozen_theory_ablation import (
    PANEL_INDICES, ProgramCatalog, PublicReplayLab, _bound_theory, _public_action, _readout,
)
from replay_theory_consolidation import (
    POINTER, ROOT, UNIT, context, declaration, load_source, roles, source_ref,
)
from replay_three_arm_historical_routing import (
    COMPUTE_BUDGET, FIT_INDICES, OBSERVATION_BUDGET, SUFFIX_INDICES, new_agent, old_candidates,
)


OUTPUT = ROOT / 'docs/evidence/question_generation_ablation_20261021.json'
COUNTER_RECEIPT = ROOT / 'docs/evidence/three_arm_historical_routing_20261021.json'
WINDOW_STATE = Path('/home/erzhu419/hegel-active-experiments/luna_window_parameter_transfer_20261017/frozen/target_000/research_state.json')
THEORY_MISSION = {'purpose': 'theory_learning', 'phase': 'theory',
    'mission_id': 'automatic_question_generation_sources',
    'theory_learning_allocation': {'compute': 30000, 'author_calls': 0, 'observations': 0}}
TASK_MISSION = {'purpose': 'task_prediction', 'mission_id': 'retained_question_generation_target'}


def source_gate(payload, source, position):
    """Use the recorded native gate, never a successor label or a new cutoff."""
    episode = payload['episodes'][position]
    threshold = payload['config']['minimum_improvement']
    compares = [(index, action['details']) for index, action in enumerate(episode['actions'])
                if action['action'] == 'compare' and action['details'].get('adopt') is True]
    checks = [(index, action['details']) for index, action in enumerate(episode['actions'])
              if action['action'] == 'prediction_check']
    supported = (episode['status'] == 'completed'
        and episode['stop_reason'] in {'adequate_and_compact', 'knowledge_reused'}
        and episode['validation_loss'] <= threshold and compares and checks
        and checks[-1][1]['revision_loss'] <= threshold)
    if not supported:
        raise ValueError(f'{episode["task_id"]}: original scoped support gate not established')
    declared = declaration(source, episode, position)
    evidence = [source_ref(source, f'episodes[{position}].{field}')
                for field in ('validation_loss', 'stop_reason')]
    evidence.extend(source_ref(source, f'episodes[{position}].actions[{index}]')
                    for index in (compares[0][0], checks[-1][0]))
    declared['support'] = {'status': 'supported_in_scope', 'context_ref': context(episode['task_id']),
                           'evidence': evidence}
    receipt = {'task_id': episode['task_id'], 'gate': 'original_completed_native_support',
        'threshold': threshold, 'threshold_ref': source_ref(source, 'config.minimum_improvement'),
        'gate_source': {'file': str(ROOT / 'src/hegel_machine/cognition/controller.py'),
                        'function': 'CognitiveAgent.run', 'condition': 'native adequate/reuse stop'},
        'validation_loss': episode['validation_loss'], 'stop_reason': episode['stop_reason'],
        'source_evidence_refs': evidence, 'adaptive_development': True,
        'independent_confirmation': False}
    return declared, receipt


def freeze_new_questions(initial_ids, stack):
    """A replay intervention on proposal IDs; tools and ordinary code stay live."""
    import hegel_machine.cognition.question_agenda as agenda
    generate = agenda.generate_questions

    def retained_only(*args, **kwargs):
        return [question for question in generate(*args, **kwargs) if question['id'] in initial_ids]

    stack.enter_context(patch('hegel_machine.cognition.question_agenda.generate_questions', retained_only))
    stack.enter_context(patch('hegel_machine.cognition.question_routing.generate_questions', retained_only))


def window_boundary():
    from hegel_machine.cognition.question_agenda import generate_questions

    if not WINDOW_STATE.exists():
        return {'source_state': str(WINDOW_STATE), 'status': 'UNKNOWN_source_unavailable'}
    retained = json.loads(WINDOW_STATE.read_text())
    proposals = generate_questions(retained['questions'], retained['research_candidates'], [],
                                  mission={'purpose': 'task_prediction'})
    return {'source_state': str(WINDOW_STATE), 'source_bytes': WINDOW_STATE.stat().st_size,
        'retained_question_ids': [question['id'] for question in retained['questions']],
        'generated_question_count': len(proposals),
        'status': 'unresolved_missing_declared_premise_and_parent_use'}


def replay_generation_ablation(pointer=POINTER, output=OUTPUT):
    payload, source = load_source(Path(pointer))
    positions, target_index = (0, 5), 14
    episodes, target = [payload['episodes'][i] for i in positions], payload['episodes'][target_index]
    if [episode['task_id'] for episode in episodes] != ['t01', 't06'] or target['task_id'] != 't15':
        raise AssertionError('The fixed historical source/successor changed')
    gated = [source_gate(payload, source, position) for position in positions]
    declarations, gates = [item[0] for item in gated], [item[1] for item in gated]
    counter_index, counter = next((index, value) for index, value in enumerate(
        payload['knowledge']['concepts'][0]['counterexamples']) if value['task_id'] == 't13')
    counter_ref = source_ref(source, f'knowledge.concepts[0].counterexamples[{counter_index}]')
    prior_receipt = json.loads(COUNTER_RECEIPT.read_text())
    retained_counter = prior_receipt['arms']['A']['counterexample']
    if retained_counter['source_ref'] != counter_ref or retained_counter['status'] != 'refuted':
        raise AssertionError('The original checked counterexample provenance changed')
    counter_checks = deepcopy(retained_counter['checks'])
    row = counter['observation']
    counter_prediction = Program.from_dict(episodes[0]['program']).predict(
        Observation(deepcopy(row['features']), row['target'], tuple(deepcopy(row.get('history', [])))))
    if (counter_checks[0]['status'] != 'failed'
            or abs(counter_prediction - counter_checks[0]['prediction']) > 1e-12
            or counter_checks[0]['actual'] != row['target']):
        raise AssertionError('The shared historical numeric failure no longer reconstructs')
    revision_index, before = next((index, action['details']['before_program'])
        for index, action in enumerate(target['actions']) if action['action'] == 'revise')
    incumbent_ref = source_ref(source, f'episodes[{target_index}].actions[{revision_index}].details.before_program')
    incumbent = _bound_theory('incumbent:t15:before', Program.from_dict(before), None,
                             {'method': 'retained_parent_before_revision', 'source_ref': incumbent_ref})
    old = []
    for episode in episodes:
        template, = extract_templates(Program.from_dict(episode['program']), f'old:{episode["task_id"]}')
        for mapping in template.mappings(('u', 'v')):
            old.append(_bound_theory(f'old:{episode["task_id"]}:{mapping["role_0"]}',
                template.instantiate(mapping), mapping['role_0'], {'method': 'existing_AbstractTemplate'}))
    panel = [{'id': target['actions'][index]['details']['probe_id'],
        'actions': [_public_action(target, index)], 'query': _readout(_public_action(target, index)),
        'tolerance': 1e-6, 'observation_cost': len(_public_action(target, index)['sequence'])}
        for index in PANEL_INDICES]
    typed_target = {'role_signature': {**roles('u'), **roles('v')},
                    'output_unit': UNIT, 'context_ref': context('t15')}
    target_question = {'id': 'retained_t15_response', 'parent_id': 'task', 'depends_on': [],
        'question': 'Which retained alternative applies at this public successor input?',
        'kind': 'value', 'queue': 'task', 'answer_type': 'number',
        'context_ref': context('t15'), 'mission': deepcopy(TASK_MISSION),
        'public_role_signature': deepcopy(typed_target['role_signature']),
        'theory_target': typed_target, 'answer_resolution_ref': {'atol': 1e-6},
        'query_semantics': {'mode': 'model_query', 'query_expression': deepcopy(panel[-1]['query'])},
        'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'task'}],
        'resolution_experiments': deepcopy(panel)}
    catalog, arms = ProgramCatalog(), {}
    with tempfile.TemporaryDirectory(prefix='hegel-question-generation-') as temporary:
        for name in ('on', 'frozen'):
            data = old_candidates(episodes, target, old, incumbent)
            agent, lab = new_agent(Path(temporary) / name), PublicReplayLab()
            register_theories(agent, declarations)
            concrete = []
            for index, entry in enumerate(data['entries']):
                item = _bound_theory(f'native:{index:02d}', entry['program'], None,
                                     {'method': entry['operation']})
                item['role_signature'] = {feature: roles(feature)[feature]
                                          for feature in entry['program'].required_features()}
                item['parameter_paths'] = []
                concrete.append(item)
            failed = _bound_theory('retained:t13:failed_instance', Program.from_dict(episodes[0]['program']),
                                   'x', {'method': 'retained_checked_failure'})
            failed.update(context_ref=context('t13'), compatible=False)
            failed['support'] = {'status': 'refuted', 'context_ref': context('t13'), 'evidence': [counter_ref]}
            alias = {**deepcopy(failed), 'id': 'ordinary_alias:failed_t13', 'compatible': True}
            alias['support'] = {'status': 'candidate', 'context_ref': context('t13'), 'evidence': []}
            register_theories(agent, [*old, incumbent, *concrete, failed, alias])
            agent.theory_consolidation = {'candidates': {}, 'transfers': [{
                'theory_ref': {'id': failed['id'], 'version': 1}, 'binding': {'context_ref': context('t13')},
                'checks': deepcopy(counter_checks), 'historical_execution_ref': {
                    'file': str(COUNTER_RECEIPT), 'json_path': ['arms', 'A', 'counterexample']}}]}
            agent.costs['compute_cost'] = data['old_propose_compute'] + data['prefix_comparison_compute'] + 1
            agent._open_question(deepcopy(target_question))
            initial_ids = frozenset(question['id'] for question in agent.questions)
            source_start = agent.costs['compute_cost']
            with ExitStack() as stack:
                if name == 'frozen':
                    freeze_new_questions(initial_ids, stack)
                source_result = agent._execute('run', {'operator': 'research_route',
                    'mission': deepcopy(THEORY_MISSION),
                    'episode_sources': [source_ref(source, f'episodes[{index}]') for index in positions],
                    'counterexamples': {'history:t01': [{'source_ref': counter_ref,
                        'observation': deepcopy(row), 'context_ref': context('t13'), 'tolerance': 1e-6}]},
                    'materials': [{'id': 'retained_t13_counterexample', 'kind': 'error', 'source_ref': counter_ref}],
                    'allocation': {'compute': 30000, 'max_groups': 4}}, lab)
                source_compute = agent.costs['compute_cost'] - source_start
                selected_source_id = agent.research_agenda['selected_question_id']
                target_result = agent._execute('run', {'operator': 'research_route',
                    'research_question': {'id': target_question['id']}, 'mission': deepcopy(TASK_MISSION)}, lab)
            selected = target_result.get('selection')
            if selected is None:
                raise AssertionError(target_result)
            recall = target_result.get('frozen_recall', {})
            pool = {item['id']: item for item in resolved_theories(agent)}
            if pool[alias['id']].get('compatible') is not False:
                raise AssertionError('The shared source counterexample was lost')
            extra_comparison = 0
            for identifier in recall.get('generated_theory_ids', []):
                program = Program.from_dict(pool[identifier]['program'])
                if any(entry['program'] == program for entry in data['entries']):
                    continue
                extra_comparison += len(data['prefix'])
                agent.costs['compute_cost'] += len(data['prefix'])
                data['entries'].append({'program': program, 'candidate_id': identifier,
                    'operation': 'automatic_frozen_recall', 'order': len(data['entries']),
                    'prefix_mse': loss(program, data['prefix'], data['fallback'], .5)})
            chosen = min(data['entries'], key=lambda item: (item['prefix_mse'], item['program'].size, item['order']))
            # All generation, fitting and plan decisions are fixed before suffix labels are read.
            suffix = [Observation(deepcopy(target['evaluation']['prediction_examples'][index]['features']),
                target['evaluation']['prediction_examples'][index]['target']) for index in SUFFIX_INDICES]
            predictions = [predict(chosen['program'], observed, data['fallback']) for observed in suffix]
            agent.costs['compute_cost'] += len(suffix)
            new_questions = [{key: deepcopy(question[key]) for key in
                ('id', 'kind', 'queue', 'origin', 'use_sites') if key in question}
                for question in agent.questions if question['id'] not in initial_ids]
            source_choice = next(q for q in agent.questions if q['id'] == selected_source_id)
            parent = next(q for q in agent.questions if q['id'] == 'task')
            arms[name] = {'initial_question_ids': sorted(initial_ids), 'new_questions': new_questions,
                'source_phase': {'status': source_result['status'], 'selected_question_id': selected_source_id,
                    'selected_origin': deepcopy(source_choice.get('origin')),
                    'frozen_candidates': deepcopy(source_result.get('frozen_candidates', [])),
                    'compute_cost': source_compute},
                'old_template_binding_count': len(old), 'raw_old_proposals': data['raw_proposal_count'],
                'candidate_program_ids': [catalog.identifier(entry['program']) for entry in data['entries']],
                'automatic_recall': {key: deepcopy(recall.get(key)) for key in
                    ('status', 'generated_theory_ids', 'work', 'author_work', 'excluded', 'new_compute_cost')},
                'selection': {key: deepcopy(selected.get(key)) for key in
                    ('status', 'scores', 'tied_experiments', 'groups', 'public_row_validation')},
                'selected_experiment_id': (selected.get('selected_experiment') or {}).get('id'),
                'parent_pending_query': deepcopy(parent.get('pending_query')),
                'parent': {'program_id': catalog.identifier(chosen['program']), 'prefix_mse': chosen['prefix_mse'],
                    'suffix_predictions': predictions, 'suffix_mse': loss(chosen['program'], suffix, data['fallback'], .5),
                    'independent_confirmation': False},
                'counterexample_retained': pool[alias['id']].get('compatible') is False,
                'source_instances_preserved': all(pool[item['id']].get('compatible') is not False for item in declarations),
                'costs': {'old_propose': data['old_propose_compute'],
                    'old_prefix_comparison': data['prefix_comparison_compute'], 'counterexample_recheck': 1,
                    'automatic_source_phase': source_compute,
                    'target_route': target_result['new_compute_cost'],
                    'new_candidate_prefix_comparison': extra_comparison,
                    'suffix_predictions': len(suffix), 'total': agent.costs['compute_cost']},
                'selected_action_executed': False, 'action_benefit': 'UNKNOWN_no_retained_panel_labels',
                'new_observations': lab.execute_calls, 'new_model_calls': agent.costs['model_calls']}
            if (agent.costs['compute_cost'] > COMPUTE_BUDGET or lab.execute_calls
                    or agent.costs['model_calls'] or agent.observations):
                raise AssertionError('The historical ablation exceeded its fixed resource contract')
    on, frozen = arms['on'], arms['frozen']
    receipt = {'status': 'completed', 'scope': 'finite_historical_question_generation_ablation', 'source': source,
        'protocol': {'intervention': 'freeze all proposal IDs outside the initial question set',
            'execution_policy': 'one automatic theory-phase route, then the same retained target route',
            'same_total_compute_budget': COMPUTE_BUDGET, 'same_observation_budget': OBSERVATION_BUDGET,
            'theory_mission': THEORY_MISSION, 'target_mission': TASK_MISSION,
            'source_episode_indices': list(positions), 'target_episode_index': target_index,
            'retained_source_episode_refs': [source_ref(source, f'episodes[{index}]') for index in positions],
            'public_panel_action_indices': list(PANEL_INDICES),
            'development_indices': list(FIT_INDICES), 'successor_indices': list(SUFFIX_INDICES),
            'retained_incumbent_ref': incumbent_ref, 'counterexample_ref': counter_ref,
            'retained_checked_failure_ref': {'file': str(COUNTER_RECEIPT), 'json_path': ['arms', 'A', 'counterexample']},
            'same_low_level_tools_and_full_old_library': True,
            'original_flags_ref': source_ref(source, 'config'),
            'new_question_or_formula_supplied_to_on': False, 'suffix_labels_used_for_generation_or_selection': False},
        'source_gates': gates, 'program_catalog': catalog.to_dict(), 'arms': arms,
        'comparison': {'same_candidate_programs': on['candidate_program_ids'] == frozen['candidate_program_ids'],
            'same_parent_program': on['parent']['program_id'] == frozen['parent']['program_id'],
            'same_suffix_predictions': on['parent']['suffix_predictions'] == frozen['parent']['suffix_predictions'],
            'same_experiment_choice': on['selected_experiment_id'] == frozen['selected_experiment_id'],
            'on_minus_frozen_compute': on['costs']['total'] - frozen['costs']['total']},
        'historical_costs_shared': {episode['task_id']: {key: episode[key] for key in
            ('observation_cost', 'compute_cost')} for episode in [*episodes, payload['episodes'][12], target]},
        'window_boundary': window_boundary(),
        'limitations': ['Operator-selected source pair, original gate interpretation and successor; no autonomous LLM discovery.',
            'Only two saved development and two historical successor examples; no new independent confirmation.',
            'The selected public action is prepared, not executed; action benefit remains UNKNOWN.']}
    Path(output).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pointer', type=Path, default=POINTER)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = replay_generation_ablation(arguments.source_pointer, arguments.output)
    print(json.dumps({'status': result['status'], 'comparison': result['comparison'],
        'new_questions': {name: len(arm['new_questions']) for name, arm in result['arms'].items()}}, ensure_ascii=False))
