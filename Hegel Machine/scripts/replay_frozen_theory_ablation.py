#!/usr/bin/env python3
"""Keep/remove one frozen family while retaining all old Program capabilities.

Plans use only public controls from saved research actions. The later native
fit/selection stage explicitly uses a saved evaluation prefix for development;
the remaining saved examples are historical checks, never new confirmation.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from hegel_machine.cognition.abstractions import extract_templates
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.evaluation import initial_prediction, loss, point_loss, predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import Concept, KnowledgeLibrary
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.question_routing import register_theories, resolved_theories
from replay_theory_consolidation import (
    FAMILY, POINTER, ROOT, UNIT, NoAcquisitionLab, context, declaration,
    load_source, roles, source_ref,
)


OUTPUT = ROOT / 'docs/evidence/frozen_theory_ablation_20261020.json'
PANEL_INDICES = (0, 1, 2, 9)


class PublicReplayLab(NoAcquisitionLab):
    def action_cost(self, action):
        return len(action['sequence'])

    def prediction_rows(self, action):
        return [Observation(deepcopy(point), 0., tuple(deepcopy(action['sequence'][:i])))
                for i, point in enumerate(action['sequence'])]


def _public_action(episode, index):
    return {'kind': 'trajectory',
            'sequence': deepcopy(episode['actions'][index]['details']['sequence'])}


def _readout(action):
    return {'op': 'Readout', 'record_index': 0, 'step': 0, 'target': 'response',
        'context_ref': context('t15'), 'units': UNIT,
        'binding': {'action': deepcopy(action), 'features': deepcopy(action['sequence'][0]), 'history': []}}


def _bound_theory(identifier, program, name, origin):
    return {'id': identifier, 'program': program.to_dict(), 'role_signature': roles(name) if name else {},
        'output_unit': UNIT, 'context_ref': context('t15'),
        'parameter_paths': [[]] if program.op == 'const' else [[0], [1, 0]], 'binding_origin': deepcopy(origin),
        'support': {'status': 'candidate', 'evidence': [], 'context_ref': context('t15')}}


class ProgramCatalog:
    """Store each actual Program once; arm records hold ordinary local ids."""

    def __init__(self):
        self.programs = []

    def identifier(self, program):
        if program not in self.programs:
            self.programs.append(program)
        return f'program_{self.programs.index(program):03d}'

    def to_dict(self):
        return {self.identifier(program): program.to_dict() for program in self.programs}


def compare_arm_results(keep, remove):
    """Report observed differences; a changed result is never forced to null."""
    comparisons = {
        'same_candidate_programs': keep['candidate_program_ids'] == remove['candidate_program_ids'],
        'same_signature_predictions': keep['selection']['signature_predictions'] == remove['selection']['signature_predictions'],
        'same_scores': keep['selection']['scores'] == remove['selection']['scores'],
        'same_selected_experiment': keep['selection']['selected_experiment_id'] == remove['selection']['selected_experiment_id'],
        'same_predictions': keep['evaluation']['prediction_matrix'] == remove['evaluation']['prediction_matrix'],
        'same_parent_selected_program': keep['native_parent_choice']['selected_program_id'] == remove['native_parent_choice']['selected_program_id'],
        'same_suffix_predictions': keep['native_parent_choice']['suffix_predictions'] == remove['native_parent_choice']['suffix_predictions'],
    }
    changed = not all(comparisons.values())
    return {**comparisons, 'incremental_effect_observed': changed,
        'incremental_behavior_effect_observed': changed,
        'same_common_compute': keep['selection']['actual_common_compute'] == remove['selection']['actual_common_compute'],
        'given_object_route_compute_difference': keep['selection']['actual_common_compute']
            - remove['selection']['actual_common_compute']}


def _native_parent_choice(source_episodes, target, extra, catalog):
    rows = [Observation(deepcopy(row['features']), row['target'], ())
            for row in target['evaluation']['prediction_examples']]
    prefix, suffix = rows[:2], rows[2:]
    library = KnowledgeLibrary(concepts=[Concept(f'history:{episode["task_id"]}',
        Program.from_dict(episode['program']), .5, episode['task_id'], 'retained_source')
        for episode in source_episodes])
    spec = TaskSpec('historical_ablation_t15', 'Same retained public numeric coordinates',
                    ('u', 'v'), (), ())
    budget = SearchBudget(20000)
    proposed = library.propose(spec, prefix, budget, preserve_parameters=True)
    before = next(action['details']['before_program'] for action in target['actions']
                  if action['action'] == 'revise')
    unique = [{'program': Program.from_dict(before), 'operation': 'retained_parent',
               'candidate_id': 'retained_parent_before_revision'}]
    for candidate in proposed:
        if not any(entry['program'] == candidate.program for entry in unique):
            unique.append({'program': candidate.program, 'operation': candidate.operation,
                           'candidate_id': candidate.candidate_id})
    for theory in extra:
        program = Program.from_dict(theory['program'])
        if not any(entry['program'] == program for entry in unique):
            unique.append({'program': program, 'operation': 'frozen_family_instantiation',
                           'candidate_id': theory['id']})
    # Development labels are admitted only after public experiment selection.
    # The strict minimum and stable insertion order are fixed before suffix use.
    fallback = initial_prediction(prefix, .5)
    comparison = SearchBudget(20000)
    if not comparison.spend(len(prefix) * len(unique)):
        raise AssertionError('Historical prefix comparison exceeded its declared allocation')
    scored = [{**entry, 'prefix_mse': loss(entry['program'], prefix, fallback, .5), 'order': i}
              for i, entry in enumerate(unique)]
    selected = min(scored, key=lambda entry: (entry['prefix_mse'], entry['program'].size, entry['order']))
    program = selected['program']
    prefix_comparison_compute = comparison.spent
    if not comparison.spend(len(suffix)):
        raise AssertionError('Historical suffix check exceeded its declared allocation')
    suffix_predictions = [predict(program, row, fallback) for row in suffix]
    return {'scope': 'historical_evaluation_prefix_development_and_saved_suffix_check',
        'fit_row_indices': [0, 1], 'selection_row_indices': [0, 1], 'suffix_row_indices': [2, 3],
        'selection_rule': 'strict minimum prefix MSE, then Program size, then original stable candidate order',
        'old_propose_raw_candidates': len(proposed), 'unique_candidate_count': len(unique),
        'raw_candidate_count_with_incumbent_and_frozen_aliases': 1 + len(proposed) + len(extra),
        'old_propose_compute': budget.spent, 'prefix_comparison_compute': prefix_comparison_compute,
        'suffix_evaluation_compute': len(suffix), 'comparison_and_suffix_budget': comparison.limit,
        'comparison_and_suffix_compute': comparison.spent,
        'candidate_scores': [{'program_id': catalog.identifier(entry['program']),
            'operation': entry['operation'], 'prefix_mse': entry['prefix_mse']} for entry in scored],
        'selected_program_id': catalog.identifier(program), 'selected_operation': selected['operation'],
        'selected_candidate_id': selected['candidate_id'], 'selected_prefix_mse': selected['prefix_mse'],
        'suffix_predictions': suffix_predictions,
        'suffix_mse': sum(point_loss(value, row.target, .5)
                          for value, row in zip(suffix_predictions, suffix)) / len(suffix),
        'suffix_evaluation_rows': len(suffix),
        'common_paid_observations_created': 0, 'independent_confirmation': False}


def replay_ablation(pointer=POINTER, output=OUTPUT):
    payload, source = load_source(pointer)
    source_positions = (0, 5)
    source_episodes = [payload['episodes'][i] for i in source_positions]
    target = payload['episodes'][14]
    counter_index, counter = next((i, example) for i, example in enumerate(
        payload['knowledge']['concepts'][0]['counterexamples']) if example['task_id'] == 't13')
    counter_ref = source_ref(source, f'knowledge.concepts[0].counterexamples[{counter_index}]')
    counter_check = {'source_ref': counter_ref, 'context_ref': context('t13'),
        'observation': deepcopy(counter['observation']), 'tolerance': 1e-6}
    mission = {'purpose': 'theory_learning', 'mission_id': 'frozen_family_same_information_ablation',
        'theory_learning_allocation': {'compute': 30000, 'author_calls': 0, 'observations': 0}}
    panel = [{'id': target['actions'][i]['details']['probe_id'],
        'actions': [_public_action(target, i)], 'query': _readout(_public_action(target, i)),
        'tolerance': 1e-6, 'observation_cost': len(_public_action(target, i)['sequence'])}
        for i in PANEL_INDICES]
    question = {'id': 'historical_t15_public_response', 'parent_id': 'task', 'depends_on': [],
        'question': 'Which frozen alternative applies at the retained public representative input?',
        'kind': 'value', 'answer_type': 'number', 'context_ref': context('t15'),
        'public_role_signature': {**roles('u'), **roles('v')},
        'answer_resolution_ref': {'atol': 1e-6},
        'query_semantics': {'mode': 'model_query', 'query_expression': _readout(_public_action(target, 9))},
        'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'task'}]}
    declarations = [declaration(source, episode, position)
                    for position, episode in zip(source_positions, source_episodes)]
    construction_arguments = {'operator': 'theory_consolidation', 'theme': 'ordered_affine',
        'theories': declarations, 'source_theory_ids': ['history:t01', 'history:t06'],
        'episode_sources': [source_ref(source, f'episodes[{i}]') for i in source_positions],
        'counterexamples': {'history:t01': [counter_check]},
        'materials': [{'id': 'shared_t13_counterexample', 'kind': 'error', 'source_ref': counter_ref}],
        'mission': mission, 'allocation': {'compute': 30000, 'max_groups': 4}}
    # Construct the intervention object once. Its measured construction cost is
    # charged to keep; neither arm repeats historical learning or acquisition.
    with tempfile.TemporaryDirectory(prefix='hegel-frozen-ablation-object-') as temporary:
        preparation_agent = CommonArchiveInquiryAgent(None, temporary, arm='hegel', source_task=False,
            config=InquiryLLMConfig(compute_budget=50000, observation_budget=8,
                                   max_model_calls=0, execution_seconds=30.))
        preparation = preparation_agent._execute('run', construction_arguments, PublicReplayLab())
        frozen_id = preparation['frozen_candidates'][0]['candidate_id']
        frozen = deepcopy(preparation_agent.theory_consolidation['candidates'][frozen_id])
        source_instance = next(item for item in frozen['instances']
                               if item['source_id'] == context('t01')['episode'])
        failed_transfer_arguments = {'operator': 'theory_transfer',
            'library_path': preparation['library_path'], 'candidate_id': frozen_id,
            'theory_id': 'shared:t13:failed_frozen_instance', 'mapping': {'role_0': 'x'},
            'parameters': source_instance['parameters'], 'role_signature': roles('x'),
            'output_unit': UNIT, 'context_ref': context('t13'), 'evidence_checks': [counter_check]}
        failed_binding = preparation_agent._execute('run', failed_transfer_arguments, PublicReplayLab())
        if failed_binding['status'] != 'refuted':
            raise AssertionError(failed_binding)
        shared_library = deepcopy(preparation_agent.theory_consolidation)
        shared_failed_theory = deepcopy(preparation_agent.theory_index[failed_binding['theory_id']])
    revision_index, before = next((i, action['details']['before_program'])
        for i, action in enumerate(target['actions']) if action['action'] == 'revise')
    incumbent = _bound_theory('incumbent:t15:before', Program.from_dict(before), None,
        {'method': 'retained_parent_before_revision', 'source_ref': source_ref(source,
         f'episodes[14].actions[{revision_index}].details.before_program')})
    counter_sequence = payload['episodes'][12]['actions'][9]['details']['sequence']
    if (counter_sequence[1] != counter['observation']['features']
            or [{name: row[name] for name in counter_sequence[0]} for row in counter['observation']['history']]
                != counter_sequence[:1]):
        raise AssertionError('The retained counterexample does not match its original public action and step')
    catalog, arms = ProgramCatalog(), {}
    for arm in ('remove', 'keep'):
        lab = PublicReplayLab()
        with tempfile.TemporaryDirectory(prefix=f'hegel-frozen-ablation-{arm}-') as temporary:
            agent = CommonArchiveInquiryAgent(None, temporary, arm='hegel', source_task=False,
                config=InquiryLLMConfig(compute_budget=50000, observation_budget=8,
                                       max_model_calls=0, execution_seconds=30.))
            agent.theory_consolidation = deepcopy(shared_library)
            register_theories(agent, declarations)
            agent.theory_index[failed_binding['theory_id']] = deepcopy(shared_failed_theory)
            old = []
            for episode in source_episodes:
                template, = extract_templates(Program.from_dict(episode['program']), f'old:{episode["task_id"]}')
                for mapping in template.mappings(('u', 'v')):
                    name = mapping['role_0']
                    old.append(_bound_theory(f'old:{episode["task_id"]}:{name}',
                        template.instantiate(mapping), name,
                        {'source_episode': episode['task_id'], 'method': 'existing_AbstractTemplate', 'mapping': mapping}))
            register_theories(agent, [*old, incumbent])
            agent._save(preparation['library_path'], json.dumps(agent.theory_consolidation))
            recalled_failure = agent._execute('run', deepcopy(failed_transfer_arguments), lab)
            if recalled_failure['status'] != 'refuted':
                raise AssertionError('Recalling the same retained target failure must not revive it')
            extra = []
            if arm == 'keep':
                for instance in frozen['instances']:
                    for name in ('u', 'v'):
                        identifier = f'frozen:{instance["source_id"]}:{name}'
                        result = agent._execute('run', {'operator': 'theory_transfer',
                            'library_path': preparation['library_path'], 'candidate_id': frozen_id,
                            'theory_id': identifier, 'mapping': {'role_0': name},
                            'parameters': instance['parameters'], 'role_signature': roles(name),
                            'output_unit': UNIT, 'context_ref': context('t15')}, lab)
                        if result['status'] != 'candidate_registered':
                            raise AssertionError(result)
                        extra.append(next(item for item in resolved_theories(agent) if item['id'] == identifier))
            else:
                del agent.theory_consolidation['candidates'][frozen_id]
                agent._save(preparation['library_path'], json.dumps(agent.theory_consolidation))
            # Refutation belongs to the executed numerical binding, independently
            # of whether its optional abstraction object remains in the library.
            alias = _bound_theory('ordinary_alias:failed_t13',
                Program.from_dict(source_episodes[0]['program']), 'x', {'method': 'ordinary_inline_declaration'})
            alias['context_ref'] = alias['support']['context_ref'] = context('t13')
            changed_parameters = _bound_theory('ordinary_alias:other_t13_parameters',
                Program.from_dict(source_episodes[1]['program']), 'x', {'method': 'ordinary_inline_declaration'})
            changed_parameters['context_ref'] = changed_parameters['support']['context_ref'] = context('t13')
            register_theories(agent, [alias, changed_parameters])
            checked_pool = {entry['id']: entry for entry in resolved_theories(agent)}
            failed_alias = checked_pool[alias['id']]
            if failed_alias.get('compatible') is not False or not failed_alias.get('scope_refutation'):
                raise AssertionError('An ordinary alias escaped an executed target counterexample')
            if (checked_pool['history:t01'].get('compatible') is False
                    or checked_pool[changed_parameters['id']].get('compatible') is False):
                raise AssertionError('A target binding failure was incorrectly extended to its source or other parameters')
            before = deepcopy(agent.costs)
            routed = agent._execute('run', {'operator': 'research_route',
                'research_question': deepcopy(question), 'experiments': deepcopy(panel), 'mission': mission}, lab)
            selection = routed['selection']
            if selection is None:
                raise AssertionError(routed)
            bound = old + [incumbent] + extra
            programs = []
            for theory in bound:
                program = Program.from_dict(theory['program'])
                if program not in programs:
                    programs.append(program)
            # Read labels only after the full plan, score and selection exist.
            evaluation_rows = target['evaluation']['prediction_examples']
            diagnostics = SearchBudget(20000)
            if not diagnostics.spend(len(programs) * len(evaluation_rows)):
                raise AssertionError('Historical prediction matrix exceeded its allocation')
            matrix = {catalog.identifier(program): [program.predict(Observation(
                deepcopy(row['features']), 0., ())) for row in evaluation_rows] for program in programs}
            native = _native_parent_choice(source_episodes, target, extra, catalog)
            selected = selection.get('selected_experiment')
            arms[arm] = {'retained_source_program_count': 2, 'old_template_count': 2,
                'old_template_binding_count': len(old), 'new_frozen_object_retained': arm == 'keep',
                'retained_incumbent_count': 1,
                'new_frozen_binding_count': len(extra), 'raw_candidate_count': len(bound),
                'total_index_count_including_original_sources': len(agent.theory_index),
                'program_count': len(programs), 'candidate_program_ids': [catalog.identifier(program) for program in programs],
                'signature_count': len(selection['groups']),
                'selection': {'status': selection['status'], 'scores': selection['scores'],
                    'signature_predictions': [{'task_answer': group['task_answer'],
                        'panel_predictions': group['panel_predictions']} for group in selection['groups']],
                    'selected_experiment_id': selected['id'] if selected else None,
                    'selected_action': selected['actions'][0] if selected else None,
                    'public_row_validation': selection['public_row_validation'],
                    'prediction_compute': selection['compute_cost'],
                    'actual_common_compute': agent.costs['compute_cost'] - before['compute_cost'],
                    'parent_plan_prepared': routed.get('parent_use', {}).get('status') == 'parent_query_prepared',
                    'selected_action_executed': False, 'action_outcome_known': False},
                'evaluation': {'row_refs': [source_ref(source,
                    f'episodes[14].evaluation.prediction_examples[{i}]') for i in range(4)],
                    'prediction_matrix': matrix, 'labels_used_for_plan_selection': False},
                'matrix_diagnostics': {'row_predictions': len(programs) * len(evaluation_rows),
                    'compute': diagnostics.spent, 'budget': diagnostics.limit},
                'native_parent_choice': native,
                'retained_counterexample': {'source_ref': counter_ref,
                    'history_preserved': counter_check['observation']['history'] == counter['observation']['history'],
                    'matched_public_action_ref': source_ref(source, 'episodes[12].actions[9].details.sequence'),
                    'matched_step': 1, 'history_has_observed_target': '__target__' in counter['observation']['history'][0],
                    'transfer_recall_status': recalled_failure['status'],
                    'transfer_recall_compute': recalled_failure['new_compute_cost'],
                    'executed_checks': recalled_failure['checks'],
                    'ordinary_alias_id': alias['id'], 'ordinary_alias_compatible': failed_alias['compatible'],
                    'ordinary_alias_scope_refutation': failed_alias['scope_refutation'],
                    'source_instance_not_withdrawn': checked_pool['history:t01'].get('compatible') is not False,
                    'other_parameters_not_withdrawn': checked_pool[changed_parameters['id']].get('compatible') is not False,
                    'source_counterexample_checks': frozen['counterexample_checks']},
                'frozen_preparation_compute': preparation['new_compute_cost'] if arm == 'keep' else 0,
                'shared_counterexample_recheck_compute': failed_binding['new_compute_cost'],
                'accounted_compute': {
                    'given_frozen_object_route': agent.costs['compute_cost'] - before['compute_cost'],
                    'old_native_propose': native['old_propose_compute'],
                    'native_prefix_comparison': native['prefix_comparison_compute'],
                    'native_suffix_evaluation': native['suffix_evaluation_compute'],
                    'matrix_diagnostics': diagnostics.spent,
                    'shared_counterexample_recheck': failed_binding['new_compute_cost'],
                    'new_object_construction': preparation['new_compute_cost'] if arm == 'keep' else 0,
                    'total_replay_compute_excluding_saved_source_learning': agent.costs['compute_cost'] - before['compute_cost']
                        + native['old_propose_compute'] + native['prefix_comparison_compute']
                        + native['suffix_evaluation_compute'] + diagnostics.spent
                        + failed_binding['new_compute_cost']
                        + (preparation['new_compute_cost'] if arm == 'keep' else 0)},
                'no_acquisition': {'execute_calls': lab.execute_calls, 'model_calls': agent.costs['model_calls'],
                    'new_observation_cost': agent.costs['observation_cost'],
                    'new_measurement_cost': agent.costs['measurement_cost'], 'common_paid_record_count': len(agent.observations)}}
            if any(arms[arm]['no_acquisition'].values()):
                raise AssertionError('The paired replay must not acquire data or call a model')
    receipt = {'status': 'passed', 'scope': 'same_information_frozen_family_keep_remove_historical_replay',
        'source': source, 'protocol': {'frozen_theme': FAMILY, 'retained_sources_in_both_arms': ['t01', 't06'],
            'retained_incumbent_ref': incumbent['binding_origin']['source_ref'],
            'shared_counterexample_ref': counter_ref, 'target': 't15', 'panel_action_indices': list(PANEL_INDICES),
            'panel_construction': 'first four retained observe actions in original order; step zero readout',
            'controls_used_for_selection': ['u', 'v'], 'labels_used_for_selection': False,
            'compute_budget': 50000, 'hypothetical_observation_budget': 8, 'actual_acquisition_allowed': False,
            'native_fit_allocation': 20000, 'native_fit_prefix': [0, 1], 'historical_suffix': [2, 3]},
        'program_catalog': catalog.to_dict(), 'arms': arms,
        'historical_costs_shared_in_both_arms': {episode['task_id']: {
            'observation_cost': episode['observation_cost'], 'compute_cost': episode['compute_cost'],
            'newly_charged_in_replay': False} for episode in [*source_episodes, payload['episodes'][12], target]},
        'comparison': compare_arm_results(arms['keep'], arms['remove']),
        'incremental_cost': {
            'new_object_construction_compute': preparation['new_compute_cost'],
            'total_keep_minus_remove_compute': arms['keep']['accounted_compute']['total_replay_compute_excluding_saved_source_learning']
                - arms['remove']['accounted_compute']['total_replay_compute_excluding_saved_source_learning'],
            'encoding_gain_bytes': frozen['encoding_cost']['gain_bytes']},
        'executed_in_harness': {
            'object_construction_runs': 1, 'object_construction_compute': preparation['new_compute_cost'],
            'shared_counterexample_recheck_runs': 1, 'shared_counterexample_recheck_compute': failed_binding['new_compute_cost'],
            'arm_route_compute': sum(a['selection']['actual_common_compute'] for a in arms.values()),
            'native_propose_compute': sum(a['native_parent_choice']['old_propose_compute'] for a in arms.values()),
            'native_prefix_comparison_compute': sum(a['native_parent_choice']['prefix_comparison_compute'] for a in arms.values()),
            'native_suffix_evaluation_compute': sum(a['native_parent_choice']['suffix_evaluation_compute'] for a in arms.values()),
            'matrix_diagnostics_compute': sum(a['matrix_diagnostics']['compute'] for a in arms.values()),
            'new_paid_observations': 0, 'new_model_calls': 0},
        'encoding': frozen['encoding_cost'],
        'limitations': ['The experiment and theme are operator-selected historical cases; this is not blind transfer or autonomous adoption.',
            'The selected representative action has public controls but no retained per-step targets; only its pending plan is prepared, so no action benefit is measured.',
            'The native second stage adapts and selects on two saved evaluation examples, then checks the other two; it creates neither paid records nor independent confirmation.',
            't01 and t06 are different measured tasks with shared abstraction lineage.']}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pointer', type=Path, default=POINTER)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    receipt = replay_ablation(args.source_pointer, args.output)
    print(json.dumps({'status': receipt['status'], 'output': str(args.output),
        'comparison': receipt['comparison'], 'selected': receipt['arms']['keep']['selection']['selected_experiment_id']}, ensure_ascii=False))
