#!/usr/bin/env python3
"""Finite historical A/B/C organization replay with a complete old t15 pool.

The two saved development examples and two saved successor examples are fixed
before execution. Public-panel plans are never acquired. This compares existing
mechanisms on retained material; it is not an autonomous discovery experiment.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from hegel_machine.cognition.abstractions import extract_templates
from hegel_machine.cognition.evaluation import initial_prediction, loss, point_loss, predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import Concept, KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.probing import choose_valuable_probe
from hegel_machine.cognition.question_routing import register_theories, resolved_theories
from replay_frozen_theory_ablation import (
    PANEL_INDICES, ProgramCatalog, PublicReplayLab, _bound_theory, _public_action, _readout,
)
from replay_theory_consolidation import (
    FAMILY, POINTER, ROOT, UNIT, context, declaration, load_source, roles, source_ref,
)
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig


OUTPUT = ROOT / 'docs/evidence/three_arm_role_mappings_20261021.json'
COMPUTE_BUDGET, OBSERVATION_BUDGET = 50000, 8
FIT_INDICES, SUFFIX_INDICES = (0, 1), (2, 3)


def new_agent(path):
    return CommonArchiveInquiryAgent(None, path, arm='hegel', source_task=False,
        config=InquiryLLMConfig(compute_budget=COMPUTE_BUDGET,
            observation_budget=OBSERVATION_BUDGET, max_model_calls=0, execution_seconds=30.))


def old_candidates(episodes, target, old, incumbent):
    prefix = [Observation(deepcopy(target['evaluation']['prediction_examples'][i]['features']),
        target['evaluation']['prediction_examples'][i]['target']) for i in FIT_INDICES]
    library = KnowledgeLibrary(concepts=[Concept(f'history:{episode["task_id"]}',
        Program.from_dict(episode['program']), .5, episode['task_id'], 'retained_source')
        for episode in episodes])
    spec = TaskSpec('historical_ablation_t15', 'Same retained public numeric coordinates',
                    ('u', 'v'), (), ())
    work = SearchBudget(20000)
    proposed = library.propose(spec, prefix, work, preserve_parameters=True)
    unique = [{'program': Program.from_dict(incumbent['program']),
               'candidate_id': 'retained_parent_before_revision', 'operation': 'retained_parent'}]
    for candidate in proposed:
        if not any(entry['program'] == candidate.program for entry in unique):
            unique.append({'program': candidate.program, 'candidate_id': candidate.candidate_id,
                           'operation': candidate.operation})
    if any(not any(Program.from_dict(item['program']) == entry['program'] for entry in unique) for item in old):
        raise AssertionError('The original KnowledgeLibrary omitted an existing template binding')
    fallback = initial_prediction(prefix, .5)
    comparison = SearchBudget(20000)
    if not comparison.spend(len(prefix) * len(unique)):
        raise AssertionError('The fixed development comparison did not fit its original allocation')
    scored = [{**entry, 'prefix_mse': loss(entry['program'], prefix, fallback, .5), 'order': index}
              for index, entry in enumerate(unique)]
    selected = min(scored, key=lambda entry: (entry['prefix_mse'], entry['program'].size, entry['order']))
    return {'entries': scored, 'selected': selected, 'prefix': prefix, 'fallback': fallback, 'spec': spec,
            'old_propose_compute': work.spent, 'prefix_comparison_compute': comparison.spent,
            'raw_proposal_count': len(proposed), 'template_count': len(library.abstracts)}


def generic_plan(data, panel, remaining):
    """Run the archived task_value policy and retain its full-rank exact ties."""
    candidates = [Candidate(entry['candidate_id'], entry['program'], entry['operation'])
                  for entry in data['entries']]
    probes = [Probe(item['id'], tuple(deepcopy(item['actions'][0]['sequence']))) for item in panel
              if item['observation_cost'] <= OBSERVATION_BUDGET]
    work = SearchBudget(remaining)
    _, details = choose_valuable_probe(probes, candidates, data['fallback'], work,
        observations=data['prefix'], spec=data['spec'])
    rows = details.get('evaluated_probes', [])
    if not rows:
        raise AssertionError('The original task_value policy did not evaluate the fixed affordable panel')
    key = lambda row: (row['value_per_observation'], row['coverage_per_observation'],
                        row['distinct_control_histories'], -row['observation_cost'])
    best = max(map(key, rows))
    tied = [row['probe_id'] for row in rows if key(row) == best]
    return {'status': 'selected' if len(tied) == 1 else 'tied',
        'selected_experiment_id': tied[0] if len(tied) == 1 else None,
        'tied_experiments': tied if len(tied) > 1 else [],
        'policy': 'original_task_value', 'details': details, 'compute_cost': work.spent,
        'selected_action_executed': False, 'action_outcome_known': False}


def parent_use_ablation(seed, question, panel, *, directory, incumbent):
    """Delete only the actual parent use from two identical historical C scenes.

    Both scenes start before the original C route. Programs, source failures,
    task budget, mission, public query and experiments stay fixed. Selecting a
    panel does not execute it or install a prediction model.
    """
    arms = {}
    original_pool = deepcopy(seed.theory_index)
    cold_start = not any(item.get('frozen_recall_execution') or item.get('routing_execution')
                         for item in seed.questions)
    for name, uses in (('keep', question['use_sites']), ('remove', [])):
        agent, lab = new_agent(Path(directory) / name), PublicReplayLab()
        agent.theory_index = deepcopy(original_pool)
        agent.theory_consolidation = deepcopy(seed.theory_consolidation)
        agent.costs = deepcopy(seed.costs)
        agent.questions = deepcopy(seed.questions)
        agent.research_mission = deepcopy(seed.research_mission)
        agent._save('shared/theory_library.json', json.dumps(agent.theory_consolidation))
        current = deepcopy(question)
        current['use_sites'] = deepcopy(uses)
        parent = next(item for item in agent.questions if item['id'] == question['parent_id'])
        before = deepcopy(parent.get('pending_query'))
        source_before = deepcopy(parent.get('pending_query_source'))
        start = agent.costs['compute_cost']
        result = agent._execute('run', {'operator': 'research_route',
            'research_question': current, 'experiments': deepcopy(panel),
            'mission': {'purpose': 'task', 'mission_id': 'three_arm_t15'}}, lab)
        parent = next(item for item in agent.questions if item['id'] == question['parent_id'])
        pool = {item['id']: item for item in resolved_theories(agent)}
        arms[name] = {'question_parent_id': current['parent_id'], 'use_sites': deepcopy(uses),
            'parent_use': deepcopy(result.get('parent_use')),
            'pending_before': before, 'pending_after': deepcopy(parent.get('pending_query')),
            'pending_source_before': source_before,
            'pending_source_after': deepcopy(parent.get('pending_query_source')),
            'selection': deepcopy(result.get('selection')),
            'automatic_recall': deepcopy(result['frozen_recall']),
            'old_pool_preserved': all(agent.theory_index[key]['program'] == item['program']
                                      for key, item in original_pool.items()),
            'retained_parent_theory_id': incumbent['id'],
            'retained_parent_program_unchanged': agent.theory_index[incumbent['id']]['program'] == incumbent['program'],
            'ordinary_counterexample_alias_blocked': pool['ordinary_alias:failed_t13'].get('compatible') is False,
            'compute_cost': agent.costs['compute_cost'] - start,
            'accounted_total_compute': agent.costs['compute_cost'],
            'same_total_compute_budget': agent.config.compute_budget,
            'new_observations': lab.execute_calls, 'new_model_calls': agent.costs['model_calls'],
            'selected_action_executed': False, 'action_outcome_known': False,
            'subsequent_prediction_effect': 'UNKNOWN_selected_panel_not_executed'}
    keep, remove = arms['keep'], arms['remove']
    same_selection = keep['selection'] == remove['selection']
    same_recall = keep['automatic_recall'] == remove['automatic_recall']
    shared_results = {}
    for field in ('selection', 'automatic_recall'):
        if keep[field] == remove[field]:
            shared_results[field] = keep[field]
            for arm in arms.values():
                arm.pop(field)
                arm[field + '_ref'] = {'receipt_field': ['parent_use_ablation', 'shared_results', field]}
    return {'scope': 'same_material_parent_use_deletion',
        'protocol': {'intervention': {'field': 'question.use_sites', 'operation': 'delete_all'},
            'parent_id_is_retained': True, 'same_task_mission': {'purpose': 'task', 'mission_id': 'three_arm_t15'},
            'cold_start_without_completed_routing_or_recall': cold_start,
            'shared_setup_accounted_compute': seed.costs['compute_cost'],
            'common_pool_ref': {'receipt_field': ['arms', 'C', 'candidate_scores']},
            'common_counterexample_ref': {'receipt_field': ['arms', 'C', 'counterexample']},
            'common_public_panel_ref': {'receipt_field': ['protocol', 'public_panel_action_indices']},
            'same_queries_panel_budget_and_starting_state': True},
        'arms': arms, 'shared_results': shared_results, 'comparison': {
            'same_selection': same_selection,
            'same_recalled_bindings': same_recall,
            'same_compute': keep['compute_cost'] == remove['compute_cost'],
            'keep_changes_parent_pending': keep['pending_after'] != keep['pending_before'],
            'remove_preserves_parent_pending': (remove['pending_after'] == remove['pending_before']
                and remove['pending_source_after'] == remove['pending_source_before'])},
        'actual_harness_compute': sum(arm['compute_cost'] for arm in arms.values()),
        'limitations': ['Only the selected experiment is prepared; it is not executed.',
            'No model adoption or downstream prediction benefit is inferred from writing a pending query.']}


def replay_three_arms(pointer=POINTER, output=OUTPUT):
    payload, source = load_source(Path(pointer))
    positions, target_index = (0, 5), 14
    episodes, target = [payload['episodes'][i] for i in positions], payload['episodes'][target_index]
    if [episode['task_id'] for episode in episodes] != ['t01', 't06'] or target['task_id'] != 't15':
        raise AssertionError('The fixed source and successor identities changed')
    if payload['config']['experiment_policy'] != 'task_value':
        raise AssertionError('The retained ordinary experiment policy changed')
    counter_index, counter = next((i, item) for i, item in enumerate(
        payload['knowledge']['concepts'][0]['counterexamples']) if item['task_id'] == 't13')
    counter_ref = source_ref(source, f'knowledge.concepts[0].counterexamples[{counter_index}]')
    counter_check = {'source_ref': counter_ref, 'context_ref': context('t13'),
        'observation': deepcopy(counter['observation']), 'tolerance': 1e-6}
    declarations = [declaration(source, episode, i) for i, episode in zip(positions, episodes)]
    revision_index, before = next((i, action['details']['before_program'])
        for i, action in enumerate(target['actions']) if action['action'] == 'revise')
    incumbent_ref = source_ref(source, f'episodes[14].actions[{revision_index}].details.before_program')
    incumbent = _bound_theory('incumbent:t15:before', Program.from_dict(before), None,
                             {'method': 'retained_parent_before_revision', 'source_ref': incumbent_ref})
    old = []
    for episode in episodes:
        template, = extract_templates(Program.from_dict(episode['program']), f'old:{episode["task_id"]}')
        for mapping in template.mappings(('u', 'v')):
            name = mapping['role_0']
            old.append(_bound_theory(f'old:{episode["task_id"]}:{name}', template.instantiate(mapping), name,
                {'method': 'existing_AbstractTemplate', 'source_episode': episode['task_id'], 'mapping': mapping}))
    panel = [{'id': target['actions'][i]['details']['probe_id'], 'actions': [_public_action(target, i)],
        'query': _readout(_public_action(target, i)), 'tolerance': 1e-6,
        'observation_cost': len(_public_action(target, i)['sequence'])} for i in PANEL_INDICES]
    typed_target = {'role_signature': {**roles('u'), **roles('v')},
                    'output_unit': UNIT, 'context_ref': context('t15')}
    question = {'id': 'three_arm_t15_public_response', 'parent_id': 'task', 'depends_on': [],
        'question': 'Which retained alternative applies at this public successor input?',
        'kind': 'value', 'answer_type': 'number', 'context_ref': context('t15'),
        'public_role_signature': deepcopy(typed_target['role_signature']),
        'theory_target': typed_target, 'answer_resolution_ref': {'atol': 1e-6},
        'query_semantics': {'mode': 'model_query', 'query_expression': deepcopy(panel[-1]['query'])},
        'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'task'}]}
    mission = {'purpose': 'theory_learning', 'mission_id': 'three_arm_retained_sources',
        'theory_learning_allocation': {'compute': 30000, 'author_calls': 0, 'observations': 0}}
    catalog, arms = ProgramCatalog(), {}
    with tempfile.TemporaryDirectory(prefix='hegel-three-arm-routing-') as temporary:
        construction = new_agent(Path(temporary) / 'construction')
        preparation = construction._execute('run', {'operator': 'theory_consolidation', 'theme': 'ordered_affine',
            'theories': declarations, 'source_theory_ids': [item['id'] for item in declarations],
            'episode_sources': [source_ref(source, f'episodes[{i}]') for i in positions],
            'counterexamples': {'history:t01': [counter_check]},
            'materials': [{'id': 'shared_t13_counterexample', 'kind': 'error', 'source_ref': counter_ref}],
            'mission': deepcopy(mission), 'allocation': {'compute': 30000, 'max_groups': 4}}, PublicReplayLab())
        # Preserve one real numeric failure in all arms; its optional family is
        # retained only in C. Selecting this check never selects a successor mapping.
        frozen_id = preparation['frozen_candidates'][0]['candidate_id']
        frozen = construction.theory_consolidation['candidates'][frozen_id]
        source_instance = next(item for item in frozen['instances'] if item['source_id'] == context('t01')['episode'])
        failed = construction._execute('run', {'operator': 'theory_transfer',
            'candidate_id': frozen_id, 'theory_id': 'shared:t13:failed_instance',
            'mapping': {'role_0': 'x'}, 'parameters': source_instance['parameters'],
            'role_signature': roles('x'), 'output_unit': UNIT, 'context_ref': context('t13'),
            'evidence_checks': [counter_check]}, PublicReplayLab())
        if failed['status'] != 'refuted':
            raise AssertionError('The original measured t13 failure was lost')
        for arm in ('A', 'B', 'C'):
            data = old_candidates(episodes, target, old, incumbent)
            shared_compute = data['old_propose_compute'] + data['prefix_comparison_compute'] + failed['new_compute_cost']
            agent, lab = new_agent(Path(temporary) / arm), PublicReplayLab()
            agent.theory_consolidation = deepcopy(construction.theory_consolidation)
            if arm != 'C':
                agent.theory_consolidation['candidates'] = {}
            register_theories(agent, declarations)
            agent.theory_index[failed['theory_id']] = deepcopy(construction.theory_index[failed['theory_id']])
            concrete = []
            for index, entry in enumerate(data['entries']):
                names = entry['program'].required_features()
                declared = _bound_theory(f'native:{index:02d}', entry['program'], None, {'method': entry['operation']})
                declared['role_signature'] = {name: roles(name)[name] for name in names}
                declared['parameter_paths'] = []
                concrete.append(declared)
            register_theories(agent, [*old, incumbent, *concrete])
            alias = _bound_theory('ordinary_alias:failed_t13', Program.from_dict(episodes[0]['program']), 'x',
                                   {'method': 'ordinary_inline_declaration'})
            alias['context_ref'] = alias['support']['context_ref'] = context('t13')
            register_theories(agent, [alias])
            pool = {item['id']: item for item in resolved_theories(agent)}
            if pool[alias['id']].get('compatible') is not False or not pool[alias['id']].get('scope_refutation'):
                raise AssertionError('An alias escaped the shared actual counterexample')
            agent.costs['compute_cost'] = shared_compute + (preparation['new_compute_cost'] if arm == 'C' else 0)
            agent._save(preparation['library_path'], json.dumps(agent.theory_consolidation))
            if arm == 'C':
                parent_ablation = parent_use_ablation(agent, question, panel,
                    directory=Path(temporary) / 'parent_use_ablation', incumbent=incumbent)
            before_route = agent.costs['compute_cost']
            if arm == 'A':
                plan = generic_plan(data, panel, COMPUTE_BUDGET - before_route)
                agent.costs['compute_cost'] += plan['compute_cost']
                recall = {'status': 'not_part_of_generic_archive', 'generated_theory_ids': [], 'new_compute_cost': 0}
            else:
                routed = agent._execute('run', {'operator': 'research_route',
                    'research_question': deepcopy(question), 'experiments': deepcopy(panel),
                    'mission': {'purpose': 'task', 'mission_id': 'three_arm_t15'}}, lab)
                selected = routed.get('selection')
                if selected is None:
                    raise AssertionError(routed)
                plan = {key: deepcopy(selected.get(key)) for key in ('status', 'scores', 'groups',
                    'tied_experiments', 'public_row_validation', 'compute_cost', 'effective_pairs')}
                plan['selected_experiment_id'] = (selected.get('selected_experiment') or {}).get('id')
                plan.update(policy='PR2_question_selection', selected_action_executed=False, action_outcome_known=False)
                recall = deepcopy(routed['frozen_recall'])
            route_compute = agent.costs['compute_cost'] - before_route
            recalled_comparison_compute = 0
            recalled_program_ids = []
            current = {item['id']: item for item in resolved_theories(agent)}
            for identifier in recall['generated_theory_ids']:
                program = Program.from_dict(current[identifier]['program'])
                if any(entry['program'] == program for entry in data['entries']):
                    continue
                if agent.costs['compute_cost'] + len(data['prefix']) > COMPUTE_BUDGET:
                    raise AssertionError('A recalled candidate could not receive the matched development comparison')
                agent.costs['compute_cost'] += len(data['prefix'])
                recalled_comparison_compute += len(data['prefix'])
                data['entries'].append({'program': program, 'candidate_id': identifier,
                    'operation': 'automatic_frozen_recall', 'order': len(data['entries']),
                    'prefix_mse': loss(program, data['prefix'], data['fallback'], .5)})
                recalled_program_ids.append(catalog.identifier(program))
            # No successor label is used until proposals, fit scores and public
            # plans are fixed. The selected parent uses the original native rule.
            chosen = min(data['entries'], key=lambda entry: (entry['prefix_mse'], entry['program'].size, entry['order']))
            successor = [Observation(deepcopy(target['evaluation']['prediction_examples'][i]['features']),
                target['evaluation']['prediction_examples'][i]['target']) for i in SUFFIX_INDICES]
            suffix_predictions = [predict(chosen['program'], row, data['fallback']) for row in successor]
            agent.costs['compute_cost'] += len(successor)
            total = agent.costs['compute_cost']
            if total > COMPUTE_BUDGET or lab.execute_calls or agent.costs['model_calls'] or agent.observations:
                raise AssertionError('The matched historical replay exceeded its budget or acquired data')
            arms[arm] = {'system': {'A': 'strong_executable_archive', 'B': 'fixed_typed_library_and_PR2',
                                   'C': 'PR2_and_cross_episode_induction'}[arm],
                'retained_source_program_count': len(episodes), 'old_template_count': data['template_count'],
                'old_template_binding_count': len(old), 'initial_parent_retained': True,
                'raw_proposal_count': data['raw_proposal_count'],
                'candidate_program_ids': [catalog.identifier(entry['program']) for entry in data['entries']],
                'new_recalled_candidate_program_ids': recalled_program_ids,
                'candidate_scores': [{'program_id': catalog.identifier(entry['program']),
                    'operation': entry['operation'], 'prefix_mse': entry['prefix_mse'], 'order': entry['order']}
                    for entry in data['entries']], 'selection': plan, 'automatic_recall': recall,
                'frozen_object_count': len(agent.theory_consolidation['candidates']),
                'counterexample': {'source_ref': counter_ref, 'status': failed['status'],
                    'checks': deepcopy(failed['checks']), 'ordinary_alias_compatible': pool[alias['id']]['compatible'],
                    'source_instance_preserved': pool['history:t01'].get('compatible') is not False},
                'parent': {'program_id': catalog.identifier(chosen['program']), 'prefix_mse': chosen['prefix_mse'],
                    'selection_rule': 'strict minimum prefix MSE, then Program size, then original stable candidate order',
                    'suffix_predictions': suffix_predictions,
                    'suffix_mse': sum(point_loss(value, row.target, .5)
                        for value, row in zip(suffix_predictions, successor)) / len(successor),
                    'successor_rows': len(successor), 'independent_confirmation': False},
                'costs': {'old_native_propose': data['old_propose_compute'],
                    'native_prefix_comparison': data['prefix_comparison_compute'],
                    'shared_counterexample_recheck': failed['new_compute_cost'],
                    'new_object_construction': preparation['new_compute_cost'] if arm == 'C' else 0,
                    'routing_including_recall': route_compute, 'native_suffix_evaluation': len(successor),
                    'recalled_prefix_comparison': recalled_comparison_compute,
                    'total': total, 'remaining': COMPUTE_BUDGET - total},
                'new_observations': 0, 'new_model_calls': 0, 'task_completion': 'not_established',
                'action_benefit': 'UNKNOWN_no_retained_panel_labels_or_action_execution'}
    receipt = {'status': 'completed', 'scope': 'finite_historical_three_arm_mechanism_comparison',
        'source': source, 'protocol': {'source_episode_indices': list(positions), 'target_episode_index': target_index,
            'preceding_comparison_ref': {'file': 'docs/evidence/three_arm_historical_routing_20261021.json'},
            'recall_method_ref': 'hegel_machine.cognition.abstractions.AbstractTemplate.mappings',
            'shared_source_program_refs': [item['program_source_ref'] for item in declarations],
            'shared_source_outcome_refs': [source_ref(source, f'episodes[{i}].evaluation') for i in positions],
            'retained_incumbent_ref': incumbent_ref, 'shared_counterexample_ref': counter_ref,
            'public_panel_action_indices': list(PANEL_INDICES), 'public_panel_readout_step': 0,
            'same_total_compute_budget': COMPUTE_BUDGET, 'same_observation_budget': OBSERVATION_BUDGET,
            'native_propose_allocation': 20000, 'native_comparison_allocation': 20000,
            'development_indices': list(FIT_INDICES), 'successor_indices': list(SUFFIX_INDICES),
            'evaluation_row_refs': [source_ref(source, f'episodes[14].evaluation.prediction_examples[{i}]') for i in range(4)],
            'full_typed_target': typed_target, 'successor_labels_used_for_fitting_or_plan_selection': False,
            'candidate_id_supplied_to_successor_recall': False, 'exact_panel_ties_retained': True},
        'program_catalog': catalog.to_dict(), 'arms': arms, 'parent_use_ablation': parent_ablation,
        'source_construction': {'status': preparation['status'], 'theme': FAMILY,
            'compute_cost': preparation['new_compute_cost'], 'frozen_candidates': preparation['frozen_candidates'],
            'source_reconstruction_preserved': all(check['exact'] for candidate in
                construction.theory_consolidation['candidates'].values() for check in candidate['source_reconstruction'])},
        'comparison': {'same_old_candidate_programs': arms['A']['candidate_program_ids'] == arms['B']['candidate_program_ids']
            == arms['C']['candidate_program_ids'],
            'same_native_development_scores': arms['A']['candidate_scores'] == arms['B']['candidate_scores']
                == arms['C']['candidate_scores'],
            'same_parent_program': len({arm['parent']['program_id'] for arm in arms.values()}) == 1,
            'same_successor_predictions': arms['A']['parent']['suffix_predictions'] == arms['B']['parent']['suffix_predictions']
                == arms['C']['parent']['suffix_predictions'],
            'C_minus_B_compute': arms['C']['costs']['total'] - arms['B']['costs']['total']},
        'historical_costs_shared': {episode['task_id']: {key: episode[key] for key in ('observation_cost', 'compute_cost')}
            for episode in [*episodes, payload['episodes'][12], target]},
        'actual_harness_compute': preparation['new_compute_cost'] + failed['new_compute_cost']
            + sum(arm['costs']['old_native_propose'] + arm['costs']['native_prefix_comparison']
                  + arm['costs']['routing_including_recall'] + arm['costs']['native_suffix_evaluation']
                  + arm['costs']['recalled_prefix_comparison'] for arm in arms.values())
            + parent_ablation['actual_harness_compute'],
        'limitations': ['Operator-selected source theme and successor; finite mechanism evidence, not autonomous research benefit.',
            'Only four retained evaluation examples: two development and two historical successor checks; no independent confirmation.',
            'All legal u/v bindings are retained; source reconstructions do not establish which binding applies.',
            'Public panel outcomes are not retained; plans are not executed and action benefit is UNKNOWN.']}
    Path(output).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


def replay_registered_groups(pointer=POINTER, retained=OUTPUT,
                             frozen_directory=ROOT / 'docs/evidence/frozen_theory_state_20261021'):
    """Check formal dispatch on the complete saved pool, without scoring gains."""
    from scripts.run_anonymous_revision_inquiry import ORGANIZATION_ARMS, protocol, run_study

    saved = json.loads(Path(retained).read_text())
    payload, origin = load_source(Path(pointer))
    target = payload['episodes'][14]
    assert target['task_id'] == 't15'
    old_ids = saved['arms']['B']['candidate_program_ids']
    declarations = []
    for identifier in old_ids:
        program = Program.from_dict(saved['program_catalog'][identifier])
        declared = _bound_theory(f'retained_old:{identifier}', program, None,
            {'method': 'complete_retained_old_pool',
             'source_ref': {'file': str(Path(retained).resolve()), 'json_path': ['program_catalog', identifier]}})
        declared['role_signature'] = {name: roles(name)[name] for name in program.required_features()}
        declared['parameter_paths'] = []
        declarations.append(declared)
    panel = [{'id': target['actions'][i]['details']['probe_id'], 'actions': [_public_action(target, i)],
        'query': _readout(_public_action(target, i)), 'tolerance': 1e-6,
        'observation_cost': len(_public_action(target, i)['sequence'])} for i in PANEL_INDICES]
    question = {'id': 'registered_retained_t15', 'parent_id': 'task', 'depends_on': [],
        'question': 'Which retained alternative applies at this public successor input?',
        'kind': 'value', 'answer_type': 'number', 'context_ref': context('t15'),
        'answer_resolution_ref': {'atol': 1e-6},
        'theory_target': {'role_signature': {**roles('u'), **roles('v')},
                         'output_unit': UNIT, 'context_ref': context('t15')},
        'query_semantics': {'mode': 'model_query', 'query_expression': deepcopy(panel[-1]['query'])},
        'resolution_experiments': panel,
        'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'task'}]}
    frozen = Path(frozen_directory).resolve()
    knowledge = {'files': {}, 'theory_index': {'file': str(frozen / 'index.json'),
                                             'json_path': ['theory_index']}}
    plan = protocol(kinds=('retained_t15',), arms=ORGANIZATION_ARMS,
        config=InquiryLLMConfig(compute_budget=COMPUTE_BUDGET, observation_budget=OBSERVATION_BUDGET,
            confirmation_rows=0, max_model_calls=0, token_budget=0, execution_seconds=30.),
        research_organization={'phase': 'target', 'theories': declarations,
                               'pr3_library_path': str(frozen / 'library.json')})
    agents, initial = {}, {}

    def prepare(agent, laboratory, case):
        agents[agent.organization_group] = agent
        initial[agent.organization_group] = {'knowledge': deepcopy(agent.knowledge),
            'theory_index': deepcopy(agent.theory_index), 'config': deepcopy(agent.config),
            'interface': agent.common_interface, 'questions': deepcopy(agent.questions)}
        agent._execute('write', {'path': 'shared/retained_question.txt',
            'code': question['question'], 'research_question': deepcopy(question)}, laboratory)
        return {}

    with tempfile.TemporaryDirectory(prefix='hegel-formal-retained-dispatch-') as temporary:
        summary = run_study(plan, temporary, provider_factory=lambda _: None, knowledge=knowledge,
            prepare_prefix=prepare, laboratory_factory=lambda _: PublicReplayLab())
        assert initial['archive'] == initial['fixed_pr2'] == initial['pr3']
        arms = {}
        for name, agent in agents.items():
            pool = {item['id'] for item in resolved_theories(agent)}
            assert all(item['id'] in pool for item in declarations)
            assert agent.model is None and agent.observations == []
            assert agent.costs['model_calls'] == agent.costs['observation_cost'] == 0
            route = agent._organization_state.get('routing')
            feedback = agent.events[route['event_index']]['feedback'] if route else {}
            recalls = feedback.get('frozen_recall', {}).get('generated_theory_ids', [])
            arms[name] = {'all_old_candidates_preserved': True,
                'automatic_route_status': feedback.get('status'),
                'automatic_recalled_ids': recalls,
                'selection': (feedback.get('selection') or {}).get('status'),
                'selected_experiment': ((feedback.get('selection') or {}).get('selected_experiment') or {}).get('id'),
                'parent_pending_prepared': any(q.get('pending_query') for q in agent.questions),
                'compute_cost': agent.costs['compute_cost'],
                'source_failed_instance_retained': agent.theory_index[
                    'history:t13:failed_frozen_instance']['compatible'] is False}
        assert arms['archive']['automatic_route_status'] is None
        assert arms['fixed_pr2']['automatic_recalled_ids'] == []
        assert arms['pr3']['automatic_recalled_ids']
        assert summary['evidence_kind'] == 'scripted_interface_check'
        return {'status': 'passed', 'scope': 'formal_dispatch_on_retained_material_not_gain_comparison',
            'entrypoint': 'scripts.run_anonymous_revision_inquiry.run_study',
            'old_pool_ref': {'file': str(Path(retained).resolve()),
                             'json_path': ['arms', 'B', 'candidate_program_ids']},
            'public_panel_source': origin, 'public_panel_action_indices': list(PANEL_INDICES),
            'frozen_library_ref': {'file': str(frozen / 'library.json'), 'json_path': []},
            'same_initial_knowledge_index_budget_interface_and_task': True,
            'registered_trials_retained': len(summary['trials']), 'arms': arms,
            'phase': 'target', 'source_construction_cost_ref': {
                'file': str(ROOT / 'docs/evidence/research_effectiveness_boundary_20261021.json'),
                'json_path': ['frozen_theory_installation', 'source_construction_compute']},
            'new_provider_calls': 0, 'new_scientific_acquisitions': 0,
            'new_predictor_adoption_or_successor_evaluation': False,
            'autonomous_discovery_or_independent_gain': 'not_established'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pointer', type=Path, default=POINTER)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = replay_three_arms(args.source_pointer, args.output)
    print(json.dumps({'status': result['status'], 'comparison': result['comparison'],
        'arms': {name: {'selection': arm['selection']['status'], 'suffix_mse': arm['parent']['suffix_mse'],
            'compute': arm['costs']['total']} for name, arm in result['arms'].items()}}, ensure_ascii=False))
