#!/usr/bin/env python3
"""Replay automatically learned revision relations against complete old search.

The two arms keep the same archive, source counterevidence, target incumbent,
old fitting/composition/startup routines and previously paid prefix labels.
No target continuation label is available until both predictions are frozen.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from hegel_machine.cognition.abstractions import extract_templates
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.evaluation import initial_prediction, loss, predict, point_loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import Concept, KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.question_routing import register_theories, resolved_theories
from hegel_machine.cognition.selection import rank_candidates, retain_history_frontier, warm_start_candidates
from hegel_machine.cognition.sparse_search import sparse_candidates
from hegel_machine.cognition.synthesis import OPERATIONS, synthesize
from replay_theory_consolidation import POINTER, ROOT, UNIT, load_source, source_ref


HISTORICAL_OUTPUT = ROOT / 'docs/evidence/cross_skeleton_relations_replay_20261020.json'
OUTPUT = ROOT / 'docs/evidence/cross_skeleton_relations_replay_20261021_parent_scope.json'
CONTEXT = {'episode': 'cognition_v3_sequence_20260908/t07',
           'purpose': 'retained_program_prediction', 'output_unit': UNIT}
ROLES = {name: {'role': f'original_public_control:{name}',
               'unit': f'original_public_numeric_coordinate:{name}', 'time': 'current'}
         for name in ('x', 'z')}
TOTAL_COMPUTE = 150000
PANEL_ACTIONS = (0, 1, 2, 14, 40, 65)
RELATION_FAMILY = 'retained_program_revision:history_coverage'


class ReplayLab:
    def __init__(self):
        self.execute_calls = 0

    def execute(self, action):
        self.execute_calls += 1
        raise AssertionError('A retained relation replay must not acquire observations')

    def public_spec(self):
        return {'controls': [{'name': name} for name in ROLES]}

    def action_cost(self, action):
        return len(action['sequence'])

    def prediction_rows(self, action):
        return [Observation(deepcopy(point), 0., tuple(deepcopy(action['sequence'][:i])))
                for i, point in enumerate(action['sequence'])]


class Catalog:
    def __init__(self):
        self.programs = []

    def add(self, program):
        if program not in self.programs:
            self.programs.append(program)
        return f'program_{self.programs.index(program):04d}'

    def export(self):
        return {self.add(program): program.to_dict() for program in self.programs}


def recovered_prefix(episode, memory_index, action_index):
    """Recover only labels literally retained in one real observation history."""
    original = episode['memory_events'][memory_index]['observation']
    sequence = episode['actions'][action_index]['details']['sequence']
    names = set(original['features'])
    public_history = [{name: row[name] for name in names} for row in original['history']]
    if (sequence[:len(public_history)] != public_history
            or sequence[len(public_history)] != original['features']):
        raise AssertionError('The retained observation is not this actual action prefix')
    rows = [Observation({name: row[name] for name in names}, row['__target__'],
                        tuple(deepcopy(original['history'][:i])))
            for i, row in enumerate(original['history'])]
    rows.append(Observation(deepcopy(original['features']), original['target'],
                            tuple(deepcopy(original['history']))))
    return rows


def old_baseline(payload, source):
    """Use actual old constructors, including rejected archive Programs."""
    episodes, archive = payload['episodes'], []
    for position, episode in enumerate(episodes[:7]):
        cutoff = 44 if position == 6 else len(episode['actions']) - 1
        for index, action in enumerate(episode['actions'][:cutoff + 1]):
            for field in ('before_program', 'after_program'):
                if field not in action.get('details', {}):
                    continue
                program = Program.from_dict(action['details'][field])
                if not any(entry['program'] == program for entry in archive):
                    archive.append({'program': program,
                        'source_ref': source_ref(source, f'episodes[{position}].actions[{index}].details.{field}'),
                        'risk_quantile': episode['evaluation']['risk_quantile']})
        if position < 6:
            program = Program.from_dict(episode['program'])
            if not any(entry['program'] == program for entry in archive):
                archive.append({'program': program, 'source_ref': source_ref(source, f'episodes[{position}].program'),
                                'risk_quantile': episode['evaluation']['risk_quantile']})
    prefix = [*recovered_prefix(episodes[6], 2, 14), *recovered_prefix(episodes[6], 3, 40)]
    concepts = [Concept(f'archive_{i:03d}', entry['program'], entry['risk_quantile'],
                        f'archive:{i}', 'retained_historical_revision') for i, entry in enumerate(archive)]
    library = KnowledgeLibrary(concepts=concepts)
    spec = TaskSpec('t07_retained_prefix', 'Only the original retained public numeric inputs',
                    ('x', 'z'), (), ())
    incumbent = Program.from_dict(episodes[6]['actions'][44]['details']['after_program'])
    raw = [Candidate(f'archive:{i}', entry['program'], 'retained_archive')
           for i, entry in enumerate(archive)]
    budget = SearchBudget(12000)
    proposed = library.propose(spec, prefix, budget, preserve_parameters=True)
    proposal_compute = budget.spent
    synthesis_budget = SearchBudget(10000)
    composed = synthesize(prefix, operations=OPERATIONS, budget=synthesis_budget,
                          incumbent=incumbent, max_history=4,
                          seed_programs=tuple(entry['program'] for entry in archive))
    sparse_budget = SearchBudget(10000)
    sparse = sparse_candidates(prefix, budget=sparse_budget, max_history=4,
                               strategy='beam', additional_terms=library.callable_atoms(spec))
    all_old = [*raw, *proposed, *composed, *sparse]
    warm_budget = SearchBudget(12000)
    starters = warm_start_candidates(all_old, prefix, budget=warm_budget,
                                     q=.5, complexity_weight=.001)
    all_old.extend(starters)
    unique = []
    for candidate in all_old:
        if not any(entry['candidate'].program == candidate.program for entry in unique):
            unique.append({'candidate': candidate, 'source': 'complete_old_baseline'})
    return {'archive': archive, 'prefix': prefix, 'candidates': unique,
        'incumbent': incumbent, 'fallback': episodes[6]['fallback'],
        'costs': {'old_propose': proposal_compute, 'old_composition': synthesis_budget.spent,
                  'old_sparse': sparse_budget.spent, 'old_warm_start': warm_budget.spent},
        'capabilities': {'archive_programs': len(archive), 'extract_template_count': len(library.abstracts),
            'old_propose_candidates': len(proposed), 'old_composition_candidates': len(composed),
            'old_sparse_candidates': len(sparse), 'old_warm_start_candidates': len(starters),
            'raw_candidates': len(all_old), 'unique_candidates': len(unique)}}


def declaration(identifier, program, *, reference=None, context=CONTEXT):
    signatures = deepcopy(ROLES)
    if any(name == '__target__' for name, _ in program.observed_variables()):
        signatures['__target__'] = {'role': 'already_observed_previous_response',
            'unit': UNIT, 'time': 'historical'}
    return {'id': identifier, 'program': program.to_dict(), 'role_signature': signatures,
        'output_unit': UNIT, 'context_ref': deepcopy(context),
        'program_source_ref': deepcopy(reference),
        'support': {'status': 'candidate', 'evidence': [], 'context_ref': deepcopy(context)}}


def source_declarations(payload, source):
    result = []
    for position, action_index in ((1, 18), (2, 34)):
        episode = payload['episodes'][position]
        ref = source_ref(source, f'episodes[{position}].actions[{action_index}].details.after_program')
        context = {**CONTEXT, 'episode': f'cognition_v3_sequence_20260908/{episode["task_id"]}'}
        item = declaration(f'source:{position}', Program.from_dict(
            episode['actions'][action_index]['details']['after_program']), reference=ref, context=context)
        item.update(context_family_ref=RELATION_FAMILY,
            episode_ref={'question_id': episode['task_id'], 'meaning_version': 1,
                         'source_ref': source_ref(source, f'episodes[{position}]')})
        result.append(item)
    return result


def public_action(target, index):
    return {'kind': 'trajectory', 'sequence': deepcopy(target['actions'][index]['details']['sequence'])}


def public_readout(action):
    return {'op': 'Readout', 'record_index': 0, 'step': 1, 'target': 'response',
        'context_ref': CONTEXT, 'units': UNIT, 'binding': {'action': deepcopy(action),
            'features': deepcopy(action['sequence'][1]), 'history': deepcopy(action['sequence'][:1])}}


def plan_inputs(target, identifiers):
    panel = [{'id': target['actions'][i]['details']['probe_id'],
        'actions': [public_action(target, i)], 'query': public_readout(public_action(target, i)),
        'tolerance': 1e-6, 'observation_cost': len(public_action(target, i)['sequence'])}
        for i in PANEL_ACTIONS]
    question = {'id': 't07_retained_history_coverage', 'parent_id': 'task', 'depends_on': [],
        'question': 'Which retained alternative applies at the next public history-dependent input?',
        'kind': 'value', 'answer_type': 'number', 'context_ref': deepcopy(CONTEXT),
        'candidate_theory_ids': list(identifiers), 'public_role_signature': deepcopy(ROLES),
        'answer_resolution_ref': {'atol': 1e-6},
        'query_semantics': {'mode': 'model_query',
                            'query_expression': public_readout(public_action(target, 65))},
        'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'task'}]}
    return question, panel


def new_agent(workspace):
    return CommonArchiveInquiryAgent(None, workspace, arm='hegel', source_task=False,
        config=InquiryLLMConfig(compute_budget=TOTAL_COMPUTE, observation_budget=8,
            max_model_calls=0, execution_seconds=30.))


def _row_dict(row):
    return {'features': deepcopy(row.features), 'target': row.target, 'history': list(deepcopy(row.history))}


def _prefix_ref(source, rows, index):
    memory_index, offset = (2, index) if index < 3 else (3, index - 3)
    original_count = 2 if memory_index == 2 else 4
    field = (f'episodes[6].memory_events[{memory_index}].observation'
             + (f'.history[{offset}]' if offset < original_count else ''))
    return source_ref(source, field)


def _selection_summary(result):
    selected = result.get('selection') or {}
    experiment = selected.get('selected_experiment')
    return {'status': result['status'], 'selected_experiment_id': experiment['id'] if experiment else None,
        'selected_action': experiment['actions'][0] if experiment else None,
        'signature_count': len(selected.get('groups', [])),
        'signature_predictions': [{key: deepcopy(group.get(key)) for key in ('task_answer', 'panel_predictions')}
                                  for group in selected.get('groups', [])],
        'scores': selected.get('scores', []), 'excluded': selected.get('excluded', []),
        'public_row_validation': selected.get('public_row_validation'),
        'matching_status': result.get('matching', {}).get('status'),
        'compute': result.get('new_compute_cost', 0),
        'parent_plan_prepared': result.get('parent_use', {}).get('status') == 'parent_query_prepared',
        'selected_action_executed': False}


def compare_arms(keep, remove):
    comparisons = {'same_effective_programs': keep['admitted_program_ids'] == remove['admitted_program_ids'],
        'same_signature_predictions': keep['plan']['signature_predictions'] == remove['plan']['signature_predictions'],
        'same_scores': keep['plan']['scores'] == remove['plan']['scores'],
        'same_selected_action': keep['plan']['selected_action'] == remove['plan']['selected_action'],
        'same_parent_program': keep['parent']['program_id'] == remove['parent']['program_id'],
        'same_continuation_prediction': keep['parent']['continuation_prediction'] == remove['parent']['continuation_prediction']}
    return {**comparisons, 'incremental_behavior_effect_observed': not all(comparisons.values()),
        'continuation_loss_keep_minus_remove': keep['parent']['continuation_loss'] - remove['parent']['continuation_loss'],
        'compute_keep_minus_remove': keep['compute']['total'] - remove['compute']['total']}


def retained_parent_selection(ranked, incumbent, frozen_config):
    """Describe screening while retaining the parent without a fresh comparison.

    This replay stops before controller's prospective comparison and adoption
    rules. The already fitted prefix cannot serve as that fresh comparison.
    """
    if frozen_config['selection_policy'] != 'prospective':
        raise ValueError('This retained replay uses the original prospective policy')
    current = next(row for row in ranked if row.candidate.program == incumbent)
    winner = ranked[0]
    gain = current.score - winner.score
    return current, winner, {
        'status': 'deferred_missing_fresh_comparison', 'action': 'keep_current_parent',
        'selection_policy': frozen_config['selection_policy'],
        'complexity_weight': frozen_config['complexity_weight'],
        'minimum_improvement': frozen_config['minimum_improvement'],
        'current_candidate_id': current.candidate.candidate_id,
        'ranking_winner_candidate_id': winner.candidate.candidate_id,
        'current_score': current.score, 'ranking_winner_score': winner.score,
        'ranking_gain': gain,
        'ranking_gain_exceeds_minimum': gain > frozen_config['minimum_improvement'],
        'fresh_comparison_rows': 0, 'controller_adoption_rules_executed': False}


def replay_relations(pointer=POINTER, output=OUTPUT):
    payload, source = load_source(pointer)
    baseline, target = old_baseline(payload, source), payload['episodes'][6]
    prefix = baseline['prefix']
    mission = {'purpose': 'theory_learning', 'mission_id': 'cross_skeleton_retained_revision_replay',
        'theory_learning_allocation': {'compute': TOTAL_COMPUTE, 'author_calls': 0, 'observations': 0}}
    sources = source_declarations(payload, source)
    catalog = Catalog()
    old = [entry['candidate'] for entry in baseline['candidates']]
    old_ids = [f'old:{i:04d}' for i in range(len(old))]
    components = [declaration(identifier, candidate.program) for identifier, candidate in zip(old_ids, old)]
    # Give shared Programs ids before intervention-created candidates enter.
    for candidate in old:
        catalog.add(candidate.program)
    with tempfile.TemporaryDirectory(prefix='hegel-cross-skeleton-prepare-') as temporary:
        preparation_agent, lab = new_agent(temporary), ReplayLab()
        preparation = preparation_agent._execute('run', {'operator': 'theory_consolidation',
            'theories': sources, 'source_theory_ids': [item['id'] for item in sources],
            'episode_sources': [{**source_ref(source, f'episodes[{position}]'), 'action_cutoff': cutoff}
                                for position, cutoff in ((1, 18), (2, 34))],
            'context_family_ref': RELATION_FAMILY, 'theme': 'cross_skeleton_history_coverage',
            'mission': mission, 'allocation': {'compute': 30000, 'max_groups': 4}}, lab)
        relation_refs = [entry for entry in preparation['frozen_candidates']
                         if entry['kind'] == 'program_context_relation']
        if len(relation_refs) != 1:
            raise AssertionError(preparation)
        frozen_id = relation_refs[0]['candidate_id']
        frozen = deepcopy(preparation_agent.theory_consolidation['candidates'][frozen_id])
        register_theories(preparation_agent, components)
        transfer_args = {'operator': 'theory_transfer', 'library_path': preparation['library_path'],
            'candidate_id': frozen_id, 'theory_id': 'relation:t07', 'component_theory_ids': old_ids,
            'context_ref': deepcopy(CONTEXT), 'role_signature': deepcopy(ROLES), 'output_unit': UNIT,
            'allocation': {'compute': 20000}}
        application = preparation_agent._execute('run', transfer_args, lab)
        generated = []
        for entry in application['generated_theories']:
            theory = preparation_agent.theory_index[entry['theory_id']]
            generated.append((entry, Program.from_dict(theory['program'])))
        # Select a falsification only from the already admitted fitting prefix.
        # This diagnostic is counted, and cannot inspect the held-out action65.
        checked_predictions, bad = 0, None
        for entry, program in generated:
            for i, row in enumerate(prefix):
                value = predict(program, row, baseline['fallback'])
                checked_predictions += 1
                if abs(value - row.target) > 1e-6:
                    bad = (entry, program, i)
                    break
            if bad:
                break
        if bad is None:
            raise AssertionError('This retained case needs an actually false generated numerical binding')
        entry, bad_program, row_index = bad
        check = {'source_ref': _prefix_ref(source, prefix, row_index),
            'observation': _row_dict(prefix[row_index]), 'context_ref': deepcopy(CONTEXT), 'tolerance': 1e-6}
        failed = preparation_agent._execute('run', {**transfer_args,
            'theory_id': entry['theory_id'], 'component_theory_ids': [],
            'component_bindings': {hole: binding['component_id'] for hole, binding in entry['binding']['components'].items()},
            'evidence_checks': [check]}, lab)
        failed_entry, = failed['generated_theories']
        if failed_entry['status'] != 'refuted':
            raise AssertionError(failed)
        shared_failure = deepcopy(preparation_agent.theory_index[failed_entry['theory_id']])
        shared_library = deepcopy(preparation_agent.theory_consolidation)
        generated_theories = [deepcopy(preparation_agent.theory_index[item['theory_id']])
                             for item, _ in generated if item['theory_id'] != failed_entry['theory_id']]
        if lab.execute_calls or preparation_agent.costs['model_calls']:
            raise AssertionError('Historical replay acquired external data')
    arms, score_table, derivations = {}, {}, []
    old_compute = sum(baseline['costs'].values())
    # The hold-out is public controls only until both parent predictions exist.
    future_features = deepcopy(target['actions'][65]['details']['sequence'][0])
    future_row = Observation(future_features, 0., ())
    for arm in ('remove', 'keep'):
        with tempfile.TemporaryDirectory(prefix=f'hegel-cross-skeleton-{arm}-') as temporary:
            agent, lab = new_agent(temporary), ReplayLab()
            agent.theory_consolidation = deepcopy(shared_library)
            # Intervention removes just this optional executable object. Failed
            # concrete bindings and their scoped counterevidence remain shared.
            if arm == 'remove':
                del agent.theory_consolidation['candidates'][frozen_id]
            agent.theory_consolidation['executions'] = {}
            agent._save(preparation['library_path'], json.dumps(agent.theory_consolidation))
            register_theories(agent, [*sources, *components])
            agent.theory_index[failed_entry['theory_id']] = deepcopy(shared_failure)
            alias = declaration('ordinary_alias:failed_binding', bad_program)
            alias['role_signature'] = deepcopy(shared_failure['role_signature'])
            register_theories(agent, [alias])
            checked_pool = {theory['id']: theory for theory in resolved_theories(agent)}
            checked_alias = checked_pool[alias['id']]
            if checked_alias.get('compatible') is not False or not checked_alias.get('scope_refutation'):
                raise AssertionError('A failed numerical alias escaped the retained scoped counterexample')
            candidates, candidate_ids = list(old), list(old_ids)
            computational_clones = 0
            if arm == 'keep':
                register_theories(agent, generated_theories)
                for theory in generated_theories:
                    if theory.get('compatible') is False:
                        continue
                    program = Program.from_dict(theory['program'])
                    # An inline or named representation of the same ordered
                    # computation gets one vote and one rank slot in both arms.
                    if not any(candidate.program.expanded() == program.expanded() for candidate in candidates):
                        candidates.append(Candidate(theory['id'], program, 'learned_revision_relation'))
                        candidate_ids.append(theory['id'])
                        derivations.append({'program_id': catalog.add(program),
                            'components': {name: binding['component_id'] for name, binding
                                           in theory['consolidation_ref']['components'].items()}})
                    else:
                        computational_clones += 1
            ranking_budget = SearchBudget(50000)
            ranked = rank_candidates(candidates, prefix, [], fallback=baseline['fallback'], q=.5,
                budget=ranking_budget, complexity_weight=.001)
            admitted, frontier = retain_history_frontier(ranked, prefix, maximum=16)
            # Resolve rank representatives back to the ordinary registered ids.
            identifiers = [candidate_ids[candidates.index(row.candidate)] for row in admitted]
            question, panel = plan_inputs(target, identifiers)
            agent.costs['compute_cost'] = old_compute + ranking_budget.spent
            if arm == 'keep':
                agent.costs['compute_cost'] += preparation['new_compute_cost'] + application['new_compute_cost']
            before = agent.costs['compute_cost']
            result = agent._execute('run', {'operator': 'research_route',
                'research_question': question, 'experiments': panel, 'mission': mission}, lab)
            current, ranking_winner, adoption = retained_parent_selection(
                ranked, baseline['incumbent'], payload['config'])
            continuation = predict(current.candidate.program, future_row, baseline['fallback'])
            prefix_scores = [{'program_id': catalog.add(row.candidate.program),
                'operation': row.candidate.operation, 'prefix_mse': row.evidence_loss, 'score': row.score,
                'admitted': row in admitted} for row in ranked]
            for row in prefix_scores:
                score_table[row['program_id']] = [row['program_id'], row['operation'], row['prefix_mse'], row['score']]
            added = len(candidates) - len(old)
            route_compute = agent.costs['compute_cost'] - before
            costs = {'old_constructors': old_compute, 'prefix_ranking': ranking_budget.spent,
                'common_route': route_compute, 'continuation_prediction': 1,
                'object_construction': preparation['new_compute_cost'] if arm == 'keep' else 0,
                'object_application': application['new_compute_cost'] if arm == 'keep' else 0,
                'shared_actual_countercheck': len(failed_entry['checks']),
                'object_binding_reinstantiation': (failed['new_compute_cost'] - len(failed_entry['checks'])) if arm == 'keep' else 0,
                'shared_counterexample_search_diagnostics': checked_predictions}
            costs['total'] = sum(costs.values())
            arms[arm] = {'new_relation_retained': arm == 'keep',
                'all_old_candidates_retained': all(candidate in candidates for candidate in old),
                'old_candidate_count': len(old), 'new_distinct_candidates': added,
                'expanded_computation_clones_merged': computational_clones,
                'candidate_count': len(candidates), 'admitted_count': len(admitted),
                'admitted_program_ids': [catalog.add(row.candidate.program) for row in admitted],
                'prefix_scores': {'table': 'prefix_score_table',
                    'rank_order': [row['program_id'] for row in prefix_scores]}, 'history_frontier': frontier,
                'plan': _selection_summary(result),
                'ranking_winner': {'program_id': catalog.add(ranking_winner.candidate.program),
                    'operation': ranking_winner.candidate.operation,
                    'candidate_id': ranking_winner.candidate.candidate_id,
                    'selection_rule': 'old rank_candidates: prefix MSE + .001 * Program size, stable original order',
                    'prefix_mse': ranking_winner.evidence_loss, 'score': ranking_winner.score},
                'adoption': adoption,
                'parent': {'program_id': catalog.add(current.candidate.program),
                    'operation': current.candidate.operation, 'candidate_id': current.candidate.candidate_id,
                    'source_ref': source_ref(source, 'episodes[6].actions[44].details.after_program'),
                    'selection_rule': 'retain original action44 incumbent pending a fresh comparison',
                    'prefix_mse': current.evidence_loss, 'score': current.score,
                    'continuation_features': future_features, 'continuation_prediction': continuation,
                    'fallback': baseline['fallback'], 'independent_confirmation': False},
                'counterexample': {'failed_theory_id': failed_entry['theory_id'],
                    'source_ref': check['source_ref'], 'actual_transfer_status': failed_entry['status'],
                    'checks': failed_entry['checks'], 'ordinary_alias_compatible': checked_alias['compatible'],
                    'ordinary_alias_scope_refutation': checked_alias['scope_refutation'],
                    'object_present': frozen_id in agent.theory_consolidation['candidates'],
                    'failed_transfer_preserved': any(item['status'] == 'refuted' for item in
                        agent.theory_consolidation['transfers'] if item['theory_id'] == failed_entry['theory_id']),
                    'source_reconstruction_preserved': all(item['exact'] for item in frozen['source_reconstruction'])},
                'compute': costs, 'no_acquisition': {'execute_calls': lab.execute_calls,
                    'model_calls': agent.costs['model_calls'], 'observation_cost': agent.costs['observation_cost'],
                    'measurement_cost': agent.costs['measurement_cost'], 'paid_record_count': len(agent.observations)}}
            if any(arms[arm]['no_acquisition'].values()) or costs['total'] > TOTAL_COMPUTE:
                raise AssertionError('Replay exceeded the common budget or acquired data')
    # Only now reveal the paid historical continuation label; no generation,
    # ranking, experiment selection or refutation can depend on this value.
    actual = target['memory_events'][0]['observation']['history'][0]['__target__']
    for arm in arms.values():
        arm['parent']['continuation_target'] = actual
        arm['parent']['continuation_loss'] = point_loss(arm['parent']['continuation_prediction'], actual, .5)
    # Export only Programs needed to interpret actual decisions; the full old
    # sources and hundreds of constructor products remain addressed, not copied.
    selected_program_ids = {arm[key]['program_id'] for arm in arms.values()
                            for key in ('parent', 'ranking_winner')}
    receipt = {'status': 'completed', 'scope': 'same_material_cross_skeleton_relation_keep_remove',
        'source': source, 'protocol': {'source_cutoffs': [
            {'episode_index': 1, 'action_cutoff': 18}, {'episode_index': 2, 'action_cutoff': 34}],
            'source_episode_boundary_exclusive': 3, 'target_episode_index': 6, 'target_action_cutoff': 44,
            'old_archive_episode_indices': list(range(7)),
            'prefix_sources': [{'memory_event_index': 2, 'action_index': 14, 'rows': 3},
                               {'memory_event_index': 3, 'action_index': 40, 'rows': 5}],
            'prefix_row_count': len(prefix), 'panel_action_indices': list(PANEL_ACTIONS),
            'public_panel_readout_step': 1, 'continuation_action_index': 65, 'continuation_step': 0,
            'future_trajectory_labels_used_for_generation_or_selection': False,
            'final_target_program_used_for_generation': False,
            'fixed_admission': 'old rank_candidates and retain_history_frontier(maximum=16)',
            'parent_consumption': 'original action44 incumbent retained; adoption deferred without a fresh comparison',
            'same_total_compute_budget': TOTAL_COMPUTE, 'relation_application_allocation': 20000,
            'actual_acquisition_allowed': False},
        'old_baseline': {'capabilities': baseline['capabilities'], 'constructor_compute': baseline['costs'],
            'archive_program_refs': [entry['source_ref'] for entry in baseline['archive']],
            'future_episode_programs_not_loaded_into_prior': True},
        'frozen_relation': frozen,
        'mainline_induction': preparation,
        'mainline_application': {'status': application['status'], 'compute': application['new_compute_cost'],
            'budget_exhausted': application['budget_exhausted'], 'excluded': application['excluded'],
            'generated_count': len(application['generated_theories']),
            'source_component_count': len(components),
            'concrete_programs_checked_through_existing_transfer': True},
        'counterexample': {'observation': check['observation'], 'source_ref': check['source_ref'],
            'actual_transfer': failed, 'failed_program': bad_program.to_dict(),
            'diagnostic_row_predictions': checked_predictions},
        'source_revision_outcomes': [
            {'episode_index': 1, 'revision_action_index': 18,
             'measured_check_ref': source_ref(source, 'episodes[1].actions[19]'),
             'measured_check': deepcopy(payload['episodes'][1]['actions'][19]['details']),
             'later_replacement_ref': source_ref(source, 'episodes[1].actions[36]'),
             'later_final_loss': payload['episodes'][1]['evaluation']['loss'],
             'final_loss_scores_the_later_program_not_source_revision': True},
            {'episode_index': 2, 'revision_action_index': 34,
             'measured_check_ref': source_ref(source, 'episodes[2].actions[35]'),
             'measured_check': deepcopy(payload['episodes'][2]['actions'][35]['details']),
             'final_evaluation_ref': source_ref(source, 'episodes[2].evaluation'),
             'final_loss': payload['episodes'][2]['evaluation']['loss']}],
        'arms': arms, 'comparison': compare_arms(arms['keep'], arms['remove']),
        'prefix_score_table': {'columns': ['program_id', 'operation', 'prefix_mse', 'score'],
                              'rows': list(score_table.values())},
        'generated_program_derivations': {'component_ids': 'old:NNNN refers to original stable old-baseline Program index NNNN',
            'rows': derivations},
        'decision_programs': {identifier: program for identifier, program in catalog.export().items()
                              if identifier in selected_program_ids},
        'historical_costs_shared': [{'episode_index': i, 'task_id': episode['task_id'],
            'observation_cost': episode['observation_cost'], 'compute_cost': episode['compute_cost'],
            'costs_ref': source_ref(source, f'episodes[{i}]'), 'newly_charged': False}
            for i, episode in enumerate(payload['episodes'][:7])],
        'executed_in_harness': {'old_constructors_runs': 1, 'old_constructors_compute': old_compute,
            'object_construction_runs': 1, 'object_construction_compute': preparation['new_compute_cost'],
            'object_application_runs': 1, 'object_application_compute': application['new_compute_cost'],
            'shared_countercheck_runs': 1, 'shared_countercheck_compute': failed['new_compute_cost'],
            'new_paid_observations': 0, 'provider_calls': 0},
        'limitations': ['The two source revisions and target are historically selected, not blind independent confirmation.',
            'The relation constrains optional Program construction; it is not an established physical law or a global rejection rule.',
            'Only eight actual labels recoverable in the two retained observation prefixes are used; these are not a complete reconstruction of all paid training data.',
            'The selected experiment is prepared without execution. Its action benefit remains unmeasured.',
            'Ranking is screening only; the controller prospective adoption rules are not executed without a fresh comparison.',
            'The one already stored continuation label checks a frozen parent prediction, not new independent scientific evidence.']}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pointer', type=Path, default=POINTER)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    receipt = replay_relations(args.source_pointer, args.output)
    print(json.dumps({'status': receipt['status'], 'output': str(args.output),
        'comparison': receipt['comparison'], 'old_baseline': receipt['old_baseline']['capabilities'],
        'generated': receipt['mainline_application']['generated_count']}, ensure_ascii=False))
