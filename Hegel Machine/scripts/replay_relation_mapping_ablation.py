#!/usr/bin/env python3
"""Permute only frozen relation bindings on a retained, bounded t07 domain.

The old 344 Programs stay available. No candidate or permutation is selected
using a later target. Native constraints, ranking, admission and the partial
historical adoption rule keep their existing implementations.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean
import tempfile

from hegel_machine.cognition.candidate_comparison import compare_candidate_adoption
from hegel_machine.cognition.evaluation import point_loss, predict
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Candidate, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.question_routing import register_theories, resolved_theories
from hegel_machine.cognition.revision_relations import instantiate_relation
from hegel_machine.cognition.selection import rank_candidates, retain_history_frontier
from audit_cross_skeleton_consumption import reconstruct_inputs, RECEIPT
from replay_cross_skeleton_relations import (
    CONTEXT, POINTER, ROOT, ROLES, TOTAL_COMPUTE, ReplayLab, declaration, new_agent,
)
from replay_cross_skeleton_selection_contract import _inputs
from retained_cross_skeleton_evidence import recover_t07_evidence


OUTPUT = ROOT / 'docs/evidence/relation_mapping_ablation_20261021.json'


def recheck_bindings(candidate, components, domain, permutation, *, compute_budget):
    """Apply a fixed argument permutation, then use the existing native checks."""
    names = tuple(Definition.from_dict(candidate['definition']).arguments)
    if len(permutation) != len(names) or set(permutation) != set(names):
        raise ValueError('the intervention must permute the same frozen arguments')
    budget = SearchBudget(compute_budget)
    proposals, costs, outcome_indices, outcomes, examples, statuses = [], [], [], [], [], Counter()
    exhausted = False
    for index, item in enumerate(domain):
        if not budget.spend(1):
            exhausted = True
            break
        bindings = {name: components[item['components'][source]]
                    for name, source in zip(names, permutation)}
        result = instantiate_relation(candidate, bindings, allocation={'compute': budget.remaining})
        budget.spend(result['actual_compute_cost'])
        status = result['status']
        statuses[status] += 1
        checks = [check for excluded in result['excluded']
                  for check in excluded.get('constraint_checks', [])]
        violated = [i for i, check in enumerate(checks) if check['status'] == 'violated']
        outcome = {'status': status, 'violated_constraint_indices': violated}
        if outcome not in outcomes:
            outcomes.append(outcome)
        costs.append(1 + result['actual_compute_cost'])
        outcome_indices.append(outcomes.index(outcome))
        if result['excluded'] and len(examples) < 3:
            examples.append({'domain_index': index, 'excluded': deepcopy(result['excluded'])})
        proposals.extend({'domain_index': index, **proposal} for proposal in result['proposals'])
        if result['budget_exhausted']:
            exhausted = True
            break
    return {'status': 'UNKNOWN' if exhausted or statuses['UNKNOWN'] else 'completed',
        'proposals': proposals, 'actual_compute_cost': budget.spent, 'budget_exhausted': exhausted,
        'processed_bindings': len(costs), 'status_counts': dict(statuses), 'excluded_examples': examples,
        'binding_checks': {'order': 'saved_domain_order', 'compute_units': costs,
                           'outcome_indices': outcome_indices, 'outcomes': outcomes}}


def _domain(receipt):
    failed, = receipt['counterexample']['actual_transfer']['generated_theories']
    domain = [{'program_id': None, 'components': {name: binding['component_id']
        for name, binding in failed['binding']['components'].items()},
        'source_field': 'counterexample.actual_transfer.generated_theories[0].binding.components'}]
    domain.extend({**deepcopy(row), 'source_field':
        f'generated_program_derivations.rows[program_id={row["program_id"]}]'}
        for row in receipt['generated_program_derivations']['rows'])
    if len(domain) != receipt['mainline_application']['generated_count']:
        raise ValueError('the saved finite binding domain is incomplete')
    return domain


def _assessment(row):
    return {'program_id': row.candidate.candidate_id, 'score': row.score,
            'prefix_mse': row.evidence_loss, 'fresh_mse': row.fresh_loss}


def replay_relation_mapping_ablation(receipt_path=RECEIPT, pointer=POINTER, output=OUTPUT):
    data = reconstruct_inputs(receipt_path, pointer)
    original, baseline = data['receipt'], data['baseline']
    payload, source, programs = data['payload'], data['source'], data['programs']
    frozen = deepcopy(original['frozen_relation'])
    names = tuple(Definition.from_dict(frozen['definition']).arguments)
    if len(names) < 2:
        raise ValueError('this retained object has no nonidentity argument permutation')
    permutations = {'original': names, 'permuted': names[1:] + names[:1]}
    domain = _domain(original)
    components = {f'old:{i:04d}': declaration(f'old:{i:04d}', entry['candidate'].program)
                  for i, entry in enumerate(baseline['candidates'])}
    evidence = recover_t07_evidence(payload, source)
    training, fresh, later = (evidence[key] for key in ('training_rows', 'comparison_rows', 'later_rows'))
    if training != baseline['prefix']:
        raise ValueError('the retained fitting rows changed')
    config = ResearchConfig(**payload['config'])
    spec = TaskSpec('retained_t07_mapping_ablation', 'Original native numeric contract', ('x', 'z'), (), ())
    incumbent_id = next(key for key, program in programs.items() if program == baseline['incumbent'])
    mission = {'purpose': 'theory_learning', 'mission_id': 'retained_relation_mapping_ablation',
        'theory_learning_allocation': {'compute': TOTAL_COMPUTE, 'author_calls': 0, 'observations': 0}}
    saved_costs = original['arms']['keep']['compute']
    retained_debit = saved_costs['total'] - sum(saved_costs[key] for key in (
        'prefix_ranking', 'common_route', 'continuation_prediction'))
    arms, forecast_table = {}, {}
    actual = sum(baseline['costs'].values())
    with tempfile.TemporaryDirectory(prefix='hegel-relation-mapping-') as temporary:
        seed, seed_lab = new_agent(Path(temporary) / 'shared'), ReplayLab()
        seed.theory_consolidation = {'candidates': {frozen['id']: frozen}, 'executions': {}}
        seed._save('shared/theory_library.json', json.dumps(seed.theory_consolidation))
        register_theories(seed, list(components.values()))
        check = {key: deepcopy(original['counterexample'][key]) for key in ('observation', 'source_ref')}
        check.update(context_ref=deepcopy(CONTEXT), tolerance=1e-6)
        if not any(check['observation'] == {'features': row.features, 'history': list(row.history),
                                           'target': row.target} for row in training):
            raise ValueError('the retained numerical counterexample is outside the fixed fitting prefix')
        failed = seed._execute('run', {'operator': 'theory_transfer', 'candidate_id': frozen['id'],
            'theory_id': 'shared_retained_failure', 'component_bindings': domain[0]['components'],
            'role_signature': deepcopy(ROLES), 'output_unit': frozen['output_unit'],
            'context_ref': deepcopy(CONTEXT), 'evidence_checks': [check],
            'allocation': {'compute': TOTAL_COMPUTE}}, seed_lab)
        failed_entry, = failed['generated_theories']
        if failed_entry['status'] != 'refuted':
            raise ValueError('the original paid numerical failure was not reproduced')
        shared_check_compute = seed.costs['compute_cost']
        actual += shared_check_compute
        for arm, permutation in permutations.items():
            agent, lab = new_agent(Path(temporary) / arm), ReplayLab()
            agent.theory_consolidation = deepcopy(seed.theory_consolidation)
            agent.theory_index = deepcopy(seed.theory_index)
            agent._save('shared/theory_library.json', json.dumps(agent.theory_consolidation))
            agent.costs['compute_cost'] = retained_debit + shared_check_compute
            checked = recheck_bindings(frozen, components, domain, permutation,
                compute_budget=max(0, TOTAL_COMPUTE - agent.costs['compute_cost']))
            agent.costs['compute_cost'] += checked['actual_compute_cost']
            declarations = []
            expected = {}
            for proposal in checked['proposals']:
                index = proposal['domain_index']
                identifier = f'mapped:{index:04d}'
                declarations.append({**declaration(identifier, Program.from_dict(proposal['program'])),
                    'role_signature': deepcopy(proposal['role_signature']),
                    'consolidation_ref': {'candidate_id': frozen['id'], 'components': proposal['bindings']}})
                expected[identifier] = domain[index]['program_id']
            register_theories(agent, declarations)
            # The actual refuted concrete Program remains present in both arms.
            # resolved_theories also excludes an inline alias of that same Program.
            pool = {item['id']: item for item in resolved_theories(agent)}
            old_candidates = [Candidate(f'program_{i:04d}', entry['candidate'].program,
                entry['candidate'].operation) for i, entry in enumerate(baseline['candidates'])]
            candidates, rejected_empirically = list(old_candidates), []
            for item in declarations:
                resolved = pool[item['id']]
                if resolved.get('compatible') is False:
                    rejected_empirically.append(item['id'])
                    continue
                program = Program.from_dict(resolved['program'])
                identifier = expected[item['id']]
                if arm == 'original' and (identifier is None or programs[identifier] != program):
                    raise ValueError('the original binding does not reconstruct its saved Program')
                identifier = identifier if arm == 'original' else f'permuted:{item["id"]}'
                programs[identifier] = program
                if not any(old.program.expanded() == program.expanded() for old in candidates):
                    candidates.append(Candidate(identifier, program, 'learned_revision_relation'))
            ranking_budget = SearchBudget(min(50000, max(0, TOTAL_COMPUTE - agent.costs['compute_cost'])))
            ranked = rank_candidates(candidates, training, [], fallback=baseline['fallback'], q=.5,
                budget=ranking_budget, complexity_weight=config.complexity_weight)
            agent.costs['compute_cost'] += ranking_budget.spent
            admitted, frontier = retain_history_frontier(ranked, training, maximum=16)
            identifiers = [row.candidate.candidate_id for row in admitted]
            register_theories(agent, [declaration(row.candidate.candidate_id, row.candidate.program) for row in ranked])
            question, panel = _inputs(payload['episodes'][6], identifiers, 'aligned_step0_native',
                                      baseline['fallback'], source)
            before = agent.costs['compute_cost']
            routed = agent._execute('run', {'operator': 'research_route', 'research_question': question,
                'experiments': panel, 'mission': deepcopy(mission)}, lab)
            route_compute = agent.costs['compute_cost'] - before
            admitted_candidates = [row.candidate for row in admitted]
            incumbent = next(row for row in admitted_candidates if row.candidate_id == incumbent_id)
            # Freeze these Programs' predictions before the label-dependent
            # partial adoption calculation; no fitting occurs after admission.
            for row in admitted:
                identifier = row.candidate.candidate_id
                forecast_table.setdefault(identifier, {
                    'comparison': [predict(row.candidate.program, item, baseline['fallback']) for item in fresh],
                    'later': [predict(row.candidate.program, item, baseline['fallback']) for item in later]})
            comparison_budget = SearchBudget(max(0, TOTAL_COMPUTE - agent.costs['compute_cost']))
            assessed = rank_candidates(admitted_candidates, training, fresh, fresh=fresh, fresh_weight=.5,
                fallback=baseline['fallback'], q=.5, budget=comparison_budget,
                complexity_weight=config.complexity_weight)
            comparison = compare_candidate_adoption(assessed, incumbent, training, [], fresh,
                fallback=baseline['fallback'], spec=spec, config=config)
            selected = comparison.best if comparison.adopt else comparison.current
            diagnostics = len(admitted) * (len(fresh) + len(later))
            support = 2 * len(fresh)
            total = agent.costs['compute_cost'] + comparison_budget.spent + diagnostics + support
            if total > TOTAL_COMPUTE or lab.execute_calls or seed_lab.execute_calls or agent.costs['model_calls']:
                raise ValueError('the unchanged historical budget or no-acquisition contract was violated')
            after = forecast_table[selected.candidate.candidate_id]['later']
            before_forecast = forecast_table[incumbent_id]['later']
            arms[arm] = {'mapping': dict(zip(names, permutation)),
                'binding_recheck': {key: value for key, value in checked.items() if key != 'proposals'},
                'old_candidate_count': len(old_candidates), 'all_old_programs_preserved': all(
                    candidate in candidates for candidate in old_candidates),
                'candidate_count': len(candidates), 'new_distinct_candidates': len(candidates) - len(old_candidates),
                'numerically_refuted_bindings': rejected_empirically,
                'retained_numeric_failure_present': pool[failed_entry['theory_id']].get('compatible') is False,
                'retained_failed_checks_present': any(item['status'] == 'refuted'
                    and item['theory_id'] == failed_entry['theory_id']
                    for item in agent.theory_consolidation['transfers']),
                'source_reconstruction_preserved': all(row['exact'] for row in frozen['source_reconstruction']),
                'admitted_program_ids': identifiers, 'history_frontier': frontier,
                'selection': {'status': routed['status'], 'tied_experiments': routed['selection']['tied_experiments'],
                    'selected_experiment': routed['selection']['selected_experiment'],
                    'pending_query_prepared': routed.get('parent_use', {}).get('status') == 'parent_query_prepared'},
                'partial_comparison': {'scope': 'two_retained_postfreeze_rows_only',
                    'ranking': [_assessment(row) for row in assessed], 'native_gate_adopt': comparison.adopt,
                    'before_program_id': incumbent_id, 'after_program_id': selected.candidate.candidate_id,
                    'support': comparison.support, 'fresh_gain': comparison.fresh_gain,
                    'margin': comparison.margin, 'actual_source_deployment_changed': False},
                'later_excerpt': {'scope': 'four_retained_final_examples_only',
                    'before_mse': mean(point_loss(value, row.target, .5) for value, row in zip(before_forecast, later)),
                    'after_mse': mean(point_loss(value, row.target, .5) for value, row in zip(after, later)),
                    'forecast_ref': f'forecast_table.{selected.candidate.candidate_id}.later',
                    'used_for_admission_or_adoption': False},
                'compute': {'shared_retained_stage_debit': retained_debit,
                    'shared_counterexample_replay': shared_check_compute,
                    'mapping_recheck': checked['actual_compute_cost'], 'prefix_ranking': ranking_budget.spent,
                    'common_route': route_compute, 'partial_native_ranking': comparison_budget.spent,
                    'native_support_predictions': support, 'frozen_forecast_diagnostics': diagnostics,
                    'total': total, 'remaining': TOTAL_COMPUTE - total},
                'new_paid_observations': 0, 'new_provider_calls': 0}
            actual += (checked['actual_compute_cost'] + ranking_budget.spent + route_compute
                       + comparison_budget.spent + support + diagnostics)
    normal, permuted = arms['original'], arms['permuted']
    receipt = {'status': ('UNKNOWN_incomplete_mapping_checks' if any(
            arm['binding_recheck']['status'] == 'UNKNOWN' for arm in arms.values()) else
            'completed_partial_historical_mapping_ablation'),
        'source_receipt': str(receipt_path), 'source_archive': source,
        'protocol': {'intervention': 'rotate existing Definition argument bindings once in original argument order',
            'argument_order': list(names), 'permutations_fixed_before_target_labels': permutations,
            'frozen_relation_ref': {'file': str(receipt_path), 'field': 'frozen_relation'},
            'hole_contracts': deepcopy(frozen['holes']), 'construction_only': frozen['construction_only'],
            'finite_domain_size': len(domain), 'finite_domain_ref': {'file': str(receipt_path),
                'fields': ['counterexample.actual_transfer.generated_theories[0].binding.components',
                           'generated_program_derivations']},
            'original_generation_allocation': original['protocol']['relation_application_allocation'],
            'original_generation_status': original['mainline_application']['status'],
            'original_generation_budget_exhausted': original['mainline_application']['budget_exhausted'],
            'old_baseline_program_count': len(components), 'same_total_compute_budget': TOTAL_COMPUTE,
            'source_cutoffs': deepcopy(original['protocol']['source_cutoffs']),
            'target_action_cutoff': 44, 'training_rows': len(training), 'comparison_rows': len(fresh),
            'later_rows': len(later), 'admission_maximum': 16, 'selection_contract': 'aligned_step0_native',
            'native_adoption_ref': 'hegel_machine.cognition.candidate_comparison.compare_candidate_adoption',
            'selection_policy': config.selection_policy, 'complexity_weight': config.complexity_weight,
            'minimum_improvement': config.minimum_improvement,
            'new_candidate_or_parameter_fitting': False, 'new_paid_observations': 0, 'new_provider_calls': 0},
        'retained_evidence': {key: deepcopy(evidence[key]) for key in (
            'training_refs', 'comparison_refs', 'later_refs', 'coverage', 'duplicates', 'scope')},
        'counterexample_replay': {'status': failed_entry['status'], 'source_ref': check['source_ref'],
            'checks': failed_entry['checks'], 'compute': shared_check_compute},
        'arms': arms, 'forecast_table': forecast_table,
        'comparison': {'same_partial_parent': normal['partial_comparison']['after_program_id']
                == permuted['partial_comparison']['after_program_id'],
            'same_later_predictions': forecast_table[normal['partial_comparison']['after_program_id']]['later']
                == forecast_table[permuted['partial_comparison']['after_program_id']]['later'],
            'permuted_constraint_rejections': permuted['binding_recheck']['status_counts'].get('not_applicable', 0),
            'scientific_gain_established': False, 'complete_historical_comparison': 'UNKNOWN_missing14of16rows'},
        'actual_harness_compute': actual,
        'limitations': ['The fixed permutation tests construction constraints, not whether a scientific relation is true.',
            'Only the previously saved finite binding domain is checked; the original full enumeration was truncated.',
            'The two comparison rows and four later examples are historical excerpts, not new independent evidence.',
            'No selected panel is executed; missing original comparison and final-evaluation rows remain unknown.']}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, default=RECEIPT)
    parser.add_argument('--pointer', type=Path, default=POINTER)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = replay_relation_mapping_ablation(arguments.receipt, arguments.pointer, arguments.output)
    print(json.dumps({'output': str(arguments.output), 'comparison': result['comparison'],
        'arms': {name: {'candidate_count': arm['candidate_count'],
            'checks': arm['binding_recheck']['status_counts'], 'compute': arm['compute']['total']}
            for name, arm in result['arms'].items()}}, ensure_ascii=False))
