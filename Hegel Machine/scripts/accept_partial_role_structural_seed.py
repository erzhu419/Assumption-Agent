#!/usr/bin/env python3
"""Resume the first registered PR3 structural question on its paid rows.

This is a paid development segment, not a target rerun. Restore the actual
bound parent from its saved dependent question and the competing partial
response from its original graph. Let the ordinary chooser decide whether a
different structural search is justified. Later fitted parents are not inputs.
"""

import argparse
from copy import deepcopy
from math import isclose
import json
from pathlib import Path
import sys
from time import monotonic, process_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.object_continuation import encode_entry
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition import unified_structure
from scripts.run_partial_role_parent import load, save


def _rows(values):
    return [Observation(row['features'], row['target'], tuple(row.get('history', ())))
            for row in values]


def recover_paid_scene(report):
    """Recover the first structural selection; never choose a later success."""
    actions = report['actions']
    choice_index, choice = next((i, item) for i, item in enumerate(actions)
        if item.get('action') == 'choose_research_action'
        and item.get('selected_action') == 'structural_research')
    compare_index = choice_index + 1
    comparison = actions[compare_index]
    assert comparison['action'] == 'compare_structural_explanations'
    feedback_index = next(i for i in range(choice_index - 1, -1, -1)
                          if actions[i].get('action') == 'research_action_feedback')
    feedback = actions[feedback_index]
    assert feedback['evidence_basis'] == 'common_unfitted_validation'
    program = Program.from_dict(feedback['baseline_program'])
    features = tuple(comparison['features'])

    training, validation, transitions, training_batches = [], [], [], []
    validation_batch = None
    for i, batch in enumerate(report['batches']):
        if set(batch['observations'][0]['features']) != set(features):
            continue
        rows = _rows(batch['observations'])
        if len(training) < feedback['fitting_rows']:
            training.extend(rows)
            transitions.extend(deepcopy(batch.get('transitions', [])))
            training_batches.append(i)
        else:
            validation, validation_batch = rows, i
            break
    assert len(training) == feedback['fitting_rows']
    assert len(validation) == feedback['evidence_rows']
    assert all(row not in training for row in validation)

    # A later revision stores the ORIGINAL graph alongside its original AST.
    # Recover that graph only when the AST and fitted-row boundary both agree.
    original_graph, graph_ref = None, None
    for i, action in enumerate(actions):
        for j, alternative in enumerate(action.get('alternatives', [])):
            revision = alternative.get('revision', {})
            graph = revision.get('original_research_program')
            if (revision.get('original_program') == feedback['baseline_program']
                    and graph and graph['fit_rows'] == len(training)):
                original_graph = deepcopy(graph)
                graph_ref = ['actions', i, 'alternatives', j, 'revision', 'original_research_program']
                break
        if original_graph is not None:
            break
    assert original_graph is not None
    assert original_graph['graph']['nodes'][original_graph['output_node']]['answer'] == program.to_dict()

    model = report['object_research']
    adoption = next(item for item in actions[:choice_index]
                    if item.get('action') == 'adopt_discovered_objects')
    assert model['support_rows'] == model['acquisition_rows'] == len(training)
    assert adoption['groups'] == model['groups']
    assert adoption['object_model_id'] == model['model_id']
    coverage, coverage_ref = None, None
    for i, frontier in enumerate(report['object_continuation_state']['candidate_frontiers']):
        for j, entry in enumerate(frontier['entries']):
            partial = entry.get('partial_object_knowledge', {})
            if (entry['kind'] == feedback['baseline_kind']
                    and partial.get('source_model_id') == model['model_id']
                    and set(partial.get('used_fields', [])) == program.required_features()):
                coverage, coverage_ref = deepcopy(partial), [
                    'object_continuation_state', 'candidate_frontiers', i, 'entries', j,
                    'partial_object_knowledge']
                break
        if coverage is not None:
            break
    assert coverage is not None
    # Only immutable role coverage comes from the later frontier. Its later
    # parameters, scores, fitted graph and selected parent are never consumed.
    baseline = {'candidate': Candidate('paid:first_response', program, 'restored_paid_response'),
        'kind': feedback['baseline_kind'], 'family_id': 'T03',
        'research_program': original_graph, 'partial_object_knowledge': coverage,
        'training_loss': original_graph['graph']['nodes'][original_graph['output_node']]['loss'],
        'validation_loss': feedback['before_loss']}
    directions_index = next(i for i in range(compare_index + 1, len(actions))
        if actions[i]['action'] == 'compare_research_directions')
    dependent_index = next(i for i in range(compare_index + 1, directions_index)
        if actions[i]['action'] == 'execute_dependent_research')
    dependent = actions[dependent_index]
    parent_program = dependent['parent_program']
    # This same parent is saved by the mechanism research BEFORE the selected
    # structural question, so the later dependent-action record is only its
    # archival carrier, not a replacement fitted on later data.
    parent_ref = next(['actions', i, 'alternatives', j, 'revision', 'original_program']
        for i, item in enumerate(actions[:choice_index])
        for j, alternative in enumerate(item.get('alternatives', []))
        if alternative.get('revision', {}).get('original_program') == parent_program)
    retained_score = next(item for item in actions[directions_index]['alternatives']
                          if item['kind'] == 'retained_response')
    parent = {'candidate': Candidate(dependent['parent_candidate_id'], Program.from_dict(parent_program),
                'retained_response'), 'kind': 'retained_response', 'family_id': 'T03',
              'validation_loss': retained_score['validation_loss']}
    return {'spec': TaskSpec(report['task_id'], 'Resume the paid structural question', features, (), (),
                context={'object_schema': json.dumps(model['schema']), 'object_model': deepcopy(model)}),
        'training': training, 'validation': validation, 'transitions': transitions,
        'baseline': baseline, 'parent': parent, 'fallback': initial_prediction(training, .5),
        'choice_index': choice_index, 'compare_index': compare_index,
        'remaining_compute': choice['compute_remaining'],
        'structural_allowance': min(90000, choice['compute_remaining'] // 2),
        'references': {'training_batches': training_batches, 'validation_batch': validation_batch,
            'baseline_program': ['actions', feedback_index, 'baseline_program'],
            'original_research_program': graph_ref, 'unchanged_role_coverage': coverage_ref,
            'original_parent_program': parent_ref,
            'bound_parent_carrier': ['actions', dependent_index, 'parent_program'],
            'original_complete_comparison': ['actions', directions_index, 'alternatives'],
            'all_original_candidate_scores': [['actions', i, 'alternatives']
                for i, item in enumerate(actions[:choice_index]) if 'alternatives' in item]}}


def run_segment(scene, agent):
    """Restore both paid scores; only a changed seed executes new research."""
    budget = SearchBudget(scene['remaining_compute'])
    training, validation = scene['training'], scene['validation']
    baseline, parent, actions = deepcopy(scene['baseline']), deepcopy(scene['parent']), []
    for entry in (baseline, parent):
        assert budget.spend(len(training) + len(validation))
        measured_train = loss(entry['candidate'].program, training, scene['fallback'], .5)
        measured_validation = loss(entry['candidate'].program, validation, scene['fallback'], .5)
        assert isclose(measured_validation, entry['validation_loss'], rel_tol=1e-10, abs_tol=1e-20)
        if 'training_loss' in entry:
            assert isclose(measured_train, entry['training_loss'], rel_tol=1e-10, abs_tol=1e-20)
        entry['training_loss'] = measured_train
    restoration_cost = budget.spent
    # The feedback immediately before the frozen choice retained this actual
    # pool winner. Comparing selectors does not require regenerating that pool.
    pool = [baseline, parent]
    selected = agent._select_structure_response_parent(scene['spec'], parent, pool,
        parent, True, actions, len(validation))
    changed = selected['candidate'].program != parent['candidate'].program
    local, checked = SearchBudget(scene['structural_allowance']), []
    if changed:
        try:
            proposals = unified_structure.propose_structures(scene['spec'], training, local,
                parent=selected, tolerance=agent.config.minimum_improvement,
                transitions=scene['transitions'])
        finally:
            budget.spend(local.spent)
        checked = agent._assess(proposals, validation, budget, scene['fallback'], .5)
    best_old, best_new = agent._best(pool), agent._best(checked)
    return {'status': 'changed_seed_researched' if changed else 'no_changed_decision',
        'scope': 'paid first structural segment; no final adoption or independent confirmation',
        'actions': actions, 'old_pool_kept_by_reference': True,
        'old_pool_replay': 'saved original winner and bound parent; full earlier pool not regenerated',
        'training_rows': len(training), 'validation_rows': len(validation),
        'original_parent': encode_entry(parent), 'selected_response': encode_entry(selected),
        'new_candidates': [encode_entry(item) for item in checked],
        'new_candidate_count': len(checked), 'old_fits_repeated': 0, 'structural_research_executed': changed,
        'original_loss': baseline['validation_loss'], 'restored_loss': baseline['validation_loss'],
        'best_new_loss': best_new['validation_loss'] if best_new else None,
        'improved': bool(best_new and best_new['validation_loss'] + agent.config.minimum_improvement
                         < best_old['validation_loss']), 'compute_cost': budget.spent,
        'restoration_scoring_compute': restoration_cost, 'structural_compute': local.spent,
        'observation_cost': 0, 'measurement_cost': 0, 'llm_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-receipt', type=Path, required=True)
    args = parser.parse_args()
    source = args.root / 'runs/106029101/pr3/response_report.json'
    registration = load(args.root / 'registration.json')
    prior = load(args.prior_receipt)
    scene = recover_paid_scene(load(source))
    agent = UnifiedResearchAgent(ResearchConfig(**registration['research_config']))
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'registration.json', {'source_report_ref': {'file': str(source)},
        'source_registration_ref': {'file': str(args.root / 'registration.json')},
        'historical_knowledge_ref': registration['historical_knowledge_ref'],
        'references': scene['references'], 'original_choice_index': scene['choice_index'],
        'compute_limit': scene['remaining_compute'], 'structural_allowance': scene['structural_allowance'],
        'cost_scope': 'paid development computation; original unsuccessful cohort and invalid recovery costs separate',
        'original_selector_ref': {'file': str(args.root / 'runtime/src/hegel_machine/cognition/unified_research.py'),
                                 'lines': [2136, 2140]},
        'fit_rows': len(scene['training']), 'validation_rows': len(scene['validation']),
        'observations_purchased': 0, 'independent_evaluation': False})
    started, cpu = monotonic(), process_time()
    result = run_segment(scene, agent)
    result.update(elapsed_seconds=monotonic() - started, process_cpu_seconds=process_time() - cpu)
    save(args.output / 'new_claims.json', result.pop('new_candidates'))
    result.update(total_compute_cost=prior['compute_cost'] + result['compute_cost'],
        prior_attempt_ref={'file': str(args.prior_receipt)},
        prior_attempt_status='invalid_parent_recovery', prior_attempt_compute_cost=prior['compute_cost'],
        correction='The original parent was supplied by ObjectResearchActions._structure_view; '
                   'the outer run variable being None did not imply this local parent was absent.',
        new_claims_ref={'file': str(args.output / 'new_claims.json')})
    save(args.output / 'receipt.json', result)
    print({key: result[key] for key in ('status', 'new_candidate_count', 'original_loss', 'restored_loss',
                                       'improved', 'compute_cost', 'elapsed_seconds')})


if __name__ == '__main__':
    main()
