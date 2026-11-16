"""Compare a matched pair of position owners; fund a single discriminating square if eligible."""
import argparse
from dataclasses import replace
from pathlib import Path
from statistics import mean

from scripts.run_paid_position_contrast import execute_position_plan, freeze_position_runtime, run
from scripts.run_partial_role_parent import load, save
from scripts.run_partial_role_successors import config, program_knowledge
from hegel_machine.cognition.common_object_research import CommonObjectResearchAgent
from hegel_machine.cognition.evaluation import initial_prediction, point_loss, predict
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.object_alternatives import enumerate_object_schemas
from hegel_machine.cognition.object_continuation import encode_entry, observations
from hegel_machine.cognition.object_induction import apply_position_experiment
from hegel_machine.cognition.object_induction_continuation import paid_object_records
from hegel_machine.cognition.object_interventions import plan_position_owner_followup


def register(previous, output):
    prior = load(previous / 'registration.json')
    receipt = load(previous / 'receipt.json')
    if receipt['status'] != 'position_domain_narrowed':
        raise ValueError('A measured narrowing is required')
    ledger = load(previous / 'closed_source_costs.json')
    hold = ledger['completed_source_hold_remaining_observations']
    total = load(prior['authorization_ref']['file'])['authorized_limits']['observation_cost']
    if hold < 2 or hold + ledger['abc_remaining_observations'] + ledger['cumulative_protocol_observations'] != total:
        raise ValueError('The closed source reservation cannot fund both missing corners')
    structure = Path(prior['previous'])
    former = load(structure / 'competition.json')
    per_fit = min(attempt['compute_limit'] for s in former['searches'] for attempt in s['fit_attempts']
                  if attempt['stage'] == 'parent_response')
    structure_reg = load(structure / 'registration.json')
    working_rows = structure_reg['comparison_rows']
    # Two executable response fits, a numerical comparison on the paid rows,
    # and a complete second role experiment, including its paid-prefix filter.
    sensors_per_reset = 1 + len(load(previous / 'plan.json')['corners'][0])
    match_limit = prior['match_limit'] + receipt['laboratory_costs']['observation_cost'] * sensors_per_reset
    coverage_cap, planning_cap = 500, 1500
    maximum = (coverage_cap + 2 * per_fit + 2 * working_rows + 3 * 4
               + planning_cap + match_limit + prior['readout_limit'] + prior['paid_prefix_filter_limit'])
    if maximum > ledger['complete_relation_unit_remaining']:
        raise ValueError('Both response fits and the complete conditional follow-up must fit the development balance')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', {
        'kind': 'remaining_position_owners', 'previous': str(previous), 'structure_source': str(structure),
        'original_arm_source': structure_reg['source'], 'seed': structure_reg['seed'], 'arm': structure_reg['arm'],
        'paid_model_ref': prior['paid_model_ref'], 'model_update_ref': str(previous / 'role_return.json'),
        'source_pool_ref': prior['full_source_pool_ref'], 'costs_ref': str(previous / 'closed_source_costs.json'),
        'completed_reservation_ref': prior['completed_reservation_ref'], 'authorization_ref': prior['authorization_ref'],
        'observation_transfer': {'from_completed_source_hold': hold, 'to_one_development_probe': 2,
            'source_hold_after_reservation': hold - 2, 'abc_hold_preserved': ledger['abc_remaining_observations'],
            'global_limit': total, 'unused_rows_return_to_source_at_close': True},
        'coverage_cap': coverage_cap, 'max_schemas': structure_reg['max_candidates'],
        'per_assignment_compute': per_fit, 'planning_cap': planning_cap,
        'readout_limit': prior['readout_limit'], 'paid_prefix_filter_limit': prior['paid_prefix_filter_limit'],
        'measurement_cost_per_row': prior['measurement_cost_per_row'], 'measurement_limit': 2 * prior['measurement_cost_per_row'],
        'compute_limit': maximum, 'model_calls': 0,
        'rule': 'First eligible pair in native order from untouched compatible paid anchors. Both assignments '
                'must be novel relative to the previous response searches and have the two surviving owners. '
                'Hold f, q and u fixed; exchange only the two necessary position bindings. Fit each on the '
                'original training trajectories with the same parent, cap and selection rule. Retain both '
                'owner labels even for identical programs. Compare frozen responses on original working rows '
                'and report all four paid corners. After both programs are delivered, fund one role square '
                'only if the two conditional foreign-effect forecasts are nonempty and disjoint, the original '
                'background and moved corner are reusable, and at most two corners are missing. Stop on any outcome.'})
    freeze_position_runtime(output, previous, extra=(Path(__file__),))


def paid_square(previous, batches):
    purchase = load(previous / 'purchase.json')
    square, added = [], []
    for i, index in enumerate(purchase['paid_batch_indices']):
        if index is None:
            batch = load(previous / f'corner_{i}.json')
            added.append(batch)
        else:
            batch = batches[index]
        square.extend(observations(batch['observations'][:1]))
    return square, added


def compare_owners(output, cfg, lab, state, scene, previous_schemas, budget):
    previous = Path(cfg['previous'])
    last = load(previous / 'receipt.json')
    measurement = load(cfg['model_update_ref'])
    prefix = scene['object_model']['schema_alternatives']['schemas']
    model = apply_position_experiment(scene['object_model'], measurement)
    spec = replace(scene['spec'], context={**scene['spec'].context, 'object_model': model})
    # Keep every old source addressable. Only unsearched compatible anchors get
    # a new fit, allowing the original training split and parent score to be reused.
    anchors = [prefix[i] for i in last['eligible_paid_prefix_indices'] if prefix[i] not in previous_schemas]
    enumeration_work = SearchBudget(cfg['coverage_cap'])
    try:
        enumeration = enumerate_object_schemas(model, enumeration_work, max_schemas=cfg['max_schemas'],
                                               preferred_schemas=anchors)
    finally:
        budget.spend(enumeration_work.spent)
    save(output / 'assignments.json', enumeration)
    schemas = enumeration['schemas']
    coverage = enumeration.get('position_owner_coverage', {})
    pair = next((pair for pair in coverage.get('pairs', [])
                 if set(pair['owners']) == set(measurement['retained_owners'])
                 and all(schemas[i] not in previous_schemas for i in pair['indices'])), None)
    if pair is None or coverage['field'] != measurement['field']:
        return {'status': 'no_novel_matched_owner_pair'}
    save(output / 'chosen_pair.json', pair)
    ancestor = load(Path(cfg['original_arm_source']) / 'registration.json')
    agent = CommonObjectResearchAgent(config(cfg['seed'], ancestor['research_config']),
        library=program_knowledge(Path(ancestor['knowledge_root']), ancestor['knowledge_repair_ref']),
        workspace=output, organization=cfg['arm'])
    parent = scene['selected']
    fallback = initial_prediction(scene['training'], spec.risk_quantile)
    frame = dict(spec=spec, training=scene['training'], comparison=scene['validation'],
        transitions=scene['transitions'], assessed=[parent], independent_validation=True,
        comparison_complete=True, native_full_pool_comparison=False, fallback=fallback, uncompared=[])
    actions = []
    compared = agent.compare_position_owner_responses(frame, enumeration, pair, budget, actions,
        per_assignment_compute=cfg['per_assignment_compute'], response_parent=parent)
    entries = compared['new_claims']
    save(output / 'responses.json', [encode_entry(item) for item in entries])
    winner = compared['selected']
    review = {'status': compared['status'], 'paired_response_complete': compared['paired_response_complete'],
        'owner_response_fits': compared['owner_response_fits'],
        'comparison_predictions': compared['comparison_predictions'], 'parent_working_loss': parent['validation_loss'],
        'selected_candidate_id': winner['candidate'].candidate_id, 'selected_working_loss': winner['validation_loss'],
        'comparison_scope': 'saved winner and the new owner fits on the same exposed 24 rows',
        'search_compute_cost': compared['search_compute_cost'], 'comparison_compute_cost': compared['comparison_compute_cost'],
        'actions': actions}
    save(output / 'response_comparison.json', review)
    summary = {'paired_response_complete': review['paired_response_complete'], 'new_response_fits': sum(item['fit_count'] for item in review['owner_response_fits']),
               'delivered_response_programs': len(entries),
               'selected_working_loss': review['selected_working_loss'], 'comparison_ref': str(output / 'response_comparison.json')}
    if not review['paired_response_complete']:
        return {**summary, 'status': 'owner_response_comparison_incomplete'}

    corners, added = paid_square(previous, state['batches'])
    corner_scores, cache = [], {}
    for entry in [parent, *entries]:
        program = entry['candidate'].program
        if program not in cache:
            if not budget.spend(len(corners)):
                raise ValueError('Four paid response rows exceed the registered allowance')
            values = [predict(program, row, fallback) for row in corners]
            cache[program] = {'predictions': values, 'loss': mean(
                point_loss(value, row.target, spec.risk_quantile) for value, row in zip(values, corners))}
        corner_scores.append({'candidate_id': entry['candidate'].candidate_id, **cache[program]})
    save(output / 'paid_corner_responses.json', {'source': str(previous), 'includes_one_training_reset': True,
        'targets': [row.target for row in corners], 'responses': corner_scores})
    batches = [*state['batches'], *added]
    records = paid_object_records(lab, batches, state['selected_channels'])
    planner = SearchBudget(cfg['planning_cap'])
    try:
        plan = plan_position_owner_followup(spec, model, records, planner,
            previous_plan=load(previous / 'plan.json'), previous_result=measurement,
            schemas=[schemas[i] for i in pair['indices']], remaining_rows=2)
    finally:
        budget.spend(planner.spent)
    result = execute_position_plan(output, cfg, lab, batches, state['selected_channels'], model, prefix,
                                   plan, budget, probe_prefix='remaining_owner_corner',
                                   prior_prefix_indices=last['eligible_paid_prefix_indices'])
    return {**summary, **result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'run'))
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'register':
        register(args.previous.resolve(), args.output.resolve())
    else:
        run(args.output.resolve())
