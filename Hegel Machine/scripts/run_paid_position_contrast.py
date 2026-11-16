"""Register and finish one reset position experiment on an interrupted, paid scene."""
import argparse
from dataclasses import asdict
from pathlib import Path

from scripts.continue_interrupted_structure import restore
from scripts.run_learned_structure_successor import freeze_runtime
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.experiment_runtime import ExecutionTiming, python_execution_deadline
from hegel_machine.cognition.models import Probe, SearchBudget
from hegel_machine.cognition.object_induction import apply_position_experiment, assess_position_experiment
from hegel_machine.cognition.object_induction_continuation import paid_object_records
from hegel_machine.cognition.object_interventions import plan_paid_position_contrast


def source_inputs(previous):
    parent = load(previous / 'registration.json')
    registered, state, scene = restore(Path(parent['source']), parent['seed'], parent['arm'])
    settings = next(t for t in registered['tasks'] if t['seed'] == parent['seed'])
    return settings, state, scene, load(previous / 'competition.json')['schemas']


def register(previous, output):
    ledger = load(previous / 'closed_source_costs.json')
    source = Path(ledger['complete_relation_allocation_ref']['file']).parent
    allocation, closed = load(source / 'registration.json'), load(source / 'receipt.json')
    limits = load(allocation['budget_protocol_ref']['file'])['authorized_limits']
    hold = allocation['maximum_observations'] - closed['observation_cost']
    rows_limit = 3
    if (closed['status'] != 'confirmed' or hold < rows_limit or
            hold + ledger['abc_remaining_observations'] + ledger['cumulative_protocol_observations']
            != limits['observation_cost']):
        raise ValueError('No verified unused reservation from the completed relation unit')
    settings, state, scene, schemas = source_inputs(previous)
    model = scene['object_model']
    with AnonymousObjectLab(**settings) as lab:
        fine_price = sum(c.cost_per_row for c in lab.channels if c.name in state['selected_channels'])
    controls = len(scene['spec'].feature_bounds)
    positions, quantities = len(model['position_candidates']), len(model['quantity_controls'])
    # Upper bounds for selecting a reset/assignment, checking all old reset
    # controls once, and measuring the response of the four isolated corners.
    plan_limit = len(scene['training']) * (1 + positions) + positions * quantities * (
        1 + max(quantities, len(schemas)) * quantities)
    match_limit = len(state['batches']) * (controls + 1)
    readout_limit = 4 * (controls + 1) + 4 * len(model['raw_sensor_features']) + 3 * quantities
    prefix = model['schema_alternatives']['schemas']
    reuse_limit = sum(len(schema['fields']) for schema in prefix)
    cap = plan_limit + match_limit + readout_limit + reuse_limit
    if cap > ledger['complete_relation_unit_remaining']:
        raise ValueError('The parent development account cannot cover the complete experiment')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', {
        'previous': str(previous), 'paid_model_ref': scene['references']['object_model'],
        'assignments_ref': {'file': str(previous / 'competition.json'), 'json_path': ['schemas']},
        'old_response_ref': scene['references']['parent'],
        'full_source_pool_ref': load(previous / 'registration.json')['source_pool_ref'],
        'costs_ref': str(previous / 'closed_source_costs.json'),
        'completed_reservation_ref': str(source / 'receipt.json'),
        'authorization_ref': allocation['budget_protocol_ref'],
        'observation_transfer': {'from_completed_source_hold': hold, 'to_one_development_probe': rows_limit,
            'source_hold_after_reservation': hold - rows_limit, 'abc_hold_preserved': ledger['abc_remaining_observations'],
            'global_limit': limits['observation_cost'], 'unused_rows_return_to_source_at_close': True},
        'plan_limit': plan_limit, 'match_limit': match_limit, 'readout_limit': readout_limit,
        'paid_prefix_filter_limit': reuse_limit,
        'compute_limit': cap, 'measurement_cost_per_row': fine_price,
        'measurement_limit': rows_limit * fine_price, 'model_calls': 0,
        'rule': 'Plan on original training resets and four actually searched assignments. Freeze conditional '
                'owner predictions and four controls before purchasing any missing reset. Stop after one '
                'square including flat, contradictory or instrument-mismatch outcomes. For a narrowed '
                'domain, filter the saved assignment prefix without a new search. No fit or final panel.'})
    freeze_position_runtime(output, previous)


def freeze_position_runtime(output, previous, extra=()):
    freeze_runtime(output, previous)
    for file in (Path(__file__), Path(__file__).with_name('continue_interrupted_structure.py'), *extra):
        name = Path('scripts') / file.name
        base = previous / 'runtime' / name
        write_text_incrementally(output / 'runtime' / name, file.read_text(), base=base if base.exists() else None)


def experiment(output, cfg, lab, state, scene, schemas, budget):
    channels = state['selected_channels']
    training = [state['batches'][ref['json_path'][-1]] for ref in scene['references']['training']]
    records = paid_object_records(lab, training, channels)
    row_cap = cfg['observation_transfer']['to_one_development_probe']
    plan = plan_paid_position_contrast(scene['spec'], scene['object_model'], records, budget,
                                       schemas=schemas, remaining_rows=row_cap)
    return execute_position_plan(output, cfg, lab, state['batches'], channels, scene['object_model'],
                                 scene['object_model']['schema_alternatives']['schemas'], plan, budget)


def execute_position_plan(output, cfg, lab, batches, channels, model, prefix, plan, budget,
                          probe_prefix='position_reset_corner', prior_prefix_indices=None):
    row_cap = cfg['observation_transfer']['to_one_development_probe']
    save(output / 'plan.json', plan)
    if plan['status'] not in ('proposed_position_experiment', 'complete_paid_position_experiment'):
        return {'status': plan['status']}
    costs_before = budget.spent
    # Query exact paid reset controls, including the old working trajectory;
    # never use its response for selection of the experiment.
    if not budget.spend(len(batches) * (1 + len(plan['corners'][0]))):
        raise ValueError('The complete reset matching step was not funded')
    controls = list(plan['corners'][0])
    old_resets = {}
    named_resets = {}
    for i, b in enumerate(batches):
        if b['channels'] == channels:
            point = tuple(b['probe']['sequence'][0][c] for c in controls)
            old_resets.setdefault(point, i)
            named_resets[(point, b['probe']['probe_id'])] = i
    refs = plan.get('paid_corner_refs', [plan['baseline_ref'], None, None, None])
    indices = []
    for corner, ref in zip(plan['corners'], refs, strict=True):
        point = tuple(corner[c] for c in controls)
        indices.append(named_resets[(point, ref['probe_id'])] if ref is not None else old_resets.get(point))
    missing = [i for i, index in enumerate(indices) if index is None]
    readout_cost = cfg['readout_limit']
    if len(missing) > row_cap or budget.remaining < readout_cost + cfg['paid_prefix_filter_limit']:
        raise ValueError('The entire purchase and readout must be affordable')
    save(output / 'purchase.json', {'paid_batch_indices': indices, 'missing_corners': missing,
        'new_observations': len(missing), 'measurement_cost': len(missing) * cfg['measurement_cost_per_row'],
        'match_compute': budget.spent - costs_before, 'readout_compute_quote': readout_cost})
    corners = []
    for i, index in enumerate(indices):
        if index is None:
            probe = Probe(f'{probe_prefix}_{i}', (plan['corners'][i],))
            acquired = lab.acquire(probe, channels)
            batch = {'channels': channels, 'probe': asdict(probe),
                     'observations': [asdict(row) for row in acquired.observations],
                     'transitions': acquired.transitions}
            save(output / f'corner_{i}.json', batch)
        else:
            b = batches[index]
            batch = {**b, 'probe': {**b['probe'], 'sequence': b['probe']['sequence'][:1]},
                     'observations': b['observations'][:1], 'transitions': b['transitions'][:1]}
        corners.extend(paid_object_records(lab, [batch], channels))
    result = assess_position_experiment(plan, model, corners, budget)
    updated = apply_position_experiment(model, result)
    save(output / 'role_return.json', result)
    eligible = None
    if result['status'] == 'position_domain_narrowed':
        if not budget.spend(cfg['paid_prefix_filter_limit']):
            raise ValueError('Insufficient paid-prefix comparison reserve')
        domains = updated['position_candidates']
        eligible = [i for i, schema in enumerate(prefix) if all(
            group['q'] in domains[group['x']]['quantities'] for group in schema['fields'].values())]
    elif result['status'] == 'position_domain_unchanged':
        eligible = list(range(len(prefix))) if prior_prefix_indices is None else list(prior_prefix_indices)
    return {'status': result['status'], 'parent_model_ref': cfg['paid_model_ref'],
            'position_update_ref': str(output / 'role_return.json'),
            'paid_prefix_size': len(prefix), 'eligible_paid_prefix_indices': eligible,
            'unresolved_roles': updated['unresolved'], 'schema_identified': updated['complete']}


def run(output):
    if Path(__file__).resolve().parents[1] != output / 'runtime':
        raise ValueError('Run the frozen entry')
    cfg = load(output / 'registration.json')
    prior = load(cfg['costs_ref'])
    settings, state, scene, schemas = source_inputs(Path(cfg.get('structure_source', cfg['previous'])))
    timing = ExecutionTiming(('experiment',))
    budget = SearchBudget(cfg['compute_limit'])
    (output / 'attempt').mkdir()  # Never repeat an interrupted purchase.
    with AnonymousObjectLab(**settings) as lab:
        try:
            with timing.phase('experiment'), python_execution_deadline(1800, activity=True):
                research = experiment
                if cfg.get('kind') == 'remaining_position_owners':
                    from scripts.compare_remaining_position_owners import compare_owners
                    research = compare_owners
                result = research(output, cfg, lab, state, scene, schemas, budget)
        except BaseException as exc:
            save(output / 'failure.json', {'error': repr(exc), 'compute_cost': budget.spent,
                                          'laboratory_costs': lab.costs})
            raise
        costs = lab.costs
    receipt = {**result, 'laboratory_costs': costs, 'compute_cost': budget.spent,
               'model_calls': 0, 'new_response_fits': result.get('new_response_fits', 0), 'independent_confirmation': False,
               'execution_timing': timing.result()}
    save(output / 'receipt.json', receipt)
    new_rows = costs['observation_cost']
    save(output / 'closed_source_costs.json', {**prior, 'status': cfg.get('kind', 'one_reset_experiment') + '_closed',
         'previous_source_costs_ref': {'file': cfg['costs_ref']}, 'receipt_ref': {'file': str(output / 'receipt.json')},
         'new_compute_interval': [budget.spent, budget.spent], 'new_compute_cost': budget.spent,
         'new_observations': new_rows, 'new_measurement_cost': costs['measurement_cost'],
         'cumulative_protocol_compute_interval': [v + budget.spent for v in prior['cumulative_protocol_compute_interval']],
         'cumulative_protocol_observations': prior['cumulative_protocol_observations'] + new_rows,
         'complete_relation_unit_remaining': prior['complete_relation_unit_remaining'] - budget.spent,
         'completed_source_hold_remaining_observations': cfg['observation_transfer']['from_completed_source_hold'] - new_rows,
         'observation_transfer_ref': {'file': str(output / 'registration.json'), 'json_path': ['observation_transfer']},
         'other_accounts_inherited_from': cfg['costs_ref']})
    print(receipt['status'], budget.spent, new_rows)


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
