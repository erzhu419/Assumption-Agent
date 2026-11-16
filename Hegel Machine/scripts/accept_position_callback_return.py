"""Replay the first exposed 603 callback and reuse its already answered Archive question."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from scripts.run_learned_structure_successor import freeze_runtime
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.experiment_runtime import ExecutionTiming
from hegel_machine.cognition.object_dependency_methods import position_method_source, propose_partial_role_experiment


def register(closed, costs, output):
    seed = load(closed / 'receipt.json')['target_seed']
    if seed != 106110603:
        raise ValueError('Expected the registered exposed callback')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', {'closed_study': str(closed), 'costs': str(costs),
        'scope': 'exposed decision repair; both Common arms share one paid position question',
        'state': 'after the first acquired corner in consumption_028', 'new_observations': 0})
    freeze_runtime(output, closed)
    write_text_incrementally(output / 'runtime/scripts' / Path(__file__).name, Path(__file__).read_text())


def replay(output):
    if Path(__file__).resolve().parents[1] != output / 'runtime':
        raise ValueError('Use the frozen repair')
    cfg = load(output / 'registration.json')
    prior = load(cfg['costs'])
    scene = Path(cfg['closed_study']) / 'runs/106110603'
    directories = [scene / arm / 'common/object_question_000' for arm in ('pr3', 'fixed_pr2')]
    d, other = directories
    values = {key: load(d / (key + '.json')) for key in
              ('spec', 'model', 'selected_values', 'consumption_028', 'paid_records')}
    if any(load(other / (key + '.json')) != val for key, val in values.items()):
        raise ValueError('The Common arms do not share this invocation')
    sources = load(d / 'sources.json')
    selected = position_method_source(sources)
    rebuilt = position_method_source(load(other / 'sources.json'))
    # Common projects frozen sources onto the fields consumed by the planner.
    if not selected or not rebuilt or any(selected[k] != rebuilt[k] for k in ('position_method', 'model_id', 'source_ref')):
        raise ValueError('The two arms did not have the same position method available')
    previous = values['consumption_028']
    plan = previous['position_experiment']
    paid_ref = scene / 'archive/common/native_position_returns/return_000.json'
    archive = load(paid_ref)
    if plan != archive['position_experiment']:
        raise ValueError('The Archive answer addresses another question')
    records = values['paid_records']
    missing = plan['missing_corner_indices']
    first = next(i for i, row in enumerate(records) if row['position'] == 0 and not row['history']
                 and row['controls'] == plan['corners'][missing[0]])
    # Consume precisely the callback that originally repeated a purchase. The
    # remaining corners already have an identical paid answer in another arm.
    remaining_rows = len(missing) - 1
    expected = [plan['corners'][i] for i in missing[1:]]
    (output / 'attempt').mkdir()
    clock = ExecutionTiming(('callback',))
    with clock.phase('callback'):
        result = propose_partial_role_experiment(SimpleNamespace(**values['spec']), values['model'],
            records[:first+1], selected_values=values['selected_values'], sources=sources,
            remaining_rows=remaining_rows, previous=previous, compute_limit=prior['complete_relation_unit_remaining'])
    spent = result['compute_cost']
    actual = [a['sequence'] for a in result.get('actions', ())]
    advanced = actual == [[p] for p in expected] and not result.get('position_assessment')
    receipt = {'status': 'callback_advanced' if advanced else 'callback_unresolved',
        'prior_callback_refs': [str(p / 'consumption_028.json') for p in directories],
        'original_proposed_missing_corners': missing, 'rows_at_first_callback': first+1,
        'next_corner_indices': [plan['corners'].index(seq[0]) for seq in actual],
        'same_frozen_question_archive_return': str(paid_ref),
        'archive_retained_owners': archive['completed_relation']['retained_owners'],
        'scope': cfg['scope'], 'compute_cost': spent, 'new_observations': 0, 'new_model_calls': 0,
        'execution_timing': clock.result()}
    save(output / 'receipt.json', receipt)
    save(output / 'closed_source_costs.json', {**prior, 'status': 'paid_603_callback_repair_closed',
         'previous_source_costs_ref': {'file': cfg['costs']}, 'receipt_ref': {'file': str(output/'receipt.json')},
         'new_compute_interval': [spent, spent], 'new_compute_cost': spent, 'new_observation_interval': [0,0],
         'new_observations': 0, 'new_measurement_interval': [0,0], 'new_measurement_cost': 0, 'new_model_calls': 0,
         'cumulative_protocol_compute_interval': [n + spent for n in prior['cumulative_protocol_compute_interval']],
         'complete_relation_unit_remaining': prior['complete_relation_unit_remaining'] - spent})
    print(receipt['status'], receipt['next_corner_indices'], spent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'replay'))
    parser.add_argument('--study', type=Path)
    parser.add_argument('--costs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'register':
        register(args.study.resolve(), args.costs.resolve(), args.output.resolve())
    else:
        replay(args.output.resolve())
