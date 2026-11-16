"""Form one position-experiment method from its two paid instrument returns."""
import argparse
from pathlib import Path

from scripts.run_learned_structure_successor import freeze_runtime
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.common_object_research import CommonObjectResearchAgent, reconstruct_partial_role_source
from hegel_machine.cognition.experiment_runtime import ExecutionTiming
from hegel_machine.cognition.models import ResearchConfig, SearchBudget


def register(unit, output):
    previous = Path(load(unit / 'registration.json')['previous'])
    ledger = load(unit / 'closed_source_costs.json')
    stages = [{'position_experiment': {'file': str(path / 'plan.json')},
               'completed_relation': {'file': str(path / 'role_return.json')}} for path in (previous, unit)]
    # One derivation for the frozen method; the two ordinary reconstruction arms
    # must independently derive the same body before a successor is admitted.
    per_derivation = sum(len(load(s['position_experiment']['file'])['candidate_owners'])
                         + len(load(s['completed_relation']['file'])['corner_refs']) for s in stages)
    if 3 * per_derivation > ledger['complete_relation_unit_remaining']:
        raise ValueError('The formation and both source-side reconstructions must fit the development account')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'raw_source.json', {'previous': {'position_history_refs': stages},
        'source_ref': {'file': str(unit / 'receipt.json')}, 'model_id': 'position_method:' + unit.name})
    save(output / 'registration.json', {'paid_unit': str(unit), 'compute_limit': 3 * per_derivation,
        'costs_ref': {'file': str(unit / 'closed_source_costs.json')}, 'observation_limit': 0,
        'physical_sources': 1, 'release': 'Same executable method from both ordinary raw-source readers and frozen uptake.'})
    freeze_runtime(output, unit)
    write_text_incrementally(output / 'runtime/scripts' / Path(__file__).name, Path(__file__).read_text())


def form(output):
    if Path(__file__).resolve().parents[1] != output / 'runtime':
        raise ValueError('Run the registered method author')
    cfg = load(output / 'registration.json')
    old = load(cfg['costs_ref']['file'])
    budget = SearchBudget(cfg['compute_limit'])
    raw = load(output / 'raw_source.json')
    (output / 'attempt').mkdir()
    clock = ExecutionTiming(('form_and_reconstruct',))
    receipt = {'status': 'method_not_formed', 'same_body_reconstructed': False, 'arms': {},
               'source_paid_rows_reused': True, 'new_observations': 0, 'model_calls': 0}
    try:
        with clock.phase('form_and_reconstruct'):
            formed = reconstruct_partial_role_source(**raw, budget=budget)
            method = formed['method']
            receipt.update(formation_compute=formed['compute_cost'], formation_status=formed['assessment']['status'])
            if method is not None:
                save(output / 'method.json', method)
                for arm in ('archive', 'fixed_pr2', 'pr3'):
                    work = CommonObjectResearchAgent(ResearchConfig(), organization=arm, workspace=output / arm,
                        raw_sources=[{'file': str(output / 'raw_source.json')}],
                        learned_methods=[method] if arm == 'pr3' else [])
                    start = budget.spent
                    available = [*work.learned_methods, *work._reconstruct_sources(budget, [])]
                    receipt['arms'][arm] = {'method_matches': available == [method], 'compute': budget.spent-start,
                                            'raw_reconstructions': len(work.source_reconstruction_receipts)}
                receipt['same_body_reconstructed'] = all(arm['method_matches'] for arm in receipt['arms'].values())
                receipt['status'] = 'frozen' if receipt['same_body_reconstructed'] else 'source_reconstruction_mismatch'
    except Exception as error:
        receipt.update(status='formation_failed', error=repr(error))
        raise
    finally:
        receipt.update(compute_cost=budget.spent, execution_timing=clock.result())
        save(output / 'receipt.json', receipt)
        save(output / 'closed_source_costs.json', {**old,'status':'position_method_formation_closed',
             'previous_source_costs_ref': cfg['costs_ref'], 'receipt_ref': {'file': str(output / 'receipt.json')},
             'new_compute_interval': [budget.spent, budget.spent], 'new_compute_cost': budget.spent,
             'new_observations': 0, 'new_measurement_cost': 0, 'new_model_calls': 0,
             'complete_relation_unit_remaining': old['complete_relation_unit_remaining'] - budget.spent,
             'cumulative_protocol_compute_interval': [v + budget.spent for v in old['cumulative_protocol_compute_interval']]})
    print(receipt['status'], receipt['compute_cost'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['register', 'form'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'register':
        register(args.source.resolve(), args.output.resolve())
    else:
        form(args.output.resolve())
