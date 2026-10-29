#!/usr/bin/env python3
"""Consume a failed successor's paid rows before any new confirmation."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from scripts.run_learned_structure_successor import freeze_runtime, load_successor_scene
from scripts.run_native_decision_pair import load_prepared, ref
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.experiment_runtime import ExecutionTiming, python_execution_deadline
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.object_continuation import decode_entry, encode_entry, observations
from hegel_machine.cognition.parallel_prediction import predict_rows


ROOT = Path(__file__).resolve().parents[1]


def restore(source):
    """Restore unique pool occurrences and already exposed trajectories by reference."""
    source = Path(source)
    scene = load_successor_scene(source)
    agent, frame, _ = load_prepared(source, scene=scene)
    registration = load(source / 'registration.json')
    pool, refs, working = [], [], {}
    parent = None
    for arm in registration['operations']:
        compared = load(source / arm / 'comparison.json')
        if not compared['full_comparison']:
            raise ValueError('The retained pool was not completely compared')
        for row in compared['scores']:
            location = row['candidate_ref']
            if location in refs:
                if arm == 'challenger' and location == compared['selected_ref']:
                    parent = pool[refs.index(location)]
                continue
            entry = decode_entry(read_method_reference(location, source))
            program = entry['candidate'].program
            pool.append(entry)
            refs.append(location)
            working[program] = row['loss']
            if arm == 'challenger' and location == compared['selected_ref']:
                parent = entry
    if parent is None:
        raise ValueError('The retained structural parent is not in the complete pool')
    parent['fallback'] = frame['fallback']
    confirmation = load(source / 'confirmation/receipt.json')
    batch = load(source / 'confirmation/observations.json')['batch']
    exposed = observations(batch['observations'])
    frozen = load(source / 'confirmation/frozen_predictors.json')
    exposed_scores = {Program.from_dict(item['program']): confirmation['arms'][arm]['mse']
                      for arm, item in frozen.items()}
    if any(item['fallback'] != frame['fallback'] for item in frozen.values()):
        raise ValueError('Retained scores use a different prediction fallback')
    # Preserve entire histories. The old confirmation is now development evidence.
    paid = [*scene['training'], *scene['validation'], *exposed]
    transitions = list(scene['transitions'])
    for location in scene['references']['validation']:
        transitions.extend(read_method_reference(location, source).get('transitions', []))
    transitions.extend(batch.get('transitions', []))
    return dict(agent=agent, frame=frame, scene=scene, parent=parent, pool=pool, refs=refs,
        working=working, exposed=exposed, exposed_scores=exposed_scores, paid=paid,
        transitions=transitions, registration=registration)


def paid_pool_scores(scene, budget):
    """Reuse original training/working losses and score only missing exposed rows."""
    nt, nv, ne = len(scene['scene']['training']), len(scene['scene']['validation']), len(scene['exposed'])
    scores, exposed = {}, dict(scene['exposed_scores'])
    before = budget.spent
    for entry in scene['pool']:
        program = entry['candidate'].program
        if program in scores:
            continue
        if program not in exposed:
            if not budget.spend(ne):
                raise RuntimeError('The complete retained comparison is not funded')
            exposed[program] = loss(program, scene['exposed'], scene['frame']['fallback'], .5)
        scores[program] = (nt * entry['training_loss'] + nv * scene['working'][program]
                          + ne * exposed[program]) / (nt + nv + ne)
    return scores, exposed, budget.spent - before


def register(source, output):
    source, output = Path(source), Path(output)
    old, receipt = load(source / 'registration.json'), load(source / 'receipt.json')
    total = old['common_compute_limit'] + len(old['operations']) * old['per_arm_total_compute_limit']
    paid = receipt['cumulative_costs']
    remaining = total - paid['compute_cost']
    if remaining <= 0:
        raise ValueError('No unused allowance remains in the original unit')
    output.mkdir(parents=True, exist_ok=False)
    registration = dict(source=str(source), source_task_ref=old['source_task_ref'],
        previous_costs=paid, total_compute_limit=total, remaining_compute=remaining,
        source_cost_refs=[ref(source / 'receipt.json'), *receipt['source_cost_refs']],
        scope='one paid development scope/boundary revision; not a new method pair',
        operations=['retain_complete_pool', 'revise_object_scope', 'revise_paid_structure'],
        scope_fit_compute_limit=12000, active_wall_seconds=1800,
        threshold=old['quality_threshold'], confirmation_rows=old['confirmation_rows'],
        confirmation_seed=old['confirmation_seed'] + 17017,
        confirmation_gate='changed complete-pool winner; every paid squared residual <= original threshold',
        confirmation_comparator='unchanged structural winner from the previous independent failure',
        development_role='all original training, working and exposed confirmation rows',
        stop_rule='one bounded revision and at most one gated fresh readout; preserve any failure',
        model_calls=0)
    save(output / 'registration.json', registration)
    freeze_runtime(output, source)
    name = Path('scripts/revise_learned_structure_successor.py')
    base = source / 'runtime' / name
    write_text_incrementally(output / 'runtime' / name, (ROOT / name).read_text(),
                             base=base if base.exists() else None)
    return registration


def run(output):
    output = Path(output)
    if ROOT != (output / 'runtime').resolve():
        raise ValueError('Execute the newly frozen current entry')
    if (output / 'started.json').exists():
        raise ValueError('Keep the prior attempt and its paid result')
    cfg = load(output / 'registration.json')
    save(output / 'started.json', {'status': 'started', 'source': cfg['source']})
    timing = ExecutionTiming(('restore', 'paid_comparison', 'scope_revision', 'boundary_revision',
                              'residual_readout', 'confirmation', 'save'))
    budget = SearchBudget(cfg['remaining_compute'])
    actions, observation_cost, measurement_cost = [], 0, 0
    try:
        with timing.phase('restore'):
            scene = restore(cfg['source'])
        agent, spec, paid = scene['agent'], scene['scene']['spec'], scene['paid']
        original_parent = deepcopy(scene['parent'])
        fallback = scene['frame']['fallback']
        with python_execution_deadline(cfg['active_wall_seconds'], activity=True):
            with timing.phase('paid_comparison'):
                scores, exposed, additional_scoring = paid_pool_scores(scene, budget)
            # Keep enough for an actual selected residual check and both frozen predictors.
            budget.reserve = len(paid) + 2 * cfg['confirmation_rows']
            with timing.phase('scope_revision'):
                local = SearchBudget(min(cfg['scope_fit_compute_limit'],
                    max(0, budget.remaining - agent.config.max_candidates * len(paid))))
                scope = agent._object_scope_claims(spec, paid, local, actions)
                budget.spend(local.spent)
                assessed = agent._assess(scope, paid, budget, fallback, spec.risk_quantile)
                if len(assessed) != len(scope):
                    raise RuntimeError('Object-scope comparison is incomplete')
                scores.update({entry['candidate'].program: entry['validation_loss'] for entry in assessed})
                save(output / 'scope_claims.json', [encode_entry(entry) for entry in assessed])
            with timing.phase('boundary_revision'):
                result = agent._revise_paid_structure(spec, scene['parent'], paid, budget, actions,
                    finalists=[*scene['pool'], *assessed], transitions=scene['transitions'], paid_scores=scores)
            budget.release_reserve()
            selected = result['selected']
            with timing.phase('residual_readout'):
                values = result.get('selected_predictions')
                if values is None and result['full_comparison']:
                    if not budget.spend(len(paid)):
                        raise RuntimeError('The selected residual readout is incomplete')
                    values = predict_rows(selected['candidate'].program, paid)
                errors = [(value - row.target) ** 2 for value, row in zip(values, paid)] if values else None
            changed = selected['candidate'].program != original_parent['candidate'].program
            gate = bool(result['full_comparison'] and changed and errors
                        and max(errors) <= cfg['threshold'])
            with timing.phase('save'):
                save(output / 'new_claims.json', [encode_entry(entry) for entry in result['new_claims']])
                save(output / 'selected_entry.json', encode_entry(selected))
                all_refs = [*scene['refs'], *[ref(output / 'scope_claims.json', i) for i in range(len(scope))],
                    *[ref(output / 'new_claims.json', i) for i in range(len(result['new_claims']))]]
                all_entries = [*scene['pool'], *assessed, *result['new_claims']]
                scored = {entry['candidate'].program: entry['validation_loss']
                          for entry in result['compared_claims']}
                save(output / 'selection.json', dict(status=result['status'],
                    full_comparison=result['full_comparison'], actions=actions,
                    retained_pool_refs=scene['refs'], source_rows_ref=ref(Path(cfg['source']) / 'prepared_state.json',
                        'source_references'), exposed_rows_ref=ref(Path(cfg['source']) / 'confirmation/observations.json'),
                    scores=[{'candidate_ref': location, 'loss': scored.get(entry['candidate'].program)}
                            for entry, location in zip(all_entries, all_refs)],
                    selected_ref=ref(output / 'selected_entry.json'), selected_paid_errors=errors,
                    confirmation_admitted=gate))
            confirmation = None
            if gate:
                with timing.phase('confirmation'):
                    save(output / 'frozen_predictors.json', dict(
                        selected=selected['candidate'].program.to_dict(),
                        incumbent=original_parent['candidate'].program.to_dict(), fallback=fallback))
                    if not budget.spend(2 * cfg['confirmation_rows']):
                        raise RuntimeError('The gated confirmation was not funded')
                    with AnonymousObjectLab(**scene['registration']['task']) as laboratory:
                        probe = laboratory.random_probe(cfg['confirmation_seed'], length=cfg['confirmation_rows'],
                            name='scope_boundary_revision_confirmation')
                        batch = laboratory.acquire(probe, ('fine',))
                        observation_cost, measurement_cost = laboratory.observation_cost, laboratory.measurement_cost
                        save(output / 'confirmation_observations.json', {'probe': asdict(probe), 'batch': asdict(batch)})
                        confirmation = {'rows': len(batch.observations), 'threshold': cfg['threshold']}
                        for name, entry in [('selected', selected), ('incumbent', original_parent)]:
                            predictions = predict_rows(entry['candidate'].program, batch.observations)
                            mse = mean((value - row.target) ** 2 for value, row in zip(predictions, batch.observations))
                            confirmation[name] = dict(mse=mse, predictions=predictions, passed=mse <= cfg['threshold'])
            receipt = dict(status='confirmed' if confirmation and confirmation['selected']['passed'] else 'unresolved',
                selection_status=result['status'], full_comparison=result['full_comparison'],
                retained_pool=len(scene['pool']), scope_candidates=len(scope), boundary_candidates=len(result['new_claims']),
                complete_pool=len(result['claims']), paid_rows=len(paid),
                old_parent_paid_mse=scores[original_parent['candidate'].program],
                selected_paid_mse=mean(errors) if errors else None,
                max_paid_squared_error=max(errors) if errors else None,
                old_exposed_mse=mean(errors[-len(scene['exposed']):]) if errors else None,
                additional_old_scoring=additional_scoring, reused_old_training_scores=len(scene['pool']),
                reused_old_working_scores=len(scene['pool']), reused_old_exposed_scores=len(scene['exposed_scores']),
                confirmation_admitted=gate, confirmation=confirmation, compute_cost=budget.spent,
                cumulative_compute=cfg['previous_costs']['compute_cost'] + budget.spent,
                observation_cost=observation_cost, measurement_cost=measurement_cost,
                source_cost_refs=cfg['source_cost_refs'], model_calls=0,
                method_condition_learned=False, execution_timing=timing.result())
            save(output / 'receipt.json', receipt)
            return receipt
    except BaseException as error:
        save(output / 'failure.json', dict(error=repr(error), compute_known=budget.spent,
            compute_upper_bound=cfg['remaining_compute'], observation_cost=observation_cost,
            measurement_cost=measurement_cost, actions=actions, execution_timing=timing.result()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(register(args.source, args.output) if args.mode == 'prepare' else run(args.output.resolve()))
