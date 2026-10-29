#!/usr/bin/env python3
"""Resume full-response interval research using paid trajectories and calibrated controls."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from scripts.run_learned_structure_successor import freeze_runtime
from scripts.run_native_decision_pair import ref
from scripts.run_partial_role_parent import load, save, plain
from scripts.successor_interval_scene import (
    restore_interval_scene, restore_interval_result, score_retained, pair_rows,
    save_confirmed_knowledge)
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.checkpoint_storage import write_json_checkpoint, write_text_incrementally
from hegel_machine.cognition.competing_object_experiments import (
    plan_competing_object_probe, shorten_discriminating_probe, assess_competing_object_probe)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.experiment_runtime import ExecutionTiming, python_execution_deadline
from hegel_machine.cognition.instrument_models import rebind_instrument_model
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.object_continuation import encode_entry
from hegel_machine.cognition.parallel_prediction import predict_rows
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch


ROOT = Path(__file__).resolve().parents[1]


def register(source, output):
    source, output = Path(source), Path(output)
    old, receipt = load(source / 'registration.json'), load(source / 'receipt.json')
    original = load(Path(old['source']) / 'registration.json')
    if not receipt['full_comparison'] or not receipt['confirmation']:
        raise ValueError('This entry consumes a complete frozen readout')
    cfg = dict(source=str(source), source_task_ref=old['source_task_ref'],
        total_compute_limit=old['total_compute_limit'], previous_compute=receipt['cumulative_compute'],
        remaining_compute=old['total_compute_limit'] - receipt['cumulative_compute'],
        previous_observations=old['previous_costs']['observation_cost'] + receipt['observation_cost'],
        previous_measurement_cost=old['previous_costs']['measurement_cost'] + receipt['measurement_cost'],
        observation_limit=original['observation_limit'], measurement_limit=original['measurement_limit'],
        threshold=old['threshold'], confirmation_rows=old['confirmation_rows'],
        confirmation_seed=old['confirmation_seed'] + 17017,
        maximum_experiments=2 * original['research_config']['max_rounds'],
        maximum_calibration_rows=2 * original['research_config']['max_rounds'],
        active_wall_seconds=old['active_wall_seconds'],
        source_cost_refs=[ref(source / 'receipt.json'), *old['source_cost_refs']],
        scope='same successor; retained coefficients and window, evidence-driven relation intervals',
        stop_rule='no fit restart or new task; stop on unresolved calibration, no distinguishing plan, or exhausted allowance',
        confirmation_gate='complete retained-pool comparison, every paid residual supported, and resolved parameter precision or no substantive competition',
        model_calls=0)
    if cfg['remaining_compute'] <= 0:
        raise ValueError('The original allowance has been consumed')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', cfg)
    freeze_runtime(output, source)
    for filename in ('continue_successor_relation_intervals.py', 'successor_interval_scene.py',
                     'revise_learned_structure_successor.py'):
        name = Path('scripts') / filename
        base = source / 'runtime' / name
        write_text_incrementally(output / 'runtime' / name, (ROOT / name).read_text(),
                                 base=base if base.exists() else None)
    return cfg


def compare_pool(agent, entries, scores, fallback):
    """Select across all retained occurrences using their current paid scope."""
    compared = [{**entry, 'training_loss': scores[entry['candidate'].program],
        'validation_loss': scores[entry['candidate'].program], 'fallback': fallback,
        'selection_basis': 'all_acquired_evidence'} for entry in entries]
    selected = agent._best(compared)
    # Equivalent predictions can retain a smaller interval and later measured
    # evidence. Preserve old occurrences while returning the latest explanation.
    latest = next((item for item in reversed(compared)
        if item['candidate'].program == selected['candidate'].program
        and item.get('parameter_values')), None)
    return latest or selected


def retain_interval_evidence(research, selected, candidates, row_count, tolerance):
    """Keep a supported pool parameter without discarding the newly narrowed domain."""
    active = next((item for item in reversed(candidates)
        if item['candidate'].program == selected['candidate'].program), None)
    if active is not None:
        return active
    values = selected.get('parameter_values')
    if (values is None or selected['validation_loss'] > tolerance or
            set(values) != set(research.values) or not all(any(low <= values[item['symbol']] <= high
                for low, high in item['radius_intervals']) for item in research.parameters)):
        return None
    index = selected['parameter_index']
    symbol = research.parameters[index]['symbol']
    interval = next(domain for domain in research.parameters[index]['radius_intervals']
                    if domain[0] <= values[symbol] <= domain[1])
    return research._entry(values, interval, index, selected['validation_loss'], row_count, 0)


def extend_scores(scores, old_count, rows, budget, fallback):
    """Keep prior row scores; price only the newly acquired suffix for each program."""
    if not budget.can_spend(len(rows) * len(scores)):
        raise RuntimeError('Cannot consume the purchase across the complete retained pool')
    for program in scores:
        budget.spend(len(rows))
        value = loss(program, rows, fallback, .5)
        scores[program] = (old_count * scores[program] + len(rows) * value) / (old_count + len(rows))


def save_interval_candidates(output, pass_number, candidates, *, previous=None):
    """Keep complete entries while sharing unchanged evidence subtrees on disk."""
    references = []
    for index, entry in enumerate(candidates):
        path = Path(output) / f'interval_candidates_{pass_number:02d}' / f'{index:03d}.json'
        # Dict-root checkpoints use the existing JSON range-sharing writer.
        # Chaining adjacent entries also shares equal evidence within a pass.
        write_json_checkpoint(path, plain(encode_entry(entry)), base=previous)
        references.append(ref(path))
        previous = path
    return references, previous


def run(output):
    output = Path(output)
    if ROOT != (output / 'runtime').resolve():
        raise ValueError('Execute this newly frozen current runtime')
    if (output / 'started.json').exists():
        raise ValueError('Preserve the existing attempt')
    cfg = load(output / 'registration.json')
    save(output / 'started.json', {'source': cfg['source']})
    budget = SearchBudget(cfg['remaining_compute'])
    timing = ExecutionTiming(('restore', 'paid_comparison', 'interval_research', 'instruments',
                              'plan', 'acquire', 'confirmation', 'knowledge'))
    actions, purchased, pool_refs, new_entries = [], [], [], []
    candidate_base = None
    observation_cost, measurement_cost, confirmation = 0, 0, None
    try:
        with timing.phase('restore'):
            scene = restore_interval_scene(cfg['source'])
        agent, spec, paid, pool = scene['agent'], scene['spec'], scene['paid'], list(scene['pool'])
        pool_refs = list(scene['refs'])
        origin = deepcopy(scene['parent'])
        selected, fallback = origin, scene['parent']['fallback']
        with timing.phase('paid_comparison'):
            scores, old_scoring = score_retained(scene, budget)
        original_loss = scores[origin['candidate'].program]
        research = RelationIntervalResearch(origin['candidate'].program, agent.library.meta_priors,
            budget, structure=origin.get('structure'), research_program=origin.get('research_program'))
        experiments = [tuple(scene['exposed'])]
        paired_batches = list(scene['paired_batches'])
        calibration, conditioning_per_row, stopped = None, 0, 'not_started'
        passes, plans, calibration_rows, full_comparison = 0, 0, 0, True
        budget.reserve = 2 * cfg['confirmation_rows'] + len(paid)
        with AnonymousObjectLab(**scene['original_registration']['task']) as lab, \
                python_execution_deadline(cfg['active_wall_seconds'], activity=True):
            command_spec = lab.spec_for(())

            def acquire_calibration(channels, purpose, *, probe):
                nonlocal observation_cost, measurement_cost, calibration_rows
                count = probe.cost
                required_compute = count * (len(scores) + 4 * len(spec.feature_names)) + conditioning_per_row * (len(paid) + count)
                if (calibration_rows + count > cfg['maximum_calibration_rows'] or
                        budget.remaining < required_compute or
                        cfg['previous_observations'] + observation_cost + count + cfg['confirmation_rows'] > cfg['observation_limit'] or
                        cfg['previous_measurement_cost'] + measurement_cost + 12 * (count + cfg['confirmation_rows']) > cfg['measurement_limit']):
                    return None
                batch = lab.acquire(probe, channels)
                observation_cost, measurement_cost = lab.observation_cost, lab.measurement_cost
                location = output / 'observations' / f'{len(purchased):02d}.json'
                save(location, {'purpose': purpose, 'probe': asdict(probe), 'batch': asdict(batch)})
                purchased.append(ref(location))
                paired_batches.append(pair_rows(probe.sequence, batch.observations))
                extend_scores(scores, len(paid), batch.observations, budget, fallback)
                paid.extend(batch.observations)
                experiments.append(tuple(batch.observations))
                calibration_rows += count
                budget.reserve = 2 * cfg['confirmation_rows'] + len(paid)
                return batch.observations

            while True:
                before = budget.spent
                with timing.phase('interval_research'):
                    candidates = research.alternatives(paid, budget, tolerance=cfg['threshold'],
                                                        experiments=experiments)
                passes += 1
                spent = budget.spent - before
                conditioning_per_row = max(conditioning_per_row, (spent + len(paid) - 1) // len(paid))
                actions.append({'action': 'condition_relation_intervals', 'status': research.status,
                    'rows': len(paid), 'experiments': len(experiments), 'candidates': len(candidates),
                    'intervals': deepcopy(research.intervals), 'compute_cost': spent})
                if not candidates:
                    stopped = research.status
                    full_comparison = research.status not in {'compute_exhausted', 'refuted'}
                    break
                # These scores were just measured by the existing interval researcher.
                for entry in candidates:
                    pool.append(entry)
                    scores[entry['candidate'].program] = entry['validation_loss']
                selected = compare_pool(agent, pool, scores, fallback)
                # Adopt the interval answer only when it is the pool's actual choice.
                active = retain_interval_evidence(research, selected, candidates, len(paid), cfg['threshold'])
                if active is not None and not any(active is item for item in candidates):
                    candidates.append(active)
                    pool.append(active)
                candidate_refs, candidate_base = save_interval_candidates(
                    output, passes, candidates, previous=candidate_base)
                pool_refs.extend(candidate_refs)
                new_entries.extend(candidates)
                if active is None:
                    stopped = 'retained_pool_preferred'
                    break
                research.adopt(active)
                questions = research.questions(candidates)
                if not questions or all(item['radius_interval'][1] - item['radius_interval'][0] <= 1e-6
                                        for item in research.intervals):
                    stopped = 'resolved_to_precision' if questions else 'no_remaining_competition'
                    break
                if plans >= cfg['maximum_experiments']:
                    stopped = 'bounded_experiment_limit'
                    break
                required = set().union(*(entry['candidate'].program.required_features() for entry in candidates))
                if calibration is None:
                    with timing.phase('instruments'):
                        calibration = agent._fit_instruments(command_spec, spec, paired_batches,
                            budget, actions, required_fields=required)
                    save(output / 'instrument_models.json', agent.library.instrument_models)
                binding = agent._instrument_binding(spec, command_spec)
                if calibration is not None:
                    binding.update(rebind_instrument_model(calibration, spec))
                if (required - set(binding) - set(command_spec.feature_names) and calibration is not None
                        and calibration_rows < cfg['maximum_calibration_rows']):
                    before_rows = len(paid)
                    calibration = agent._distinguish_instruments(lab, calibration, ('fine',),
                        acquire_calibration, lambda channels: paired_batches, budget, actions,
                        observation_allowance=cfg['maximum_calibration_rows'] - calibration_rows,
                        required_fields=required)
                    save(output / 'instrument_models.json', agent.library.instrument_models)
                    binding = agent._instrument_binding(spec, command_spec)
                    if calibration is not None:
                        binding.update(rebind_instrument_model(calibration, spec))
                    if len(paid) != before_rows:
                        continue
                if required - set(binding) - set(command_spec.feature_names):
                    stopped = 'instrument_mapping_unresolved'
                    break
                with timing.phase('plan'):
                    plan = plan_competing_object_probe(command_spec, candidates, budget, binding=binding,
                        serial=passes, length=5, tolerance=cfg['threshold'], questions=questions,
                        methods=agent.library.program_experiment_methods,
                        task_id=f'{spec.task_id}:{agent.config.seed}')
                    plan = shorten_discriminating_probe(plan, budget, tolerance=cfg['threshold'])
                save(output / f'plan_{passes:02d}.json', {**plan,
                    'probe': asdict(plan['probe']) if plan['probe'] else None})
                actions.append({'action': 'plan_relation_experiment', 'plan_ref': ref(output / f'plan_{passes:02d}.json'),
                    'forecast_status': plan['forecast_status'], 'predicted_separation': plan['predicted_separation']})
                if plan['probe'] is None or plan['forecast_status'] != 'predicted_distinction':
                    stopped = plan['forecast_status']
                    break
                count = plan['probe'].cost
                # Pay for the complete response assessment, every retained candidate's
                # new rows, subsequent conditioning and a final frozen readout.
                required_compute = count * (len(scores) + len(plan['hypotheses']) +
                    len(plan['predicted_fields']) + 1) + conditioning_per_row * (len(paid) + count)
                if (budget.remaining < required_compute or
                        cfg['previous_observations'] + observation_cost + count + cfg['confirmation_rows'] > cfg['observation_limit'] or
                        cfg['previous_measurement_cost'] + measurement_cost + 12 * (count + cfg['confirmation_rows']) > cfg['measurement_limit']):
                    stopped = 'preserve_update_and_confirmation_allowance'
                    break
                with timing.phase('acquire'):
                    batch = lab.acquire(plan['probe'], ('fine',))
                    observation_cost, measurement_cost = lab.observation_cost, lab.measurement_cost
                    location = output / 'observations' / f'{len(purchased):02d}.json'
                    save(location, {'probe': asdict(plan['probe']), 'batch': asdict(batch)})
                    pairs = pair_rows(plan['probe'].sequence, batch.observations)
                    evidence = assess_competing_object_probe(plan, pairs, budget, tolerance=cfg['threshold'])
                    feedback = agent.library.program_experiment_methods.observe(plan, evidence,
                        task_id=f'{spec.task_id}:{agent.config.seed}', command_spec=command_spec,
                        binding=binding, tolerance=cfg['threshold'])
                    actions.append({'action': 'test_relation_experiment', 'observations_ref': ref(location),
                                    'evidence': evidence, 'method_feedback': feedback})
                    purchased.append(ref(location))
                    paired_batches.append(pairs)
                    extend_scores(scores, len(paid), batch.observations, budget, fallback)
                    paid.extend(batch.observations)
                    experiments.append(tuple(batch.observations))
                    plans += 1
                    # A failed input forecast is its own research problem. Recheck
                    # the now-paid calibration before proposing another experiment.
                    if evidence['calibration_status'] == 'counterexample':
                        calibration = None
                    budget.reserve = 2 * cfg['confirmation_rows'] + len(paid)
                save(output / 'progress.json', {'actions': actions, 'compute_cost': budget.spent,
                    'paid_rows': len(paid), 'pool_size': len(pool), 'new_observation_refs': purchased})
            budget.release_reserve()
            selected = compare_pool(agent, pool, scores, fallback)
            if not budget.spend(len(paid)):
                raise RuntimeError('Selected full-response readout not funded')
            predictions = predict_rows(selected['candidate'].program, paid)
            errors = [(value - row.target) ** 2 for value, row in zip(predictions, paid)]
            gate = full_comparison and max(errors) <= cfg['threshold'] and stopped in {
                'resolved_to_precision', 'no_remaining_competition', 'algebraically_shared'}
            save(output / 'selected_entry.json', encode_entry(selected))
            save(output / 'selection.json', {'stop_reason': stopped, 'full_comparison': full_comparison,
                'source_ref': ref(Path(cfg['source']) / 'selection.json'), 'new_observation_refs': purchased,
                'scores': [{'candidate_ref': location, 'loss': scores[entry['candidate'].program]}
                           for entry, location in zip(pool, pool_refs)],
                'selected_ref': ref(output / 'selected_entry.json'), 'actions': actions,
                'selected_paid_errors': errors, 'fresh_confirmation_admitted': gate})
            if gate:
                with timing.phase('confirmation'):
                    save(output / 'frozen_predictors.json', {'selected': selected['candidate'].program.to_dict(),
                        'incumbent': origin['candidate'].program.to_dict(), 'fallback': fallback})
                    if not budget.spend(2 * cfg['confirmation_rows']):
                        raise RuntimeError('Frozen readout not funded')
                    probe = lab.random_probe(cfg['confirmation_seed'], length=cfg['confirmation_rows'],
                                             name='consuming_interval_confirmation')
                    batch = lab.acquire(probe, ('fine',))
                    observation_cost, measurement_cost = lab.observation_cost, lab.measurement_cost
                    save(output / 'confirmation_observations.json', {'probe': asdict(probe), 'batch': asdict(batch)})
                    confirmation = {'rows': len(batch.observations), 'threshold': cfg['threshold']}
                    for name, entry in [('selected', selected), ('incumbent', origin)]:
                        values = predict_rows(entry['candidate'].program, batch.observations)
                        mse = mean((value - row.target) ** 2 for value, row in zip(values, batch.observations))
                        confirmation[name] = {'mse': mse, 'passed': mse <= cfg['threshold'], 'predictions': values}
            receipt = dict(status='confirmed' if confirmation and confirmation['selected']['passed'] else 'unresolved',
                stop_reason=stopped, full_comparison=full_comparison, retained_pool=len(scene['pool']),
                complete_pool=len(pool), interval_candidates=len(new_entries), conditioning_passes=passes,
                actual_experiments=plans, paid_rows=len(paid), old_parent_on_initial245_mse=original_loss,
                calibration_rows=calibration_rows,
                selected_paid_mse=mean(errors), max_paid_squared_error=max(errors),
                radius_intervals=research.intervals, additional_old_scoring=old_scoring,
                confirmation_admitted=gate, confirmation=confirmation,
                compute_cost=budget.spent, cumulative_compute=cfg['previous_compute'] + budget.spent,
                observation_cost=observation_cost, measurement_cost=measurement_cost,
                source_cost_refs=cfg['source_cost_refs'], model_calls=0, response_refits=0,
                method_condition_learned=False, execution_timing=timing.result())
            save(output / 'receipt.json', receipt)
            save(output / 'program_experiment_methods.json', agent.library.program_experiment_methods.to_dict())
            if confirmation and confirmation['selected']['passed']:
                from scripts.successor_interval_scene import save_confirmed_knowledge

                with timing.phase('knowledge'):
                    receipt['knowledge'] = save_confirmed_knowledge(agent, selected, spec, paid, errors,
                        confirmation, output, source=cfg['source'], new_observation_refs=purchased)
                receipt['execution_timing'] = timing.result()
                save(output / 'receipt.json', receipt)
            return receipt
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'compute_known': budget.spent,
            'compute_upper_bound': cfg['remaining_compute'], 'observation_cost': observation_cost,
            'measurement_cost': measurement_cost, 'actions': actions,
            'paid_candidate_refs': pool_refs, 'execution_timing': timing.result()})
        raise


def register_resume(source, output):
    """Continue the paid pending experiment without extending its original contract."""
    source, output = Path(source), Path(output)
    old, result = load(source / 'registration.json'), load(source / 'receipt.json')
    if not result['full_comparison'] or result['confirmation'] is not None:
        raise ValueError('Resume only the unconfirmed complete paid interval result')
    previous_experiments = result.get('cumulative_experiments', result['actual_experiments'])
    if previous_experiments >= old['maximum_experiments']:
        raise ValueError('The registered experiment count has been consumed')
    cfg = {**old, 'resume_result': str(source),
        'previous_compute': result['cumulative_compute'],
        'remaining_compute': old['total_compute_limit'] - result['cumulative_compute'],
        'previous_observations': old['previous_observations'] + result['observation_cost'],
        'previous_measurement_cost': old['previous_measurement_cost'] + result['measurement_cost'],
        'previous_experiments': previous_experiments, 'relation_precision': 1e-6,
        'source_cost_refs': [ref(source / 'receipt.json'), *old['source_cost_refs']],
        'resume_scope': 'one already planned experiment; reuse completed partition and scores; unchanged confirmation seed and gates',
        'update_quote': 'one predicted new cut; three representatives per cell plus exact incumbent; unexpected geometry may exhaust the unchanged allowance'}
    if cfg['remaining_compute'] <= 0:
        raise ValueError('The original allowance has been consumed')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', cfg)
    freeze_runtime(output, source)
    for filename in ('continue_successor_relation_intervals.py', 'successor_interval_scene.py',
                     'revise_learned_structure_successor.py'):
        name = Path('scripts') / filename
        write_text_incrementally(output / 'runtime' / name, (ROOT / name).read_text(),
                                 base=source / 'runtime' / name)
    return cfg


def resume(output):
    """Consume the existing plan, then confirm only a complete supported answer."""
    output = Path(output)
    if ROOT != (output / 'runtime').resolve():
        raise ValueError('Execute this newly frozen current runtime')
    if (output / 'started.json').exists():
        raise ValueError('Preserve the existing attempt')
    cfg = load(output / 'registration.json')
    save(output / 'started.json', {'source': cfg['resume_result']})
    budget = SearchBudget(cfg['remaining_compute'])
    timing = ExecutionTiming(('restore', 'acquire', 'interval_research', 'paid_comparison',
                              'confirmation', 'knowledge'))
    actions, purchased, pool_refs = [], [], []
    observation_cost = measurement_cost = 0
    try:
        with timing.phase('restore'):
            scene = restore_interval_result(cfg['resume_result'])
        agent, spec, paid = scene['agent'], scene['spec'], list(scene['paid'])
        pool, pool_refs, scores = list(scene['pool']), list(scene['refs']), dict(scene['scores'])
        parent, selected = scene['parent'], scene['parent']
        fallback, old_count = parent['fallback'], len(paid)
        # The original frozen comparator remains the parent before interval research.
        origin = scene['source_scene']['parent']
        plan, experiments = scene['pending_plan'], list(scene['experiments'])
        if not plan or plan['forecast_status'] != 'predicted_distinction':
            raise ValueError('There is no paid pending distinguishing experiment')
        research = RelationIntervalResearch(parent['candidate'].program, agent.library.meta_priors,
            budget, structure=parent.get('structure'), research_program=parent.get('research_program'))
        reused = research.reuse_paid_partition(paid, parent)
        if not reused:
            raise ValueError('The paid partition does not belong to this complete response and evidence')
        count = plan['probe'].cost
        rows_after = old_count + count
        budget.reserve = rows_after + 2 * cfg['confirmation_rows']
        # The saved plan targets one cell boundary; a changed calibration or
        # unexpected extra cuts stop this bounded unit, never increase its limit.
        update_quote = count * len(spec.feature_names) + 7 * rows_after
        consume_quote = count * (len(scores) + len(plan['hypotheses']) + len(plan['predicted_fields']))
        funded = (budget.can_spend(update_quote + consume_quote)
            and cfg['previous_experiments'] < cfg['maximum_experiments']
            and cfg['previous_observations'] + count + cfg['confirmation_rows'] <= cfg['observation_limit']
            and cfg['previous_measurement_cost'] + 12 * (count + cfg['confirmation_rows']) <= cfg['measurement_limit'])
        actions.append({'action': 'reuse_paid_interval_state', 'rows': old_count,
            'candidate_occurrences': len(pool), 'program_scores': len(scores),
            'plan_ref': scene['pending_plan_ref'], 'partition_reused': reused,
            'update_quote': update_quote, 'consume_quote': consume_quote, 'funded': funded})
        full_comparison, stopped, candidates, confirmation = True, 'preserve_update_and_confirmation_allowance', [], None
        errors = list(scene['paid_errors'])
        with AnonymousObjectLab(**scene['original_registration']['task']) as lab, \
                python_execution_deadline(cfg['active_wall_seconds'], activity=True):
            if funded:
                with timing.phase('acquire'):
                    batch = lab.acquire(plan['probe'], ('fine',))
                    observation_cost, measurement_cost = lab.observation_cost, lab.measurement_cost
                    location = output / 'observations' / '00.json'
                    save(location, {'probe': asdict(plan['probe']), 'batch': asdict(batch)})
                    purchased.append(ref(location))
                    pairs = pair_rows(plan['probe'].sequence, batch.observations)
                    evidence = assess_competing_object_probe(plan, pairs, budget, tolerance=cfg['threshold'])
                    binding = agent._instrument_binding(spec, lab.spec_for(()))
                    binding.update(rebind_instrument_model(scene['calibration'], spec))
                    feedback = agent.library.program_experiment_methods.observe(plan, evidence,
                        task_id=f'{spec.task_id}:{agent.config.seed}', command_spec=lab.spec_for(()),
                        binding=binding, tolerance=cfg['threshold'])
                    actions.append({'action': 'test_relation_experiment', 'plan_ref': scene['pending_plan_ref'],
                        'observations_ref': ref(location), 'evidence': evidence, 'method_feedback': feedback})
                with timing.phase('paid_comparison'):
                    extend_scores(scores, old_count, batch.observations, budget, fallback)
                paid.extend(batch.observations)
                experiments.append(tuple(batch.observations))
                if evidence['calibration_status'] == 'counterexample':
                    stopped = 'instrument_mapping_unresolved'
                else:
                    before = budget.spent
                    with timing.phase('interval_research'):
                        candidates = research.alternatives(paid, budget, tolerance=cfg['threshold'],
                                                           experiments=experiments)
                    actions.append({'action': 'condition_relation_intervals', 'status': research.status,
                        'rows': len(paid), 'candidates': len(candidates), 'compute_cost': budget.spent - before,
                        'intervals': deepcopy(research.intervals)})
                    full_comparison = research.status not in {'compute_exhausted', 'refuted'}
                    for entry in candidates:
                        pool.append(entry)
                        scores[entry['candidate'].program] = entry['validation_loss']
                    selected = compare_pool(agent, pool, scores, fallback)
                    active = retain_interval_evidence(research, selected, candidates, len(paid), cfg['threshold'])
                    if active is not None:
                        if not any(active is entry for entry in candidates):
                            candidates.append(active)
                            pool.append(active)
                        research.adopt(active)
                    candidate_refs, _ = save_interval_candidates(output, 1, candidates,
                        previous=Path(cfg['resume_result']) / 'selected_entry.json')
                    pool_refs.extend(candidate_refs)
                    stopped = ('resolved_to_precision' if active is not None and all(
                        item['radius_interval'][1] - item['radius_interval'][0] <= cfg['relation_precision']
                        for item in research.intervals) else research.status)
                budget.release_reserve()
                selected = compare_pool(agent, pool, scores, fallback)
                if not budget.spend(len(paid)):
                    raise RuntimeError('Selected full-response readout not funded')
                errors = [(value-row.target)**2 for value, row in zip(
                    predict_rows(selected['candidate'].program, paid), paid)]
            else:
                budget.release_reserve()
            gate = full_comparison and max(errors) <= cfg['threshold'] and stopped == 'resolved_to_precision'
            save(output / 'selected_entry.json', encode_entry(selected))
            save(output / 'selection.json', {'stop_reason': stopped, 'full_comparison': full_comparison,
                'source_ref': scene['source_refs']['selection'],
                'new_observation_refs': [*scene['knowledge_observation_refs'], *purchased],
                'scores': [{'candidate_ref': location, 'loss': scores[entry['candidate'].program]}
                           for entry, location in zip(pool, pool_refs)],
                'selected_ref': ref(output / 'selected_entry.json'), 'actions': actions,
                'selected_paid_errors': errors, 'fresh_confirmation_admitted': gate})
            if gate:
                with timing.phase('confirmation'):
                    save(output / 'frozen_predictors.json', {'selected': selected['candidate'].program.to_dict(),
                        'incumbent': origin['candidate'].program.to_dict(), 'fallback': fallback})
                    if not budget.spend(2 * cfg['confirmation_rows']):
                        raise RuntimeError('Frozen readout not funded')
                    probe = lab.random_probe(cfg['confirmation_seed'], length=cfg['confirmation_rows'],
                                             name='consuming_interval_confirmation')
                    batch = lab.acquire(probe, ('fine',))
                    observation_cost, measurement_cost = lab.observation_cost, lab.measurement_cost
                    save(output / 'confirmation_observations.json', {'probe': asdict(probe), 'batch': asdict(batch)})
                    confirmation = {'rows': len(batch.observations), 'threshold': cfg['threshold']}
                    for name, entry in [('selected', selected), ('incumbent', origin)]:
                        values = predict_rows(entry['candidate'].program, batch.observations)
                        mse = mean((value-row.target)**2 for value, row in zip(values, batch.observations))
                        confirmation[name] = {'mse': mse, 'passed': mse <= cfg['threshold'], 'predictions': values}
            receipt = {'status': 'confirmed' if confirmation and confirmation['selected']['passed'] else 'unresolved',
                'stop_reason': stopped, 'full_comparison': full_comparison,
                'retained_pool': len(scene['pool']), 'complete_pool': len(pool),
                'interval_candidates': len(candidates), 'paid_rows': len(paid),
                'actual_experiments': len(purchased), 'cumulative_experiments': cfg['previous_experiments'] + len(purchased),
                'selected_paid_mse': mean(errors), 'max_paid_squared_error': max(errors),
                'radius_intervals': research.intervals, 'reused_program_scores': len(scene['scores']),
                'evidence_reuse': dict(research.reuse_statistics),
                'confirmation_admitted': gate, 'confirmation': confirmation,
                'compute_cost': budget.spent, 'cumulative_compute': cfg['previous_compute'] + budget.spent,
                'observation_cost': observation_cost, 'measurement_cost': measurement_cost,
                'source_cost_refs': cfg['source_cost_refs'], 'model_calls': 0, 'response_refits': 0,
                'method_condition_learned': False, 'execution_timing': timing.result()}
            save(output / 'receipt.json', receipt)
            save(output / 'instrument_models.json', agent.library.instrument_models)
            save(output / 'program_experiment_methods.json', agent.library.program_experiment_methods.to_dict())
            if confirmation and confirmation['selected']['passed']:
                with timing.phase('knowledge'):
                    receipt['knowledge'] = save_confirmed_knowledge(agent, selected, spec, paid, errors,
                        confirmation, output, source=scene['knowledge_source'],
                        new_observation_refs=[*scene['knowledge_observation_refs'], *purchased])
                receipt['execution_timing'] = timing.result()
                save(output / 'receipt.json', receipt)
            return receipt
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'compute_known': budget.spent,
            'compute_upper_bound': cfg['remaining_compute'], 'observation_cost': observation_cost,
            'measurement_cost': measurement_cost, 'actions': actions, 'new_observation_refs': purchased,
            'paid_candidate_refs': pool_refs, 'execution_timing': timing.result()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'prepare-resume', 'resume'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode in {'prepare', 'prepare-resume'}:
        result = (register if args.mode == 'prepare' else register_resume)(args.source, args.output)
    else:
        result = (run if args.mode == 'run' else resume)(args.output.resolve())
    print(result)
