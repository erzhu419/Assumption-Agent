#!/usr/bin/env python3
"""One frozen-knowledge successor: shared discovery, paired research, fresh readout."""

import argparse
from copy import deepcopy
from math import ceil
from dataclasses import asdict, replace
import json
from pathlib import Path
from random import Random

from scripts.retain_confirmed_structure import load_retained_knowledge
from scripts.run_native_decision_pair import (
    finish_prepared_pair, load_prepared, ref, save_prepared,
)
from scripts.run_partial_role_parent import load, plain, save
from scripts.run_partial_role_successors import knowledge_changes
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.common_object_research import CommonObjectResearchAgent
from hegel_machine.cognition.effective_history import HistoryStore
from hegel_machine.cognition.evaluation import predict
from hegel_machine.cognition.experiment_runtime import ExecutionTiming, python_execution_deadline
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.object_continuation import observations


ROOT = Path(__file__).resolve().parents[1]
SEED = 106029104
CONFIRMATION_ROWS = 48


def task_settings(seed=SEED):
    """The same public task-generation distribution as the previous three roots."""
    rng = Random(seed + 17011)
    return dict(seed=seed, objects=4, apparatus_seed=seed,
        name_seed=seed + 63857, wiring_seed=seed + 3709,
        variant='exchange', profile='confirmation', readout='window_total_event',
        actuator_curvature=rng.uniform(.08, .24), relation_radius=rng.uniform(.70, 1.20))


def register(knowledge_root, source_root, prior_review, output):
    knowledge_root, source_root, prior_review, output = map(Path,
        (knowledge_root, source_root, prior_review, output))
    retained = load(knowledge_root / 'receipt.json')
    if retained['status'] != 'confirmed_structure_method_retained':
        raise ValueError('The declared source knowledge has not completed paid retention')
    old = load(source_root / 'registration.json')
    if any(item['seed'] == SEED for item in old['tasks']):
        raise ValueError('This fixed successor was already part of the source cohort')
    knowledge_delta = load(knowledge_root / 'knowledge_delta.json')
    source = Path(knowledge_delta['source'])
    if load(source / 'registration.json')['seed'] == SEED:
        raise ValueError('The declared successor is the knowledge source itself')
    historical = knowledge_delta['historical_knowledge_ref']
    store = HistoryStore(historical['store'], readonly=True)
    try:
        previous = [row[0] for row in store.connection.execute(
            'SELECT unit FROM entries WHERE unit LIKE ? LIMIT 3', ('%' + str(SEED) + '%',))]
    finally:
        store.close()
    if previous:
        raise ValueError('The fixed successor already occurs in the retained historical index')
    topic = next(q for q in load(prior_review / 'common/questions.json')
                 if q['id'] == 'task:method_choice')
    review = 2 + len(topic.get('method_use_plan', {}).get('native_return_refs', []))
    cfg = ResearchConfig(seed=SEED, observation_budget=512, compute_budget=90000,
        minimum_improvement=1e-7, max_rounds=6, max_history=4, active_probes=True)
    output.mkdir(parents=True, exist_ok=False)
    registration = {
        'source': str(source_root), 'source_seed': SEED, 'task': task_settings(),
        'source_task_ref': ref(output / 'registration.json', 'task'),
        'knowledge_root': str(knowledge_root),
        'knowledge_contract_ref': ref(knowledge_root / 'knowledge_delta.json'),
        'historical_knowledge_ref': historical,
        'raw_source_ref': ref(source_root / 'inputs/raw_source.json'),
        'role_method_ref': ref(source_root / 'inputs/method.json'),
        'source_cost_refs': [ref(source_root / 'inputs/source_costs.json'),
            ref(source_root / 'registration.json', 'historical_cumulative_costs'),
            ref(knowledge_root / 'receipt.json'), ref(source / 'receipt.json')],
        'source_cost_scope': 'original sources and failed attempts stay referenced; count each source once',
        'prior_method_review': str(prior_review), 'method_version': 2,
        'operations': {'baseline': 'program_replay', 'challenger': 'structural_research'},
        'research_config': asdict(cfg), 'preparation_compute_limit': 90000,
        'method_review_compute_limit': review, 'common_compute_limit': 90000 + review,
        'per_arm_compute_limit': 90000 - CONFIRMATION_ROWS,
        'per_arm_total_compute_limit': 90000,
        'per_arm_active_wall_seconds': 1800, 'preparation_active_wall_seconds': 1800,
        'observation_limit': 512, 'measurement_limit': 6144,
        'confirmation_rows': CONFIRMATION_ROWS, 'confirmation_seed': SEED + 1000003,
        'confirmation_compute_per_arm': CONFIRMATION_ROWS,
        'quality_threshold': cfg.minimum_improvement,
        'unexposed_check': {'historical_index_matches': previous,
            'source_cohort_seeds': [item['seed'] for item in old['tasks']],
            'scope': 'declared source cohort and historical index; not a scan of all past files'},
        'evaluation_role': 'new_shared_working_trajectory',
        'scope': 'one new physical successor; same K1, data and pool, two fixed research segments',
        'stop_rule': 'stop if common pool solves, preparation is incomplete or a segment fails; no new seed or extension',
        'confirmation_contract': 'one shared 48-row acquisition after both returned predictors are frozen; later method feedback only',
        'model_calls': 0,
    }
    save(output / 'registration.json', registration)
    freeze_runtime(output, knowledge_root)
    return registration


def register_column_successor(knowledge_root, repaired_entry_ref, source_root, prior_review, output):
    """Freeze one new successor with only the retained method columns changed."""
    from scripts.successor_interval_scene import load_repaired_knowledge

    knowledge_root, source_root, prior_review, output = map(Path,
        (knowledge_root, source_root, prior_review, output))
    seed = 106029105
    old = load(source_root / 'registration.json')
    historical = old['historical_knowledge_ref']
    store = HistoryStore(historical['store'], readonly=True)
    try:
        previous = [row[0] for row in store.connection.execute(
            'SELECT unit FROM entries WHERE unit LIKE ? LIMIT 3', ('%' + str(seed) + '%',))]
    finally:
        store.close()
    if previous or seed in [item['seed'] for item in old['tasks']]:
        raise ValueError('The registered successor already appears in source history')
    library, retained = load_repaired_knowledge(knowledge_root, repaired_entry_ref)
    topic = next(q for q in load(prior_review / 'common/questions.json') if q['id'] == 'task:method_choice')
    review = 2 + len(topic.get('method_use_plan', {}).get('native_return_refs', []))
    cfg = ResearchConfig(seed=seed, observation_budget=512, compute_budget=110000,
        minimum_improvement=1e-7, max_rounds=6, max_history=4, active_probes=True)
    output.mkdir(parents=True, exist_ok=False)
    patch = {**retained, 'prior': deepcopy(library.meta_priors[retained['prior_index']])}
    save(output / 'method_revision.json', patch)
    previous_pair = load(prior_review / 'pair.json')
    previous_baseline = read_method_reference(previous_pair['baseline_ref'], prior_review)
    registration = {
        'source': str(source_root), 'source_seed': seed, 'task': task_settings(seed),
        'source_task_ref': ref(output / 'registration.json', 'task'),
        'knowledge_root': str(knowledge_root), 'knowledge_loader': 'confirmed_interval',
        'knowledge_contract_ref': ref(knowledge_root / 'knowledge_delta.json'),
        'historical_knowledge_ref': historical,
        'raw_source_ref': ref(source_root / 'inputs/raw_source.json'),
        'role_method_ref': ref(source_root / 'inputs/method.json'),
        'source_cost_refs': [ref(source_root / 'inputs/source_costs.json'),
            ref(source_root / 'registration.json', 'historical_cumulative_costs'),
            ref(knowledge_root / 'receipt.json'),
            ref(Path(repaired_entry_ref['file']).parent / 'receipt.json')],
        'prior_method_review': str(prior_review),
        'method_versions': {'baseline': 3, 'challenger': 4},
        'retained_method_artifact_refs': {'baseline': previous_baseline['method_artifact_ref']},
        'method_program_revisions': {'challenger': ref(output / 'method_revision.json')},
        'operations': {'baseline': 'program_replay', 'challenger': 'program_replay'},
        'research_config': asdict(cfg), 'preparation_compute_limit': 110000,
        'method_review_compute_limit': review, 'common_compute_limit': 110000 + review,
        'per_arm_compute_limit': 80000 - CONFIRMATION_ROWS, 'per_arm_total_compute_limit': 80000,
        'per_arm_active_wall_seconds': 1800, 'preparation_active_wall_seconds': 1800,
        'observation_limit': 512, 'measurement_limit': 6144,
        'confirmation_rows': CONFIRMATION_ROWS, 'confirmation_seed': seed + 1000003,
        'confirmation_compute_per_arm': CONFIRMATION_ROWS, 'quality_threshold': cfg.minimum_improvement,
        'unexposed_check': {'historical_index_matches': previous,
            'source_cohort_seeds': [item['seed'] for item in old['tasks']]},
        'evaluation_role': 'new_shared_working_trajectory',
        'scope': 'one prospective successor; frozen K2 single-column versus retained multi-column replay',
        'stop_rule': 'common solved or incomplete: stop; otherwise one matched replay and frozen readout; no second seed',
        'boundary_followup': 'review target residuals after fixed-data replay; no within-pair acquisitions or structural search',
        'confirmation_contract': 'freeze both full-pool winners then one shared 48-row readout; later method feedback only',
        'model_calls': 0,
    }
    save(output / 'registration.json', registration)
    freeze_runtime(output, knowledge_root)
    return registration


def load_study_knowledge(registration):
    if registration.get('knowledge_loader') == 'confirmed_interval':
        from scripts.successor_interval_scene import load_confirmed_knowledge
        return load_confirmed_knowledge(Path(registration['knowledge_root']))
    return load_retained_knowledge(Path(registration['knowledge_root']))


def freeze_runtime(output, source):
    names = [p.relative_to(ROOT) for p in (ROOT / 'src/hegel_machine').rglob('*')
             if p.is_file() and p.suffix in {'.py', '.json'}]
    names.extend(Path('scripts') / name for name in (
        'run_learned_structure_successor.py', 'retain_confirmed_structure.py',
        'run_native_decision_pair.py', 'native_decision_scene.py',
        'run_partial_role_parent.py', 'run_partial_role_successors.py',
        'partial_role_paid_scene.py', 'accept_partial_role_structural_seed.py',
        'successor_interval_scene.py', 'revise_learned_structure_successor.py'))
    for name in names:
        target, base = output / 'runtime' / name, source / 'runtime' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text_incrementally(target, (ROOT / name).read_text(), base=base if base.exists() else None)


def register_continuation(previous, output):
    """Complete paid comparison within the original total, before either arm starts."""
    previous, output = Path(previous), Path(output)
    old = load(previous / 'registration.json')
    if load(previous / 'receipt.json')['status'] != 'preparation_incomplete':
        raise ValueError('This continuation only consumes an incomplete common comparison')
    if any((previous / arm).exists() for arm in old['operations']):
        raise ValueError('Do not reallocate after a research arm has started')
    state = load(previous / 'prepared_state.json')
    scene = load_successor_scene(previous)
    required = len(state['uncompared']) * len(scene['validation'])
    if not required:
        raise ValueError('There is no paid pending comparison to complete')
    costs = load(previous / 'common_receipt.json')['costs']
    per_arm_transfer = ceil(max(0, costs['compute_cost'] + required
                               - old['preparation_compute_limit']) / len(old['operations']))
    registration = deepcopy(old)
    registration['preparation_compute_limit'] += per_arm_transfer * len(old['operations'])
    registration['common_compute_limit'] += per_arm_transfer * len(old['operations'])
    registration['per_arm_compute_limit'] -= per_arm_transfer
    registration['per_arm_total_compute_limit'] -= per_arm_transfer
    registration.update(previous_unit=str(previous), prior_paid_costs=costs,
        comparison_resume={'pending_candidates': len(state['uncompared']), 'compute_required': required,
            'per_arm_transfer_to_common': per_arm_transfer,
            'total_compute_limit': old['common_compute_limit']
                + len(old['operations']) * old['per_arm_total_compute_limit'],
            'new_fits': 0, 'new_observations': 0},
        scope='same new physical successor after explicit preparation-budget reallocation; not a clean first attempt')
    if registration['per_arm_compute_limit'] <= 0:
        raise ValueError('No complete paired allocation remains')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', registration)
    freeze_runtime(output, previous)
    return registration


def acquire_common(agent, laboratory, output, registration, budget, actions, batches):
    """Use ordinary probes and store each purchased trajectory only once."""
    rates = {item.name: item.cost_per_row for item in laboratory.channels}
    reserve = registration['confirmation_rows']
    reserve_measurement = reserve * rates['fine']

    def acquire(channels, purpose, candidates=(), *, rows_count=None, probe=None, questions=()):
        count = probe.cost if probe is not None else 24 if rows_count is None else rows_count
        rate = sum(rates[name] for name in channels)
        if (laboratory.observation_cost + count > registration['observation_limit'] - reserve
                or laboratory.measurement_cost + count * rate >
                    registration['measurement_limit'] - reserve_measurement):
            return None
        if probe is None:
            probe = agent._probe(agent._spec(laboratory, channels), len(batches), count,
                () if purpose == 'compare_on_new_trajectory' else candidates, budget, actions)
        batch = laboratory.acquire(probe, channels)
        payload = {'probe': asdict(probe), 'channels': list(channels),
            'observations': [asdict(row) for row in batch.observations],
            'transitions': getattr(batch, 'transitions', [])}
        location = output / 'observations' / f'{len(batches):03d}.json'
        # Save the purchase before any interpretation or next research step.
        save(location, payload)
        batches.append(payload)
        actions.append({'action': 'acquire', 'purpose': purpose, 'probe_id': probe.probe_id,
            'channels': list(channels), 'observation_cost': batch.observation_cost,
            'measurement_cost': batch.measurement_cost, 'batch_ref': ref(location)})
        save(output / 'acquisition_receipt.json', {'batches': [ref(output / 'observations' / f'{i:03d}.json')
            for i in range(len(batches))], 'costs': laboratory.costs})
        return batch.observations

    return acquire


def load_successor_scene(output):
    """Recover this new paid prefix from disk, without acquiring or fitting again."""
    output = Path(output)
    registration = load(output / 'registration.json')
    state = load(output / 'prepared_state.json')
    references = state['source_references']
    train_batches = [read_method_reference(item, output) for item in references['training']]
    compare_batches = [read_method_reference(item, output) for item in references['validation']]
    model = read_method_reference(references['object_model'], output)
    # The ordinary measured view predates the library's membership examples.
    measured = deepcopy(model)
    measured.pop('membership_witnesses', None)
    with AnonymousObjectLab(**registration['task']) as laboratory:
        public = laboratory.spec_for(('fine',))
    context = {**public.context, 'object_model': measured}
    if model.get('complete') and model.get('schema'):
        context['object_schema'] = json.dumps(model['schema'])
    spec = replace(public, context=context)
    library = load_study_knowledge(registration)
    return {'spec': spec,
        'training': [row for batch in train_batches for row in observations(batch['observations'])],
        'validation': [row for batch in compare_batches for row in observations(batch['observations'])],
        'transitions': [deepcopy(item) for batch in train_batches for item in batch.get('transitions', [])],
        'parent': None, 'object_model': model, 'knowledge': plain(library.to_dict()),
        'knowledge_base_ref': registration['knowledge_contract_ref'],
        'knowledge_additions': [references['object_model']],
        'research_config': registration['research_config'],
        'historical_knowledge_ref': registration['historical_knowledge_ref'],
        'source_task_ref': registration['source_task_ref'], 'references': references}


def prepare_successor(output, registration, library, budget, timing):
    """Discover roles from public measurements, then freeze the entire common pool."""
    knowledge = plain(library.to_dict())
    agent = CommonObjectResearchAgent(ResearchConfig(**registration['research_config']),
        measurement_budget=registration['measurement_limit'], library=library,
        workspace=output / 'common_roles', organization='pr3',
        source_task_ref=registration['source_task_ref'],
        raw_sources=[registration['raw_source_ref']],
        learned_methods=[read_method_reference(registration['role_method_ref'], output)])
    actions, batches = [], []
    with AnonymousObjectLab(**registration['task']) as laboratory:
        try:
            with timing.phase('preparation'), python_execution_deadline(
                    registration['preparation_active_wall_seconds'], activity=True):
                agent._learn_instruments = agent._instrument_knowledge(laboratory)
                agent._revise_inputs = agent._revisable_inputs(laboratory)
                acquire = acquire_common(agent, laboratory, output, registration, budget, actions, batches)
                model = agent._research_objects(laboratory, ('fine',), acquire, batches, budget, actions,
                    remaining_rows=registration['observation_limit'] - registration['confirmation_rows'])
                save(output / 'object_model.json', model)
                training = [row for batch in batches for row in observations(batch['observations'])]
                transitions = [deepcopy(item) for batch in batches for item in batch.get('transitions', [])]
                training_refs = [ref(output / 'observations' / f'{i:03d}.json') for i in range(len(batches))]
                if not model.get('groups') or len(training) < 6:
                    return None, None, None
                validation = acquire(('fine',), 'compare_on_new_trajectory', rows_count=24)
                if validation is None:
                    return None, None, None
                spec = agent._spec(laboratory, ('fine',))
                scene = {'spec': spec, 'training': training, 'validation': validation,
                    'transitions': transitions, 'parent': None, 'object_model': model,
                    'knowledge': knowledge, 'knowledge_additions': [ref(output / 'object_model.json')],
                    'knowledge_base_ref': registration['knowledge_contract_ref'],
                    'research_config': registration['research_config'],
                    'historical_knowledge_ref': registration['historical_knowledge_ref'],
                    'source_task_ref': registration['source_task_ref'],
                    'references': {'report': ref(output / 'common_receipt.json'),
                        'registration': ref(output / 'registration.json'),
                        'choice': ref(output / 'prepared_state.json', 'directions'),
                        'training': training_refs,
                        'validation': [ref(output / 'observations' / f'{len(batches) - 1:03d}.json')],
                        'object_model': ref(output / 'object_model.json'), 'parent_original': None}}
                frame = agent.prepare_research_decision(spec, training, validation, budget, actions,
                    parent=None, transitions=transitions,
                    observation_remaining=registration['observation_limit'] - laboratory.observation_cost,
                    measurement_remaining=registration['measurement_limit'] - laboratory.measurement_cost)
                save_prepared(output, agent, frame, scene, actions)
                recovered = load_successor_scene(output)
                restored, prepared, _ = load_prepared(output, scene=recovered)
                return restored, prepared, recovered
        finally:
            save(output / 'common_actions.json', actions)
            if (output / 'prepared_state.json').exists():
                knowledge_ref = ref(output / 'prepared_state.json', 'knowledge_delta')
            else:
                save(output / 'common_knowledge_delta.json', {
                    'base_ref': registration['knowledge_contract_ref'],
                    'changes': knowledge_changes(knowledge, plain(agent.library.to_dict()))})
                knowledge_ref = ref(output / 'common_knowledge_delta.json', 'changes')
            save(output / 'common_receipt.json', {
                'costs': {**laboratory.costs, 'compute_cost': budget.spent},
                'batches': [ref(output / 'observations' / f'{i:03d}.json') for i in range(len(batches))],
                'source_cost_refs': registration['source_cost_refs'],
                'knowledge_base_ref': registration['knowledge_contract_ref'],
                'knowledge_delta_ref': knowledge_ref,
                'source_reconstruction_receipts': agent.source_reconstruction_receipts,
                'execution_timing': timing.result(), 'model_calls': 0})


def complete_paid_preparation(output, registration, budget, timing):
    previous = Path(registration['previous_unit'])
    scene = load_successor_scene(previous)
    agent, frame, _ = load_prepared(previous, scene=scene)
    before, pending = budget.spent, frame['uncompared']
    actions = []
    with timing.phase('preparation'):
        checked = agent._assess(pending, frame['comparison'], budget,
                                frame['fallback'], frame['spec'].risk_quantile)
        frame['assessed'].extend(checked)
        frame['uncompared'] = pending[len(checked):]
        frame['comparison_complete'] = not frame['uncompared']
        frame['terminal'] = agent._research_pool_explained(frame)
        frame['prepare_compute_cost'] += budget.spent - before
        frame['remaining_compute'] = budget.remaining
        actions.append({'action': 'complete_paid_preparation_comparison',
            'previous_state_ref': ref(previous / 'prepared_state.json'),
            'new_scores': len(checked), 'compute_cost': budget.spent - before,
            'new_fits': 0, 'new_observations': 0})
        scene['references'] = {**scene['references'], 'report': ref(output / 'common_receipt.json'),
            'registration': ref(output / 'registration.json'),
            'choice': ref(output / 'prepared_state.json', 'directions')}
        save_prepared(output, agent, frame, scene, actions)
        save(output / 'common_receipt.json', {
            'costs': {**registration['prior_paid_costs'], 'compute_cost': budget.spent},
            'incremental_compute': budget.spent - before,
            'previous_receipt_ref': ref(previous / 'common_receipt.json'),
            'comparison_resume': registration['comparison_resume'],
            'actions': actions, 'source_cost_refs': registration['source_cost_refs']})
        recovered = load_successor_scene(output)
        restored, actual, _ = load_prepared(output, scene=recovered)
    return restored, actual, recovered


def confirm_pair(output, registration):
    """Freeze both delivered winners before buying one shared confirmation."""
    directory = output / 'confirmation'
    directory.mkdir(exist_ok=False)
    frozen = {}
    for arm in registration['operations']:
        outcome = load(output / arm / 'outcome.json')
        if not outcome['execution_complete'] or not outcome['full_comparison'] or not outcome['selected_ref']:
            result = {'status': 'not_run_incomplete_pair', 'arm': arm, 'new_compute': 0,
                'new_observations': 0}
            save(directory / 'receipt.json', result)
            return result
        entry = read_method_reference(outcome['selected_ref'], output)
        frozen[arm] = {'program': entry['candidate']['program'],
            'fallback': load(output / 'prepared_state.json')['fallback'],
            'selected_ref': outcome['selected_ref']}
    save(directory / 'frozen_predictors.json', frozen)
    rows_count, compute = registration['confirmation_rows'], 0
    scored = {}
    with AnonymousObjectLab(**registration['task']) as laboratory:
        try:
            probe = laboratory.random_probe(registration['confirmation_seed'], length=rows_count,
                name='shared_frozen_confirmation')
            batch = laboratory.acquire(probe, ('fine',))
            save(directory / 'observations.json', {'probe': asdict(probe), 'batch': asdict(batch)})
            for arm, payload in frozen.items():
                program = Program.from_dict(payload['program'])
                values = []
                scored[arm] = {'predictions': values}
                for row in batch.observations:
                    compute += 1
                    values.append(predict(program, row, payload['fallback']))
                mse = sum((value - row.target) ** 2 for value, row in zip(values, batch.observations)) / rows_count
                scored[arm].update(mse=mse, passed=mse <= registration['quality_threshold'])
        except BaseException as error:
            save(directory / 'failure.json', {'error': repr(error), 'costs': {
                **laboratory.costs, 'compute_cost': compute}, 'partial_scores': scored})
            raise
        result = {'status': 'frozen_pair_readout_completed', 'rows': rows_count, 'arms': scored,
            'all_passed': all(item['passed'] for item in scored.values()),
            'shared_actual_costs': {**laboratory.costs, 'compute_cost': compute},
            'new_fits': 0, 'model_calls': 0, 'used_for_selection_or_condition_learning': False}
    save(directory / 'receipt.json', result)
    return result


def consume_frozen_pair(output, source, *, pair_ref=None, prior_review=None, compute_limit):
    """Return an already completed readout to the original method question."""
    from hegel_machine.cognition.common_object_research import (
        CommonObjectDependencyMethods, record_native_parent_research_return,
    )
    from hegel_machine.cognition.method_feedback import ensure_python_method_topic
    from hegel_machine.cognition.question_routing import consume_method_return_agenda

    output, source = Path(output), Path(source)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'reused': True, 'new_compute_cost': 0}
    pair_ref = pair_ref or ref(source / 'pair.json')
    pair = read_method_reference(pair_ref, source)
    confirmation = load(source / 'confirmation/receipt.json')
    if confirmation.get('status') != 'frozen_pair_readout_completed':
        return {'status': 'no_completed_frozen_pair', 'new_compute_cost': 0}
    previous = Path(prior_review) if prior_review else source
    owner = next(q for q in load(previous / 'common/questions.json')
                 if q['id'] == 'task:method_choice')
    # The native reviewer consumes all retained returns once, including this
    # later result. Insufficient allowance does not silently create a label.
    required = 2 + len(owner['method_use_plan']['native_return_refs'])
    if compute_limit < required:
        return {'status': 'post_freeze_method_return_not_funded',
            'required_compute': required, 'new_compute_cost': 0}
    output.mkdir(parents=True, exist_ok=False)
    updated = {**deepcopy(pair), 'previous_working_pair_ref': pair_ref,
        'paired_future_ref': ref(source / 'confirmation/receipt.json'),
        'frozen_predictors_ref': ref(source / 'confirmation/frozen_predictors.json'),
        'evaluation_role': 'paired_future', 'data_role': 'development',
        'evaluation_lifecycle': {'all_outputs_frozen_before_labels': True,
            'labels_revealed': True, 'role': 'development_after_freeze'},
        'shared_evaluation_cost_ref': ref(source / 'confirmation/receipt.json', 'shared_actual_costs')}
    save(output / 'pair.json', updated)
    challenger = read_method_reference(pair['challenger_ref'], source)
    save(output / 'research_receipt.json', {
        'status': 'frozen_evaluation_returned', 'source_task_ref': pair['physical_source_ref'],
        'method_pair_ref': ref(output / 'pair.json'),
        'method_ref': challenger['method_ref'],
        'method_artifact_ref': challenger['method_artifact_ref'],
        'confirmation_ref': updated['paired_future_ref'],
        'original_pair_ref': pair_ref, 'world_execution_repeated': False})
    save(output / 'previous_topic.json', {'question': owner, 'questions': [owner], 'methods': []})
    common = CommonObjectDependencyMethods(output / 'common', [],
        budget=SearchBudget(compute_limit), actions=[],
        source_task_ref=pair['physical_source_ref']).common
    ensure_python_method_topic(common, parent_question_ref={'question_id': 'task'},
        topic_ref=ref(output / 'previous_topic.json'))
    mission = deepcopy(owner['mission'])
    mission['allocation']['compute'] += compute_limit
    feedback = record_native_parent_research_return(common,
        source_task_ref=pair['physical_source_ref'], result_ref=ref(output / 'research_receipt.json'),
        confirmation_ref=updated['paired_future_ref'],
        cost_refs=[updated['shared_evaluation_cost_ref']], mission=mission)
    result = consume_method_return_agenda(common, feedback, None, compute_limit=compute_limit)
    receipt = {'status': 'post_freeze_method_return_completed', 'method_return': result,
        'new_compute_cost': common.costs['compute_cost'], 'new_fits': 0, 'new_predictions': 0,
        'new_observations': 0, 'model_calls': 0, 'source': str(source),
        'original_independent_result_ref': updated['paired_future_ref'],
        'pair_ref': ref(output / 'pair.json'),
        'data_role_after_return': 'exposed_method_development',
        'old_world_results_unchanged': True}
    save(output / 'receipt.json', receipt)
    return receipt


def run(output):
    output = Path(output)
    if ROOT != (output / 'runtime').resolve():
        raise ValueError('Run the newly frozen successor entry')
    if (output / 'started.json').exists():
        raise ValueError('Retain this attempt and its paid data; no implicit restart')
    registration = load(output / 'registration.json')
    save(output / 'started.json', {'status': 'started', 'task_ref': registration['source_task_ref']})
    paid = registration.get('prior_paid_costs', {})
    budget = SearchBudget(registration['preparation_compute_limit'], spent=paid.get('compute_cost', 0))
    timing = ExecutionTiming(('restore', 'preparation', 'paired_research', 'method_return', 'confirmation'))
    stage = 'restore'
    try:
        stage = 'preparation'
        if registration.get('previous_unit'):
            agent, frame, scene = complete_paid_preparation(output, registration, budget, timing)
        else:
            with timing.phase('restore'):
                library = load_study_knowledge(registration)
            agent, frame, scene = prepare_successor(output, registration, library, budget, timing)
        if frame is None or not frame['comparison_complete'] or frame['terminal']:
            result = {'status': 'common_pool_already_explained' if frame and frame['terminal']
                else 'preparation_incomplete', 'arms_started': False,
                'compute_cost': budget.spent, 'source_cost_refs': registration['source_cost_refs'],
                'execution_timing': timing.result()}
            save(output / 'receipt.json', result)
            return result
        if not set(registration['operations'].values()) <= set(frame['directions']):
            result = {'status': 'paired_directions_unavailable', 'directions': frame['directions'],
                'arms_started': False, 'compute_cost': budget.spent}
            save(output / 'receipt.json', result)
            return result
        stage = 'paired_research'
        paired = finish_prepared_pair(output, registration, scene, agent, frame, budget.spent, timing)
        save(output / 'paired_receipt.json', paired)
        stage = 'confirmation'
        with timing.phase('confirmation'):
            confirmed = confirm_pair(output, registration)
        stage = 'post_freeze_method_return'
        with timing.phase('method_return'):
            allocation = load(output / 'method_review_allocation.json')
            returned = consume_frozen_pair(output / 'post_freeze_method_return', output,
                compute_limit=max(0, allocation['method_review_compute_limit']
                                  - paired['method_review_compute']))
        common_costs = load(output / 'common_receipt.json')['costs']
        confirmation_costs = confirmed.get('shared_actual_costs', {})
        result = {'status': 'successor_completed', 'paired_result_ref': ref(output / 'paired_receipt.json'),
            'paired_status': paired['status'], 'confirmation_ref': ref(output / 'confirmation/receipt.json'),
            'confirmation_status': confirmed['status'], 'source_cost_refs': registration['source_cost_refs'],
            'post_freeze_method_return': returned,
            'common_costs': common_costs, 'execution_timing': timing.result(), 'model_calls': 0,
            'cumulative_costs': {'compute_cost': paired['compute_cost'] + confirmation_costs.get('compute_cost', 0)
                + returned['new_compute_cost'],
                **{key: common_costs[key] + confirmation_costs.get(key, 0)
                   for key in ('observation_cost', 'measurement_cost')}},
            'scope': registration['scope'], 'general_method_condition_proven': False}
        result['incremental_costs'] = {key: value - paid.get(key, 0)
                                       for key, value in result['cumulative_costs'].items()}
        save(output / 'receipt.json', result)
        return result
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'stage': stage,
            'common_compute_known': budget.spent, 'common_compute_upper_bound': budget.limit,
            'source_cost_refs': registration['source_cost_refs'], 'execution_timing': timing.result()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'register-columns', 'register-continuation', 'run',
                                        'consume-frozen'))
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--knowledge-root', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--prior-review', type=Path)
    parser.add_argument('--repaired-entry', type=Path)
    parser.add_argument('--pair', type=Path)
    parser.add_argument('--compute-limit', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(consume_frozen_pair(args.output.resolve(), args.source_root.resolve(),
        pair_ref=ref(args.pair.resolve()) if args.pair else None, prior_review=args.prior_review,
        compute_limit=args.compute_limit) if args.mode == 'consume-frozen' else
        register_column_successor(args.knowledge_root.resolve(), ref(args.repaired_entry.resolve()),
        args.source_root.resolve(), args.prior_review.resolve(), args.output.resolve())
        if args.mode == 'register-columns' else register_continuation(args.previous.resolve(), args.output.resolve())
        if args.mode == 'register-continuation' else register(args.knowledge_root.resolve(), args.source_root.resolve(),
        args.prior_review.resolve(), args.output.resolve()) if args.mode == 'register'
        else run(args.output.resolve()))
