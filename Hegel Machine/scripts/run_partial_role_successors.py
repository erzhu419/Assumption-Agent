#!/usr/bin/env python3
"""Registered finite object-method successors; no implicit experiment launch.

Archive and fixed Common reconstruct the same method from the same source
observations. PR3 consumes its already formed version. Every arm keeps the full
declared historical library and the same ordinary parent response researcher.
This compares added organization and method reuse, not LLM Code agents.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.common_object_research import CommonObjectResearchAgent
from hegel_machine.cognition.effective_history import HistoryStore
from hegel_machine.cognition.experiment_runtime import (
    ExecutionTiming, python_execution_deadline, run_process_trials,
)
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from run_partial_role_parent import evaluate_reports, load, plain, save


ARMS = ('archive', 'fixed_pr2', 'pr3')
ROOTS = (106029101, 106029102, 106029103)
K0_UNIT = ('C/dependency_method_revision_20260920/development/runs/'
           'rewritten_methods/95902002/checkpoint.json')
LIMITS = {'observation_cost': 512, 'compute_cost': 480000, 'measurement_cost': 6144}
ACTIVITY_SECONDS = 1800
_SHARED = None


def tasks():
    result = []
    for seed, count in zip(ROOTS, (3, 4, 4)):
        rng = Random(seed + 17011)
        result.append({'seed': seed, 'objects': count, 'apparatus_seed': seed,
            'name_seed': seed + 63857, 'wiring_seed': seed + 3709,
            'variant': 'exchange', 'profile': 'confirmation',
            'readout': 'window_total_event', 'actuator_curvature': rng.uniform(.08, .24),
            'relation_radius': rng.uniform(.70, 1.20)})
    return result


def config(seed, registered=None):
    if registered is not None:
        return ResearchConfig(**{**registered, 'seed': seed})
    return ResearchConfig(seed=seed, observation_budget=LIMITS['observation_cost'],
        compute_budget=LIMITS['compute_cost'], minimum_improvement=1e-7,
        max_rounds=6, max_history=4, active_probes=True)


def program_successor_protocol(source_task):
    """Four declared single-factor physical sources; never selected by scores."""
    factors = ('000', '100', '010', '001')
    cases = []
    for index, factor in enumerate(factors):
        seed = 106030201 + index
        task = {**deepcopy(source_task), 'seed': seed, 'apparatus_seed': seed}
        if factor == '100':
            task['relation_radius'] = source_task['relation_radius'] * 1.25
        if factor == '010':
            task['relation_direction'] = ('left' if source_task.get('relation_direction', 'symmetric') == 'right'
                                          else 'right')
        if factor == '001':
            task.update(name_seed=seed + 63857, wiring_seed=seed + 3709,
                        actuator_curvature=source_task['actuator_curvature'] + .05)
        cases.append({'factor': factor, 'task': task})
    # Existing random-plus-control-step terminal. One shared measurement per
    # actual bank, at most coarse+fine. No additional target feedback is legal.
    terminal = {'length': 48, 'trajectories': 1, 'random_seed_offset': 1000003,
        'random_backgrounds': 0, 'background_seed_offset': 2000003,
        'sweep_fractions': [0., .25, .5, .75, 1.], 'mse_tolerance': 1e-7}
    controls = 3 * source_task['objects'] + 1
    terminal_rows = 48 + 10 * controls
    count = len(cases) * len(ARMS)
    return {'scope': 'prospective_single_factor_native_ABC_not_LLM_Code',
        'factor_cases': cases, 'tasks': [item['task'] for item in cases],
        'limits': LIMITS, 'arms': ARMS, 'activity_seconds': ACTIVITY_SECONDS,
        'terminal': terminal, 'research_config': {**asdict(config(cases[0]['task']['seed'])),
                                               'max_candidates': 4},
        'budget_quote': {'training_compute_upper': count * LIMITS['compute_cost'],
            'terminal_compute_upper': len(cases) * len(ARMS) * terminal_rows,
            'total_new_compute_upper': count * LIMITS['compute_cost'] + len(cases) * len(ARMS) * terminal_rows,
            'training_observation_upper': count * LIMITS['observation_cost'],
            'shared_terminal_observation_upper': 2 * len(cases) * terminal_rows,
            'total_new_observation_upper': count * LIMITS['observation_cost'] + 2 * len(cases) * terminal_rows,
            'total_new_measurement_upper': count * LIMITS['measurement_cost']
                + len(cases) * terminal_rows * (3 + 3 * source_task['objects']),
            'training_activity_seconds_upper': count * ACTIVITY_SECONDS,
            'source_costs': 'existing occurrences separate; not included a second time',
            'model_calls': 0},
        'selection_boundary': 'No result-based source replacement; source factors remain laboratory-side',
        'nuisance_contract': 'Every cell has fresh readout/coupling coefficients. P means the relation radius only; '
            'S changes relation direction; M jointly changes names, wiring and instrument curvature.',
        'current_budget_admitted': False}


def program_knowledge(root, repair_ref=None):
    from successor_interval_scene import load_confirmed_knowledge, load_repaired_knowledge
    if repair_ref:
        return load_repaired_knowledge(root, repair_ref)[0]
    return load_confirmed_knowledge(root)


def frozen_program_bank(source, method, proposal):
    """Retain every actually formed source method, including the first stage."""
    record = {'source_concept_id': source['source_concept_id'],
              'source_ref': source['source_ref'],
              **({'source_model_ref': source['source_model_ref']}
                 if source.get('source_model_ref') else {})}
    bodies = [method['research_program'],
              *(item['research_program'] for item in proposal['candidates']),
              proposal['baseline_research_program']]
    result = []
    for body in bodies:
        if not any(item['research_program'] == body for item in result):
            result.append({**record, 'research_program': body})
    return result


def register_programs(args):
    """Prepare an inspectable registration without acquiring or fitting anything."""
    from hegel_machine.cognition.method_action_research import read_method_reference
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Preserve the existing registration')
    raw = load(args.program_source)
    method = load(args.method)
    task_source = load(args.task_source)
    source_task = task_source['task']
    plan = program_successor_protocol(source_task)
    repair_ref = task_source.get('repaired_entry_ref')
    if not repair_ref and task_source.get('method_program_revisions', {}).get('challenger'):
        repair_ref = read_method_reference(task_source['method_program_revisions']['challenger'],
                                          args.task_source.parent)['repaired_entry_ref']
    knowledge = plain(program_knowledge(args.knowledge_root, repair_ref).to_dict())
    if not any(prior['source_concept_id'] == raw['source_concept_id']
               for prior in knowledge['meta_priors']):
        raise ValueError('The source program must belong to the retained executable library')
    pool_ref = {'file': str(args.source_pool.resolve())}
    pool = read_method_reference(pool_ref, output)['scores']
    if not pool:
        raise ValueError('Declare the complete retained candidate pool')
    base = task_source['historical_knowledge_ref']
    store = HistoryStore(base['store'], readonly=True)
    try:
        matches = {task['seed']: [row[0] for row in store.connection.execute(
            'SELECT unit FROM entries WHERE unit LIKE ? LIMIT 1', ('%' + str(task['seed']) + '%',))]
            for task in plan['tasks']}
    finally:
        store.close()
    if any(matches.values()) or source_task['seed'] in matches:
        raise ValueError('A registered source identity is already exposed')
    output.mkdir(parents=True)
    save(output / 'inputs/program_source.json', raw)
    proposal = read_method_reference(raw['proposal_ref'], args.program_source.parent)['proposal']
    frozen_methods = frozen_program_bank(raw, method, proposal)
    for index, record in enumerate(frozen_methods):
        record['formation_ref'] = ({'file': str(args.method.resolve())}
                                   if index == 0 else raw['proposal_ref'])
    save(output / 'inputs/program_method.json', frozen_methods)
    save(output / 'inputs/source_costs.json', load(args.source_costs))
    save(output / 'inputs/raw_source.json', load(args.raw_source) if args.raw_source else None)
    save(output / 'inputs/method.json', {})
    conditions = load(args.method_conditions) if args.method_conditions else []
    if isinstance(conditions, dict):
        conditions = read_method_reference(conditions, args.method_conditions.parent)
    save(output / 'inputs/method_conditions.json', conditions)
    from run_learned_structure_successor import freeze_runtime
    freeze_runtime(output, args.knowledge_root)
    admitted = False
    if args.budget_contract:
        allocation = load(args.budget_contract)['allocations']['abc']
        if (allocation['compute_cost'] < plan['budget_quote']['total_new_compute_upper']
                or allocation['observation_cost'] < plan['budget_quote']['total_new_observation_upper']):
            raise ValueError('The declared ABC allocation cannot fund its frozen full protocol')
        admitted = True
    registration = {**plan, 'registered_at': datetime.now(timezone.utc).isoformat(),
        'method_kind': 'transfer_partition', 'workers': args.workers,
        'knowledge_root': str(args.knowledge_root.resolve()),
        'knowledge_repair_ref': deepcopy(repair_ref),
        'historical_knowledge_ref': base, 'historical_cumulative_costs': None,
        'source_pool_ref': pool_ref, 'source_pool_size': len(pool),
        'shared_source_ref': raw['source_ref'], 'source_costs_ref': {'file': 'inputs/source_costs.json'},
        'unexposed_check': matches,
        'contrast': {'archive': 'ordinary adaptive author reconstructs from the same real source returns',
            'fixed_pr2': 'Common organization with the same executable reconstruction and all ordinary solvers',
            'pr3': 'same organization plus the actually frozen learned program and validated conditions'},
        'knowledge_contract': 'same full library, paid raw sources and full candidate pool; per-target additions stay local',
        'evaluation_contract': 'all twelve outcomes freeze before shared target labels are scored',
        'failure_contract': 'keep all twelve; no retries, replacement, cost-zero failures or post-hoc arm switching',
        'claim_boundary': 'finite native organization contrast, not complete LLM agent superiority',
        'model_calls': 0, 'current_budget_admitted': admitted,
        'budget_contract_ref': {'file': str(args.budget_contract.resolve()), 'json_path': ['allocations', 'abc']}
            if args.budget_contract else None}
    save(output / 'registration.json', registration)
    return registration


def historical_knowledge(reference):
    """Restore one actual checkpoint in memory, never export its large history."""
    store = HistoryStore(reference['store'], readonly=True)
    try:
        row = store.connection.execute('SELECT sequence FROM entries WHERE unit=?',
                                       (reference['unit'],)).fetchone()
        if row is None:
            raise ValueError('The complete declared historical library is unavailable')
        if reference.get('sequence') is not None and row[0] != reference['sequence']:
            raise ValueError('Historical entry is not the registered sequence')
        checkpoint = json.loads(b''.join(store.read_sequence(row[0])))
        return checkpoint['knowledge'], checkpoint['costs'], row[0]
    finally:
        store.close()


def validate_release(parent_root, repair):
    """The release receipt is explicit; missing fields never admit a cohort."""
    parent_root = Path(parent_root).resolve()
    source = load(parent_root / 'source/summary.json')
    terminal = load(parent_root / 'terminal/summary.json')['arms']['source']
    if source['status'] != 'completed' or terminal['prediction_pass'] is not True:
        raise ValueError('The actual source parent and its frozen prediction must pass')
    if (terminal['threshold'] != 1e-6 or not all(value <= 1e-6 for value in
            (terminal['random_mse'], terminal['public_control_step_mse']))):
        raise ValueError('The original object terminal contract must be retained')
    if (repair.get('passed') is not True
            or Path(repair.get('source_parent_root', '')).resolve() != parent_root
            or repair.get('repair_receipt_type') != 'paid_parent_candidate_retention'
            or not repair.get('selected_candidate_origin')
            or repair.get('paid_evidence_rows') != 68
            or repair.get('compute_cost', -1) < 0):
        raise ValueError('An explicit passing repair receipt for this parent is required')
    checked = repair.get('evaluation', {})
    if (checked.get('kind') != 'paid_predecision_confirmation'
            or checked.get('rows') != 24 or checked.get('threshold') != 1e-7
            or checked.get('passed') is not True
            or not checked.get('current_mse', float('inf')) <= 1e-7):
        raise ValueError('The stored candidate must pass the actual old confirmation and paid decision acceptance')
    return {'source_summary_ref': {'file': str(parent_root / 'source/summary.json')},
        'source_terminal_ref': {'file': str(parent_root / 'terminal/summary.json'),
                                'json_path': ['arms', 'source']},
        'source_status': source['status'], 'source_terminal': terminal,
        'repair_scope': 'paid predecision acceptance only; original warm failure is not rewritten or promoted',
        'repair_receipt': deepcopy(repair)}


def register(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Preserve the existing registration and its attempts')
    repair = load(args.repair_receipt)
    release = validate_release(args.source_parent_root, repair)
    raw, method = load(args.raw_source), load(args.method)
    required = {'spec', 'model', 'records', 'previous', 'source_ref', 'model_id'}
    if set(raw) != required or not raw['previous'].get('paired_contract'):
        raise ValueError('Raw source must bind the actual completed paired experiment')
    if method.get('source_ref') != raw['source_ref'] or method.get('model_id') != raw['model_id']:
        raise ValueError('Raw reconstruction and frozen method must identify the same source')
    if not any(item.get('status') == 'discriminating'
               for item in method.get('dependency_methods', [])):
        raise ValueError('The source has not formed a supported experiment method')
    # Source role bindings are reconstruction premises, not target answers.
    # Keep the actual method and every failed recipe; do not install its schema.
    method = {key: deepcopy(method[key]) for key in
              ('model_id', 'source_ref', 'dependency_methods', 'status') if key in method}
    reference = {'store': str(args.history_store.resolve()), 'unit': K0_UNIT,
                 'json_path': ['knowledge']}
    knowledge, old_costs, sequence = historical_knowledge(reference)
    reference['sequence'] = sequence
    store = HistoryStore(reference['store'], readonly=True)
    try:
        prior_names = {seed: [row[0] for row in store.connection.execute(
            'SELECT unit FROM entries WHERE unit LIKE ? LIMIT 3', ('%' + str(seed) + '%',))]
            for seed in ROOTS}
    finally:
        store.close()
    if any(prior_names.values()):
        raise ValueError('A declared target appears in the historical index; do not replace it silently')
    source_costs = load(args.source_costs)
    output.mkdir(parents=True)
    save(output / 'inputs/raw_source.json', raw)
    save(output / 'inputs/method.json', method)
    save(output / 'inputs/source_costs.json', source_costs)
    from run_observation_action_study import freeze
    freeze(output)
    (output / 'runtime/scripts').mkdir(parents=True, exist_ok=True)
    for name in ('run_partial_role_successors.py', 'run_partial_role_parent.py'):
        write_text_incrementally(output / 'runtime/scripts' / name,
                                 (ROOT / 'scripts' / name).read_text())
    registration = {'registered_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'three new apparatus roots; finite native organization and source-method reuse',
        'arms': ARMS, 'tasks': tasks(), 'limits': LIMITS,
        'research_config': asdict(config(ROOTS[0])), 'activity_seconds': ACTIVITY_SECONDS,
        'workers': args.workers, 'release': release,
        'raw_source_input_ref': {'file': str(args.raw_source.resolve())},
        'method_input_ref': {'file': str(args.method.resolve())},
        'shared_source_ref': raw['source_ref'], 'historical_knowledge_ref': reference,
        'historical_knowledge_counts': {key: len(value) for key, value in knowledge.items()
                                      if isinstance(value, (dict, list))},
        'historical_cumulative_costs': old_costs,
        'historical_cost_scope': 'already paid; counted once by original source reference',
        'source_costs_ref': {'file': 'inputs/source_costs.json'},
        'unknown_costs': 'retained in source ledger; no complete net-cost claim from target costs alone',
        'unexposed_check': {'historical_index_matches': prior_names,
            'scope': 'index names checked before first acquisition; not all historical JSON bodies'},
        'terminal': {'length': 48, 'trajectories': 4, 'random_seed_offset': 1000003,
            'random_backgrounds': 2, 'background_seed_offset': 2000003,
            'sweep_fractions': [0., .25, .5, .75, 1.], 'mse_tolerance': 1e-6},
        'contrast': {'archive': 'complete old adaptive author plus actual raw-source reconstruction',
            'fixed_pr2': 'same author and raw reconstruction plus Common paired child organization',
            'pr3': 'same Common author and old library plus frozen source method; no reconstruction'},
        'failure_contract': 'All nine remain in denominator; no retry, seed replacement or extra allowance',
        'evaluation_contract': 'Freeze all nine outcomes before any target truth is evaluated; no feedback',
        'knowledge_contract': 'full K0 copied in memory, never pruned; target libraries do not cross between jobs',
        'claim_boundary': 'not complete Hegel-versus-Code or an LLM comparison',
        'model_calls': 0}
    save(output / 'registration.json', registration)
    return registration


def knowledge_changes(before, after):
    """Record actual changed entries; unchanged old libraries stay by reference."""
    result = {}
    for key, value in after.items():
        old = before.get(key)
        if value == old:
            continue
        if isinstance(value, list) and isinstance(old, list):
            result[key] = {'length': len(value), 'updated': {str(i): item
                for i, item in enumerate(value) if i >= len(old) or item != old[i]}}
        elif isinstance(value, dict) and isinstance(old, dict):
            result[key] = {'updated': {k: v for k, v in value.items() if k not in old or v != old[k]},
                           'removed': [k for k in old if k not in value]}
        else:
            result[key] = {'value': value}
    return result


def apply_knowledge_changes(knowledge, changes):
    """Restore recorded additions while leaving the shared historical library intact."""
    knowledge = deepcopy(knowledge)
    for key, delta in changes.items():
        if 'value' in delta:
            knowledge[key] = deepcopy(delta['value'])
        elif 'length' in delta:
            values = knowledge[key][:delta['length']]
            values.extend([None] * (delta['length'] - len(values)))
            for index, value in delta['updated'].items():
                values[int(index)] = deepcopy(value)
            knowledge[key] = values
        else:
            for removed in delta['removed']:
                knowledge[key].pop(removed)
            knowledge[key].update(deepcopy(delta['updated']))
    return knowledge


def train_one(job):
    seed, arm = job
    state = _SHARED
    output = state['output'] / 'runs' / str(seed) / arm
    output.mkdir(parents=True, exist_ok=False)
    timing = ExecutionTiming(('restore', 'research', 'save'))
    task = next(item for item in state['registration']['tasks'] if item['seed'] == seed)
    with timing.phase('restore'):
        library = KnowledgeLibrary.from_dict(deepcopy(state['knowledge']))
        extra = {}
        if state['registration'].get('method_kind') == 'transfer_partition':
            from hegel_machine.cognition.method_conditions import native_condition_bindings
            extra = {'program_sources': [state['program_source']],
                'learned_programs': state['program_method'] if arm == 'pr3' else [],
                'retained_pool': state['retained_pool'],
                'condition_sources': state['method_conditions'],
                'native_method_conditions': native_condition_bindings(state['method_conditions'], state['output'])
                    if arm == 'pr3' else []}
        agent = CommonObjectResearchAgent(config(seed, state['registration'].get('research_config')),
            measurement_budget=state['registration']['limits']['measurement_cost'],
            library=library, workspace=output / 'common', organization=arm,
            source_task_ref={'file': str(state['output'] / 'registration.json'),
                'json_path': ['tasks', state['registration']['tasks'].index(task)]},
            raw_sources=[deepcopy(state['raw'])] if state['raw'] else [],
            learned_methods=[deepcopy(state['method'])] if arm == 'pr3' and state['method'] else [], **extra)
    with AnonymousObjectLab(**task) as laboratory:
        try:
            with timing.phase('research'), python_execution_deadline(ACTIVITY_SECONDS, activity=True):
                report = agent.run(laboratory)
        except Exception as error:
            import traceback
            with timing.phase('save'):
                save(output / 'failure.json', {'error': repr(error),
                    'traceback': traceback.format_exc(),
                    'paid_state_ref': agent.failure_evidence_ref,
                    'confirmed_compute_lower_bound': agent.failure_compute_lower_bound,
                    'actual_acquisition_costs': laboratory.costs,
                    'new_acquisitions': laboratory.acquisitions,
                    'compute_cost': None, 'compute_upper_bound': LIMITS['compute_cost'],
                    'compute_scope': 'native run did not return its account',
                    'execution_timing': timing.result()})
            summary = {'seed': seed, 'arm': arm, 'status': 'failed',
                'frozen': True, 'predictor_available': False,
                'costs': {**laboratory.costs, 'compute_cost': None},
                'confirmed_compute_lower_bound': agent.failure_compute_lower_bound,
                'compute_unknown_upper_bound': LIMITS['compute_cost'],
                'failure_ref': {'file': 'failure.json'}, 'execution_timing': timing.result()}
        else:
            with timing.phase('save'):
                save(output / 'response_report.json', report)
                save(output / 'knowledge_changes.json', {
                    'base_ref': state['registration']['historical_knowledge_ref'],
                    'changes': knowledge_changes(state['knowledge'], plain(library.to_dict()))})
                available = report.get('program') is not None
                if available:
                    save(output / 'frozen_predictor.json', {key: report[key] for key in
                        ('program', 'fallback', 'selected_channels', 'selected_kind', 'status')})
            summary = {'seed': seed, 'arm': arm, 'status': report['status'],
                'frozen': True, 'predictor_available': available,
                'confirmation_loss': report.get('confirmation_loss'), 'costs': report['costs'],
                'source_reconstruction_cost': sum(item['compute_cost']
                    for item in [*agent.source_reconstruction_receipts, *agent.program_reconstruction_receipts])
                    + sum(item['new_compute_cost'] for item in agent.condition_reconstruction_receipts),
                'common_child_compute_cost': sum(item.common.costs['compute_cost']
                    for item in agent.common_object_consumers),
                'nested_cost_scope': 'included in parent compute, never added again',
                'report_ref': {'file': 'response_report.json'},
                'execution_timing': timing.result()}
        save(output / 'summary.json', summary)
    return {'seed': seed, 'arm': arm, 'summary_ref': str(output / 'summary.json')}


def run(output):
    global _SHARED
    output = Path(output).resolve()
    registration = load(output / 'registration.json')
    if registration.get('method_kind') == 'transfer_partition' and not registration.get('current_budget_admitted'):
        raise ValueError('The new twelve-run budget is not admitted; registration does not launch research')
    jobs = [(task['seed'], arm) for task in registration['tasks'] for arm in ARMS]
    missing = []
    for seed, arm in jobs:
        directory = output / 'runs' / str(seed) / arm
        if (directory / 'summary.json').exists():
            continue
        if directory.exists():
            save(directory / 'summary.json', {'seed': seed, 'arm': arm,
                'status': 'interrupted_without_final_account', 'frozen': True,
                'predictor_available': False,
                'costs': {key: None for key in LIMITS},
                'unknown_upper_bounds': LIMITS,
                'reason': 'Retain interrupted attempt; do not rerun within this cohort'})
        else:
            missing.append((seed, arm))
    if missing:
        timing = ExecutionTiming(('restore_once', 'trials', 'save'))
        with timing.phase('restore_once'):
            if registration.get('knowledge_root'):
                knowledge = plain(program_knowledge(Path(registration['knowledge_root']),
                    registration.get('knowledge_repair_ref')).to_dict())
            else:
                knowledge, _, _ = historical_knowledge(registration['historical_knowledge_ref'])
            _SHARED = {'output': output, 'registration': registration, 'knowledge': knowledge,
                'raw': load(output / 'inputs/raw_source.json'),
                'method': load(output / 'inputs/method.json')}
            if registration.get('method_kind') == 'transfer_partition':
                from hegel_machine.cognition.method_action_research import read_method_reference
                from hegel_machine.cognition.object_continuation import decode_entry
                from hegel_machine.cognition.language import Program
                from hegel_machine.cognition.models import Candidate
                pool = read_method_reference(registration['source_pool_ref'], output)['scores']
                entries = []
                for index, row in enumerate(pool):
                    payload = read_method_reference(row['candidate_ref'], output)
                    entries.append(decode_entry(payload) if 'candidate' in payload else {
                        'candidate': Candidate(f'retained:{index}', Program.from_dict(payload['program']), 'research_program'),
                        'kind': 'research_program', 'family_id': 'T03',
                        'research_program': payload['research_program'],
                        'training_loss': payload['training_loss'], 'validation_loss': payload['validation_loss']})
                    entries[-1]['retained_candidate_ref'] = row['candidate_ref']
                _SHARED.update(program_source=load(output / 'inputs/program_source.json'),
                    program_method=load(output / 'inputs/program_method.json'), retained_pool=entries,
                    method_conditions=load(output / 'inputs/method_conditions.json'))
        try:
            with timing.phase('trials'):
                run_process_trials(train_one, missing, registration['workers'])
        finally:
            _SHARED = None
            with timing.phase('save'):
                save(output / 'run_timing.json', timing.result())
    summaries = [load(output / 'runs' / str(seed) / arm / 'summary.json') for seed, arm in jobs]
    save(output / 'training_summary.json', {'planned': len(jobs), 'outcomes': summaries,
        'all_training_frozen': all(item['frozen'] for item in summaries),
        'terminal_started': (output / 'terminal').exists(),
        'source_costs_ref': registration['source_costs_ref']})


def evaluate(output):
    output = Path(output).resolve()
    registration = load(output / 'registration.json')
    all_summaries = {(task['seed'], arm): load(output / 'runs' / str(task['seed']) / arm / 'summary.json')
                     for task in registration['tasks'] for arm in ARMS}
    if not all(item['frozen'] for item in all_summaries.values()):
        raise ValueError('All nine research outcomes must be frozen before terminal evaluation')
    results = []
    for task in registration['tasks']:
        directory = output / 'terminal' / str(task['seed'])
        if (directory / 'summary.json').exists():
            results.append(load(directory / 'summary.json'))
            continue
        if directory.exists():
            raise ValueError('Interrupted terminal observations are paid; inspect them instead of repurchasing')
        reports = {arm: load(output / 'runs' / str(task['seed']) / arm / 'frozen_predictor.json')
            for arm in ARMS if all_summaries[task['seed'], arm]['predictor_available']}
        terminal = deepcopy(registration['terminal'])
        terminal['random_seed'] = task['seed'] + terminal.pop('random_seed_offset')
        terminal['background_seed'] = task['seed'] + terminal.pop('background_seed_offset')
        if reports:
            result = evaluate_reports(task, reports, terminal, directory)
        else:
            result = {'arms': {}, 'shared_actual_costs': {key: 0 for key in LIMITS},
                      'reason': 'No predictor delivered by any arm; no terminal purchase'}
        for arm in ARMS:
            if arm not in result['arms']:
                result['arms'][arm] = {'prediction_pass': False, 'prediction_measured': False,
                    'reason': 'No returned predictor', 'native_status': all_summaries[task['seed'], arm]['status']}
        result['seed'] = task['seed']
        save(directory / 'summary.json', result)
        results.append(result)
    target = {arm: {'tasks': len(registration['tasks']),
        'prediction_passes': sum(row['arms'][arm]['prediction_pass'] for row in results),
        'native_completed': sum(all_summaries[task['seed'], arm]['status'] == 'completed'
                                for task in registration['tasks']),
        'known_costs': {key: sum(row['costs'][key] for (seed, name), row in all_summaries.items()
                                  if name == arm and row['costs'].get(key) is not None)
                        for key in LIMITS},
        'unknown_cost_runs': [seed for (seed, name), row in all_summaries.items()
                              if name == arm and any(row['costs'].get(key) is None for key in LIMITS)]}
        for arm in ARMS}
    summary = {'scope': registration['scope'], 'targets': target, 'paired_terminal': results,
        'source_costs_ref': registration['source_costs_ref'],
        'old_library_costs': registration['historical_cumulative_costs'],
        'old_library_ref': registration['historical_knowledge_ref'],
        'source_and_unknown_costs_kept_separate': True,
        'full_lifecycle_net_benefit': 'not inferred automatically; reconcile source and unknown ledger',
        'claim_boundary': registration['claim_boundary'], 'model_calls': 0}
    summary['lifecycle_costs'] = lifecycle_readout(output, registration, all_summaries, results)
    save(output / 'summary.json', summary)
    return summary


def lifecycle_readout(output, registration, training, terminal):
    from hegel_machine.cognition.research_lifecycle_costs import summarize_research_lifecycle
    ledger = load(output / registration['source_costs_ref']['file'])
    occurrences = deepcopy(ledger.get('occurrences', []))
    if not occurrences:
        # The old free-form bill remains addressed. Its unallocated source
        # costs cannot cancel between A/B/C or become an apparent zero.
        occurrences = [{'occurrence_ref': {**registration['source_costs_ref'], 'unresolved_arm': arm},
            'scope': 'incremental_execution', 'phase': 'source', 'charged_to': [arm],
            'costs': {}, 'reason': 'source_increment_and_attribution_not_resolved'} for arm in ARMS]
    outcomes = []
    for (seed, arm), row in training.items():
        costs = {**row['costs'], 'model_calls': 0, 'reported_tokens': 0}
        interval = {}
        if costs.get('compute_cost') is None:
            interval['compute_cost'] = [row.get('confirmed_compute_lower_bound', 0), row.get('compute_unknown_upper_bound',
                row.get('unknown_upper_bounds', {}).get('compute_cost'))]
        occurrences.append({'occurrence_ref': {'file': str(output / 'runs' / str(seed) / arm / 'summary.json')},
            'scope': 'incremental_execution', 'phase': 'successor', 'charged_to': [arm],
            'failed': row['status'] != 'completed', 'costs': costs, 'cost_intervals': interval})
        evaluated = next(item for item in terminal if item['seed'] == seed)['arms'][arm]
        outcomes.append({'arm': arm, 'seed': seed, 'predictor_available': row['predictor_available'],
            'prediction_pass': evaluated['prediction_pass'],
            'prediction_measured': evaluated.get('prediction_measured', row['predictor_available'])})
    for row in terminal:
        shared = {**row['shared_actual_costs'], 'model_calls': 0, 'reported_tokens': 0}
        prediction_compute = shared.pop('compute_cost')
        shared['compute_cost'] = 0
        occurrences.append({'occurrence_ref': {'file': str(output / 'terminal' / str(row['seed']) / 'summary.json')},
            'scope': 'incremental_execution', 'phase': 'terminal', 'charged_to': list(ARMS),
            'costs': shared})
        predicted = [arm for arm in ARMS if training[row['seed'], arm]['predictor_available']]
        for arm in predicted:
            occurrences.append({'occurrence_ref': {'file': str(output / 'terminal' / str(row['seed']) / 'summary.json'),
                'json_path': ['arms', arm]}, 'scope': 'incremental_execution', 'phase': 'terminal',
                'charged_to': [arm], 'costs': {'compute_cost': prediction_compute / len(predicted),
                    'observation_cost': 0, 'measurement_cost': 0, 'model_calls': 0, 'reported_tokens': 0}})
    return summarize_research_lifecycle(occurrences, arms=ARMS, outcomes=outcomes)


def freeze_terminal_reader(output, terminal_script):
    """Freeze the paid-observation retention fix without changing training."""
    output, terminal_script = Path(output).resolve(), Path(terminal_script).resolve()
    registration = load(output / 'registration.json')
    for task in registration['tasks']:
        for arm in ARMS:
            summary = load(output / 'runs' / str(task['seed']) / arm / 'summary.json')
            if not summary['frozen']:
                raise ValueError('Finish all nine training outcomes before freezing terminal execution')
    directory = output / 'terminal_runtime/scripts'
    launcher = directory / 'evaluate.py'
    if launcher.exists():
        return launcher
    if (output / 'terminal').exists():
        raise ValueError('Retain the existing terminal attempt; do not replace its evaluator')
    directory.mkdir(parents=True, exist_ok=False)
    write_text_incrementally(directory / 'run_partial_role_parent.py', terminal_script.read_text())
    write_text_incrementally(launcher,
        "from pathlib import Path\nimport sys\n"
        "output = Path(__file__).resolve().parents[2]\n"
        "sys.path[:0] = [str(output / 'runtime/src'), "
        "str(output / 'terminal_runtime/scripts'), str(output / 'runtime/scripts')]\n"
        "from run_partial_role_successors import evaluate\n"
        "evaluate(output)\n")
    save(output / 'terminal_runtime/change_note.json', {
        'change': 'Retain each actually acquired terminal sensor bank with observations, histories '
                  'and transitions once; share references across all frozen predictors.',
        'training_runtime_ref': {'file': str(output / 'runtime/scripts/run_partial_role_successors.py')},
        'evaluation_source_ref': {'file': str(terminal_script)},
        'evaluation_frozen_ref': {'file': str(directory / 'run_partial_role_parent.py')},
        'unchanged': ['training outcomes', 'registered probes and seeds', 'predictors',
                      'selection', 'prediction metrics', 'scientific fee rules'],
        'registered_at': datetime.now(timezone.utc).isoformat()})
    return launcher


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=('register', 'register-programs', 'run', 'evaluate'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-parent-root', type=Path)
    parser.add_argument('--repair-receipt', type=Path)
    parser.add_argument('--raw-source', type=Path)
    parser.add_argument('--method', type=Path)
    parser.add_argument('--source-costs', type=Path)
    parser.add_argument('--program-source', type=Path)
    parser.add_argument('--source-pool', type=Path)
    parser.add_argument('--knowledge-root', type=Path)
    parser.add_argument('--task-source', type=Path)
    parser.add_argument('--method-conditions', type=Path,
        help='Actual paired_condition_research list or its retained file/json_path reference')
    parser.add_argument('--budget-contract', type=Path,
        help='Authorized separate total protocol with an allocations.abc computation and observation allowance')
    parser.add_argument('--history-store', type=Path,
        default=Path('/mnt/d/Hegel-History-Archive-20261012/effective_history.sqlite3'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--terminal-script', type=Path,
        help='Evaluate only: separately freeze the observation-retaining evaluator; keep training frozen')
    args = parser.parse_args()
    if args.terminal_script is not None:
        if args.mode != 'evaluate':
            parser.error('--terminal-script applies only to the frozen terminal stage')
        launcher = freeze_terminal_reader(args.output, args.terminal_script)
        os.execv(sys.executable, [sys.executable, str(launcher)])
    if args.mode in {'register', 'register-programs'}:
        required = (('source_parent_root', 'repair_receipt', 'raw_source', 'method', 'source_costs')
            if args.mode == 'register' else
            ('program_source', 'source_pool', 'knowledge_root', 'task_source', 'method', 'source_costs'))
        if any(getattr(args, name) is None for name in required):
            parser.error('register requires explicit source, repair, raw material, method and cost receipts')
        if not 1 <= args.workers <= 8:
            parser.error('use one to eight physical workers')
        (register if args.mode == 'register' else register_programs)(args)
    else:
        frozen = args.output.resolve() / 'runtime/scripts' / Path(__file__).name
        if Path(__file__).resolve() != frozen:
            if not frozen.exists():
                parser.error('The cohort must be registered before execution')
            os.execv(sys.executable, [sys.executable, str(frozen), *sys.argv[1:]])
        globals()[args.mode](args.output)


if __name__ == '__main__':
    main()
