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


def config(seed):
    return ResearchConfig(seed=seed, observation_budget=LIMITS['observation_cost'],
        compute_budget=LIMITS['compute_cost'], minimum_improvement=1e-7,
        max_rounds=6, max_history=4, active_probes=True)


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
        agent = CommonObjectResearchAgent(config(seed), measurement_budget=LIMITS['measurement_cost'],
            library=library, workspace=output / 'common', organization=arm,
            source_task_ref={'file': str(state['output'] / 'registration.json'),
                'json_path': ['tasks', state['registration']['tasks'].index(task)]},
            raw_sources=[deepcopy(state['raw'])],
            learned_methods=[deepcopy(state['method'])] if arm == 'pr3' else [])
    with AnonymousObjectLab(**task) as laboratory:
        try:
            with timing.phase('research'), python_execution_deadline(ACTIVITY_SECONDS, activity=True):
                report = agent.run(laboratory)
        except Exception as error:
            with timing.phase('save'):
                save(output / 'failure.json', {'error': repr(error),
                    'actual_acquisition_costs': laboratory.costs,
                    'new_acquisitions': laboratory.acquisitions,
                    'compute_cost': None, 'compute_upper_bound': LIMITS['compute_cost'],
                    'compute_scope': 'native run did not return its account',
                    'execution_timing': timing.result()})
            summary = {'seed': seed, 'arm': arm, 'status': 'failed',
                'frozen': True, 'predictor_available': False,
                'costs': {**laboratory.costs, 'compute_cost': None},
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
                    for item in agent.source_reconstruction_receipts),
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
            knowledge, _, _ = historical_knowledge(registration['historical_knowledge_ref'])
            _SHARED = {'output': output, 'registration': registration, 'knowledge': knowledge,
                'raw': load(output / 'inputs/raw_source.json'),
                'method': load(output / 'inputs/method.json')}
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
    save(output / 'summary.json', summary)
    return summary


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
    parser.add_argument('mode', choices=('register', 'run', 'evaluate'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-parent-root', type=Path)
    parser.add_argument('--repair-receipt', type=Path)
    parser.add_argument('--raw-source', type=Path)
    parser.add_argument('--method', type=Path)
    parser.add_argument('--source-costs', type=Path)
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
    if args.mode == 'register':
        required = ('source_parent_root', 'repair_receipt', 'raw_source', 'method', 'source_costs')
        if any(getattr(args, name) is None for name in required):
            parser.error('register requires explicit source, repair, raw material, method and cost receipts')
        if not 1 <= args.workers <= 8:
            parser.error('use one to eight physical workers')
        register(args)
    else:
        frozen = args.output.resolve() / 'runtime/scripts' / Path(__file__).name
        if Path(__file__).resolve() != frozen:
            if not frozen.exists():
                parser.error('The cohort must be registered before execution')
            os.execv(sys.executable, [sys.executable, str(frozen), *sys.argv[1:]])
        globals()[args.mode](args.output)


if __name__ == '__main__':
    main()
