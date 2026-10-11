#!/usr/bin/env python3
"""Compare exact-only and revised-band source knowledge on twelve new conditions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]
from scripts import replay_response_condition_retention as replay
from scripts import run_response_pending_band_revision as previous

OUTPUT, PHASES = replay.OUTPUT, ('before', 'after')
SIGNALS, CONDITIONS = ((8.25, 16.25), (9.25, 18.25)), ('source_gain', 'shifted_gain')
read, write, aggregate = previous.read, previous.write, previous.aggregate
load_runtime, load_execution = previous.load_runtime, previous.load_execution


def prepare(output):
    assert not (output / 'protocol.json').exists(), 'Preserve prepared inputs'
    for folder, modules, source in (
            ('runtime', previous.MODULES, ROOT / 'src/hegel_machine/cognition'),
            ('execution', previous.EXPORTERS, ROOT / 'scripts')):
        directory = output / folder
        directory.mkdir(parents=True, exist_ok=True)
        for part in modules:
            shutil.copyfile(source / (part + '.py'), directory / (part + '.py'))
    load_runtime(output / 'runtime')
    execution = load_execution(output / 'execution')
    from scripts.run_response_matched_adoption import prepare_bundle
    from scripts.run_response_composition_transfer import declarations
    from hegel_machine.cognition.world_response_transfer import combine_response_knowledge, export_response_knowledge

    cases, inventory, source_rows, limits = {}, [], [], None
    for cohort, source_case in replay.CASES:
        base = replay.OUTPUT / 'replays' / cohort
        source = base / source_case / 'frozen_numeric'
        protocol = read(base / 'protocol.json')
        if limits is None:
            limits = protocol['limits']
        assert limits == protocol['limits'], 'Source limits must agree'
        config = protocol['cases'][source_case]
        source_id = cohort + '__' + source_case
        inherited = read(base / 'sources' / config['source_id'] / 'numeric_bundle.json')
        inputs = read(source / 'final_export_inputs.json')
        exact = read(source / 'learned_response_knowledge.json')
        rebuilt = export_response_knowledge(inputs['state'], inputs['estimators'], inputs['source_id'])
        old = json.loads(inherited['files'][execution.KNOWLEDGE])
        archives = {'before': combine_response_knowledge(old, exact),
                    'after': combine_response_knowledge(old, rebuilt)}
        copied = output / 'source_inputs' / source_id
        for name in ('final_export_inputs.json', 'final_controller.py',
                     'learned_response_knowledge.json', 'brief.json', 'replay_comparison.json'):
            copied.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, copied / name)
        write(copied / 'inherited_bundle.json', inherited)
        write(copied / 'source_config.json', config)
        write(copied / 'rebuilt_response_knowledge.json', rebuilt)
        source_rows.append(read(source / 'brief.json'))
        controller = (source / 'final_controller.py').read_text()
        package = deepcopy(inherited)
        package['files'][execution.METHOD] = controller
        package = prepare_bundle(package)
        assert declarations(package['files'][execution.METHOD]) == declarations(controller)
        assert list(declarations(controller)) == inputs['estimators']
        bundles = {}
        for phase, knowledge in archives.items():
            bundles[phase] = deepcopy(package)
            bundles[phase]['files'][execution.KNOWLEDGE] = json.dumps(knowledge)
            write(output / phase / 'sources' / source_id / 'numeric_bundle.json', bundles[phase])
        nonknowledge = lambda bundle: {**bundle, 'files': {
            key: value for key, value in bundle['files'].items() if key != execution.KNOWLEDGE}}
        assert nonknowledge(bundles['before']) == nonknowledge(bundles['after'])
        inventory.append({'source_id': source_id, 'cohort': cohort, 'source_case': source_case,
            'replay_artifact': str(source), 'copied_inputs': str(copied), 'source_gain': config['gain'],
            'inherited_models': len(old['models']), 'exact_models': len(exact['models']),
            'rebuilt_models': len(rebuilt['models']),
            'combined_models': {phase: len(archive['models']) for phase, archive in archives.items()},
            'source_response_count': rebuilt['source_response_count'],
            'source_execution_costs': source_rows[-1]['costs'],
            'exact_export_compute': exact['compute_cost'], 'rebuilt_export_compute': rebuilt['compute_cost'],
            'revised_model_count': rebuilt.get('revised_model_count', 0),
            'band_revisions': rebuilt.get('band_revisions', []), 'nonknowledge_inputs_equal': True})
        for index, signals in enumerate(SIGNALS, 1):
            for condition, factor in zip(CONDITIONS, (1., 1.01)):
                target = deepcopy(config)
                target.update(source_id=source_id, condition=condition, source_cohort=cohort,
                    retention_source_case=source_case, target_object_pair=index,
                    gain=config['gain'] * factor, gain_factor=config.get('gain_factor', 1.) * factor,
                    gain_relative_to_replay_source=factor)
                target['signal_values'].update({'4': signals[0], '5': signals[1]})
                cases[f'{source_id}__objects_{index}__{condition}'] = target
    for phase in PHASES:
        write(output / phase / 'protocol.json', {'cases': cases, 'limits': limits,
            'arms': ['frozen_numeric'], 'knowledge_phase': phase, 'planned_executions': len(cases)})
    write(output / 'source_inventory.json', inventory)
    write(output / 'protocol.json', {'cases': cases, 'limits': limits, 'phases': PHASES,
        'planned_executions': 24, 'planned_pairs': 12, 'maximum_workers': 2, 'model_calls': 0,
        'runtime_modules': list(previous.MODULES), 'execution_exporters': list(previous.EXPORTERS),
        'source': str(replay.OUTPUT / 'replays'), 'source_count': len(inventory),
        'source_execution': aggregate(source_rows), 'target_signals': SIGNALS,
        'intervention': 'Both phases use the same frozen runtime, final source controller and declarations. Before combines inherited knowledge with the replay exact export; after combines inherited knowledge with current export from the same final paid source inputs.',
        'cost_scope': 'Successor execution costs, previously paid source execution and source re-export compute are separate. Terminal exports only provide diagnostics in both phases.',
        'scope': 'Three completed development branches, two new object pairs and two gains; not independent mechanisms or an LLM adoption study. Zero new source execution or API/model calls.'})
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    summarize(output)
    print(json.dumps({'status': 'prepared', 'sources': len(inventory), 'planned': 24}), flush=True)


def summarize(output):
    protocol, rows, pairs = read(output / 'protocol.json'), [], []
    for case, config in protocol['cases'].items():
        found = {}
        for phase in PHASES:
            path = output / phase / case / 'frozen_numeric/brief.json'
            if path.exists():
                found[phase] = read(path)
                rows.append({**found[phase], 'phase': phase, 'target_object_pair': config['target_object_pair']})
        if len(found) == 2:
            before, after = found['before'], found['after']
            pairs.append({'case': case, 'source_id': config['source_id'], 'condition': config['condition'],
                'target_object_pair': config['target_object_pair'],
                'quality': {phase: {key: found[phase][key] for key in (
                    'success', 'status', 'requested_point_confirmed', 'unsupported_positive_verdicts')} for phase in PHASES},
                'actions': {phase: found[phase]['costs']['environment_steps'] for phase in PHASES},
                'research_compute': {phase: found[phase]['costs']['research_compute_units'] for phase in PHASES},
                'actions_difference': after['costs']['environment_steps'] - before['costs']['environment_steps'],
                'compute_difference': after['costs']['research_compute_units'] - before['costs']['research_compute_units']})
    errors = read(output / 'batch_errors.json') if (output / 'batch_errors.json').exists() else []
    inventory = read(output / 'source_inventory.json')
    summary = {'status': 'execution_failures' if errors else 'finished' if len(rows) == 24 else 'running' if rows else 'prepared',
        'completed': len(rows), 'planned': 24, 'paired_completed': len(pairs), 'planned_pairs': 12,
        'source_count': protocol['source_count'], 'source_execution': protocol['source_execution'],
        'source_reexport_compute': {phase: sum(item[key] for item in inventory)
            for phase, key in (('before', 'exact_export_compute'), ('after', 'rebuilt_export_compute'))},
        'source_scope': protocol['scope'], 'new_source_executions': 0,
        'model_calls': sum(row['costs'].get('model_calls', 0) for row in rows),
        'overall': {phase: aggregate([row for row in rows if row['phase'] == phase]) for phase in PHASES},
        'conditions': {condition: {phase: aggregate([row for row in rows
            if row['phase'] == phase and row['condition'] == condition]) for phase in PHASES} for condition in CONDITIONS},
        'paired': pairs, 'execution_errors': errors}
    write(output / 'brief.json', rows)
    write(output / 'summary.json', summary)
    return summary


def run(output):
    jobs = [(phase, case) for case in read(output / 'protocol.json')['cases'] for phase in PHASES]
    assert not any((output / phase / case / 'frozen_numeric').exists() for phase, case in jobs), 'Preserve started runs'
    def launch(phase, case):
        log = output / 'logs' / phase / (case + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as stream:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--output', str(output),
                '--phase', phase, '--case', case], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        return result.returncode, str(log)
    errors = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(launch, *job): job for job in jobs}
        for future in as_completed(pending):
            phase, case = pending[future]
            code, log = future.result()
            if code:
                errors.append({'phase': phase, 'case': case, 'returncode': code, 'log': log})
            write(output / 'batch_errors.json', errors)
            summary = summarize(output)
            print(json.dumps({'phase': phase, 'case': case, 'returncode': code,
                              'completed': summary['completed']}), flush=True)
    if errors:
        raise SystemExit('Recorded execution failures')


def execute(output, phase, case):
    load_runtime(output / 'runtime')
    execution = load_execution(output / 'execution')
    from hegel_machine.cognition import world_response_transfer as transfer
    original = transfer.export_response_knowledge
    def capture(state, estimators, source_id):
        write(output / phase / case / 'frozen_numeric/final_paid_inputs.json', {
            'response_settings': deepcopy(state.get('response_settings', [])),
            'research_table': deepcopy(state.get('research_table', {})),
            'estimators': deepcopy(estimators), 'source_id': source_id})
        return original(state, estimators, source_id)
    transfer.export_response_knowledge = capture
    execution.run_case(output / phase, case, 'frozen_numeric')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    for option in ('prepare', 'run', 'summarize'):
        parser.add_argument('--' + option, action='store_true')
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--case')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.case and args.phase:
        execute(output, args.phase, args.case)
    elif args.prepare:
        prepare(output)
    elif args.run:
        run(output)
    elif args.summarize:
        print(json.dumps(summarize(output), indent=2))
    else:
        parser.error('Choose --prepare, --run, --summarize, or --phase with --case')
