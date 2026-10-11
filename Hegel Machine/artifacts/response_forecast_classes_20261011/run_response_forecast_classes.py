#!/usr/bin/env python3
"""Matched finite-forecast selection on the existing 26 retained-archive cases."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
from statistics import pvariance
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]
from scripts import run_response_revision_successors as previous

SOURCE = ROOT / 'artifacts/response_exhausted_stage_20261011'
OUTPUT = ROOT / 'artifacts/response_forecast_classes_20261011'
PHASES = ('before', 'after')
MODULES = ('world_experiment_choice', *previous.MODULES)
CHANGED = {'world_experiment_choice', 'world_response_hypotheses', 'world_response_consequences'}
PILOT_SOURCES = ('cubic_original_4__gain_higher', 'cubic_shifted_2__gain_higher',
                 'cubic_shifted_4__gain_higher')
read, write = previous.read, previous.write


def load_runtime(directory):
    import hegel_machine.cognition as package
    for part in MODULES:
        name = 'hegel_machine.cognition.' + part
        spec = importlib.util.spec_from_file_location(name, directory / (part + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        setattr(package, part, module)
        spec.loader.exec_module(module)


def diagnosis(output):
    """Reproduce actual first decisions from paid prefixes, then quotient columns."""
    load_runtime(output / 'runtime/before')
    from hegel_machine.cognition.world_response_transfer import reuse_response_knowledge
    from hegel_machine.cognition.world_response_consequences import _contrast_order

    rows = []
    for case in read(output / 'protocol.json')['cases']:
        trace = read(SOURCE / 'after/retained' / case / 'frozen_numeric/research_trace.json')
        question = next(q for q in trace['questions'] if q.get('answer', {}).get('frozen_transfer'))
        state = deepcopy(trace['state'])
        state['completed'] = []
        state['response_settings'] = [r for r in state['response_settings']
            if max(r.get('evidence', [0])) < question['opened_before_observation']]
        knowledge = state['frozen_response_knowledge']
        result = reuse_response_knowledge(state, [{'measurements': knowledge['measurements']}],
                                          'shared/consolidated_method.py')
        experiments = result['experiments']
        classes = {}
        for hypothesis in result['hypotheses']:
            column = tuple(e['predictions'][hypothesis['id']] for e in experiments)
            classes.setdefault(column, []).append(hypothesis['id'])
        chosen = result['selection']['selected_experiment']
        archived = trace['comparisons'][0]['selected_experiment']
        key = lambda e: {'number': e['number'], 'prediction': e['prediction']}
        assert chosen == archived, case
        ordered, _ = _contrast_order(experiments)
        rows.append({'case': case, 'raw_decision_reproduced': True,
            'prefix_response_count': len(state['response_settings']),
            'raw_hypothesis_count': len(result['hypotheses']),
            'forecast_class_count': len(classes), 'class_sizes': [len(c) for c in classes.values()],
            'source_counts': result['frozen_transfer'].get('source_counts', {}),
            'is_mixed_archive': knowledge['source_id'] == 'retained_response_archives',
            'raw_selection': key(chosen), 'unique_selection': key(ordered[0]),
            'selection_changed': key(chosen) != key(ordered[0]),
            'experiments': [{**key(e), 'raw_positive_count': sum(e['predictions'].values()),
                'raw_variance': pvariance(e['predictions'].values()),
                'unique_positive_count': sum(e['predictions'][ids[0]] for ids in classes.values()),
                'unique_variance': pvariance(e['predictions'][ids[0]] for ids in classes.values())}
                for e in experiments]})
    assert len(rows) == 26 and sum(r['selection_changed'] for r in rows) == 6
    write(output / 'initial_decision_diagnosis.json', {'cases': rows, 'reproduced': 26,
        'changed': 6, 'mixed_archive_cases': sum(r['is_mixed_archive'] for r in rows),
        'scope': 'Saved paid prefixes only; no new physical observations and no task outcome inference.'})


def prepare(output):
    from scripts.run_response_matched_adoption import prepare_bundle
    from scripts.run_response_composition_transfer import declarations

    old = read(SOURCE / 'protocol.json')
    assert not (output / 'protocol.json').exists(), 'Preserve the prepared protocol'
    # Explicit pre-edit copies avoid a race with production implementation work.
    for module in CHANGED:
        shutil.copyfile(output / (module + '_before.py'), output / 'runtime/before' / (module + '.py'))
    cases = old['cases']
    pilot = [case for case, spec in cases.items() if spec['source_id'] in PILOT_SOURCES]
    write(output / 'protocol.json', {'cases': cases, 'phases': list(PHASES), 'knowledge_arm': 'retained',
        'limits': old['limits'], 'planned_executions': 52, 'pilot_cases': pilot,
        'pilot_executions': 12, 'remaining_executions': 40, 'maximum_workers': 2,
        'source': str(SOURCE), 'runtime_modules': list(MODULES),
        'intervention': 'Use distinct prospective forecast columns in finite response experiment selection; retain every hypothesis and physical prediction. The existing full-family contrast ordering shares the same extracted grouping helper.',
        'control': 'Both phases rerun all 26 retained cases with the same current point-finalization controller, exact numerical archives, declarations, tools and limits. Only the three frozen chooser modules differ.',
        'pilot_rule': 'First run both phases for the six prefix decisions affected by grouping. After inspection run the other twenty cases in both phases; count pilot once.',
        'scope': 'Consumed development conditions; 18 mixed archives and 8 old-only fallbacks. No independent mechanism, LLM or held-out task claim.',
        'cost_scope': 'Report charged research computation, physical actions, confirmations and execution limits. Diagnostic exports are separate.',
        'model_calls': 0})
    refreshed = []
    for source_id in dict.fromkeys(spec['source_id'] for spec in cases.values()):
        old_bundle = read(SOURCE / 'after/retained/sources' / source_id / 'numeric_bundle.json')
        bundle = prepare_bundle(old_bundle)
        method = previous.original.METHOD
        assert declarations(old_bundle['files'][method]) == declarations(bundle['files'][method])
        assert {k: v for k, v in old_bundle.items() if k != 'files'} == {k: v for k, v in bundle.items() if k != 'files'}
        assert {k: v for k, v in old_bundle['files'].items() if k != method} == {k: v for k, v in bundle['files'].items() if k != method}
        for phase in PHASES:
            write(output / phase / 'sources' / source_id / 'numeric_bundle.json', bundle)
        refreshed.append({'source': source_id, 'common_controller_refreshed': True,
                          'knowledge_declarations_methods_preserved': True})
    for phase in PHASES:
        write(output / phase / 'protocol.json', {'cases': cases, 'limits': old['limits'],
            'arms': ['frozen_numeric'], 'planned_executions': 26})
    write(output / 'bundle_refresh.json', refreshed)
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    diagnosis(output)
    summarize(output)
    print(json.dumps({'status': 'prepared', 'planned': 52, 'pilot': 12}), flush=True)


def freeze_after(output):
    directory = output / 'runtime/after'
    directory.mkdir(exist_ok=False)
    changed = []
    for module in MODULES:
        target = directory / (module + '.py')
        shutil.copyfile(ROOT / 'src/hegel_machine/cognition' / target.name, target)
        if target.read_text() != (output / 'runtime/before' / target.name).read_text():
            changed.append(module)
    assert set(changed) == CHANGED, changed
    write(output / 'runtime_difference.json', {'changed_modules': changed,
        'unchanged_modules': sorted(set(MODULES) - set(changed))})
    print(json.dumps({'status': 'after_runtime_frozen', 'changed_modules': changed}), flush=True)


def location(output, phase, case):
    return output / phase / case / 'frozen_numeric'


def summarize(output):
    protocol = read(output / 'protocol.json')
    rows, pairs = [], []
    for case, config in protocol['cases'].items():
        found = {}
        for phase in PHASES:
            path = location(output, phase, case) / 'brief.json'
            if path.exists():
                found[phase] = {**read(path), 'phase': phase}
                rows.append(found[phase])
        if len(found) != 2:
            continue
        a, b = found['before'], found['after']
        old_actions, new_actions = [read(location(output, phase, case) / 'action_trace.json') for phase in PHASES]
        pairs.append({'case': case, 'condition': config['condition'],
            'success_before': a['success'], 'success_after': b['success'],
            'confirmations_before': a['requested_point_confirmed'], 'confirmations_after': b['requested_point_confirmed'],
            'actions_difference': b['costs']['environment_steps'] - a['costs']['environment_steps'],
            'compute_difference': b['costs']['research_compute_units'] - a['costs']['research_compute_units'],
            'full_actions_preserved': old_actions == new_actions,
            'first_different_action': next((i for i, (x, y) in enumerate(zip(old_actions, new_actions))
                if x['action'] != y['action']), None),
            'status_before': a['status'], 'status_after': b['status']})
    summary = {'status': 'finished' if len(rows) == 52 else 'running' if rows else 'prepared',
        'completed': len(rows), 'planned': 52,
        'conditions': {condition: {phase: previous.aggregate([r for r in rows
            if r['condition'] == condition and r['phase'] == phase]) for phase in PHASES}
            for condition in previous.CONDITIONS}, 'paired': pairs}
    write(output / 'brief.json', rows)
    write(output / 'summary.json', summary)
    return summary


def run(output, pilot):
    protocol = read(output / 'protocol.json')
    # Avoid mixing pilot and remaining observations from different implementations.
    for module in MODULES:
        assert (output / 'runtime/after' / (module + '.py')).read_text() == (ROOT / 'src/hegel_machine/cognition' / (module + '.py')).read_text(), module
    assert (output / 'run_response_numeric_transfer.py').read_text() == (ROOT / 'scripts/run_response_numeric_transfer.py').read_text()
    cases = [case for case in protocol['cases'] if (case in protocol['pilot_cases']) == pilot]
    if not pilot:
        assert all((location(output, phase, case) / 'brief.json').exists()
                   for case in protocol['pilot_cases'] for phase in PHASES), 'Complete pilot first'
    jobs = [(phase, case) for case in cases for phase in PHASES]
    assert not any(location(output, phase, case).exists() for phase, case in jobs), 'Preserve started runs'
    def launch(phase, case):
        log = output / 'logs' / phase / (case + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as stream:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--output', str(output),
                '--case', case, '--phase', phase], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        return result.returncode, str(log)
    errors = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(launch, phase, case): (phase, case) for phase, case in jobs}
        for future in as_completed(pending):
            phase, case = pending[future]
            code, log = future.result()
            if code:
                errors.append({'phase': phase, 'case': case, 'returncode': code, 'log': log})
            write(output / ('pilot_errors.json' if pilot else 'batch_errors.json'), errors)
            summary = summarize(output)
            print(json.dumps({'phase': phase, 'case': case, 'returncode': code, 'completed': summary['completed']}), flush=True)
    if errors:
        raise SystemExit('Recorded execution failures')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--freeze-after', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--case')
    parser.add_argument('--phase', choices=PHASES)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.case and args.phase:
        load_runtime(output / 'runtime' / args.phase)
        previous.original.run_case(output / args.phase, args.case, 'frozen_numeric')
    elif args.prepare:
        prepare(output)
    elif args.freeze_after:
        freeze_after(output)
    elif args.pilot or args.run:
        run(output, args.pilot)
    elif args.summarize:
        print(json.dumps(summarize(output), indent=2))
    else:
        parser.error('Choose prepare, freeze-after, pilot, run or summarize')
