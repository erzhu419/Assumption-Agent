#!/usr/bin/env python3
"""Matched endpoint versus paid-menu trajectory selection on the same 26 cases."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]
from scripts import run_response_forecast_classes as previous

SOURCE = previous.OUTPUT
OUTPUT = ROOT / 'artifacts/response_trajectory_choice_20261011'
PHASES = previous.PHASES
MODULES = ('world_experiment_choice', 'world_reference_recalibration', 'world_response_trajectory',
    'world_response_consequences', 'world_response_transfer', 'world_response_hypotheses',
    'world_reference_hypothesis', 'world_reference_expansion', 'world_reference_response')
CHANGED = {'world_experiment_choice', 'world_reference_recalibration', 'world_response_trajectory',
           'world_response_hypotheses', 'world_reference_response'}
PILOT = ('cubic_shifted_2__gain_higher__adapted_law',
         'cubic_shifted_2__gain_higher__original_law_return',
         'cubic_original_3__gain_lower__adapted_law',
         'cubic_original_4__gain_higher__adapted_law')
read, write = previous.read, previous.write
aggregate = previous.previous.aggregate
numeric = previous.previous.original


def load_runtime(directory):
    # Each execution starts a fresh process before the Agent/consumer imports.
    # Bind dependencies first; notably trajectory captures _tuning_choice.
    import hegel_machine.cognition as package
    for part in MODULES:
        name = 'hegel_machine.cognition.' + part
        sys.modules.pop(name, None)
        if hasattr(package, part):
            delattr(package, part)
    for part in MODULES:
        path = directory / (part + '.py')
        if not path.exists():  # The endpoint runtime predates this adapter.
            continue
        name = 'hegel_machine.cognition.' + part
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        setattr(package, part, module)
        spec.loader.exec_module(module)


def prepare(output):
    from scripts.run_response_matched_adoption import prepare_bundle
    from scripts.run_response_composition_transfer import declarations

    assert not (output / 'protocol.json').exists(), 'Preserve prepared inputs'
    old = read(SOURCE / 'protocol.json')
    cases = deepcopy(old['cases'])
    inventory = {}
    for source_id in dict.fromkeys(case['source_id'] for case in cases.values()):
        old_bundle = read(SOURCE / 'after/sources' / source_id / 'numeric_bundle.json')
        bundle = prepare_bundle(old_bundle)
        method = numeric.METHOD
        assert declarations(old_bundle['files'][method]) == declarations(bundle['files'][method])
        assert {k: v for k, v in old_bundle.items() if k != 'files'} == {k: v for k, v in bundle.items() if k != 'files'}
        assert {k: v for k, v in old_bundle['files'].items() if k != method} == {k: v for k, v in bundle['files'].items() if k != method}
        archive = json.loads(bundle['files'][numeric.KNOWLEDGE])
        inventory[source_id] = {'models': len(archive['models']),
            'stratum': 'mixed' if archive['source_id'] == 'retained_response_archives' else 'empty_revision_fallback'}
        for phase in PHASES:
            write(output / phase / 'sources' / source_id / 'numeric_bundle.json', bundle)
    for part in MODULES:
        if part == 'world_response_trajectory':
            continue
        archived = output / (part + '_before.py')
        source = archived if archived.exists() else SOURCE / 'runtime/after' / (part + '.py')
        target = output / 'runtime/before' / (part + '.py')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    strata = {name: inventory[case['source_id']]['stratum'] for name, case in cases.items()}
    assert len(cases) == 26 and sum(s == 'mixed' for s in strata.values()) == 18
    write(output / 'protocol.json', {'cases': cases, 'limits': old['limits'], 'phases': list(PHASES),
        'planned_executions': 52, 'pilot_cases': list(PILOT), 'pilot_executions': 8,
        'remaining_executions': 44, 'maximum_workers': 2, 'model_calls': 0,
        'case_strata': strata, 'source': str(SOURCE), 'runtime_modules': list(MODULES),
        'intervention': 'Finite response selection compares predicted tuning readouts and length under each object own paid menu; retain endpoint forecasts and full candidate/source records.',
        'control': 'Both phases use identical source archives, declarations and equivalent current exported tuning helpers. The before host uses the frozen endpoint chooser; the after host binds the trajectory adapter.',
        'scope': 'Same consumed 26 development conditions: 18 mixed archives and 8 empty-revision fallbacks. No independent mechanism or model-adoption claim.',
        'cost_scope': 'Charged research computation, all actual physical actions, point confirmations and limits. Predicted tuning cost excludes navigation/installation; diagnostic exports remain separate.',
        'pilot_rule': 'Inspect all eight fixed pilot executions before running the remaining forty-four. Preserve negative results; no case replacement or tuning within the cohort.'})
    for phase in PHASES:
        write(output / phase / 'protocol.json', {'cases': cases, 'limits': old['limits'],
            'arms': ['frozen_numeric'], 'planned_executions': 26})
    write(output / 'source_inventory.json', inventory)
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    shutil.copyfile(ROOT / 'scripts/run_response_numeric_transfer.py', output / 'run_response_numeric_transfer.py')
    shutil.copyfile(ROOT / 'scripts/run_response_matched_adoption.py', output / 'run_response_matched_adoption.py')
    summarize(output)
    print(json.dumps({'status': 'prepared', 'planned': 52, 'pilot': 8}), flush=True)


def freeze_after(output):
    directory = output / 'runtime/after'
    directory.mkdir(parents=True, exist_ok=False)
    changed = []
    for module in MODULES:
        target = directory / (module + '.py')
        shutil.copyfile(ROOT / 'src/hegel_machine/cognition' / target.name, target)
        before = output / 'runtime/before' / target.name
        if not before.exists() or before.read_text() != target.read_text():
            changed.append(module)
    assert set(changed) == CHANGED, changed
    write(output / 'runtime_difference.json', {'changed_modules': changed,
        'unchanged_modules': sorted(set(MODULES) - set(changed))})
    print(json.dumps({'status': 'after_runtime_frozen', 'changed_modules': changed}), flush=True)


def path_matches(directory):
    trace, records = read(directory / 'research_trace.json'), read(directory / 'control_observations.json')
    questions = {q['id']: q for q in trace['questions']}
    comparisons = trace['comparisons']
    rows = []
    for position, comparison in enumerate(comparisons):
        experiment = comparison.get('selected_experiment') or {}
        if not experiment.get('tuning_path'):
            continue
        qid = comparison['question_id']
        start = questions[qid]['opened_before_observation']
        later = [questions[c['question_id']]['opened_before_observation'] for c in comparisons[position + 1:]
                 if c['question_id'] != qid and questions[c['question_id']]['opened_before_observation'] > start]
        stop = min(later, default=float('inf'))
        menu = comparison['trajectory_inputs'][str(experiment['number'])]
        options = set(map(int, menu['steps']))
        actual = [r for r in records if start <= r['index'] < stop and r['number'] == experiment['number']
                  and r['action'].get('chosen_dialog_option_int') in options]
        planned = experiment['tuning_path']
        matches = [a['action']['chosen_dialog_option_int'] == p['chosen_dialog_option_int']
                   and abs(a['frequency'] - p['frequency']) <= 1e-7 for a, p in zip(actual, planned)]
        rows.append({'question_id': qid, 'number': experiment['number'], 'setting': experiment['prediction'],
            'control_evidence': experiment['control_evidence'], 'selection_before_observation': start,
            'all_input_evidence_precedes_choice': all(m['evidence'] < start for m in comparison['trajectory_inputs'].values()),
            'predicted_adjustments': len(planned), 'actual_adjustments_before_next_choice': len(actual),
            'compared_steps': len(matches), 'matched_steps': sum(matches),
            'full_path_matched': len(actual) == len(planned) and all(matches),
            'predicted_segment_matched': len(actual) >= len(planned) and all(matches),
            'additional_actual_adjustments': max(0, len(actual) - len(planned)),
            'matching_prefix_only': len(actual) < len(planned) and all(matches),
            'actual_indices': [r['index'] for r in actual],
            'mismatch_positions': [i for i, same in enumerate(matches) if not same]})
    return rows


def summarize(output):
    summary = previous.summarize(output)
    protocol = read(output / 'protocol.json')
    rows = read(output / 'brief.json')
    for row in rows:
        row['stratum'] = protocol['case_strata'][row['case']]
    summary['strata'] = {stratum: {phase: aggregate([r for r in rows
        if r['stratum'] == stratum and r['phase'] == phase]) for phase in PHASES}
        for stratum in ('mixed', 'empty_revision_fallback')}
    diagnostics = []
    preservation = []
    for row in rows:
        directory = previous.location(output, row['phase'], row['case'])
        details = path_matches(directory)
        diagnostics.append({'case': row['case'], 'phase': row['phase'], 'paths': details})
        if row['phase'] == 'before':
            old_directory = previous.location(SOURCE, 'after', row['case'])
            old = read(old_directory / 'brief.json')
            preservation.append({'case': row['case'],
                'actions_preserved': read(directory / 'action_trace.json') == read(old_directory / 'action_trace.json'),
                'endpoints_preserved': all(row[k] == old[k] for k in ('success', 'status',
                    'point_confirmations', 'requested_point_confirmed', 'unsupported_positive_verdicts')),
                'compute_preserved': row['costs']['research_compute_units'] == old['costs']['research_compute_units']})
    summary['trajectory_validation'] = {'selected_paths': sum(len(d['paths']) for d in diagnostics),
        'compared_steps': sum(p['compared_steps'] for d in diagnostics for p in d['paths']),
        'matched_steps': sum(p['matched_steps'] for d in diagnostics for p in d['paths']),
        'full_paths_matched': sum(p['full_path_matched'] for d in diagnostics for p in d['paths']),
        'predicted_segments_matched': sum(p['predicted_segment_matched'] for d in diagnostics for p in d['paths']),
        'additional_actual_adjustments': sum(p['additional_actual_adjustments'] for d in diagnostics for p in d['paths']),
        'matching_prefix_only': sum(p['matching_prefix_only'] for d in diagnostics for p in d['paths']),
        'input_evidence_failures': sum(not p['all_input_evidence_precedes_choice'] for d in diagnostics for p in d['paths'])}
    write(output / 'brief.json', rows)
    write(output / 'trajectory_matches.json', diagnostics)
    write(output / 'baseline_preservation.json', preservation)
    write(output / 'summary.json', summary)
    return summary


def run(output, pilot):
    protocol = read(output / 'protocol.json')
    for module in MODULES:
        assert (output / 'runtime/after' / (module + '.py')).read_text() == (ROOT / 'src/hegel_machine/cognition' / (module + '.py')).read_text(), module
    for filename in ('run_response_numeric_transfer.py', 'run_response_matched_adoption.py'):
        assert (output / filename).read_text() == (ROOT / 'scripts' / filename).read_text(), filename
    cases = [case for case in protocol['cases'] if (case in protocol['pilot_cases']) == pilot]
    if not pilot:
        assert all((previous.location(output, phase, case) / 'brief.json').exists()
            for phase in PHASES for case in protocol['pilot_cases']), 'Complete pilot first'
    jobs = [(phase, case) for case in cases for phase in PHASES]
    assert not any(previous.location(output, phase, case).exists() for phase, case in jobs), 'Preserve started runs'
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
    for option in ('prepare', 'freeze-after', 'pilot', 'run', 'summarize'):
        parser.add_argument('--' + option, action='store_true')
    parser.add_argument('--case')
    parser.add_argument('--phase', choices=PHASES)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.case and args.phase:
        load_runtime(output / 'runtime' / args.phase)
        numeric.run_case(output / args.phase, args.case, 'frozen_numeric')
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
