#!/usr/bin/env python3
"""Matched frozen-member target consequences on the existing 26 conditions."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]
from scripts import run_response_trajectory_choice as previous

SOURCE = previous.OUTPUT
OUTPUT = ROOT / 'artifacts/response_target_consequences_20261011'
PHASES = ('before', 'after')
MODULES = tuple(part for part in previous.MODULES if part != 'world_response_trajectory')
CHANGED = {'world_response_consequences', 'world_response_transfer', 'world_reference_response'}
EXPORTERS = ('run_response_matched_adoption', 'run_response_numeric_transfer')
read, write, aggregate = previous.read, previous.write, previous.aggregate


def location(output, phase, case):
    return output / phase / case / 'frozen_numeric'


def prepare(output):
    """Freeze both arms once, preserving existing before copies and all inputs."""
    assert not (output / 'protocol.json').exists(), 'Preserve prepared inputs'
    old = read(SOURCE / 'protocol.json')
    cases = deepcopy(old['cases'])
    # The preceding baseline reproduces the forecast-class result, but its
    # actual exporters differ from the restored scripts. Use a new paired
    # baseline rather than claiming exact archived execution equivalence.
    same_modules = []
    for part in MODULES:
        before = output / 'runtime/before' / (part + '.py')
        assert before.read_text() == (SOURCE / 'runtime/before' / before.name).read_text(), part
        same_modules.append(part)
    changed = []
    after = output / 'runtime/after'
    after.mkdir(parents=True, exist_ok=False)
    for part in MODULES:
        target = after / (part + '.py')
        shutil.copyfile(ROOT / 'src/hegel_machine/cognition' / target.name, target)
        if target.read_text() != (output / 'runtime/before' / target.name).read_text():
            changed.append(part)
    assert set(changed) == CHANGED, changed
    for phase in PHASES:
        execution = output / 'execution' / phase
        execution.mkdir(parents=True, exist_ok=False)
        for part in EXPORTERS:
            source = output / (part + '_before.py') if phase == 'before' else ROOT / 'scripts' / (part + '.py')
            shutil.copyfile(source, execution / (part + '.py'))
    matched = 'run_response_matched_adoption.py'
    assert (output / 'execution/before' / matched).read_text() == (output / 'execution/after' / matched).read_text()
    inventory = {}
    for source_id in dict.fromkeys(case['source_id'] for case in cases.values()):
        source = SOURCE / 'before/sources' / source_id / 'numeric_bundle.json'
        bundle = read(source)
        archive = json.loads(bundle['files'][previous.numeric.KNOWLEDGE])
        inventory[source_id] = {'source': str(source), 'models': len(archive['models']),
            'stratum': 'mixed' if archive['source_id'] == 'retained_response_archives' else 'empty_revision_fallback'}
        for phase in PHASES:
            target = output / phase / 'sources' / source_id / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    strata = {case: inventory[spec['source_id']]['stratum'] for case, spec in cases.items()}
    assert len(cases) == 26 and sum(s == 'mixed' for s in strata.values()) == 18
    write(output / 'protocol.json', {'cases': cases, 'limits': old['limits'], 'phases': list(PHASES),
        'planned_executions': 52, 'maximum_workers': 2, 'model_calls': 0, 'case_strata': strata,
        'source': str(SOURCE), 'runtime_modules': list(MODULES),
        'intervention': 'Only frozen response reuse compares attainable common-positive target trials, then two-outcome projected target signatures excluding the tested point; guaranteed reduction precedes possible reduction, with incumbent ties retained.',
        'control': 'Run both frozen runtimes on identical unchanged source bundles and matched-adoption exporter. Numeric exporter only adds member_consequences to saved diagnostics.',
        'baseline_choice': 'Use 52 new matched executions: archived forecast-class results have an identical numerical exporter but no matched-adoption snapshot; the intervening trajectory run added export diagnostics and prepare helpers. No noncontemporaneous result is counted as this control.',
        'cohort_rule': 'Fixed 26 conditions in both arms, no pilot tuning, replacements, or retries. Preserve every scientific and execution failure.',
        'scope': 'Consumed development cohort: 18 mixed archives and 8 empty-revision fallbacks. Finite numerical members only; common positive is a proposed real trial, not a confirmed outcome or family theorem.',
        'cost_scope': 'All actual physical actions, charged research computation including target comparison, requested-point confirmation, execution limits; diagnostic exports separate. Host elapsed time is not a speedup claim.'})
    for phase in PHASES:
        write(output / phase / 'protocol.json', {'cases': cases, 'limits': old['limits'],
            'arms': ['frozen_numeric'], 'planned_executions': 26})
    write(output / 'source_inventory.json', inventory)
    write(output / 'runtime_difference.json', {'changed_modules': changed,
        'unchanged_modules': sorted(set(MODULES) - set(changed)), 'baseline_runtime_matches_previous': same_modules,
        'matched_exporter_identical': True})
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    summarize(output)
    print(json.dumps({'status': 'prepared', 'planned': 52, 'source_bundles': len(inventory)}), flush=True)


def case_diagnostics(output, phase, case):
    """Read the completed research trace once and retain a compact summary."""
    directory = location(output, phase, case)
    row = read(directory / 'brief.json')
    trace = read(directory / 'research_trace.json')
    selections = []
    for comparison in trace['comparisons']:
        details = comparison.get('member_consequences')
        if details is None:
            continue
        experiment = comparison.get('selected_experiment') or {}
        points = [p for p in row['point_confirmations']
            if p['question_id'] == comparison['question_id'] + '_physical']
        selections.append({'question_id': comparison['question_id'], 'rule': details['rule'],
            'number': experiment.get('number'), 'setting': experiment.get('prediction'),
            'changed': details['selected_experiment_id'] != details['incumbent_experiment_id'],
            'compute_cost': details['compute_cost'],
            'physically_confirmed': any(p['physically_confirmed'] for p in points),
            'physical_verdict_recorded': bool(points)})
    diagnostic = {'case': case, 'phase': phase, 'selections': selections,
        'selection_count': len(selections), 'changed_selections': sum(s['changed'] for s in selections),
        'rules': dict(Counter(s['rule'] for s in selections)),
        'common_positive_trials': sum(s['rule'] == 'common_positive' for s in selections),
        'common_positive_physically_confirmed': sum(s['rule'] == 'common_positive' and s['physically_confirmed'] for s in selections),
        'comparison_compute': sum(s['compute_cost'] for s in selections)}
    write(directory / 'diagnostic_brief.json', diagnostic)
    return diagnostic


def pair_diagnostics(output, case, config):
    directories = {phase: location(output, phase, case) for phase in PHASES}
    rows = {phase: read(path / 'brief.json') for phase, path in directories.items()}
    a, b = rows['before'], rows['after']
    old_actions, new_actions = [read(directories[phase] / 'action_trace.json') for phase in PHASES]
    result = {'case': case, 'condition': config['condition'],
        'success_before': a['success'], 'success_after': b['success'],
        'confirmations_before': a['requested_point_confirmed'], 'confirmations_after': b['requested_point_confirmed'],
        'actions_difference': b['costs']['environment_steps'] - a['costs']['environment_steps'],
        'compute_difference': b['costs']['research_compute_units'] - a['costs']['research_compute_units'],
        'full_actions_preserved': old_actions == new_actions,
        'first_different_action': next((i for i, (x, y) in enumerate(zip(old_actions, new_actions)) if x['action'] != y['action']), None),
        'status_before': a['status'], 'status_after': b['status']}
    write(output / 'paired' / (case + '.json'), result)
    return result


def summarize(output):
    protocol = read(output / 'protocol.json')
    rows, pairs, diagnostics = [], [], []
    for case, config in protocol['cases'].items():
        found = []
        for phase in PHASES:
            directory = location(output, phase, case)
            if not (directory / 'brief.json').exists():
                continue
            row = {**read(directory / 'brief.json'), 'phase': phase, 'stratum': protocol['case_strata'][case]}
            rows.append(row); found.append(phase)
            diagnostic = directory / 'diagnostic_brief.json'
            if diagnostic.exists():
                diagnostics.append(read(diagnostic))
        if len(found) == 2:
            saved = output / 'paired' / (case + '.json')
            pairs.append(read(saved) if saved.exists() else pair_diagnostics(output, case, config))
    finished = len(rows) == protocol['planned_executions']
    errors = read(output / 'batch_errors.json') if (output / 'batch_errors.json').exists() else []
    status = 'execution_failures' if errors else 'finished' if finished else 'running' if rows else 'prepared'
    summary = {'status': status, 'completed': len(rows), 'planned': protocol['planned_executions'],
        'paired_completed': len(pairs),
        'overall': {phase: aggregate([r for r in rows if r['phase'] == phase]) for phase in PHASES},
        'conditions': {condition: {phase: aggregate([r for r in rows
            if r['condition'] == condition and r['phase'] == phase]) for phase in PHASES}
            for condition in ('adapted_law', 'original_law_return')},
        'strata': {stratum: {phase: aggregate([r for r in rows
            if r['stratum'] == stratum and r['phase'] == phase]) for phase in PHASES}
            for stratum in ('mixed', 'empty_revision_fallback')}, 'paired': pairs,
        'member_consequences': {phase: {
            'diagnosed_cases': sum(d['phase'] == phase for d in diagnostics),
            **{key: sum(d[key] for d in diagnostics if d['phase'] == phase) for key in
                ('selection_count', 'changed_selections', 'common_positive_trials', 'common_positive_physically_confirmed', 'comparison_compute')},
            'rules': dict(sum((Counter(d['rules']) for d in diagnostics if d['phase'] == phase), Counter()))}
            for phase in PHASES}, 'execution_errors': errors}
    write(output / 'brief.json', rows)
    write(output / 'diagnostic_brief.json', diagnostics)
    write(output / 'summary.json', summary)
    return summary


def run(output):
    protocol = read(output / 'protocol.json')
    jobs = [(phase, case) for case in protocol['cases'] for phase in PHASES]
    assert not any(location(output, phase, case).exists() for phase, case in jobs), 'Preserve started runs'
    errors = []
    def launch(phase, case):
        log = output / 'logs' / phase / (case + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as stream:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                '--output', str(output), '--case', case, '--phase', phase], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT)
        return result.returncode, str(log)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(launch, phase, case): (phase, case) for phase, case in jobs}
        for future in as_completed(pending):
            phase, case = pending[future]
            code, log = future.result()
            if code:
                errors.append({'phase': phase, 'case': case, 'returncode': code, 'log': log})
            write(output / 'batch_errors.json', errors)
            summary = summarize(output)
            print(json.dumps({'phase': phase, 'case': case, 'returncode': code,
                'completed': summary['completed'], 'paired_completed': summary['paired_completed']}), flush=True)
    if errors:
        raise SystemExit('Recorded execution failures')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    for option in ('prepare', 'run', 'summarize'):
        parser.add_argument('--' + option, action='store_true')
    parser.add_argument('--case')
    parser.add_argument('--phase', choices=PHASES)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.case and args.phase:
        previous.load_runtime(output / 'runtime' / args.phase)
        previous.load_execution(output / 'execution' / args.phase).run_case(output / args.phase, args.case, 'frozen_numeric')
        case_diagnostics(output, args.phase, args.case)
    elif args.prepare:
        prepare(output)
    elif args.run:
        run(output)
    elif args.summarize:
        print(json.dumps(summarize(output), indent=2))
    else:
        parser.error('Choose prepare, run, summarize, or case with phase')
