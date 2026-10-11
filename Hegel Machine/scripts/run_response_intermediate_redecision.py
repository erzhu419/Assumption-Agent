#!/usr/bin/env python3
"""Matched return to parent research when an unfinished trial loses support."""
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
from scripts import run_response_target_consequences as previous

SOURCE = previous.OUTPUT
OUTPUT = ROOT / 'artifacts/response_intermediate_redecision_20261011'
PHASES, MODULES, EXPORTERS = previous.PHASES, previous.MODULES, previous.EXPORTERS
read, write, aggregate = previous.read, previous.write, previous.aggregate
location, pair_diagnostics = previous.location, previous.pair_diagnostics
load_runtime, load_execution = previous.previous.load_runtime, previous.previous.load_execution


def prepare(output):
    """Bind each arm's controller exporter to that arm's frozen runtime."""
    from scripts.run_response_composition_transfer import declarations

    assert not (output / 'protocol.json').exists(), 'Preserve prepared inputs'
    old = read(SOURCE / 'protocol.json')
    cases, inventory, changed = deepcopy(old['cases']), {}, []
    after = output / 'runtime/after'
    after.mkdir(parents=True, exist_ok=False)
    for part in MODULES:
        before = output / 'runtime/before' / (part + '.py')
        assert before.read_text() == (SOURCE / 'runtime/before' / before.name).read_text(), part
        target = after / before.name
        shutil.copyfile(ROOT / 'src/hegel_machine/cognition' / target.name, target)
        if before.read_text() != target.read_text():
            changed.append(part)
    assert changed == ['world_reference_recalibration'], changed
    for phase in PHASES:
        execution = output / 'execution' / phase
        execution.mkdir(parents=True, exist_ok=False)
        for part in EXPORTERS:
            source = output / (part + '_before.py') if phase == 'before' else ROOT / 'scripts' / (part + '.py')
            shutil.copyfile(source, execution / (part + '.py'))
        load_runtime(output / 'runtime' / phase)
        load_execution(execution)
        from scripts.run_response_matched_adoption import prepare_bundle

        for source_id in dict.fromkeys(case['source_id'] for case in cases.values()):
            source = SOURCE / 'before/sources' / source_id / 'numeric_bundle.json'
            original = read(source)
            bundle = prepare_bundle(original)
            method = previous.previous.numeric.METHOD
            assert declarations(original['files'][method]) == declarations(bundle['files'][method])
            assert {k: v for k, v in original.items() if k != 'files'} == {k: v for k, v in bundle.items() if k != 'files'}
            assert {k: v for k, v in original['files'].items() if k != method} == {k: v for k, v in bundle['files'].items() if k != method}
            archive = json.loads(bundle['files'][previous.previous.numeric.KNOWLEDGE])
            item = inventory.setdefault(source_id, {'source': str(source), 'models': len(archive['models']),
                'stratum': 'mixed' if archive['source_id'] == 'retained_response_archives' else 'empty_revision_fallback'})
            item[phase + '_method_matches_source'] = bundle['files'][method] == original['files'][method]
            write(output / phase / 'sources' / source_id / source.name, bundle)
        write(output / phase / 'protocol.json', {'cases': cases, 'limits': old['limits'],
            'arms': ['frozen_numeric'], 'planned_executions': 26})
    write(output / 'protocol.json', {**{key: old[key] for key in ('cases', 'limits', 'case_strata')},
        'phases': list(PHASES), 'planned_executions': 52, 'maximum_workers': 2, 'model_calls': 0,
        'source': str(SOURCE), 'runtime_modules': list(MODULES),
        'intervention': 'Return an unfinished trial to parent research after paid intermediate evidence eliminates every frozen member that supported its requested setting. Existing parent selection, transfer and recovery continue.',
        'control': 'Each phase freezes its own runtime and controller exporter. Both prepare the same archived source bundle; scientific declarations and all other knowledge files remain identical.',
        'baseline_choice': '52 new paired executions because the exported physical controller changes. Compare the completed before arm once with the preceding baseline, without rerunning that archived cohort.',
        'cohort_rule': 'Fixed 26 development conditions, no pilot tuning, case replacements or retries. Retain every scientific and execution failure.',
        'scope': '18 mixed archives and 8 empty-revision fallbacks; zero model/API calls. Withdrawing an unfinished trial does not refute its unobserved endpoint.',
        'cost_scope': 'Actual physical actions, charged research computation, requested-point confirmation and execution limits. Elapsed host time is not a speedup claim.'})
    write(output / 'source_inventory.json', inventory)
    write(output / 'runtime_difference.json', {'changed_modules': changed,
        'unchanged_modules': sorted(set(MODULES) - set(changed)),
        'execution_exporters': list(EXPORTERS), 'controller_prepared_separately_for_each_phase': True})
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    summarize(output)
    print(json.dumps({'status': 'prepared', 'planned': 52, 'source_bundles': len(inventory)}), flush=True)


def case_diagnostics(output, phase, case):
    """Read the complete trace once, then aggregate only compact diagnostics."""
    directory = location(output, phase, case)
    trace = read(directory / 'research_trace.json')
    questions = {q['id']: q for q in trace['questions']}
    state, retirements = trace['state'], []
    for trial in state.get('tests', []):
        if trial.get('retirement', {}).get('reason') != 'intermediate_responses_refuted_all_requested_members':
            continue
        question = questions.get(trial['question'], {})
        comparison_id = trial.get('response_question')
        comparisons = trace['comparisons']
        position = next((i for i, item in enumerate(comparisons) if item['question_id'] == comparison_id), -1)
        followup = comparisons[position + 1] if position >= 0 and position + 1 < len(comparisons) else {}
        retirements.append({'question_id': trial['question'], 'number': trial['number'],
            'requested_setting': trial['prediction'], 'retirement': trial['retirement'],
            'trial_status': trial.get('status'), 'question_status': question.get('status'),
            'evidence': question.get('evidence', []), 'next_comparison_question': followup.get('question_id'),
            'next_status': followup.get('status'), 'next_selected_experiment': followup.get('selected_experiment'),
            'next_frozen_transfer': followup.get('frozen_transfer')})
    knowledge = read(directory / 'learned_response_knowledge.json')
    diagnostic = {'case': case, 'phase': phase, 'retirements': retirements,
        'retirement_count': len(retirements),
        'inconclusive_retirements': sum(item['question_status'] == 'inconclusive' for item in retirements),
        'parent_reselections': sum(bool(item['next_selected_experiment']) for item in retirements),
        'next_research_stages': dict(Counter((item['next_frozen_transfer'] or {}).get('research_stage', 'not_frozen') for item in retirements)),
        'completed_targets': state.get('completed', []),
        'retained_member_count': len(knowledge.get('models', [])), 'pending_trial': state.get('test')}
    write(directory / 'diagnostic_brief.json', diagnostic)
    return diagnostic


def baseline_preservation(output, cases):
    path = output / 'baseline_preservation.json'
    if path.exists():
        return read(path)
    rows = []
    for case in cases:
        current, archived = location(output, 'before', case), location(SOURCE, 'before', case)
        a, b = read(current / 'brief.json'), read(archived / 'brief.json')
        rows.append({'case': case,
            'full_action_trace_preserved': read(current / 'action_trace.json') == read(archived / 'action_trace.json'),
            'endpoints_preserved': all(a[key] == b[key] for key in ('success', 'status',
                'point_confirmations', 'requested_point_confirmed', 'unsupported_positive_verdicts')),
            'research_compute_preserved': a['costs']['research_compute_units'] == b['costs']['research_compute_units']})
    write(path, rows)
    return rows


def summarize(output):
    protocol, rows, pairs, diagnostics = read(output / 'protocol.json'), [], [], []
    for case, config in protocol['cases'].items():
        found = []
        for phase in PHASES:
            directory = location(output, phase, case)
            if not (directory / 'brief.json').exists():
                continue
            rows.append({**read(directory / 'brief.json'), 'phase': phase, 'stratum': protocol['case_strata'][case]})
            found.append(phase)
            if (directory / 'diagnostic_brief.json').exists():
                diagnostics.append(read(directory / 'diagnostic_brief.json'))
        if len(found) == 2:
            saved = output / 'paired' / (case + '.json')
            pairs.append(read(saved) if saved.exists() else pair_diagnostics(output, case, config))
    finished = len(rows) == protocol['planned_executions']
    errors = read(output / 'batch_errors.json') if (output / 'batch_errors.json').exists() else []
    summary = {'status': 'execution_failures' if errors else 'finished' if finished else 'running' if rows else 'prepared',
        'completed': len(rows), 'planned': protocol['planned_executions'], 'paired_completed': len(pairs),
        'overall': {phase: aggregate([r for r in rows if r['phase'] == phase]) for phase in PHASES},
        'strata': {stratum: {phase: aggregate([r for r in rows if r['phase'] == phase and r['stratum'] == stratum])
            for phase in PHASES} for stratum in ('mixed', 'empty_revision_fallback')},
        'conditions': {condition: {phase: aggregate([r for r in rows if r['phase'] == phase and r['condition'] == condition])
            for phase in PHASES} for condition in ('adapted_law', 'original_law_return')},
        'paired': pairs, 'execution_errors': errors,
        'intermediate_redecision': {phase: {
            **{key: sum(d[key] for d in diagnostics if d['phase'] == phase) for key in
               ('retirement_count', 'inconclusive_retirements', 'parent_reselections')},
            'next_research_stages': dict(sum((Counter(d['next_research_stages']) for d in diagnostics if d['phase'] == phase), Counter()))}
            for phase in PHASES}}
    if finished:
        summary['baseline_preservation'] = baseline_preservation(output, protocol['cases'])
    write(output / 'brief.json', rows)
    write(output / 'diagnostic_brief.json', diagnostics)
    write(output / 'summary.json', summary)
    return summary


def run(output):
    jobs = [(phase, case) for case in read(output / 'protocol.json')['cases'] for phase in PHASES]
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
            write(output / 'batch_errors.json', errors)
            summary = summarize(output)
            print(json.dumps({'phase': phase, 'case': case, 'returncode': code, 'completed': summary['completed']}), flush=True)
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
        load_runtime(output / 'runtime' / args.phase)
        load_execution(output / 'execution' / args.phase).run_case(output / args.phase, args.case, 'frozen_numeric')
        case_diagnostics(output, args.phase, args.case)
    elif args.prepare:
        prepare(output)
    elif args.run:
        run(output)
    elif args.summarize:
        print(json.dumps(summarize(output), indent=2))
    else:
        parser.error('Choose prepare, run, summarize, or case with phase')
