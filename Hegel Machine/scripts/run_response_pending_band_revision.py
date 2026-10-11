#!/usr/bin/env python3
"""Fixed-program band revision: 26 preserved cases and 26 fresh object pairs."""
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
from scripts import run_response_intermediate_redecision as previous

SOURCE = previous.OUTPUT
OUTPUT = ROOT / 'artifacts/response_pending_band_revision_20261011'
COHORTS, PHASES = ('preservation', 'new_objects'), previous.PHASES
MODULES, EXPORTERS = previous.MODULES, previous.EXPORTERS
CHANGED = {'world_reference_recalibration', 'world_reference_response',
           'world_response_hypotheses', 'world_response_transfer'}
read, write, aggregate = previous.read, previous.write, previous.aggregate
load_runtime, load_execution = previous.load_runtime, previous.load_execution
location, pair_diagnostics = previous.location, previous.pair_diagnostics


def prepare(output):
    from scripts.run_response_composition_transfer import declarations
    from scripts.run_response_numeric_transfer import METHOD, KNOWLEDGE

    assert not (output / 'protocol.json').exists(), 'Preserve prepared inputs'
    old = read(SOURCE / 'protocol.json')
    fresh = read(output / 'recommended_new_object_cases.json')
    cases = {'preservation': deepcopy(old['cases']), 'new_objects': deepcopy(fresh['cases'])}
    assert all(len(items) == 26 for items in cases.values())
    assert cases['preservation'].keys() == cases['new_objects'].keys()
    changed, inventory = [], {}
    after = output / 'runtime/after'
    after.mkdir(parents=True, exist_ok=False)
    for part in MODULES:
        before = output / 'runtime/before' / (part + '.py')
        assert before.read_text() == (SOURCE / 'runtime/after' / before.name).read_text(), part
        target = after / before.name
        shutil.copyfile(ROOT / 'src/hegel_machine/cognition' / target.name, target)
        if before.read_text() != target.read_text():
            changed.append(part)
    assert set(changed) == CHANGED, changed
    for phase in PHASES:
        execution = output / 'execution' / phase
        execution.mkdir(parents=True, exist_ok=False)
        for part in EXPORTERS:
            source = output / (part + '_before.py') if phase == 'before' else ROOT / 'scripts' / (part + '.py')
            if phase == 'before':
                assert source.read_text() == (SOURCE / 'execution/after' / (part + '.py')).read_text(), part
            shutil.copyfile(source, execution / (part + '.py'))
        load_runtime(output / 'runtime' / phase)
        load_execution(execution)
        from scripts.run_response_matched_adoption import prepare_bundle

        for source_id in dict.fromkeys(config['source_id'] for config in old['cases'].values()):
            source = SOURCE / 'after/sources' / source_id / 'numeric_bundle.json'
            original = read(source)
            bundle = prepare_bundle(original)
            assert declarations(original['files'][METHOD]) == declarations(bundle['files'][METHOD])
            assert {k: v for k, v in original.items() if k != 'files'} == {k: v for k, v in bundle.items() if k != 'files'}
            assert {k: v for k, v in original['files'].items() if k != METHOD} == {k: v for k, v in bundle['files'].items() if k != METHOD}
            archive = json.loads(bundle['files'][KNOWLEDGE])
            inventory.setdefault(source_id, {'source': str(source), 'models': len(archive['models']),
                'stratum': 'mixed' if archive['source_id'] == 'retained_response_archives' else 'empty_revision_fallback'})
            for cohort in COHORTS:
                if cohort == 'preservation' and phase == 'before':
                    continue
                write(output / cohort / phase / 'sources' / source_id / source.name, bundle)
        for cohort in COHORTS:
            if cohort == 'preservation' and phase == 'before':
                continue
            write(output / cohort / phase / 'protocol.json', {'cases': cases[cohort],
                'limits': old['limits'], 'arms': ['frozen_numeric'], 'planned_executions': 26})
    # Reuse the completed control in place. No result or input under this link
    # is written by the runner; only the current cohort's paired records change.
    (output / 'preservation/before').symlink_to(SOURCE / 'after', target_is_directory=True)
    write(output / 'protocol.json', {'cohorts': COHORTS, 'cases': cases,
        'phases': PHASES, 'limits': old['limits'], 'case_strata': old['case_strata'],
        'planned_executions': 78, 'archived_control_executions': 26,
        'planned_pairs': 52, 'maximum_workers': 2, 'model_calls': 0,
        'source': str(SOURCE), 'runtime_modules': list(MODULES),
        'new_target_signals': fresh['new_target_signals'],
        'intervention': 'After old-band support is exhausted, retain fixed coefficient programs and test compatible minimal/wider bands at the pending setting; already active targets return directly to parent research. Failed band continuation preserves the prior stage choice.',
        'control': 'The current intermediate-retirement controller, with identical frozen source knowledge and declarations. Its archived after results supply the 26 preservation controls; new objects run both frozen phases.',
        'baseline_reuse': 'Runtime modules and both execution exporters exactly match the archived after control. The preservation/before path points to that completed archive; no copied or rerun baseline traces.',
        'cohort_rule': '26 consumed regression conditions plus the same 13 archives and two laws at fresh target signals 7.25/14.75. Fixed conditions, no case replacement, tuning or reruns.',
        'scope': 'New objects within a deterministic development fixture, not independent random seeds, new mechanism families or an external benchmark.',
        'cost_scope': 'All paid physical actions and charged computation, including band revision. Export diagnostics are reported separately. Zero API/model calls.'})
    write(output / 'source_inventory.json', inventory)
    write(output / 'runtime_difference.json', {'changed_modules': changed,
        'unchanged_modules': sorted(set(MODULES) - set(changed)),
        'archived_control_runtime_and_exporters_match': True})
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    summarize(output)
    print(json.dumps({'status': 'prepared', 'new_executions': 78, 'archived_controls': 26}), flush=True)


def case_diagnostics(output, cohort, phase, case):
    directory = location(output / cohort, phase, case)
    trace, brief = read(directory / 'research_trace.json'), read(directory / 'brief.json')
    state, revisions = trace['state'], []
    questions = {q['id']: q for q in trace['questions']}
    trials = [*state.get('tests', []), *([state['test']] if state.get('test') else [])]
    for trial in trials:
        revision = trial.get('band_revision')
        if not revision:
            continue
        question = questions.get(revision.get('question_id'), {})
        answer = questions.get(trial['question'], {}).get('answer', {})
        revised_feedback = answer.get('response_feedback', answer).get('band_revision', {})
        point = next((p for p in brief['point_confirmations'] if p['question_id'] == trial['question']), {})
        revisions.append({'physical_question_id': trial['question'],
            'revision_question_id': revision.get('question_id'), 'number': trial['number'],
            'requested_setting': trial['prediction'], 'revision_status': revision.get('status'),
            'research_question_status': question.get('status'),
            'evidence': revision.get('evidence', []), 'compute_cost': revision.get('compute_cost', 0),
            'candidate_count': len(revision.get('candidates', [])),
            'hypothesis_count': len(revision.get('hypotheses', [])),
            'positive_supporter_count': len(revision.get('positive_supporters', [])),
            'revised_feedback': revised_feedback,
            'physical_question_status': questions.get(trial['question'], {}).get('status'),
            'physically_confirmed': point.get('physically_confirmed', False),
            'positive_evidence': point.get('actual_positive_evidence', [])})
    retirements = [t for t in trials if t.get('retirement')]
    knowledge = read(directory / 'learned_response_knowledge.json')
    revision_ids = {r['revision_question_id'] for r in revisions}
    diagnostic = {'cohort': cohort, 'phase': phase, 'case': case, 'revisions': revisions,
        'revision_count': len(revisions), 'revision_compute': sum(r['compute_cost'] for r in revisions),
        'revised_trials_confirmed': sum(r['physically_confirmed'] for r in revisions),
        'revision_statuses': dict(Counter(r['revision_status'] for r in revisions)),
        'exported_revision_models': sum(m.get('source_question') in revision_ids for m in knowledge['models']),
        'retirement_count': len(retirements),
        'inconclusive_retirements': sum(questions[t['question']]['status'] == 'inconclusive' for t in retirements),
        'retained_member_count': len(knowledge['models']), 'completed_targets': state.get('completed', [])}
    write(directory / 'band_revision_diagnostic.json', diagnostic)
    return diagnostic


def summarize(output):
    protocol = read(output / 'protocol.json')
    rows, diagnostics, pairs = [], [], []
    for cohort, cases in protocol['cases'].items():
        for case, config in cases.items():
            found = []
            for phase in PHASES:
                directory = location(output / cohort, phase, case)
                if not (directory / 'brief.json').exists():
                    continue
                rows.append({**read(directory / 'brief.json'), 'cohort': cohort, 'phase': phase,
                    'stratum': protocol['case_strata'][case],
                    'archived_control': cohort == 'preservation' and phase == 'before'})
                found.append(phase)
                if (directory / 'band_revision_diagnostic.json').exists():
                    diagnostics.append(read(directory / 'band_revision_diagnostic.json'))
                elif cohort == 'preservation' and phase == 'before':
                    old = read(directory / 'diagnostic_brief.json')
                    diagnostics.append({**old, 'cohort': cohort, 'phase': phase, 'revisions': [],
                        'revision_count': 0, 'revision_compute': 0, 'revised_trials_confirmed': 0,
                        'revision_statuses': {}, 'exported_revision_models': 0})
            if len(found) == 2:
                saved = output / cohort / 'paired' / (case + '.json')
                pairs.append({'cohort': cohort, **(read(saved) if saved.exists()
                             else pair_diagnostics(output / cohort, case, config))})
    completed = sum(not r['archived_control'] for r in rows)
    errors = read(output / 'batch_errors.json') if (output / 'batch_errors.json').exists() else []
    summary = {'status': 'execution_failures' if errors else 'finished' if completed == 78 else 'running' if completed else 'prepared',
        'completed': completed, 'planned': 78, 'archived_controls': sum(r['archived_control'] for r in rows),
        'paired_completed': len(pairs), 'planned_pairs': 52,
        'cohorts': {cohort: {
            'overall': {phase: aggregate([r for r in rows if r['cohort'] == cohort and r['phase'] == phase]) for phase in PHASES},
            'strata': {stratum: {phase: aggregate([r for r in rows if r['cohort'] == cohort and r['phase'] == phase and r['stratum'] == stratum])
                for phase in PHASES} for stratum in ('mixed', 'empty_revision_fallback')},
            'band_revision': {phase: {
                **{key: sum(d[key] for d in diagnostics if d['cohort'] == cohort and d['phase'] == phase)
                   for key in ('revision_count', 'revision_compute', 'revised_trials_confirmed', 'exported_revision_models', 'retirement_count', 'inconclusive_retirements')},
                'statuses': dict(sum((Counter(d['revision_statuses']) for d in diagnostics if d['cohort'] == cohort and d['phase'] == phase), Counter()))}
                for phase in PHASES}} for cohort in COHORTS}, 'paired': pairs, 'execution_errors': errors}
    write(output / 'brief.json', rows)
    write(output / 'band_revision_diagnostic.json', diagnostics)
    write(output / 'summary.json', summary)
    return summary


def run(output):
    protocol = read(output / 'protocol.json')
    jobs = [(cohort, phase, case) for cohort in COHORTS for case in protocol['cases'][cohort]
            for phase in PHASES if not (cohort == 'preservation' and phase == 'before')]
    assert not any(location(output / cohort, phase, case).exists() for cohort, phase, case in jobs), 'Preserve started runs'
    def launch(cohort, phase, case):
        log = output / 'logs' / cohort / phase / (case + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as stream:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--output', str(output),
                '--cohort', cohort, '--phase', phase, '--case', case], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT)
        return result.returncode, str(log)
    errors = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(launch, *job): job for job in jobs}
        for future in as_completed(pending):
            cohort, phase, case = pending[future]
            code, log = future.result()
            if code:
                errors.append({'cohort': cohort, 'phase': phase, 'case': case, 'returncode': code, 'log': log})
            write(output / 'batch_errors.json', errors)
            summary = summarize(output)
            print(json.dumps({'cohort': cohort, 'phase': phase, 'case': case,
                              'returncode': code, 'completed': summary['completed']}), flush=True)
    if errors:
        raise SystemExit('Recorded execution failures')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    for option in ('prepare', 'run', 'summarize'):
        parser.add_argument('--' + option, action='store_true')
    parser.add_argument('--cohort', choices=COHORTS)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--case')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.case and args.phase and args.cohort:
        load_runtime(output / 'runtime' / args.phase)
        load_execution(output / 'execution' / args.phase).run_case(output / args.cohort / args.phase, args.case, 'frozen_numeric')
        case_diagnostics(output, args.cohort, args.phase, args.case)
    elif args.prepare:
        prepare(output)
    elif args.run:
        run(output)
    elif args.summarize:
        print(json.dumps(summarize(output), indent=2))
    else:
        parser.error('Choose prepare, run, summarize, or case with cohort and phase')
