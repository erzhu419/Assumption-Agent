"""Fixed offline cohorts; serialize each learning snapshot once, gzip level one."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PRIOR = BASE.parent / 'independent_confirmation_20261003'
sys.path.insert(0, str(ROOT / 'scripts'))
from run_knowledge_lineage_study import read, save
COHORTS = ('successor_regression',)


def require_learning_complete(output, protocol):
    receipt = BASE / 'learning_completion.json'
    if not receipt.exists() or not read(receipt)['complete']:
        raise ValueError('all 24 new learning chains must finish before terminal scoring')


def worker(cohort, phase, arm, root):
    import run_observed_condition_decoding_study as runner
    import run_research_investment_attribution_study as attribution
    attribution.confirm_learning_complete = require_learning_complete
    output = BASE / cohort
    protocol = read(output / 'protocol.json')
    def fast_save(path, value):
        if path.suffix != '.gz':
            save(path, value)
            return
        if path.name != 'checkpoint.json.gz':
            return  # The checkpoint contains the complete paid trajectory.
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        payload = gzip.compress(json.dumps(value, separators=(',', ':')).encode(), compresslevel=1)
        temporary.write_bytes(payload)
        temporary.replace(path)
        save(path.parent / 'learning_status.json', {
            'complete': value['study']['learning_complete'],
            'steps': value['study']['steps'], 'costs': value['costs'],
            'research_cpu_seconds': value['study']['cpu_seconds'],
            'revisits': len(value['revisits']),
        })

    runner.save = fast_save
    if phase == 'learn' and arm == 'prior_sampling':
        original_load = runner.load_runtime
        def load(protocol, arm):
            module = original_load(protocol, arm)
            from sampling_ablation import install_prior_sampling
            install_prior_sampling()
            return module
        runner.load_runtime = load
    start = time.monotonic()
    getattr(runner, phase)(output, protocol, arm, root)
    save(output / 'runs' / arm / root / (phase + '_execution.json'), {
        'complete': True, 'wall_seconds': time.monotonic() - start,
        'external_model_calls': 0,
    })


def research_evidence(state):
    """Keep the fields used by the detailed summary, omitting raw libraries/rows."""
    from finish_summary import trim
    from confirmation_evidence import confirmation_diagnostics
    def report_view(report):
        uncertainty = report.get('condition_uncertainty') or {}
        return {'status': report['status'], 'validation_loss': report.get('validation_loss'),
            'condition_uncertainty': {key: uncertainty.get(key)
                for key in ('status', 'expected_prediction_ambiguity')},
            'actions': [trim(action) for action in report['actions']]}
    experience = {}
    for context, actions in state['knowledge'].get('action_experience', {}).items():
        if ':condition_followup_research:' in context:
            experience[context] = {'condition_followup_research': {
                key: value for key, value in actions.get('condition_followup_research', {}).items()
                if isinstance(value, (int, float)) or key in ('instrument_revisits', 'instrument_cycles')}}
    return {'costs': state['costs'], 'study': state['study'],
        'posthoc_confirmation_diagnostics': [entry for item in state['results']
            for entry in confirmation_diagnostics(item['research'], item['task_position'])],
        'screening_compute_cost': state.get('screening_compute_cost', 0),
        'scheduler_events': state['scheduler_events'],
        'condition_revisit_quotes': [None] * len(state.get('condition_revisit_quotes', [])),
        'knowledge': {'action_experience': experience},
        'results': [{'task_position': row['task_position'], 'research': report_view(row['research'])}
            for row in state['results']],
        'revisits': [{**{key: value for key, value in visit.items() if key not in ('candidate', 'previous')},
            'previous': report_view(visit['previous']),
            'method_draft_id': visit.get('candidate', {}).get('research_draft_id'),
            'method_draft_ids': visit.get('candidate', {}).get('method_opportunity', {}).get('draft_ids', []),
            'instrument_cost_quote': {
                'expected_resource_costs': visit.get('candidate', {}).get('instrument_condition_experiment', {}).get('decision', {}).get('expected_cycle_costs'),
                'screening_compute_cost': visit.get('candidate', {}).get('instrument_condition_experiment', {}).get('screening_compute_cost', 0),
                'sampling_context': visit.get('candidate', {}).get('instrument_condition_experiment', {}).get('plan', {}).get('sampling_context')}}
            for visit in state['revisits']]}


def summarize(protocols):
    import run_observed_condition_decoding_study as runner
    original_read = runner.read
    def read_with_evidence(path):
        value = original_read(path)
        if path.name == 'checkpoint.json.gz':
            target = path.parent / 'research_evidence.json.gz'
            payload = gzip.compress(json.dumps(research_evidence(value), separators=(',', ':')).encode(),
                compresslevel=1)
            target.write_bytes(payload)
        return value
    runner.read = read_with_evidence
    for cohort, protocol in protocols.items():
        current = deepcopy(protocol)
        current['arms'] = {name: settings for name, settings in protocol['arms'].items()
            if settings['result_source'] == 'new run'}
        runner.load_runtime(current, 'new_cumulative')
        runner.summarize(BASE / cohort, current)
        summary = read(BASE / cohort / 'summary.json')
        previous = read(PRIOR / cohort / 'summary.json')
        for arm, source in (('anchor_164', 'anchor_164'), ('old_cumulative', 'new_cumulative')):
            summary['totals'][arm] = previous['totals'][source]
            summary['rows'].extend({**row, 'arm': arm} for row in previous['rows'] if row['arm'] == source)
            for stage, values in previous['by_stage'].items():
                summary['by_stage'][stage][arm] = values[source]
        summary['reference_sources'] = 'Reused compact 0.171 summaries, including the unchanged 0.164 quality reference.'
        save(BASE / cohort / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('learn', 'evaluate', 'summarize'), required=True)
    parser.add_argument('--group', choices=('baseline', 'candidate'))
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--cohort', choices=COHORTS)
    parser.add_argument('--arm')
    parser.add_argument('--root')
    args = parser.parse_args()
    if args.root:
        worker(args.cohort, args.phase, args.arm, args.root)
        return
    protocols = {name: read(BASE / name / 'protocol.json') for name in COHORTS}
    if args.phase == 'summarize':
        summarize(protocols)
        return
    if args.phase == 'evaluate':
        require_learning_complete(BASE, protocols[COHORTS[0]])
    jobs = [(cohort, arm, root) for cohort, protocol in protocols.items()
        for arm, settings in protocol['arms'].items() if settings['result_source'] == 'new run'
        for root in protocol['roots']]
    if args.group == 'baseline':
        raise ValueError('Both baselines are reused; do not rerun settled learning or terminal scores.')
    def run(job):
        cohort, arm, root = job
        directory = BASE / cohort / 'runs' / arm / root
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / (args.phase + '.log')).open('a') as stream:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--phase', args.phase,
                '--cohort', cohort, '--arm', arm, '--root', root], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT)
        record = {'cohort': cohort, 'arm': arm, 'root': root, 'returncode': result.returncode}
        print(record, flush=True)
        return record
    with ThreadPoolExecutor(max_workers=min(12, args.workers)) as pool:
        records = list(pool.map(run, jobs))
    save(BASE / (args.phase + '_' + (args.group or 'all') + '_jobs.json'), records)
    if any(record['returncode'] for record in records):
        raise SystemExit(1)
    if args.phase == 'learn':
        complete = all(read(BASE / cohort / 'runs' / arm / root / 'learning_status.json')['complete']
            for cohort, arm, root in jobs)
        if not complete:
            raise RuntimeError('A learning chain returned without finishing.')
        save(BASE / 'learning_completion.json', {'complete': True, 'new_learning_chains': len(jobs),
            'new_terminal_task_arm_scores': 336, 'baseline_results': 'Reused 24 previously completed long chains.'})


if __name__ == '__main__':
    main()
