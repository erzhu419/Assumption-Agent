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
PRIOR = BASE.parent / 'shared_condition_research_20261003'
sys.path.insert(0, str(ROOT / 'scripts'))
from run_knowledge_lineage_study import read, save
COHORTS = ('seen_regression', 'successor_regression')


def require_learning_complete(output, protocol):
    receipt = BASE / 'learning_completion.json'
    if not receipt.exists() or not read(receipt)['complete']:
        raise ValueError('all 36 new learning chains must finish before terminal scoring')


def worker(cohort, phase, arm, root):
    import run_observed_condition_decoding_study as runner
    import run_research_investment_attribution_study as attribution
    attribution.confirm_learning_complete = require_learning_complete
    output = BASE / cohort
    protocol = read(output / 'protocol.json')
    cached = {}

    def fast_save(path, value):
        if path.suffix != '.gz':
            save(path, value)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name.startswith('visit'):
            cached['payload'] = gzip.compress(json.dumps(value, separators=(',', ':')).encode(), compresslevel=1)
        payload = cached['payload'] if path.name == 'checkpoint.json.gz' else cached.get('payload')
        if payload is None:
            payload = gzip.compress(json.dumps(value, separators=(',', ':')).encode(), compresslevel=1)
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_bytes(payload)
        temporary.replace(path)
        if path.name == 'checkpoint.json.gz':
            save(path.parent / 'learning_status.json', {
                'complete': value['study']['learning_complete'],
                'steps': value['study']['steps'], 'costs': value['costs'],
                'research_cpu_seconds': value['study']['cpu_seconds'],
                'revisits': len(value['revisits']),
            })

    runner.save = fast_save
    if phase == 'learn' and arm == 'no_marginal_history':
        original_load = runner.load_runtime
        def load(protocol, arm):
            module = original_load(protocol, arm)
            parent = module.UnifiedResearchSession
            class WithoutMarginalHistory(parent):
                def run(self, *args, **kwargs):
                    prefix = protocol['arms'][arm]['remove_action_experience_context']
                    keys = [key for key in self.library.action_experience if prefix in key]
                    if keys:
                        self.state.setdefault('study', {}).setdefault('marginal_history_removals', []).append({
                            'step': self.state.get('study', {}).get('steps', 0),
                            'contexts': keys, 'removed_contexts': len(keys)})
                        for key in keys:
                            del self.library.action_experience[key]
                    return super().run(*args, **kwargs)
            module.UnifiedResearchSession = WithoutMarginalHistory
            return module
        runner.load_runtime = load
    start = time.monotonic()
    getattr(runner, phase)(output, protocol, arm, root)
    save(output / 'runs' / arm / root / (phase + '_execution.json'), {
        'complete': True, 'wall_seconds': time.monotonic() - start,
        'external_model_calls': 0,
    })


def summarize(protocols):
    import run_observed_condition_decoding_study as runner
    for cohort, protocol in protocols.items():
        current = deepcopy(protocol)
        current['arms'] = {name: settings for name, settings in protocol['arms'].items()
            if settings['result_source'] == 'new run'}
        runner.summarize(BASE / cohort, current)
        summary = read(BASE / cohort / 'summary.json')
        previous = read(PRIOR / cohort / 'summary.json')
        for arm, source in (('anchor_162', 'anchor_162'), ('old_cumulative', 'new_cumulative')):
            summary['totals'][arm] = previous['totals'][source]
            summary['rows'].extend({**row, 'arm': arm} for row in previous['rows'] if row['arm'] == source)
            for stage, values in previous['by_stage'].items():
                summary['by_stage'][stage][arm] = values[source]
        summary['reference_sources'] = 'Reused compact 0.164 summaries, including the unchanged 0.162 anchor.'
        save(BASE / cohort / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('learn', 'evaluate', 'summarize'), required=True)
    parser.add_argument('--group', choices=('baseline', 'candidate'))
    parser.add_argument('--workers', type=int, default=16)
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
    with ThreadPoolExecutor(max_workers=min(16, args.workers)) as pool:
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
            'new_terminal_task_arm_scores': 420, 'baseline_results': 'Reused 48 previously completed chains.'})


if __name__ == '__main__':
    main()
