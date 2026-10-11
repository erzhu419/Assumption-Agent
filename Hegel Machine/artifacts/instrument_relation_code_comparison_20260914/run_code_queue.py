#!/usr/bin/env python3
"""Bounded Code-only queue; completed or budget-stopped cases are retained."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=('development', 'development_v2', 'development_v3', 'formal', 'formal_v2', 'formal_v3'))
    args = parser.parse_args()
    protocol = STUDY / ('protocol.json' if args.cohort == 'formal' else f'{args.cohort}_protocol.json')
    config = json.loads(protocol.read_text())
    output = STUDY / args.cohort
    logs = output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, case, index) for seed in config['development_apparatus']
            for index, case in enumerate(config['cases'])]
    results = []

    def run(job):
        seed, case, index = job
        checkpoint = output / 'code' / str(seed) / f'case_{index}' / 'code_checkpoint.json'
        if checkpoint.exists() and json.loads(checkpoint.read_text())['session']['results']:
            return {'seed': seed, 'case': case, 'status': 'already_learned', 'returncode': 0}
        command = [sys.executable, str(ROOT / 'scripts/run_instrument_relation_code_comparison.py'),
                   'learn', '--protocol', str(protocol), '--output', str(output),
                   '--seeds', str(seed), '--cases', case]
        start = monotonic()
        print(json.dumps({'event': 'start', 'seed': seed, 'case_index': index,
                          'time': datetime.now(timezone.utc).isoformat()}), flush=True)
        with (logs / f'{seed}_case_{index}_learn.log').open('a') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row = {'seed': seed, 'case': case, 'returncode': result.returncode,
               'process_wall_seconds': monotonic() - start, 'command': command}
        print(json.dumps({'event': 'end', **row}), flush=True)
        return row

    # Verify that the first actual pair can obtain actions before dispatching
    # further paid cases. A shared startup failure stops the remaining queue.
    first_pair = jobs[:2]

    def first_pair_has_action():
        for seed, case, index in first_pair:
            path = output / 'code' / str(seed) / f'case_{index}' / 'code_checkpoint.json'
            if path.exists():
                state = json.loads(path.read_text())['session']
                current = state['results'][0] if state['results'] else state['active']['agent_state']
                if any(event['action'] != 'provider_error' and not event.get('usage', {}).get('runtime_errors')
                       for event in current['events']):
                    return True
        return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(run, job) for job in first_pair}
        remaining = iter(jobs[2:])
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
            save(output / 'learn_queue_results.json', results)
            if first_pair_has_action():
                for _ in range(2 - len(pending)):
                    job = next(remaining, None)
                    if job is not None:
                        pending.add(pool.submit(run, job))
            elif not pending:
                raise SystemExit('First pair had no model actions; remaining cases were not dispatched.')
    if any(row['returncode'] for row in results):
        raise SystemExit('At least one process failed; retained checkpoints and logs, no automatic retry.')
    command = [sys.executable, str(ROOT / 'scripts/run_instrument_relation_code_comparison.py'),
               'summarize', '--protocol', str(protocol), '--output', str(output)]
    subprocess.run(command, cwd=ROOT, check=True, stdout=(logs / 'learning_summary.log').open('w'))
    print(json.dumps({'status': 'all_learning_processes_stopped', 'cases': len(jobs),
                      'terminal_dispatched': False}), flush=True)


if __name__ == '__main__':
    main()
