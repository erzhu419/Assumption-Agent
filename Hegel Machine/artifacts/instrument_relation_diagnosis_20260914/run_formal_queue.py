#!/usr/bin/env python3
"""Fixed apparatus jobs, two workers, all learning before terminal acquisition."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PROTOCOL = json.loads((BASE / 'protocol.json').read_text())
RESULTS = BASE / 'formal'
DRIVER = ROOT / 'scripts/run_instrument_relation_diagnosis.py'
ARMS = ('baseline', 'revised')


def complete(mode, arm, seed):
    files = [RESULTS / arm / str(seed) / (case + '.json') for case in PROTOCOL['cases']]
    return all(path.exists() and (mode == 'learn' or 'terminal' in json.loads(path.read_text())) for path in files)


def run(job):
    mode, arm, seed = job
    command = [sys.executable, str(DRIVER), mode, '--arm', arm, '--seeds', str(seed),
               '--protocol', str(BASE / 'protocol.json'), '--results-directory', str(RESULTS)]
    path = RESULTS / 'logs' / f'{arm}_{seed}_{mode}.log'
    with path.open('w') as stream:
        process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record = {'mode': mode, 'arm': arm, 'seed': seed, 'exit_code': process.returncode,
              'command': command, 'log': str(path.relative_to(BASE))}
    print(json.dumps(record), flush=True)
    return record


for mode in ('learn', 'evaluate'):
    if mode == 'evaluate' and any(not complete('learn', arm, seed)
        for seed in PROTOCOL['development_apparatus'] for arm in ARMS):
        raise SystemExit('Incomplete learning retained; terminal not started.')
    jobs = [(mode, arm, seed) for seed in PROTOCOL['development_apparatus'] for arm in ARMS
            if not complete(mode, arm, seed)]
    print(json.dumps({'mode': mode, 'jobs': len(jobs), 'workers': 2}), flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(run, jobs))
    (RESULTS / f'{mode}_queue_results.json').write_text(json.dumps(rows, indent=2) + '\n')
    if any(row['exit_code'] for row in rows):
        raise SystemExit('Failed processes retained without automatic retries.')
subprocess.run([sys.executable, str(DRIVER), 'summarize', '--protocol', str(BASE / 'protocol.json'),
                '--results-directory', str(RESULTS)], cwd=ROOT, check=True)
