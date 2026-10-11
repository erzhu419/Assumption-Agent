"""Eight-worker execution queue; all jobs are learning, never terminal scoring."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import gzip
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SCRIPT = ROOT / 'scripts/run_research_investment_attribution_study.py'

def complete(cohort, arm, root):
    path = BASE / cohort / 'runs' / arm / root / 'checkpoint.json.gz'
    if not path.exists():
        return False
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)['study']['learning_complete']


def run(job):
    cohort, root = job
    for arm in ('candidate', 'no_target_withdrawal', 'delete_consumed_sources'):
        if complete(cohort, arm, root):
            continue
        directory = BASE / cohort / 'runs' / arm / root
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'learn.log').open('a') as log:
            process = subprocess.run([sys.executable, str(SCRIPT), '--output', str(BASE / cohort),
                '--phase', 'learn', '--arm', arm, '--root', root], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT)
        print({'cohort': cohort, 'root': root, 'arm': arm, 'returncode': process.returncode}, flush=True)
        if process.returncode:
            return process.returncode
    return 0

jobs = [(cohort, root) for cohort in ('fresh_replication',)
    for root in json.loads((BASE / cohort / 'protocol.json').read_text())['roots']]
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(run, jobs))
(BASE / 'candidate_learning_jobs.json').write_text(json.dumps(
    [{'cohort': c, 'root': r, 'returncode': code} for (c, r), code in zip(jobs, results)], indent=2) + '\n')
sys.exit(int(any(results)))
