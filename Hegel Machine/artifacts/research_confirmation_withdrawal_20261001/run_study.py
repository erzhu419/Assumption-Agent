"""Run the two fixed confirmation/withdrawal cohorts through the existing runner."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_research_investment_attribution_study import read, save, confirm_learning_complete

BASE = Path(__file__).resolve().parent
RUNNER = ROOT / 'scripts/run_research_investment_attribution_study.py'
COHORTS = ('seen_regression', 'fresh_replication')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('learn', 'evaluate', 'summarize'), required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    protocols = {name: read(BASE / name / 'protocol.json') for name in COHORTS}
    if args.phase == 'summarize':
        for cohort in COHORTS:
            subprocess.run([sys.executable, str(RUNNER), '--output', str(BASE / cohort),
                '--phase', 'summarize'], cwd=ROOT, check=True)
        return
    if args.phase == 'evaluate':
        confirm_learning_complete(BASE / COHORTS[0], protocols[COHORTS[0]])
    jobs = [(cohort, arm, root) for cohort, protocol in protocols.items()
        for arm in protocol['arms'] for root in protocol['roots']
        if not (args.phase == 'learn' and arm == 'baseline_157_1')]
    def run(job):
        cohort, arm, root = job
        log = BASE / cohort / 'runs' / arm / root / (args.phase + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            result = subprocess.run([sys.executable, str(RUNNER), '--output', str(BASE / cohort),
                '--phase', args.phase, '--arm', arm, '--root', root], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT)
        record = {'cohort': cohort, 'arm': arm, 'root': root, 'returncode': result.returncode}
        print(record, flush=True)
        return record
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, jobs))
    save(BASE / (args.phase + '_jobs.json'), results)
    if any(result['returncode'] for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
