"""Run fixed cohorts with shared learning completion before new terminals."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_research_investment_attribution_study import read, save, confirm_learning_complete
BASE = Path(__file__).resolve().parent
COHORTS = ('mixed_regression','instrument_seen','instrument_fresh')

def runner(cohort):
    return ROOT / 'scripts' / ('run_observed_condition_decoding_study.py' if cohort.startswith('instrument_') else 'run_research_investment_attribution_study.py')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('learn','evaluate','summarize'), required=True)
    parser.add_argument('--arm', choices=('baseline_159','candidate'))
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    protocols = {name:read(BASE/name/'protocol.json') for name in COHORTS}
    if args.phase == 'summarize':
        for name in COHORTS:
            subprocess.run([sys.executable,str(runner(name)),'--output',str(BASE/name),'--phase','summarize'],cwd=ROOT,check=True)
        return
    if args.phase == 'evaluate':
        confirm_learning_complete(BASE/COHORTS[0],protocols[COHORTS[0]])
    jobs = []
    for cohort,p in protocols.items():
        for arm,settings in p['arms'].items():
            group = 'candidate' if arm in ('candidate','new_cumulative') else 'baseline_159'
            if args.arm is not None and args.arm != group:
                continue
            if settings['result_source'].startswith('reused'):
                continue
            jobs.extend((cohort,arm,root) for root in p['roots'])
    def run(job):
        cohort,arm,root = job
        log = BASE/cohort/'runs'/arm/root/(args.phase+'.log')
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            result = subprocess.run([sys.executable,str(runner(cohort)),'--output',str(BASE/cohort),'--phase',args.phase,'--arm',arm,'--root',root],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        record = {'cohort':cohort,'arm':arm,'root':root,'returncode':result.returncode}
        print(record,flush=True)
        return record
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(run,jobs))
    save(BASE/(args.phase+'_'+(args.arm or 'all')+'_jobs.json'),records)
    if any(r['returncode'] for r in records):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
