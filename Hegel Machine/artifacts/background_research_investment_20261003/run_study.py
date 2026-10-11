"""Fixed offline cohorts; independent processes and one global learning barrier."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from run_research_investment_attribution_study import read, save, confirm_learning_complete
COHORTS = ('seen_regression','successor_regression')

def worker(cohort, phase, arm, root):
    import run_observed_condition_decoding_study as runner
    output = BASE/cohort
    protocol = read(output/'protocol.json')
    if phase == 'evaluate':
        confirm_learning_complete(output, protocol)
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
                        self.state.setdefault('study',{}).setdefault('marginal_history_removals',[]).append({
                            'step':self.state.get('study',{}).get('steps',0), 'contexts':keys,
                            'removed_contexts':len(keys)})
                        for key in keys:
                            del self.library.action_experience[key]
                    return super().run(*args, **kwargs)
            module.UnifiedResearchSession = WithoutMarginalHistory
            return module
        runner.load_runtime = load
    getattr(runner, phase)(output, protocol, arm, root)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('learn','evaluate','summarize'), required=True)
    parser.add_argument('--group', choices=('baseline','candidate'))
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--cohort', choices=COHORTS)
    parser.add_argument('--arm')
    parser.add_argument('--root')
    args = parser.parse_args()
    if args.root:
        worker(args.cohort,args.phase,args.arm,args.root)
        return
    protocols = {name:read(BASE/name/'protocol.json') for name in COHORTS}
    if args.phase == 'summarize':
        import run_observed_condition_decoding_study as runner
        for name,protocol in protocols.items():
            runner.summarize(BASE/name,protocol)
        return
    if args.phase == 'evaluate':
        confirm_learning_complete(BASE/COHORTS[0],protocols[COHORTS[0]])
    jobs = []
    for cohort,protocol in protocols.items():
        for arm,settings in protocol['arms'].items():
            group = 'baseline' if arm == 'old_cumulative' else 'candidate'
            if args.group is not None and args.group != group:
                continue
            if settings['result_source'].startswith('reused'):
                continue
            jobs.extend((cohort,arm,root) for root in protocol['roots'])
    def run(job):
        cohort,arm,root = job
        log = BASE/cohort/'runs'/arm/root/(args.phase+'.log')
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            result = subprocess.run([sys.executable,str(Path(__file__).resolve()),'--phase',args.phase,
                '--cohort',cohort,'--arm',arm,'--root',root],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        record = {'cohort':cohort,'arm':arm,'root':root,'returncode':result.returncode}
        print(record,flush=True)
        return record
    with ThreadPoolExecutor(max_workers=min(16,args.workers)) as pool:
        records = list(pool.map(run,jobs))
    save(BASE/(args.phase+'_'+(args.group or 'all')+'_jobs.json'),records)
    if any(record['returncode'] for record in records):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
