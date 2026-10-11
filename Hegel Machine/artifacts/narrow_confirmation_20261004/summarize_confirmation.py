"""Aggregate existing paid allocator outcomes without repeating prediction work."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ENTRY=Path(__file__).resolve()
BASE=Path(os.environ.get('HEGEL_PORTFOLIO_OUTPUT',str(ENTRY.parent)))
ROOT=ENTRY.parents[2]

def read(path):
    if path.suffix=='.gz':
        return json.loads(gzip.decompress(path.read_bytes()))
    return json.loads(path.read_text())

def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def diagnose(arm,protocol):
    cpu=time.process_time()
    tolerance=protocol['config']['minimum_improvement']
    records=[];plans=[];pool_counts=Counter();stops=Counter();decisions=Counter();replan_compute=0;pair_eval_compute=0
    for root in protocol['roots']:
        receipt=read(BASE/'runs'/arm/root/'pair_confirmation.json.gz')
        for visit in receipt['visits']:
            by_probe={plan['probe_id']:plan for plan in visit['plans']}
            for plan in visit['plans']:
                if plan.get('confirmation_pair') is not None:
                    selection=plan.get('residual_pool_selection')
                    pool_counts[(selection or {}).get('selection','baseline_local')]+=1
                    plans.append({'root':root,'position':visit['position'],
                        'after_revisits':visit['after_revisits'],'probe_id':plan['probe_id'],
                        'expected_disagreement':plan['expected_disagreement'],
                        'fixed_control_background':plan.get('fixed_control_background'),
                        'residual_pool_selection':selection,
                        'interval_exposure':plan.get('interval_exposure'),
                        'planning_compute_cost':plan.get('planning_compute_cost'),
                        'planner_candidate_count':plan.get('planner_candidate_count')})
            # The caller re-scores the final probe. Count each allocator probe once.
            for allocation in visit['pair_allocations']:
                pair=allocation['confirmation_pair']
                outcomes={outcome['probe_id']:outcome for outcome in allocation['outcomes']}
                stops[allocation.get('stop_reason','unknown')]+=1
                for decision in allocation['decisions']:
                    decisions[decision.get('stage','unknown')+':'+decision['reason']]+=1
                replan_compute+=allocation['planning_compute_cost']
                pair_eval_compute+=allocation['evaluation_compute_cost']
                for batch_index,probe in enumerate(allocation['probe_ids']):
                    batch=visit['fresh_batches_by_probe'][probe]
                    outcome=outcomes.get(probe)
                    disagreement=outcome['prediction_disagreement'] if outcome else None
                    old_loss=outcome['baseline_loss'] if outcome else None
                    new_loss=outcome['selected_loss'] if outcome else None
                    plan=by_probe.get(probe,{})
                    records.append({'root':root,'position':visit['position'],
                        'after_revisits':visit['after_revisits'],'probe_id':probe,
                        'batch_index_in_allocation':batch_index,'rows':len(batch['observations']),
                        'prediction_disagreement':disagreement,
                        'distinguished':outcome['distinguished'] if outcome else None,'before_loss':old_loss,'after_loss':new_loss,
                        'positive_gain':old_loss-new_loss>tolerance if outcome else None,
                        'negative_gain':new_loss-old_loss>tolerance if outcome else None,
                        'recorded_outcome':outcomes.get(probe),
                        'fixed_control_background':plan.get('fixed_control_background'),
                        'residual_pool_selection':plan.get('residual_pool_selection'),
                        'expected_disagreement':plan.get('expected_disagreement'),
                        'interval_exposure':plan.get('interval_exposure'),
                        'stop_reason':allocation.get('stop_reason')})
    result={'arm':arm,'roots':list(protocol['roots']),
        'scope':'Aggregated already paid allocator outcomes. No new sampling, fitting, prediction evaluation or learning.',
        'posthoc_aggregation_cpu_seconds':time.process_time()-cpu,
        'pair_plans':len(plans),'residual_pool_plan_choices':dict(pool_counts),
        'actual_pair_batches':len(records),'actual_pair_rows':sum(r['rows'] for r in records),
        'actual_distinguishing_batches':sum(r['distinguished'] is True for r in records),
        'actual_positive_gain_batches':sum(r['positive_gain'] is True for r in records),
        'actual_negative_gain_batches':sum(r['negative_gain'] is True for r in records),
        'non_distinguishing_rows':sum(r['rows'] for r in records if r['distinguished'] is False),
        'actual_extra_batches':sum(r['batch_index_in_allocation']>0 for r in records),
        'actual_pair_replanning_compute':replan_compute,'actual_pair_evaluation_compute':pair_eval_compute,
        'missing_allocator_outcomes':sum(r['recorded_outcome'] is None for r in records),
        'allocator_stop_reasons':dict(stops),'investment_decision_reasons':dict(decisions),
        'plans':plans,'batches':records}
    save(BASE/(arm+'_pair_diagnostics.json'),result)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('baseline','candidate'))
    args=parser.parse_args();protocol=read(BASE/'development_protocol.json')
    if args.arm:
        diagnose(args.arm,protocol);return
    for arm in protocol['arms']:
        subprocess.run([sys.executable,str(ENTRY),'--arm',arm],check=True,cwd=ROOT)
    summary=read(BASE/'development_summary.json')
    diagnostics={arm:read(BASE/(arm+'_pair_diagnostics.json')) for arm in protocol['arms']}
    result={'complete':True,'candidate_version':protocol['candidate_version'],
        'scope':protocol['scope'],'arms':summary['arms'],'pair_diagnostics':diagnostics,
        'pairs':summary['pairs'],
        'new_failures':[p for p in summary['pairs'] if p['baseline_mse']<=protocol['pass_mse']<p['candidate_mse']],
        'recoveries':[p for p in summary['pairs'] if p['candidate_mse']<=protocol['pass_mse']<p['baseline_mse']]}
    a,c=summary['arms']['baseline'],summary['arms']['candidate']
    result['candidate_minus_baseline']={'passed':c['passed']-a['passed'],'mse':c['mse']-a['mse'],
        'all_costs':{key:c['all_costs'][key]-a['all_costs'][key] for key in c['all_costs']},
        'measured_cpu_seconds':c['measured_cpu_seconds']-a['measured_cpu_seconds']}
    save(BASE/'compact_results.json',result)
    print({arm:{key:diagnostics[arm][key] for key in ('pair_plans','actual_pair_batches','actual_pair_rows',
        'actual_distinguishing_batches','actual_positive_gain_batches','non_distinguishing_rows')}
        for arm in protocol['arms']})

if __name__=='__main__':
    main()
