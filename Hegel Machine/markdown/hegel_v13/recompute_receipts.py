#!/usr/bin/env python3
"""Recompute outcomes and paired, source-chain-level summaries from saved v12 receipts.
No fitting, simulator, model/API calls, or experiment reruns.
"""
from pathlib import Path
from itertools import product
from statistics import mean,median
import json,hashlib
import os
BUNDLE=Path(os.environ.get('HEGEL_V12_BUNDLE', '/mnt/data/hegel_v12_input')).expanduser().resolve()
ROOT=BUNDLE/'receipts/v12_autonomous_reconciliation_and_utility_20261117'
OUT=Path(__file__).parent

def main():
 p=ROOT/'deliverable_c_v12_knowledge_utility_20261117.json';d=json.loads(p.read_text())
 chains=d['unseen_evaluation_chains'];arms=list(chains[0]['arms_target_results'])
 result={'kind':'saved_receipt_recalculation','no_scientific_rerun':True,
         'receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'independent_clusters':len(chains),'targets_per_cluster':3,'arms':{},'paired':{},'improved_target_cases':[]}
 for a in arms:
  es=[e for c in chains for e in c['arms_target_results'][a]]
  result['arms'][a]={'n':len(es),'passes':sum(e['strict_pass'] for e in es),
   'false_confirmations':sum(e['false_internal_confirmation'] for e in es),
   'mean_compute':mean(e['measured_costs']['compute_cost'] for e in es),
   'mean_observations':mean(e['measured_costs']['observation_cost'] for e in es),
   'mean_measurements':mean(e['measured_costs']['measurement_cost'] for e in es),
   'mean_wall_seconds':mean(e['measured_costs']['wall_seconds'] for e in es),
   'mean_external_mse':mean(e['external_fresh48_mse'] for e in es),
   'within_budget_all':all(e['within_predeclared_budget'] for e in es)}
 for ref in ('R_ordinary_same_api','K_cold','K_object_deleted','K_relations_deleted'):
  sd=[];cd=[];pairs=[];wins=losses=0
  for c in chains:
   rr=c['arms_target_results'][ref];kk=c['arms_target_results']['K_reconcile']
   diff=sum(int(k['strict_pass'])-int(r['strict_pass']) for k,r in zip(kk,rr))
   dc=sum(r['measured_costs']['compute_cost']-k['measured_costs']['compute_cost'] for k,r in zip(kk,rr))
   sd.append(diff);cd.append(dc/3)
   for r,k in zip(rr,kk):
    win=int(k['strict_pass'])-int(r['strict_pass']);wins+=win==1;losses+=win==-1
    if ref=='R_ordinary_same_api' and win:
     result['improved_target_cases'].append({'chain':c['chain_id'],'target':k['target_regime_role'],
       'reference':{x:r[x] for x in ('status','confirmation_loss','external_fresh48_mse','strict_pass')},
       'reconcile':{x:k[x] for x in ('status','confirmation_loss','external_fresh48_mse','strict_pass')},
       'compute_ref':r['measured_costs']['compute_cost'],'compute_reconcile':k['measured_costs']['compute_cost']})
  obs=sum(sd);vals=[abs(x) for x in sd if x]
  exact_p=(sum(abs(sum(s*v for s,v in zip(signs,vals)))>=abs(obs)-1e-12 for signs in product((-1,1),repeat=len(vals)))/(2**len(vals))) if vals else 1.
  base=result['arms'][ref];full=result['arms']['K_reconcile']
  result['paired'][ref]={'success_difference_per_chain':sd,'target_compute_saving_per_chain_mean':cd,
      'task_wins':wins,'task_losses':losses,'task_ties':24-wins-losses,
      'success_rate_difference_pp':100*sum(sd)/24,
      'source_cluster_sign_flip_two_sided_p':exact_p,
      'sign_flip_assumptions':'paired independent source chains; signs exchangeable under zero-effect null; descriptive small-sample diagnostic, not proof of random task sampling',
      'mean_target_compute_saving':mean(cd),'median_chain_mean_compute_saving':median(cd),
      'compute_saving_percent':100*(base['mean_compute']-full['mean_compute'])/base['mean_compute'],
      'wall_change_percent':100*(full['mean_wall_seconds']-base['mean_wall_seconds'])/base['mean_wall_seconds'],
      'observation_change_percent':100*(full['mean_observations']-base['mean_observations'])/base['mean_observations']}
 # Reconstruct explicitly reported incremental maintenance ratio; do not call it end-to-end payback.
 maint=mean(c['source_formation_measured_ledger']['reconciliation_subset_compute_measured'] for c in chains)
 source=mean(c['source_formation_measured_ledger']['source_total_compute_measured'] for c in chains)
 save=result['paired']['R_ordinary_same_api']['mean_target_compute_saving']
 result['amortization']={'reported_source_mean':source,'reported_maintenance_subset_mean':maint,
   'maintenance_only_break_even_targets':maint/save,
   'scope':'conditional on same source experiences already available; excludes extra acquisition costs and does not establish whole-system net utility'}
 # Source and target outcome details and cache residual compute
 recon=[]
 for ch in chains:
  for e in ch['arms_target_results']['R_ordinary_same_api']:
   s=(e.get('branch_dispatch_telemetry') or {}).get('on_demand_archive_family_reconstruction')
   if s:recon.append({'episode':e['episode_tag'],**s})
 result['ordinary_reconstruction_last_dispatch_stats']=recon
 result['saved_vs_raw_summary_matches']=all(
  d['unseen_8chain_aggregate']['arm_aggregates'][a]['strict_passes']==result['arms'][a]['passes']
  and abs(d['unseen_8chain_aggregate']['arm_aggregates'][a]['measured_target_costs']['mean_compute']-result['arms'][a]['mean_compute'])<.011 for a in arms)
 (OUT/'recomputed_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('ordinary_reconstruction_last_dispatch_stats',)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
