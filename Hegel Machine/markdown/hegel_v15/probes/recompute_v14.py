#!/usr/bin/env python3
"""Recompute V14 gates and summaries from saved MSE/cost receipts.

This does not recompute prediction MSE from trajectories, run scientific
experiments, or call models. Pairing and prefix arithmetic are descriptive;
they do not establish causal attribution or semantic parity of replay.
"""
from __future__ import annotations
import argparse
from collections import Counter
from itertools import product
from pathlib import Path
from statistics import mean,median
import hashlib,json

SUB=Path('receipts/v14_continuous_object_and_fair_comparison_20261118')
FILES={'D2':'deliverable_2_unconfounded_noH_source_paired_study_20261118.json','D3':'deliverable_3_continuous_repaired_object_study_20261118.json'}
TOL=1e-7
BUDGETS={'D2_source':(660,13200,740000),'D2_target':(620,12400,680000),'D3_markov':(520,11000,540000),'D3_conflict':(560,11500,880000),'D3_target':(580,11800,840000)}
K='K_eager_noH';R='R_lazy_noH';COLD='K_cold';BR='K_minimal_branch_removed_raw_kept';OBJ='K_minimal_repaired_object_removed_other_sources_kept'

def gates(e,kind,mismatches):
 b=BUDGETS[kind];cost=e['measured_costs'];within=all(cost[k]<=v for k,v in zip(('observation_cost','measurement_cost','compute_cost'),b))
 internal=e['status']=='completed' and e['confirmation_loss']is not None and e['confirmation_loss']<=TOL
 ext=e['external_fresh48_mse']<=TOL
 calc={'internally_confirmed':bool(internal),'strict_pass':bool(internal and ext and within),'false_internal_confirmation':bool(internal and not ext),'within_predeclared_budget':within,'unresolved_or_unconfirmed':not internal,'unconfirmed_yet_accurate_on_external':bool(not internal and ext)}
 for k,v in calc.items():
  if e[k]!=v:mismatches.append({'arm':e['arm'],'episode':e['episode_tag'],'field':k,'reported':e[k],'recomputed':v})
  e[k]=v

def brief(e):
 return {k:e.get(k)for k in ('arm','episode_tag','episode_uid','status','strict_pass','internally_confirmed','confirmation_loss','external_fresh48_mse','unconfirmed_yet_accurate_on_external','measured_costs','selected_kind')}

def agg(es):
 ans={'n':len(es),'strict_passes':sum(e['strict_pass']for e in es),'internally_confirmed':sum(e['internally_confirmed']for e in es),'false_confirmations':sum(e['false_internal_confirmation']for e in es),'unresolved':sum(e['unresolved_or_unconfirmed']for e in es),'unconfirmed_yet_external_accurate':sum(e['unconfirmed_yet_accurate_on_external']for e in es),'all_within_budget':all(e['within_predeclared_budget']for e in es),'costs':{},'failures':[brief(e)for e in es if not e['strict_pass']]}
 for key in es[0]['measured_costs']:
  vals=[e['measured_costs'][key]for e in es];ans['costs'][key]={'total':sum(vals),'mean':mean(vals),'median':median(vals)}
 ans['boundary_probes_executed']=sum(e.get('boundary_probes_executed_count',0)for e in es)
 ans['reconstruction_compute']=sum(e.get('ordinary_lazy_episode_reconstruction_compute',0)for e in es)
 ans['ordinary_family_layer2_built']=sum(e.get('ordinary_lazy_layer2_families_built',0)for e in es)
 return ans

def sf(v):
 vals=[x for x in v if x]
 return sum(abs(sum(s*abs(x)for s,x in zip(signs,vals)))>=abs(sum(vals))-1e-9 for signs in product((-1,1),repeat=len(vals)))/2**len(vals)if vals else 1.

def paired(chains,focal,ref):
 success=[];costs=[];changed=[];wins=losses=ties=0;same=[]
 for ch in chains:
  fs=ch['arms_target_results'][focal];rs=ch['arms_target_results'][ref];assert len(fs)==len(rs)
  success.append(sum(int(f['strict_pass'])-int(r['strict_pass'])for f,r in zip(fs,rs)))
  costs.append(mean(r['measured_costs']['compute_cost']-f['measured_costs']['compute_cost']for f,r in zip(fs,rs)))
  for f,r in zip(fs,rs):
   assert f['episode_tag']==r['episode_tag']
   delta=int(f['strict_pass'])-int(r['strict_pass']);wins+=delta>0;losses+=delta<0;ties+=delta==0
   if delta:changed.append({'chain':ch['chain_id'],'role':f['target_regime_role'],'focal':brief(f),'reference':brief(r)})
   if f['strict_pass']and r['strict_pass']:same.append(r['measured_costs']['compute_cost']-f['measured_costs']['compute_cost'])
 return {'focal':focal,'reference':ref,'source_clusters':len(chains),'success_difference_per_chain':success,'target_compute_saving_per_chain_mean':costs,'wins_losses_ties':[wins,losses,ties],'success_sign_flip_p':sf(success),'compute_sign_flip_p_exploratory':sf(costs),'mean_compute_saving':mean(costs),'median_chain_mean_compute_saving':median(costs),'changed_cases':changed,'both_success_pairs_n':len(same),'both_success_pairs_mean_saving':mean(same)if same else None,'both_success_subset_scope':'Post-outcome selected descriptive subset; not a causal estimate'}

def source_typed_events(e,cid):
 if cid is None:return []
 return [x for x in (e.get('causal_consumption_telemetry')or{}).get('consumption_events',[]) if cid in (x.get('source_concept_ids')or[])]

def run(bundle):
 data={};hashes={}
 for k,f in FILES.items():
  p=bundle/SUB/f;raw=p.read_bytes();data[k]=json.loads(raw);hashes[k]=hashlib.sha256(raw).hexdigest()
 out={'scope':'Recomputed decisions/aggregates from saved per-episode MSE and costs; no scientific rerun and no recalculation of individual MSE from trajectories','receipt_sha256':hashes,'gate_mismatches':[],'source_budgets_obs_meas_compute':BUDGETS,'tolerance':TOL,'statistical_scope':'Source-cluster comparisons are descriptive and exploratory; sign flips assume independent clusters and exchangeable signs; no multiplicity adjustment; nonsignificance does not establish equivalence'}
 all_es=[];groups={}
 d2cs=data['D2']['unexposed_evaluation_chains'];d3cs=data['D3']['all_attempted_continuous_chains']
 for label,cs,srcname in [('D2',d2cs,'ordinary_source_episodes'),('D3',d3cs,'ordinary_markov_source_episodes')]:
  g={'anchors':[c['calibration_anchor_episode']for c in cs],'sources':[e for c in cs for e in c[srcname]]}
  if label=='D3':
   for arm in ('ordinary_failure_retry_equal_api','subquestion_driven_repair_equal_api'):g['QA_'+arm]=[c['question_A_parent_repair_equal_api'][arm]for c in cs]
  for arm in cs[0]['arms_target_results']:g[arm]=[e for c in cs for e in c['arms_target_results'][arm]]
  groups[label]=g;all_label=[]
  for kind,es in g.items():
   if label=='D2':budget='D2_source'if kind in ('anchors','sources')else'D2_target'
   else:budget='D3_markov'if kind in ('anchors','sources')else('D3_conflict'if kind.startswith('QA_')else'D3_target')
   for e in es:gates(e,budget,out['gate_mismatches'])
   all_label+=es
  all_es+=all_label
  out[label]={'records':len(all_label),'groups':{k:agg(es)for k,es in g.items()},'unique_arm_and_episode_tag':len({(e['arm'],e['episode_tag'])for e in all_label}),'unique_episode_uid':len({e['episode_uid']for e in all_label}),'unique_episode_tag':len({e['episode_tag']for e in all_label}),'strict_passes_all':sum(e['strict_pass']for e in all_label),'false_confirmations_all':sum(e['false_internal_confirmation']for e in all_label),'target_by_role':{},'paired':{}}
  for arm in cs[0]['arms_target_results']:
   out[label]['target_by_role'][arm]={role:agg([e for e in g[arm]if e['target_regime_role']==role])for role in sorted({e['target_regime_role']for e in g[arm]})}
  pairs=[(K,R),(K,COLD),(R,COLD)]if label=='D2'else[(K,R),(K,BR),(K,OBJ),(K,COLD)]
  for f,r in pairs:out[label]['paired'][f+'__vs__'+r]=paired(cs,f,r)
 out['episode_count_audit']={'total_arm_episode_records':len(all_es),'unique_arm_and_episode_tag':len({(e['arm'],e['episode_tag'])for e in all_es}),'unique_episode_uid':len({e['episode_uid']for e in all_es}),'unique_episode_tag':len({e['episode_tag']for e in all_es}),'D2_D3_episode_uid_overlap':sorted({e['episode_uid']for es in groups['D2'].values()for e in es}&{e['episode_uid']for es in groups['D3'].values()for e in es}),'false_confirmations_all':sum(e['false_internal_confirmation']for e in all_es),'strict_passes_all':sum(e['strict_pass']for e in all_es),'internally_confirmed_all':sum(e['internally_confirmed']for e in all_es),'arithmetic_origin_of_214':'112 + 120 - (D3 6 anchors + 12 Markov sources) = 214; these 18 are distinct from D2, so exclusion is not source deduplication','independence_caution':'232 is number of arm-episode judgments, not independent scientific trials; source clusters govern paired inference'}
 prefixes=[];accounting=[]
 for c in d2cs:
  led=c['source_formation_measured_ledger'];maint=led['source_reconciliation_maintenance_compute_C_K_maint'];sums={a:0 for a in (R,K)};vals=[]
  for ix in range(3):
   for a in (R,K):sums[a]+=c['arms_target_results'][a][ix]['measured_costs']['compute_cost']
   delta=sums[R]-sums[K]-maint;vals.append(delta)
   if delta!=c['prefix_full_cost_comparison_R_minus_K'][f'k_{ix+1}']['delta_C_c_k_R_minus_K']:raise ValueError('Prefix arithmetic mismatch')
  prefixes.append({'chain':c['chain_id'],'maintenance_compute':maint,'prefix_R_minus_K':vals})
  anchors=c['calibration_anchor_episode']['measured_costs']['compute_cost'];sources=sum(e['measured_costs']['compute_cost']for e in c['ordinary_source_episodes'])
  accounting.append({'chain':c['chain_id'],'anchor_compute':anchors,'source_compute_including_maintenance':sources,'source_maintenance_compute':maint,'source_compute_excluding_maintenance':sources-maint,'all_K_actual_source_plus_3target_compute':anchors+sources+sums[K]})
 prefixagg={str(k+1):{'per_chain':[p['prefix_R_minus_K'][k]for p in prefixes],'mean':mean(p['prefix_R_minus_K'][k]for p in prefixes),'median':median(p['prefix_R_minus_K'][k]for p in prefixes),'K_cheaper_chains':sum(p['prefix_R_minus_K'][k]>0 for p in prefixes),'sign_flip_p_exploratory':sf([p['prefix_R_minus_K'][k]for p in prefixes])}for k in range(3)}
 out['D2']['source_and_prefix_cost']={'per_chain':accounting,'mean_source_compute_including_maintenance':mean(x['source_compute_including_maintenance']for x in accounting),'mean_source_compute_excluding_maintenance':mean(x['source_compute_excluding_maintenance']for x in accounting),'mean_anchor_compute':mean(x['anchor_compute']for x in accounting),'mean_maintenance_compute':mean(x['source_maintenance_compute']for x in accounting),'prefixes':prefixes,'prefix_aggregates':prefixagg,'scope':'Recomputes reported maintenance allocation conditional on shared raw source history. Full prefix here is charged compute, not all acquisition/time resources and not an actually run ordinary source-history counterfactual. No double charging of R reconstruction already included in target costs.'}
 # D3 Question A paired comparison and full source costs, not spliced into a fictitious end-to-end R run.
 qa=[];d3source=[];obj=[];narrative=[];consumer=[];scopefields=[]
 for c in d3cs:
  a=c['question_A_parent_repair_equal_api'];rr=a['ordinary_failure_retry_equal_api'];kk=a['subquestion_driven_repair_equal_api']
  qa.append({'chain':c['chain_id'],'success_delta_subquestion_minus_ordinary':int(kk['strict_pass'])-int(rr['strict_pass']),'compute_extra_subquestion_minus_ordinary':kk['measured_costs']['compute_cost']-rr['measured_costs']['compute_cost'],'observation_extra_subquestion_minus_ordinary':kk['measured_costs']['observation_cost']-rr['measured_costs']['observation_cost'],'ordinary':brief(rr),'subquestion':brief(kk)})
  anc=c['calibration_anchor_episode']['measured_costs']['compute_cost'];mark=sum(e['measured_costs']['compute_cost']for e in c['ordinary_markov_source_episodes'])
  d3source.append({'chain':c['chain_id'],'anchor_compute':anc,'ordinary_markov_compute':mark,'subquestion_conflict_compute':kk['measured_costs']['compute_cost'],'ordinary_conflict_compute':rr['measured_costs']['compute_cost'],'actual_K_source_compute_with_anchor':anc+mark+kk['measured_costs']['compute_cost'],'actual_K_source_compute_without_anchor':mark+kk['measured_costs']['compute_cost'],'ordinary_source_alternative_compute_with_anchor':anc+mark+rr['measured_costs']['compute_cost'],'actual_K_source_plus_K_targets_compute':anc+mark+kk['measured_costs']['compute_cost']+sum(e['measured_costs']['compute_cost']for e in c['arms_target_results'][K])})
  ob=c['continuous_repaired_object_ledger'];cid=ob['repaired_concept_id'];obj.append({'chain':c['chain_id'],**{k:ob[k]for k in ('produced_qualified_repaired_object','repaired_concept_id','repaired_semantic_version','contains_accumulate_operator','repaired_prior_present_in_library','post_subresearch_re_reconciliation_event')}})
  if not ob['produced_qualified_repaired_object']and 'passed an independent' in c['seven_sentence_factual_ledger']['5_fresh_confirmation_and_historical_branch_checks']:
   narrative.append({'chain':c['chain_id'],'incorrect_sentence_5':c['seven_sentence_factual_ledger']['5_fresh_confirmation_and_historical_branch_checks'],'actual_status':kk['status'],'actual_external_mse':kk['external_fresh48_mse']})
  for arm,es in c['arms_target_results'].items():
   for e in es:
    ev=source_typed_events(e,cid);gen=[x for x in ev if x.get('event_stage')=='candidate_generation'];ad=[x for x in ev if x.get('adopted_confirmed')]
    consumer.append({'chain':c['chain_id'],'arm':arm,'role':e['target_regime_role'],'repaired_cid':cid,'reported_invoked_by_string':e['invoked_repaired_object'],'typed_events_from_repaired_source':len(ev),'typed_candidate_generation_from_repaired_source':bool(gen),'typed_adopted_confirmed_from_repaired_source':bool(ad),'typed_event_versions':[x.get('semantic_version')for x in gen],'strict_pass':e['strict_pass']})
 for label,cs in [('D2',d2cs),('D3',d3cs)]:
  for c in cs:
   for arm,es in c['arms_target_results'].items():
    for e in es:
     tel=e.get('branch_dispatch_telemetry')or{}
     if tel.get('scope_not_established')or tel.get('empirically_excluded_on_context'):
      scopefields.append({'study':label,'chain':c['chain_id'],'arm':arm,'episode_tag':e['episode_tag'],'reported_scope_unknown_events':e.get('scope_unknown_events'),'first_dispatch_scope_not_established':tel.get('scope_not_established'),'reported_excluded_ids':e.get('excluded_out_of_scope_branch_ids'),'first_dispatch_empirically_excluded':tel.get('empirically_excluded_on_context')})
 out['D3']['question_A_paired']={'per_chain':qa,'success_signflip_p':sf([x['success_delta_subquestion_minus_ordinary']for x in qa]),'mean_compute_extra':mean(x['compute_extra_subquestion_minus_ordinary']for x in qa),'median_compute_extra':median(x['compute_extra_subquestion_minus_ordinary']for x in qa),'compute_signflip_p_exploratory':sf([x['compute_extra_subquestion_minus_ordinary']for x in qa]),'mean_observation_extra':mean(x['observation_extra_subquestion_minus_ordinary']for x in qa),'comparison_scope':'Both can reach existing latent API and share total budget; subquestion policy changes max_rounds, expanded revisions, latent and threshold flags, so this compares policy packages rather than a single boolean.'}
 out['D3']['source_costs']={'per_chain':d3source,'means':{key:mean(x[key]for x in d3source)for key in d3source[0]if key!='chain'},'maintenance_decomposition_available':False,'scope':'Saved receipts permit whole source episode costs but do not export a complete maintenance/re-reconciliation cost decomposition for D3. Question B R uses the subquestion-derived source library, not the library produced by Question A ordinary retry; do not splice A-ordinary and B-R into an allegedly executed end-to-end run.'}
 out['D3']['repaired_objects']=obj;out['D3']['incorrect_success_narratives']=narrative;out['D3']['typed_consumption_audit']=consumer
 out['scope_metric_field_mismatches']=scopefields
 return out

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('recomputed_results.json'));args=p.parse_args()
 out=run(args.bundle.resolve());args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'output':str(args.output.resolve()),'counts':out['episode_count_audit'],'gate_mismatches':out['gate_mismatches'],'D2_prefixes':out['D2']['source_and_prefix_cost']['prefix_aggregates'],'D3_question_A_extra_compute':out['D3']['question_A_paired']['mean_compute_extra'],'D3_source_cost_means':out['D3']['source_costs']['means'],'incorrect_narratives':len(out['D3']['incorrect_success_narratives'])},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
