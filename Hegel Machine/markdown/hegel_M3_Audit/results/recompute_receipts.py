"""Independent arithmetic/provenance audit; no scientific calls or repository edits."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import mean

REPO = Path(os.environ.get('HEGEL_REPO', '/workspace/scratch/3f14ad2da0ed/repo_m3_612d8eea5/Hegel Machine'))
OUT = Path(os.environ.get('AUDIT_OUTPUT_DIR', Path(__file__).resolve().parent))
OUT.mkdir(parents=True, exist_ok=True)

def jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def close(a,b,atol=0.0002):
    return abs(float(a)-float(b)) <= atol

result = {'fixed_commit':'612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8','studies':{}}
for label, folder in [('m2_5','v16_9_m2_audited_open_workspace_20261119'),('m3','v17_0_m3_hipporag_wormhole_scale_20261119')]:
    d = REPO/'docs/evidence'/folder
    manifest=json.loads((d/'experiment_manifest.json').read_text())
    paired=json.loads((d/'paired_results.json').read_text())
    source=jsonl(d/'source_episodes.jsonl'); targets=jsonl(d/'target_episodes.jsonl'); ledger=jsonl(d/'cost_ledger.jsonl')
    mismatches=[]; lookup={r['record_id']:r for r in ledger}
    for r in source+targets:
        if r['record_id'] not in lookup:
            mismatches.append({'kind':'ledger_missing','id':r['record_id']});continue
        l=lookup[r['record_id']]; c=r['costs']
        for k in ('observation_cost','measurement_cost','compute_cost','composite_cost'):
            if k in c and not close(c[k],l.get(k,0)):
                mismatches.append({'kind':'ledger_mismatch','id':r['record_id'],'field':k,'episode':c[k],'ledger':l.get(k)})
        if not close(c['composite_cost'],c['measurement_cost']+.05*c['compute_cost']):
            mismatches.append({'kind':'composite','id':r['record_id']})
        for k,pf in [('observation_cost','obs'),('measurement_cost','meas'),('compute_cost','compute')]:
            if c.get('phase_breakdown') and not close(c[k],sum(x[pf] for x in c['phase_breakdown'].values())):
                mismatches.append({'kind':'phase_breakdown','id':r['record_id'],'field':k})
    for l in ledger:
        if 'measurement_cost' in l and not close(l['composite_cost'],l['measurement_cost']+.05*l['compute_cost']):
            mismatches.append({'kind':'ledger_composite','id':l['record_id']})
    tg=defaultdict(list)
    for r in targets:tg[(r['cluster_id'],r['arm'])].append(r)
    for cs in paired['heldout_cluster_summaries']:
        cid=cs['cluster_id']
        for arm,vals in cs['arms'].items():
            rs=tg[(cid,arm)]
            fields={'strict_pass_count':sum(r['strict_pass'] for r in rs),
                    'total_targets':len(rs),'false_confirmations':sum(r['false_confirmation'] for r in rs),
                    'target_observation_cost':sum(r['costs']['observation_cost'] for r in rs),
                    'target_measurement_cost':sum(r['costs']['measurement_cost'] for r in rs),
                    'target_logical_compute_cost':sum(r['costs']['compute_cost'] for r in rs),
                    'target_composite_cost':sum(r['costs']['composite_cost'] for r in rs)}
            for k,v in fields.items():
                if not close(v,vals[k]):mismatches.append({'kind':'cluster_sum','cluster':cid,'arm':arm,'field':k,'actual':v,'declared':vals[k]})
    cmp_checks={}
    for name,cmp in paired['key_paired_comparisons'].items():
        left,right=cmp['comparison'].split('_minus_',1);keys=cmp['metric'].split('.')
        diffs=[]
        for cs in paired['heldout_cluster_summaries']:
            l,r=cs['arms'][left],cs['arms'][right]
            for k in keys:l,r=l[k],r[k]
            diffs.append(float(l)-float(r))
        ok=close(mean(diffs),cmp['mean_diff']) and all(close(a,b) for a,b in zip(diffs,cmp['per_cluster_diffs']))
        cmp_checks[name]={'mean_recomputed':round(mean(diffs),6),'ok':ok}
        if not ok:mismatches.append({'kind':'paired_comparison','name':name})
    source_counts=Counter((r['split'],r['arm'],r['phase']) for r in source)
    study={'source_records':len(source),'target_records':len(targets),'ledger_records':len(ledger),
           'strict_target_passes':sum(r['strict_pass'] for r in targets),
           'false_confirmations':sum(r['false_confirmation'] for r in targets),
           'hash_checks':{p:hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h for p,h in manifest['module_sha256'].items()},
           'source_counts':[{'split':s,'arm':a,'phase':p,'count':n} for (s,a,p),n in source_counts.items()],
           'paired_checks':cmp_checks,'arithmetic_mismatches':mismatches,
           'declared_dev_summary':paired['dev_calibration_summary']}
    if label=='m2_5':
        open_runs=json.loads((d/'open_author_revision_results.json').read_text())
        open_targets=[x for run in open_runs for arms in run['downstream_targets'].values() for x in arms.values()]
        study['open_revision_target_count']=len(open_targets)
        study['open_revision_pass_count']=sum(x['strict_pass'] for x in open_targets)
        study['open_revision_model_calls']=sum(x['costs']['model_calls'] for x in open_targets)
        study['open_revision_old_source_rmse_range']=[min(c['backward_source_rmse'] for run in open_runs for c in run['chains'].values()),max(c['backward_source_rmse'] for run in open_runs for c in run['chains'].values())]
    else:
        diagnostics=json.loads((d/'index_and_retrieval_diagnostics.json').read_text())
        recall={}
        for arm in diagnostics['arms']:
            rs=[r['retrieval_diagnostic'] for r in targets if r['arm']==arm and r.get('retrieval_diagnostic')]
            post=[r for r in rs if not r.get('blocked_by_scale_map_pre_pilot')]
            hit1=sum(bool(r.get('authorized_top_schemas')) and r['authorized_top_schemas'][0]==r.get('truth_schema_audit_only') for r in post)
            hit2=sum(r.get('truth_schema_audit_only') in r.get('authorized_top_schemas',[])[:2] for r in post)
            recall[arm]={'all_queries':len(rs),'post_pilot_queries':len(post),'hit1':hit1,'hit2':hit2,
                         'recall1':hit1/len(post),'recall2':hit2/len(post),
                         'wrong_trials':sum(r.get('wrong_candidate_trials',0) for r in rs),
                         'filtered':sum(r.get('contract_filtered_count',0) for r in rs),
                         'definition':'Exact equality of declared source schema tag; conditional on non-pre-pilot-ScaleMap-blocked queries.'}
        study['recall_recomputed']=recall
        oos=json.loads((d/'out_of_scope_honesty.json').read_text())
        study['oos']={'count':len(oos),'statuses':dict(Counter(r['submission_status'] for r in oos)),
                      'false_confirmations':sum(r['false_confirmation'] for r in oos),
                      'diagnostic_composite_cost_not_in_main_ledger':sum(r['costs']['composite_cost'] for r in oos)}
        cs=paired['heldout_cluster_summaries'];h='Hegel_M3_HippoPPR';b='Flat_BM25_EqualMetadata';ordinary='O_strong_AllValidatedPriors'
        categories=sorted({r['target_category'] for r in targets})
        study['category_diffs_vs_flat']={cat:mean(sum(r['costs']['composite_cost'] for r in tg[(x['cluster_id'],h)] if r['target_category']==cat)-sum(r['costs']['composite_cost'] for r in tg[(x['cluster_id'],b)] if r['target_category']==cat) for x in cs) for cat in categories}
        ordered_names=[r['target_name'] for r in tg[(cs[0]['cluster_id'],h)]]
        study['per_target_diffs_vs_flat']={name:mean(next(r['costs']['composite_cost'] for r in tg[(x['cluster_id'],h)] if r['target_name']==name)-next(r['costs']['composite_cost'] for r in tg[(x['cluster_id'],b)] if r['target_name']==name) for x in cs) for name in ordered_names}
        order_sensitivity={}
        for ref in (b,ordinary):
            source_delta=mean(x['arms'][h]['source_and_index_composite_cost']-x['arms'][ref]['source_and_index_composite_cost'] for x in cs)
            delta=[mean(next(r['costs']['composite_cost'] for r in tg[(x['cluster_id'],h)] if r['target_name']==name)-next(r['costs']['composite_cost'] for r in tg[(x['cluster_id'],ref)] if r['target_name']==name) for x in cs) for name in ordered_names]
            fixed_curve=[source_delta+sum(delta[:k]) for k in range(1,25)]
            # Reconstruction is charged once to first original target. Move it to
            # upfront cost for algebraic reordering so it remains paid at every prefix.
            reconstruction=0.0
            if ref==ordinary:
                reconstruction=mean(tg[(x['cluster_id'],ref)][0]['costs'].get('phase_breakdown',{}).get('on_demand_reconstruction',{}).get('compute',0)*.05 for x in cs)
                delta[0]+=reconstruction;source_delta-=reconstruction
            rng=random.Random(612);firsts=[];sustained=[]
            for _ in range(20000):
                ds=list(delta);rng.shuffle(ds)
                curve=[source_delta+sum(ds[:k]) for k in range(1,25)]
                firsts.append(next((i+1 for i,v in enumerate(curve) if v<0),25))
                sustained.append(next((i+1 for i in range(24) if all(v<0 for v in curve[i:])),25))
            def summarize(xs):
                xs=sorted(xs);return {'min':xs[0],'p05':xs[int(.05*len(xs))],'median':xs[len(xs)//2],'p95':xs[int(.95*len(xs))],'max':xs[-1]}
            order_sensitivity[ref]={'declared_fixed_order_first_crossing':next((i+1 for i,v in enumerate(fixed_curve) if v<0),None),
                                    'fixed_order_stays_negative_after_first_crossing':all(v<0 for v in fixed_curve[next((i for i,v in enumerate(fixed_curve) if v<0),24):]),
                                    'fixed_order_curve':fixed_curve,'fixed_cost_permutation_first_crossing':summarize(firsts),
                                    'fixed_cost_permutation_sustained_crossing':summarize(sustained),
                                    'permutations':20000,'ordinary_once_reconstruction_moved_upfront':reconstruction,
                                    'limitation':'Algebraic ordering of preserved per-task costs, not a new online rerun; order-dependent gate behavior is not re-executed.'}
        study['prefix_order_sensitivity']=order_sensitivity
        per_cluster_h=sum(x['arms'][h]['source_and_index_composite_cost'] for x in cs)/4
        dev=paired['dev_calibration_summary']['dev_total_composite_cost']
        study['dev_amortization']={}
        for ref in (ordinary,b):
            saved=-mean(x['arms'][h]['end_to_end_total_composite_cost']-x['arms'][ref]['end_to_end_total_composite_cost'] for x in cs)
            study['dev_amortization'][ref]={'saved_per_current_24_target_cluster_given_gate':saved,
               'dev_charged_only_to_hegel':dev,'net_difference_dev_split_across_four_clusters':-saved+dev/4,
               'identical_cluster_count_to_amortize_dev_if_all_other_costs_repeat':math.ceil(dev/saved)}
    result['studies'][label]=study
(OUT/'receipt_recomputation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:{'source':v['source_records'],'targets':v['target_records'],'ledger':v['ledger_records'],'pass':v['strict_target_passes'],'false':v['false_confirmations'],'hashes_ok':all(v['hash_checks'].values()),'arithmetic_mismatches':v['arithmetic_mismatches']} for k,v in result['studies'].items()},ensure_ascii=False,indent=2))
