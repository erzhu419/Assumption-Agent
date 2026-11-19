"""Focused post-hoc retrieval controls on the committed eval04 forest (no new model calls).

All variants use the exact committed node contents. This is diagnostic ranking,
not a new frozen benchmark or runtime speed comparison.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

REPO = Path(os.environ.get('HEGEL_REPO', '/workspace/scratch/3f14ad2da0ed/repo_m3_612d8eea5/Hegel Machine'))
OUT = Path(os.environ.get('AUDIT_OUTPUT_DIR', str(Path(__file__).resolve().parent)))
sys.path.insert(0, str(REPO / 'src'))
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex, FlatEqualMetadataIndex
from hegel_machine.cognition.m3_scale_universe import make_m3_large_scale_suite, build_m3_pilot_points, extract_m3_observable_signatures
from hegel_machine.cognition.open_workspace_wormhole import _estimate_observation_lens, _active_and_fixed_features
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest, authorize_abstraction_use, compose_wormhole_chain

EVID = REPO / 'docs/evidence/v17_0_m3_hipporag_wormhole_scale_20261119'

def load_forest():
    d = json.loads((EVID / 'wormhole_forest_graph.json').read_text())
    f = PhilosophicalForest()
    for k in ('nodes', 'local_implication_edges', 'horizontal_wormhole_edges'):
        setattr(f, k, deepcopy(d[k]))
    return f

def schema(c):
    return c['target_node']['claim_contract']['conclusion_or_action']['schema_tag']

class FlatComponentSum(FlatEqualMetadataIndex):
    """Use the same BM25 per-node scores but sum within the same live-WH class as PPR.

    Calling the upstream scorer one record at a time is only a way to reuse the
    exact formula in this small ranking control; wall time/cost is not compared.
    """
    def __init__(self, forest, components):
        super().__init__(forest, mode='bm25')
        self.components = components

    def retrieve(self, observed_signatures, *, task_view, top_k=5):
        original_records = self.records
        scored = []
        for rec in original_records:
            self.records = [rec]
            scored.extend(super().retrieve(observed_signatures, task_view=task_view, top_k=1)['candidates'])
        self.records = original_records
        node_scores = {c['node_id']: c['score'] for c in scored}
        for c in scored:
            c['node_score'] = c['score']
            c['score'] = sum(node_scores.get(n, 0.0) for n in self.components[c['node_id']])
        scored.sort(key=lambda c: (-c['score'], -c['node_score'], c['node_id']))
        seen = set(); kept = []
        for c in scored:
            if schema(c) not in seen:
                seen.add(schema(c)); kept.append(c)
                if len(kept) == top_k: break
        return {'candidates':kept}

def renamed_forest(f):
    g = deepcopy(f)
    names=sorted(g.nodes)
    mapping={old:f'node_{i:03d}' for i,old in enumerate(reversed(names))}
    g.nodes={mapping[n]:{**v,'node_id':mapping[n]} for n,v in g.nodes.items()}
    for e in g.local_implication_edges:
        for key in ('stronger_node_id','weaker_node_id'):e[key]=mapping[e[key]]
    for e in g.horizontal_wormhole_edges.values():
        for key in ('node_a_id','node_b_id'):e[key]=mapping[e[key]]
    return g

def authorized(f, candidates, lab, info):
    kept=[]
    for c in candidates:
        st=schema(c); wh=c.get('wormhole_edge'); chain=c.get('wormhole_path_edges',[])
        if info.get('high_load_rollover_detected') and st=='rational_monod_1d':
            hop2=[e for e in f.horizontal_wormhole_edges.values() if e.get('role_and_observation_map',{}).get('bridge_from_schema')=='rational_monod_1d']
            if wh is not None and hop2:
                if not compose_wormhole_chain([wh,hop2[0]],target_tolerance=lab.task_view['tolerance'])['within_tolerance']:continue
        elif len(chain)>=2 and st==chain[-1].get('role_and_observation_map',{}).get('bridge_from_schema'):
            if not compose_wormhole_chain(chain,target_tolerance=lab.task_view['tolerance'])['within_tolerance']:continue
        auth=authorize_abstraction_use(requested_tier='bounded_trial',task_view=lab.task_view,target_features={**lab.domain_regime_attrs,**info},claim_contract=c['target_node']['claim_contract'],wormhole_edge=wh)
        if auth['granted']:kept.append(c)
    return kept

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    f=load_forest(); ppr=HippoRAGWormholeIndex(f)
    no_pool=HippoRAGWormholeIndex(f); no_pool.wh_component_of={n:[n] for n in no_pool.node_ids}
    no_h_walk=HippoRAGWormholeIndex(f,allow_horizontal_wormholes=False);no_h_walk.wh_component_of=deepcopy(ppr.wh_component_of)
    rf=renamed_forest(f)
    variants={
        'PPR_original':(f,ppr),
        'PPR_alpha0_same_pool':(f,HippoRAGWormholeIndex(f,damping_alpha=0.0)),
        'PPR_walk_without_pool':(f,no_pool),
        'PPR_vertical_walk_same_pool':(f,no_h_walk),
        'Flat_BM25_original':(f,FlatEqualMetadataIndex(f)),
        'Flat_BM25_same_component_sum':(f,FlatComponentSum(f,ppr.wh_component_of)),
        'Flat_BM25_relabel_ids':(rf,FlatEqualMetadataIndex(rf)),
        'PPR_relabel_ids':(rf,HippoRAGWormholeIndex(rf)),
    }
    records=[json.loads(l) for l in (EVID/'target_episodes.jsonl').read_text().splitlines()]
    orig={r['target_name']:r for r in records if r['cluster_id']=='m3_eval_04' and r['arm']=='Hegel_M3_HippoPPR'}
    suite=make_m3_large_scale_suite(7027404)
    results=[]; mismatch=[]; direct_all=[]
    for cat,name,(fac,_) in suite['targets']:
        base=orig[name]
        diag=base['retrieval_diagnostic']
        if diag.get('blocked_by_scale_map_pre_pilot'):continue
        lab=fac();lens=_estimate_observation_lens(lab,OUT/'signature_workspaces'/name)
        lab.acquire(build_m3_pilot_points(lab))
        sigs,roles,info=extract_m3_observable_signatures(lab,lens)
        if sigs!=diag['observed_signatures']:mismatch.append({'target':name,'kind':'signature'})
        truth=diag['truth_schema_audit_only'];tv={**lab.task_view,'active_arity':len(_active_and_fixed_features(lab)[0])}
        row={'target':name,'category':cat,'truth_used_only_for_scoring':truth,'variants':{}}
        for key,(forest,index) in variants.items():
            r=index.retrieve(sigs,task_view=tv,top_k=6)
            ranks=[schema(c) for c in r['candidates']]
            allowed=authorized(forest,r['candidates'],lab,info)
            aschemas=[schema(c) for c in allowed]
            if key=='PPR_original':
                if ranks[:4]!=diag['raw_top_schemas']:mismatch.append({'target':name,'kind':'raw_rank','now':ranks[:4],'published':diag['raw_top_schemas']})
                if aschemas[:4]!=diag['authorized_top_schemas']:mismatch.append({'target':name,'kind':'authorized_rank','now':aschemas[:4],'published':diag['authorized_top_schemas']})
                direct={n for s in sigs for n in index.sig_to_nodes.get(s,[])}
                direct_all.extend([c['node_id'] in direct for c in r['candidates']])
            row['variants'][key]={'raw_top_schemas':ranks,'authorized_top_schemas':aschemas,'rank':aschemas.index(truth)+1 if truth in aschemas else None,'scores':[c['score'] for c in r['candidates']]}
        results.append(row)
    summary={}
    for key in variants:
        ranks=[r['variants'][key]['rank'] for r in results]
        summary[key]={'n':len(ranks),'hit_at_1':sum(x is not None and x<=1 for x in ranks),'hit_at_2':sum(x is not None and x<=2 for x in ranks),'hit_at_3':sum(x is not None and x<=3 for x in ranks),'hit_at_4':sum(x is not None and x<=4 for x in ranks)}
    out={'fixed_commit':'612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8','scope':'posthoc ranking control on committed eval04 graph; no heldout truth used by rankers; no new benchmark claim','summary':summary,'published_reconstruction_mismatches':mismatch,'path_direct_hit_checks':{'selected_candidates':len(direct_all),'direct_hit_candidates':sum(direct_all),'BFS_needed_candidates':len(direct_all)-sum(direct_all)},'forest':{'nodes':len(f.nodes),'trees':len({n['tree_id'] for n in f.nodes.values()}),'schemas':len({n['claim_contract']['conclusion_or_action']['schema_tag'] for n in f.nodes.values()}),'mismatch_digest_nodes':[n for n,v in f.nodes.items() if v['observation_map']['parameter_semantics']['ast_skeleton_digest'].startswith('mismatch:')]},'targets':results}
    (OUT/'retrieval_factorization_result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='targets'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
