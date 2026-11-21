"""Cheap targeted audit controls. No source edits and no full benchmark rerun."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',default=os.environ.get('HEGEL_AUDIT_REPO_ROOT','.'),help='Assumption-Agent checkout root, or its Hegel Machine subdirectory')
parser.add_argument('--evidence-dir',default=os.environ.get('HEGEL_AUDIT_EVIDENCE_DIR'),help='Committed evidence directory; defaults to repository docs/evidence/v18_0_m5_final_temporal_multiscale_and_base_subtree_20261120')
parser.add_argument('--output-dir',default=str(Path(__file__).resolve().parent),help='Audit output directory')
args=parser.parse_args()
PROJECT=Path(args.repo_root).expanduser().resolve()
if (PROJECT/'Hegel Machine').is_dir():
    PROJECT=PROJECT/'Hegel Machine'
ROOT=Path(args.output_dir).expanduser().resolve()
ROOT.mkdir(parents=True,exist_ok=True)
EV=Path(args.evidence_dir).expanduser().resolve() if args.evidence_dir else PROJECT/'docs/evidence/v18_0_m5_final_temporal_multiscale_and_base_subtree_20261120'
sys.path.insert(0,str(PROJECT/'src'))
from hegel_machine.cognition import m5_temporal_multiscale_and_base_subtree as m
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex, FlatEqualMetadataIndex

f, pb = m._build_seed_forest_and_priors()
hi, fi = HippoRAGWormholeIndex(f), FlatEqualMetadataIndex(f)
full = 'Hegel_M5_Full_TemporalSubtree_Forest'
original_fit = m._fit_candidate_model_class

def run(ds, arm=full, mock_fit=False):
    calls=[]
    def observed_fit(cls, train, **kwargs):
        r = dict(model_class=cls,nfev=1,compute_cost=1.0,predict_fn=lambda row:0.0) if mock_fit else original_fit(cls,train,**kwargs)
        calls.append(dict(model_class=cls,train_rows=len(train),warm_prior_present=kwargs['warm_prior'] is not None,nfev=r['nfev']))
        return r
    m._fit_candidate_model_class = observed_fit
    try:
        result=m._run_genuine_solver_pipeline(ds,arm,forest=f,priors_bundle=pb,hippo_idx=hi,flat_idx=fi,rng=random.Random(817))
    finally:
        m._fit_candidate_model_class=original_fit
    return dict(receipt=result,fit_calls=calls,actual_fit_nfev=sum(c['nfev'] for c in calls))

ds=m._build_disjoint_dataset('Audit_C_same_numeric_rows','C_BaseSubtree','C_two_layer_conjunctive_subtree',False,
    ['sig:base_field_C1','sig:sink_gradient_C2','sig:monotone_concave'],False,
    dict(vmax=6.4,rho=.70,local_branch_coeff=.38,tau_scale=1.),random.Random(3181))
numeric_digest=hashlib.sha256(json.dumps([ds.pilot_rows,ds.extra_dynamic_rows,ds.probe_pool,ds.confirm_rows,ds.heldout_rows],sort_keys=True).encode()).hexdigest()
signature_control=dict(numeric_data_sha256=numeric_digest,
    with_generator_signatures=run(ds),without_generator_signatures=run(replace(ds,probe_signatures=[])))

# Same numeric training set and model, instrument individual optimizer starts.
original_ls=m.least_squares
warm_cold={}
for tag, prior in [('source_soft_prior',pb['siphon_warm_prior']),('wide_cold',None)]:
    starts=[]
    def observed_ls(*args,**kwargs):
        r=original_ls(*args,**kwargs)
        starts.append(dict(nfev=int(r.nfev),final_rmse=float((sum(float(v)**2 for v in r.fun)/len(r.fun))**.5)))
        return r
    m.least_squares=observed_ls
    try:
        fit=m._fit_candidate_model_class('M6_SiphonSubtree_SqrtBranch',ds.pilot_rows,tau_scale=1.0,warm_prior=prior,tolerance=.045,rng=random.Random(817))
    finally:
        m.least_squares=original_ls
    warm_cold[tag]=dict(total_nfev=fit['nfev'],individual_starts=starts,
        heldout_rmse=m._rmse([fit['predict_fn'](r) for r in ds.heldout_rows],[r['y'] for r in ds.heldout_rows]))

static_ds=m._build_disjoint_dataset('Audit_static','A_TemporalMemory','A_snapshot_only',False,
    ['sig:static_equilibrium_consistent','sig:zero_history_gap'],False,dict(c0=2.3,c1=1.2),random.Random(101))
blind_nfev=run(static_ds,'Ab_Blind_Parameter_Copy_Transfer')
boundary=replace(static_ds,
    pilot_rows=[{**r,'y':0.0} for r in static_ds.pilot_rows],
    confirm_rows=[{**r,'y':0.0} for r in static_ds.confirm_rows],
    heldout_rows=[{**r,'y':0.055} for r in static_ds.heldout_rows])
boundary_controls={a:run(boundary,a,mock_fit=True) for a in [full,'Strong_Dynamic_StateSpace_AllHistory','Ab_Memoryless_Static_Baseline']}

hash_code="import json; print(json.dumps({'seed': hash((101,'Hegel_M5_Full_TemporalSubtree_Forest')) & 0xFFFFFFFF, 'python_hash_seed_is_set': __import__('os').environ.get('PYTHONHASHSEED')}))"
seeds=[json.loads(subprocess.check_output([sys.executable,'-c',hash_code],text=True)) for _ in range(3)]

records=[json.loads(x) for x in (EV/'part2_track1_anonymous_receipts.jsonl').read_text().splitlines()]
committed_boundary=[r for r in records if r.get('heldout_rmse') is not None and .045 < r['heldout_rmse'] <= .0675]
out=dict(signature_control=signature_control,blind_reported_nfev_control=blind_nfev,
    same_model_same_training_rows_warm_cold_start_control=warm_cold,
    false_confirmation_boundary_controls=boundary_controls,committed_receipts_in_threshold_disagreement_band=committed_boundary,
    same_declared_seed_different_fresh_process_seeds=seeds)
(ROOT/'evidence_targeted_controls.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2))
