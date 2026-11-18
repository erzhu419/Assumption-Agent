"""V14 bounded AST controls with explicit stubs; no full physical-agent replay."""
from __future__ import annotations
import argparse,ast,runpy,hashlib,json,math
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle', required=True, type=Path, help='Directory containing the extracted V14 review bundle.')
parser.add_argument('--output', type=Path, default=Path('scope_replay_probe_results.json'), help='Output JSON path; defaults to the current directory.')
args = parser.parse_args()
ROOT = args.bundle.resolve()
SRC=ROOT/'bundled_sources/src/hegel_machine/cognition'
B=SRC/'branch_reconciliation.py'; I=SRC/'premise_commitment_index.py'
v=runpy.run_path(str(ROOT/'05_STANDALONE_VERIFY_V14.py'))
P=v['_MinimalProgram']; O=v['_MinimalObservation']
def req(self):
    return ({self.name} if self.op in {'feature','arg'} else set()).union(*(req(ch) for ch in self.children))
P.required_features=req
ns={'Program':P,'Observation':O,'deepcopy':deepcopy,'isfinite':math.isfinite,'hashlib':hashlib,'json':json,'Any':Any}
def load(path,names,constants=()):
    tr=ast.parse(path.read_text());nodes=[]
    for n in tr.body:
        if isinstance(n,ast.FunctionDef) and n.name in names:nodes.append(n)
        targets=[]
        if isinstance(n,ast.Assign):targets=[x.id for x in n.targets if isinstance(x,ast.Name)]
        elif isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name):targets=[n.target.id]
        if set(targets)&set(constants):nodes.append(n)
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod),str(path),'exec'),ns)
load(B,{'_deserialize_obs_rows','_serialize_obs_rows','evaluate_scope_guard','validate_observable_scope_guard','extract_guard_ast_dependencies','shrink_claim_to_observable_scope','replay_evidence_batches_onto_branches'},['FORBIDDEN_SCOPE_TOKENS','PUBLIC_OBSERVABLE_GUARD_MANIFEST','OBSERVABLE_REGIME_FEATURES'])
load(I,{'compute_evidence_archive_fingerprint'})
# Identical observable summaries cannot separate two incompatible bound instances;
# a refittable-template check is available but never invoked on this fallback branch.
reg={'obs_signed_direction_flux':.5,'obs_unconditional_block_rate':.2,'_meta':{'protocol_id':'paid_episode_transitions','pilot_calibration_available':True}}
ns['compute_observable_regime_features']=lambda *a,**k:deepcopy(reg)
ns['instantiate_aligned_claim_program']=lambda b,sc:{'aligned':True,'program':P('const',value=b['value'])}
ns['loss']=lambda p,rows,*a:sum((p.predict(r)-r.target)**2 for r in rows)/len(rows)
tmpl_calls=[]
def template_supports(*a,**k):
    tmpl_calls.append(1)
    return {'evaluated':True,'empirically_refuted':False,'verdict':'supported_on_batch'}
ns['_cross_evaluate_refittable_template_on_neighbor']=template_supports
ns['_refine_coexisting_branch_scopes_and_routing']=lambda bs,**k:bs
bs=[{'claim_id':'branch:old','source_concept_id':'old','value':1.,'status':'universal','scope_claim_level':'bound_instance','independent_empirical_support':True,'pilot_regime_features':[deepcopy(reg)]}, {'claim_id':'branch:new','source_concept_id':'new','value':0.,'status':'universal','scope_claim_level':'bound_instance','independent_empirical_support':True,'pilot_regime_features':[deepcopy(reg)]}]
ev=[{'evidence_id':'ev:new','schema':{'defined':True},'observations_serialized':[{'features':{},'target':0.}], 'transitions':[{'dummy':True}], 'protocol_id':'paid_episode_transitions','radius_value':.75,'commitment':{'radius_parameters':[{'direction':'right'}]}}]
shr=ns['shrink_claim_to_observable_scope'](bs[0],[reg],[reg],claim_level='bound_instance')
rb,stats=ns['replay_evidence_batches_onto_branches'](bs,ev)
# Parameter bindings actually read by replay are omitted from evidence hash.
fp=ns['compute_evidence_archive_fingerprint'](ev)
ev_radius=deepcopy(ev);ev_radius[0]['radius_value']=.9
fp_radius=ns['compute_evidence_archive_fingerprint'](ev_radius)
ev_comm=deepcopy(ev);ev_comm[0]['commitment']['radius_parameters'][0]['direction']='left'
fp_comm=ns['compute_evidence_archive_fingerprint'](ev_comm)
ev_data=deepcopy(ev);ev_data[0]['observations_serialized'][0]['target']=.1
fp_data=ns['compute_evidence_archive_fingerprint'](ev_data)
out={'mode':'actual_AST_function_bodies_with_explicit_alignment_summary_and_refinement_stubs','full_physical_rerun':False,'source_sha256':hashlib.sha256(B.read_bytes()).hexdigest(),'scope_unidentified_replay_fallback':{'direct_shrink_status':shr['status'],'replayed_old_status':rb[0]['status'],'template_check_calls':len(tmpl_calls),'old_source_support_preserved_but_globally_refuted':rb[0]['independent_empirical_support'],'stats':stats,'limitation':'The summary is forced identical and post-replay all-pairs refinement is an identity stub to isolate the production fallback; no claim that this explains every recorded target.'},'evidence_hash_dependencies':{'base':fp,'radius_changed':fp_radius,'commitment_changed':fp_comm,'observation_changed':fp_data,'radius_edit_invalidates':fp!=fp_radius,'commitment_edit_invalidates':fp!=fp_comm,'observation_edit_invalidates':fp!=fp_data}}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
