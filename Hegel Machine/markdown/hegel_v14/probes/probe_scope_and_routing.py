#!/usr/bin/env python3
"""Bounded V13 source-excerpt probes; not a full repository or physical-agent replay.

Usage:
    python probe_scope_and_routing.py --bundle /path/to/unzipped_v13 --output results.json

Uses actual AST function bodies from the supplied bundle, the actual SearchBudget,
and explicit minimal dependency stubs. It makes no network or model calls and does
not edit the bundle. Results establish the specified control-flow behaviors only;
they do not show that every recorded scientific episode exercised those paths.
Non-finite numeric values are encoded as strings for strict JSON compatibility.
"""
from __future__ import annotations
import argparse
import ast, importlib.util, runpy, sys, types, math, json, hashlib
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle', type=Path, required=True, help='Unzipped V13 handoff bundle directory')
parser.add_argument('--output', type=Path, default=Path('scope_and_routing_probe_results.json'), help='Strict JSON result path')
args = parser.parse_args()
ROOT = args.bundle.expanduser().resolve()
if not ROOT.is_dir():
    parser.error(f'Bundle directory does not exist: {ROOT}')
SRC=ROOT/'bundled_sources/src/hegel_machine/cognition'
bpath=SRC/'branch_reconciliation.py'
ver=runpy.run_path(str(ROOT/'05_STANDALONE_VERIFY_V13.py'))
Program=ver['_MinimalProgram']; Observation=ver['_MinimalObservation']
pkg=types.ModuleType('_audit'); pkg.__path__=[];sys.modules['_audit']=pkg
spec=importlib.util.spec_from_file_location('_audit.models',SRC/'models.py')
models=importlib.util.module_from_spec(spec);sys.modules[spec.name]=models;spec.loader.exec_module(models)
world=types.ModuleType('_audit.world_structure_commitments')
world.is_commitment_empirically_consistent=lambda *a,**k:True
sys.modules[world.__name__]=world
ns={'__package__':'_audit','Program':Program,'Observation':Observation,'Any':Any,'deepcopy':deepcopy,'isfinite':math.isfinite,'math':math,'hashlib':hashlib,'json':json,'SearchBudget':models.SearchBudget,'TaskSpec':models.TaskSpec,'_as_schema':lambda x:x}
tree=ast.parse(bpath.read_text())
def load(names):
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert len(nodes)==len(names)
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod),str(bpath),'exec'),ns)
for n in tree.body:
    names=[]
    if isinstance(n,ast.Assign): names=[x.id for x in n.targets if isinstance(x,ast.Name)]
    elif isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name):names=[n.target.id]
    if {'FORBIDDEN_SCOPE_TOKENS','PUBLIC_OBSERVABLE_GUARD_MANIFEST','OBSERVABLE_REGIME_FEATURES'}.intersection(names):
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),n],type_ignores=[])),str(bpath),'exec'),ns)
load({'_deserialize_obs_rows','validate_parametric_family_calibration_support','_infer_slot_role_in_skeleton','instantiate_parametric_family_on_target','_cross_evaluate_refittable_family_on_neighbor','evaluate_scope_guard','validate_observable_scope_guard','extract_guard_ast_dependencies','shrink_claim_to_observable_scope','dispatch_branches_and_families_for_target','_cross_evaluate_fixed_instance_on_neighbor','_program_semantic_version','compute_observable_regime_features'})
# Exact production search and wrapper with a zero budget; dependency stubs just return a valid constant term.
ns['_bind_arg_slots']=lambda *a,**k:Program('const',value=1.)
ns['_has_unbound_theta_slot']=lambda *a:False
ns['instantiate_commitment_terms']=lambda *a,**k:[Program('const',value=1.)]
sc=SimpleNamespace(feature_names=(),objects=(),fields={})
fam={'commitment_id':'H_star','anti_unified_predicate_skeleton':Program('lt',children=(Program('arg',name='theta_0'),Program('arg',name='edge_source_quantity'))).to_dict(),'parameter_slots':['theta_0']}
nb={'schema':sc,'support_rows_serialized':[{'features':{},'target':0.}], 'commitment':{}}
budget=models.SearchBudget(0)
verdict=ns['_cross_evaluate_refittable_family_on_neighbor'](fam,nb,budget=budget)
out={'mode':'actual_source_excerpt_execution_with_explicit_dependency_stubs','full_repository_scientific_rerun':False,'source_sha256':hashlib.sha256(bpath.read_bytes()).hexdigest(),'empty_family_search':{'budget_limit':0,'budget_spent':budget.spent,'denied_requests':budget.denied_requests,'last_denied_request':budget.last_denied_request,'verdict':verdict,'last_diagnostic':fam.get('_last_instantiation_diagnostic')}}
# Actual dispatch: a router miss must stay in tail, but aggregate commitment is marked seen before append.
ns['compute_observable_regime_features']=lambda *a,**k:{'obs_signed_direction_flux':-1.}
ns['propagate_premise_retraction_invalidation']=lambda xs,**k:[{**x,'support_validity':'valid_grounded'} for x in xs]
guard=Program('lt',children=(Program('const',value=0.),Program('feature',name='obs_signed_direction_flux'))).to_dict()
b={'claim_id':'branch:A','commitment':{'commitment_id':'branch_comm:A','source_concept_ids':['A']},'source_concept_id':'A','status':'universal','routing_guard':guard}
lib=SimpleNamespace(_active_reconciled_branches=[b],_generalized_branch_commitments=[])
cons={'commitment_id':'world_commitment:0000','source_concept_ids':['A'],'unique_payload':'ordinary_aggregate_template'}
d=ns['dispatch_branches_and_families_for_target'](library=lib,consistent_comms=[cons],transitions=[],schema=None,audit=None,task_sig='T',comm_budget=models.SearchBudget(100))
out['router_deferred_aggregate_dropped']={'telemetry':d['telemetry'],'ordered_consistent_comms':d['ordered_consistent_comms'],'expected_retained_id':cons['commitment_id'],'ordinary_aggregate_retained':any(x['commitment_id']==cons['commitment_id'] for x in d['ordered_consistent_comms'])}
# Family router misses are returned separately; ordinary UnifiedResearchAgent uses scheduled_families only.
lib=SimpleNamespace(_active_reconciled_branches=[],_generalized_branch_commitments=[{'commitment_id':'family:test','routing_guard':guard}])
d2=ns['dispatch_branches_and_families_for_target'](library=lib,consistent_comms=[],transitions=[],schema=None,audit=None,task_sig='T',comm_budget=models.SearchBudget(100))
out['router_deferred_family_not_in_consumed_list']={'scheduled_families':d2['scheduled_families'],'deferred_router_families':d2['deferred_router_families'],'consumer_reference_count':(SRC/'unified_research.py').read_text().count('deferred_router_families')}
# Actual cross-instance function ignores denied spend and performs fitting.
fit_calls=[]
ns['instantiate_aligned_claim_program']=lambda *a,**k:{'aligned':True,'terms':[Program('feature',name='x')]}
def linear_fit(cols,ys):
    fit_calls.append(1)
    xs=cols[0]; mx=sum(xs)/len(xs); my=sum(ys)/len(ys); var=sum((v-mx)**2 for v in xs)
    slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/var if var else 0.
    return [my-slope*mx,slope]
ns['_linear_fit']=linear_fit
ns['_sum_terms']=lambda ts,ws,i:Program('add',children=(Program('const',value=i),*[Program('mul',children=(Program('const',value=w),t)) for t,w in zip(ts,ws)]))
ns['loss']=lambda p,rs,*a:sum((p.predict(r)-r.target)**2 for r in rs)/len(rs)
bud2=models.SearchBudget(0)
xver=ns['_cross_evaluate_fixed_instance_on_neighbor']({}, {'schema':sc,'support_rows_serialized':[{'features':{'x':0.},'target':0.},{'features':{'x':1.},'target':1.}]},budget=bud2)
out['denied_fixed_cross_check_still_executed']={'budget_spent':bud2.spent,'denied_requests':bud2.denied_requests,'fit_calls':len(fit_calls),'verdict':xver}
# Isolated parameter-semantic example: fixed old radius is refuted, legal new binding fits exactly.
ns['instantiate_aligned_claim_program']=lambda c,s:{'aligned':True,'terms':[Program('lt',children=(Program('feature',name='x'),Program('const',value=c['radius_value'])))]}
rows=[{'features':{'x':x},'target':float(x<.75)} for x in (0.,.25,.5,.75,1.)]
fixed=ns['_cross_evaluate_fixed_instance_on_neighbor']({'radius_value':.5},{'schema':sc,'support_rows_serialized':rows},budget=models.SearchBudget(100))
rebound=ns['_cross_evaluate_fixed_instance_on_neighbor']({'radius_value':.75},{'schema':sc,'support_rows_serialized':rows},budget=models.SearchBudget(100))
scoped=ns['shrink_claim_to_observable_scope']({'claim_id':'branch:A','status':'universal','commitment':{'commitment_id':'A','source_concept_ids':['A']}},[{'obs_unconditional_block_rate':0.}],{'obs_unconditional_block_rate':.5})
ns['compute_observable_regime_features']=lambda *a,**k:{'obs_unconditional_block_rate':.5}
d3=ns['dispatch_branches_and_families_for_target'](library=SimpleNamespace(_active_reconciled_branches=[scoped],_generalized_branch_commitments=[]),consistent_comms=[scoped['commitment']],transitions=[],schema=None,audit=None,task_sig='T',comm_budget=models.SearchBudget(100))
out['fixed_instance_vs_refittable_template_control']={'old_radius':.5,'allowed_new_radius':.75,'fixed_cross':fixed,'rebound_cross':rebound,'derived_scope':scoped['scope_guard'],'dispatched_templates':d3['ordered_consistent_comms'],'limitation':'Simple threshold basis with alignment and fitting stubs; demonstrates production scope/dispatch control flow and object-level mismatch, not a full physical-agent replay.'}
# Protocol mismatch is optional only; callers omit expected_protocol.
out['protocol_binding_is_optional']={'when_requested':ns['evaluate_scope_guard'](guard,{'obs_signed_direction_flux':1.,'_meta':{'protocol_id':'protocol_B','pilot_calibration_available':True}},expected_protocol='protocol_A'),'ordinary_call':ns['evaluate_scope_guard'](guard,{'obs_signed_direction_flux':1.,'_meta':{'protocol_id':'protocol_B','pilot_calibration_available':True}})}
def strict_json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return 'NaN'
        return 'Infinity' if value > 0 else '-Infinity'
    if isinstance(value, dict):
        return {key: strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_value(item) for item in value]
    return value
out['non_finite_value_encoding'] = 'Non-finite values in source function outputs are string labels, not JSON numbers.'
out['scope_limitations'] = {
    'dependency_stubs': 'Minimal Program/Observation from the bundled verifier; schema, term construction, consistency, grounding and linear-fit dependencies are explicitly stubbed in this script.',
    'budget_probe': 'Actual production family-instantiation body, cross-check wrapper and SearchBudget; no full simulator execution.',
    'parameter_example': 'Simplified threshold basis with explicit alignment and fit stubs; shows object-level control flow, not a replay of the physical benchmark.',
}
rendered = json.dumps(strict_json_value(out), indent=2, ensure_ascii=False, allow_nan=False)
path = args.output.expanduser()
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(rendered + '\n', encoding='utf-8')
print(rendered)
