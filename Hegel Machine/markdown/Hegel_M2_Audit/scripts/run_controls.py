"""Local audit interventions; fixed source files are not modified.

These controls reuse existing M2 functions and paid source schema records. They
are diagnostic counterfactuals, not a replacement preregistered performance study.
"""

from _common import HEGEL_REPO, AUDIT_OUTPUT_DIR
from copy import deepcopy
import inspect
import json
from pathlib import Path
import runpy
import statistics
import textwrap
import types

import hegel_machine.cognition.open_workspace_wormhole as ow
from hegel_machine.cognition.open_workspace_labs import make_m2_open_workspace_suite

OUT = AUDIT_OUTPUT_DIR
REPO = HEGEL_REPO
study = runpy.run_path(str(REPO/'scripts/run_v16_8_m2_open_workspace_wormhole_study.py'))

# Exactly one policy change to the existing Flat implementation: consume the
# same Stage-1 gate that its constructor already copies from Hegel.
flat_source = textwrap.dedent(inspect.getsource(ow.OpenWorkspaceFlatEqualContent.solve_target))
assert flat_source.count('if matches:') == 1
flat_source = flat_source.replace('if matches:', '''gate_dec = self.gate.decide_stage1(
        sigs[-1], remaining_budget=lab.meter.remaining_meas(),
        estimated_probe_and_bind_cost=lab.cost_per_obs,
    )
    self.audit_gate_decisions.append({"target": name, "signature": sigs[-1], **gate_dec})
    if matches and gate_dec["admit_transfer"]:''')
flat_ns = dict(vars(ow))
exec(compile(flat_source, '<audit_flat_same_gate>', 'exec'), flat_ns)
class FlatSameGate(ow.OpenWorkspaceFlatEqualContent):
    def __init__(self, source):
        super().__init__(source)
        self.audit_gate_decisions=[]
    solve_target = flat_ns['solve_target']

# Keep the original full schemas, source observations, guard, and confirmation;
# only allow a matching local node from another named domain without a bridge.
class NoWhLocalContent(ow.OpenWorkspaceHegelResearcher):
    def _filter_cross_domain(self, recalled, target_domain):
        return recalled

# Same ordinary solver, but retrieve all validated same-arity source schemas,
# including one-domain near-miss repairs. No oracle-derived target labels.
direct_src = inspect.getsource(ow.solve_open_workspace_direct)
direct_src = direct_src.replace('def solve_open_workspace_direct(', 'def solve_direct_multi(')
assert direct_src.count('allowed_schemas=[reconstructed_prior_schema]') == 2
direct_src = direct_src.replace('allowed_schemas=[reconstructed_prior_schema]', 'allowed_schemas=reconstructed_prior_schema', 1)
direct_src = direct_src.replace('allowed_schemas=[reconstructed_prior_schema]', 'allowed_schemas=[prior_fit["selected_schema"]]', 1)

def synth_charge_per_candidate(*args, **kwargs):
    schemas=kwargs.get('allowed_schemas')
    if schemas and len(schemas)>1:
        args[1].meter.charge('archive_extra_schema_search', compute=10.0*(len(schemas)-1))
    return ow.run_workspace_python_synthesizer(*args, **kwargs)

direct_ns=dict(vars(ow))
direct_ns['run_workspace_python_synthesizer']=synth_charge_per_candidate
exec(compile(direct_src, '<audit_multi_prior>', 'exec'),direct_ns)
ordinary_solve_src = textwrap.dedent(inspect.getsource(ow.OpenWorkspaceStrongOrdinary.solve_target))
ordinary_solve_src=ordinary_solve_src.replace('return solve_open_workspace_direct(', 'return solve_direct_multi(')
exec(compile(ordinary_solve_src, '<audit_multi_prior_target>', 'exec'),direct_ns)

class OrdinaryMultiPrior(ow.OpenWorkspaceStrongOrdinary):
    def _reconstruct_prior_by_arity(self, lab):
        if not self.allow_archive_reconstruction or not self.raw_archive:
            return None
        arity=len(ow._active_and_fixed_features(lab)[0])
        if self.rebuilt_cache.get('archive_len')!=len(self.raw_archive):
            if not lab.meter.charge('on_demand_reconstruction',compute=26.0):
                return None
            by_arity={}
            for ep in self.raw_archive:
                if ep.get('strict_pass') and ep.get('selected_schema'):
                    by_arity.setdefault(int(ep['active_arity']),set()).add(ep['selected_schema'])
            self.rebuilt_cache={'archive_len':len(self.raw_archive),'priors_by_arity':{a:sorted(s) for a,s in by_arity.items()}}
        return self.rebuilt_cache['priors_by_arity'].get(arity)
    solve_target=direct_ns['solve_target']


def main():
    gate,dev=study['_calibrate_m2_gate'](OUT/'control_workspaces/dev')
    result={'scope':'Post-hoc diagnostic controls at the fixed 9115 commit; not a new frozen benchmark.',
            'dev_summary':{k:v for k,v in dev.items() if k not in ['source_rows','rollout_rows']},
            'clusters':[], 'targets':[]}
    for cid,seed in study['EVAL_CLUSTERS']:
        suite=make_m2_open_workspace_suite(seed)
        ws=OUT/'control_workspaces'/cid
        h=ow.OpenWorkspaceHegelResearcher(gate=deepcopy(gate))
        hs=h.form_and_revise_in_workspaces(cid,suite,ws/'H_source')
        flat=FlatSameGate(h)
        no_wh=deepcopy(h)
        no_wh.__class__=NoWhLocalContent
        no_wh.allow_horizontal_wormholes=False
        ordinary=OrdinaryMultiPrior()
        os=ordinary.run_source_and_nearmiss_phase(suite,ws/'O_source')
        arms={'Hegel_ref':h,'Flat_same_gate':flat,'NoWh_allow_local_content':no_wh,'Ordinary_all_validated_priors':ordinary}
        for family,name,(fac,evaluator) in suite['targets']:
            for arm,obj in arms.items():
                lab=fac()
                target_ws=ws/arm/name
                submission=obj.solve_target(name,lab,target_ws)
                evaluation=evaluator(target_ws,submission,lab.observations)
                row={'cluster_id':cid,'target_name':name,'arm':arm,'strict_pass':evaluation['strict_pass'],
                     'evaluation':evaluation,'submission':submission,'costs':lab.meter.snapshot()}
                result['targets'].append(row)
        summary={'cluster_id':cid,'arms':{},'flat_gate_decisions':flat.audit_gate_decisions,'ordinary_cache':ordinary.rebuilt_cache}
        for arm in arms:
            rows=[r for r in result['targets'] if r['cluster_id']==cid and r['arm']==arm]
            src=sum(r['costs']['composite_cost'] for r in (os if arm=='Ordinary_all_validated_priors' else hs))
            summary['arms'][arm]={'strict_passes':sum(r['strict_pass'] for r in rows),
                 'measurement_cost':sum(r['costs']['measurement_cost'] for r in rows),
                 'compute_cost':sum(r['costs']['compute_cost'] for r in rows),
                 'target_composite_cost':sum(r['costs']['composite_cost'] for r in rows),
                 'source_composite_cost':src,
                 'e2e_composite_cost':src+sum(r['costs']['composite_cost'] for r in rows)}
        result['clusters'].append(summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
        (OUT/'controls_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    result['paired_comparisons']={}
    for arm in ['Flat_same_gate','NoWh_allow_local_content','Ordinary_all_validated_priors']:
        result['paired_comparisons'][arm]={}
        for metric in ['measurement_cost','target_composite_cost','e2e_composite_cost']:
            diffs=[c['arms']['Hegel_ref'][metric]-c['arms'][arm][metric] for c in result['clusters']]
            result['paired_comparisons'][arm][metric]={'per_cluster_Hegel_minus_control':diffs,'mean':statistics.mean(diffs)}
    (OUT/'controls_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['paired_comparisons'],indent=2),flush=True)

if __name__=='__main__':
    main()
