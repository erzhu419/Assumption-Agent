#!/usr/bin/env python3
"""Bounded source-excerpt controls, NOT a full Agent/scientific rerun.
Extract actual functions from the supplied 6e963206 bundle. Minimal value types
and explicit environment stubs isolate control-flow/semantic questions.
"""
from __future__ import annotations
import ast, json, math, copy, hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import os
BUNDLE=Path(os.environ.get('HEGEL_V12_BUNDLE', '/mnt/data/hegel_v12_input')).expanduser().resolve()
ROOT=BUNDLE/'bundled_sources'
OUT=Path(__file__).parent
BPATH=ROOT/'src/hegel_machine/cognition/branch_reconciliation.py'
UPATH=ROOT/'src/hegel_machine/cognition/unified_research.py'
@dataclass(frozen=True)
class Observation:
    features: dict
    target: float
    history: tuple=()
@dataclass(frozen=True)
class Program:
    op:str
    value:float|None=None
    name:str|None=None
    lag:int=0
    children:tuple=()
    definition:Any=None
    def predict(self,row):
        d=row.features if hasattr(row,'features') else row
        if self.op=='const':return float(self.value or 0.)
        if self.op in ('feature','arg'):return float(d[self.name])
        c=[p.predict(row) for p in self.children]
        if self.op=='lt':return float(c[0]<c[1])
        if self.op=='and':return float(all(x>=.5 for x in c))
        if self.op=='or':return float(any(x>=.5 for x in c))
        if self.op=='not':return float(c[0]<.5)
        if self.op=='add':return sum(c)
        if self.op=='sub':return c[0]-c[1]
        if self.op=='mul':return math.prod(c)
        raise ValueError(self.op)
    def to_dict(self):
        return {'op':self.op,'value':self.value,'name':self.name,'lag':self.lag,
                'children':[c.to_dict() for c in self.children]}
    @classmethod
    def from_dict(cls,d):
        return cls(d['op'],d.get('value'),d.get('name'),d.get('lag',0),tuple(cls.from_dict(c) for c in d.get('children',())))
    def simplified(self):return self
@dataclass
class ResearchQuestion:
    question:str
    evidence:dict
    operations:tuple

def loss(prog,rows,*args):
    return sum((prog.predict(r)-r.target)**2 for r in rows)/len(rows)

def funcs(path,names,ns):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(nodes)!=len(names):raise ValueError('missing function')
    pre=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pre,*nodes],type_ignores=[])),str(path),'exec'),ns)

def main():
    ns={'Program':Program,'Observation':Observation,'ResearchQuestion':ResearchQuestion,
        'deepcopy':copy.deepcopy,'isfinite':math.isfinite,'loss':loss,'_as_schema':lambda x:x}
    tree=ast.parse(BPATH.read_text())
    varnames={'FORBIDDEN_SCOPE_TOKENS','PUBLIC_OBSERVABLE_GUARD_MANIFEST','OBSERVABLE_REGIME_FEATURES'}
    for n in tree.body:
        names=[]
        if isinstance(n,ast.Assign):names=[t.id for t in n.targets if isinstance(t,ast.Name)]
        if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name):names=[n.target.id]
        if varnames.intersection(names):
            exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),n],type_ignores=[])),str(BPATH),'exec'),ns)
    funcs(BPATH,{'validate_observable_scope_guard','compute_observable_regime_features','evaluate_scope_guard','shrink_claim_to_observable_scope','adjudicate_branches_on_paid_evidence','reconcile_competing_branches'},ns)
    out={'commit':'6e963206fd2822476c894fd2d0f09db4be70223c','mode':'source_excerpt_execution_with_explicit_stubs','full_repository_executed':False,'source_sha256':hashlib.sha256(BPATH.read_bytes()).hexdigest()}
    g=Program('lt',children=(Program('feature',name='obs_unconditional_block_rate'),Program('const',value=.1)))
    empty=ns['compute_observable_regime_features']([],None)
    out['empty_pilot_guard']={
        'explicit_no_pilot_validator':ns['validate_observable_scope_guard'](g,pilot_calibration_available=False),
        'computed_empty_block_rate':empty['obs_unconditional_block_rate'],
        'ordinary_evaluate_scope_guard_accepts':ns['evaluate_scope_guard'](g,empty),
        'scope':'actual functions, empty paid transition set; not evidence all recorded episodes hit this state'}
    # Exact nested acquisition callback: acquisition denied -> old evidence returned.
    u_tree=ast.parse(UPATH.read_text())
    fn=next(n for n in ast.walk(u_tree) if isinstance(n,ast.FunctionDef) and n.name=='_exec_reconcile_probe')
    logs=[]
    def deny(*args,**kwargs):logs.append({'args':str(args),'probe':getattr(kwargs.get('probe'),'probe_id',None)});return None
    oldrows=[Observation({'x':0},0.)];oldtrans=[{'old_record':True}]
    cbns={'reconcile_channels':('fine',),'acquire':deny,'ep_obs_rec':oldrows,'ep_trans_rec':oldtrans,'batches':[]}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(UPATH),'exec'),cbns)
    requested=SimpleNamespace(probe_id='new_probe_not_executed',sequence=({'x':1},))
    actual_return=cbns['_exec_reconcile_probe'](requested)
    out['denied_probe_fallback']={'requested_id':requested.probe_id,'acquire_returned':None,'callback_returns_old_rows':actual_return[0] is oldrows,'callback_returns_old_transitions':actual_return[1] is oldtrans}
    # Actual reconcile + actual adjudication: both programs fail, ResearchQuestion discarded.
    old={'claim_id':'old','status':'universal','independent_empirical_support':True}
    new={'claim_id':'new','status':'universal','independent_empirical_support':True}
    a={'program':Program('const',value=1.)};b={'program':Program('const',value=2.)}
    ns['compute_observable_regime_features']=lambda *a,**k:{'obs_signed_direction_flux':1.}
    direct=ns['adjudicate_branches_on_paid_evidence'](new,old,a,b,oldrows,oldtrans,None)
    ns['align_semantics_roles_units_and_time']=lambda *args,**kw:{'aligned_new':{'aligned':True,**a},'compatible_candidates':[{'raw_claim':old,'aligned':b}],'incompatible_semantics':[]}
    ns['assess_scope_overlap']=lambda *args,**kw:{'certified_disjoint':False,'unknown':False}
    ns['find_feasible_prediction_difference']=lambda *args,**kw:{'proved_absent':False,'witness_available':True,'probe':requested,'max_abs_disagreement':1.}
    rec=ns['reconcile_competing_branches'](new,[old],None,None,execute_probe_fn=lambda p:cbns['_exec_reconcile_probe'](p))
    out['both_refuted_question_and_false_probe_ledger']={
        'adjudicator_has_research_question':'research_question' in direct,
        'reconciler_has_research_question':'research_question' in rec,
        'verdict':direct['verdict'],
        'old_status_after':rec['updated_old_claims'][0]['status'],
        'new_status_after':rec['updated_new_claim']['status'],
        'recorded_paid_probes':rec['paid_probes_executed'],
        'actual_successful_acquisitions':0,
        'scope':'minimal constants and denied acquisition control; not a replay of recorded scientific tasks'}
    # Execute actual block filtering commitment IDs using branch-claim IDs.
    lines=UPATH.read_text().splitlines()
    st=next(i for i,l in enumerate(lines) if l.strip()=='if matched_branch_comms:')
    en=next(i for i in range(st+1,len(lines)) if lines[i].strip().startswith('branch_dispatch_telemetry ='))
    import textwrap
    code=textwrap.dedent('\n'.join(lines[st:en]))
    fns={'matched_branch_comms':[],'checked_branches':[{'claim_id':'branch:concept_A'}],
         'consistent_comms':[{'commitment_id':'world_commitment:0000','source_concept_ids':['concept_A']}],
         'pruned_scope_ids':['branch:concept_A'],'pruned_retracted_ids':[]}
    exec(code,fns)
    out['branch_scope_id_vs_commitment_id']={'branch_reported_pruned':'branch:concept_A','commitment_still_in_candidate_pool':fns['consistent_comms'],'block_lines':[st+1,en]}
    # Multi-clause complement control. Use original shrink and adjudication; replace observed summary only.
    old_guard=Program('lt',children=(Program('const',value=0),Program('feature',name='obs_min_active_donor_q')))
    old_multi={'claim_id':'oldscope','status':'scoped_conditional','independent_empirical_support':True,
               'scope_guard':old_guard.to_dict(),'observable_bounds':{'obs_min_active_donor_q':[0,1e6]},'scope_shrink_features':['obs_min_active_donor_q']}
    good={'obs_min_active_donor_q':1.,'obs_signed_direction_flux':1.}
    new_reg={'obs_min_active_donor_q':1.,'obs_signed_direction_flux':-1.}
    ns['compute_observable_regime_features']=lambda *args,**kw:new_reg
    rr=ns['adjudicate_branches_on_paid_evidence']({'claim_id':'newscope','status':'universal'},old_multi,
        {'program':Program('const',value=0.)},{'program':Program('const',value=1.)},oldrows,oldtrans,None,old_support_regime_features=[good])
    ng=rr['updated_new_claim'].get('scope_guard')
    out['new_scope_complement_control']={
       'verdict':rr['verdict'],'old_guard':rr['updated_old_claim']['scope_guard'],
       'new_guard':ng,'newly_supporting_regime_accepted':ns['evaluate_scope_guard'](ng,new_reg),
       'note':'This example confirms the newly chosen second feature; no general logical complement proof implied.'}
    (OUT/'probe_results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
