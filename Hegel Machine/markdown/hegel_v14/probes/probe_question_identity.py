"""Bounded V13 actual adjudicator probe: no full Agent or scientific rerun."""
from pathlib import Path
import ast,json,hashlib
from copy import deepcopy
from dataclasses import dataclass,field
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle', type=Path, required=True, help='Directory containing extracted V13 bundled_sources')
parser.add_argument('--output', type=Path, default=Path('probe_question_identity.json'), help='Destination JSON file (default: current working directory)')
args=parser.parse_args()
root=args.bundle.expanduser().resolve()
mp=root/'bundled_sources/src/hegel_machine/cognition/models.py'
bp=root/'bundled_sources/src/hegel_machine/cognition/branch_reconciliation.py'
@dataclass
class Program:
 value:float
 def predict(self,row):return self.value
 def to_dict(self):return {'op':'const','value':self.value}
eval_calls=[]
def loss(prog,rows,*a):
 eval_calls.append(prog.value)
 return sum((prog.predict(r)-r.target)**2 for r in rows)/len(rows)
ns={'dataclass':dataclass,'field':field,'Program':Program,'json':json,'hashlib':hashlib,'deepcopy':deepcopy,'loss':loss,'compute_observable_regime_features':lambda *a,**kw:{'observed':1.}}
def load(p,names):
 t=ast.parse(p.read_text());nodes=[n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 pre=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[pre,*nodes],type_ignores=[])),str(p),'exec'),ns)
load(mp,{'SearchBudget','ResearchQuestion','Observation'})
load(bp,{'_program_semantic_version','adjudicate_branches_on_paid_evidence'})
new={'claim_id':'unconfirmed_candidate:anonymous_process','status':'candidate'}
old={'claim_id':'branch:concept_0001','status':'universal'}
evidence={'status':'paid_episode_observations','actual_probe_id':'paid_episode:anonymous_process'}
b=ns['SearchBudget'](0)
results=[]
for val in [1.,3.]:
 r=ns['adjudicate_branches_on_paid_evidence'](new,old,{'program':Program(val)},{'program':Program(2.)},[ns['Observation']({},0.)],[],None,evidence_ref=evidence,budget=b)
 results.append({'question_id':r['pending_question']['question_id'],'versions':r['pending_question']['tested_program_versions'],'status':r['updated_old_claim']['status']})
out={'scope':'Exact adjudicator, semantic-version and SearchBudget source with minimal constant-program/value stubs; not a replay of research task','questions_for_different_parent_programs':results,'identical_question_ids':results[0]['question_id']==results[1]['question_id'],'zero_budget':{'spent':b.spent,'denied_requests':b.denied_requests,'predictive_loss_evaluations_still_executed':len(eval_calls),'claims_mutated_after_denial':all(r['status']!='universal' for r in results)}}
print(json.dumps(out,ensure_ascii=False,indent=2))
args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2), encoding='utf-8')
