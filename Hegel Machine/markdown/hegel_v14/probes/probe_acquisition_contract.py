"""Run exact nested callback from V13 using actual acquire batch schema; no Agent/scientific rerun."""
from pathlib import Path
from dataclasses import dataclass, asdict
from types import SimpleNamespace
import ast,json
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle', type=Path, required=True, help='Directory containing extracted V13 bundled_sources')
parser.add_argument('--output', type=Path, default=Path('probe_acquisition_contract.json'), help='Destination JSON file (default: current working directory)')
args=parser.parse_args()
root=args.bundle.expanduser().resolve()
p=root/'bundled_sources/src/hegel_machine/cognition/unified_research.py'
t=ast.parse(p.read_text());f=next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='_exec_reconcile_probe')
@dataclass
class O:
 features:dict
 target:float
 history:tuple=()
row=O({'x':.5},0.)
req=SimpleNamespace(probe_id='requested_identical_protocol',sequence=({'c0':.5},),reset=True)
real_schema_batch={'probe':{'probe_id':'paid_original_probe','sequence':[{'c0':.5}]},'channels':['fine'],'observations':[asdict(row)],'transitions':[{'before':{'x':0.},'after':{'x':.5}}]}
flat_schema_batch={'probe_id':'paid_original_probe','sequence':[{'c0':.5}],'channels':['fine'],'observations':[row],'transitions':[{'before':{'x':0.},'after':{'x':.5}}]}
acq_calls=[]
def deny(*a,**kw): acq_calls.append(kw.get('probe').probe_id);return None
ns={'reconcile_channels':('fine',),'batches':[real_schema_batch],'acquire':deny}
exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),str(p),'exec'),ns)
a=ns['_exec_reconcile_probe'](req,['v1'])
real={'status':a['status'],'actual_probe_id':a['actual_probe_id'],'new_acquisition_attempts':len(acq_calls)}
ns['batches']=[flat_schema_batch];acq_calls.clear()
b=ns['_exec_reconcile_probe'](req,['v1'])
flat={'status':b['status'],'actual_probe_id':b['actual_probe_id'],'new_acquisition_attempts':len(acq_calls)}
def acquire_new(*a,**kw):
 ns['batches'].append(real_schema_batch)
 return [row]
ns['batches']=[];ns['acquire']=acquire_new
c=ns['_exec_reconcile_probe'](req,['v1'])
new={'status':c['status'],'reported_cost_ref':c['cost_ref'],'actual_production_batch_has_measurement_cost_key':'measurement_cost' in real_schema_batch}
out={'scope':'AST-extracted exact function; explicit environment stub matching production acquire storage schema; not full scientific rerun','real_production_batch':real,'test_fixture_flat_batch':flat,'new_acquisition_cost_ref':new}
print(json.dumps(out,ensure_ascii=False,indent=2))
args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2), encoding='utf-8')
