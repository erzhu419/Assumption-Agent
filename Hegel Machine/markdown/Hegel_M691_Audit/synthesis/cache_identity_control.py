"""Execute the exact study-runner forest fingerprint with two different skill bodies."""
import ast, hashlib, json, os
from pathlib import Path
from types import SimpleNamespace
here=Path(__file__).resolve().parent
repo=Path(os.environ.get('HEGEL_AUDIT_REPO',os.environ.get('HEGEL_M691_REPO',str(here.parents[1]/'repo_m691_4001bef56'))))
source=repo/'Hegel Machine/scripts/run_m6_9_a_two_branch_study.py'
tree=ast.parse(source.read_text())
fn=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='_forest_sha16')
archive={'node_A':{'wave1_executable_modules':[{'filename':'skill.py','code':'def op(x): return x'}],'wave1_consolidated_nogoods':[]}}
ns={'hashlib':hashlib,'json':json,'f_arch':archive,'f_obj':SimpleNamespace(shared_base_subtrees={})}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),ns)
a=ns['_forest_sha16']()
archive['node_A']['wave1_executable_modules'][0]['code']='def op(x): raise RuntimeError("different skill")'
b=ns['_forest_sha16']()
r={'before_skill_body_hash':a,'after_different_skill_body_hash':b,'identical_cache_identity':a==b,'scope':'latent cache invalidation defect; not evidence saved current study reused stale episodes'}
(here/'cache_identity_control.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
