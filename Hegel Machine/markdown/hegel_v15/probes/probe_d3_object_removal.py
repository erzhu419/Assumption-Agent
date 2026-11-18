"""V14 D3 exact arm-construction function with a minimal list-shaped KnowledgeLibrary stub.
This checks retention/removal only, not a full Agent run or downstream scientific result.
"""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import ast, argparse, json
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--bundle',type=Path,required=True)
p.add_argument('--output',type=Path,default=Path('probe_d3_object_removal.json'))
a=p.parse_args()
src=a.bundle/'bundled_sources/scripts/run_v14_fair_comparison_and_continuous_object.py'
t=ast.parse(src.read_text());node=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_build_d3_target_libraries')
pre=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
ns={'deepcopy':deepcopy}
exec(compile(ast.fix_missing_locations(ast.Module(body=[pre,node],type_ignores=[])),str(src),'exec'),ns)
ids=['concept_0001','concept_0004']
lib=SimpleNamespace(
 meta_priors=[{'source_concept_id':c} for c in ids],
 concepts=[SimpleNamespace(concept_id=c,status='available') for c in ids],
 abstracts=[SimpleNamespace(template_id=f'{c}:abstract:0',source_concept_id=c,status='available') for c in ids],
 definitions={f'{c}:definition':SimpleNamespace(source_concept_id=c) for c in ids},
 definition_bindings={f'{c}:definition':{'template_id':f'{c}:abstract:0'} for c in ids},
 _active_reconciled_branches=[{'claim_id':f'branch:{c}','source_concept_id':c,'source_concept_ids':[c]} for c in ids],
 _generalized_branch_commitments=[]
)
removed=ns['_build_d3_target_libraries'](deepcopy(lib),deepcopy(lib),'concept_0004')['K_minimal_repaired_object_removed_other_sources_kept']
res={
 'scope':'Exact D3 arm-construction source executed with explicit minimal stub matching KnowledgeLibrary list-shaped concepts. No full scientific rerun; no claim retained object was actually selected in original runs.',
 'removed_id':'concept_0004',
 'meta_prior_remaining_ids':[x['source_concept_id'] for x in removed.meta_priors],
 'branch_remaining_ids':[x['claim_id'] for x in removed._active_reconciled_branches],
 'concept_remaining_ids':[x.concept_id for x in removed.concepts],
 'abstract_remaining_source_ids':[x.source_concept_id for x in removed.abstracts],
 'definition_remaining_source_ids':[x.source_concept_id for x in removed.definitions.values()],
 'definition_binding_keys':list(removed.definition_bindings)
}
a.output.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(res,ensure_ascii=False,indent=2))
