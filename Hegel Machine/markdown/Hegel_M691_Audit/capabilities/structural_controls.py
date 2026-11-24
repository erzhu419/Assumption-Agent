"""Read-only controls of M6.9.1 capability semantics. No models, Docker, or task truths."""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
REPO = Path(os.environ.get('HEGEL_AUDIT_REPO', ROOT / 'repo_m691_4001bef56'))
HM = REPO if REPO.name == 'Hegel Machine' else REPO / 'Hegel Machine'
sys.path.insert(0, str(HM / 'src'))
from hegel_machine.cognition import m6_8_b_prime_capability_bridge as bridge
from hegel_machine.cognition import m6_4_executable_wormhole_library as lib
from hegel_machine.cognition import m6_frontier_real_benchmarks as frontier
from hegel_machine.cognition import m6_3_sovereign_causal_wormhole_engine as tracka

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()

out={}

# Change the complete learned law while preserving function count and a temporal label.
# A mechanism-derived abstraction should reflect the changed operator; current proxy is invariant.
subtrees=[]
for variant in (0,1):
    forest, archive, _ = frontier.build_m6_frontier_general_forest()
    with tempfile.TemporaryDirectory(prefix='m691_shared_semantics_') as td:
        pairs=[]
        for i, fam in enumerate(['fam_symbolic_and_operator_dynamics','fam_robust_control_and_calibration']):
            ep=Path(td)/str(i); (ep/'04b_producer_skill_manifest').mkdir(parents=True)
            code = f'def learned_law(x):\n    return {i+1}.0*x + 1.0\n' if variant==0 else f'def learned_law(x):\n    return np.sin({i+2}.0*x*x)\n'
            manifest={'temporal':{'temporal_readout_mode':'mori_zwanzig_or_stage_history'},'structure':{'full_producer_count':1,'reusable_ast_function_closures':[{'name':'learned_law','code':code}]}}
            (ep/'04b_producer_skill_manifest/executable_skill_manifest.json').write_text(json.dumps(manifest))
            pairs.append(({'task_id':f'toy_{i}','short_name':f'toy_{i}','wormhole_family':fam,'strict_pass':True,'benchmark':'toy'},ep))
        rec=bridge.consolidate_wave1_skill_manifests_with_reconciliation(wave1_episode_dirs=pairs,forest=forest,archive=archive)
        subtrees.append(deepcopy(forest.shared_base_subtrees))
out['shared_subtree_ignores_actual_law']={'same_subtree':subtrees[0]==subtrees[1],'hash_linear_sources':digest(subtrees[0]),'hash_sinusoidal_sources':digest(subtrees[1]),'registered_record':subtrees[0]}

# Generated frontier slice selector ignores incumbent-only slices; invalid data can be omitted.
ns={'track_wormhole_skill':lambda _:lambda fn:fn}
exec(bridge.SKILL_07_MULTISCALE_RECONCILIATION.replace('from . import track_wormhole_skill',''),ns)
out['missing_slice_challenger_promoted']=ns['adjudicate_incumbent_vs_challenger_slices']({'majority':1.0,'minority':0.1},{'majority':0.5})

# An explicit causal cutoff with too few prefix points is silently replaced by an index split.
t=np.arange(10,dtype=float); u=np.sin(t); y=np.cos(t)
out['pi_fold_invalid_cutoff_silently_ignored']=ns['pi_fold_temporal_regime_selector'](t,u,y,cutoff_t0=0.0)
out['pi_fold_invalid_cutoff_silently_ignored']['required_actual_prefix_count']=int(np.sum(t<=0))
out['pi_fold_invalid_cutoff_silently_ignored']['actual_fallback_prefix_count']=8

# Scope checker still permits np.load (ordinary file I/O) despite claiming all pure functions are path-free.
source='def numeric_reader(path):\n    """Load a numeric array from a caller supplied file."""\n    return np.load(path)\n'
out['pure_ast_np_load']={'accepted_functions':[n for n,_ in lib.extract_pure_reusable_functions_from_python(source)]}

# New Track-A shadow evaluator now uses old formula, but one broad bin makes opposite laws look consistent.
forest,archive=tracka.build_m6_1_clean_k40_forest()
for v in archive.values():
    v['verified_mechanism_note']='old law y = -z_1'; v.pop('historical_L1_shadow',None)
rows=[{'z_1':float(x),'y':float(x)} for x in [-2,-1,0,1,2]]
rec={'status':'confirmed_ok','strict_pass':True,'task_id':'toy_symmetric_projection','winning_python_code':"def predict(row):\n    return row['z_1']\n",'confirm_rmse':0.0,'recalled_node_ids':['node_C01_TreeA_prime_electro'],'observed_signatures':['sig:c01_treeAprime_electro'],'violated_root_ids':[],'pilot_rows_snapshot':rows}
res=tracka.consolidate_wave1_sovereign_m6_3(wave1_receipts=[rec],arm='Hegel_M6_3_Sovereign_LiveGPT',forest=forest,node_archive=archive)
out['opposite_historical_laws_pass_mean_bin']={'receipt':res['events'][0]['empirical_retroactive_projection'],'pointwise_prediction_rmse':float(np.sqrt(np.mean((2*np.arange(-2,3))**2)))}

# Host nested-CV probe: scanner returns 'term', while probe reads nonexistent h['col'].
# Construct a strong univariate signal so the real scanner produces a significant univariate hit.
with tempfile.TemporaryDirectory(prefix='m691_nested_probe_') as td:
    td=Path(td); model=td/'model.py'; model.write_text('def predict(X_train,y_train,X_test):\n    return (X_test[:,0]>0).astype(float)\n')
    rng=np.random.default_rng(31415); X=rng.normal(size=(120,3)); y=(X[:,0]>0).astype(int)
    data=td/'train.npz'; np.savez(data,X=X,y=y)
    scns={'track_wormhole_skill':lambda _:lambda fn:fn}
    exec(lib.SKILL_01_SYMBOLIC_DYNAMICAL.replace('from . import track_wormhole_skill',''),scns)
    modname='hegel_wormhole_skills.symbolic_and_dynamical_wormhole'
    fake=types.ModuleType(modname); fake.null_calibrated_symbolic_scanner=scns['null_calibrated_symbolic_scanner']
    runner_ns={'__name__':'audit_control'}
    exec(lib.HOST_MECHANICAL_WORMHOLE_GATE_RUNNER_PY,runner_ns)
    runner_ns['_find_public_files']=lambda pat:[str(data)]
    reports=[]
    with patch.dict(sys.modules,{modname:fake}):
        runner_ns['_probe_symbolic_predictor']([str(model)],reports)
    out['symbolic_nested_probe_scanner_contract']={'reports':reports}

# Flow-sensitive straight-line cases improved, but may-alias at control-flow merge changes local dictionaries.
ast_cases={
    'conditional_mixed_row_and_local_dictionary': "def predict(row):\n    r=row\n    if row['z']>0:\n        r={'x':7.0}\n    return r['x']\n",
    'shared_helper_for_row_and_parameter_dictionary': "def helper(q):\n    return q['x']\ndef predict(row):\n    return helper(row)+helper({'x':3.0})\n",
}
out['remaining_ast_role_translation_cases']={}
for label,code in ast_cases.items():
    tr=tracka.extract_and_translate_authored_ast_v2(code,{'x':'target_x','z':'target_z'})
    ns={}; exec(tr['translated_code'],ns)
    try: value=ns['predict']({'target_x':2.0,'target_z':1.0}); error=None
    except Exception as exc: value=None; error=f'{type(exc).__name__}: {exc}'
    out['remaining_ast_role_translation_cases'][label]={'translation':tr,'value':value,'error':error}

Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(out,indent=2,ensure_ascii=False))
