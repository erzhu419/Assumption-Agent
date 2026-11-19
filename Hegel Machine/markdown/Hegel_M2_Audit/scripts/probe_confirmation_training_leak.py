"""A legal fit/predict model exposes confirmation rows included in training."""

from _common import HEGEL_REPO, AUDIT_OUTPUT_DIR
import json
from pathlib import Path
from hegel_machine.cognition.cross_domain_clusters import CostMeter
from hegel_machine.cognition.open_workspace_labs import OpenWorkspaceTaskLab
from hegel_machine.cognition.open_workspace_wormhole import validate_workspace_model_on_rows
from hegel_machine.cognition.code_agent import evaluate_python_model

ws = AUDIT_OUTPUT_DIR / 'confirmation_training_leak'
ws.mkdir(parents=True, exist_ok=True)
lab = OpenWorkspaceTaskLab(
    task_id='fit_contract', surface_domain='fit_contract', feature_names=('u',),
    feature_bounds={'u':(1.0,10.0)}, cost_per_obs=5,
    task_view={'goal_kind':'open_workspace_mechanism_prediction','tolerance':0.01},
    meter=CostMeter(obs_budget=10,meas_budget=100,compute_budget=100),
    private_oracle_fn=lambda p:p['u']**2,
)
lab.acquire([{'u':1.0},{'u':2.0}],workspace=ws)
prefix = list(lab.observations)
(ws/'memorize.py').write_text("memory={}\ndef fit(observations):\n    global memory\n    memory={r['features']['u']:r['target'] for r in observations}\ndef predict(row):\n    return memory.get(row['features']['u'],0.0)\n")
confirm = lab.acquire([{'u':3.0}], workspace=ws, phase='confirm')
wrapper = validate_workspace_model_on_rows(ws,'memorize.py',confirm,lab)
prefix_only = evaluate_python_model(ws,'memorize.py',confirm,training_observations=prefix,require_fit=False)
out={'wrapper_result':wrapper,'training_prefix_only_result':prefix_only,'training_prefix_count':len(prefix),'lab_observations_after_confirm':len(lab.observations),'confirmation_target':confirm[0].target}
out_path=ws.parent/'confirmation_training_leak_result.json'
out_path.write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
