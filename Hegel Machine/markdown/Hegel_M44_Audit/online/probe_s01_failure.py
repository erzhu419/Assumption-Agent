"""Bounded control: stop Part3 before S02, forcing S01 failure to test writeback guard.
No source changes, no physical/scientific observations and no model API calls.
Run with HEGEL_AUDIT_REPO=... PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$HEGEL_AUDIT_REPO/src" python ...
"""
from copy import deepcopy
import json
import os
from pathlib import Path
from unittest.mock import patch
from hegel_machine.cognition import m4_4_instance_role_closure_and_online_writeback as m44

BASE = Path(os.environ.get("HEGEL_AUDIT_REPO", "/workspace/scratch/3f14ad2da0ed/repo_m44_b965f022b/Hegel Machine"))
OUT = Path(os.environ.get("HEGEL_AUDIT_OUT", "/workspace/scratch/3f14ad2da0ed/audit_m44_b965f022b/online"))
EVID = BASE / "docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120"
forest = m44.load_location_independent_forest(EVID / "relocatable_forest_snapshots/post_stage2_forest_snapshot.json", EVID / "persisted_authored_workspaces")
node_id = "stage1_W4_monod_shunt"
control = {"control": "S01 failed evaluator result; stop immediately before S02", "observations_acquired": 0, "model_api_calls": 0}
class StopAfterS01(Exception): pass

def evaluator(target, **kwargs):
    nd = kwargs["forest"].nodes[node_id]
    if target.target_id == "M44_S01_W4_LiveRevisionTrigger":
        control["before"] = deepcopy(nd)
        control["forced_receipt"] = {"confirmed_ok": False, "strict_pass": False, "false_confirmation": False,
           "status": "scoped_unknown", "cost": {"measurement_cost": 0.0, "compute_cost": 0.0,
           "composite_cost": 0.0, "phase_breakdown": {}}}
        return deepcopy(control["forced_receipt"])
    control["after"] = deepcopy(nd)
    control["node_history_after"] = deepcopy(kwargs["forest"].node_history.get(node_id, []))
    raise StopAfterS01

with patch.object(m44, "evaluate_m4_3_target_under_arm", evaluator):
    try:
        m44.run_m4_4_online_writeback_and_multicue_stream(OUT / "s01_failed_control_workspace", {"forest": forest, "ordinary_history_forward": []})
    except StopAfterS01:
        pass
b, a = control["before"], control["after"]
control["version_incremented_on_failure"] = a["version"] > b["version"]
control["preconditions_unchanged"] = a["claim_contract"]["preconditions"] == b["claim_contract"]["preconditions"]
control["protocol_before"] = b["claim_contract"].get("protocol_spec")
control["protocol_after"] = a["claim_contract"].get("protocol_spec")
control["conclusion_action_unchanged"] = a["claim_contract"]["conclusion_or_action"] == b["claim_contract"]["conclusion_or_action"]
control["b_1_6_precondition_added"] = any(x.get("field") in {"b", "b_bleed", "b_bleed_ceiling_safe"} for x in a["claim_contract"]["preconditions"])
(OUT / "s01_failed_control.json").write_text(json.dumps(control, ensure_ascii=False, indent=2, sort_keys=True))
print(json.dumps({k:v for k,v in control.items() if k not in {"before", "after", "node_history_after"}},ensure_ascii=False,indent=2))
