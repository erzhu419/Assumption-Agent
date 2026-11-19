"""Small read-only boundary controls against commit 11961f702, reusing M3 audit protocols.

No network/model calls; all generated workspaces live beside this probe.
PYTHONPATH=<fixed worktree>/Hegel Machine/src python3 probe_boundaries.py
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from probe_closure_original import ast_pair
from hegel_machine.cognition.cross_domain_clusters import CostMeter
from hegel_machine.cognition.m3_1_walk_factorization import make_m3_1_factorized_walk_suite, build_m3_1_pilot_points, build_m3_1_high_load_confirmation_point
from hegel_machine.cognition.open_workspace_labs import OpenWorkspaceTaskLab
from hegel_machine.cognition.open_workspace_wormhole import validate_workspace_model_on_rows
from hegel_machine.cognition.wormhole_forest import authorize_abstraction_use

ROOT = Path(__file__).resolve().parent
COMMIT = "11961f702739ab272661ff59b77fb5bc8e3589b8"


def ast_boundaries():
    duplicate = "def predict(row):\n    return row['features']['x']\ndef predict(row):\n    return row['features']['x'] ** {p}\n"
    attr_shadow = "import math\nmath.exp = lambda x: x ** {p}\ndef predict(row):\n    return math.exp(row['features']['x'])\n"
    original_exp = math.exp
    first = ast_pair("duplicate_predict_definition", duplicate.format(p=2), duplicate.format(p=3), 2.0)
    try:
        second = ast_pair("module_attribute_shadowing", attr_shadow.format(p=2), attr_shadow.format(p=3), 2.0)
        for side in ('a', 'b'):
            ns = {}
            exec(second[f'model_{side}'], ns)
            second[f'prediction_{side}'] = ns['predict']({'features': {'x': 2.0}})
    finally:
        math.exp = original_exp
    return [first, second]


def receipt_boundary():
    cc = {"claim_id": "c", "family_plugin": "m3_open_workspace_ast", "permission_tier": "local_solver_guidance", "evidence_status": "SUPPORTED_IN_SCOPE", "preconditions": []}
    tv = {"goal_kind": "open_workspace_mechanism_prediction", "tolerance": .01}
    return {
        "no_receipt_no_target_boolean_true": authorize_abstraction_use(requested_tier="local_solver_guidance", task_view=tv, target_features={}, claim_contract=cc, heldout_confirmed=True),
        "no_receipt_target_present_boolean_true": authorize_abstraction_use(requested_tier="local_solver_guidance", task_view={**tv, "target_id": "t"}, target_features={}, claim_contract=cc, heldout_confirmed=True),
    }


def absolute_path_confirmation_boundary():
    ws = ROOT / "absolute_path_workspace"
    ws.mkdir(exist_ok=True)
    lab = OpenWorkspaceTaskLab(task_id="absolute_path", surface_domain="boundary", feature_names=("u",), feature_bounds={"u": (1., 10.)}, cost_per_obs=5., task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": .01}, meter=CostMeter(obs_budget=10, meas_budget=100., compute_budget=100.), private_oracle_fn=lambda p: p["u"] ** 2)
    lab.acquire([{"u": 1.}, {"u": 2.}], workspace=ws)
    prefix = list(lab.observations)
    code = "import json\nmemory={}\ndef fit(observations):\n    global memory\n    rows=json.load(open('observations.json'))\n    memory={r['features']['u']: r['target'] for r in rows}\ndef predict(row):\n    return memory.get(row['features']['u'], 0.0)\n"
    (ws / "file_memorizer.py").write_text(code)
    confirm = lab.acquire([{"u": 3.}], workspace=ws, phase="confirm")
    return {"relative_path": validate_workspace_model_on_rows(ws, "file_memorizer.py", confirm, lab, training_prefix=prefix), "absolute_path": validate_workspace_model_on_rows(ws, str(ws / "file_memorizer.py"), confirm, lab, training_prefix=prefix)}


def safe_policy():
    suite = make_m3_1_factorized_walk_suite(8_028_101)
    rows = []
    for _, name, (fac, _) in suite["walk_targets"]:
        lab = fac()
        pilot = build_m3_1_pilot_points(lab)
        normalize = lambda p: {k: round((v-lab.feature_bounds[k][0]) / (lab.feature_bounds[k][1] - lab.feature_bounds[k][0]), 9) for k, v in p.items()}
        high = {f: lo + .9*(hi-lo) for f, (lo, hi) in lab.feature_bounds.items()}
        acquired = lab.acquire([high], phase="presolve")
        rows.append({"target": name, "public_safe_bounds": {k: v for k,v in lab.task_view.items() if k.startswith('safe_')}, "pilot_fractions": [normalize(p) for p in pilot], "unconditionally_available_confirmation_fractions": normalize(build_m3_1_high_load_confirmation_point(fac(), symmetric_roles=name.startswith('T25'))), "initial_0_90_load_accepted": acquired is not None, "accepted_fractions": [normalize(r.features) for r in acquired or []]})
    return rows


def main():
    # The attribute-shadow test monkey-patches process-global math.exp, so run it last.
    out = {"commit": COMMIT, "receipt_boundary": receipt_boundary(), "absolute_path_confirmation_boundary": absolute_path_confirmation_boundary(), "safe_policy": safe_policy(), "ast_boundaries": ast_boundaries()}
    (ROOT / "boundary_probe_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"ast": [{"name": x["name"], "same_digest": x["same_digest"], "statuses": [x["canonical_a"]["skeleton_status"], x["canonical_b"]["skeleton_status"]], "predictions": [x["prediction_a"],x["prediction_b"]]} for x in out["ast_boundaries"]], "receipt": out["receipt_boundary"], "confirmation_rmse": {k:v["rmse"] for k,v in out["absolute_path_confirmation_boundary"].items()}, "high_load_accepted": [r["initial_0_90_load_accepted"] for r in out["safe_policy"]]}, indent=2))


if __name__ == "__main__":
    main()
