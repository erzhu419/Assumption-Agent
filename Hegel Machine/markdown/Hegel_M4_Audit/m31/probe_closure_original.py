"""Small, no-network boundary probes against the fixed V16.9/V17.0 checkout.

Run with PYTHONPATH=<checkout>/Hegel Machine/src. No repository edits.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from hegel_machine.cognition.cross_domain_clusters import CostMeter
from hegel_machine.cognition.open_workspace_labs import OpenWorkspaceTaskLab, _evaluate_workspace_submission
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.open_workspace_wormhole import (
    canonicalize_python_model_ast,
    validate_workspace_model_on_rows,
)
from hegel_machine.cognition.open_python_revision import ScriptedOpenGrammarCodeProvider, _run_code_agent_with_prefix_confirm
from hegel_machine.cognition.wormhole_forest import authorize_abstraction_use

ROOT = Path(__file__).resolve().parent


def ast_pair(name, a, b, x):
    ca, cb = canonicalize_python_model_ast(a), canonicalize_python_model_ast(b)
    na, nb = {}, {}
    exec(a, na)
    exec(b, nb)
    return {
        "name": name,
        "model_a": a,
        "model_b": b,
        "canonical_a": ca,
        "canonical_b": cb,
        "same_digest": ca["ast_skeleton_digest"] == cb["ast_skeleton_digest"],
        "x": x,
        "prediction_a": na["predict"]({"features": {"x": x}}),
        "prediction_b": nb["predict"]({"features": {"x": x}}),
    }


def probe_confirmation_files():
    ws = ROOT / "file_memorizer_workspace"
    ws.mkdir(exist_ok=True)
    lab = OpenWorkspaceTaskLab(
        task_id="file_fit_contract", surface_domain="file_fit_contract",
        feature_names=("u",), feature_bounds={"u": (1.0, 10.0)}, cost_per_obs=5.0,
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.01},
        meter=CostMeter(obs_budget=10, meas_budget=100.0, compute_budget=100.0),
        private_oracle_fn=lambda p: p["u"] ** 2,
    )
    lab.acquire([{"u": 1.0}, {"u": 2.0}], workspace=ws)
    prefix = list(lab.observations)
    module_prefix = "import json\nmemory={}\n"
    module_suffix = (
        "    global memory\n"
        "    memory={r['features']['u']: r['target'] for r in rows}\n"
        "def predict(row):\n"
        "    return memory.get(row['features']['u'], 0.0)\n"
    )
    for fname, rows_line in [
        ("argument_memorizer.py", "    rows=observations\n"),
        ("file_memorizer.py", "    rows=json.load(open('observations.json'))\n"),
        ("model_file_memorizer.py", "    rows=json.load(open('model_observations.json'))\n"),
    ]:
        (ws / fname).write_text(module_prefix + "def fit(observations):\n" + rows_line + module_suffix)
    confirm = lab.acquire([{"u": 3.0}], workspace=ws, phase="confirm")
    assert confirm is not None
    res = {
        fname: validate_workspace_model_on_rows(ws, fname, confirm, lab, training_prefix=prefix)
        for fname in ("argument_memorizer.py", "file_memorizer.py", "model_file_memorizer.py")
    }
    res["disk_observations"] = json.loads((ws / "observations.json").read_text())
    res["disk_model_observations"] = json.loads((ws / "model_observations.json").read_text())
    res["training_prefix_count"] = len(prefix)
    res["confirmation_target"] = confirm[0].target
    return res


def probe_provider_dataflow():
    base = ScriptedOpenGrammarCodeProvider(
        query_batches=[], candidate_specs=[("candidate.py", "def predict(row): return 1.0\n")],
        diagnostic_script="print('diagnostic')",
    )
    first = base.next_action({"events": []})
    a, b = deepcopy(base), deepcopy(base)
    action_a = a.next_action({"events": [{"action": "run", "feedback": {"stdout": "high_drive_turnover_with_bypass"}}]})
    action_b = b.next_action({"events": [{"action": "run", "feedback": {"stdout": "pure_oscillation_no_bypass"}}]})
    return {"diagnostic_action": first, "after_feedback_a": action_a,
            "after_feedback_b": action_b, "same_next_action": action_a == action_b}


def probe_code_agent_file_fit_lifecycle():
    ws = ROOT / "code_agent_file_fit_workspace"
    lab = OpenWorkspaceTaskLab(
        task_id="code_agent_file_fit_contract", surface_domain="file_fit_contract",
        feature_names=("u",), feature_bounds={"u": (1.0, 10.0)}, cost_per_obs=5.0,
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.01},
        meter=CostMeter(obs_budget=10, meas_budget=100.0, compute_budget=300.0),
        private_oracle_fn=lambda p: p["u"] ** 2,
    )
    code = (
        "import json\nmemory={}\n"
        "def fit(observations):\n"
        "    global memory\n"
        "    rows=json.load(open('observations.json'))\n"
        "    memory={r['features']['u']:r['target'] for r in rows}\n"
        "def predict(row):\n"
        "    return memory.get(row['features']['u'],0.0)\n"
    )
    provider = ScriptedOpenGrammarCodeProvider(
        query_batches=[[{"u": 1.0}, {"u": 2.0}]],
        candidate_specs=[("file_fit_model.py", code)], tolerance=.01,
    )
    sub = _run_code_agent_with_prefix_confirm(lab, ws, provider, {"u": 3.0})
    external = _evaluate_workspace_submission(
        ws, sub.get("model_path"), lab.observations,
        [Observation(features={"u": 4.0}, target=16.0, history=())], .01,
        submission_status=sub["status"],
    )
    return {"submission": sub, "external_evaluation": external}


def probe_receipt_optional_bindings():
    cc = {"claim_id": "c", "permission_tier": "local_solver_guidance", "evidence_status": "SUPPORTED_IN_SCOPE", "preconditions": []}
    receipt = {"claim_id": "c", "passed": True, "observed_error": 0.002,
               "target_id": "old_target", "node_version": 1, "model_sha256": "old_code"}
    result = {}
    for name, tv, heldout in [
        ("legacy_view_all_bindings_omitted", {"goal_kind": "open_workspace_mechanism_prediction", "tolerance": .01}, False),
        ("target_binding_present", {"goal_kind": "open_workspace_mechanism_prediction", "tolerance": .01, "target_id": "new_target"}, False),
        ("missing_confirmation_digest_accepted", {"goal_kind": "open_workspace_mechanism_prediction", "tolerance": .01, "target_id": "old_target", "node_version": 1, "expected_model_sha256": "old_code"}, False),
        ("legacy_boolean_bypass", {"goal_kind": "open_workspace_mechanism_prediction", "tolerance": .01, "target_id": "new_target"}, True),
    ]:
        result[name] = authorize_abstraction_use(
            requested_tier="local_solver_guidance", task_view=tv, target_features={},
            claim_contract=cc, evidence_receipt=receipt, heldout_confirmed=heldout,
        )
    return result


def main():
    results = {
        "commit": "612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8",
        "ast": [
            ast_pair("whitelisted_function_shadowing",
                "def exp(x):\n    return x*x\ndef predict(row):\n    return exp(row['features']['x'])\n",
                "def exp(x):\n    return x*x*x\ndef predict(row):\n    return exp(row['features']['x'])\n", 2.0),
            ast_pair("keyword_semantics_discarded",
                "def predict(row):\n    return round(row['features']['x'], ndigits=1)\n",
                "def predict(row):\n    return round(row['features']['x'], ndigits=3)\n", 1.2345),
        ],
        "confirmation_files": probe_confirmation_files(),
        "provider_dataflow": probe_provider_dataflow(),
        "code_agent_file_fit_lifecycle": probe_code_agent_file_fit_lifecycle(),
        "receipt_optional_bindings": probe_receipt_optional_bindings(),
    }
    (ROOT / "closure_probe_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "ast_collisions": [{"name": r["name"], "statuses": [r["canonical_a"]["skeleton_status"], r["canonical_b"]["skeleton_status"]], "same_digest": r["same_digest"], "predictions": [r["prediction_a"], r["prediction_b"]]} for r in results["ast"]],
        "memorizer_rmse": {k: v["rmse"] for k, v in results["confirmation_files"].items() if k.endswith('.py')},
        "provider_ignores_diagnostic_stdout": results["provider_dataflow"]["same_next_action"],
        "code_agent_file_fit_lifecycle": results["code_agent_file_fit_lifecycle"],
        "receipt_granted": {k: v["granted"] for k, v in results["receipt_optional_bindings"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
