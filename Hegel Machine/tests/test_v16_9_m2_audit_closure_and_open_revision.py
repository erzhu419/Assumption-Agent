"""V16.9 M2 Red-Team Audit Closure & Open Python Revision Loop Tests.

Directly regression-tests all focused counterexamples from `Hegel_M2_Audit` (`F1`–`F11`)
plus the `CodeAgent` + `PhilosophicalForest` open compositional Python revision chains.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.cross_domain_clusters import CostMeter
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.open_python_revision import run_open_revision_chain_experiment
from hegel_machine.cognition.open_workspace_labs import (
    OpenWorkspaceTaskLab,
    _evaluate_workspace_submission,
    make_m2_open_workspace_suite,
)
from hegel_machine.cognition.open_workspace_wormhole import (
    OpenWorkspaceHegelResearcher,
    canonicalize_python_model_ast,
    solve_open_workspace_direct,
    validate_workspace_model_on_rows,
)
from hegel_machine.cognition.wormhole_forest import authorize_abstraction_use


def test_v16_9_ast_canonicalizer_blocks_all_three_audit_collisions_and_keeps_lens_unification() -> None:
    """Verifies F2 fix: annotation spoofing, un-annotated local bindings, and x**2 vs x**-2 never collide."""
    # 1. Stale/mismatched _CORE_LAW_EXPR vs actual predict(row)
    p_a = "_CORE_LAW_EXPR = \"f['x'] / (0.28 * f['x'] + 1.6)\"\ndef predict(row):\n    return row['features']['x']\n"
    p_b = "_CORE_LAW_EXPR = \"f['x'] / (0.28 * f['x'] + 1.6)\"\ndef predict(row):\n    return 100000.0\n"
    ca, cb = canonicalize_python_model_ast(p_a), canonicalize_python_model_ast(p_b)
    assert ca["skeleton_status"] == "ANNOTATION_CODE_MISMATCH"
    assert cb["skeleton_status"] == "ANNOTATION_CODE_MISMATCH"
    assert ca["ast_skeleton_digest"] != cb["ast_skeleton_digest"]

    # 2. No _CORE_LAW_EXPR: local `latent = ...` must be inlined before outer affine lens stripping!
    q_a = "def predict(row):\n    f = row['features']\n    latent = f['x'] / (0.28 * f['x'] + 1.6)\n    return 0.0 + (1.0) * (1.0) * latent\n"
    q_b = "def predict(row):\n    f = row['features']\n    latent = f['x'] ** 5 + f['y']\n    return 0.0 + (1.0) * (1.0) * latent\n"
    qa, qb = canonicalize_python_model_ast(q_a), canonicalize_python_model_ast(q_b)
    assert qa["skeleton_status"] == "VERIFIED_PURE_AST"
    assert qb["skeleton_status"] == "VERIFIED_PURE_AST"
    assert qa["feature_role_order"] == ["x"]
    assert qb["feature_role_order"] == ["x", "y"]
    assert qa["ast_skeleton_digest"] != qb["ast_skeleton_digest"]

    # 3. Structural exponent sign preservation (`x ** 2` != `x ** -2`)
    e_pos = canonicalize_python_model_ast("_CORE_LAW_EXPR = \"f['x'] ** 2\"\n")
    e_neg = canonicalize_python_model_ast("_CORE_LAW_EXPR = \"f['x'] ** -2\"\n")
    assert e_pos["ast_skeleton_digest"] != e_neg["ast_skeleton_digest"]

    # 4. Opposite-sign (+1 vs -1) and shifted readout lenses still unify cleanly!
    chem = "import math\ndef predict(row):\n    f = row['features']\n    latent = 0.9 * f['p'] + 1.2 * f['p'] * math.exp(-f['t'] / 1.7)\n    return 0.0 + (1.0) * (1.0) * latent\n"
    lenz = "import math\ndef predict(row):\n    f = row['features']\n    latent = 0.8 * f['flux'] + 1.4 * f['flux'] * math.exp(-f['dt'] / 1.5)\n    return 3.0 + (-1.0) * (1.5) * latent\n"
    cc, cl = canonicalize_python_model_ast(chem), canonicalize_python_model_ast(lenz)
    assert cc["skeleton_status"] == "VERIFIED_PURE_AST"
    assert cc["ast_skeleton_digest"] == cl["ast_skeleton_digest"]


def test_v16_9_confirmation_prefix_isolation_honest_unknown_and_bound_receipts(tmp_path: Path) -> None:
    """Verifies F5, F7, F11: memorizer blocked (RMSE=9), sin(u) returns scoped_unknown, stale/NaN receipts denied."""
    # F11: Stateful `fit(observations)` memorizer must NOT see confirmation target!
    ws_mem = tmp_path / "memorizer_ws"
    ws_mem.mkdir(parents=True, exist_ok=True)
    lab_mem = OpenWorkspaceTaskLab(
        task_id="fit_contract",
        surface_domain="fit_contract",
        feature_names=("u",),
        feature_bounds={"u": (1.0, 10.0)},
        cost_per_obs=5.0,
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.01},
        meter=CostMeter(obs_budget=10, meas_budget=100.0, compute_budget=100.0),
        private_oracle_fn=lambda p: p["u"] ** 2,
    )
    lab_mem.acquire([{"u": 1.0}, {"u": 2.0}], workspace=ws_mem)
    prefix = list(lab_mem.observations)
    (ws_mem / "memorize.py").write_text(
        "memory={}\n"
        "def fit(observations):\n"
        "    global memory\n"
        "    memory={r['features']['u']: r['target'] for r in observations}\n"
        "def predict(row):\n"
        "    return memory.get(row['features']['u'], 0.0)\n",
        encoding="utf-8",
    )
    confirm = lab_mem.acquire([{"u": 3.0}], workspace=ws_mem, phase="confirm")
    assert confirm is not None
    wrap_res = validate_workspace_model_on_rows(ws_mem, "memorize.py", confirm, lab_mem)
    direct_res = evaluate_python_model(ws_mem, "memorize.py", confirm, training_observations=prefix, require_fit=False)
    assert wrap_res["training_prefix_count"] == 2
    assert wrap_res["rmse"] == 9.0
    assert math.sqrt(direct_res["mse"]) == 9.0

    # F5: Out-of-grammar `5 + sin(u)` fails confirmation -> returns `scoped_unknown`, `false_confirmation == False`!
    ws_sin = tmp_path / "sinusoid_ws"
    lab_sin = OpenWorkspaceTaskLab(
        task_id="audit_sinusoid",
        surface_domain="audit_sinusoid",
        feature_names=("u",),
        feature_bounds={"u": (1.0, 10.0)},
        cost_per_obs=5.0,
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "target_metric": "rmse_original_scale", "tolerance": 0.01},
        meter=CostMeter(obs_budget=24, meas_budget=120.0, compute_budget=800.0),
        private_oracle_fn=lambda p: 5.0 + math.sin(p["u"]),
    )
    sub_sin = solve_open_workspace_direct(lab_sin, ws_sin)
    assert sub_sin["status"] == "scoped_unknown"
    assert sub_sin["model_path"] is None
    eval_rows = [Observation(features={"u": 1.0 + i * 9.0 / 30.0}, target=5.0 + math.sin(1.0 + i * 9.0 / 30.0), history=()) for i in range(31)]
    ext_sin = _evaluate_workspace_submission(
        ws_sin, sub_sin.get("model_path"), lab_sin.observations, eval_rows, 0.01, submission_status=sub_sin["status"]
    )
    assert ext_sin["strict_pass"] is False
    assert ext_sin["false_confirmation"] is False

    # F7: Evidence receipt must bind target_id, node_version, model_sha256, and finite observed_error <= tol!
    cc = {"claim_id": "same_law", "permission_tier": "local_solver_guidance", "evidence_status": "SUPPORTED_IN_SCOPE", "preconditions": []}
    tv = {"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.01, "target_id": "new_target", "node_version": 2, "expected_model_sha256": "abc123"}
    bad_receipts = [
        {"claim_id": "same_law", "passed": True, "target_id": "old_target", "node_version": 1, "model_sha256": "abc123", "observed_error": 0.0},
        {"claim_id": "same_law", "passed": True, "target_id": "new_target", "node_version": 2, "model_sha256": "abc123"},
        {"claim_id": "same_law", "passed": True, "target_id": "new_target", "node_version": 2, "model_sha256": "abc123", "observed_error": float("nan")},
        {"claim_id": "*", "passed": True, "target_id": "new_target", "node_version": 2, "model_sha256": "abc123", "observed_error": 0.002},
        {"claim_id": "same_law", "passed": True, "target_id": "new_target", "node_version": 2, "model_sha256": "wrong_sha", "observed_error": 0.002},
    ]
    for r_bad in bad_receipts:
        auth = authorize_abstraction_use(requested_tier="local_solver_guidance", task_view=tv, target_features={}, claim_contract=cc, evidence_receipt=r_bad)
        assert auth["granted"] is False

    good_receipt = {"claim_id": "same_law", "passed": True, "target_id": "new_target", "node_version": 2, "model_sha256": "abc123", "observed_error": 0.004}
    good_auth = authorize_abstraction_use(requested_tier="local_solver_guidance", task_view=tv, target_features={}, claim_contract=cc, evidence_receipt=good_receipt)
    assert good_auth["granted"] is True


def test_v16_9_rng_fork_coordinate_permutation_and_program_steps_execution(tmp_path: Path) -> None:
    """Verifies F6, F8, F9: independent RNG deepcopy, reversed feature order pass, & interpreted program_steps."""
    suite = make_m2_open_workspace_suite(5026101)
    fac_a1, ev_a1 = suite["targets"][0][2]

    # F9: Independent RNG state across deepcopy forks
    base_lab = fac_a1()
    copy_a, copy_b = deepcopy(base_lab), deepcopy(base_lab)
    assert copy_a._oracle is not copy_b._oracle
    pt_mid = {k: 0.5 * (lo + hi) for k, (lo, hi) in base_lab.feature_bounds.items()}
    ya = copy_a.acquire([pt_mid])[0].target
    yb = copy_b.acquire([pt_mid])[0].target
    assert ya == yb

    # Form forest once
    hegel = OpenWorkspaceHegelResearcher()
    hegel.form_and_revise_in_workspaces("perm_check", suite, tmp_path / "formation")

    # F6: Original vs reversed `feature_names` on T_M2A1 both pass with identical measurement & compute cost!
    lab_orig = fac_a1()
    sub_orig = hegel.solve_target("T_orig", lab_orig, tmp_path / "ws_orig")
    res_orig = ev_a1(tmp_path / "ws_orig", sub_orig, lab_orig.observations)

    hegel_rev = deepcopy(hegel)
    lab_rev = fac_a1()
    lab_rev.feature_names = tuple(reversed(lab_rev.feature_names))
    lab_rev.task_view["feature_names"] = list(lab_rev.feature_names)
    sub_rev = hegel_rev.solve_target("T_rev", lab_rev, tmp_path / "ws_rev")
    res_rev = ev_a1(tmp_path / "ws_rev", sub_rev, lab_rev.observations)

    assert res_orig["strict_pass"] is True
    assert res_rev["strict_pass"] is True
    assert lab_orig.meter.meas_spent == lab_rev.meter.meas_spent
    assert lab_orig.meter.compute_spent == lab_rev.meter.compute_spent

    # F8: Deleting `calibrate_observation_lens` & `subprocess_confirm_and_refit` from `program_steps` changes execution!
    hegel_del = deepcopy(hegel)
    for node in hegel_del.forest.nodes.values():
        node["program_steps"] = [s for s in node.get("program_steps", []) if s.get("op") == "synthesize_scoped_python_schema"]
    lab_del = fac_a1()
    sub_del = hegel_del.solve_target("T_del", lab_del, tmp_path / "ws_del")
    res_del = ev_a1(tmp_path / "ws_del", sub_del, lab_del.observations)
    assert res_del["strict_pass"] is True
    assert lab_del.meter.meas_spent > lab_orig.meter.meas_spent


def test_v16_9_open_author_code_agent_revision_chains(tmp_path: Path) -> None:
    """Verifies F1 & F3: real CodeAgent + stateful scipy.optimize.least_squares open AST revision & reuse!"""
    report = run_open_revision_chain_experiment(5026201, tmp_path / "open_chains")
    for ckey in ("chain_O1", "chain_O2"):
        c = report["chains"][ckey]
        assert c["incumbent_witness_rmse"] > 0.5
        assert c["backward_source_rmse"] < 0.01
        assert c["witness_eval_rmse"] < 0.02

    for tname, arms in report["downstream_targets"].items():
        h = arms["Hegel_OpenRevision_Forest"]
        abl = arms["Ablated_RemoveRevisedNode"]
        ord_s = arms["Continuing_Ordinary_CodeSession"]
        assert h["strict_pass"] is True and h["false_confirmation"] is False
        assert abl["strict_pass"] is True and abl["false_confirmation"] is False
        assert ord_s["strict_pass"] is True and ord_s["false_confirmation"] is False
        assert h["costs"]["measurement_cost"] < abl["costs"]["measurement_cost"]
        assert h["costs"]["composite_cost"] < ord_s["costs"]["composite_cost"]
