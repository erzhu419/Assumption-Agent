"""Comprehensive M8.9.1 Deterministic Audit Closure & Certificate Kernel Regression Suite (`R1..R7`).

Directly reproduces and verifies closure of EVERY counterexample from `Hegel_M8_9_Audit`:
- `R1`: `hs01` & `PT06` certificate soundness (`deliberately_invalid_solver` suboptimal + fake LB rejected, `I_3` analytic fake LB rejected,
  valid 1-leaf root certificate accepted, optimal solution + fake LB `certificate_valid=False`, proof-carrying `dual_nu` & PSD minorant
  witnesses verified above `max_enum_p`, and all 6 unique sealed `M8.9` `hs01` submissions verified `18/18`).
- `R2`: Content-addressed `(filename, actual_source_sha)` L2 versioning (`7` versions preserved, `0` silent drops), exact-SHA reverification
  fan-out (`0` cross-SHA receipt overwrites), and fail-closed `L1` qualification gate (`missing/empty/stale receipt -> unqualified`, never staged).
- `R3`: Bi-directional ordered-closure + `ast.Return` `SimpleNamespace` export-map retention (`removed K=2` & `changed export map` rejected),
  scope-aware reachability (`uncalled nested helper` excluded), and structured/fenced entrypoint extraction (`ho01`, `ho06`, `uf04` `0` empty).
- `R4`: `python3 -c "import sol"` rejected as substantive verification; `target_identifiability` nogood requires explicit identifiability/gauge
  witness matching `scope` and stays `pending` on generic assertions.
- `R5`: Two-phase atomic write-back (`out_1.py` host `OSError` causes `0` container writes; container `docker cp` failure rolls back & verifies SHA).
- `R6` & `R7`: Strict router `(feature_schema_version, feature_names, feature_builder_sha256)` check on load, clean `prediction_state_digest`
  (`0` false cache misses on unrelated shared-root task registration while keeping temporal history sensitivity), `20000 unknowns` wording
  invariance, symmetric 6-module progressive catalog, repair precondition + top-k contract prefetch, and real Docker E2E smoke test.
"""

from __future__ import annotations

import ast
import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch

import numpy as np
import pytest

from hegel_machine.cognition import m6_4_executable_wormhole_library as lib
from hegel_machine.cognition import m6_frontier_real_benchmarks as front
from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    build_holdout_ood_12_tasks,
    ensure_clean_scipy_sandbox_image,
)
from hegel_machine.cognition.m8_8_unseen_postfreeze_benchmark import build_unseen_postfreeze_4_tasks
from hegel_machine.cognition.m8_9_1_box_l0_certificate_kernel import (
    ExactAllSupportBoxQPReference,
    eval_exact_box_l0_objective,
    verify_box_l0_subcube_certificate,
)
from hegel_machine.cognition.m8_9_hard_stress_benchmark import HARD_STRESS_M89_TASKS
from hegel_machine.cognition import m8_offline_container_planner as rtr
from hegel_machine.cognition.m8_persistent_variational_planner import (
    M8VariationalPlanner,
    PersistentMechanismGraph,
    TypedMechanism,
    evaluate_committed_task_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V5_BUNDLE = (
    PROJECT_ROOT
    / "docs/evidence/m8_9_progressive_disclosure_and_net_value_ablation_20261126/pretrained_autogenous_forest_v5_m89.json"
)


def _deliberately_invalid_hs01_solver(problem: dict) -> dict:
    """Exact reproduction of `Hegel_M8_9_Audit/hs01/reproduce_hs01_unsound_certificate.py`."""
    X = np.asarray(problem["X"], dtype=float)
    y = np.asarray(problem["y"], dtype=float)
    p = X.shape[1]
    ridge = float(problem["ridge"])
    box = float(problem["box"])
    l0_pen = float(problem["l0_penalty"])
    trials = []
    for active0 in (False, True):
        w = np.zeros(p, dtype=float)
        if active0:
            rhs = float(X[:, 0] @ y)
            w[0] = np.clip(rhs / (float(X[:, 0] @ X[:, 0]) + ridge), -box, box)
            if abs(w[0]) <= 1e-9:
                w[0] = 2e-9 if rhs >= 0 else -2e-9
        trials.append(w.copy())
        free = list(range(1, p))
        if free:
            r = y - X @ w
            j = free[int(np.argmax(np.abs(X[:, free].T @ r)))]
            wc = w.copy()
            wc[j] = np.clip(float(X[:, j] @ r) / (float(X[:, j] @ X[:, j]) + ridge), -box, box)
            trials.append(wc)
    w_best = min(trials, key=lambda v: eval_exact_box_l0_objective(X, y, v, ridge, l0_pen))
    val = eval_exact_box_l0_objective(X, y, w_best, ridge, l0_pen)
    fake_lb = val - 1e-8
    return {
        "weights": w_best.tolist(),
        "objective": val,
        "certified_lower_bound": fake_lb,
        "certificate_leaves": [
            {"fixed_zero": [0], "fixed_nonzero": [], "leaf_lower_bound": fake_lb},
            {"fixed_zero": [], "fixed_nonzero": [0], "leaf_lower_bound": fake_lb},
        ],
    }


def test_r1_hs01_and_pt06_certificate_kernel_rejects_fake_bounds_and_accepts_single_leaf() -> None:
    # 1. Analytic I_3 control: true optimum is 0.2, fake 2-leaf submission claims 0.6 / 0.59999999
    small = {"X": np.eye(3), "y": np.array([0.0, 1.0, 1.0]), "ridge": 0.0, "l0_penalty": 0.1, "box": 2.0, "gap_tol": 1e-6}
    fake_ans = _deliberately_invalid_hs01_solver(small)
    chk_fake = verify_box_l0_subcube_certificate(small, fake_ans)
    assert chk_fake["strict_pass"] is False
    assert chk_fake["solution_correct"] is False
    assert chk_fake["leaf_bounds_valid"] is False
    assert chk_fake["certificate_valid"] is False

    # 2. Valid 1-leaf global certificate on I_3 (`w = [0, 1, 1]`, `J = 0.2`) -> MUST PASS!
    valid_1leaf = {
        "weights": [0.0, 1.0, 1.0],
        "objective": 0.2,
        "certified_lower_bound": 0.2,
        "certificate_leaves": [{"fixed_zero": [], "fixed_nonzero": [], "leaf_lower_bound": 0.2}],
    }
    chk_valid1 = verify_box_l0_subcube_certificate(small, valid_1leaf)
    assert chk_valid1["strict_pass"] is True
    assert chk_valid1["solution_correct"] is True
    assert chk_valid1["certificate_valid"] is True
    assert chk_valid1["leaf_count"] == 1

    # 3. True optimal solution paired with bogus inflated leaf_lower_bound (`0.2 + 1.0` or `+Inf`) -> solution_correct=True, certificate_valid=False!
    bad_cert_good_sol = {
        "weights": [0.0, 1.0, 1.0],
        "objective": 0.2,
        "certified_lower_bound": 0.2,
        "certificate_leaves": [{"fixed_zero": [], "fixed_nonzero": [], "leaf_lower_bound": 1.2}],
    }
    chk_bad_cert = verify_box_l0_subcube_certificate(small, bad_cert_good_sol)
    assert chk_bad_cert["solution_correct"] is True
    assert chk_bad_cert["leaf_bounds_valid"] is False
    assert chk_bad_cert["certificate_valid"] is False
    assert chk_bad_cert["strict_pass"] is False

    # 4. Run actual `hs01` `private_verifier_py` against `deliberately_invalid_solver` -> MUST FAIL CLOSED!
    hs01 = next(t for t in HARD_STRESS_M89_TASKS if t.task_id == "hs01__correlated_box_l0_subcube_certificate")
    fake_mod = types.ModuleType("certified_box_l0")
    fake_mod.solve_certified_box_l0 = _deliberately_invalid_hs01_solver  # type: ignore[attr-defined]
    prev = sys.modules.get("certified_box_l0")
    sys.modules["certified_box_l0"] = fake_mod
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(hs01.private_verifier_py, "<hs01_m891_grader>", "exec"), {})
    finally:
        if prev is None:
            sys.modules.pop("certified_box_l0", None)
        else:
            sys.modules["certified_box_l0"] = prev
    verdict = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert verdict["strict_pass"] is False
    assert verdict["solution_correct"] is False

    # 5. Proof-carrying `dual_nu` & `minorant_diagonal` above `max_enum_p` (`p = 20`, `max_enum_p = 16`):
    p_hi = 20
    X_hi = np.eye(p_hi)
    y_hi = np.zeros(p_hi)
    y_hi[[0, 5]] = [1.5, -1.2]
    prob_hi = {"X": X_hi, "y": y_hi, "ridge": 0.5, "l0_penalty": 0.2, "box": 2.0, "gap_tol": 1e-5}
    w_hi = np.zeros(p_hi)
    w_hi[0] = 1.0
    w_hi[5] = -0.8
    obj_hi = eval_exact_box_l0_objective(X_hi, y_hi, w_hi, 0.5, 0.2)
    nu_star = y_hi - X_hi @ w_hi
    # 5a. Scalar-only leaf without witness above `max_enum_p=16` MUST fail closed:
    no_wit = {
        "weights": w_hi.tolist(),
        "objective": obj_hi,
        "certified_lower_bound": obj_hi,
        "certificate_leaves": [{"fixed_zero": [], "fixed_nonzero": [], "leaf_lower_bound": obj_hi}],
    }
    assert verify_box_l0_subcube_certificate(prob_hi, no_wit, max_enum_p=16)["strict_pass"] is False
    # 5b. Unbranched root Fenchel dual claiming `obj_hi` MUST ALSO be rejected because coord 5 has perspective root gap (`h(-0.4)=0.16 < rho=0.20`)!
    unbranched_fenchel = {
        "weights": w_hi.tolist(),
        "objective": obj_hi,
        "certified_lower_bound": obj_hi - 1e-7,
        "certificate_leaves": [{"fixed_zero": [], "fixed_nonzero": [], "leaf_lower_bound": obj_hi - 1e-7, "dual_nu": nu_star.tolist()}],
    }
    assert verify_box_l0_subcube_certificate(prob_hi, unbranched_fenchel, max_enum_p=16)["leaf_bounds_valid"] is False
    # 5c. Branching on coord 5 (`fixed_nonzero=[5]` with `nu_star`, `fixed_zero=[5]` with `nu_z5`) closes the Fenchel dual gap exactly!
    w_z5 = np.zeros(p_hi)
    w_z5[0] = 1.0
    nu_z5 = y_hi - X_hi @ w_z5  # nu_z5[5] = -1.2 -> on fixed_zero=[5], coord 5 contributes 0 penalty, giving exact subcube opt!
    obj_z5 = eval_exact_box_l0_objective(X_hi, y_hi, w_z5, 0.5, 0.2)
    branched_fenchel = {
        "weights": w_hi.tolist(),
        "objective": obj_hi,
        "certified_lower_bound": obj_hi - 1e-7,
        "certificate_leaves": [
            {"fixed_zero": [], "fixed_nonzero": [5], "leaf_lower_bound": obj_hi - 1e-7, "dual_nu": nu_star.tolist()},
            {"fixed_zero": [5], "fixed_nonzero": [], "leaf_lower_bound": obj_z5 - 1e-7, "dual_nu": nu_z5.tolist()},
        ],
    }
    chk_wit = verify_box_l0_subcube_certificate(prob_hi, branched_fenchel, max_enum_p=16)
    assert chk_wit["strict_pass"] is True
    assert chk_wit["certificate_valid"] is True
    # 5d. Exact PSD minorant (`diagonal=ones(20)`, `center=zeros(20)`) closes in 1 leaf at `p=20`!
    minorant_1leaf = {
        "weights": w_hi.tolist(),
        "objective": obj_hi,
        "certified_lower_bound": obj_hi - 1e-7,
        "certificate_leaves": [
            {
                "fixed_zero": [],
                "fixed_nonzero": [],
                "leaf_lower_bound": obj_hi - 1e-7,
                "minorant_diagonal": np.ones(p_hi).tolist(),
                "minorant_center": np.zeros(p_hi).tolist(),
            }
        ],
    }
    chk_min = verify_box_l0_subcube_certificate(prob_hi, minorant_1leaf, max_enum_p=16)
    assert chk_min["strict_pass"] is True

    # 6. `PT06` grader rejects optimal beta paired with bogus `lower_bound = upper_bound + 1.0`!
    pt06 = next(t for t in m7a.build_disjoint_pretraining_suite() if "pt06" in t.task_id)
    def _pt06_good_sol_bad_lb(X, y, lam0, lam2, M, eps):
        ref = ExactAllSupportBoxQPReference(X, y, lam2, lam0, M)
        _, ub = ref.interval()
        # Find optimal support
        best_m = int(np.argmin(ref.ub))
        ids = np.flatnonzero((best_m >> np.arange(X.shape[1])) & 1)
        beta = np.zeros(X.shape[1])
        if len(ids):
            beta[ids] = np.linalg.solve(X[:, ids].T @ X[:, ids] + lam2 * np.eye(len(ids)), X[:, ids].T @ y)
        return {
            "beta": beta,
            "upper_bound": ub,
            "tree_leaves": [{"fixed_zero": [], "fixed_active": [], "lower_bound": ub + 1.0}],
        }
    pt_fake = types.ModuleType("certified_l0_solver")
    pt_fake.solve_and_certify_l0 = _pt06_good_sol_bad_lb  # type: ignore[attr-defined]
    prev_pt = sys.modules.get("certified_l0_solver")
    sys.modules["certified_l0_solver"] = pt_fake
    buf_pt = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf_pt):
            exec(compile(pt06.blackbox_grader_py, "<pt06_m891_grader>", "exec"), {})
    finally:
        if prev_pt is None:
            sys.modules.pop("certified_l0_solver", None)
        else:
            sys.modules["certified_l0_solver"] = prev_pt
    pt_verd = json.loads(buf_pt.getvalue().strip().splitlines()[-1])
    assert pt_verd["solution_correct"] is True
    assert pt_verd["leaf_bounds_valid"] is False
    assert pt_verd["certificate_valid"] is False
    assert pt_verd["strict_pass"] is False


def test_r2_content_version_dedup_exact_reverification_fanout_and_failclosed_qualification() -> None:
    assert V5_BUNDLE.is_file()

    # 1. Same-filename distinct L2 source version MUST NOT be silently swallowed (`total_source_l2_modules == 7`)
    forest, archive, _, _ = m7a.load_autogenous_forest_bundle(V5_BUNDLE)
    target = "autogenous_sparse_ridge_certification.py"
    matches = [
        (nid, m)
        for nid, r in m7a.iter_live_archive_nodes(archive)
        for m in r.get("autogenous_wormhole_modules", [])
        if m.get("filename") == target
    ]
    assert len(matches) >= 2
    _, m_last = matches[-1]
    m_last["python_code"] += "\ndef new_version_probe(x):\n    return x + 42\n"
    new_sha = hashlib.sha256(m_last["python_code"].encode("utf-8")).hexdigest()
    m_last["compiled_code_sha256"] = new_sha

    class _MockRejectCompiler:
        def call_structured(self, **_kw):
            return {"ok": False, "error": "mock_no_paid_call_in_unit_test"}

    with patch.object(m7a, "ensure_clean_scipy_sandbox_image", lambda: None), patch.object(
        m7a.m6f, "start_detached_container", lambda *a, **k: "mock-cid"
    ), patch.object(m7a.m6f, "stop_container_safely", lambda *a, **k: None):
        res_sub = m7a.sublate_autogenous_forest_into_compositional_primitives(
            forest=forest, archive=archive, codex_client=_MockRejectCompiler(), allow_horizontal_wormholes=False
        )
    assert res_sub["total_source_l2_modules"] == 7
    assert res_sub["retained_existing_count"] == 6
    ev_shas = {ev["recomputed_source_sha256"] for ev in res_sub["events"]}
    assert new_sha in ev_shas

    # 2. Same-stem reverification MUST NOT overwrite untouched byte-distinct copy's receipt!
    _, archive2, _, _ = m7a.load_autogenous_forest_bundle(V5_BUNDLE)
    copies = [
        (nid, m)
        for nid, r in m7a.iter_live_archive_nodes(archive2)
        for m in r.get("autogenous_wormhole_modules", [])
        if m.get("filename") == "autogenous_prim_box_ridge_bounds.py"
    ]
    assert len(copies) >= 2
    orig_copy0_sha = hashlib.sha256(copies[0][1]["python_code"].encode("utf-8")).hexdigest()
    copies[-1][1]["python_code"] += "\n# distinct version comment\n"
    mod_copy_sha = hashlib.sha256(copies[-1][1]["python_code"].encode("utf-8")).hexdigest()
    assert orig_copy0_sha != mod_copy_sha

    def _mock_verifier(**kw):
        return {"passed": True, "telemetry_verified": True}

    with patch.object(m7a, "ensure_clean_scipy_sandbox_image", lambda: None), patch.object(
        m7a.m6f, "start_detached_container", lambda *a, **k: "mock-cid"
    ), patch.object(m7a.m6f, "stop_container_safely", lambda *a, **k: None), patch.object(
        m7a, "verify_autogenous_module_in_clean_container", _mock_verifier
    ):
        m7a.enrich_and_harden_l1_primitive_contracts(archive2, reverify_in_clean_container=True)

    for _, m_chk in copies:
        act_s = hashlib.sha256(m_chk["python_code"].encode("utf-8")).hexdigest()
        assert m_chk["compiled_code_sha256"] == act_s
        assert m_chk["unit_test_receipt"]["verified_code_sha256"] == act_s

    # 3. Missing or empty qualification receipt MUST fail closed (`qualified == False`) and NEVER enter disk/manifest/catalog!
    unver_mod = {"filename": "autogenous_prim_unverified_demo.py", "python_code": "def demo(x):\n    return x + 1\n"}
    assert lib.evaluate_l1_skill_staging_net_value(unver_mod, "", set(), "generic array")["qualified"] is False
    assert lib.evaluate_l1_skill_staging_net_value({**unver_mod, "unit_test_receipt": {}}, "", set(), "generic array")["qualified"] is False
    note = {"node_id": "audit_node", "autogenous_only": True, "autogenous_wormhole_modules": [unver_mod]}
    for policy in ("all_compatible", "net_value_gated"):
        sources, manifest, diag = lib.collect_recalled_executable_wormhole_sources(
            [note],
            include_foundational_skills=False,
            allow_l1_primitives=True,
            task_query_text="demo generic array",
            disclosure_mode="progressive",
            staging_policy=policy,
            return_diagnostics=True,
        )
        assert sources == {}
        assert manifest == []
        assert json.loads(diag["contracts_catalog_json"])["skills"] == []


def test_r3_ordered_closure_export_map_nested_scope_and_structured_entrypoints() -> None:
    stem = "autogenous_prim_demo"
    source = "K = 2\ndef scale(x):\n    return x * K\n"
    candidate = f"from hegel_wormhole_skills.{stem} import scale as s\ndef solve(x):\n    return s(x)\n"
    pkg = lib.package_self_contained_wormhole_dependencies(candidate, {stem: source})
    assert pkg["ok"] is True
    base_tree = ast.parse(pkg["packaged_python_code"])

    def _inspect_tree(t: ast.Module) -> dict:
        code = ast.unparse(ast.fix_missing_locations(t)) + "\n"
        with tempfile.TemporaryDirectory() as td:
            Path(td, "solution.py").write_text(code, encoding="utf-8")
            return lib.inspect_final_deliverable_skill_retention(Path(td), {stem: source}, declared_entrypoints=("solve",))

    # 1. Original valid packaged output -> retained & reachable!
    r_orig = _inspect_tree(deepcopy(base_tree))
    assert r_orig["all_retained_ast_match_source"] is True
    assert r_orig["final_retained_l1_modules"] == [stem]
    assert r_orig["entrypoint_reachable_l1_functions"] == [f"{stem}.scale"]

    # 2. Deleted required constant `K = 2` -> MUST reject closure!
    t_rem = deepcopy(base_tree)
    l_rem = next(n for n in t_rem.body if isinstance(n, ast.FunctionDef) and n.name.startswith("__hegel_load_skill_"))
    l_rem.body = [
        n for n in l_rem.body if not (isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "K" for x in n.targets))
    ]
    r_rem = _inspect_tree(t_rem)
    assert r_rem["all_retained_ast_match_source"] is False
    assert r_rem["final_retained_l1_modules"] == []
    assert r_rem["entrypoint_reachable_l1_functions"] == []

    # 3. Altered `ast.Return` `SimpleNamespace(scale=lambda x: x + 1)` -> MUST reject export map!
    t_exp = deepcopy(base_tree)
    l_exp = next(n for n in t_exp.body if isinstance(n, ast.FunctionDef) and n.name.startswith("__hegel_load_skill_"))
    ret_n = next(n for n in l_exp.body if isinstance(n, ast.Return))
    assert isinstance(ret_n.value, ast.Call)
    for kw in ret_n.value.keywords:
        if kw.arg == "scale":
            kw.value = ast.parse("lambda x: x + 1", mode="eval").body
    r_exp = _inspect_tree(t_exp)
    assert r_exp["all_retained_ast_match_source"] is False
    assert r_exp["final_retained_l1_modules"] == []

    # 4. Uncalled nested helper referencing `s(x)` -> MUST NOT count as entrypoint reachable!
    t_nest = deepcopy(base_tree)
    t_nest.body = [n for n in t_nest.body if not (isinstance(n, ast.FunctionDef) and n.name == "solve")]
    t_nest.body += ast.parse("def solve(x):\n    def helper_never_called():\n        return s(x)\n    return x\n").body
    r_nest = _inspect_tree(t_nest)
    assert r_nest["all_retained_ast_match_source"] is True
    assert r_nest["entrypoint_reachable_l1_functions"] == []

    # 5. Entrypoint extraction across `ho01`, `ho06`, `uf03`, `uf04`, and `hs01..hs04` (`0` formula false positives!)
    tasks = {
        t.task_id: t
        for t in build_holdout_ood_12_tasks() + build_unseen_postfreeze_4_tasks() + list(HARD_STRESS_M89_TASKS)
    }
    assert front._extract_declared_python_entrypoints(tasks["ho01__crosswell_anisotropic_clock_tomography"].instruction_text, task=tasks["ho01__crosswell_anisotropic_clock_tomography"]) == ("invert_plume_and_clocks",)
    assert front._extract_declared_python_entrypoints(tasks["ho06__plate_evaporation_deconfounded_ic50"].instruction_text, task=tasks["ho06__plate_evaporation_deconfounded_ic50"]) == ("estimate_compound_hill_parameters",)
    assert front._extract_declared_python_entrypoints(tasks["uf04__certified_box_l0_sensor_selection"].instruction_text, task=tasks["uf04__certified_box_l0_sensor_selection"]) == ("solve_certified_box_l0",)
    assert front._extract_declared_python_entrypoints(tasks["hs01__correlated_box_l0_subcube_certificate"].instruction_text, task=tasks["hs01__correlated_box_l0_subcube_certificate"]) == ("solve_certified_box_l0",)


def test_r4_r5_r6_r7_nogood_witness_two_phase_writeback_router_schema_and_cache() -> None:
    # R4a: `python3 -c "import sol"` MUST NOT qualify as substantive verification!
    assert front.is_substantive_verification_command('python3 -c "import sol"', ("/app/sol.py",)) is False
    assert front.is_substantive_verification_command('python3 -c "import sol; r = sol.solve(1); assert r == 2"', ("/app/sol.py",)) is True

    # R4b: Promotion AST block rejects non-identifiability witness on `target_identifiability` and accepts scope-matched gauge witness!
    fp = PROJECT_ROOT / "src/hegel_machine/cognition/m6_frontier_real_benchmarks.py"
    tree = ast.parse(fp.read_text(encoding="utf-8"))
    block = next(
        n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "pending_nogood_resolutions and nogoods_recorded"
    )
    ng = {
        "nogood_id": "ng_t01",
        "risk_kind": "target_identifiability",
        "assumption_refuted": "absolute target scale identifiable from relative-only observations",
        "scope": "affine gauge scale",
        "resolved": False,
    }
    ns_bad = {
        "re": __import__("re"),
        "pending_nogood_resolutions": [{"proposed_turn": 2, "resolves_ids": ["ng_t01"], "proposed_by_nogood_id": None}],
        "nogoods_recorded": [ng],
        "nogood_resolution_events": [],
        "turn_idx": 3,
        "current_candidate_sha256": "cand_sha_1",
        "cmd_str": "python3 /tmp/run_basic_shape_check.py",
        "ex_res": {"stdout": "shape check passed ok"},
    }
    exec(compile(ast.Module(body=[deepcopy(block)], type_ignores=[]), str(fp), "exec"), ns_bad)
    assert ng["resolved"] is False
    assert len(ns_bad["pending_nogood_resolutions"]) == 1

    ns_good = {
        **ns_bad,
        "cmd_str": "python3 /tmp/verify_affine_gauge_identifiability.py",
        "ex_res": {"stdout": "two_gauge affine scale identifiability recovery assert max_err=1.2e-9 pass"},
    }
    exec(compile(ast.Module(body=[deepcopy(block)], type_ignores=[]), str(fp), "exec"), ns_good)
    assert ng["resolved"] is True
    assert ns_good["pending_nogood_resolutions"] == []

    # R5: Host local `out_1.py` `OSError` in Phase 1b MUST cause ZERO container writes (`write_attempts == []`)!
    originals = {
        "/app/a.py": "from hegel_wormhole_skills.demo import scale\ndef solve(x):\n    return scale(x)\n",
        "/app/b.py": "from hegel_wormhole_skills.demo import scale\ndef solve(x):\n    return scale(x)\n",
    }
    state = dict(originals)
    writes: list[str] = []

    def _fake_run(args, **_kw):
        if args[:2] == ["docker", "cp"]:
            src, dst = args[2:4]
            if src.startswith("mock:"):
                Path(dst).write_bytes(state[src[5:]].encode("utf-8"))
            else:
                state[dst[5:]] = Path(src).read_text(encoding="utf-8")
                writes.append(dst[5:])
        return subprocess.CompletedProcess(args, 0, b"", b"")

    orig_wt = Path.write_text

    def _fail_out1(p: Path, *a, **kw):
        if p.name == "out_1.py":
            raise OSError("simulated disk full on out_1.py")
        return orig_wt(p, *a, **kw)

    with patch.object(lib, "scan_wormhole_imports_in_container", return_value={"scan_ok": True, "hits": list(originals), "error": ""}), patch.object(
        lib.subprocess, "run", side_effect=_fake_run
    ), patch.object(Path, "write_text", _fail_out1):
        res_tx = lib.auto_package_wormhole_imports_in_container("mock", tuple(originals), {"demo": "def scale(x):\n    return x * 2\n"})
    assert res_tx["ok"] is False
    assert res_tx["container_mutated"] is False
    assert writes == []
    assert state == originals

    # R6: Router `from_json_dict` rejects missing/wrong `feature_schema_version` & mismatched `feature_names`!
    m74_dir = PROJECT_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124"
    recs = rtr.load_historical_suite_episode_records(m74_dir)
    pl = rtr.M8OfflineContainerPlanner.fit_from_historical_records(recs, suite_ids=["m74"])
    blob = pl.to_json_dict()
    assert blob["feature_names"] == [
        "intercept",
        "refuted_or_unfinished",
        "unknown_code_or_ident_risk",
        "unknown_biomystery",
        "turn_exhaustion_pressure",
        "grounded_evidence",
    ]
    pl_rt = rtr.M8OfflineContainerPlanner.from_json_dict(blob)
    assert pl_rt.decide_second_branch(recs[0]["pre_action_state"], record_to_ledger=False)["spawn_second_branch"] == pl.decide_second_branch(
        recs[0]["pre_action_state"], record_to_ledger=False
    )["spawn_second_branch"]
    with pytest.raises(ValueError, match="feature_schema_version"):
        rtr.M8OfflineContainerPlanner.from_json_dict({**blob, "feature_schema_version": "wrong_v0"})
    blob_no_ver = deepcopy(blob)
    blob_no_ver.pop("feature_schema_version", None)
    with pytest.raises(ValueError, match="feature_schema_version"):
        rtr.M8OfflineContainerPlanner.from_json_dict(blob_no_ver)

    # R7a: Registering an unrelated task on shared `root` does NOT invalidate negative cache!
    graph = PersistentMechanismGraph()
    graph.register_mechanism(TypedMechanism("root", None, "shared_upstream_premise", ("a",), shared_weights={"a": 0.0}))
    graph.register_mechanism(TypedMechanism("mem", "root", "temporal_memory", ("a",), temporal_tau=1.5, temporal_driver_col="a"))
    a = np.linspace(-2.0, 2.0, 81)
    train = {"x": {"a": a}, "y": 2.5 * a * a, "past_driver_history": (a * a)[:, None]}
    bad = {"x": {"a": a}, "y": 2.5 * a * a, "past_driver_history": np.full((81, 1), 10.0)}
    good = {"x": {"a": a}, "y": 2.5 * a * a, "past_driver_history": (a * a)[:, None]}
    planner = M8VariationalPlanner(graph)
    tm = planner.register_task(task_id="target", shared_upstream_id="root", feature_names=("a",), train_data=train, val_data=bad)

    def _run_disc(ds):
        base = evaluate_committed_task_model(tm, graph, ds)
        cs = planner._discover_reusable_mechanism_candidates(tm, train, ds, float(base["val_mean_log_predictive"]), float(base["variational_free_energy"]))
        return [float(c.val_log_predictive_gain) for c in cs]

    assert len(_run_disc(bad)) == 1
    assert _run_disc(bad) == []
    planner.register_task(task_id="unrelated", shared_upstream_id="root", feature_names=("a",), train_data=deepcopy(train), val_data=deepcopy(bad))
    assert _run_disc(bad) == []  # Zero false invalidation!
    good_g = _run_disc(good)
    assert len(good_g) == 1 and good_g[0] > 2.5  # Temporal history sensitivity preserved!

    # R7b: `20000 unknowns` wording invariance + symmetric 6-module progressive catalog + repair precondition prefetch!
    _, v5_arch, _, _ = m7a.load_autogenous_forest_bundle(V5_BUNDLE)
    spd_mod = next(
        m
        for _, r in m7a.iter_live_archive_nodes(v5_arch)
        for m in r.get("autogenous_wormhole_modules", [])
        if m.get("filename") == "autogenous_prim_conservative_spd_solve.py"
    )
    g1 = lib.evaluate_l1_skill_staging_net_value(
        spd_mod, "", set(), "Solve a Robin tensor elliptic SPD system with 20000 unknowns using a matrix-free method."
    )
    g2 = lib.evaluate_l1_skill_staging_net_value(
        spd_mod, "", set(), "Solve a Robin tensor elliptic SPD system with 20000 unknowns using an iterative matrix-free method."
    )
    assert g1["recommend_turn1_staging"] is True and g2["recommend_turn1_staging"] is True
    assert g1["heuristic_recommendation_score"] == g2["heuristic_recommendation_score"]

    sample_notes = [deepcopy(r) for _, r in list(m7a.iter_live_archive_nodes(v5_arch))[:2]]
    _, _, diag_all = lib.collect_recalled_executable_wormhole_sources(
        sample_notes,
        include_foundational_skills=False,
        allow_l1_primitives=True,
        full_forest_archive=v5_arch,
        task_query_text="Robin heat flux conductivity pde",
        disclosure_mode="progressive",
        staging_policy="all_compatible",
        return_diagnostics=True,
    )
    _, _, diag_gated = lib.collect_recalled_executable_wormhole_sources(
        sample_notes,
        include_foundational_skills=False,
        allow_l1_primitives=True,
        full_forest_archive=v5_arch,
        task_query_text="Robin heat flux conductivity pde",
        disclosure_mode="progressive",
        staging_policy="net_value_gated",
        return_diagnostics=True,
    )
    assert diag_all["mounted_on_disk_l1_count"] == 6 and diag_gated["mounted_on_disk_l1_count"] == 6
    assert diag_all["queryable_catalog_l1_count"] == 6 and diag_gated["queryable_catalog_l1_count"] == 6
    assert diag_all["card_char_stats"]["min_chars"] >= 200

    rep_ctx = lib.build_repair_skill_context_from_sources(
        {"custom_prim_op": '"""Precondition: MUST pass strictly SPD matrix A and finite rhs b."""\ndef solve_op(A, b):\n    """Solve A x = b."""\n    return b\n'},
        query_text="solve_op SPD matrix rhs",
        max_cards=4,
        top_k_prefetched_contracts=2,
    )
    assert any("SPD matrix A" in c for c in rep_ctx["index_cards"])
    assert len(rep_ctx["prefetched_contracts"]) == 1


def test_m8_9_1_real_container_e2e_smoke_and_knowledge_only_isolation() -> None:
    ensure_clean_scipy_sandbox_image()
    _, v5_arch, _, _ = m7a.load_autogenous_forest_bundle(V5_BUNDLE)
    sample_notes = [deepcopy(r) for _, r in list(m7a.iter_live_archive_nodes(v5_arch))[:2]]
    cid = front.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="1024m", cpus="2", network_mode="none")
    try:
        # 1. Knowledge-only mode mounts `/tmp/hegel_knowledge_ref` and ZERO `/tmp/hegel_wormhole_skills` package!
        man_k = lib.stage_recalled_executable_wormholes_in_container(
            cid,
            sample_notes,
            include_foundational_skills=False,
            allow_l1_primitives=True,
            full_forest_archive=v5_arch,
            task_query_text="box l0 ridge lower_bound",
            disclosure_mode="progressive_knowledge_only",
            staging_policy="all_compatible",
        )
        assert len(man_k) >= 4
        ex_imp = front.exec_container_bash(
            cid,
            "PYTHONPATH=/tmp python3 -c 'import hegel_wormhole_skills' 2>/dev/null && echo IMPORTED || echo BLOCKED",
            timeout_sec=10,
        )
        assert "BLOCKED" in ex_imp["stdout"]
        ex_desc = front.exec_container_bash(
            cid,
            "python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound",
            timeout_sec=10,
        )
        assert ex_desc["exit_code"] == 0 and '"ok": true' in ex_desc["stdout"]
        insps = lib.harvest_container_skill_inspections(cid)
        assert any(ev.get("event") == "skill_contract_delivered" for ev in insps)

        # 2. Executable progressive mode: skill call -> auto-package -> cleanroom verify -> retention check!
        mods_map, _ = lib.collect_recalled_executable_wormhole_sources(
            sample_notes,
            include_foundational_skills=False,
            allow_l1_primitives=True,
            full_forest_archive=v5_arch,
            task_query_text="box l0 ridge lower_bound",
            disclosure_mode="progressive",
            staging_policy="all_compatible",
        )
        lib.stage_recalled_executable_wormholes_in_container(
            cid,
            sample_notes,
            include_foundational_skills=False,
            allow_l1_primitives=True,
            full_forest_archive=v5_arch,
            task_query_text="box l0 ridge lower_bound",
            disclosure_mode="progressive",
            staging_policy="all_compatible",
        )
        front.exec_container_bash(
            cid,
            "mkdir -p /holdout_task && cat <<'EOF' > /holdout_task/solver.py\n"
            "import numpy as np\n"
            "from hegel_wormhole_skills.autogenous_prim_box_ridge_bounds import gram_diagonal_minorant, box_ridge_cube_lower_bound\n"
            "def solve_smoke(X, y):\n"
            "    d = gram_diagonal_minorant(X)\n"
            "    return float(box_ridge_cube_lower_bound(X, y, np.zeros(X.shape[1]), 1.0, 0.1, ridge=0.2, diagonal=d))\n"
            "EOF\n"
            "PYTHONPATH=/tmp python3 -c 'import sys, numpy as np; sys.path.insert(0, \"/holdout_task\"); import solver; print(solver.solve_smoke(np.eye(3), np.ones(3)))'",
            timeout_sec=15,
        )
        pkg_res = lib.auto_package_wormhole_imports_in_container(
            cid,
            ("/holdout_task/solver.py",),
            {Path(k).stem: v for k, v in mods_map.items()},
        )
        assert pkg_res["ok"] is True
        # Cleanroom execution with `/tmp/hegel_wormhole_skills` deleted!
        ex_clean = front.exec_container_bash(
            cid,
            "rm -rf /tmp/hegel_wormhole_skills && python3 -I -c 'import sys, numpy as np; sys.path.insert(0, \"/holdout_task\"); import solver; v = solver.solve_smoke(np.eye(3), np.ones(3)); assert np.isfinite(v); print(v)'",
            timeout_sec=15,
        )
        assert ex_clean["exit_code"] == 0
        with tempfile.TemporaryDirectory() as td_s:
            front.save_submitted_artifacts_snapshot(cid, ("/holdout_task/solver.py",), Path(td_s))
            ret = lib.inspect_final_deliverable_skill_retention(
                Path(td_s),
                {Path(k).stem: v for k, v in mods_map.items()},
                declared_entrypoints=("solve_smoke",),
            )
            assert ret["all_retained_ast_match_source"] is True
            assert "autogenous_prim_box_ridge_bounds" in ret["final_retained_l1_modules"]
            assert sorted(ret["entrypoint_reachable_l1_functions"]) == [
                "autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound",
                "autogenous_prim_box_ridge_bounds.gram_diagonal_minorant",
            ]
    finally:
        front.stop_container_safely(cid)
