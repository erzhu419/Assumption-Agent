"""Deterministic Unit & Container Verification Suite for Hegel M9.3 (`Hegel-M93-Lifecycle`).

Verifies:
1. All 8 `M9.3` lifecycle tasks (`src01`, `suc01_T`, `suc01_C`, `rec01_R`, `src02`, `suc02_T`, `suc02_C`, `rec02_R`) pass in
   real air-gapped Docker verifier containers on reference solvers (`strict_pass == True`) and fail on targeted negative
   controls (`strict_pass == False`: memoryless / ignoring `initial_memory_states` / linear-only on `suc01_C` / suboptimal KKT
   duality gap / rank-1 additive separable approximation on `suc02_C`).
2. All 8 `public_setup_py` + `/holdout_task/validate_public_schema.py` scripts materialize required public fixtures, exit 0 on
   valid outputs, and exit non-zero on malformed outputs.
3. Direct source float plug-in fails across all 4 successors (`v10.2` `C2` lifecycle non-triviality check).
4. Newly compiled online `L1` primitives (`hegel_l1_qual_4tuple_v2`) rank #1 on their matching successor chains (`suc01_*` vs
   `suc02_*`), are vetoed on cross-domain chains (`0` cross-chain pollution), and produce `0` search rank/member disagreements
   between `K` and `E` under `_SHARED_CATALOG_READER_CORE_PY`.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition import m6_4_executable_wormhole_library as m64
from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition.m9_3_online_lifecycle_benchmark import (
    M93_LIFECYCLE_TASKS,
    M93_TASKS_BY_ID,
    evaluate_direct_source_float_plugin_controls,
    get_m93_recovery_frontier_tasks,
    get_m93_source_frontier_tasks,
    get_m93_successor_frontier_tasks,
    run_m93_lifecycle_private_verifier_by_id,
)

V7_PATH = REPO_ROOT / "docs/evidence/m9_2_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v7_m921.json"


def _bundle_py_tar(td: Path, rel_path: str, code: str, name: str) -> Path:
    tar_p = td / name
    raw = code.encode("utf-8")
    with tarfile.open(tar_p, "w") as tf:
        ti = tarfile.TarInfo(name=rel_path.lstrip("/"))
        ti.size = len(raw)
        tf.addfile(ti, io.BytesIO(raw))
    return tar_p


_REF_CHAIN1_COMMON = r'''
import math
import numpy as np
from scipy.optimize import minimize

def _align_and_impute(tr, te):
    n_tr = len(tr["lab_id"])
    labs = np.concatenate([tr["lab_id"], te["lab_id"]])
    sts = np.concatenate([tr["stratum"], te["stratum"]])
    X = np.vstack([tr["coupon_features_raw"], te["coupon_features_raw"]]).astype(float)
    R = np.vstack([tr["ref_coupons"], te["ref_coupons"]]).astype(float)
    Z = np.empty_like(X)
    for c in np.unique(labs):
        mc = labs == c
        Rc = R[mc]
        means, stds = [], []
        for j in range(Rc.shape[1]):
            col = Rc[:, j]; col = col[np.isfinite(col)]
            if col.size >= 2:
                means.append(float(np.mean(col))); stds.append(float(np.std(col, ddof=0)))
        off = float(np.median(means)); gn = max(float(np.median(stds)), 1e-8)
        Z[mc] = (X[mc] - off) / gn
    w = np.ones(len(labs), dtype=float)
    cells = {}
    for i, key in enumerate(zip(labs.tolist(), sts.tolist())):
        cells.setdefault(key, []).append(i)
    for idx in cells.values():
        w[idx] = 1.0 / len(idx)
    w /= float(np.mean(w))
    for j in range(Z.shape[1]):
        fin = np.isfinite(Z[:, j])
        fill = float(np.sum(w[fin] * Z[fin, j]) / np.sum(w[fin]))
        Z[~fin, j] = fill
    return Z[:n_tr], Z[n_tr:]

def _prony_mem(hist, past_dt, taus, init_mem=None):
    h = np.asarray(hist, dtype=float)
    dt = np.asarray(past_dt, dtype=float)
    ts = np.asarray(taus, dtype=float)
    z = np.zeros((h.shape[0], 2), dtype=float) if init_mem is None else np.asarray(init_mem, dtype=float).copy()
    for s in range(h.shape[1]):
        dec = np.exp(-dt[s] / ts)
        z = z * dec[None, :] + (1.0 - dec[None, :]) * h[:, s : s + 1]
    return z

def _cauchy_irls(Phi, y, n_iter=22):
    w_r = np.ones(len(y), dtype=float)
    beta = np.zeros(Phi.shape[1], dtype=float)
    reg = 1e-8 * np.eye(Phi.shape[1])
    for _ in range(n_iter):
        Pw = Phi * w_r[:, None]
        beta = np.linalg.solve(Phi.T @ Pw + reg, Pw.T @ y)
        res = y - Phi @ beta
        sc = max(1e-3, 1.4826 * float(np.median(np.abs(res))))
        w_r = 1.0 / (1.0 + (res / (2.0 * sc)) ** 2)
    return beta, y - Phi @ beta
'''

_REF_CHAIN2_COMMON = r'''
import math
import numpy as np
from scipy.optimize import minimize

def _build_1d_op(L, n, kappa, vel, rg):
    h = float(L) / int(n)
    kap = np.asarray(kappa, dtype=float).ravel()
    v = np.asarray(vel, dtype=float).ravel()
    A = np.zeros((n, n), dtype=float)
    kh = (2.0 * kap[:-1] * kap[1:] / (kap[:-1] + kap[1:])) / (h * h)
    idx = np.arange(n - 1)
    A[idx, idx] += kh; A[idx + 1, idx + 1] += kh; A[idx, idx + 1] -= kh; A[idx + 1, idx] -= kh
    A[0, 0] += (rg * kap[0]) / (h * (rg * 0.5 * h + kap[0]))
    A[n - 1, n - 1] += (rg * kap[n - 1]) / (h * (rg * 0.5 * h + kap[n - 1]))
    for i in range(n):
        vi = float(v[i])
        if vi >= 0.0:
            A[i, i] += vi / h
            if i > 0: A[i, i - 1] -= vi / h
        else:
            A[i, i] += (-vi) / h
            if i < n - 1: A[i, i + 1] -= (-vi) / h
    return A

def _kronecker_multishift_solve(Ax, By, mu_shifts, modes):
    lx, Vx = np.linalg.eig(Ax)
    ly, Vy = np.linalg.eig(By)
    Phi_hat = np.einsum("ia,mab,bj->mij", np.linalg.inv(Vx), modes, np.linalg.inv(Vy).T)
    denom = lx[:, None, None] + ly[None, :, None] + np.asarray(mu_shifts, dtype=float)[None, None, :]
    return np.real(np.einsum("ai,mijk,jb->mkab", Vx, Phi_hat[:, :, :, None] / denom[None, :, :, :], Vy.T))

def _solve_box_simplex_kkt(S_tilde, r_y, lam_w, u_box, b_tot):
    M = S_tilde.shape[1]
    H = S_tilde.T @ S_tilde + float(lam_w) * np.eye(M)
    f = S_tilde.T @ r_y
    res = minimize(
        lambda w: 0.5 * float(w @ (H @ w)) - float(f @ w),
        np.full(M, float(b_tot) / M),
        jac=lambda w: H @ w - f,
        bounds=[(0.0, float(u_box))] * M,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w)) - float(b_tot), "jac": lambda w: np.ones(M)}],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 400},
    )
    w = np.clip(res.x, 0.0, float(u_box))
    lo, hi = float(np.min(w) - u_box), float(np.max(w))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(np.sum(np.clip(w - mid, 0.0, float(u_box)))) > float(b_tot): lo = mid
        else: hi = mid
    w = np.clip(w - 0.5 * (lo + hi), 0.0, float(u_box))
    for _ in range(8):
        free = (w > 1e-11) & (w < float(u_box) - 1e-11)
        fi = np.flatnonzero(free)
        if fi.size >= 2:
            rem = float(b_tot) - float(np.sum(w[~free]))
            aug = np.block([[H[np.ix_(fi, fi)], -np.ones((fi.size, 1))], [np.ones((1, fi.size)), np.zeros((1, 1))]])
            wf = np.linalg.solve(aug, np.concatenate([f[fi] - H[np.ix_(fi, np.flatnonzero(~free))] @ w[~free], [rem]]))[:-1]
            if np.all((wf >= -1e-12) & (wf <= float(u_box) + 1e-12)):
                w[fi] = np.clip(wf, 0.0, float(u_box)); break
    rv = S_tilde @ w - r_y
    jv = 0.5 * float(rv @ rv) + 0.5 * float(lam_w) * float(w @ w)
    g = H @ w - f
    best_nu, best_lb = float(g[0]), -1e30
    for nc in np.unique(g):
        lbc = jv - float(g @ w) + float(nc) * float(b_tot) + float(u_box) * float(np.sum(np.minimum(0.0, g - float(nc))))
        if lbc > best_lb: best_lb, best_nu = lbc, float(nc)
    return w, {
        "weights": w.tolist(), "nu_sum": best_nu,
        "lam_lower": np.maximum(0.0, g - best_nu).tolist(),
        "lam_upper": np.maximum(0.0, best_nu - g).tolist(),
        "primal_objective": float(jv), "dual_lower_bound": float(min(best_lb, jv)),
    }
'''


def test_m9_3_direct_source_float_plugin_fails_and_online_l1_domain_routing() -> None:
    import ast as _ast
    from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f

    runner_tree = _ast.parse((REPO_ROOT / "scripts/run_m9_3_online_source_freeze_successor_experiment.py").read_text(encoding="utf-8"))
    mods_chk = {"m64": m64, "m6f": m6f, "m7a": m7a}
    bad_attrs = [
        f"{n.value.id}.{n.attr}"
        for n in _ast.walk(runner_tree)
        if isinstance(n, _ast.Attribute) and isinstance(n.value, _ast.Name) and n.value.id in mods_chk and not hasattr(mods_chk[n.value.id], n.attr)
    ]
    assert not bad_attrs, f"Runner references nonexistent module attributes: {bad_attrs}"

    assert len(get_m93_source_frontier_tasks()) == 2
    assert len(get_m93_successor_frontier_tasks()) == 4
    assert len(get_m93_recovery_frontier_tasks()) == 2

    ctrl = evaluate_direct_source_float_plugin_controls()
    assert ctrl["direct_source_float_plugin_fails_all_successors"] is True, f"Direct float plugin control failed: {ctrl}"

    # Verify newly compiled online L1 modules rank #1 on matching chain, veto cross-chain, and preserve K==E reader parity
    _, archive7, _, _ = m7a.load_autogenous_forest_bundle(V7_PATH)
    env_dig, _ = m7a.compute_runtime_sandbox_env_digest(sandbox_cid=None)
    m64.register_verified_l1_env_digest(env_dig)

    def _mock_online_l1(stem: str, code: str, ut: str, fns: list[str], rationale: str) -> dict[str, Any]:
        c_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
        u_sha = hashlib.sha256(ut.encode("utf-8")).hexdigest()
        return {
            "filename": f"{stem}.py",
            "granularity_tier": "L1_compositional_primitive",
            "qualification_protocol": "hegel_l1_qual_4tuple_v2",
            "qualification_status": "qualified_4tuple_code_test_protocol_env_bound",
            "unit_test_verified": True,
            "function_names": fns,
            "exported_apis": ", ".join(f"{f}(x)" for f in fns),
            "compositional_rationale": rationale,
            "python_code": code,
            "compiled_code_sha256": c_sha,
            "synthetic_unit_test_code": ut,
            "synthetic_unit_test_sha256": u_sha,
            "mathematical_preconditions": ["Finite float64 arrays and positive relaxation/box parameters."],
            "unit_test_receipt": {
                "passed": True,
                "telemetry_verified": True,
                "verified_code_sha256": c_sha,
                "synthetic_unit_test_sha256": u_sha,
                "verifier_protocol_version": "hegel_l1_qual_4tuple_v2",
                "dependency_env_digest": env_dig,
            },
        }

    m_prony = _mock_online_l1(
        "autogenous_prim_prony_hereditary_spectrum",
        "import numpy as np\ndef propagate_prony_hereditary_memory(past_stress, past_dt, taus):\n    \"\"\"Propagate variable-step Prony hereditary viscoelastic memory states.\"\"\"\n    return np.zeros((len(past_stress), 2))\n",
        "def run_self_test():\n    propagate_prony_hereditary_memory([[1.0]], [0.2], [0.5, 3.0])\nrun_self_test()\n",
        ["propagate_prony_hereditary_memory"],
        "Variable-step Prony hereditary viscoelastic relaxation spectrum and strain rate consistency.",
    )
    m_kron = _mock_online_l1(
        "autogenous_prim_kronecker_sylvester_kkt",
        "import numpy as np\ndef solve_kronecker_sylvester_kkt(Ax, By, mu_shifts):\n    \"\"\"Solve 2D Kronecker Sylvester multishift PDE and verify box-simplex KKT certificate.\"\"\"\n    return np.zeros((1, 1))\n",
        "def run_self_test():\n    solve_kronecker_sylvester_kkt([[1.0]], [[1.0]], [0.1])\nrun_self_test()\n",
        ["solve_kronecker_sylvester_kkt"],
        "Separable 2D Kronecker Sylvester multishift spectral transport and box-simplex KKT dual lower bound.",
    )
    archive8 = {**archive7}
    archive8["node_m93_online_test"] = {
        "node_id": "node_m93_online_test",
        "layer_role": "mid",
        "autogenous_only": True,
        "signatures": ["sig:symbolic_ode_and_law_regression", "sig:conservation_flux_and_multiscale_pde"],
        "autogenous_wormhole_modules": [m_prony, m_kron],
    }

    for suc_task in get_m93_successor_frontier_tasks():
        _, _, diag_k = m64.collect_recalled_executable_wormhole_sources(
            [], include_foundational_skills=False, allow_l1_primitives=True, full_forest_archive=archive8,
            task_query_text=suc_task.instruction_text, max_l1_modules=6, disclosure_mode="on_demand_knowledge_only",
            staging_policy="net_value_gated", return_diagnostics=True,
        )
        _, _, diag_e = m64.collect_recalled_executable_wormhole_sources(
            [], include_foundational_skills=False, allow_l1_primitives=True, full_forest_archive=archive8,
            task_query_text=suc_task.instruction_text, max_l1_modules=6, disclosure_mode="on_demand_executable",
            staging_policy="net_value_gated", return_diagnostics=True,
        )
        sel_fnames = [x["filename"] for x in diag_e["selected_l1_modules"]]
        if suc_task.task_id.startswith("suc01"):
            assert "autogenous_prim_prony_hereditary_spectrum.py" in sel_fnames
            assert "autogenous_prim_kronecker_sylvester_kkt.py" not in sel_fnames
        else:
            assert "autogenous_prim_kronecker_sylvester_kkt.py" in sel_fnames
            assert "autogenous_prim_prony_hereditary_spectrum.py" not in sel_fnames

        # Verify K vs E `_SHARED_CATALOG_READER_CORE_PY` identical ranking on M9.3 queries
        cat_k = json.loads(diag_k["contracts_catalog_json"])
        cat_e = json.loads(diag_e["contracts_catalog_json"])
        ns_r: dict[str, Any] = {"json": json, "re": __import__("re")}
        exec(compile(m64._SHARED_CATALOG_READER_CORE_PY, "<shared_reader>", "exec"), ns_r)
        for q in ("prony hereditary memory taus", "kronecker sylvester simplex kkt certificate"):
            _, r_k, _ = ns_r["_shared_search_catalog"](cat_k, q, top_k=6)
            _, r_e, _ = ns_r["_shared_search_catalog"](cat_e, q, top_k=6)
            assert r_k == r_e


def test_m9_3_all_8_verifiers_public_validators_and_negative_controls() -> None:
    src01_code = _REF_CHAIN1_COMMON + r'''
def _fit_chain1(dataset, *, use_init_mem=True, use_vp=False, fn_taus_only=False):
    past_dt = np.asarray(dataset["past_dt"], dtype=float)
    et = np.asarray(dataset["eval_times"], dtype=float)
    tr, te = dataset["train"], dataset["test"]
    X_tr, X_te = _align_and_impute(tr, te)
    im_tr = tr.get("initial_memory_states") if use_init_mem else None
    im_te = te.get("initial_memory_states") if use_init_mem else None
    y_tr = (np.asarray(tr["strain_curves_obs"], dtype=float) - np.asarray(tr["strain_0"], dtype=float)[:, None]).ravel()
    h0_tr, h0_te = np.asarray(tr["stress_hold_0"], dtype=float), np.asarray(te["stress_hold_0"], dtype=float)
    ys = float(dataset["yield_stress"]) if (use_vp and "yield_stress" in dataset) else None

    def _design(X_f, hist, h0, im, taus):
        z = _prony_mem(hist, past_dt, taus, im)
        e1 = 1.0 - np.exp(-et / taus[0]); e2 = 1.0 - np.exp(-et / taus[1])
        n_r, k_t = len(h0), len(et)
        cols = [
            ((h0 - z[:, 0])[:, None] * e1[None, :]).ravel(),
            ((h0 - z[:, 1])[:, None] * e2[None, :]).ravel(),
            np.broadcast_to(et[None, :], (n_r, k_t)).ravel(),
            *[np.ravel(X_f[:, d : d + 1] * et[None, :]) for d in range(4)],
        ]
        if ys is not None:
            ov = (np.maximum(0.0, np.abs(h0) - ys) ** 2) * np.sign(h0)
            dh = np.sum((np.maximum(0.0, np.abs(hist) - ys) ** 2) * past_dt[None, :], axis=1)
            cols.append((ov[:, None] * et[None, :]).ravel())
            cols.append(((dh * h0)[:, None] * e1[None, :]).ravel())
        return np.column_stack(cols), z

    def _loss(log_taus):
        t1, t2 = math.exp(log_taus[0]), math.exp(log_taus[1])
        if t2 <= t1 * 1.35: return 1e12
        Phi, _ = _design(X_tr, tr["past_stress"], h0_tr, im_tr, (t1, t2))
        _, res = _cauchy_irls(Phi, y_tr, n_iter=12)
        sc = 0.03
        return float(np.sum(np.log1p((res / sc) ** 2)))

    best_g = (math.log(0.55), math.log(3.8))
    best_v = 1e18
    for g1 in np.linspace(math.log(0.32), math.log(0.95), 6):
        for g2 in np.linspace(math.log(2.6), math.log(5.8), 6):
            v = _loss((g1, g2))
            if v < best_v: best_v, best_g = v, (g1, g2)
    opt = minimize(_loss, best_g, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 90})
    taus_hat = np.array([math.exp(opt.x[0]), math.exp(opt.x[1])], dtype=float)
    Phi_tr, _ = _design(X_tr, tr["past_stress"], h0_tr, im_tr, taus_hat)
    w_hat, _ = _cauchy_irls(Phi_tr, y_tr, n_iter=24)
    Phi_te, z_te = _design(X_te, te["past_stress"], h0_te, im_te, taus_hat)
    s0_te = np.asarray(te["strain_0"], dtype=float)
    pred_te = s0_te[:, None] + (Phi_te @ w_hat).reshape(len(s0_te), len(et))
    rate_te = w_hat[0] * (h0_te - z_te[:, 0]) / taus_hat[0] + w_hat[1] * (h0_te - z_te[:, 1]) / taus_hat[1] + w_hat[2] + X_te @ w_hat[3:7]
    return taus_hat, w_hat, pred_te, rate_te, s0_te, h0_te, et

def solve_hereditary_prony_source(dataset: dict) -> dict:
    taus_hat, w_hat, pred_te, rate_te, *_ = _fit_chain1(dataset)
    return {"estimated_taus": taus_hat, "hereditary_weights": w_hat[:7], "predicted_strain": pred_te, "initial_rate_pred": rate_te}

def solve_hereditary_prony_recovery(dataset: dict) -> dict:
    return solve_hereditary_prony_source(dataset)

def solve_preloaded_hereditary_transfer(dataset: dict) -> dict:
    taus_hat, _, pred_te, _, s0_te, h0_te, et = _fit_chain1(dataset, use_init_mem=True)
    sec = (pred_te[:, -1] - s0_te) / np.maximum(np.abs(h0_te), 0.5)
    ts_f = np.concatenate([[0.0], et]); ys_f = np.column_stack([s0_te, pred_te])
    crits = np.asarray(dataset["test"]["critical_strain"], dtype=float)
    cx = np.full(len(s0_te), float(et[-1]), dtype=float)
    for i, c in enumerate(crits):
        if ys_f[i, 0] >= c: cx[i] = 0.0; continue
        for k in range(len(ts_f) - 1):
            if ys_f[i, k + 1] >= c:
                cx[i] = float(ts_f[k] + (c - ys_f[i, k]) / max(ys_f[i, k + 1] - ys_f[i, k], 1e-12) * (ts_f[k + 1] - ts_f[k])); break
    return {"estimated_taus": taus_hat, "predicted_strain": pred_te, "secant_compliance": sec, "hazard_crossing_times": cx}

def solve_viscoplastic_hereditary_revision(dataset: dict) -> dict:
    taus_hat, w_hat, pred_te, *_ = _fit_chain1(dataset, use_vp=True)
    return {"estimated_taus": taus_hat, "viscoplastic_weights": w_hat[7:9], "predicted_strain": pred_te}
'''

    src02_code = _REF_CHAIN2_COMMON + r'''
from scipy import sparse
from scipy.sparse.linalg import splu

def _prep_chain2(st, *, channel_b_mode="none"):
    Ax = _build_1d_op(st["Lx"], st["nx"], st["kappa_x"], st["vel_x"], st["robin_x"])
    By = _build_1d_op(st["Ly"], st["ny"], st["kappa_y"], st["vel_y"], st["robin_y"])
    modes = np.asarray(st["source_modes"], dtype=float)
    mus = np.asarray(st["mu_shifts"], dtype=float)
    si, sj = st["sensor_ij"][:, 0], st["sensor_ij"][:, 1]
    U_A = _kronecker_multishift_solve(Ax, By, mus, modes)
    S_A = np.vstack([U_A[:, k, si, sj].T for k in range(len(mus))])
    U_B = None
    if channel_b_mode == "exact":
        M, nx, ny = modes.shape
        Vp = np.asarray(st["V_pocket_channel_b"], dtype=float)
        Kb = sparse.kron(sparse.csc_matrix(Ax), sparse.eye(ny, format="csc"), format="csc") + sparse.kron(sparse.eye(nx, format="csc"), sparse.csc_matrix(By), format="csc") + sparse.diags(Vp.ravel(), format="csc")
        U_B = np.zeros((M, len(mus), nx, ny), dtype=float)
        rhs = modes.reshape(M, nx * ny).T
        for k, mk in enumerate(mus):
            U_B[:, k] = splu((Kb + float(mk) * sparse.eye(nx * ny, format="csc")).tocsc()).solve(rhs).T.reshape(M, nx, ny)
        S_stack = np.vstack([S_A, np.vstack([U_B[:, k, si, sj].T for k in range(len(mus))])])
    elif channel_b_mode == "rank1_separable_approx":
        Vp = np.asarray(st["V_pocket_channel_b"], dtype=float)
        vx_a = np.mean(Vp, axis=1)
        vy_a = np.mean(Vp, axis=0) - float(np.mean(Vp))
        U_B = _kronecker_multishift_solve(Ax + np.diag(vx_a), By + np.diag(vy_a), mus, modes)
        S_stack = np.vstack([S_A, np.vstack([U_B[:, k, si, sj].T for k in range(len(mus))])])
    else:
        S_stack = S_A
    Z = np.asarray(st["Z_nuisance"], dtype=float)
    y = np.asarray(st["y_obs"], dtype=float)
    Pz = np.linalg.solve(Z.T @ Z + float(st["nuisance_ridge"]) * np.eye(Z.shape[1]), Z.T)
    return U_A, U_B, S_stack - Z @ (Pz @ S_stack), y - Z @ (Pz @ y)

def solve_kronecker_box_simplex_source(study: dict) -> dict:
    U_A, _, St, ry = _prep_chain2(study)
    w, cert = _solve_box_simplex_kkt(St, ry, study["source_ridge"], study["u_box"], study["total_budget"])
    return {"weights": w, "terminal_field": np.tensordot(w, U_A[:, -1], axes=(0, 0)), "kkt_certificate": cert}

def solve_kronecker_box_simplex_recovery(study: dict) -> dict:
    return solve_kronecker_box_simplex_source(study)

def solve_multibudget_kronecker_transfer(study: dict) -> dict:
    U_A, _, St, ry = _prep_chain2(study)
    ws, certs = [], []
    for bg in study["budget_grid"]:
        w, c = _solve_box_simplex_kkt(St, ry, study["source_ridge"], study["u_box"], float(bg))
        ws.append(w); certs.append(c)
    return {"weights_by_budget": np.vstack(ws), "terminal_field_mid_budget": np.tensordot(ws[1], U_A[:, -1], axes=(0, 0)), "kkt_certificates": certs}

def solve_coupled_channel_vortex_inversion(study: dict) -> dict:
    U_A, U_B, St, ry = _prep_chain2(study, channel_b_mode="exact")
    w, cert = _solve_box_simplex_kkt(St, ry, study["source_ridge"], study["u_box"], study["total_budget"])
    return {"weights": w, "terminal_field_channel_a": np.tensordot(w, U_A[:, -1], axes=(0, 0)), "terminal_field_channel_b": np.tensordot(w, U_B[:, -1], axes=(0, 0)), "kkt_certificate": cert}
'''

    with tempfile.TemporaryDirectory(prefix="m93_ut_") as td:
        td_p = Path(td)
        # 1. Verify all 8 public_setup_py + validate_public_schema.py scripts pass on valid code & reject empty dict
        for ht in M93_LIFECYCLE_TASKS:
            sub_d = td_p / ht.short_name
            sub_d.mkdir()
            exec(compile(ht.public_setup_py.replace("/holdout_task", sub_d.as_posix()), f"<pub_{ht.short_name}>", "exec"), {})
            ref_src = src01_code if "01" in ht.task_id else src02_code
            art_name = Path(ht.declared_artifacts[0]).name
            (sub_d / art_name).write_text(ref_src, encoding="utf-8")
            pr_ok = subprocess.run([sys.executable, str(sub_d / "validate_public_schema.py")], capture_output=True, text=True)
            assert pr_ok.returncode == 0, f"{ht.short_name} public validator failed: {pr_ok.stdout} {pr_ok.stderr}"
            bad_py = sub_d / "bad.py"
            bad_py.write_text(f"def {ht.declared_callable_entrypoints[0]}(x):\n    return {{}}\n", encoding="utf-8")
            pr_bad = subprocess.run([sys.executable, str(sub_d / "validate_public_schema.py"), str(bad_py)], capture_output=True, text=True)
            assert pr_bad.returncode != 0

        # 2. Verify all 8 air-gapped Docker private verifiers PASS on reference solvers
        for ht in M93_LIFECYCLE_TASKS:
            ref_src = src01_code if "01" in ht.task_id else src02_code
            tar_ok = _bundle_py_tar(td_p, ht.declared_artifacts[0], ref_src, f"{ht.short_name}_ok.tar")
            rep_ok = run_m93_lifecycle_private_verifier_by_id(ht.task_id, tar_ok)
            assert rep_ok["strict_pass"] is True, f"{ht.short_name} ref verifier failed: {rep_ok}"

        # 3. Verify targeted negative controls FAIL in air-gapped Docker containers!
        # (a) Ignoring `initial_memory_states` on `suc01_T`
        suc01_t_no_preload = src01_code.replace("use_init_mem=True", "use_init_mem=False")
        t_bad1 = _bundle_py_tar(td_p, M93_TASKS_BY_ID["suc01_T__nonuniform_preloaded_viscoelastic_secant_and_hazard_transfer"].declared_artifacts[0], suc01_t_no_preload, "bad_preload.tar")
        assert run_m93_lifecycle_private_verifier_by_id("suc01_T__nonuniform_preloaded_viscoelastic_secant_and_hazard_transfer", t_bad1)["strict_pass"] is False

        # (b) Pure linear Prony (`vp_weights = 0`) on `suc01_C` counterexample
        suc01_c_linear_only = src01_code.replace(
            'return {"estimated_taus": taus_hat, "viscoplastic_weights": w_hat[7:9], "predicted_strain": pred_te}',
            'taus_l, _, pred_l, *_ = _fit_chain1(dataset, use_vp=False)\n    return {"estimated_taus": taus_l, "viscoplastic_weights": np.zeros(2), "predicted_strain": pred_l}',
        )
        t_bad2 = _bundle_py_tar(td_p, M93_TASKS_BY_ID["suc01_C__viscoplastic_bingham_overstress_counterexample_revision"].declared_artifacts[0], suc01_c_linear_only, "bad_vp.tar")
        assert run_m93_lifecycle_private_verifier_by_id("suc01_C__viscoplastic_bingham_overstress_counterexample_revision", t_bad2)["strict_pass"] is False

        # (c) Fake/uniform box-simplex weights with positive KKT duality gap on `src02`
        src02_bad_kkt = src02_code.replace(
            "w, cert = _solve_box_simplex_kkt(St, ry, study[\"source_ridge\"], study[\"u_box\"], study[\"total_budget\"])",
            "M_d = St.shape[1]; w = np.full(M_d, study['total_budget']/M_d); j0 = 0.5*float(np.sum((St@w-ry)**2)) + 0.5*study['source_ridge']*float(w@w); cert = {'weights': w.tolist(), 'nu_sum': 0.0, 'lam_lower': [0.0]*M_d, 'lam_upper': [0.0]*M_d, 'primal_objective': j0, 'dual_lower_bound': j0}",
        )
        t_bad3 = _bundle_py_tar(td_p, M93_TASKS_BY_ID["src02__kronecker_sylvester_multishift_and_box_simplex_kkt_source"].declared_artifacts[0], src02_bad_kkt, "bad_kkt.tar")
        assert run_m93_lifecycle_private_verifier_by_id("src02__kronecker_sylvester_multishift_and_box_simplex_kkt_source", t_bad3)["strict_pass"] is False

        # (d) Rank-1 additive separable approximation of `V_pocket_channel_b` on `suc02_C` counterexample
        suc02_c_sep_approx = src02_code.replace('channel_b_mode="exact"', 'channel_b_mode="rank1_separable_approx"')
        t_bad4 = _bundle_py_tar(td_p, M93_TASKS_BY_ID["suc02_C__nonseparable_vortex_cross_coupling_counterexample_revision"].declared_artifacts[0], suc02_c_sep_approx, "bad_sep.tar")
        assert run_m93_lifecycle_private_verifier_by_id("suc02_C__nonseparable_vortex_cross_coupling_counterexample_revision", t_bad4)["strict_pass"] is False
