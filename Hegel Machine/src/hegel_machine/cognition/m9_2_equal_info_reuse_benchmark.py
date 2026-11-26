"""Hegel M9.2 Equal-Information On-Demand Reuse, Causal Coupling & Temporal Memory Suite (`Hegel-M92-EqualInfo-4`).

Evaluated under strict single-branch symmetry (direct 1x `run_frontier_episode`, `max_turns = 18` symmetric across
`bare_no_library` [B], `on_demand_knowledge_only` [K], and `on_demand_executable_library` [E]):

Equal-Information Guarantees (`0` artificial solver bans, `100%` symmetric deliverable schema validation):
- Every task writes `/holdout_task/validate_public_schema.py` inside EVERY arm's container (`B`, `K`, `E`) so any agent can
  verify exact deliverable keys, array shapes, `bool[p]` certificate types, finiteness, and cube partition completeness (`0`
  private truth access; `--describe` never acts as a format backchannel).
- Standard SciPy/NumPy algorithms (`splu`, `spsolve`, `spilu`, `cg`, `gmres`, `lsq_linear`, block decomposition) are 100%
  legal and unbanned across all tasks.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

from . import m6_frontier_real_benchmarks as m6f
from .m7_holdout_ood_blind_benchmark import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    HoldoutOODTask,
    ensure_clean_scipy_sandbox_image,
    run_holdout_private_verifier,
)


EQUAL_INFO_M92_TASKS: tuple[HoldoutOODTask, ...] = (
    # =========================================================================
    # EQ01: 3D Multi-Operator Parametric Robin Continuation & Nuisance Profiling
    # =========================================================================
    HoldoutOODTask(
        task_id="eq01__3d_multi_operator_robin_continuation_and_nuisance_tomography",
        short_name="eq01_3d_multi_operator_robin_continuation",
        wave="m92_equal_info",
        scientific_domain="3D Parametric Elliptic PDE Continuation, Reusable Preconditioning & Profiled Nuisance Tomography",
        wormhole_family="fam_multiscale_pde_and_preconditioned_operators",
        declared_artifacts=("/holdout_task/robin_continuation_tomography.py",),
        declared_callable_entrypoints=("solve_robin_continuation_tomography",),
        instruction_text=(
            "Implement `/holdout_task/robin_continuation_tomography.py` exposing:\n"
            "`def solve_robin_continuation_tomography(study: dict) -> dict:`\n\n"
            "3D Multi-Operator Parametric Robin Diffusion-Reaction Continuation & Nuisance-Profiled Tomography Contract:\n"
            "1. PARAMETRIC 3D ROBIN OPERATOR FAMILY (`(nx, ny, nz) = (28, 28, 28) = 21,952` cell-centered unknowns on `[0, Lx] x [0, Ly] x [0, Lz]`):\n"
            "   - `study` provides grid dimensions `'Lx'`, `'Ly'`, `'Lz'`, base/modulation 3D positive arrays `'kappa_base'`, `'kappa_mod1'`, `'kappa_mod2'`\n"
            "     (`shape (nx, ny, nz)` each, base contrast up to `~3,500x`), `'sigma_base'`, `'sigma_mod'` (`shape (nx, ny, nz)`), Robin wall transfer\n"
            "     `'gamma_robin'` (`float > 0`), continuation parameter vector `'thetas'` (`shape (K,)`, `K = 9` distinct states in `[0.15, 1.85]`),\n"
            "     and `'source_modes'` (`shape (M, nx, ny, nz)`, `M = 3`).\n"
            "   - For each continuation index `k in range(K)` with `th = float(thetas[k])`:\n"
            "     `kappa_k = kappa_base + th * kappa_mod1 + (th ** 2) * kappa_mod2`, and `sigma_k = sigma_base + th * sigma_mod`.\n"
            "   - Exact cell-centered finite-volume operator `A_k` (`21,952 x 21,952` real SPD matrix in C-order `ravel()`, `hx=Lx/nx, hy=Ly/ny, hz=Lz/nz`, `vol=hx*hy*hz`):\n"
            "     * Reaction diagonal on cell `c`: `vol * sigma_k[c]`.\n"
            "     * Interior face between adjacent cells `c` and `d` along axis `a in {0, 1, 2}` (`area_a = vol / h_a`): harmonic conductance\n"
            "       `w_cd = (area_a / h_a) * (2.0 * kappa_k[c] * kappa_k[d] / (kappa_k[c] + kappa_k[d]))`, contributing `+w_cd` to diagonals `(c,c),(d,d)` and `-w_cd` to `(c,d),(d,c)`.\n"
            "     * Each of the 6 exterior domain boundary walls along axis `a` on boundary cell `c` adds series half-cell Robin conductance:\n"
            "       `w_rob = area_a * (gamma_robin * kappa_k[c]) / (gamma_robin * 0.5 * h_a + kappa_k[c])` to diagonal `(c, c)`.\n"
            "   - Solve `A_k @ u_{k, m} = vol * source_modes[m].ravel()` for all `k = 0..K-1` and `m = 0..M-1` (`K * M = 27` 3D solves per study;\n"
            "     every solve must satisfy relative residual `||A_k u - b||_2 / ||b||_2 <= 1e-7`, and total runtime across 2 private studies must stay `<= 25.0s`.\n"
            "     All standard NumPy/SciPy methods are allowed; note that factoring 18 separate `21,952`-DOF 3D sparse LU matrices sequentially costs ~35s on CPU\n"
            "     while unpreconditioned CG stalls on the `3,500x` conductivity contrast, whereas reusing a spectrally equivalent coarse/multigrid preconditioner\n"
            "     built at a reference `theta` solves all `K` operators rapidly).\n"
            "2. STACKED SENSOR TOMOGRAPHY & RIDGE-WHITENED NUISANCE AMPLITUDE PROFILING:\n"
            "   - Let `S_stack` (`shape (K * N_obs, M)`) stack `[u_{k, 0}[sensor_ijk], ..., u_{k, M-1}[sensor_ijk]]` vertically for `k = 0..K-1`, where\n"
            "     `sensor_ijk` has shape `(N_obs, 3)`.\n"
            "   - `study` provides `'Z_nuisance'` (`shape (K * N_obs, Q_nuis)`), `'Y_obs'` (`shape (K * N_obs, S_snap)`), `'nuisance_ridge'` (`lam_z > 0`),\n"
            "     and `'source_ridge'` (`lam_s > 0`).\n"
            "   - Define the ridge nuisance-projected residual operator `P_perp = I - Z @ (Z^T Z + lam_z I)^{-1} Z^T`. For each candidate mode `m in 0..M-1` and\n"
            "     snapshot `q in 0..S_snap-1`, compute the exact 1-mode profiled least-squares amplitude:\n"
            "     `profiled_amplitudes[m, q] = (s_m^T P_perp y_q) / (s_m^T P_perp s_m)` (`rel L2 error <= 1e-5`).\n"
            "   - Also solve the joint `(Z_nuisance, S_stack)` block ridge system minimizing\n"
            "     `0.5 * ||Z G + S_stack W - Y_obs||_F^2 + 0.5 * lam_z * ||G||_F^2 + 0.5 * lam_s * ||W||_F^2` for `'joint_source_weights'` `W` (`shape (M, S_snap)`, `rel L2 <= 0.03`).\n"
            "- Return:\n"
            "  - `'terminal_fields'`: float array `(M, nx, ny, nz)` for the last continuation state `k = K - 1`,\n"
            "  - `'continuation_sensor_predictions'`: float array `(K * N_obs, M)` equal to `S_stack`,\n"
            "  - `'profiled_amplitudes'`: float array `(M, S_snap)`,\n"
            "  - `'joint_source_weights'`: float array `(M, S_snap)`.\n"
            "- Public development study: `/holdout_task/public_eq01_studies.pkl`. Shared public output schema validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        public_setup_py=r'''
import importlib.util, json, math, pickle, sys
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, splu

def _fast_2grid_prec(A, nx, ny, nz):
    N = nx * ny * nz
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    agg = ((np.arange(nx)[:, None, None] // 2) * (cy * cz) + (np.arange(ny)[None, :, None] // 2) * cz + (np.arange(nz)[None, None, :] // 2)).ravel()
    P = sparse.coo_matrix((np.ones(N, dtype=float), (np.arange(N), agg)), shape=(N, cx * cy * cz)).tocsr()
    R = P.T.tocsr()
    inv_d = 1.0 / np.maximum(A.diagonal(), 1e-12)
    lu_c = splu((R @ A @ P).tocsc())
    w = 0.67
    def _mv(r):
        x = w * inv_d * r
        x = x + P @ lu_c.solve(R @ (r - A @ x))
        x = x + w * inv_d * (r - A @ x)
        return x
    return LinearOperator((N, N), matvec=_mv, dtype=float)

def _assemble_3d(Lx, Ly, Lz, kappa, sigma, gamma_robin):
    nx, ny, nz = kappa.shape
    steps = (Lx / nx, Ly / ny, Lz / nz)
    vol = steps[0] * steps[1] * steps[2]
    N = kappa.size
    ids = np.arange(N).reshape(nx, ny, nz)
    diag = vol * sigma.ravel().astype(float)
    rows, cols, data = [], [], []
    kf = kappa.ravel()
    for axis, h_a in enumerate(steps):
        area_a = vol / h_a
        i0 = np.take(ids, np.arange(kappa.shape[axis] - 1), axis=axis).ravel()
        i1 = np.take(ids, np.arange(1, kappa.shape[axis]), axis=axis).ravel()
        w = (area_a / h_a) * (2.0 * kf[i0] * kf[i1] / (kf[i0] + kf[i1]))
        diag[i0] += w; diag[i1] += w
        rows.extend([i0, i1]); cols.extend([i1, i0]); data.extend([-w, -w])
        for end in (0, -1):
            ib = np.take(ids, end, axis=axis).ravel()
            kb = kf[ib]
            diag[ib] += area_a * (gamma_robin * kb) / (gamma_robin * 0.5 * h_a + kb)
    rows.append(ids.ravel()); cols.append(ids.ravel()); data.append(diag)
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)), vol

def _make_study(seed):
    rng = np.random.default_rng(seed)
    nx = ny = nz = 28
    Lx, Ly, Lz = 1.1, 1.0, 0.95
    xs = (np.arange(nx) + 0.5) * (Lx / nx)
    ys = (np.arange(ny) + 0.5) * (Ly / ny)
    zs = (np.arange(nz) + 0.5) * (Lz / nz)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    inc1 = ((XX - 0.34 * Lx) ** 2 + (YY - 0.44 * Ly) ** 2 + (ZZ - 0.50 * Lz) ** 2) < 0.042
    inc2 = ((XX - 0.69 * Lx) ** 2 + (YY - 0.61 * Ly) ** 2 + (ZZ - 0.40 * Lz) ** 2) < 0.036
    low_b = (np.abs(XX - 0.52 * Lx) < 0.06) & (YY > 0.2 * Ly) & (YY < 0.8 * Ly)
    kappa_base = np.where(low_b, 0.025, 1.0) + 75.0 * inc1.astype(float) + 52.0 * inc2.astype(float)
    kappa_mod1 = 0.55 * kappa_base * (0.5 + 0.5 * np.sin(2.0 * math.pi * XX / Lx) * np.cos(math.pi * YY / Ly))
    kappa_mod2 = 0.30 * kappa_base * (0.5 + 0.5 * np.cos(math.pi * ZZ / Lz))
    sigma_base = 0.08 + 0.05 * np.cos(math.pi * XX / Lx) ** 2
    sigma_mod = 0.03 * (0.5 + 0.5 * np.sin(math.pi * YY / Ly) ** 2)
    gamma_robin = 2.2
    thetas = np.linspace(0.15, 1.85, 9)

    M = 3
    ctrs = np.array([[0.25 * Lx, 0.30 * Ly, 0.35 * Lz], [0.74 * Lx, 0.68 * Ly, 0.45 * Lz], [0.48 * Lx, 0.52 * Ly, 0.72 * Lz]])
    source_modes = np.zeros((M, nx, ny, nz), dtype=float)
    for m in range(M):
        r2 = (XX - ctrs[m, 0]) ** 2 + (YY - ctrs[m, 1]) ** 2 + (ZZ - ctrs[m, 2]) ** 2
        source_modes[m] = 35.0 * np.exp(-r2 / (2.0 * (0.095 ** 2)))

    N_obs = 90
    fc = rng.choice(nx * ny * nz, size=N_obs, replace=False)
    sensor_ijk = np.column_stack([fc // (ny * nz), (fc // nz) % ny, fc % nz])

    th_m = float(thetas[len(thetas) // 2])
    A_m, _ = _assemble_3d(Lx, Ly, Lz, kappa_base + th_m * kappa_mod1 + (th_m ** 2) * kappa_mod2, sigma_base + th_m * sigma_mod, gamma_robin)
    M_gt = _fast_2grid_prec(A_m, nx, ny, nz)
    S_blocks = []
    term_fields = np.zeros((M, nx, ny, nz), dtype=float)
    for k_i, th in enumerate(thetas):
        kk = kappa_base + float(th) * kappa_mod1 + (float(th) ** 2) * kappa_mod2
        sk = sigma_base + float(th) * sigma_mod
        Ak, vol = _assemble_3d(Lx, Ly, Lz, kk, sk, gamma_robin)
        uk = np.column_stack([cg(Ak, vol * source_modes[m].ravel(), M=M_gt, rtol=1e-12, maxiter=250)[0] for m in range(M)])
        S_blocks.append(uk[fc, :])
        if k_i == len(thetas) - 1:
            term_fields = uk.T.reshape(M, nx, ny, nz)
    S_stack = np.vstack(S_blocks)

    K_tot = S_stack.shape[0]
    Q_nuis = 6
    S_snap = 4
    Z_nuis = rng.normal(size=(K_tot, Q_nuis))
    G_true = rng.normal(scale=0.6, size=(Q_nuis, S_snap))
    W_true = rng.uniform(-1.5, 1.5, size=(M, S_snap))
    Y_obs = Z_nuis @ G_true + S_stack @ W_true + rng.normal(scale=1e-3, size=(K_tot, S_snap))
    lam_z = 0.25
    lam_s = 1e-4

    P_z = np.eye(K_tot) - Z_nuis @ np.linalg.solve(Z_nuis.T @ Z_nuis + lam_z * np.eye(Q_nuis), Z_nuis.T)
    PS = P_z @ S_stack
    PY = P_z @ Y_obs
    prof_gt = (S_stack.T @ PY) / np.sum(S_stack * PS, axis=0)[:, None]
    H_blk = np.block([[Z_nuis.T @ Z_nuis + lam_z * np.eye(Q_nuis), Z_nuis.T @ S_stack], [S_stack.T @ Z_nuis, S_stack.T @ S_stack + lam_s * np.eye(M)]])
    W_gt = np.linalg.solve(H_blk, np.vstack([Z_nuis.T @ Y_obs, S_stack.T @ Y_obs]))[Q_nuis:, :]

    return {
        "study": {
            "Lx": Lx, "Ly": Ly, "Lz": Lz,
            "kappa_base": kappa_base, "kappa_mod1": kappa_mod1, "kappa_mod2": kappa_mod2,
            "sigma_base": sigma_base, "sigma_mod": sigma_mod, "gamma_robin": gamma_robin,
            "thetas": thetas, "source_modes": source_modes, "sensor_ijk": sensor_ijk,
            "Z_nuisance": Z_nuis, "Y_obs": Y_obs, "nuisance_ridge": lam_z, "source_ridge": lam_s,
        },
        "gt": {
            "terminal_fields": term_fields,
            "continuation_sensor_predictions": S_stack,
            "profiled_amplitudes": prof_gt,
            "joint_source_weights": W_gt,
        },
    }

pub = [_make_study(9201)]
with open("/holdout_task/public_eq01_studies.pkl", "wb") as fh:
    pickle.dump(pub, fh)

open("/holdout_task/validate_public_schema.py", "w").write("""import importlib.util, json, pickle, sys
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np

cand_p = sys.argv[1] if len(sys.argv) > 1 else "/holdout_task/robin_continuation_tomography.py"
spec = importlib.util.spec_from_file_location("cand_eq01", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
st = pickle.load(open("/holdout_task/public_eq01_studies.pkl", "rb"))[0]["study"]
out = mod.solve_robin_continuation_tomography(st)
req = {"terminal_fields": (3, 28, 28, 28), "continuation_sensor_predictions": (9 * len(st["sensor_ijk"]), 3), "profiled_amplitudes": (3, st["Y_obs"].shape[1]), "joint_source_weights": (3, st["Y_obs"].shape[1])}
errs = []
if not isinstance(out, dict):
    errs.append("output_not_dict")
else:
    for k, shp in req.items():
        if k not in out:
            errs.append(f"missing:{k}")
        else:
            arr = np.asarray(out[k], dtype=float)
            if arr.shape != shp or not np.all(np.isfinite(arr)):
                errs.append(f"bad_shape_or_nan:{k}:{arr.shape}!={shp}")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
""")
''',
        private_verifier_py=r'''
import json, math, sys, time
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, splu

def _fast_2grid_prec(A, nx, ny, nz):
    N = nx * ny * nz
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    agg = ((np.arange(nx)[:, None, None] // 2) * (cy * cz) + (np.arange(ny)[None, :, None] // 2) * cz + (np.arange(nz)[None, None, :] // 2)).ravel()
    P = sparse.coo_matrix((np.ones(N, dtype=float), (np.arange(N), agg)), shape=(N, cx * cy * cz)).tocsr()
    R = P.T.tocsr()
    inv_d = 1.0 / np.maximum(A.diagonal(), 1e-12)
    lu_c = splu((R @ A @ P).tocsc())
    w = 0.67
    def _mv(r):
        x = w * inv_d * r
        x = x + P @ lu_c.solve(R @ (r - A @ x))
        x = x + w * inv_d * (r - A @ x)
        return x
    return LinearOperator((N, N), matvec=_mv, dtype=float)

sys.path.insert(0, "/holdout_task")
import robin_continuation_tomography

def _assemble_3d(Lx, Ly, Lz, kappa, sigma, gamma_robin):
    nx, ny, nz = kappa.shape
    steps = (Lx / nx, Ly / ny, Lz / nz)
    vol = steps[0] * steps[1] * steps[2]
    N = kappa.size
    ids = np.arange(N).reshape(nx, ny, nz)
    diag = vol * sigma.ravel().astype(float)
    rows, cols, data = [], [], []
    kf = kappa.ravel()
    for axis, h_a in enumerate(steps):
        area_a = vol / h_a
        i0 = np.take(ids, np.arange(kappa.shape[axis] - 1), axis=axis).ravel()
        i1 = np.take(ids, np.arange(1, kappa.shape[axis]), axis=axis).ravel()
        w = (area_a / h_a) * (2.0 * kf[i0] * kf[i1] / (kf[i0] + kf[i1]))
        diag[i0] += w; diag[i1] += w
        rows.extend([i0, i1]); cols.extend([i1, i0]); data.extend([-w, -w])
        for end in (0, -1):
            ib = np.take(ids, end, axis=axis).ravel()
            kb = kf[ib]
            diag[ib] += area_a * (gamma_robin * kb) / (gamma_robin * 0.5 * h_a + kb)
    rows.append(ids.ravel()); cols.append(ids.ravel()); data.append(diag)
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)), vol

def _make_priv(seed):
    rng = np.random.default_rng(seed)
    nx = ny = nz = 28
    Lx, Ly, Lz = 1.05, 1.12, 0.92
    xs = (np.arange(nx) + 0.5) * (Lx / nx)
    ys = (np.arange(ny) + 0.5) * (Ly / ny)
    zs = (np.arange(nz) + 0.5) * (Lz / nz)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    inc1 = ((XX - 0.36 * Lx) ** 2 + (YY - 0.42 * Ly) ** 2 + (ZZ - 0.48 * Lz) ** 2) < 0.044
    inc2 = ((XX - 0.67 * Lx) ** 2 + (YY - 0.63 * Ly) ** 2 + (ZZ - 0.41 * Lz) ** 2) < 0.038
    low_b = (np.abs(XX - 0.51 * Lx) < 0.055) & (YY > 0.18 * Ly) & (YY < 0.82 * Ly)
    kappa_base = np.where(low_b, 0.024, 1.0) + 78.0 * inc1.astype(float) + 54.0 * inc2.astype(float)
    kappa_mod1 = 0.56 * kappa_base * (0.5 + 0.5 * np.sin(2.0 * math.pi * XX / Lx) * np.cos(math.pi * YY / Ly))
    kappa_mod2 = 0.32 * kappa_base * (0.5 + 0.5 * np.cos(math.pi * ZZ / Lz))
    sigma_base = 0.085 + 0.045 * np.cos(math.pi * XX / Lx) ** 2
    sigma_mod = 0.032 * (0.5 + 0.5 * np.sin(math.pi * YY / Ly) ** 2)
    gamma_robin = 2.35
    thetas = np.linspace(0.15, 1.85, 9)

    M = 3
    ctrs = np.array([[0.24 * Lx, 0.32 * Ly, 0.34 * Lz], [0.75 * Lx, 0.66 * Ly, 0.46 * Lz], [0.49 * Lx, 0.50 * Ly, 0.74 * Lz]]) + rng.uniform(-0.025, 0.025, size=(M, 3))
    source_modes = np.zeros((M, nx, ny, nz), dtype=float)
    for m in range(M):
        r2 = (XX - ctrs[m, 0]) ** 2 + (YY - ctrs[m, 1]) ** 2 + (ZZ - ctrs[m, 2]) ** 2
        source_modes[m] = 35.0 * np.exp(-r2 / (2.0 * (0.095 ** 2)))

    N_obs = 92
    fc = rng.choice(nx * ny * nz, size=N_obs, replace=False)
    sensor_ijk = np.column_stack([fc // (ny * nz), (fc // nz) % ny, fc % nz])

    th_m = float(thetas[len(thetas) // 2])
    A_m, _ = _assemble_3d(Lx, Ly, Lz, kappa_base + th_m * kappa_mod1 + (th_m ** 2) * kappa_mod2, sigma_base + th_m * sigma_mod, gamma_robin)
    M_gt = _fast_2grid_prec(A_m, nx, ny, nz)
    S_blocks = []
    A_last, vol_last = None, 0.0
    for k_i, th in enumerate(thetas):
        kk = kappa_base + float(th) * kappa_mod1 + (float(th) ** 2) * kappa_mod2
        sk = sigma_base + float(th) * sigma_mod
        Ak, vol = _assemble_3d(Lx, Ly, Lz, kk, sk, gamma_robin)
        uk = np.column_stack([cg(Ak, vol * source_modes[m].ravel(), M=M_gt, rtol=1e-12, maxiter=250)[0] for m in range(M)])
        S_blocks.append(uk[fc, :])
        if k_i == len(thetas) - 1:
            A_last, vol_last = Ak, vol
    S_stack = np.vstack(S_blocks)

    K_tot = S_stack.shape[0]
    Q_nuis = 6
    S_snap = 4
    Z_nuis = rng.normal(size=(K_tot, Q_nuis))
    G_true = rng.normal(scale=0.6, size=(Q_nuis, S_snap))
    W_true = rng.uniform(-1.5, 1.5, size=(M, S_snap))
    Y_obs = Z_nuis @ G_true + S_stack @ W_true + rng.normal(scale=1e-3, size=(K_tot, S_snap))
    lam_z = 0.25
    lam_s = 1e-4

    P_z = np.eye(K_tot) - Z_nuis @ np.linalg.solve(Z_nuis.T @ Z_nuis + lam_z * np.eye(Q_nuis), Z_nuis.T)
    PS = P_z @ S_stack
    PY = P_z @ Y_obs
    prof_gt = (S_stack.T @ PY) / np.sum(S_stack * PS, axis=0)[:, None]
    H_blk = np.block([[Z_nuis.T @ Z_nuis + lam_z * np.eye(Q_nuis), Z_nuis.T @ S_stack], [S_stack.T @ Z_nuis, S_stack.T @ S_stack + lam_s * np.eye(M)]])
    W_gt = np.linalg.solve(H_blk, np.vstack([Z_nuis.T @ Y_obs, S_stack.T @ Y_obs]))[Q_nuis:, :]

    study = {
        "Lx": Lx, "Ly": Ly, "Lz": Lz,
        "kappa_base": kappa_base, "kappa_mod1": kappa_mod1, "kappa_mod2": kappa_mod2,
        "sigma_base": sigma_base, "sigma_mod": sigma_mod, "gamma_robin": gamma_robin,
        "thetas": thetas, "source_modes": source_modes, "sensor_ijk": sensor_ijk,
        "Z_nuisance": Z_nuis, "Y_obs": Y_obs, "nuisance_ridge": lam_z, "source_ridge": lam_s,
    }
    return study, A_last, vol_last, S_stack, prof_gt, W_gt

cases = [_make_priv(99401), _make_priv(99402)]
t0 = time.monotonic()
valid = True
max_pde_res = 0.0
max_sens_err = 0.0
max_prof_err = 0.0
max_jw_err = 0.0
diag_rows = []
for sd, (st, A_last, vol_last, S_gt, prof_gt, W_gt) in zip((99401, 99402), cases):
    out = robin_continuation_tomography.solve_robin_continuation_tomography(st)
    if not isinstance(out, dict) or not {"terminal_fields", "continuation_sensor_predictions", "profiled_amplitudes", "joint_source_weights"} <= set(out.keys()):
        valid = False; diag_rows.append({"seed": sd, "error": "missing_keys"}); continue
    tf = np.asarray(out["terminal_fields"], dtype=float)
    sp = np.asarray(out["continuation_sensor_predictions"], dtype=float)
    pa = np.asarray(out["profiled_amplitudes"], dtype=float)
    jw = np.asarray(out["joint_source_weights"], dtype=float)
    if tf.shape != (3, 28, 28, 28) or sp.shape != S_gt.shape or pa.shape != prof_gt.shape or jw.shape != W_gt.shape:
        valid = False; diag_rows.append({"seed": sd, "error": "shape_mismatch"}); continue
    if not (np.all(np.isfinite(tf)) and np.all(np.isfinite(sp)) and np.all(np.isfinite(pa)) and np.all(np.isfinite(jw))):
        valid = False; diag_rows.append({"seed": sd, "error": "non_finite"}); continue
    m_res = 0.0
    for m in range(3):
        b_m = vol_last * st["source_modes"][m].ravel()
        r_m = float(np.linalg.norm(A_last @ tf[m].ravel() - b_m) / np.linalg.norm(b_m))
        m_res = max(m_res, r_m)
    s_err = float(np.linalg.norm(sp - S_gt) / np.linalg.norm(S_gt))
    p_err = float(np.linalg.norm(pa - prof_gt) / np.linalg.norm(prof_gt))
    j_err = float(np.linalg.norm(jw - W_gt) / np.linalg.norm(W_gt))
    max_pde_res = max(max_pde_res, m_res)
    max_sens_err = max(max_sens_err, s_err)
    max_prof_err = max(max_prof_err, p_err)
    max_jw_err = max(max_jw_err, j_err)
    diag_rows.append({"seed": sd, "terminal_pde_rel_res": round(m_res, 10), "sensor_stack_rel_l2": round(s_err, 8), "profiled_rel_l2": round(p_err, 8), "joint_rel_l2": round(j_err, 6)})

student_elapsed = time.monotonic() - t0
strict_pass = bool(
    valid
    and len(diag_rows) == 2
    and max_pde_res <= 1e-7
    and max_sens_err <= 1e-5
    and max_prof_err <= 1e-5
    and max_jw_err <= 0.03
    and student_elapsed <= 25.0
)
print(json.dumps({
    "strict_pass": strict_pass,
    "reward": 1.0 if strict_pass else 0.0,
    "all_valid": valid,
    "max_terminal_pde_rel_res": (round(float(max_pde_res), 10) if math.isfinite(max_pde_res) else -1.0),
    "max_sensor_stack_rel_l2": (round(float(max_sens_err), 8) if math.isfinite(max_sens_err) else -1.0),
    "max_profiled_rel_l2": (round(float(max_prof_err), 8) if math.isfinite(max_prof_err) else -1.0),
    "max_joint_weights_rel_l2": (round(float(max_jw_err), 6) if math.isfinite(max_jw_err) else -1.0),
    "student_elapsed_sec": round(float(student_elapsed), 4),
    "instance_diagnostics": diag_rows,
}, allow_nan=False))
''',
    ),

    # =========================================================================
    # EQ02: End-to-End Causally Coupled Panel -> Interferometry+Cauchy -> Connected Box-L0
    # =========================================================================
    HoldoutOODTask(
        task_id="eq02__coupled_panel_interferometry_cauchy_and_connected_box_l0",
        short_name="eq02_coupled_interferometry_cauchy_box_l0",
        wave="m92_equal_info",
        scientific_domain="Causally Coupled Multi-Center Panel Deconfounding, Interferometric Demodulation & Connected Box-L0 Certification",
        wormhole_family="fam_sparse_cert_and_combinatorial_pruning",
        declared_artifacts=("/holdout_task/coupled_interferometry_certified_l0.py",),
        declared_callable_entrypoints=("solve_coupled_interferometry_l0",),
        instruction_text=(
            "Implement `/holdout_task/coupled_interferometry_certified_l0.py` exposing:\n"
            "`def solve_coupled_interferometry_l0(study: dict) -> dict:`\n\n"
            "End-to-End Causally Coupled Multi-Center Panel, Optical Phase Demodulation, Cauchy Calibration, & Connected Box-L0 Contract:\n"
            "1. STAGE 1 (`Multi-Center Reference-Panel Alignment & Cell-Balanced Imputation`):\n"
            "   - `study['panel']` contains `'center_id'` (`shape (N,)`, `C = 5` centers), `'stratum'` (`shape (N,)`, `S = 3` strata `0, 1, 2`,\n"
            "     all `(center, stratum)` cells nonempty), `'X_panel_raw'` (`shape (N, D)`, `D = 8` real float64 biomarkers with ~2.5% `NaN`s),\n"
            "     and `'controls'` (`shape (N, Q)`, `Q = 18` reference standards with ~5% `NaN`s, drawn from `N(0, I_18)` per center).\n"
            "   - In each center `c`, align `Z_aligned = (X_panel_raw - offset[c]) / gain[c]` where `offset[c]` is the median across valid control\n"
            "     columns (`>= 2` finite obs) of the column mean and `gain[c]` is the median across valid control columns of the column population\n"
            "     std (`ddof=0`). Compute inverse-cell-frequency weights `w_cell` (`mean == 1.0`) giving equal total weight to each `(center, stratum)`\n"
            "     cell, and impute each `NaN` in `Z_aligned` with its `w_cell`-weighted finite column mean -> `'aligned_panel'` (`(N, D)`, `MAE <= 0.11`).\n"
            "2. STAGE 2 (`Subproblem-Scoped complex128 Optical Phase Demodulation + Real float64 Cauchy IRLS Deconfounding`):\n"
            "   - Stage 2a (`complex128` optical carrier demodulation): `study['interferometry']` stores `'signal_phasor'` (`complex128`, `shape (N, 2)`)\n"
            "     and `'ref_phasor'` (`complex128`, `shape (N, 2)`). Demodulate real float64 phase `demod_phase = np.angle(signal_phasor * np.conj(ref_phasor))` (`(N, 2)`).\n"
            "   - Stage 2b (Real `float64` Cauchy IRLS calibration fit): Build the real float64 calibration design\n"
            "     `H_cal = np.column_stack([np.ones(N), (stratum == 1).astype(float), (stratum == 2).astype(float), aligned_panel, demod_phase])` (`shape (N, 13)`).\n"
            "     `study['y_assay_raw']` (`shape (N,)`, real float64) obeys `H_cal_true @ theta_cal + sparse_signal + noise`, corrupted on ~12% of calibration\n"
            "     rows (`cal_idx = study['calibration_indices']`, where `sparse_signal == 0`) by heavy Cauchy sensor glitches (`+-11.0`)! Fit `theta_cal_hat`\n"
            "     (`shape (13,)`) on `(H_cal[cal_idx], y_assay_raw[cal_idx])` via Cauchy/Lorentzian IRLS (`rel L2 <= 0.14`).\n"
            "   - CAUSAL HANDOFF TO STAGE 3: On the target discovery cohort `s3_idx = study['stage3_indices']`, subtract the fitted calibration baseline:\n"
            "     `y_deconfounded = y_assay_raw[s3_idx] - H_cal[s3_idx] @ theta_cal_hat` (`shape (n_s,)`).\n"
            "     (WARNING: Biased Stage-1 alignment or non-robust Stage-2 OLS corrupts `y_deconfounded` and breaks Stage-3 sparse recovery!)\n"
            "3. STAGE 3 (`Single Connected Block p in {18, 19} Certified Symmetric-Box L0-Ridge` on `(X_cand, y_deconfounded)`):\n"
            "   - Using `X_cand = study['X_cand']` (`shape (n_s, p)`, `p in {18, 19}`, single connected column-incidence component `len(blocks) == 1`!),\n"
            "     `y = y_deconfounded`, `lam2 = study['lam2']`, `lam0 = study['lam0']`, and `M = study['M']`, globally minimize over `w in [-M, M]^p`:\n"
            "     `F(w) = 0.5 * ||X_cand @ w - y||_2^2 + 0.5 * lam2 * ||w||_2^2 + lam0 * count(|w_j| > 1e-8)` (student runtime `<= 18.0s` across 2 studies).\n"
            "   - Return `'certified_selection'` dict (`hegel_box_l0_cert_schema_v2`, validated by `/holdout_task/validate_public_schema.py`):\n"
            "     * `'weights'`: `(p,)` float sequence in `[-M, M]^p`, `'objective'`: float `F(weights)`,\n"
            "     * `'certified_lower_bound'`: float `L` (`0 <= F(weights) - L <= 1e-5`),\n"
            "     * `'subcube_leaves'`: list (`1 <= len <= 2048`) partitioning `{0, 1}^p`, each leaf having STRICT length-`p` boolean sequences\n"
            "       `'fixed_zero': bool[p]`, `'fixed_active': bool[p]` (disjoint), `'lb': float`, AND EITHER `('minorant_diagonal', 'minorant_center')`\n"
            "       (`shape (p,)` each, `d >= 0` with `X_cand.T @ X_cand - diag(d)` PSD) OR `'dual_nu'` (`shape (n_s,)`).\n"
            "- Return `{'aligned_panel': (N, D), 'calibration_weights': theta_cal_hat (13,), 'deconfounded_target': y_deconfounded (n_s,), 'certified_selection': dict}`.\n"
            "- Public studies: `/holdout_task/public_eq02_studies.pkl`. Shared public schema validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        public_setup_py=r'''
import heapq, json, math, pickle, sys
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np
from scipy.optimize import lsq_linear

def _eval_primal(X, y, M, lam0, lam2, mask_b):
    p = X.shape[1]
    idx = np.flatnonzero(mask_b)
    w = np.zeros(p, dtype=float)
    if idx.size > 0:
        Xs = X[:, idx]
        w[idx] = lsq_linear(np.vstack([Xs, math.sqrt(lam2) * np.eye(idx.size)]), np.concatenate([y, np.zeros(idx.size)]), bounds=(-M, M), tol=1e-12).x
    sup = np.abs(w) > 1e-8
    w = np.where(sup, w, 0.0)
    r = X @ w - y
    return 0.5 * float(r @ r) + 0.5 * lam2 * float(w @ w) + lam0 * int(np.sum(sup)), w

def _exact_bnb(X, y, M, lam0, lam2):
    n, p = X.shape
    G = X.T @ X
    Xty = X.T @ y
    off = np.abs(G).copy(); np.fill_diagonal(off, 0.0)
    d = np.maximum(np.diag(G) - np.sum(off, axis=1), 0.0)
    best_u, best_w = _eval_primal(X, y, M, lam0, lam2, np.zeros(p, dtype=bool))
    def _lb(fz, fa):
        keep = np.flatnonzero(~fz)
        cf = np.zeros(p, dtype=float)
        if keep.size > 0:
            cf[keep] = np.clip(np.linalg.solve(G[np.ix_(keep, keep)] + lam2 * np.eye(keep.size), Xty[keep]), -M, M)
        ca = np.zeros(p, dtype=float)
        act_i = np.flatnonzero(fa)
        if act_i.size > 0:
            ca[act_i] = np.clip(np.linalg.solve(G[np.ix_(act_i, act_i)] + lam2 * np.eye(act_i.size), Xty[act_i]), -M, M)
        best_tot = -1e30
        for c_vec in (cf, ca):
            r = y - X @ c_vec
            g = X.T @ r
            const = 0.5 * float(r @ r + (d * c_vec) @ c_vec) + float(g @ c_vec)
            z = np.abs(g + d * c_vec)
            kap = d + lam2
            b = np.minimum(M, z / np.maximum(kap, 1e-15))
            gain = b * (z - 0.5 * kap * b)
            on = lam0 - gain
            free = ~(fz | fa)
            tot = const + float(np.sum(on[fa])) + float(np.sum(np.minimum(on[free], 0.0)))
            if tot > best_tot:
                best_tot = tot
        return float(best_tot), cf
    lb0, c0 = _lb(np.zeros(p, dtype=bool), np.zeros(p, dtype=bool))
    for thr_h in (0.15, 0.25, 0.35):
        uh, wh = _eval_primal(X, y, M, lam0, lam2, np.abs(c0) > thr_h * M)
        if uh < best_u:
            best_u, best_w = uh, wh
    pq = [(lb0, 0, np.zeros(p, dtype=bool), np.zeros(p, dtype=bool), c0)]
    uid = 0
    while pq:
        lbn, _, fz, fa, cf = heapq.heappop(pq)
        if lbn >= best_u - 1e-9:
            continue
        uc, wc = _eval_primal(X, y, M, lam0, lam2, fa | ((~fz) & (np.abs(cf) > 0.2 * M)))
        if uc < best_u:
            best_u, best_w = uc, wc
        if lbn >= best_u - 1e-9:
            continue
        free = np.flatnonzero(~(fz | fa))
        if free.size == 0:
            continue
        jb = int(free[int(np.argmax(np.abs(cf[free])))])
        for act in (False, True):
            fzn, fan = fz.copy(), fa.copy()
            if act: fan[jb] = True
            else: fzn[jb] = True
            u1, w1 = _eval_primal(X, y, M, lam0, lam2, fan)
            if u1 < best_u: best_u, best_w = u1, w1
            lbc, cfc = _lb(fzn, fan)
            uid += 1
            heapq.heappush(pq, (lbc, uid, fzn, fan, cfc))
    return best_u, best_w

def _make_study(seed, p_dim):
    rng = np.random.default_rng(seed)
    D, Q = 8, 18
    offsets = np.array([4.1, -3.8, 3.6, -4.2, 3.3], dtype=float)
    gains = np.array([1.35, 0.76, 1.42, 0.80, 1.25], dtype=float)
    cell_counts = [[36, 10, 9], [9, 34, 10], [10, 9, 35], [32, 11, 10], [10, 31, 11]]  # 267 rows -> 177 cal + 90 s3
    strat_means = np.array([[-1.1, 0.8, -0.7, 0.6, -0.5, 0.4, -0.3, 0.3], [0.1, -0.1, 0.2, -0.2, 0.1, 0.0, 0.1, -0.1], [1.2, -0.9, 0.8, -0.7, 0.6, -0.5, 0.4, -0.4]])

    cids, strata, Z_list, X_list, ctrl_list = [], [], [], [], []
    for c in range(5):
        for s in range(3):
            cnt = cell_counts[c][s]
            z_cs = strat_means[s] + rng.normal(scale=0.85, size=(cnt, D))
            x_cs = offsets[c] + gains[c] * z_cs
            ct_cs = offsets[c] + gains[c] * rng.normal(size=(cnt, Q))
            cids.append(np.full(cnt, c, dtype=int))
            strata.append(np.full(cnt, s, dtype=int))
            Z_list.append(z_cs); X_list.append(x_cs); ctrl_list.append(ct_cs)

    perm = rng.permutation(sum(sum(r) for r in cell_counts))
    center_id = np.concatenate(cids)[perm]
    stratum = np.concatenate(strata)[perm]
    Z_true = np.vstack(Z_list)[perm]
    X_raw = np.vstack(X_list)[perm]
    controls = np.vstack(ctrl_list)[perm]
    N = len(center_id)

    obs_mask = rng.uniform(size=(N, D)) >= 0.025
    X_raw_nan = X_raw.copy(); X_raw_nan[~obs_mask] = np.nan
    ctrl_nan = controls.copy(); ctrl_nan[rng.uniform(size=(N, Q)) < 0.05] = np.nan

    # Stage 2a: complex128 interferometry
    phi_ref = rng.uniform(-math.pi, math.pi, size=(N, 2))
    amp_ref = rng.uniform(0.8, 1.4, size=(N, 2))
    phi_true = rng.uniform(-2.1, 2.1, size=(N, 2))
    ref_phasor = (amp_ref * np.exp(1j * phi_ref)).astype(np.complex128)
    signal_phasor = (rng.uniform(0.9, 1.3, size=(N, 2)) * np.exp(1j * (phi_ref + phi_true))).astype(np.complex128)

    cal_idx = np.arange(0, 177, dtype=int)
    s3_idx = np.arange(177, N, dtype=int)
    n_s = len(s3_idx)

    theta_cal = np.array([0.85, -0.75, 0.95, 1.15, -0.95, 0.80, -0.70, 0.60, -0.50, 0.45, -0.40, 1.25, -1.10], dtype=float)
    H_true = np.column_stack([np.ones(N), (stratum == 1).astype(float), (stratum == 2).astype(float), Z_true, phi_true])

    # Single connected block X_cand (`n_s x p_dim`) with correlated chain/chord couplings & positive diagonal minorant
    U_raw = rng.normal(size=(n_s, p_dim))
    Q_orth = np.zeros((n_s, p_dim), dtype=float)
    for j in range(p_dim):
        vj = U_raw[:, j].copy()
        for k in range(j):
            vj -= float(Q_orth[:, k] @ vj) * Q_orth[:, k]
        Q_orth[:, j] = vj / max(float(np.linalg.norm(vj)), 1e-12)
    M_mix = np.eye(p_dim, dtype=float)
    for j in range(p_dim - 1):
        M_mix[j, j + 1] = 0.14; M_mix[j + 1, j] = 0.14
    for j in range(p_dim - 2):
        M_mix[j, j + 2] = -0.06; M_mix[j + 2, j] = -0.06
    X_cand = math.sqrt(28.0) * (Q_orth @ M_mix)
    w_true = np.zeros(p_dim, dtype=float)
    w_true[[2, 6, 11, 15]] = [0.76, -0.68, 0.62, -0.71]
    M_box = 1.0
    lam0 = 0.24
    lam2 = 0.20

    y_assay = H_true @ theta_cal + rng.normal(scale=0.055, size=N)
    y_assay[s3_idx] += X_cand @ w_true
    glitch_rows = rng.choice(cal_idx, size=13, replace=False)
    y_assay[glitch_rows] += rng.choice([-1.0, 1.0], size=13) * rng.uniform(9.5, 13.5, size=13)

    y_s3_clean = y_assay[s3_idx] - H_true[s3_idx] @ theta_cal
    gt_obj, gt_w = _eval_primal(X_cand, y_s3_clean, M_box, lam0, lam2, np.abs(w_true) > 1e-8)

    return {
        "study": {
            "panel": {"center_id": center_id, "stratum": stratum, "X_panel_raw": X_raw_nan, "controls": ctrl_nan},
            "interferometry": {"signal_phasor": signal_phasor, "ref_phasor": ref_phasor},
            "y_assay_raw": y_assay,
            "calibration_indices": cal_idx,
            "stage3_indices": s3_idx,
            "X_cand": X_cand,
            "lam2": lam2,
            "lam0": lam0,
            "M": M_box,
        },
        "gt": {
            "Z_true": Z_true,
            "obs_mask": obs_mask,
            "theta_cal": theta_cal,
            "y_s3_clean": y_s3_clean,
            "w_opt_clean": gt_w,
            "obj_opt_clean": gt_obj,
        },
    }

pub = [_make_study(9211, 18)]
with open("/holdout_task/public_eq02_studies.pkl", "wb") as fh:
    pickle.dump(pub, fh)

open("/holdout_task/validate_public_schema.py", "w").write("""import importlib.util, json, pickle, sys
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np

def check_bool_mask(seq, p):
    if isinstance(seq, np.ndarray):
        if seq.dtype != np.bool_ or seq.shape != (p,): return None
        return [bool(x) for x in seq.tolist()]
    if isinstance(seq, (list, tuple)) and len(seq) == p and all(isinstance(x, (bool, np.bool_)) for x in seq):
        return [bool(x) for x in seq]
    return None

cand_p = sys.argv[1] if len(sys.argv) > 1 else "/holdout_task/coupled_interferometry_certified_l0.py"
spec = importlib.util.spec_from_file_location("cand_eq02", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
st = pickle.load(open("/holdout_task/public_eq02_studies.pkl", "rb"))[0]["study"]
out = mod.solve_coupled_interferometry_l0(st)
N, D = st["panel"]["X_panel_raw"].shape
n_s, p = st["X_cand"].shape
errs = []
if not isinstance(out, dict):
    errs.append("output_not_dict")
else:
    for k, shp in (("aligned_panel", (N, D)), ("calibration_weights", (13,)), ("deconfounded_target", (n_s,))):
        if k not in out: errs.append(f"missing:{k}")
        else:
            a = np.asarray(out[k], dtype=float)
            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    cs = out.get("certified_selection")
    if not isinstance(cs, dict):
        errs.append("missing_certified_selection")
    else:
        w = np.asarray(cs.get("weights", []), dtype=float)
        if w.shape != (p,) or not np.all(np.isfinite(w)) or np.max(np.abs(w)) > st["M"] + 1e-7:
            errs.append("bad_weights")
        lvs = cs.get("subcube_leaves")
        if not isinstance(lvs, list) or not (1 <= len(lvs) <= 2048):
            errs.append("bad_subcube_leaves_len")
        else:
            cov = 0.0
            for idx_l, lf in enumerate(lvs):
                fz = check_bool_mask(lf.get("fixed_zero"), p)
                fa = check_bool_mask(lf.get("fixed_active"), p)
                if fz is None or fa is None or any(z and a for z, a in zip(fz, fa)):
                    errs.append(f"leaf_{idx_l}_requires_disjoint_bool_len_p_masks"); break
                if not (("minorant_diagonal" in lf and "minorant_center" in lf) or ("dual_nu" in lf)):
                    errs.append(f"leaf_{idx_l}_missing_witness"); break
                cov += 2.0 ** (-(sum(fz) + sum(fa)))
            if abs(cov - 1.0) > 1e-9:
                errs.append(f"partition_coverage_{cov}!=1.0")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
""")
''',
        private_verifier_py=r'''
import heapq, json, math, sys, time
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, "/holdout_task")
import coupled_interferometry_certified_l0

def _eval_primal(X, y, M, lam0, lam2, mask_b):
    p = X.shape[1]
    idx = np.flatnonzero(mask_b)
    w = np.zeros(p, dtype=float)
    if idx.size > 0:
        Xs = X[:, idx]
        w[idx] = lsq_linear(np.vstack([Xs, math.sqrt(lam2) * np.eye(idx.size)]), np.concatenate([y, np.zeros(idx.size)]), bounds=(-M, M), tol=1e-12).x
    sup = np.abs(w) > 1e-8
    w = np.where(sup, w, 0.0)
    r = X @ w - y
    return 0.5 * float(r @ r) + 0.5 * lam2 * float(w @ w) + lam0 * int(np.sum(sup)), w

def _exact_bnb(X, y, M, lam0, lam2):
    n, p = X.shape
    G = X.T @ X
    Xty = X.T @ y
    off = np.abs(G).copy(); np.fill_diagonal(off, 0.0)
    d = np.maximum(np.diag(G) - np.sum(off, axis=1), 0.0)
    best_u, best_w = _eval_primal(X, y, M, lam0, lam2, np.zeros(p, dtype=bool))
    def _lb(fz, fa):
        keep = np.flatnonzero(~fz)
        cf = np.zeros(p, dtype=float)
        if keep.size > 0:
            cf[keep] = np.clip(np.linalg.solve(G[np.ix_(keep, keep)] + lam2 * np.eye(keep.size), Xty[keep]), -M, M)
        ca = np.zeros(p, dtype=float)
        act_i = np.flatnonzero(fa)
        if act_i.size > 0:
            ca[act_i] = np.clip(np.linalg.solve(G[np.ix_(act_i, act_i)] + lam2 * np.eye(act_i.size), Xty[act_i]), -M, M)
        best_tot = -1e30
        for c_vec in (cf, ca):
            r = y - X @ c_vec
            g = X.T @ r
            const = 0.5 * float(r @ r + (d * c_vec) @ c_vec) + float(g @ c_vec)
            z = np.abs(g + d * c_vec)
            kap = d + lam2
            b = np.minimum(M, z / np.maximum(kap, 1e-15))
            gain = b * (z - 0.5 * kap * b)
            on = lam0 - gain
            free = ~(fz | fa)
            tot = const + float(np.sum(on[fa])) + float(np.sum(np.minimum(on[free], 0.0)))
            if tot > best_tot:
                best_tot = tot
        return float(best_tot), cf
    lb0, c0 = _lb(np.zeros(p, dtype=bool), np.zeros(p, dtype=bool))
    for thr_h in (0.15, 0.25, 0.35):
        uh, wh = _eval_primal(X, y, M, lam0, lam2, np.abs(c0) > thr_h * M)
        if uh < best_u:
            best_u, best_w = uh, wh
    pq = [(lb0, 0, np.zeros(p, dtype=bool), np.zeros(p, dtype=bool), c0)]
    uid = 0
    while pq:
        lbn, _, fz, fa, cf = heapq.heappop(pq)
        if lbn >= best_u - 1e-9: continue
        uc, wc = _eval_primal(X, y, M, lam0, lam2, fa | ((~fz) & (np.abs(cf) > 0.2 * M)))
        if uc < best_u: best_u, best_w = uc, wc
        if lbn >= best_u - 1e-9: continue
        free = np.flatnonzero(~(fz | fa))
        if free.size == 0: continue
        jb = int(free[int(np.argmax(np.abs(cf[free])))])
        for act in (False, True):
            fzn, fan = fz.copy(), fa.copy()
            if act: fan[jb] = True
            else: fzn[jb] = True
            u1, w1 = _eval_primal(X, y, M, lam0, lam2, fan)
            if u1 < best_u: best_u, best_w = u1, w1
            lbc, cfc = _lb(fzn, fan)
            uid += 1
            heapq.heappush(pq, (lbc, uid, fzn, fan, cfc))
    return best_u, best_w

def _verify_cert_v2(X, y, M, lam0, lam2, cert):
    n, p = X.shape
    if not isinstance(cert, dict): return False, "not_dict", 1e9
    for k in ("weights", "objective", "certified_lower_bound", "subcube_leaves"):
        if k not in cert: return False, f"missing_{k}", 1e9
    w = np.asarray(cert["weights"], dtype=float)
    if w.shape != (p,) or not np.all(np.isfinite(w)) or float(np.max(np.abs(w))) > M + 1e-7:
        return False, "bad_weights", 1e9
    r = X @ w - y
    true_obj = 0.5 * float(r @ r) + 0.5 * lam2 * float(w @ w) + lam0 * int(np.sum(np.abs(w) > 1e-8))
    obj = float(cert["objective"]); clb = float(cert["certified_lower_bound"])
    if not (math.isfinite(obj) and math.isfinite(clb)) or abs(obj - true_obj) > 1e-5:
        return False, "obj_mismatch", 1e9
    leaves = cert["subcube_leaves"]
    if not isinstance(leaves, list) or not (1 <= len(leaves) <= 2048):
        return False, "bad_leaf_count", 1e9
    G = X.T @ X
    parsed_masks = []
    min_lb = float("inf")
    for lf in leaves:
        if not isinstance(lf, dict): return False, "leaf_not_dict", 1e9
        raw_fz, raw_fa = lf.get("fixed_zero"), lf.get("fixed_active")
        def _to_bool(s):
            if isinstance(s, np.ndarray):
                return ([bool(x) for x in s.tolist()] if (s.dtype == np.bool_ and s.shape == (p,)) else None)
            if isinstance(s, (list, tuple)) and len(s) == p and all(isinstance(x, (bool, np.bool_)) for x in s):
                return [bool(x) for x in s]
            return None
        fzb, fab = _to_bool(raw_fz), _to_bool(raw_fa)
        if fzb is None or fab is None: return False, "non_bool_len_p_mask", 1e9
        fz = np.asarray(fzb, dtype=bool); fa = np.asarray(fab, dtype=bool)
        if np.any(fz & fa): return False, "overlap_zero_active", 1e9
        parsed_masks.append((fz, fa))
        lb_decl = float(lf.get("lb", float("nan")))
        if not math.isfinite(lb_decl): return False, "nan_lb", 1e9
        if "minorant_diagonal" in lf and "minorant_center" in lf:
            d = np.asarray(lf["minorant_diagonal"], dtype=float)
            c = np.asarray(lf["minorant_center"], dtype=float)
            if d.shape != (p,) or c.shape != (p,) or not (np.all(np.isfinite(d)) and np.all(np.isfinite(c))) or np.any(d < -1e-9):
                return False, "bad_minorant", 1e9
            if float(np.linalg.eigvalsh(G - np.diag(d))[0]) < -1e-6:
                return False, "minorant_not_psd", 1e9
            d_pos = np.maximum(d, 0.0)
            r_c = y - X @ c
            g_c = X.T @ r_c
            const_c = 0.5 * float(r_c @ r_c + (d_pos * c) @ c) + float(g_c @ c)
            z_c = np.abs(g_c + d_pos * c)
            kap_c = d_pos + lam2
            b_c = np.minimum(M, z_c / np.maximum(kap_c, 1e-15))
            gain_c = b_c * (z_c - 0.5 * kap_c * b_c)
            on_c = lam0 - gain_c
            free_c = ~(fz | fa)
            t_lb = const_c + float(np.sum(on_c[fa])) + float(np.sum(np.minimum(on_c[free_c], 0.0)))
        elif "dual_nu" in lf:
            nu = np.asarray(lf["dual_nu"], dtype=float)
            if nu.shape != (n,) or not np.all(np.isfinite(nu)): return False, "bad_dual_nu", 1e9
            xi = X.T @ nu
            t_lb = -0.5 * float(nu @ nu) - float(nu @ y)
            for j in range(p):
                wu = float(np.clip(xi[j] / lam2, -M, M))
                gain_a = abs(xi[j]) * wu - 0.5 * lam2 * (wu ** 2) - lam0
                if fz[j]: pass
                elif fa[j]: t_lb -= gain_a
                else: t_lb -= max(0.0, gain_a)
        else:
            return False, "missing_witness", 1e9
        if lb_decl > t_lb + 1e-5: return False, "invalid_leaf_lb", 1e9
        min_lb = min(min_lb, float(t_lb))
    cov = sum(2.0 ** (-int(np.sum(fz) + np.sum(fa))) for fz, fa in parsed_masks)
    if abs(cov - 1.0) > 1e-8: return False, "incomplete_partition", 1e9
    for i in range(len(parsed_masks)):
        fzi, fai = parsed_masks[i]
        for j in range(i + 1, len(parsed_masks)):
            fzj, faj = parsed_masks[j]
            if not (np.any(fzi & faj) or np.any(fai & fzj)):
                return False, "overlapping_leaves", 1e9
    if clb > min_lb + 1e-5: return False, "clb_above_min_leaf", 1e9
    gap = true_obj - min(clb, min_lb)
    if gap > 1e-5 or gap < -1e-5: return False, f"duality_gap_{gap:.2e}", gap
    return True, "ok", max(0.0, gap)

def _make_priv(seed, p_dim):
    rng = np.random.default_rng(seed)
    D, Q = 8, 18
    offsets = np.array([3.8, -4.1, 3.9, -3.7, 3.5], dtype=float)
    gains = np.array([1.38, 0.78, 1.44, 0.82, 1.28], dtype=float)
    cell_counts = [[37, 10, 9], [9, 35, 10], [10, 9, 36], [33, 11, 10], [10, 32, 11]]  # 272 rows -> 182 cal + 90 s3
    strat_means = np.array([[-1.15, 0.85, -0.75, 0.60, -0.50, 0.40, -0.35, 0.30], [0.10, -0.10, 0.20, -0.20, 0.10, 0.0, 0.10, -0.10], [1.25, -0.95, 0.80, -0.70, 0.60, -0.50, 0.40, -0.40]])
    cids, strata, Z_list, X_list, ctrl_list = [], [], [], [], []
    for c in range(5):
        for s in range(3):
            cnt = cell_counts[c][s]
            z_cs = strat_means[s] + rng.normal(scale=0.85, size=(cnt, D))
            x_cs = offsets[c] + gains[c] * z_cs
            ct_cs = offsets[c] + gains[c] * rng.normal(size=(cnt, Q))
            cids.append(np.full(cnt, c, dtype=int)); strata.append(np.full(cnt, s, dtype=int))
            Z_list.append(z_cs); X_list.append(x_cs); ctrl_list.append(ct_cs)
    perm = rng.permutation(sum(sum(r) for r in cell_counts))
    center_id = np.concatenate(cids)[perm]; stratum = np.concatenate(strata)[perm]
    Z_true = np.vstack(Z_list)[perm]; X_raw = np.vstack(X_list)[perm]; controls = np.vstack(ctrl_list)[perm]
    N = len(center_id)
    obs_mask = rng.uniform(size=(N, D)) >= 0.025
    X_raw_nan = X_raw.copy(); X_raw_nan[~obs_mask] = np.nan
    ctrl_nan = controls.copy(); ctrl_nan[rng.uniform(size=(N, Q)) < 0.05] = np.nan
    phi_ref = rng.uniform(-math.pi, math.pi, size=(N, 2))
    amp_ref = rng.uniform(0.8, 1.4, size=(N, 2))
    phi_true = rng.uniform(-2.1, 2.1, size=(N, 2))
    ref_phasor = (amp_ref * np.exp(1j * phi_ref)).astype(np.complex128)
    signal_phasor = (rng.uniform(0.9, 1.3, size=(N, 2)) * np.exp(1j * (phi_ref + phi_true))).astype(np.complex128)
    cal_idx = np.arange(0, 182, dtype=int); s3_idx = np.arange(182, N, dtype=int)
    n_s = len(s3_idx)
    theta_cal = np.array([0.90, -0.80, 1.00, 1.18, -0.92, 0.82, -0.72, 0.62, -0.52, 0.46, -0.42, 1.28, -1.12], dtype=float)
    H_true = np.column_stack([np.ones(N), (stratum == 1).astype(float), (stratum == 2).astype(float), Z_true, phi_true])
    U_raw = rng.normal(size=(n_s, p_dim))
    Q_orth = np.zeros((n_s, p_dim), dtype=float)
    for j in range(p_dim):
        vj = U_raw[:, j].copy()
        for k in range(j):
            vj -= float(Q_orth[:, k] @ vj) * Q_orth[:, k]
        Q_orth[:, j] = vj / max(float(np.linalg.norm(vj)), 1e-12)
    M_mix = np.eye(p_dim, dtype=float)
    for j in range(p_dim - 1):
        M_mix[j, j + 1] = 0.14; M_mix[j + 1, j] = 0.14
    for j in range(p_dim - 2):
        M_mix[j, j + 2] = -0.06; M_mix[j + 2, j] = -0.06
    X_cand = math.sqrt(28.0) * (Q_orth @ M_mix)
    w_true = np.zeros(p_dim, dtype=float)
    w_true[[2, 6, 11, 15]] = [0.78, -0.70, 0.64, -0.72]
    M_box = 1.0; lam0 = 0.24; lam2 = 0.20
    y_assay = H_true @ theta_cal + rng.normal(scale=0.055, size=N)
    y_assay[s3_idx] += X_cand @ w_true
    glitch_rows = rng.choice(cal_idx, size=14, replace=False)
    y_assay[glitch_rows] += rng.choice([-1.0, 1.0], size=14) * rng.uniform(9.5, 13.5, size=14)
    y_s3_clean = y_assay[s3_idx] - H_true[s3_idx] @ theta_cal
    _, w_opt_clean = _eval_primal(X_cand, y_s3_clean, M_box, lam0, lam2, np.abs(w_true) > 1e-8)
    study = {
        "panel": {"center_id": center_id, "stratum": stratum, "X_panel_raw": X_raw_nan, "controls": ctrl_nan},
        "interferometry": {"signal_phasor": signal_phasor, "ref_phasor": ref_phasor},
        "y_assay_raw": y_assay, "calibration_indices": cal_idx, "stage3_indices": s3_idx,
        "X_cand": X_cand, "lam2": lam2, "lam0": lam0, "M": M_box,
    }
    return study, Z_true, obs_mask, theta_cal, y_s3_clean, w_opt_clean

priv_cases = [_make_priv(99411, 18), _make_priv(99412, 19)]
t0 = time.monotonic()
valid = True
max_panel_mae = 0.0
max_cal_rel = 0.0
max_deconf_mae = 0.0
max_w_rel = 0.0
max_gap = 0.0
diag_rows = []
for (sd, p_dim), (st, Z_gt, obs_m, th_gt, y_clean_gt, w_clean_gt) in zip(((99411, 18), (99412, 19)), priv_cases):
    out = coupled_interferometry_certified_l0.solve_coupled_interferometry_l0(st)
    if not isinstance(out, dict) or not {"aligned_panel", "calibration_weights", "deconfounded_target", "certified_selection"} <= set(out.keys()):
        valid = False; diag_rows.append({"seed": sd, "error": "missing_keys"}); continue
    ap = np.asarray(out["aligned_panel"], dtype=float)
    cw = np.asarray(out["calibration_weights"], dtype=float)
    dt = np.asarray(out["deconfounded_target"], dtype=float)
    if ap.shape != Z_gt.shape or cw.shape != (13,) or dt.shape != y_clean_gt.shape:
        valid = False; diag_rows.append({"seed": sd, "error": "shape_mismatch"}); continue
    if not (np.all(np.isfinite(ap)) and np.all(np.isfinite(cw)) and np.all(np.isfinite(dt))):
        valid = False; diag_rows.append({"seed": sd, "error": "non_finite"}); continue
    p_mae = float(np.mean(np.abs(ap[obs_m] - Z_gt[obs_m])))
    c_rel = float(np.linalg.norm(cw - th_gt) / np.linalg.norm(th_gt))
    d_mae = float(np.mean(np.abs(dt - y_clean_gt)))
    ok_c, rsn_c, gap_c = _verify_cert_v2(st["X_cand"], dt, float(st["M"]), float(st["lam0"]), float(st["lam2"]), out["certified_selection"])
    w_sub = np.asarray(out["certified_selection"].get("weights", np.zeros(p_dim)), dtype=float)
    w_rel = float(np.linalg.norm(w_sub - w_clean_gt) / max(np.linalg.norm(w_clean_gt), 1e-8)) if w_sub.shape == (p_dim,) else 1e9
    if not ok_c: valid = False
    max_panel_mae = max(max_panel_mae, p_mae)
    max_cal_rel = max(max_cal_rel, c_rel)
    max_deconf_mae = max(max_deconf_mae, d_mae)
    max_w_rel = max(max_w_rel, w_rel)
    max_gap = max(max_gap, gap_c)
    diag_rows.append({"seed": sd, "p": p_dim, "panel_mae": round(p_mae, 6), "cal_rel_l2": round(c_rel, 6), "deconf_mae": round(d_mae, 6), "w_vs_clean_rel_l2": round(w_rel, 6), "cert_ok": ok_c, "cert_reason": rsn_c})

student_elapsed = time.monotonic() - t0
strict_pass = bool(
    valid
    and len(diag_rows) == 2
    and max_panel_mae <= 0.11
    and max_cal_rel <= 0.14
    and max_deconf_mae <= 0.34
    and max_w_rel <= 0.28
    and max_gap <= 1e-5
    and student_elapsed <= 18.0
)
print(json.dumps({
    "strict_pass": strict_pass,
    "reward": 1.0 if strict_pass else 0.0,
    "all_valid": valid,
    "max_panel_mae": round(float(max_panel_mae), 6),
    "max_calibration_rel_l2": round(float(max_cal_rel), 6),
    "max_deconfounded_mae": round(float(max_deconf_mae), 6),
    "max_stage3_w_vs_clean_rel_l2": round(float(max_w_rel), 6),
    "max_duality_gap": (round(float(max_gap), 9) if math.isfinite(max_gap) else -1.0),
    "student_elapsed_sec": round(float(student_elapsed), 4),
    "instance_diagnostics": diag_rows,
}, allow_nan=False))
''',
    ),

    # =========================================================================
    # EQ03: Viscoelastic Hereditary Memory (`Same Present / Different Past`)
    # =========================================================================
    HoldoutOODTask(
        task_id="eq03__viscoelastic_hereditary_memory_same_present_different_past",
        short_name="eq03_viscoelastic_hereditary_memory",
        wave="m92_equal_info",
        scientific_domain="Multi-Scale Viscoelastic Hereditary Memory Rebinding & Coupon-Calibrated Creep Hazard Forecasting",
        wormhole_family="fam_multiscale_memory_and_outlier_deconfounding",
        declared_artifacts=("/holdout_task/viscoelastic_hereditary_forecast.py",),
        declared_callable_entrypoints=("solve_viscoelastic_hereditary_forecast",),
        instruction_text=(
            "Implement `/holdout_task/viscoelastic_hereditary_forecast.py` exposing:\n"
            "`def solve_viscoelastic_hereditary_forecast(dataset: dict) -> dict:`\n\n"
            "Multi-Laboratory Viscoelastic Hereditary Memory (`Same-Present / Different-Past`) & Creep Hazard Contract:\n"
            "1. LAB REFERENCE COUPON ALIGNMENT & BALANCED IMPUTATION (`train` and `test`):\n"
            "   - `dataset['train']` (`N_tr` coupons) and `dataset['test']` (`N_te` coupons) come from the same `L = 4` laboratories (`'lab_id' in 0..3`,\n"
            "     `'stratum' in {0, 1}`). Pool `ref_coupons` (`shape (N_tr + N_te, 4)`, ~8% `NaN`s, drawn from `N(0, I_4)` per lab) across `train`+`test`\n"
            "     by `lab_id` to estimate per-lab `offset[l]` (median of valid control column means) and `gain[l]` (median of valid control column\n"
            "     population stds, `ddof=0`), and align `X_aligned = (coupon_features_raw - offset[lab_id]) / gain[lab_id]` (`D = 4`, ~5% `NaN`s).\n"
            "   - Compute inverse-cell-frequency `(lab_id, stratum)` weights on pooled `(train + test)` (`mean == 1.0`) and impute any `NaN` in\n"
            "     `X_aligned` with its cell-balanced finite column mean -> return test slice as `'aligned_test_features'` (`shape (N_te, 4)`, `MAE <= 0.15`).\n"
            "2. CAUSAL HEREDITARY PRONY MEMORY KERNEL (`Same Instantaneous Present, Divergent Pre-Window History`):\n"
            "   - Each coupon `i` provides pre-window loading history `'past_stress_history'` (`shape (N, L_hist)`, `L_hist = 40`, uniform step\n"
            "     `dt = dataset['history_dt'] = 0.25` chronologically from `t = -L_hist * dt` up to `t = 0^-`) AND instantaneous `t = 0` observables\n"
            "     `'stress_hold_0'` (`shape (N,)`, held constant for `t >= 0`), `'strain_0'` (`shape (N,)`), and `'instant_rate_0'` (`shape (N,)`).\n"
            "   - Two hereditary Prony memory modes with known relaxation times `tau_1, tau_2 = dataset['memory_taus']` (`[1.0, 4.5]`) evolve causally\n"
            "     from `z_m = 0.0` across `s = 0 .. L_hist - 1` via exact zero-order-hold recurrence:\n"
            "     `dec_m = exp(-dt / tau_m);  z_m <- dec_m * z_m + (1.0 - dec_m) * past_stress_history[:, s]` -> terminal pre-window memory `z_m(0)`.\n"
            "   - For `t > 0` on `eval_times = dataset['eval_times']` (`shape (K_t,)`, strictly increasing in `(0, 8.0]`), clean viscoelastic creep obeys:\n"
            "     `strain(i, t) = strain_0[i] + a_1 * (stress_hold_0[i] - z_1[i]) * (1 - exp(-t / tau_1))`\n"
            "     `               + a_2 * (stress_hold_0[i] - z_2[i]) * (1 - exp(-t / tau_2)) + (b_0 + X_aligned[i] @ b_mat) * t`,\n"
            "     parameterized by the 7-vector `theta_hered = [a_1, a_2, b_0, b_mat_0, b_mat_1, b_mat_2, b_mat_3]`.\n"
            "   - `dataset['train']['strain_curves_obs']` (`shape (N_tr, K_t)`, real float64) contains ~9% heavy Cauchy transducer spikes (`+-0.95`)!\n"
            "     Fit `theta_hered` via Cauchy/Lorentzian IRLS over all `N_tr * K_t` training observations (`rel L2 <= 0.10`).\n"
            "     (CRITICAL: `dataset['test']` contains twin coupons with IDENTICAL `(lab_id, stratum, coupon_features_raw, stress_hold_0, strain_0, instant_rate_0)`\n"
            "     at `t = 0` that diverge strongly for `t > 0` solely through `past_stress_history` — any memoryless `t = 0` model fails completely!)\n"
            "3. TEST TRAJECTORY FORECAST & THRESHOLD HAZARD CROSSING TIME:\n"
            "   - Predict `'predicted_strain'` (`shape (N_te, K_t)`) at `eval_times` (`overall MAE <= 0.08`, twin-pair MAE `<= 0.08`).\n"
            "   - For each test coupon `i` with threshold `crit = dataset['test']['critical_strain'][i]`, compute `'crossing_times'` (`shape (N_te,)`)\n"
            "     on the piecewise-linear trajectory through knots `[(0.0, strain_0[i]), *(zip(eval_times, predicted_strain[i]))]`:\n"
            "     if `strain_0[i] >= crit`, `0.0`; else first segment `(t_a, y_a) -> (t_b, y_b)` with `y_b >= crit` gives `t_a + (crit - y_a) / (y_b - y_a) * (t_b - t_a)`;\n"
            "     if never reached by `eval_times[-1]`, return `float(eval_times[-1])` (`crossing MAE <= 0.18`).\n"
            "- Return `{'aligned_test_features': (N_te, 4), 'hereditary_weights': (7,), 'predicted_strain': (N_te, K_t), 'crossing_times': (N_te,)}`.\n"
            "- Public dataset: `/holdout_task/public_eq03_datasets.pkl`. Shared public schema validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        public_setup_py=r'''
import importlib.util, json, math, pickle, sys
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np

def _crossing_times(t_grid, s0, curves, crits):
    ts = np.concatenate([[0.0], np.asarray(t_grid, dtype=float)])
    ys = np.column_stack([np.asarray(s0, dtype=float), np.asarray(curves, dtype=float)])
    out = np.full(len(s0), float(ts[-1]), dtype=float)
    for i in range(len(s0)):
        c = float(crits[i])
        if ys[i, 0] >= c:
            out[i] = 0.0
            continue
        for k in range(len(ts) - 1):
            if ys[i, k + 1] >= c:
                dy = max(ys[i, k + 1] - ys[i, k], 1e-12)
                out[i] = float(ts[k] + (c - ys[i, k]) / dy * (ts[k + 1] - ts[k]))
                break
    return out

def _make_dataset(seed):
    rng = np.random.default_rng(seed)
    taus = np.array([1.0, 4.5], dtype=float)
    dt = 0.25
    L_hist = 40
    eval_times = np.linspace(0.5, 8.0, 16)
    offsets = np.array([2.8, -3.1, 3.4, -2.6], dtype=float)
    gains = np.array([1.30, 0.78, 1.40, 0.84], dtype=float)
    theta_true = np.array([0.42, 0.31, 0.085, 0.045, -0.038, 0.032, -0.028], dtype=float)

    def _gen_coupons(counts_ls):
        labs, strats, X_lat, X_raw, refs = [], [], [], [], []
        s_means = np.array([[-0.8, 0.6, -0.5, 0.4], [0.9, -0.7, 0.6, -0.4]], dtype=float)
        for l_id in range(4):
            for st_id in range(2):
                cnt = counts_ls[l_id][st_id]
                xl = s_means[st_id] + rng.normal(scale=0.6, size=(cnt, 4))
                xr = offsets[l_id] + gains[l_id] * xl
                rc = offsets[l_id] + gains[l_id] * rng.normal(size=(cnt, 4))
                labs.append(np.full(cnt, l_id, dtype=int))
                strats.append(np.full(cnt, st_id, dtype=int))
                X_lat.append(xl); X_raw.append(xr); refs.append(rc)
        N = sum(sum(r) for r in counts_ls)
        pm = rng.permutation(N)
        lab = np.concatenate(labs)[pm]; st = np.concatenate(strats)[pm]
        xl = np.vstack(X_lat)[pm]; xr = np.vstack(X_raw)[pm]; rc = np.vstack(refs)[pm]
        obs_m = rng.uniform(size=(N, 4)) >= 0.05
        xr_n = xr.copy(); xr_n[~obs_m] = np.nan
        rc_n = rc.copy(); rc_n[rng.uniform(size=(N, 4)) < 0.08] = np.nan
        return lab, st, xl, xr_n, rc_n, obs_m

    tr_lab, tr_st, tr_xl, tr_xr, tr_rc, _ = _gen_coupons([[14, 5], [5, 14], [13, 6], [6, 13]])
    te_lab, te_st, te_xl, te_xr, te_rc, te_obs_m = _gen_coupons([[4, 2], [2, 4], [4, 2], [2, 4]])  # 24 base + 12 twins = 36

    # Append 6 twin pairs (`12` test coupons) with IDENTICAL present state at t=0, opposite pre-window histories!
    twin_base_idx = np.arange(6, dtype=int)
    te_lab = np.concatenate([te_lab, te_lab[twin_base_idx], te_lab[twin_base_idx]])
    te_st = np.concatenate([te_st, te_st[twin_base_idx], te_st[twin_base_idx]])
    te_xl = np.vstack([te_xl, te_xl[twin_base_idx], te_xl[twin_base_idx]])
    te_xr = np.vstack([te_xr, te_xr[twin_base_idx], te_xr[twin_base_idx]])
    te_rc = np.vstack([te_rc, te_rc[twin_base_idx], te_rc[twin_base_idx]])
    te_obs_m = np.vstack([te_obs_m, te_obs_m[twin_base_idx], te_obs_m[twin_base_idx]])
    twin_hi_idx = np.arange(24, 30, dtype=int)
    twin_lo_idx = np.arange(30, 36, dtype=int)

    def _sim_curves(xl, n_rows, force_twins=False):
        hist = rng.normal(scale=1.2, size=(n_rows, L_hist))
        for r in range(n_rows):
            lvl = rng.uniform(-2.2, 2.2)
            hist[r, :20] += lvl
            hist[r, 20:] += rng.uniform(-2.2, 2.2)
        sh0 = rng.uniform(0.6, 2.4, size=n_rows)
        s0 = rng.uniform(0.1, 0.5, size=n_rows)
        ir0 = 0.15 * sh0 + rng.normal(scale=0.02, size=n_rows)
        if force_twins:
            for k_t in range(6):
                ih, il = int(twin_hi_idx[k_t]), int(twin_lo_idx[k_t])
                sh0[il] = sh0[ih]
                s0[il] = s0[ih]
                ir0[il] = ir0[ih]
                hist[ih] = np.linspace(2.6, 3.2, L_hist)
                hist[il] = np.linspace(-2.8, -2.4, L_hist)
        z_modes = []
        for tau_m in taus:
            dec = math.exp(-dt / float(tau_m))
            zm = np.zeros(n_rows, dtype=float)
            for s in range(L_hist):
                zm = dec * zm + (1.0 - dec) * hist[:, s]
            z_modes.append(zm)
        z1, z2 = z_modes[0], z_modes[1]
        e1 = 1.0 - np.exp(-eval_times / taus[0])
        e2 = 1.0 - np.exp(-eval_times / taus[1])
        drift_rate = theta_true[2] + xl @ theta_true[3:]
        clean = (
            s0[:, None]
            + theta_true[0] * ((sh0 - z1)[:, None] * e1[None, :])
            + theta_true[1] * ((sh0 - z2)[:, None] * e2[None, :])
            + drift_rate[:, None] * eval_times[None, :]
        )
        return hist, sh0, s0, ir0, clean

    tr_h, tr_sh0, tr_s0, tr_ir0, tr_clean = _sim_curves(tr_xl, len(tr_lab), force_twins=False)
    tr_obs = tr_clean + rng.normal(scale=0.008, size=tr_clean.shape)
    n_sp = int(0.09 * tr_obs.size)
    sp_flat = rng.choice(tr_obs.size, size=n_sp, replace=False)
    tr_obs.ravel()[sp_flat] += rng.choice([-1.0, 1.0], size=n_sp) * rng.uniform(0.75, 1.25, size=n_sp)

    te_h, te_sh0, te_s0, te_ir0, te_clean = _sim_curves(te_xl, len(te_lab), force_twins=True)
    crits = te_s0 + 0.55 * (te_clean[:, -1] - te_s0) + 0.12
    for k_t in range(6):
        ih, il = int(twin_hi_idx[k_t]), int(twin_lo_idx[k_t])
        crits[ih] = crits[il] = 0.5 * (float(te_clean[ih, 10]) + float(te_clean[il, 10]))
    te_cross = _crossing_times(eval_times, te_s0, te_clean, crits)

    return {
        "dataset": {
            "history_dt": dt,
            "memory_taus": taus.copy(),
            "eval_times": eval_times.copy(),
            "train": {
                "lab_id": tr_lab, "stratum": tr_st, "coupon_features_raw": tr_xr, "ref_coupons": tr_rc,
                "past_stress_history": tr_h, "stress_hold_0": tr_sh0, "strain_0": tr_s0, "instant_rate_0": tr_ir0,
                "strain_curves_obs": tr_obs,
            },
            "test": {
                "lab_id": te_lab, "stratum": te_st, "coupon_features_raw": te_xr, "ref_coupons": te_rc,
                "past_stress_history": te_h, "stress_hold_0": te_sh0, "strain_0": te_s0, "instant_rate_0": te_ir0,
                "critical_strain": crits,
            },
        },
        "gt": {
            "te_xl": te_xl, "te_obs_m": te_obs_m, "theta_true": theta_true,
            "te_clean": te_clean, "te_cross": te_cross,
            "twin_hi_idx": twin_hi_idx, "twin_lo_idx": twin_lo_idx,
        },
    }

pub = [_make_dataset(9221)]
with open("/holdout_task/public_eq03_datasets.pkl", "wb") as fh:
    pickle.dump(pub, fh)

open("/holdout_task/validate_public_schema.py", "w").write("""import importlib.util, json, pickle, sys
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np

cand_p = sys.argv[1] if len(sys.argv) > 1 else "/holdout_task/viscoelastic_hereditary_forecast.py"
spec = importlib.util.spec_from_file_location("cand_eq03", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ds = pickle.load(open("/holdout_task/public_eq03_datasets.pkl", "rb"))[0]["dataset"]
out = mod.solve_viscoelastic_hereditary_forecast(ds)
N_te = len(ds["test"]["lab_id"]); K_t = len(ds["eval_times"])
req = {"aligned_test_features": (N_te, 4), "hereditary_weights": (7,), "predicted_strain": (N_te, K_t), "crossing_times": (N_te,)}
errs = []
if not isinstance(out, dict): errs.append("output_not_dict")
else:
    for k, shp in req.items():
        if k not in out: errs.append(f"missing:{k}")
        else:
            a = np.asarray(out[k], dtype=float)
            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
""")
''',
        private_verifier_py=r'''
import json, math, sys, time
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np

sys.path.insert(0, "/holdout_task")
import viscoelastic_hereditary_forecast

def _crossing_times(t_grid, s0, curves, crits):
    ts = np.concatenate([[0.0], np.asarray(t_grid, dtype=float)])
    ys = np.column_stack([np.asarray(s0, dtype=float), np.asarray(curves, dtype=float)])
    out = np.full(len(s0), float(ts[-1]), dtype=float)
    for i in range(len(s0)):
        c = float(crits[i])
        if ys[i, 0] >= c:
            out[i] = 0.0; continue
        for k in range(len(ts) - 1):
            if ys[i, k + 1] >= c:
                dy = max(ys[i, k + 1] - ys[i, k], 1e-12)
                out[i] = float(ts[k] + (c - ys[i, k]) / dy * (ts[k + 1] - ts[k])); break
    return out

def _make_priv(seed):
    rng = np.random.default_rng(seed)
    taus = np.array([1.0, 4.5], dtype=float)
    dt = 0.25; L_hist = 40
    eval_times = np.linspace(0.5, 8.0, 16)
    offsets = np.array([3.1, -2.9, 3.6, -2.8], dtype=float)
    gains = np.array([1.34, 0.76, 1.42, 0.82], dtype=float)
    theta_true = np.array([0.44, 0.33, 0.090, 0.048, -0.040, 0.034, -0.030], dtype=float)

    def _gen_coupons(counts_ls):
        labs, strats, X_lat, X_raw, refs = [], [], [], [], []
        s_means = np.array([[-0.85, 0.65, -0.50, 0.40], [0.95, -0.75, 0.60, -0.45]], dtype=float)
        for l_id in range(4):
            for st_id in range(2):
                cnt = counts_ls[l_id][st_id]
                xl = s_means[st_id] + rng.normal(scale=0.6, size=(cnt, 4))
                xr = offsets[l_id] + gains[l_id] * xl
                rc = offsets[l_id] + gains[l_id] * rng.normal(size=(cnt, 4))
                labs.append(np.full(cnt, l_id, dtype=int)); strats.append(np.full(cnt, st_id, dtype=int))
                X_lat.append(xl); X_raw.append(xr); refs.append(rc)
        N = sum(sum(r) for r in counts_ls)
        pm = rng.permutation(N)
        lab = np.concatenate(labs)[pm]; st = np.concatenate(strats)[pm]
        xl = np.vstack(X_lat)[pm]; xr = np.vstack(X_raw)[pm]; rc = np.vstack(refs)[pm]
        obs_m = rng.uniform(size=(N, 4)) >= 0.05
        xr_n = xr.copy(); xr_n[~obs_m] = np.nan
        rc_n = rc.copy(); rc_n[rng.uniform(size=(N, 4)) < 0.08] = np.nan
        return lab, st, xl, xr_n, rc_n, obs_m

    tr_lab, tr_st, tr_xl, tr_xr, tr_rc, _ = _gen_coupons([[15, 5], [5, 15], [14, 6], [6, 14]])
    te_lab, te_st, te_xl, te_xr, te_rc, te_obs_m = _gen_coupons([[4, 2], [2, 4], [4, 2], [2, 4]])
    twin_base = np.arange(8, dtype=int)
    te_lab = np.concatenate([te_lab, te_lab[twin_base], te_lab[twin_base]])
    te_st = np.concatenate([te_st, te_st[twin_base], te_st[twin_base]])
    te_xl = np.vstack([te_xl, te_xl[twin_base], te_xl[twin_base]])
    te_xr = np.vstack([te_xr, te_xr[twin_base], te_xr[twin_base]])
    te_rc = np.vstack([te_rc, te_rc[twin_base], te_rc[twin_base]])
    te_obs_m = np.vstack([te_obs_m, te_obs_m[twin_base], te_obs_m[twin_base]])
    twin_hi = np.arange(24, 32, dtype=int)
    twin_lo = np.arange(32, 40, dtype=int)

    def _sim_curves(xl, n_rows, force_twins=False):
        hist = rng.normal(scale=1.2, size=(n_rows, L_hist))
        for r in range(n_rows):
            hist[r, :20] += rng.uniform(-2.3, 2.3)
            hist[r, 20:] += rng.uniform(-2.3, 2.3)
        sh0 = rng.uniform(0.6, 2.4, size=n_rows)
        s0 = rng.uniform(0.1, 0.5, size=n_rows)
        ir0 = 0.15 * sh0 + rng.normal(scale=0.02, size=n_rows)
        if force_twins:
            for k_t in range(8):
                ih, il = int(twin_hi[k_t]), int(twin_lo[k_t])
                sh0[il] = sh0[ih]; s0[il] = s0[ih]; ir0[il] = ir0[ih]
                hist[ih] = np.linspace(2.7, 3.3, L_hist)
                hist[il] = np.linspace(-2.9, -2.5, L_hist)
        z_modes = []
        for tau_m in taus:
            dec = math.exp(-dt / float(tau_m))
            zm = np.zeros(n_rows, dtype=float)
            for s in range(L_hist):
                zm = dec * zm + (1.0 - dec) * hist[:, s]
            z_modes.append(zm)
        z1, z2 = z_modes[0], z_modes[1]
        e1 = 1.0 - np.exp(-eval_times / taus[0])
        e2 = 1.0 - np.exp(-eval_times / taus[1])
        drift_rate = theta_true[2] + xl @ theta_true[3:]
        clean = (
            s0[:, None]
            + theta_true[0] * ((sh0 - z1)[:, None] * e1[None, :])
            + theta_true[1] * ((sh0 - z2)[:, None] * e2[None, :])
            + drift_rate[:, None] * eval_times[None, :]
        )
        return hist, sh0, s0, ir0, clean

    tr_h, tr_sh0, tr_s0, tr_ir0, tr_clean = _sim_curves(tr_xl, len(tr_lab), force_twins=False)
    tr_obs = tr_clean + rng.normal(scale=0.008, size=tr_clean.shape)
    n_sp = int(0.09 * tr_obs.size)
    sp_flat = rng.choice(tr_obs.size, size=n_sp, replace=False)
    tr_obs.ravel()[sp_flat] += rng.choice([-1.0, 1.0], size=n_sp) * rng.uniform(0.75, 1.25, size=n_sp)
    te_h, te_sh0, te_s0, te_ir0, te_clean = _sim_curves(te_xl, len(te_lab), force_twins=True)
    crits = te_s0 + 0.55 * (te_clean[:, -1] - te_s0) + 0.12
    for k_t in range(8):
        ih, il = int(twin_hi[k_t]), int(twin_lo[k_t])
        crits[ih] = crits[il] = 0.5 * (float(te_clean[ih, 10]) + float(te_clean[il, 10]))
    te_cross = _crossing_times(eval_times, te_s0, te_clean, crits)
    ds = {
        "history_dt": dt, "memory_taus": taus.copy(), "eval_times": eval_times.copy(),
        "train": {"lab_id": tr_lab, "stratum": tr_st, "coupon_features_raw": tr_xr, "ref_coupons": tr_rc, "past_stress_history": tr_h, "stress_hold_0": tr_sh0, "strain_0": tr_s0, "instant_rate_0": tr_ir0, "strain_curves_obs": tr_obs},
        "test": {"lab_id": te_lab, "stratum": te_st, "coupon_features_raw": te_xr, "ref_coupons": te_rc, "past_stress_history": te_h, "stress_hold_0": te_sh0, "strain_0": te_s0, "instant_rate_0": te_ir0, "critical_strain": crits},
    }
    return ds, te_xl, te_obs_m, theta_true, te_clean, te_cross, twin_hi, twin_lo

t0 = time.monotonic()
valid = True
max_feat_mae = 0.0
max_w_rel = 0.0
max_strain_mae = 0.0
max_twin_mae = 0.0
max_cross_mae = 0.0
diag_rows = []
for sd in (99421, 99422, 99423):
    ds, te_xl, te_obs_m, th_gt, clean_gt, cross_gt, t_hi, t_lo = _make_priv(sd)
    out = viscoelastic_hereditary_forecast.solve_viscoelastic_hereditary_forecast(ds)
    if not isinstance(out, dict) or not {"aligned_test_features", "hereditary_weights", "predicted_strain", "crossing_times"} <= set(out.keys()):
        valid = False; diag_rows.append({"seed": sd, "error": "missing_keys"}); continue
    af = np.asarray(out["aligned_test_features"], dtype=float)
    hw = np.asarray(out["hereditary_weights"], dtype=float)
    ps = np.asarray(out["predicted_strain"], dtype=float)
    ct = np.asarray(out["crossing_times"], dtype=float)
    if af.shape != te_xl.shape or hw.shape != (7,) or ps.shape != clean_gt.shape or ct.shape != cross_gt.shape:
        valid = False; diag_rows.append({"seed": sd, "error": "shape_mismatch"}); continue
    if not (np.all(np.isfinite(af)) and np.all(np.isfinite(hw)) and np.all(np.isfinite(ps)) and np.all(np.isfinite(ct))):
        valid = False; diag_rows.append({"seed": sd, "error": "non_finite"}); continue
    f_mae = float(np.mean(np.abs(af[te_obs_m] - te_xl[te_obs_m])))
    w_rel = float(np.linalg.norm(hw - th_gt) / np.linalg.norm(th_gt))
    s_mae = float(np.mean(np.abs(ps - clean_gt)))
    tw_idx = np.concatenate([t_hi, t_lo])
    tw_mae = float(np.mean(np.abs(ps[tw_idx] - clean_gt[tw_idx])))
    c_mae = float(np.mean(np.abs(ct - cross_gt)))
    max_feat_mae = max(max_feat_mae, f_mae)
    max_w_rel = max(max_w_rel, w_rel)
    max_strain_mae = max(max_strain_mae, s_mae)
    max_twin_mae = max(max_twin_mae, tw_mae)
    max_cross_mae = max(max_cross_mae, c_mae)
    diag_rows.append({"seed": sd, "feat_mae": round(f_mae, 5), "w_rel_l2": round(w_rel, 5), "strain_mae": round(s_mae, 5), "twin_mae": round(tw_mae, 5), "cross_mae": round(c_mae, 5)})

student_elapsed = time.monotonic() - t0
strict_pass = bool(
    valid
    and len(diag_rows) == 3
    and max_feat_mae <= 0.15
    and max_w_rel <= 0.10
    and max_strain_mae <= 0.08
    and max_twin_mae <= 0.08
    and max_cross_mae <= 0.18
)
print(json.dumps({
    "strict_pass": strict_pass,
    "reward": 1.0 if strict_pass else 0.0,
    "all_valid": valid,
    "max_aligned_feat_mae": round(float(max_feat_mae), 5),
    "max_hereditary_w_rel_l2": round(float(max_w_rel), 5),
    "max_strain_mae": round(float(max_strain_mae), 5),
    "max_twin_same_present_mae": round(float(max_twin_mae), 5),
    "max_crossing_time_mae": round(float(max_cross_mae), 5),
    "student_elapsed_sec": round(float(student_elapsed), 4),
    "instance_diagnostics": diag_rows,
}, allow_nan=False))
''',
    ),

    # =========================================================================
    # EQ04: Non-Hermitian Complex Advection + Non-Separable Pairwise Penalty (Negative Control)
    # =========================================================================
    HoldoutOODTask(
        task_id="eq04__nonhermitian_complex_advection_and_coupled_ising_negative_control",
        short_name="eq04_nonhermitian_complex_coupled_negctrl",
        wave="m92_equal_info",
        scientific_domain="Non-Hermitian Complex Advection-Reaction PDE & Non-Separable Pairwise Co-Activation Subset Selection",
        wormhole_family="fam_indefinite_wave_scattering_negative_control",
        declared_artifacts=("/holdout_task/nonhermitian_advection_coupled_select.py",),
        declared_callable_entrypoints=("solve_nonhermitian_coupled_inversion",),
        instruction_text=(
            "Implement `/holdout_task/nonhermitian_advection_coupled_select.py` exposing:\n"
            "`def solve_nonhermitian_coupled_inversion(problem: dict) -> dict:`\n\n"
            "Non-Hermitian Complex128 Oscillatory Advection-Reaction PDE & Coupled Pairwise Co-Activation Selection Contract:\n"
            "1. 2D NON-HERMITIAN COMPLEX128 ADVECTION-DIFFUSION-REACTION OPERATOR (`(nx, ny) = (32, 32) = 1,024` interior nodes):\n"
            "   - On `[0, Lx] x [0, Ly]` with uniform grid spacing `hx = Lx / (nx + 1)`, `hy = Ly / (ny + 1)` and zero Dirichlet exterior\n"
            "     boundary `psi = 0`, `problem` provides diffusivity `'nu'` (`float > 0`), velocity arrays `'vx'`, `'vy'` (`shape (nx, ny)`),\n"
            "     damping `'sigma'` (`shape (nx, ny)`), angular frequency `'omega'` (`float != 0`), and `'basis'` (`shape (M, nx, ny)`, `M = 11`, float).\n"
            "   - Assemble the `1,024 x 1,024` complex128 non-Hermitian matrix `A` (`A != A.conj().T` and indefinite) in C-order `ravel()` (`id = i * ny + j`):\n"
            "     * Central diffusion (`-nu * Laplacian`): diagonal `+(2*nu/hx**2 + 2*nu/hy**2)`; if `i > 0`, `(id, id - ny)` gets `-nu/hx**2`;\n"
            "       if `i < nx - 1`, `(id, id + ny)` gets `-nu/hx**2`; if `j > 0`, `(id, id - 1)` gets `-nu/hy**2`; if `j < ny - 1`, `(id, id + 1)` gets `-nu/hy**2`.\n"
            "     * First-order upwind advection (`vx * d_x psi + vy * d_y psi`): on cell `(i, j)` with `vx_c = vx[i, j]`, `vy_c = vy[i, j]`:\n"
            "       - if `vx_c >= 0`: add `+vx_c / hx` to `(id, id)`, and if `i > 0` add `-vx_c / hx` to `(id, id - ny)`;\n"
            "       - if `vx_c < 0`: add `-vx_c / hx` to `(id, id)` (`== |vx_c|/hx`), and if `i < nx - 1` add `+vx_c / hx` to `(id, id + ny)`;\n"
            "       - if `vy_c >= 0`: add `+vy_c / hy` to `(id, id)`, and if `j > 0` add `-vy_c / hy` to `(id, id - 1)`;\n"
            "       - if `vy_c < 0`: add `-vy_c / hy` to `(id, id)`, and if `j < ny - 1` add `+vy_c / hy` to `(id, id + 1)`.\n"
            "     * Complex oscillatory reaction diagonal: add `sigma[i, j] + 1j * omega` to `(id, id)`.\n"
            "     (WARNING: `A` is complex128 and non-Hermitian — real SPD conjugate-gradient / AMG routines fail preconditions; use sparse direct LU / `spsolve`.)\n"
            "   - Solve `A @ g_m = basis[m].ravel()` (`complex128`) for `m = 0..M-1` and form the complex sensor matrix `G` (`shape (K_sens, M)`, `complex128`)\n"
            "     at `sensor_ij = problem['sensor_ij']` (`shape (K_sens, 2)`).\n"
            "2. EXACT COMPLEX SUBSET SELECTION WITH NON-SEPARABLE PAIRWISE GROUP PENALTY (`non_separable_cardinality` / `pairwise_group_penalty`):\n"
            "   - Given `y_obs = problem['y_obs_re'] + 1j * problem['y_obs_im']` (`shape (K_sens,)`, `complex128`), `lam2 = problem['lam2'] > 0`,\n"
            "     `lam0 = problem['lam0'] > 0`, and `pair_penalties = problem['pair_penalties']` (list of `[u, v, gamma_uv]` with `gamma_uv > 0`),\n"
            "     globally minimize over complex weights `w in C^M` (`M = 11`, `2^11 = 2,048` supports):\n"
            "     `J(w) = 0.5 * ||G @ w - y_obs||_2^2 + 0.5 * lam2 * ||w||_2^2 + lam0 * sum_m I(|w_m| > 1e-8)`\n"
            "     `       + sum_{(u, v, gamma_uv)} gamma_uv * I(|w_u| > 1e-8 and |w_v| > 1e-8)`\n"
            "     (where `||z||_2^2 = sum |z_k|^2 = real(z.conj() @ z)`; on any active support `S`, optimal `w_S` solves `(G_S^H G_S + lam2 I) w_S = G_S^H y_obs`).\n"
            "- Return:\n"
            "  - `'weights_real'`: `(M,)` float array `real(w_opt)`,\n"
            "  - `'weights_imag'`: `(M,)` float array `imag(w_opt)`,\n"
            "  - `'objective'`: float `J(w_opt)` (`abs error <= 1e-6`),\n"
            "  - `'predicted_field_abs'`: `(nx, ny)` float array `|sum_m w_opt[m] * g_m.reshape(nx, ny)|` (`rel L2 <= 1e-5`).\n"
            "- Public cases: `/holdout_task/public_eq04_cases.pkl`. Shared public schema validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        public_setup_py=r'''
import importlib.util, json, math, pickle, sys
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, splu

def _fast_2grid_prec(A, nx, ny, nz):
    N = nx * ny * nz
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    agg = ((np.arange(nx)[:, None, None] // 2) * (cy * cz) + (np.arange(ny)[None, :, None] // 2) * cz + (np.arange(nz)[None, None, :] // 2)).ravel()
    P = sparse.coo_matrix((np.ones(N, dtype=float), (np.arange(N), agg)), shape=(N, cx * cy * cz)).tocsr()
    R = P.T.tocsr()
    inv_d = 1.0 / np.maximum(A.diagonal(), 1e-12)
    lu_c = splu((R @ A @ P).tocsc())
    w = 0.67
    def _mv(r):
        x = w * inv_d * r
        x = x + P @ lu_c.solve(R @ (r - A @ x))
        x = x + w * inv_d * (r - A @ x)
        return x
    return LinearOperator((N, N), matvec=_mv, dtype=float)

def _assemble_adv(Lx, Ly, nu, vx, vy, sigma, omega):
    nx, ny = sigma.shape
    hx, hy = Lx / (nx + 1), Ly / (ny + 1)
    N = nx * ny
    ids = np.arange(N).reshape(nx, ny)
    diag = np.full(N, 2.0 * nu / (hx ** 2) + 2.0 * nu / (hy ** 2), dtype=np.complex128) + sigma.ravel().astype(np.complex128) + 1j * float(omega)
    rows, cols, data = [], [], []

    i0, i1 = ids[:-1, :].ravel(), ids[1:, :].ravel()
    rows.extend([i0, i1]); cols.extend([i1, i0])
    data.extend([np.full(i0.size, -nu / (hx ** 2), dtype=np.complex128), np.full(i0.size, -nu / (hx ** 2), dtype=np.complex128)])
    j0, j1 = ids[:, :-1].ravel(), ids[:, 1:].ravel()
    rows.extend([j0, j1]); cols.extend([j1, j0])
    data.extend([np.full(j0.size, -nu / (hy ** 2), dtype=np.complex128), np.full(j0.size, -nu / (hy ** 2), dtype=np.complex128)])

    vxf = vx.ravel(); vyf = vy.ravel()
    diag += np.abs(vxf) / hx + np.abs(vyf) / hy

    pos_x = vxf[i1] >= 0.0
    if np.any(pos_x):
        rows.append(i1[pos_x]); cols.append(i0[pos_x]); data.append((-vxf[i1[pos_x]] / hx).astype(np.complex128))
    neg_x = vxf[i0] < 0.0
    if np.any(neg_x):
        rows.append(i0[neg_x]); cols.append(i1[neg_x]); data.append((vxf[i0[neg_x]] / hx).astype(np.complex128))

    pos_y = vyf[j1] >= 0.0
    if np.any(pos_y):
        rows.append(j1[pos_y]); cols.append(j0[pos_y]); data.append((-vyf[j1[pos_y]] / hy).astype(np.complex128))
    neg_y = vyf[j0] < 0.0
    if np.any(neg_y):
        rows.append(j0[neg_y]); cols.append(j1[neg_y]); data.append((vyf[j0[neg_y]] / hy).astype(np.complex128))

    rows.append(ids.ravel()); cols.append(ids.ravel()); data.append(diag)
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))

def _solve_exact(G, y, lam2, lam0, pairs):
    M = G.shape[1]
    GH_G = G.conj().T @ G
    GH_y = G.conj().T @ y
    yy = 0.5 * float(np.real(np.vdot(y, y)))
    best_j = yy
    best_w = np.zeros(M, dtype=np.complex128)
    for mask in range(1, 1 << M):
        idx = [m for m in range(M) if (mask >> m) & 1]
        k = len(idx)
        H_s = GH_G[np.ix_(idx, idx)] + lam2 * np.eye(k, dtype=np.complex128)
        w_s = np.linalg.solve(H_s, GH_y[idx])
        quad = yy - 0.5 * float(np.real(np.vdot(GH_y[idx], w_s)))
        pen = lam0 * k + sum(float(gam) for u, v, gam in pairs if ((mask >> int(u)) & 1) and ((mask >> int(v)) & 1))
        val = quad + pen
        if val < best_j:
            best_j = val
            best_w = np.zeros(M, dtype=np.complex128)
            best_w[idx] = w_s
    return best_w, float(best_j)

def _make_case(seed):
    rng = np.random.default_rng(seed)
    nx = ny = 32
    Lx, Ly = 1.0, 1.0
    xs = np.arange(1, nx + 1) * (Lx / (nx + 1))
    ys = np.arange(1, ny + 1) * (Ly / (ny + 1))
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    nu = 0.035
    vx = 1.6 * np.cos(math.pi * YY)
    vy = -1.2 * np.sin(math.pi * XX)
    sigma = 0.25 + 0.15 * (XX ** 2 + YY ** 2)
    omega = 6.5

    M = 11
    basis = np.zeros((M, nx, ny), dtype=float)
    for m in range(M):
        cx = 0.18 + 0.64 * ((m * 3) % 11) / 10.0
        cy = 0.18 + 0.64 * ((m * 7 + 2) % 11) / 10.0
        basis[m] = 18.0 * np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2.0 * (0.085 ** 2)))

    A = _assemble_adv(Lx, Ly, nu, vx, vy, sigma, omega)
    lu = splu(A.tocsc())
    G_full = np.zeros((M, nx, ny), dtype=np.complex128)
    for m in range(M):
        G_full[m] = lu.solve(basis[m].ravel().astype(np.complex128)).reshape(nx, ny)

    K_s = 80
    fc = rng.choice(nx * ny, size=K_s, replace=False)
    sensor_ij = np.column_stack([fc // ny, fc % ny])
    G_sens = G_full[:, sensor_ij[:, 0], sensor_ij[:, 1]].T

    w_gt_gen = np.zeros(M, dtype=np.complex128)
    w_gt_gen[[1, 4, 8]] = [1.2 - 0.8j, -0.9 + 1.1j, 1.0 + 0.7j]
    y_obs = G_sens @ w_gt_gen + (rng.normal(scale=0.015, size=K_s) + 1j * rng.normal(scale=0.015, size=K_s))
    lam2 = 0.08
    lam0 = 0.09
    pairs = [[1, 2, 0.22], [4, 5, 0.25], [8, 9, 0.20], [0, 4, 0.18], [3, 7, 0.24]]

    w_opt, j_opt = _solve_exact(G_sens, y_obs, lam2, lam0, pairs)
    psi_opt = np.tensordot(w_opt, G_full, axes=(0, 0))
    return {
        "problem": {
            "Lx": Lx, "Ly": Ly, "nu": nu, "vx": vx, "vy": vy, "sigma": sigma, "omega": omega,
            "basis": basis, "sensor_ij": sensor_ij,
            "y_obs_re": y_obs.real.copy(), "y_obs_im": y_obs.imag.copy(),
            "lam2": lam2, "lam0": lam0, "pair_penalties": pairs,
        },
        "gt": {
            "weights_real": w_opt.real.copy(),
            "weights_imag": w_opt.imag.copy(),
            "objective": j_opt,
            "predicted_field_abs": np.abs(psi_opt),
        },
    }

pub = [_make_case(9231)]
with open("/holdout_task/public_eq04_cases.pkl", "wb") as fh:
    pickle.dump(pub, fh)

open("/holdout_task/validate_public_schema.py", "w").write("""import importlib.util, json, pickle, sys
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np

cand_p = sys.argv[1] if len(sys.argv) > 1 else "/holdout_task/nonhermitian_advection_coupled_select.py"
spec = importlib.util.spec_from_file_location("cand_eq04", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
pr = pickle.load(open("/holdout_task/public_eq04_cases.pkl", "rb"))[0]["problem"]
out = mod.solve_nonhermitian_coupled_inversion(pr)
M, nx, ny = pr["basis"].shape
errs = []
if not isinstance(out, dict): errs.append("output_not_dict")
else:
    for k, shp in (("weights_real", (M,)), ("weights_imag", (M,)), ("predicted_field_abs", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}")
        else:
            a = np.asarray(out[k], dtype=float)
            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    if "objective" not in out or not np.isfinite(float(out["objective"])):
        errs.append("bad_objective")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
""")
''',
        private_verifier_py=r'''
import json, math, sys, time
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, splu

def _fast_2grid_prec(A, nx, ny, nz):
    N = nx * ny * nz
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    agg = ((np.arange(nx)[:, None, None] // 2) * (cy * cz) + (np.arange(ny)[None, :, None] // 2) * cz + (np.arange(nz)[None, None, :] // 2)).ravel()
    P = sparse.coo_matrix((np.ones(N, dtype=float), (np.arange(N), agg)), shape=(N, cx * cy * cz)).tocsr()
    R = P.T.tocsr()
    inv_d = 1.0 / np.maximum(A.diagonal(), 1e-12)
    lu_c = splu((R @ A @ P).tocsc())
    w = 0.67
    def _mv(r):
        x = w * inv_d * r
        x = x + P @ lu_c.solve(R @ (r - A @ x))
        x = x + w * inv_d * (r - A @ x)
        return x
    return LinearOperator((N, N), matvec=_mv, dtype=float)

sys.path.insert(0, "/holdout_task")
import nonhermitian_advection_coupled_select

def _assemble_adv(Lx, Ly, nu, vx, vy, sigma, omega):
    nx, ny = sigma.shape
    hx, hy = Lx / (nx + 1), Ly / (ny + 1)
    N = nx * ny
    ids = np.arange(N).reshape(nx, ny)
    diag = np.full(N, 2.0 * nu / (hx ** 2) + 2.0 * nu / (hy ** 2), dtype=np.complex128) + sigma.ravel().astype(np.complex128) + 1j * float(omega)
    rows, cols, data = [], [], []
    i0, i1 = ids[:-1, :].ravel(), ids[1:, :].ravel()
    rows.extend([i0, i1]); cols.extend([i1, i0])
    data.extend([np.full(i0.size, -nu / (hx ** 2), dtype=np.complex128), np.full(i0.size, -nu / (hx ** 2), dtype=np.complex128)])
    j0, j1 = ids[:, :-1].ravel(), ids[:, 1:].ravel()
    rows.extend([j0, j1]); cols.extend([j1, j0])
    data.extend([np.full(j0.size, -nu / (hy ** 2), dtype=np.complex128), np.full(j0.size, -nu / (hy ** 2), dtype=np.complex128)])
    vxf = vx.ravel(); vyf = vy.ravel()
    diag += np.abs(vxf) / hx + np.abs(vyf) / hy
    pos_x = vxf[i1] >= 0.0
    if np.any(pos_x): rows.append(i1[pos_x]); cols.append(i0[pos_x]); data.append((-vxf[i1[pos_x]] / hx).astype(np.complex128))
    neg_x = vxf[i0] < 0.0
    if np.any(neg_x): rows.append(i0[neg_x]); cols.append(i1[neg_x]); data.append((vxf[i0[neg_x]] / hx).astype(np.complex128))
    pos_y = vyf[j1] >= 0.0
    if np.any(pos_y): rows.append(j1[pos_y]); cols.append(j0[pos_y]); data.append((-vyf[j1[pos_y]] / hy).astype(np.complex128))
    neg_y = vyf[j0] < 0.0
    if np.any(neg_y): rows.append(j0[neg_y]); cols.append(j1[neg_y]); data.append((vyf[j0[neg_y]] / hy).astype(np.complex128))
    rows.append(ids.ravel()); cols.append(ids.ravel()); data.append(diag)
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))

def _solve_exact(G, y, lam2, lam0, pairs):
    M = G.shape[1]
    GH_G = G.conj().T @ G; GH_y = G.conj().T @ y
    yy = 0.5 * float(np.real(np.vdot(y, y)))
    best_j = yy; best_w = np.zeros(M, dtype=np.complex128)
    for mask in range(1, 1 << M):
        idx = [m for m in range(M) if (mask >> m) & 1]
        k = len(idx)
        H_s = GH_G[np.ix_(idx, idx)] + lam2 * np.eye(k, dtype=np.complex128)
        w_s = np.linalg.solve(H_s, GH_y[idx])
        quad = yy - 0.5 * float(np.real(np.vdot(GH_y[idx], w_s)))
        pen = lam0 * k + sum(float(gam) for u, v, gam in pairs if ((mask >> int(u)) & 1) and ((mask >> int(v)) & 1))
        val = quad + pen
        if val < best_j:
            best_j = val; best_w = np.zeros(M, dtype=np.complex128); best_w[idx] = w_s
    return best_w, float(best_j)

def _make_priv(seed):
    rng = np.random.default_rng(seed)
    nx = ny = 32; Lx, Ly = 1.05, 0.95
    xs = np.arange(1, nx + 1) * (Lx / (nx + 1)); ys = np.arange(1, ny + 1) * (Ly / (ny + 1))
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    nu = 0.032
    vx = 1.7 * np.cos(math.pi * YY / Ly)
    vy = -1.3 * np.sin(math.pi * XX / Lx)
    sigma = 0.24 + 0.16 * ((XX / Lx) ** 2 + (YY / Ly) ** 2)
    omega = 6.8
    M = 11
    basis = np.zeros((M, nx, ny), dtype=float)
    for m in range(M):
        cx = Lx * (0.17 + 0.66 * ((m * 4 + 1) % 11) / 10.0)
        cy = Ly * (0.17 + 0.66 * ((m * 6 + 3) % 11) / 10.0)
        basis[m] = 18.0 * np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2.0 * (0.085 ** 2)))
    A = _assemble_adv(Lx, Ly, nu, vx, vy, sigma, omega)
    lu = splu(A.tocsc())
    G_full = np.zeros((M, nx, ny), dtype=np.complex128)
    for m in range(M):
        G_full[m] = lu.solve(basis[m].ravel().astype(np.complex128)).reshape(nx, ny)
    K_s = 82
    fc = rng.choice(nx * ny, size=K_s, replace=False)
    sensor_ij = np.column_stack([fc // ny, fc % ny])
    G_sens = G_full[:, sensor_ij[:, 0], sensor_ij[:, 1]].T
    w_gt_gen = np.zeros(M, dtype=np.complex128)
    w_gt_gen[[2, 5, 9]] = [1.25 - 0.75j, -0.95 + 1.05j, 1.05 + 0.65j]
    y_obs = G_sens @ w_gt_gen + (rng.normal(scale=0.015, size=K_s) + 1j * rng.normal(scale=0.015, size=K_s))
    lam2 = 0.08; lam0 = 0.09
    pairs = [[2, 3, 0.23], [5, 6, 0.25], [9, 10, 0.21], [1, 5, 0.19], [0, 8, 0.24]]
    w_opt, j_opt = _solve_exact(G_sens, y_obs, lam2, lam0, pairs)
    psi_opt = np.tensordot(w_opt, G_full, axes=(0, 0))
    prob = {
        "Lx": Lx, "Ly": Ly, "nu": nu, "vx": vx, "vy": vy, "sigma": sigma, "omega": omega,
        "basis": basis, "sensor_ij": sensor_ij, "y_obs_re": y_obs.real.copy(), "y_obs_im": y_obs.imag.copy(),
        "lam2": lam2, "lam0": lam0, "pair_penalties": pairs,
    }
    return prob, w_opt, j_opt, np.abs(psi_opt)

t0 = time.monotonic()
valid = True
max_w_err = 0.0
max_obj_err = 0.0
max_field_err = 0.0
diag_rows = []
for sd in (99431, 99432, 99433):
    pr, w_gt, j_gt, f_gt = _make_priv(sd)
    out = nonhermitian_advection_coupled_select.solve_nonhermitian_coupled_inversion(pr)
    if not isinstance(out, dict) or not {"weights_real", "weights_imag", "objective", "predicted_field_abs"} <= set(out.keys()):
        valid = False; diag_rows.append({"seed": sd, "error": "missing_keys"}); continue
    wr = np.asarray(out["weights_real"], dtype=float)
    wi = np.asarray(out["weights_imag"], dtype=float)
    pf = np.asarray(out["predicted_field_abs"], dtype=float)
    obj = float(out["objective"])
    if wr.shape != (11,) or wi.shape != (11,) or pf.shape != (32, 32) or not (np.all(np.isfinite(wr)) and np.all(np.isfinite(wi)) and np.all(np.isfinite(pf)) and math.isfinite(obj)):
        valid = False; diag_rows.append({"seed": sd, "error": "shape_or_nan"}); continue
    wc = wr + 1j * wi
    w_err = float(np.linalg.norm(wc - w_gt) / np.linalg.norm(w_gt))
    o_err = abs(obj - j_gt)
    f_err = float(np.linalg.norm(pf - f_gt) / np.linalg.norm(f_gt))
    max_w_err = max(max_w_err, w_err)
    max_obj_err = max(max_obj_err, o_err)
    max_field_err = max(max_field_err, f_err)
    diag_rows.append({"seed": sd, "w_rel_l2": round(w_err, 8), "obj_abs_err": round(o_err, 9), "field_rel_l2": round(f_err, 8)})

student_elapsed = time.monotonic() - t0
strict_pass = bool(valid and len(diag_rows) == 3 and max_w_err <= 1e-5 and max_obj_err <= 1e-6 and max_field_err <= 1e-5)
print(json.dumps({
    "strict_pass": strict_pass,
    "reward": 1.0 if strict_pass else 0.0,
    "all_valid": valid,
    "max_weights_rel_l2": round(float(max_w_err), 8),
    "max_objective_abs_err": round(float(max_obj_err), 9),
    "max_field_rel_l2": round(float(max_field_err), 8),
    "student_elapsed_sec": round(float(student_elapsed), 4),
    "instance_diagnostics": diag_rows,
}, allow_nan=False))
''',
    ),
)


EQUAL_INFO_M92_BY_ID: dict[str, HoldoutOODTask] = {t.task_id: t for t in EQUAL_INFO_M92_TASKS}


def equal_info_m92_task_to_frontier_task(ht: HoldoutOODTask) -> m6f.FrontierBenchmarkTask:
    ensure_clean_scipy_sandbox_image()
    return m6f.FrontierBenchmarkTask(
        task_id=ht.task_id,
        short_name=ht.short_name,
        benchmark="Hegel-M92-EqualInfo-4",
        wave=ht.wave,
        scientific_domain=ht.scientific_domain,
        wormhole_family=ht.wormhole_family,
        instruction_text=ht.instruction_text,
        agent_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
        verifier_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
        declared_artifacts=ht.declared_artifacts,
        declared_callable_entrypoints=ht.declared_callable_entrypoints,
        holdout_public_setup_py=ht.public_setup_py,
        agent_cpus=2,
        agent_memory_mb=2560,
        verifier_cpus=2,
        verifier_memory_mb=2560,
        verifier_timeout_sec=120,
        verifier_network_mode="none",
    )


def get_equal_info_m92_frontier_tasks() -> list[m6f.FrontierBenchmarkTask]:
    return [equal_info_m92_task_to_frontier_task(t) for t in EQUAL_INFO_M92_TASKS]


def run_equal_info_m92_private_verifier_by_id(task_id: str, submitted_tar_path: Path | None) -> dict[str, Any]:
    ht = EQUAL_INFO_M92_BY_ID[task_id]
    rep = run_holdout_private_verifier(ht, submitted_tar_path)
    rep["benchmark"] = "Hegel-M92-EqualInfo-4"
    return rep


def run_equal_info_m92_private_verifier_for_container(task_id: str, agent_cid: str) -> dict[str, Any]:
    ht = EQUAL_INFO_M92_BY_ID[task_id]
    with tempfile.TemporaryDirectory(prefix="hegel_eq_m92_ver_") as td:
        tar_p = Path(td) / "submitted.tar"
        m6f.export_declared_artifacts_tar(agent_cid, ht.declared_artifacts, tar_p)
        rep = run_holdout_private_verifier(ht, tar_p)
        rep["benchmark"] = "Hegel-M92-EqualInfo-4"
        return rep
