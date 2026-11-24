"""Hegel M7.3 Pre-Registered 12-Task Unseen Hold-out OOD Blind Scientific Benchmark (`Hegel-OOD-Holdout-12`).

STRICT BLIND BENCHMARK INTEGRITY INVARIANTS:
1. **100% Disjoint Problem Formulations**:
   - Zero overlap in physics equations, schemas, file layouts, or verifiers with ALL 20 `Terminal-Bench-Science` /
     `BioMysteryBench` tasks AND ALL 6 `pt01..pt06` offline pretraining tasks.
   - Spans 6 scientific domains across `Wave 1` (`ho01..ho06`) and `Wave 2` (`ho07..ho12`):
     * Inverse Wave & Tomographic Physics (`ho01` Crosswell Thomsen Anisotropy, `ho07` Speed-Shifted Thermoacoustics)
     * Nonlinear & Memory Dynamical Systems (`ho02` Rational Enzyme Kinetics, `ho08` Prony Viscoelastic Spectrum)
     * Multiscale Advection/Conservation PDEs (`ho03` High-Peclet Upwind Plume, `ho09` Shallow-Water Sill & Gate)
     * Robust Calibration & Safety Control (`ho04` Hard/Soft-Iron Magnetometer, `ho10` Haldane Bioreactor Safety)
     * Latent Deconfounding & Exact Certificates (`ho05` Sparse+Low-Rank Graph, `ho11` Certified FacilityMIP)
     * Quantitative Assay & Multi-Isolate Genomics (`ho06` 384-Well Spatial Edge IC50, `ho12` Multicopy Plasmid Consensus)
2. **Air-Gapped Execution**:
   - Agent executes inside `hegel_clean_scipy_sandbox:v1` (`/holdout_task`) with ONLY `public_setup_py` data materialized.
   - Post-submission grading runs in a SEPARATE fresh `hegel_clean_scipy_sandbox:v1` container receiving ONLY
     `declared_artifacts` and evaluating on unseen private test seeds (`private_verifier_py`).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from time import monotonic
from typing import Any

from . import m6_frontier_real_benchmarks as m6f
from .m7_autogenous_forest_pretraining import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    ensure_clean_scipy_sandbox_image,
)


@dataclass(frozen=True)
class HoldoutOODTask:
    task_id: str
    short_name: str
    wave: str  # "wave_1" | "wave_2"
    scientific_domain: str
    wormhole_family: str
    declared_artifacts: tuple[str, ...]
    instruction_text: str
    public_setup_py: str
    private_verifier_py: str


def build_holdout_ood_12_tasks() -> list[HoldoutOODTask]:
    """Return the pre-registered 12-task `Hegel-OOD-Holdout-12` benchmark (`6` Wave-1 + `6` Wave-2)."""
    return [
        # ====================================================================
        # WAVE 1 (`ho01` .. `ho06`)
        # ====================================================================
        HoldoutOODTask(
            task_id="ho01__crosswell_anisotropic_clock_tomography",
            short_name="crosswell_anisotropic_tomography",
            wave="wave_1",
            scientific_domain="Geophysics / Borehole Traveltime Tomography",
            wormhole_family="fam_guided_wave_shm_localization",
            declared_artifacts=("/holdout_task/tomography_solver.py",),
            instruction_text=(
                "Write `/holdout_task/tomography_solver.py` exposing:\n"
                "  `invert_plume_and_clocks(survey: dict) -> dict`\n"
                "A cross-well seismic survey fires `N_src` sources at `(x=0, z_src)` across `x in [0, W_m]` to `N_rcv` "
                "receivers at `(x=W_m, z_rcv)` (`z in [0, H_m]`).\n"
                "- Background medium has weak elliptical Thomsen anisotropy: group slowness at ray angle `theta` "
                "(`tan(theta) = dz / dx`) is `s_bg(theta) = s0 * (1.0 - aniso_eps * sin(theta)**2)`.\n"
                "- A compact circular CO2 plume centered at `(x_p, z_p)` with Gaussian radius `R_p = 5.5 m` adds slowness "
                "perturbation `ds(x, z) = delta_s * exp(-0.5 * ((x - x_p)**2 + (z - z_p)**2) / R_p**2)` (`delta_s > 0`).\n"
                "- Measured pick time for `(i_src, j_rcv)` is:\n"
                "  `T_obs[i, j] = integral_ray (s_bg(theta_ij) + ds(x, z)) dl + rcv_clock_bias_s[j] + noise`\n"
                "  where `sum(rcv_clock_bias_s) == 0.0`, `|rcv_clock_bias_s[j]| <= 0.004` s, AND `~4%` of picks suffer "
                "cycle-skipping gross outliers (`+-0.006` s)!\n"
                "Inspect `/holdout_task/public_surveys.pkl` and return:\n"
                "  `{'plume_xz_m': [float(x_p), float(z_p)], 'rcv_clock_bias_s': list[float]}`.\n"
                "Target: Plume center error `<= 1.8 m` and clock bias RMSE `<= 0.00025 s` on held-out surveys (`< 15s`)."
            ),
            public_setup_py=r'''
import math, pickle, numpy as np

def make_survey(seed, xp, zp, s0, eps_a, ds_amp):
    rng = np.random.default_rng(seed)
    W, H = 40.0, 80.0
    z_src = np.linspace(6.0, 74.0, 15)
    z_rcv = np.linspace(5.0, 75.0, 16)
    clk = rng.uniform(-0.0032, 0.0032, size=len(z_rcv))
    clk -= np.mean(clk)
    T = np.zeros((len(z_src), len(z_rcv)), dtype=float)
    u_q = np.linspace(0.0, 1.0, 81)
    for i, zs in enumerate(z_src):
        for j, zr in enumerate(z_rcv):
            dx, dz = W, zr - zs
            L = math.hypot(dx, dz)
            sin2 = (dz / L) ** 2
            t_bg = L * s0 * (1.0 - eps_a * sin2)
            xq = u_q * W; zq = zs + u_q * dz
            g = ds_amp * np.exp(-0.5 * ((xq - xp) ** 2 + (zq - zp) ** 2) / (5.5 ** 2))
            t_pl = float(0.5 * np.sum((g[:-1] + g[1:]) * np.diff(u_q)) * L)
            T[i, j] = t_bg + t_pl + clk[j] + rng.normal(0.0, 3.5e-5)
    # 4% cycle-skip outliers
    n_out = 10
    flat_idx = rng.choice(T.size, size=n_out, replace=False)
    for fi in flat_idx:
        T.flat[int(fi)] += float(rng.choice([-0.0055, 0.0055]))
    return {"W_m": W, "H_m": H, "z_src_m": z_src, "z_rcv_m": z_rcv, "T_obs_s": T, "R_plume_m": 5.5}, {"plume_xz_m": [xp, zp], "rcv_clock_bias_s": clk}

pub = [make_survey(11, 16.5, 34.0, 0.00042, 0.14, 6.5e-5), make_survey(12, 26.0, 51.5, 0.00038, 0.18, 7.2e-5)]
with open("/holdout_task/public_surveys.pkl", "wb") as fh:
    pickle.dump([{"survey": s, "gt": g} for s, g in pub], fh)
''',
            private_verifier_py=r'''
import json, math, sys, numpy as np
sys.path.insert(0, "/holdout_task")
import tomography_solver

def make_survey(seed, xp, zp, s0, eps_a, ds_amp):
    rng = np.random.default_rng(seed)
    W, H = 44.0, 84.0
    z_src = np.linspace(6.0, 78.0, 16)
    z_rcv = np.linspace(5.0, 79.0, 17)
    clk = rng.uniform(-0.0035, 0.0035, size=len(z_rcv)); clk -= np.mean(clk)
    T = np.zeros((len(z_src), len(z_rcv)), dtype=float)
    u_q = np.linspace(0.0, 1.0, 81)
    for i, zs in enumerate(z_src):
        for j, zr in enumerate(z_rcv):
            dx, dz = W, zr - zs; L = math.hypot(dx, dz); sin2 = (dz / L) ** 2
            xq = u_q * W; zq = zs + u_q * dz
            g = ds_amp * np.exp(-0.5 * ((xq - xp) ** 2 + (zq - zp) ** 2) / (5.5 ** 2))
            t_pl = float(0.5 * np.sum((g[:-1] + g[1:]) * np.diff(u_q)) * L)
            T[i, j] = L * s0 * (1.0 - eps_a * sin2) + t_pl + clk[j] + rng.normal(0.0, 3.5e-5)
    for fi in rng.choice(T.size, size=12, replace=False):
        T.flat[int(fi)] += float(rng.choice([-0.006, 0.006]))
    return {"W_m": W, "H_m": H, "z_src_m": z_src, "z_rcv_m": z_rcv, "T_obs_s": T, "R_plume_m": 5.5}, np.array([xp, zp]), clk

pos_errs, clk_errs = [], []
for sd, xp, zp, s0, ep, ds in ((801, 14.0, 58.0, 0.00040, 0.16, 6.8e-5), (802, 29.5, 26.5, 0.00045, 0.12, 7.5e-5), (803, 21.0, 43.0, 0.00036, 0.20, 6.2e-5)):
    sv, gt_xz, gt_c = make_survey(sd, xp, zp, s0, ep, ds)
    out = tomography_solver.invert_plume_and_clocks(sv)
    pos_errs.append(float(np.linalg.norm(np.asarray(out["plume_xz_m"], float) - gt_xz)))
    c_hat = np.asarray(out["rcv_clock_bias_s"], float); c_hat = c_hat - np.mean(c_hat)
    clk_errs.append(float(np.sqrt(np.mean((c_hat - gt_c) ** 2))))
passed = bool(max(pos_errs) <= 1.8 and max(clk_errs) <= 0.00025)
print(json.dumps({"strict_pass": passed, "max_pos_err_m": round(max(pos_errs), 3), "max_clk_rmse_s": round(max(clk_errs), 6)}))
''',
        ),

        HoldoutOODTask(
            task_id="ho02__inhibited_enzyme_rational_kinetics",
            short_name="inhibited_enzyme_kinetics",
            wave="wave_1",
            scientific_domain="Systems Biochemistry / Nonlinear ODE Identification",
            wormhole_family="fam_symbolic_and_operator_dynamics",
            declared_artifacts=("/holdout_task/kinetics_id.py",),
            instruction_text=(
                "Write `/holdout_task/kinetics_id.py` exposing:\n"
                "  `identify_and_forecast(dataset: dict) -> dict`\n"
                "A 3-species enzymatic cascade `(S, I, P)` obeys rational product-inhibited mass-balance ODEs:\n"
                "  `dS/dt = -v1 + u_feed(t)`,  `dI/dt = v1 - v2 - k_leak * I`,  `dP/dt = v2`\n"
                "where `v1 = Vmax1 * S / (Km1 + S + alpha_p * P)` (competitive product inhibition by `P`!) and "
                "`v2 = Vmax2 * I / (Km2 + I)`.\n"
                "`dataset` provides 4 noisy training trajectories (`~2.5%` Gaussian observation noise at irregular `t_obs`, "
                "known piecewise-constant `u_feed`), plus an unseen test initial state `y0_test`, `t_test`, and `u_feed_test`.\n"
                "Return `{'params': {'Vmax1': ..., 'Km1': ..., 'alpha_p': ..., 'Vmax2': ..., 'Km2': ..., 'k_leak': ...}, "
                "'forecast_SIP': (len(t_test), 3) float array}`.\n"
                "Inspect `/holdout_task/public_enzyme.pkl`. Target: Held-out trajectory RMSE `<= 0.035` and max relative parameter error `<= 0.12`."
            ),
            public_setup_py=r'''
import pickle, numpy as np
from scipy.integrate import solve_ivp

def sim_enz(params, y0, t_eval, feed_amp, feed_t1):
    Vm1, Km1, ap, Vm2, Km2, kl = params
    def rhs(t, y):
        S, I, P = np.maximum(y, 0.0)
        uf = feed_amp if (2.5 <= t <= feed_t1) else 0.0
        v1 = Vm1 * S / (Km1 + S + ap * P)
        v2 = Vm2 * I / (Km2 + I)
        return [-v1 + uf, v1 - v2 - kl * I, v2]
    sol = solve_ivp(rhs, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.y.T

rng = np.random.default_rng(202)
gt_p = (1.8, 0.65, 1.25, 1.35, 0.45, 0.12)
trajs = []
for y0, fa, ft in (([3.2, 0.0, 0.0], 0.5, 6.0), ([1.4, 0.5, 0.2], 0.0, 5.0), ([2.5, 0.1, 1.8], 0.8, 7.0), ([0.8, 1.2, 0.0], 0.3, 4.5)):
    t_obs = np.sort(np.r_[0.0, rng.uniform(0.15, 11.8, size=58), 12.0])
    clean = sim_enz(gt_p, y0, t_obs, fa, ft)
    noisy = np.maximum(clean + rng.normal(0.0, 0.022, size=clean.shape), 0.0)
    trajs.append({"t_obs": t_obs, "y_obs": noisy, "y0_approx": noisy[0], "feed_amp": fa, "feed_window": [2.5, ft]})
t_te = np.linspace(0.0, 14.0, 71)
gt_te = sim_enz(gt_p, [2.8, 0.2, 0.9], t_te, 0.65, 6.5)
ds = {"train_trajectories": trajs, "y0_test": [2.8, 0.2, 0.9], "t_test": t_te, "test_feed_amp": 0.65, "test_feed_window": [2.5, 6.5]}
with open("/holdout_task/public_enzyme.pkl", "wb") as fh:
    pickle.dump({"dataset": ds, "gt_params": gt_p, "gt_forecast": gt_te}, fh)
''',
            private_verifier_py=r'''
import json, sys, numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, "/holdout_task")
import kinetics_id

def sim_enz(params, y0, t_eval, fa, w0, w1):
    Vm1, Km1, ap, Vm2, Km2, kl = params
    def rhs(t, y):
        S, I, P = np.maximum(y, 0.0)
        uf = fa if (w0 <= t <= w1) else 0.0
        v1 = Vm1 * S / (Km1 + S + ap * P)
        v2 = Vm2 * I / (Km2 + I)
        return [-v1 + uf, v1 - v2 - kl * I, v2]
    return solve_ivp(rhs, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, rtol=1e-8, atol=1e-10).y.T

rng = np.random.default_rng(9202)
gt_p = (2.15, 0.55, 1.55, 1.50, 0.52, 0.16)
names = ("Vmax1", "Km1", "alpha_p", "Vmax2", "Km2", "k_leak")
trajs = []
for y0, fa, ft in (([3.6, 0.0, 0.0], 0.6, 6.5), ([1.2, 0.6, 0.1], 0.0, 5.0), ([2.2, 0.0, 2.2], 0.9, 7.5), ([0.7, 1.5, 0.3], 0.4, 5.5)):
    t_obs = np.sort(np.r_[0.0, rng.uniform(0.15, 11.8, size=60), 12.0])
    clean = sim_enz(gt_p, y0, t_obs, fa, 2.5, ft)
    noisy = np.maximum(clean + rng.normal(0.0, 0.022, size=clean.shape), 0.0)
    trajs.append({"t_obs": t_obs, "y_obs": noisy, "y0_approx": noisy[0], "feed_amp": fa, "feed_window": [2.5, ft]})
t_te = np.linspace(0.0, 15.0, 76)
gt_te = sim_enz(gt_p, [3.0, 0.1, 1.2], t_te, 0.75, 2.5, 7.0)
ds = {"train_trajectories": trajs, "y0_test": [3.0, 0.1, 1.2], "t_test": t_te, "test_feed_amp": 0.75, "test_feed_window": [2.5, 7.0]}
res = kinetics_id.identify_and_forecast(ds)
fc = np.asarray(res["forecast_SIP"], float)
rmse = float(np.sqrt(np.mean((fc - gt_te) ** 2)))
rel_p = max(abs(float(res["params"][k]) - v) / v for k, v in zip(names, gt_p))
passed = bool(rmse <= 0.035 and rel_p <= 0.12)
print(json.dumps({"strict_pass": passed, "forecast_rmse": round(rmse, 5), "max_param_rel_err": round(rel_p, 4)}))
''',
        ),

        HoldoutOODTask(
            task_id="ho03__high_peclet_upwind_plume_inversion",
            short_name="high_peclet_plume_inversion",
            wave="wave_1",
            scientific_domain="Computational Transport PDE / Source Inversion",
            wormhole_family="fam_multiscale_pde_and_combinatorial_physics",
            declared_artifacts=("/holdout_task/peclet_plume.py",),
            instruction_text=(
                "Write `/holdout_task/peclet_plume.py` exposing:\n"
                "  `forward_peclet_field(ux_face, uy_face, diffusivity, kappa, source_map, h) -> np.ndarray`\n"
                "  `invert_point_source(problem: dict) -> dict`\n"
                "On an `N x N` cell-centered grid (`x, y in (0, L)`, `h = L / N`), steady advection-diffusion-decay with "
                "zero-Dirichlet (`c = 0`) ghost cells (`distance h` outside cell centers) satisfies:\n"
                "  `div( u * c - D * grad(c) ) + kappa * c == source_map`\n"
                "using the exact 1D exponential Scharfetter-Gummel / Il'in flux across every face (for face velocity `V_f` from `a -> b`, "
                "`Pe_f = V_f * h / D`, face flux `J_{a->b} = (D / h) * (B(-Pe_f) * c_a - B(Pe_f) * c_b)` where Bernoulli function "
                "`B(z) = z / (exp(z) - 1)` with `B(0) = 1`, and boundary ghost cell `c_ghost = 0`!).\n"
                "`problem` gives `N=96`, `h`, spatially varying divergence-free swirl velocity `(ux_face, uy_face)`, small `D` (`Pe ~ 80`), "
                "`kappa`, and 18 noisy interior sensor readings `sensor_ij`, `c_obs` (including 2 faulty sensor outliers!) from a Gaussian "
                "source `Q * exp(-0.5 * ((X - x_s)**2 + (Y - y_s)**2) / sigma_s**2)`.\n"
                "Return `{'source_xy': [float(x_s), float(y_s)], 'strength_Q': float}` (`pos_err <= 0.025 * L`, `|Q_hat - Q|/Q <= 0.08`, total runtime `< 4.0s`)."
            ),
            public_setup_py=r'''
import pickle, numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def bernoulli(z):
    z = np.asarray(z, dtype=float)
    out = np.ones_like(z)
    small = np.abs(z) < 1e-4
    out[small] = 1.0 - 0.5 * z[small] + (z[small] ** 2) / 12.0
    zs = z[~small]
    out[~small] = np.where(zs > 40.0, zs * np.exp(-zs), np.where(zs < -40.0, -zs, zs / np.expm1(zs)))
    return out

def exact_sg_solve(N, h, ux_f, uy_f, D, kappa, src):
    # ux_f: (N+1, N) faces in x; uy_f: (N, N+1) faces in y
    A = lil_matrix((N * N, N * N), dtype=float)
    for i in range(N):
        for j in range(N):
            row = i * N + j
            diag = kappa
            # Right face (i+1, j)
            pe_r = ux_f[i + 1, j] * h / D
            diag += (D / (h * h)) * bernoulli(-pe_r)
            if i + 1 < N:
                A[row, (i + 1) * N + j] = -(D / (h * h)) * bernoulli(pe_r)
            # Left face (i, j): outward is -x, flux a->left is -J_{left->a}
            pe_l = ux_f[i, j] * h / D
            diag += (D / (h * h)) * bernoulli(pe_l)
            if i - 1 >= 0:
                A[row, (i - 1) * N + j] = -(D / (h * h)) * bernoulli(-pe_l)
            # Top face (i, j+1)
            pe_t = uy_f[i, j + 1] * h / D
            diag += (D / (h * h)) * bernoulli(-pe_t)
            if j + 1 < N:
                A[row, i * N + (j + 1)] = -(D / (h * h)) * bernoulli(pe_t)
            # Bottom face (i, j)
            pe_b = uy_f[i, j] * h / D
            diag += (D / (h * h)) * bernoulli(pe_b)
            if j - 1 >= 0:
                A[row, i * N + (j - 1)] = -(D / (h * h)) * bernoulli(-pe_b)
            A[row, row] = diag
    return spsolve(A.tocsc(), src.ravel()).reshape(N, N)

rng = np.random.default_rng(303)
N = 96; L = 1.0; h = L / N
xc = (np.arange(N) + 0.5) * h
X, Y = np.meshgrid(xc, xc, indexing="ij")
xf = np.arange(N + 1) * h
Xf_x, Yf_x = np.meshgrid(xf, xc, indexing="ij")
Xf_y, Yf_y = np.meshgrid(xc, xf, indexing="ij")
ux_f = 1.6 + 0.4 * np.sin(np.pi * Xf_x) * np.cos(np.pi * Yf_x)
uy_f = 0.7 - 0.4 * np.cos(np.pi * Xf_y) * np.sin(np.pi * Yf_y)
D, kappa, sig_s = 0.0025, 0.25, 0.045
xs, ys, Q = 0.28, 0.36, 4.2
src = Q * np.exp(-0.5 * ((X - xs) ** 2 + (Y - ys) ** 2) / (sig_s ** 2))
c_full = exact_sg_solve(N, h, ux_f, uy_f, D, kappa, src)
s_ij = np.array([[i, j] for i in (18, 34, 50, 66, 80) for j in (20, 42, 64, 82)], dtype=int)[:18]
c_obs = c_full[s_ij[:, 0], s_ij[:, 1]] + rng.normal(0.0, 0.004, size=18)
c_obs[[3, 14]] += np.array([0.85, -0.70])
prob = {"N": N, "L": L, "h": h, "ux_face": ux_f, "uy_face": uy_f, "D": D, "kappa": kappa, "sigma_s": sig_s, "sensor_ij": s_ij, "c_obs": c_obs}
with open("/holdout_task/public_peclet.pkl", "wb") as fh:
    pickle.dump({"problem": prob, "gt_field": c_full, "gt_source_xy": [xs, ys], "gt_Q": Q}, fh)
''',
            private_verifier_py=r'''
import json, sys, time, numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
sys.path.insert(0, "/holdout_task")
import peclet_plume

def bernoulli(z):
    z = np.asarray(z, dtype=float); out = np.ones_like(z); sm = np.abs(z) < 1e-4
    out[sm] = 1.0 - 0.5 * z[sm] + (z[sm] ** 2) / 12.0; zs = z[~sm]
    out[~sm] = np.where(zs > 40.0, zs * np.exp(-zs), np.where(zs < -40.0, -zs, zs / np.expm1(zs)))
    return out

def sg_ref(N, h, ux_f, uy_f, D, kappa, src):
    A = lil_matrix((N * N, N * N), dtype=float)
    for i in range(N):
        for j in range(N):
            r = i * N + j; d = kappa
            pr, pl = ux_f[i + 1, j] * h / D, ux_f[i, j] * h / D
            pt, pb = uy_f[i, j + 1] * h / D, uy_f[i, j] * h / D
            d += (D / (h * h)) * (bernoulli(-pr) + bernoulli(pl) + bernoulli(-pt) + bernoulli(pb))
            if i + 1 < N: A[r, (i + 1) * N + j] = -(D / (h * h)) * bernoulli(pr)
            if i - 1 >= 0: A[r, (i - 1) * N + j] = -(D / (h * h)) * bernoulli(-pl)
            if j + 1 < N: A[r, i * N + j + 1] = -(D / (h * h)) * bernoulli(pt)
            if j - 1 >= 0: A[r, i * N + j - 1] = -(D / (h * h)) * bernoulli(-pb)
            A[r, r] = d
    return spsolve(A.tocsc(), src.ravel()).reshape(N, N)

# 1. Verify forward Scharfetter-Gummel stencil accuracy on a FRESH UNSEEN PDE field (never public_peclet.pkl!)
N_f = 96; L_f = 1.0; h_f = L_f / N_f
xc_f = (np.arange(N_f) + 0.5) * h_f; xf_f = np.arange(N_f + 1) * h_f
Xf, Yf = np.meshgrid(xc_f, xc_f, indexing="ij")
Xfx, Yfx = np.meshgrid(xf_f, xc_f, indexing="ij"); Xfy, Yfy = np.meshgrid(xc_f, xf_f, indexing="ij")
ux_fwd = -1.15 + 0.45 * np.cos(np.pi * Xfx) * np.sin(np.pi * Yfx)
uy_fwd = 0.95 + 0.30 * np.sin(np.pi * Xfy) * np.cos(np.pi * Yfy)
D_fwd, kap_fwd = 0.0028, 0.30
src_fwd = 3.7 * np.exp(-0.5 * ((Xf - 0.62) ** 2 + (Yf - 0.41) ** 2) / (0.05 ** 2))
gt_fwd_priv = sg_ref(N_f, h_f, ux_fwd, uy_fwd, D_fwd, kap_fwd, src_fwd)
c_fwd = np.asarray(peclet_plume.forward_peclet_field(ux_fwd, uy_fwd, D_fwd, kap_fwd, src_fwd, h_f), float)
fwd_rel = float(np.linalg.norm(c_fwd - gt_fwd_priv) / max(np.linalg.norm(gt_fwd_priv), 1e-12))

# 2. Generate unseen inverse problem on shifted velocity & source!
rng = np.random.default_rng(9303)
N = 96; L = 1.0; h = L / N; xc = (np.arange(N) + 0.5) * h; xf = np.arange(N + 1) * h
X, Y = np.meshgrid(xc, xc, indexing="ij")
Xf_x, Yf_x = np.meshgrid(xf, xc, indexing="ij"); Xf_y, Yf_y = np.meshgrid(xc, xf, indexing="ij")
ux_f = 1.35 + 0.35 * np.sin(np.pi * Xf_x) * np.cos(np.pi * Yf_x)
uy_f = -0.55 - 0.35 * np.cos(np.pi * Xf_y) * np.sin(np.pi * Yf_y)
D, kappa, sig_s = 0.0022, 0.25, 0.045
xs, ys, Q = 0.34, 0.64, 5.1
src_u = Q * np.exp(-0.5 * ((X - xs) ** 2 + (Y - ys) ** 2) / (sig_s ** 2))
cu = sg_ref(N, h, ux_f, uy_f, D, kappa, src_u)
s_ij = np.array([[i, j] for i in (20, 38, 56, 72, 84) for j in (16, 36, 54, 74)], dtype=int)[:18]
c_obs = cu[s_ij[:, 0], s_ij[:, 1]] + rng.normal(0.0, 0.004, size=18)
c_obs[[2, 11]] += np.array([-0.9, 1.1])
prob_u = {"N": N, "L": L, "h": h, "ux_face": ux_f, "uy_face": uy_f, "D": D, "kappa": kappa, "sigma_s": sig_s, "sensor_ij": s_ij, "c_obs": c_obs}
t0 = time.monotonic()
inv = peclet_plume.invert_point_source(prob_u)
dt = time.monotonic() - t0
pos_err = float(np.linalg.norm(np.asarray(inv["source_xy"], float) - np.array([xs, ys])))
q_err = abs(float(inv["strength_Q"]) - Q) / Q
passed = bool(np.isfinite(fwd_rel) and np.isfinite(pos_err) and np.isfinite(q_err) and fwd_rel <= 1e-5 and pos_err <= 0.025 and q_err <= 0.08 and dt <= 6.0)
print(json.dumps({"strict_pass": passed, "fwd_rel_err": round(fwd_rel, 7), "pos_err": round(pos_err, 4), "q_rel_err": round(q_err, 4), "wall_s": round(dt, 3)}))
''',
        ),

        HoldoutOODTask(
            task_id="ho04__magnetometer_hard_soft_iron_calibration",
            short_name="magnetometer_ellipsoid_calibration",
            wave="wave_1",
            scientific_domain="Navigation & Sensor Fusion / Ellipsoid Calibration",
            wormhole_family="fam_robust_control_and_safety",
            declared_artifacts=("/holdout_task/mag_calibrator.py",),
            instruction_text=(
                "Write `/holdout_task/mag_calibrator.py` exposing:\n"
                "  `calibrate_and_estimate_heading(session: dict) -> dict`\n"
                "A 3-axis strapdown magnetometer mounted on an autonomous vehicle measures:\n"
                "  `m_raw[k] = S_soft @ (R_body_from_ned[k] @ B_ned_uT) + b_hard + noise` (`k = 0..K-1`)\n"
                "where `S_soft` is a unknown symmetric positive-definite `3x3` soft-iron matrix with **`det(S_soft) == 1.0`** "
                "(fixes scale ambiguity!), `b_hard` is the `(3,)` hard-iron bias in `uT`, `B_ned_uT = [B_north, 0.0, B_down]` "
                "(`B_north > 0` horizontal geomagnetic component in NED), and `4%` of calibration samples suffer transient "
                "current-spike dipole interference (`outliers`)!\n"
                "During calibration (`calib_m_raw`, `calib_roll_pitch_rad`), ONLY roll & pitch (`phi, theta`) are known "
                "(yaw `psi` is UNKNOWN during calibration, so `||S_soft^{-1} (m_raw - b_hard)||_2 == ||B_ned||_2` AND horizontal/vertical "
                "projection after roll-pitch leveling pins down `S_soft` and `b_hard` without rotational gauge ambiguity!).\n"
                "Then on `test_m_raw` and `test_roll_pitch_rad`, compute calibrated magnetic yaw `heading_deg` in `(-180, 180]` "
                "(`atan2(-m_level_y, m_level_x)` in degrees).\n"
                "Inspect `/holdout_task/public_mag.pkl`. Target: Held-out circular heading RMSE `<= 0.45 deg` and `||b_hard_hat - b_hard||_2 <= 0.35 uT`."
            ),
            public_setup_py=r'''
import math, pickle, numpy as np

def rot_zyx(yaw, pitch, roll):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    # NED to body is (Rz @ Ry @ Rx).T
    return (Rz @ Ry @ Rx).T

def gen_mag_session(seed):
    rng = np.random.default_rng(seed)
    A = np.array([[1.18, 0.07, -0.04], [0.07, 0.92, 0.05], [-0.04, 0.05, 0.94]])
    S = A / (np.linalg.det(A) ** (1.0 / 3.0))
    b_hard = np.array([28.5, -41.0, 16.2])
    B_ned = np.array([23.5, 0.0, 42.0])
    K = 420
    yaws = rng.uniform(-math.pi, math.pi, size=K)
    pitches = rng.uniform(-0.65, 0.65, size=K)
    rolls = rng.uniform(-0.75, 0.75, size=K)
    m_cal = np.zeros((K, 3))
    for k in range(K):
        R_bn = rot_zyx(yaws[k], pitches[k], rolls[k])
        m_cal[k] = S @ (R_bn @ B_ned) + b_hard + rng.normal(0.0, 0.08, size=3)
    sp = rng.choice(K, size=16, replace=False)
    m_cal[sp] += rng.normal(0.0, 18.0, size=(16, 3))

    K_te = 120
    y_te = np.linspace(-math.pi + 0.1, math.pi - 0.1, K_te)
    p_te = 0.35 * np.sin(2 * y_te)
    r_te = 0.40 * np.cos(3 * y_te)
    m_te = np.array([S @ (rot_zyx(y_te[k], p_te[k], r_te[k]) @ B_ned) + b_hard + rng.normal(0.0, 0.06, size=3) for k in range(K_te)])
    sess = {
        "calib_m_raw": m_cal,
        "calib_roll_pitch_rad": np.column_stack([rolls, pitches]),
        "test_m_raw": m_te,
        "test_roll_pitch_rad": np.column_stack([r_te, p_te]),
    }
    return sess, {"S_soft": S, "b_hard": b_hard, "gt_heading_deg": np.degrees(y_te)}

s0, g0 = gen_mag_session(404)
with open("/holdout_task/public_mag.pkl", "wb") as fh:
    pickle.dump({"session": s0, "gt": g0}, fh)
''',
            private_verifier_py=r'''
import json, math, sys, numpy as np
sys.path.insert(0, "/holdout_task")
import mag_calibrator

def rot_zyx(yaw, pitch, roll):
    cy, sy = math.cos(yaw), math.sin(yaw); cp, sp = math.cos(pitch), math.sin(pitch); cr, sr = math.cos(roll), math.sin(roll)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])).T

rng = np.random.default_rng(9404)
A = np.array([[0.88, -0.08, 0.05], [-0.08, 1.16, -0.03], [0.05, -0.03, 1.01]])
S = A / (np.linalg.det(A) ** (1.0 / 3.0))
b_hard = np.array([-34.2, 22.8, -19.5])
B_ned = np.array([21.0, 0.0, 45.5])
K = 450
yaws = rng.uniform(-math.pi, math.pi, size=K); pitches = rng.uniform(-0.65, 0.65, size=K); rolls = rng.uniform(-0.75, 0.75, size=K)
m_cal = np.array([S @ (rot_zyx(yaws[k], pitches[k], rolls[k]) @ B_ned) + b_hard + rng.normal(0.0, 0.08, size=3) for k in range(K)])
sp = rng.choice(K, size=18, replace=False)
m_cal[sp] += rng.normal(0.0, 20.0, size=(18, 3))
K_te = 140
y_te = np.linspace(-math.pi + 0.08, math.pi - 0.08, K_te)
p_te = 0.30 * np.cos(2 * y_te); r_te = -0.38 * np.sin(2.5 * y_te)
m_te = np.array([S @ (rot_zyx(y_te[k], p_te[k], r_te[k]) @ B_ned) + b_hard + rng.normal(0.0, 0.06, size=3) for k in range(K_te)])

res = mag_calibrator.calibrate_and_estimate_heading({
    "calib_m_raw": m_cal, "calib_roll_pitch_rad": np.column_stack([rolls, pitches]),
    "test_m_raw": m_te, "test_roll_pitch_rad": np.column_stack([r_te, p_te]),
})
b_err = float(np.linalg.norm(np.asarray(res["b_hard"], float) - b_hard))
dh = (np.asarray(res["heading_deg"], float) - np.degrees(y_te) + 180.0) % 360.0 - 180.0
h_rmse = float(np.sqrt(np.mean(dh ** 2)))
passed = bool(h_rmse <= 0.45 and b_err <= 0.35)
print(json.dumps({"strict_pass": passed, "heading_rmse_deg": round(h_rmse, 4), "b_hard_err_uT": round(b_err, 4)}))
''',
        ),

        HoldoutOODTask(
            task_id="ho05__latent_confounded_graphical_precision",
            short_name="latent_confounded_precision_graph",
            wave="wave_1",
            scientific_domain="Statistical Learning / Sparse + Low-Rank Graphical Models",
            wormhole_family="fam_sparse_assimilation_and_latent_recovery",
            declared_artifacts=("/holdout_task/deconfound_graph.py",),
            instruction_text=(
                "Write `/holdout_task/deconfound_graph.py` exposing:\n"
                "  `recover_sparse_precision_support(X_samples: np.ndarray) -> dict`\n"
                "Observed data `X_samples` (`n x p`, `p = 24`, `n = 900`) are generated from a Gaussian graphical model "
                "with sparse conditional-independence precision matrix `S_true` (`~26` undirected off-diagonal edges) "
                "CONFOUNDED by `r = 2` unobserved global latent factors (`h ~ N(0, I_2)` affecting nearly all variables, so the "
                "marginal precision matrix `Sigma_obs^{-1} = S_true - L_lowrank` appears completely DENSE under naive Graphical Lasso / partial correlation!).\n"
                "Decompose `Sigma_obs^{-1} = S - L` (`S` sparse symmetric positive-definite, `L >= 0` rank-`r` low-rank confounder!) "
                "and return `{'adjacency': (p, p) bool symmetric hollow matrix, 'sparse_precision': (p, p) float array}`.\n"
                "Inspect `/holdout_task/public_graph.npz`. Target: Off-diagonal edge recovery `F1 >= 0.86` on unseen graphs."
            ),
            public_setup_py=r'''
import numpy as np
rng = np.random.default_rng(505)
p, n = 24, 900
S = np.eye(p) * 2.2
edges = [(i, (i + 1) % p) for i in range(p)] + [(0, 6), (3, 12), (8, 17), (11, 20)]
adj = np.zeros((p, p), dtype=bool)
for u, v in edges:
    w = float(rng.choice([-0.52, 0.52]))
    S[u, v] = S[v, u] = w
    adj[u, v] = adj[v, u] = True
# Latent loading B: (p, 2) -> marginal cov = S^{-1} + S^{-1} B B^T S^{-1} (or X = X_clean + H @ F.T -> Sigma = S^{-1} + F F^T -> inv = S - S F (I + F^T S F)^{-1} F^T S, exact S - L!)
F = rng.normal(0.0, 0.85, size=(p, 2))
Sigma = np.linalg.inv(S) + F @ F.T
X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
np.savez("/holdout_task/public_graph.npz", X=X, gt_adj=adj, gt_S=S)
''',
            private_verifier_py=r'''
import json, sys, numpy as np
sys.path.insert(0, "/holdout_task")
import deconfound_graph

rng = np.random.default_rng(9505)
f1s = []
prec_valid_all = True
data_dep_all = True
for sd in (1, 2):
    p, n = 24, 950
    S_base = np.eye(p) * 2.35
    base_edges = [(i, (i + 1) % p) for i in range(p)] + [(0, 8), (3, 14), (6, 17), (10, 21)]
    adj_base = np.zeros((p, p), dtype=bool)
    for u, v in base_edges:
        w = float(rng.choice([-0.54, 0.54]))
        S_base[u, v] = S_base[v, u] = w
        adj_base[u, v] = adj_base[v, u] = True
    # Secret random vertex permutation eliminates any fixed-ring / index-template shortcut!
    perm = rng.permutation(p)
    S = S_base[np.ix_(perm, perm)]
    adj = adj_base[np.ix_(perm, perm)]
    F = rng.normal(0.0, 0.90, size=(p, 2))
    Sigma = np.linalg.inv(S) + F @ F.T
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    res = deconfound_graph.recover_sparse_precision_support(X.copy())
    if not isinstance(res, dict) or "adjacency" not in res or "sparse_precision" not in res:
        prec_valid_all = False
        f1s.append(0.0)
        continue
    pa = np.asarray(res["adjacency"], dtype=bool)
    S_hat = np.asarray(res["sparse_precision"], dtype=float)
    if (
        pa.shape != (p, p)
        or S_hat.shape != (p, p)
        or np.any(np.diag(pa))
        or not np.array_equal(pa, pa.T)
        or not np.all(np.isfinite(S_hat))
        or not np.allclose(S_hat, S_hat.T, atol=1e-3)
        or float(np.linalg.eigvalsh(0.5 * (S_hat + S_hat.T))[0]) <= 1e-6
        or (np.any(pa) and float(np.min(np.abs(S_hat[pa]))) <= 1e-5)
    ):
        prec_valid_all = False
    # Verify genuine data dependence: permuting columns of X must correspondingly change predicted support!
    alt_perm = np.roll(np.arange(p), 5)
    res_alt = deconfound_graph.recover_sparse_precision_support(X[:, alt_perm].copy())
    pa_alt = np.asarray(res_alt.get("adjacency", np.zeros_like(pa)), dtype=bool)
    if np.array_equal(pa, pa_alt):
        data_dep_all = False
    iu = np.triu_indices(p, 1)
    tp = int(np.sum(pa[iu] & adj[iu])); fp = int(np.sum(pa[iu] & ~adj[iu])); fn = int(np.sum(~pa[iu] & adj[iu]))
    f1s.append((2.0 * tp) / max(2 * tp + fp + fn, 1))
passed = bool(prec_valid_all and data_dep_all and len(f1s) == 2 and min(f1s) >= 0.86)
print(json.dumps({"strict_pass": passed, "prec_valid": bool(prec_valid_all), "data_dependent": bool(data_dep_all), "min_edge_f1": round(min(f1s), 4), "f1s": [round(v, 4) for v in f1s]}))
''',
        ),

        HoldoutOODTask(
            task_id="ho06__plate_evaporation_deconfounded_ic50",
            short_name="plate_evaporation_ic50_assay",
            wave="wave_1",
            scientific_domain="Quantitative Pharmacology / Confounded High-Throughput Assay",
            wormhole_family="fam_biomystery_structural_and_regulatory",
            declared_artifacts=("/holdout_task/assay_ic50.py",),
            instruction_text=(
                "Write `/holdout_task/assay_ic50.py` exposing:\n"
                "  `estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]`\n"
                "A 384-well (`16 rows x 24 cols`) viability screen assays `12` compounds across `4` plates.\n"
                "- Each plate suffers a multiplicative **2D perimeter evaporation & illumination gradient** `G_plate(row, col)` "
                "(smooth across rows/cols, elevating outer-edge fluorescence by up to `+42%`!) AND plate-specific baseline background `B_plate`:\n"
                "  `Readout = B_plate + G_plate(r, c) * Signal_max_plate * Viability(compound, log10_conc_uM) + noise`\n"
                "- Each plate includes `DMSO` negative-control wells (`viability == 1.0`) and `POS_KILL` positive-control wells "
                "(`viability == 0.0`) scattered across both interior and border rows/cols, PLUS `~3%` dead/clogged pipetting outlier wells!\n"
                "- True Hill law: `Viability(c) = 1.0 / (1.0 + 10.0 ** (hill_slope * (log10_conc_uM - log10_ic50_uM)))`.\n"
                "- Note: Potent vs weak compounds were non-randomly plated near borders vs center (`Simpson's spatial confounding`!), "
                "so un-deconfounded raw plate normalization biases `log10_ic50_uM` by `> 0.45` log units!\n"
                "Inspect `/holdout_task/public_screen.pkl` and return `{compound_id: {'log10_ic50_uM': float, 'hill_slope': float}}`.\n"
                "Target: Max `|log10_ic50_hat - log10_ic50| <= 0.09` and max `|hill_slope_hat - hill_slope| <= 0.18` across all 12 compounds."
            ),
            public_setup_py=r'''
import pickle, numpy as np

def make_screen(seed):
    rng = np.random.default_rng(seed)
    compounds = {f"CMP_{i:02d}": {"log10_ic50_uM": round(-2.2 + 0.36 * i, 3), "hill_slope": round(0.85 + 0.11 * (i % 5), 3)} for i in range(12)}
    plates = []
    concs = np.linspace(-3.0, 2.0, 8)
    for p_idx in range(4):
        R, C = np.meshgrid(np.arange(16, dtype=float), np.arange(24, dtype=float), indexing="ij")
        d_edge = np.minimum(np.minimum(R, 15.0 - R), np.minimum(C, 23.0 - C))
        G = 1.0 + 0.42 * np.exp(-d_edge / 1.8) + 0.06 * (R / 15.0) - 0.05 * (C / 23.0)
        B = 140.0 + 35.0 * p_idx
        Smax = 1800.0 * (1.0 + 0.12 * p_idx)
        wells = []
        used = set()
        # Place controls across border & interior
        for r in range(0, 16, 3):
            for c in range(0, 24, 4):
                used.add((r, c))
                kind = "DMSO" if (r + c) % 2 == 0 else "POS_KILL"
                v = 1.0 if kind == "DMSO" else 0.0
                y = B + G[r, c] * Smax * v + rng.normal(0.0, 14.0)
                wells.append({"row": r, "col": c, "well_type": kind, "compound_id": None, "log10_conc_uM": None, "fluorescence": float(y)})
        # Confounded spatial layout: early compounds near border, late compounds near center!
        free_cells = [(r, c) for r in range(16) for c in range(24) if (r, c) not in used]
        free_cells.sort(key=lambda rc: min(rc[0], 15 - rc[0], rc[1], 23 - rc[1]))
        ptr = 0
        for c_i, (cid, par) in enumerate(compounds.items()):
            for rep in range(2):
                for lc in concs:
                    r, c = free_cells[ptr]; ptr += 1
                    viab = 1.0 / (1.0 + 10.0 ** (par["hill_slope"] * (lc - par["log10_ic50_uM"])))
                    y = B + G[r, c] * Smax * viab + rng.normal(0.0, 14.0)
                    if rng.uniform() < 0.03:
                        y += float(rng.choice([-950.0, 950.0]))
                    wells.append({"row": r, "col": c, "well_type": "COMPOUND", "compound_id": cid, "log10_conc_uM": float(lc), "fluorescence": float(y)})
        plates.append({"plate_id": f"P{p_idx}", "wells": wells})
    return plates, compounds

pl, gt = make_screen(606)
with open("/holdout_task/public_screen.pkl", "wb") as fh:
    pickle.dump({"plates": pl, "gt": gt}, fh)
''',
            private_verifier_py=r'''
import json, sys, numpy as np
sys.path.insert(0, "/holdout_task")
import assay_ic50

rng = np.random.default_rng(9606)
compounds = {f"CMP_{i:02d}": {"log10_ic50_uM": round(-2.0 + 0.34 * i, 3), "hill_slope": round(0.90 + 0.12 * ((i + 2) % 5), 3)} for i in range(12)}
plates = []
concs = np.linspace(-3.0, 2.0, 8)
for p_idx in range(4):
    R, C = np.meshgrid(np.arange(16, dtype=float), np.arange(24, dtype=float), indexing="ij")
    d_edge = np.minimum(np.minimum(R, 15.0 - R), np.minimum(C, 23.0 - C))
    G = 1.0 + 0.45 * np.exp(-d_edge / 1.9) - 0.05 * (R / 15.0) + 0.07 * (C / 23.0)
    B = 160.0 + 30.0 * p_idx; Smax = 1950.0 * (1.0 + 0.10 * p_idx)
    wells = []; used = set()
    for r in range(0, 16, 3):
        for c in range(0, 24, 4):
            used.add((r, c))
            kind = "DMSO" if (r + c) % 2 == 0 else "POS_KILL"
            v = 1.0 if kind == "DMSO" else 0.0
            wells.append({"row": r, "col": c, "well_type": kind, "compound_id": None, "log10_conc_uM": None, "fluorescence": float(B + G[r, c] * Smax * v + rng.normal(0.0, 15.0))})
    free_cells = [(r, c) for r in range(16) for c in range(24) if (r, c) not in used]
    free_cells.sort(key=lambda rc: min(rc[0], 15 - rc[0], rc[1], 23 - rc[1]))
    ptr = 0
    for cid, par in compounds.items():
        for _ in range(2):
            for lc in concs:
                r, c = free_cells[ptr]; ptr += 1
                viab = 1.0 / (1.0 + 10.0 ** (par["hill_slope"] * (lc - par["log10_ic50_uM"])))
                y = B + G[r, c] * Smax * viab + rng.normal(0.0, 15.0)
                if rng.uniform() < 0.03:
                    y += float(rng.choice([-1000.0, 1000.0]))
                wells.append({"row": r, "col": c, "well_type": "COMPOUND", "compound_id": cid, "log10_conc_uM": float(lc), "fluorescence": float(y)})
    plates.append({"plate_id": f"P{p_idx}", "wells": wells})

est = assay_ic50.estimate_compound_hill_parameters(plates)
ic_errs = [abs(float(est[k]["log10_ic50_uM"]) - v["log10_ic50_uM"]) for k, v in compounds.items()]
sl_errs = [abs(float(est[k]["hill_slope"]) - v["hill_slope"]) for k, v in compounds.items()]
passed = bool(max(ic_errs) <= 0.09 and max(sl_errs) <= 0.18)
print(json.dumps({"strict_pass": passed, "max_ic50_err": round(max(ic_errs), 4), "max_slope_err": round(max(sl_errs), 4)}))
''',
        ),

        # ====================================================================
        # WAVE 2 (`ho07` .. `ho12`)
        # ====================================================================
        HoldoutOODTask(
            task_id="ho07__thermoacoustic_speed_shifted_ring",
            short_name="thermoacoustic_speed_shifted_ring",
            wave="wave_2",
            scientific_domain="Medical Physics / Speed-Uncertain Thermoacoustic Localization",
            wormhole_family="fam_guided_wave_shm_localization",
            declared_artifacts=("/holdout_task/thermoacoustic_recon.py",),
            instruction_text=(
                "Write `/holdout_task/thermoacoustic_recon.py` exposing:\n"
                "  `reconstruct_two_absorbers_mm(scan: dict) -> dict`\n"
                "A circular 16-transducer thermoacoustic ring (`radius_mm = 45.0`, `fs_hz = 20_000_000`) images two compact "
                "optical absorbers inside a tissue phantom (`r < 32 mm`).\n"
                "- Nominal calibration water-bath sound speed was `c_nominal_m_per_s = 1480.0`, BUT bath temperature changed "
                "between calibration and scan so true `c_true_m_per_s in [1435, 1545]` is UNKNOWN (must be jointly calibrated "
                "from multi-angle arrival focus coherence!).\n"
                "- Furthermore, `2` of the `16` transducers have **polarity-inverted (`-1x`) & gain-drifted** preamplifiers, "
                "and each bipolar N-wave pulse (`p(t) ~ -(t - r/c) * exp(-0.5 * ((t - r/c)/tau)^2)`) reflects off a known "
                "acoustic reflector pin at `calib_pin_xy_mm`!\n"
                "Inspect `/holdout_task/public_tat.pkl` and return:\n"
                "  `{'sound_speed_m_per_s': float, 'absorbers_xy_mm': [[x1, y1], [x2, y2]]}` (sorted by `x`).\n"
                "Target: Hausdorff absorber localization error `<= 0.85 mm` and sound-speed error `<= 2.5 m/s`!"
            ),
            public_setup_py=r'''
import math, pickle, numpy as np

def make_tat(seed, c_true, p1, p2, inv_sensors):
    rng = np.random.default_rng(seed)
    fs = 20_000_000.0
    n_t = 1600
    t = np.arange(n_t) / fs
    ang = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    s_xy = np.column_stack([45.0 * np.cos(ang), 45.0 * np.sin(ang)])
    pin_xy = np.array([0.0, -34.0])
    gains = np.exp(rng.normal(0.0, 0.15, size=16))
    for k in inv_sensors:
        gains[k] *= -1.35
    tau = 2.2e-7
    sig = np.zeros((n_t, 16), dtype=float)
    sources = [(pin_xy, 0.55), (np.array(p1), 1.0), (np.array(p2), 0.78)]
    for j in range(16):
        w = np.zeros(n_t)
        for src_pt, amp in sources:
            d_m = float(np.linalg.norm(s_xy[j] - src_pt)) * 1e-3
            dt = t - d_m / c_true
            w += amp * (-(dt / tau) * np.exp(-0.5 * (dt / tau) ** 2)) / math.sqrt(max(d_m, 0.005))
        sig[:, j] = gains[j] * w + rng.normal(0.0, 0.08, size=n_t)
    scan = {"fs_hz": fs, "sensor_xy_mm": s_xy, "calib_pin_xy_mm": pin_xy, "c_nominal_m_per_s": 1480.0, "signals": sig}
    pts = sorted([list(p1), list(p2)], key=lambda q: q[0])
    return scan, {"sound_speed_m_per_s": c_true, "absorbers_xy_mm": pts}

pub = [make_tat(71, 1522.0, [-11.5, 8.0], [14.0, -5.5], [2, 11]), make_tat(72, 1448.0, [-8.0, -12.5], [16.5, 10.5], [5, 14])]
with open("/holdout_task/public_tat.pkl", "wb") as fh:
    pickle.dump([{"scan": s, "gt": g} for s, g in pub], fh)
''',
            private_verifier_py=r'''
import json, math, sys, numpy as np
sys.path.insert(0, "/holdout_task")
import thermoacoustic_recon

def make_tat(seed, c_true, p1, p2, inv_sensors):
    rng = np.random.default_rng(seed)
    fs = 20_000_000.0; n_t = 1600; t = np.arange(n_t) / fs
    ang = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    s_xy = np.column_stack([45.0 * np.cos(ang), 45.0 * np.sin(ang)])
    pin_xy = np.array([0.0, -34.0])
    gains = np.exp(rng.normal(0.0, 0.16, size=16))
    for k in inv_sensors: gains[k] *= -1.4
    tau = 2.2e-7; sig = np.zeros((n_t, 16))
    for j in range(16):
        w = np.zeros(n_t)
        for src_pt, amp in ((pin_xy, 0.55), (np.array(p1), 0.95), (np.array(p2), 0.72)):
            d_m = float(np.linalg.norm(s_xy[j] - src_pt)) * 1e-3; dt = t - d_m / c_true
            w += amp * (-(dt / tau) * np.exp(-0.5 * (dt / tau) ** 2)) / math.sqrt(max(d_m, 0.005))
        sig[:, j] = gains[j] * w + rng.normal(0.0, 0.08, size=n_t)
    return {"fs_hz": fs, "sensor_xy_mm": s_xy, "calib_pin_xy_mm": pin_xy, "c_nominal_m_per_s": 1480.0, "signals": sig}, c_true, np.array(sorted([list(p1), list(p2)], key=lambda q: q[0]))

p_errs, c_errs = [], []
for sd, ct, a1, a2, invs in ((971, 1534.5, [-14.2, 6.5], [12.8, 13.5], [1, 9]), (972, 1441.0, [-9.5, -14.0], [18.0, -4.5], [4, 13])):
    sc, gt_c, gt_pts = make_tat(sd, ct, a1, a2, invs)
    res = thermoacoustic_recon.reconstruct_two_absorbers_mm(sc)
    c_errs.append(abs(float(res["sound_speed_m_per_s"]) - gt_c))
    pr = np.array(sorted([list(v) for v in res["absorbers_xy_mm"]], key=lambda q: q[0]), float)
    e_direct = max(np.linalg.norm(pr[0] - gt_pts[0]), np.linalg.norm(pr[1] - gt_pts[1]))
    e_swap = max(np.linalg.norm(pr[0] - gt_pts[1]), np.linalg.norm(pr[1] - gt_pts[0]))
    p_errs.append(float(min(e_direct, e_swap)))
passed = bool(max(p_errs) <= 0.85 and max(c_errs) <= 2.5)
print(json.dumps({"strict_pass": passed, "max_absorber_err_mm": round(max(p_errs), 4), "max_c_err": round(max(c_errs), 4)}))
''',
        ),

        HoldoutOODTask(
            task_id="ho08__prony_viscoelastic_memory_inversion",
            short_name="prony_viscoelastic_memory",
            wave="wave_2",
            scientific_domain="Rheology / Mori-Zwanzig Prony Memory Kernel Identification",
            wormhole_family="fam_symbolic_and_operator_dynamics",
            declared_artifacts=("/holdout_task/prony_rheology.py",),
            instruction_text=(
                "Write `/holdout_task/prony_rheology.py` exposing:\n"
                "  `fit_prony_and_predict_stress(experiment: dict) -> dict`\n"
                "A linear viscoelastic polymer with Boltzmann hereditary integral stress:\n"
                "  `sigma(t) = G_inf * eps(t) + sum_{k=1..M} G_k * int_0^t exp(-(t - s)/tau_k) * (deps/ds) ds` (`G_inf >= 0, G_k >= 0`)\n"
                "is probed on `train_t`, `train_strain` (`eps(0) == 0`), where `train_stress_raw` is corrupted by an unknown "
                "affine load-cell thermal drift `b0 + b1 * t` PLUS 2% spike outliers!\n"
                "Given candidate relaxation-time grid range `[0.02, 20.0] s` and unseen multi-frequency chirp/step `test_t`, `test_strain` "
                "(zero thermal drift on target prediction!), predict clean physical stress `test_stress` and equilibrium modulus `G_inf`.\n"
                "Inspect `/holdout_task/public_rheology.pkl`. Target: Unseen `test_stress` relative RMSE `<= 0.022` and `|G_inf_hat - G_inf| / G_inf <= 0.06`."
            ),
            public_setup_py=r'''
import pickle, numpy as np

def sim_prony(t, eps, G_inf, Gs, taus):
    dt = np.diff(t)
    deps = np.diff(eps)
    sig = G_inf * eps.copy()
    for gk, tk in zip(Gs, taus):
        z = np.zeros_like(t)
        for i in range(len(dt)):
            decay = np.exp(-dt[i] / tk)
            # Exact piecewise-linear strain increment convolution over [t_i, t_{i+1}]
            rate = deps[i] / max(dt[i], 1e-12)
            z[i + 1] = decay * z[i] + tk * (1.0 - decay) * rate
        sig += gk * z
    return sig

rng = np.random.default_rng(808)
t_tr = np.linspace(0.0, 30.0, 601)
eps_tr = np.where(t_tr < 2.0, 0.05 * (t_tr / 2.0), np.where(t_tr < 14.0, 0.05, 0.05 - 0.03 * np.sin(0.6 * (t_tr - 14.0))))
G_inf = 12.5; Gs = [28.0, 16.5, 9.0]; taus = [0.18, 1.4, 8.5]
clean_tr = sim_prony(t_tr, eps_tr, G_inf, Gs, taus)
raw_tr = clean_tr + (-1.8 + 0.14 * t_tr) + rng.normal(0.0, 0.015, size=len(t_tr))
sp = rng.choice(np.arange(10, 590), size=12, replace=False)
raw_tr[sp] += rng.choice([-4.5, 4.5], size=12)

t_te = np.linspace(0.0, 24.0, 481)
eps_te = 0.04 * np.sin(0.35 * t_te + 0.03 * t_te ** 2) * (1.0 - np.exp(-t_te / 1.0))
gt_te = sim_prony(t_te, eps_te, G_inf, Gs, taus)
with open("/holdout_task/public_rheology.pkl", "wb") as fh:
    pickle.dump({"experiment": {"train_t": t_tr, "train_strain": eps_tr, "train_stress_raw": raw_tr, "test_t": t_te, "test_strain": eps_te}, "gt_G_inf": G_inf, "gt_test_stress": gt_te}, fh)
''',
            private_verifier_py=r'''
import json, sys, numpy as np
sys.path.insert(0, "/holdout_task")
import prony_rheology

def sim_prony(t, eps, G_inf, Gs, taus):
    dt = np.diff(t); deps = np.diff(eps); sig = G_inf * eps.copy()
    for gk, tk in zip(Gs, taus):
        z = np.zeros_like(t)
        for i in range(len(dt)):
            dec = np.exp(-dt[i] / tk)
            z[i + 1] = dec * z[i] + tk * (1.0 - dec) * (deps[i] / max(dt[i], 1e-12))
        sig += gk * z
    return sig

rng = np.random.default_rng(9808)
t_tr = np.linspace(0.0, 32.0, 641)
eps_tr = np.where(t_tr < 1.8, 0.06 * (t_tr / 1.8), np.where(t_tr < 15.0, 0.06, 0.06 - 0.035 * np.sin(0.55 * (t_tr - 15.0))))
G_inf = 15.2; Gs = [24.0, 19.0, 11.5]; taus = [0.14, 1.15, 7.2]
raw_tr = sim_prony(t_tr, eps_tr, G_inf, Gs, taus) + (2.4 - 0.19 * t_tr) + rng.normal(0.0, 0.015, size=len(t_tr))
sp = rng.choice(np.arange(15, 625), size=13, replace=False)
raw_tr[sp] += rng.choice([-5.0, 5.0], size=13)

t_te = np.linspace(0.0, 25.0, 501)
eps_te = 0.045 * np.sin(0.4 * t_te + 0.025 * t_te ** 2) * (1.0 - np.exp(-t_te / 0.8))
gt_te = sim_prony(t_te, eps_te, G_inf, Gs, taus)
res = prony_rheology.fit_prony_and_predict_stress({"train_t": t_tr, "train_strain": eps_tr, "train_stress_raw": raw_tr, "test_t": t_te, "test_strain": eps_te})
pred = np.asarray(res["test_stress"], float)
rel_rmse = float(np.sqrt(np.mean((pred - gt_te) ** 2)) / np.sqrt(np.mean(gt_te ** 2)))
g_err = abs(float(res["G_inf"]) - G_inf) / G_inf
passed = bool(rel_rmse <= 0.022 and g_err <= 0.06)
print(json.dumps({"strict_pass": passed, "rel_rmse": round(rel_rmse, 5), "G_inf_rel_err": round(g_err, 5)}))
''',
        ),

        HoldoutOODTask(
            task_id="ho09__shallow_water_sill_and_gate_inversion",
            short_name="shallow_water_sill_inversion",
            wave="wave_2",
            scientific_domain="Environmental Fluid Mechanics / Hyperbolic Conservation Inversion",
            wormhole_family="fam_symbolic_and_operator_dynamics",
            declared_artifacts=("/holdout_task/canal_inversion.py",),
            instruction_text=(
                "Write `/holdout_task/canal_inversion.py` exposing:\n"
                "  `invert_canal_regime(obs_bundle: dict) -> dict`\n"
                "A 1D frictionless shallow-water canal (`N = 80` finite-volume cells on `[0, 800] m`, `dx = 10 m`, `g = 9.81`, "
                "reflective solid right wall `u = 0` at `x = 800 m`) evolves `h_t + (h u)_x = 0` and `(h u)_t + (h u^2 + 0.5 g h^2)_x = -g h b_x` "
                "using the exact hydrostatic-reconstruction Rusanov / Local-Lax-Friedrichs flux with flat bed EXCEPT a Gaussian submerged sill "
                "`b(x) = sill_height_m * exp(-0.5 * ((x - 400.0) / 45.0)**2)` (`sill_height_m in [0.0, 0.75]`).\n"
                "At `t = 0`, water is at rest with free-surface elevation `eta = h + b == 2.0 m` (`u = 0`), EXCEPT the upstream sluice bay "
                "`x < 80 m` (`cells 0..7`) starts at elevated `eta_0 = 2.65 m` if `sluice_high_mode == True` OR depressed `eta_0 = 1.45 m` "
                "if `sluice_high_mode == False` (reflective wall at `x = 0` too once released!).\n"
                "`obs_bundle` provides noisy free-surface `eta_obs` at ONLY `3` downstream gauges (`cells [48, 62, 74]`) on a **causally delayed** "
                "time window `t in [35.0, 95.0] s` (`dt = 0.5 s` steps)!\n"
                "Inspect `/holdout_task/public_canal.pkl` (which includes the exact forward `step_hydrostatic_rusanov` definition in its metadata docstring "
                "so flux conventions are 100% unambiguous!) and return `{'sluice_high_mode': bool, 'sill_height_m': float, 'eta_full': (T_steps, 80) array}`.\n"
                "Target: Exact `sluice_high_mode`, `|sill_height_hat - sill_height| <= 0.035 m`, and full-domain `eta` RMSE `<= 0.012 m`."
            ),
            public_setup_py=r'''
import pickle, numpy as np

FORWARD_DOC = """Exact forward discretization (`dt = 0.5 s`, `dx = 10.0 m`, `g = 9.81`):
State `U = [h, q]` (`q = h * u`), bed `b[i] = sill_h * exp(-0.5 * ((xc[i] - 400.0) / 45.0)**2)`.
Reflective ghost cells: `h_L_ghost = h[0], b_L_ghost = b[0], q_L_ghost = -q[0]`; same at right `N-1`.
Hydrostatic reconstruction Audusse et al. (2004) at each interface `i - 1/2` between left `(hL, qL, bL)` and right `(hR, qR, bR)`:
  `uL = qL / max(hL, 1e-9)`, `uR = qR / max(hR, 1e-9)`
  `b_star = max(bL, bR)`
  `hL_s = max(0.0, hL + bL - b_star)`, `hR_s = max(0.0, hR + bR - b_star)`
  `FL = [hL_s * uL, hL_s * uL**2 + 0.5 * g * hL_s**2]`, `FR = [hR_s * uR, hR_s * uR**2 + 0.5 * g * hR_s**2]`
  `s_max = max(abs(uL) + sqrt(g * hL_s), abs(uR) + sqrt(g * hR_s), 1e-6)`
  `F_num = 0.5 * (FL + FR) - 0.5 * s_max * ([hR_s - hL_s, hR_s * uR - hL_s * uL])`
  Cell `i` update (Euler step `dt = 0.5`):
  `S_left_b = [0.0, 0.5 * g * (hL_s_at_left_face_right_side**2 - h[i]**2)]` -> specifically for cell `i`:
  Let `b_ip = max(b[i], b[i+1])`, `h_i_minus = max(0.0, h[i] + b[i] - b_ip)` (left side of right face `i+1/2`),
  and `b_im = max(b[i-1], b[i])`, `h_i_plus = max(0.0, h[i] + b[i] - b_im)` (right side of left face `i-1/2`).
  `U_next[i] = U[i] - (dt / dx) * (F_num[i+1/2] - F_num[i-1/2]) + (dt / dx) * [0.0, 0.5 * g * (h_i_minus**2 - h_i_plus**2)]`!
"""

def rollout_canal(sluice_high, sill_h, T_steps=191):
    N, dx, dt, g = 80, 10.0, 0.5, 9.81
    xc = (np.arange(N) + 0.5) * dx
    b = sill_h * np.exp(-0.5 * ((xc - 400.0) / 45.0) ** 2)
    eta0 = np.full(N, 2.0)
    eta0[:8] = 2.65 if sluice_high else 1.45
    h = eta0 - b
    q = np.zeros(N)
    etas = np.zeros((T_steps, N))
    etas[0] = h + b
    for t_k in range(T_steps - 1):
        h_ext = np.r_[h[0], h, h[-1]]
        b_ext = np.r_[b[0], b, b[-1]]
        q_ext = np.r_[-q[0], q, -q[-1]]
        hL, hR = h_ext[:-1], h_ext[1:]
        bL, bR = b_ext[:-1], b_ext[1:]
        qL, qR = q_ext[:-1], q_ext[1:]
        uL = qL / np.maximum(hL, 1e-9); uR = qR / np.maximum(hR, 1e-9)
        bs = np.maximum(bL, bR)
        hLs = np.maximum(0.0, hL + bL - bs); hRs = np.maximum(0.0, hR + bR - bs)
        FL0, FL1 = hLs * uL, hLs * uL ** 2 + 0.5 * g * hLs ** 2
        FR0, FR1 = hRs * uR, hRs * uR ** 2 + 0.5 * g * hRs ** 2
        sm = np.maximum(np.maximum(np.abs(uL) + np.sqrt(g * hLs), np.abs(uR) + np.sqrt(g * hRs)), 1e-6)
        Fn0 = 0.5 * (FL0 + FR0) - 0.5 * sm * (hRs - hLs)
        Fn1 = 0.5 * (FL1 + FR1) - 0.5 * sm * (hRs * uR - hLs * uL)
        src1 = 0.5 * g * (hLs[1:] ** 2 - hRs[:-1] ** 2)
        h = h - (dt / dx) * (Fn0[1:] - Fn0[:-1])
        q = q - (dt / dx) * (Fn1[1:] - Fn1[:-1]) + (dt / dx) * src1
        etas[t_k + 1] = h + b
    return etas

rng = np.random.default_rng(909)
pub = []
for sh_mode, sill_v in ((True, 0.48), (False, 0.26)):
    et = rollout_canal(sh_mode, sill_v)
    gauges = [48, 62, 74]
    t_idx = np.arange(70, 191)
    obs = et[np.ix_(t_idx, gauges)] + rng.normal(0.0, 0.004, size=(len(t_idx), 3))
    pub.append({
        "bundle": {"T_steps": 191, "dt": 0.5, "dx": 10.0, "N": 80, "gauge_cells": gauges, "obs_step_indices": t_idx, "eta_obs": obs, "spec_docstring": FORWARD_DOC},
        "gt": {"sluice_high_mode": sh_mode, "sill_height_m": sill_v, "eta_full": et},
    })
with open("/holdout_task/public_canal.pkl", "wb") as fh:
    pickle.dump(pub, fh)
''',
            private_verifier_py=r'''
import json, pickle, sys, numpy as np
sys.path.insert(0, "/holdout_task")
import canal_inversion

with open("/holdout_task/public_canal.pkl", "rb") as fh:
    spec_doc = pickle.load(fh)[0]["bundle"]["spec_docstring"]

def rollout_canal(sluice_high, sill_h, T_steps=191):
    N, dx, dt, g = 80, 10.0, 0.5, 9.81
    xc = (np.arange(N) + 0.5) * dx; b = sill_h * np.exp(-0.5 * ((xc - 400.0) / 45.0) ** 2)
    eta0 = np.full(N, 2.0); eta0[:8] = 2.65 if sluice_high else 1.45
    h = eta0 - b; q = np.zeros(N); etas = np.zeros((T_steps, N)); etas[0] = h + b
    for t_k in range(T_steps - 1):
        h_ext = np.r_[h[0], h, h[-1]]; b_ext = np.r_[b[0], b, b[-1]]; q_ext = np.r_[-q[0], q, -q[-1]]
        hL, hR = h_ext[:-1], h_ext[1:]; bL, bR = b_ext[:-1], b_ext[1:]; qL, qR = q_ext[:-1], q_ext[1:]
        uL = qL / np.maximum(hL, 1e-9); uR = qR / np.maximum(hR, 1e-9); bs = np.maximum(bL, bR)
        hLs = np.maximum(0.0, hL + bL - bs); hRs = np.maximum(0.0, hR + bR - bs)
        FL0, FL1 = hLs * uL, hLs * uL ** 2 + 0.5 * g * hLs ** 2
        FR0, FR1 = hRs * uR, hRs * uR ** 2 + 0.5 * g * hRs ** 2
        sm = np.maximum(np.maximum(np.abs(uL) + np.sqrt(g * hLs), np.abs(uR) + np.sqrt(g * hRs)), 1e-6)
        Fn0 = 0.5 * (FL0 + FR0) - 0.5 * sm * (hRs - hLs)
        Fn1 = 0.5 * (FL1 + FR1) - 0.5 * sm * (hRs * uR - hLs * uL)
        src1 = 0.5 * g * (hLs[1:] ** 2 - hRs[:-1] ** 2)
        h = h - (dt / dx) * (Fn0[1:] - Fn0[:-1])
        q = q - (dt / dx) * (Fn1[1:] - Fn1[:-1]) + (dt / dx) * src1
        etas[t_k + 1] = h + b
    return etas

rng = np.random.default_rng(9909)
mode_ok = True; sill_errs = []; eta_rmses = []
for sh_mode, sill_v in ((False, 0.58), (True, 0.19), (False, 0.37)):
    et = rollout_canal(sh_mode, sill_v)
    gauges = [48, 62, 74]; t_idx = np.arange(70, 191)
    obs = et[np.ix_(t_idx, gauges)] + rng.normal(0.0, 0.004, size=(len(t_idx), 3))
    out = canal_inversion.invert_canal_regime({"T_steps": 191, "dt": 0.5, "dx": 10.0, "N": 80, "gauge_cells": gauges, "obs_step_indices": t_idx, "eta_obs": obs, "spec_docstring": spec_doc})
    if bool(out["sluice_high_mode"]) != sh_mode: mode_ok = False
    sill_errs.append(abs(float(out["sill_height_m"]) - sill_v))
    eta_rmses.append(float(np.sqrt(np.mean((np.asarray(out["eta_full"], float) - et) ** 2))))
passed = bool(mode_ok and max(sill_errs) <= 0.035 and max(eta_rmses) <= 0.012)
print(json.dumps({"strict_pass": passed, "mode_exact": mode_ok, "max_sill_err": round(max(sill_errs), 4), "max_eta_rmse": round(max(eta_rmses), 5)}))
''',
        ),

        HoldoutOODTask(
            task_id="ho10__haldane_bioreactor_runaway_control",
            short_name="haldane_bioreactor_safety_control",
            wave="wave_2",
            scientific_domain="Process Control / Substrate-Inhibited Exothermic Bioreactor",
            wormhole_family="fam_robust_control_and_safety",
            declared_artifacts=("/holdout_task/bioreactor_ctrl.py",),
            instruction_text=(
                "Write `/holdout_task/bioreactor_ctrl.py` exposing:\n"
                "  `compute_control_step(obs: dict, state_memory: dict) -> tuple[dict, dict]`\n"
                "controlling a substrate-inhibited exothermic fed-batch bioreactor over `t in [0, 20.0] h` (`dt = 0.05 h`, `400` steps).\n"
                "Plant dynamics (exact nominal constants & simulation harness provided in `/holdout_task/plant_sim.py`!):\n"
                "- Specific growth rate has **non-monotonic Haldane inhibition** `mu(S, T) = mu_0 * exp(0.065 * (T - 30.0)) * S / (K_s + S + S^2 / K_i)` "
                "(above `S* = sqrt(K_s * K_i) ~ 3.5 g/L`, excess substrate `S` INHIBITS growth, causing substrate accumulation & sudden exothermic runaway when `S` drops back into the peak zone!).\n"
                "- Control inputs returned each step: `{'feed_rate_L_per_h': float in [0.0, 2.5], 'coolant_flow_frac': float in [0.0, 1.0]}` (with `1`-step actuator delay and noisy `S_meas`, `T_meas`, `X_meas`, `V_L`).\n"
                "- **HARD SAFETY & PRODUCTIVITY REQUIREMENTS** across ALL nominal AND unseen perturbed scenarios (`mu_0 +-15%`, `K_i +-20%`, ambient heat disturbance pulse, coolant fouling `-25%` capacity):\n"
                "  1. Zero safety violations: `T_true(t) <= 34.5 C` AND `V_L(t) <= 12.0 L` at EVERY step;\n"
                "  2. Final total biomass `X(T_end) * V(T_end) >= 62.0 g` on EVERY scenario!"
            ),
            public_setup_py=r'''
SIM_CODE = """import numpy as np

NOMINAL_SCENARIOS = [
    {"name": "nominal", "mu0": 0.42, "Ks": 0.8, "Ki": 15.0, "ua_scale": 1.0, "heat_pulse": 0.0},
    {"name": "high_mu_low_Ki", "mu0": 0.48, "Ks": 0.8, "Ki": 12.0, "ua_scale": 0.82, "heat_pulse": 2.2},
    {"name": "low_mu_fouled", "mu0": 0.36, "Ks": 0.9, "Ki": 16.0, "ua_scale": 0.75, "heat_pulse": 1.5},
]

def run_closed_loop(compute_control_step, scen, seed=10):
    rng = np.random.default_rng(seed)
    dt = 0.05
    X, S, V, T = 1.5, 2.2, 4.0, 30.0
    Sf = 45.0; Yxs = 0.50; k_heat = 2.8; T_cool = 18.0; T_amb = 29.0
    mem = {}
    prev_u = {"feed_rate_L_per_h": 0.2, "coolant_flow_frac": 0.3}
    max_T, max_V = T, V
    for step in range(400):
        t = step * dt
        obs = {
            "t_h": t, "dt_h": dt, "V_L": V,
            "X_meas_g_L": max(0.0, X + rng.normal(0.0, 0.04)),
            "S_meas_g_L": max(0.0, S + rng.normal(0.0, 0.08)),
            "T_meas_C": T + rng.normal(0.0, 0.05),
        }
        u_cmd, mem = compute_control_step(obs, mem)
        F = float(np.clip(prev_u["feed_rate_L_per_h"], 0.0, 2.5))
        uc = float(np.clip(prev_u["coolant_flow_frac"], 0.0, 1.0))
        prev_u = u_cmd
        # Sub-step RK2
        for _ in range(5):
            hs = dt / 5.0
            mu = scen["mu0"] * np.exp(0.065 * (T - 30.0)) * max(S, 0.0) / (scen["Ks"] + max(S, 0.0) + (max(S, 0.0) ** 2) / scen["Ki"])
            dV = F
            dX = mu * X - (F / V) * X
            dS = -(mu / Yxs) * X + (F / V) * (Sf - S)
            q_rxn = k_heat * mu * X
            q_cool = 3.6 * scen["ua_scale"] * uc * (T - T_cool) / (V / 4.0)
            q_dist = scen["heat_pulse"] if (9.0 <= t <= 12.5) else 0.0
            dT = q_rxn - q_cool + 0.15 * (T_amb - T) + (F / V) * (28.0 - T) + q_dist
            V += hs * dV; X = max(1e-4, X + hs * dX); S = max(0.0, S + hs * dS); T += hs * dT
            max_T = max(max_T, T); max_V = max(max_V, V)
    return {"final_biomass_g": float(X * V), "max_T_C": float(max_T), "max_V_L": float(max_V), "safe": bool(max_T <= 34.5 and max_V <= 12.0)}
"""
with open("/holdout_task/plant_sim.py", "w", encoding="utf-8") as fh:
    fh.write(SIM_CODE)
''',
            private_verifier_py=r'''
import json, sys, numpy as np
sys.path.insert(0, "/holdout_task")
import plant_sim, bioreactor_ctrl

UNSEEN_SCENARIOS = [
    *plant_sim.NOMINAL_SCENARIOS,
    {"name": "corner_hot_runaway", "mu0": 0.483, "Ks": 0.75, "Ki": 12.2, "ua_scale": 0.76, "heat_pulse": 2.4},
    {"name": "corner_slow_inhibited", "mu0": 0.357, "Ks": 0.92, "Ki": 12.5, "ua_scale": 0.80, "heat_pulse": 0.8},
    {"name": "corner_fast_nominal", "mu0": 0.46, "Ks": 0.78, "Ki": 18.0, "ua_scale": 0.75, "heat_pulse": 2.0},
]
results = [plant_sim.run_closed_loop(bioreactor_ctrl.compute_control_step, sc, seed=100 + i) for i, sc in enumerate(UNSEEN_SCENARIOS)]
all_safe = all(r["safe"] for r in results)
min_bio = min(r["final_biomass_g"] for r in results)
max_T = max(r["max_T_C"] for r in results)
passed = bool(all_safe and min_bio >= 62.0)
print(json.dumps({"strict_pass": passed, "all_safe": all_safe, "min_biomass_g": round(min_bio, 3), "worst_max_T_C": round(max_T, 3)}))
''',
        ),

        HoldoutOODTask(
            task_id="ho11__certified_capacitated_facility_bnb",
            short_name="certified_facility_bnb",
            wave="wave_2",
            scientific_domain="Operations Research / Exact Mixed-Integer Convex Certificate",
            wormhole_family="fam_sparse_assimilation_and_latent_recovery",
            declared_artifacts=("/holdout_task/facility_cert.py",),
            instruction_text=(
                "Write `/holdout_task/facility_cert.py` exposing:\n"
                "  `solve_and_certify_facilities(instance: dict) -> dict`\n"
                "Given `m = 12` candidate facilities (`fixed_cost[i] > 0`, `capacity[i] > 0`) and `n = 24` customers (`demand[j] > 0`, "
                "unit transport cost `cost_matrix[i, j] >= 0`), minimize over binary `y in {0, 1}^m` and flows `X >= 0` (`m x n`):\n"
                "  `J(y, X) = sum_i fixed_cost[i] * y[i] + sum_{i,j} cost_matrix[i, j] * X[i, j]`\n"
                "subject to `sum_i X[i, j] == demand[j]` for all `j`, and `sum_j X[i, j] <= capacity[i] * y[i]` for all `i`.\n"
                "Return:\n"
                "- `'open_y'`: list of `m` ints in `{0, 1}`;\n"
                "- `'flows_X'`: `(m, n)` float array;\n"
                "- `'upper_bound'`: float exact optimal objective `J*`;\n"
                "- `'proof_leaves'`: list of disjoint partition leaves `{'fixed_closed': list[int], 'fixed_open': list[int], 'lower_bound': float}` "
                "covering `{0, 1}^m` EXACTLY (`sum 2^(m - |C| - |O|) == 2^m`, pairwise disjoint), where EVERY leaf has a mathematically sound "
                "LP-relaxation / capacity-knapsack lower bound `lower_bound >= upper_bound - 1e-5` (or exact infeasibility `+inf` when "
                "`sum_{i not in fixed_closed} capacity[i] < sum_j demand[j]`), compacted bottom-up via sibling merging so `len(proof_leaves) <= 80`!\n"
                "Test on `/holdout_task/public_facility.pkl`."
            ),
            public_setup_py=r'''
import pickle, numpy as np
rng = np.random.default_rng(1101)
m, n = 12, 24
fac_xy = rng.uniform(0.0, 10.0, size=(m, 2))
cus_xy = rng.uniform(0.0, 10.0, size=(n, 2))
cost = np.linalg.norm(fac_xy[:, None, :] - cus_xy[None, :, :], axis=2)
demand = rng.uniform(1.0, 3.0, size=n)
cap = rng.uniform(8.0, 13.0, size=m)
fcost = rng.uniform(14.0, 26.0, size=m)
with open("/holdout_task/public_facility.pkl", "wb") as fh:
    pickle.dump({"m": m, "n": n, "fixed_cost": fcost, "capacity": cap, "demand": demand, "cost_matrix": cost}, fh)
''',
            private_verifier_py=r'''
import json, sys, numpy as np
from scipy.optimize import linprog
sys.path.insert(0, "/holdout_task")
import facility_cert

rng = np.random.default_rng(91101)
m, n = 12, 22
fac_xy = rng.uniform(0.0, 10.0, size=(m, 2)); cus_xy = rng.uniform(0.0, 10.0, size=(n, 2))
cost = np.linalg.norm(fac_xy[:, None, :] - cus_xy[None, :, :], axis=2)
demand = rng.uniform(1.1, 2.8, size=n); cap = rng.uniform(8.5, 13.5, size=m); fcost = rng.uniform(15.0, 27.0, size=m)
inst = {"m": m, "n": n, "fixed_cost": fcost, "capacity": cap, "demand": demand, "cost_matrix": cost}

res = facility_cert.solve_and_certify_facilities(inst)
y_hat = np.asarray(res["open_y"], int)
X_hat = np.asarray(res["flows_X"], float)
ub_rep = float(res["upper_bound"])
feas = (
    y_hat.shape == (m,) and set(y_hat.tolist()).issubset({0, 1})
    and np.all(X_hat >= -1e-7)
    and np.max(np.abs(X_hat.sum(axis=0) - demand)) <= 1e-5
    and np.all(X_hat.sum(axis=1) <= cap * y_hat + 1e-5)
)
ub_true = float(fcost @ y_hat + np.sum(cost * X_hat))

# Brute-force exact global MIP optimum over all 4096 binary y in {0, 1}^12!
tot_dem = float(np.sum(demand))
A_eq = np.zeros((n, m * n))
for j in range(n):
    A_eq[j, j::n] = 1.0
A_ub = np.zeros((m, m * n))
for i in range(m):
    A_ub[i, i * n:(i + 1) * n] = 1.0
c_vec = cost.ravel()
min_by_mask = np.full(1 << m, np.inf)
for mask in range(1 << m):
    ym = np.array([(mask >> i) & 1 for i in range(m)], dtype=float)
    if float(cap @ ym) < tot_dem - 1e-9:
        continue
    lp = linprog(c_vec, A_ub=A_ub, b_ub=cap * ym, A_eq=A_eq, b_eq=demand, bounds=(0, None), method="highs")
    if lp.success:
        min_by_mask[mask] = float(fcost @ ym + lp.fun)
best_global = float(np.min(min_by_mask))

leaves = res["proof_leaves"]
cov = np.zeros(1 << m, dtype=int)
lb_valid = bool(np.isfinite(ub_rep) and np.isfinite(ub_true) and len(leaves) > 0)
for lf in (leaves if lb_valid else []):
    fc = set(int(i) for i in lf["fixed_closed"]); fo = set(int(i) for i in lf["fixed_open"])
    if (fc & fo) or any(i < 0 or i >= m for i in (fc | fo)):
        lb_valid = False; break
    sub_min = np.inf
    for mask in range(1 << m):
        if any((mask >> i) & 1 for i in fc) or any(not ((mask >> i) & 1) for i in fo):
            continue
        cov[mask] += 1
        if min_by_mask[mask] < sub_min:
            sub_min = min_by_mask[mask]
    rep_lb = float(lf["lower_bound"])
    if (not np.isfinite(rep_lb)) or rep_lb > sub_min + 1e-4 or rep_lb < ub_rep - 1e-4:
        lb_valid = False; break

passed = bool(feas and np.isfinite(ub_rep) and abs(ub_true - best_global) <= 1e-4 and abs(ub_rep - ub_true) <= 1e-4 and np.all(cov == 1) and lb_valid and len(leaves) <= 80)
print(json.dumps({"strict_pass": passed, "feas": bool(feas), "opt_exact": bool(abs(ub_true - best_global) <= 1e-4), "partition_exact": bool(np.all(cov == 1)), "lb_valid": bool(lb_valid), "leaf_count": int(len(leaves))}))
''',
        ),

        HoldoutOODTask(
            task_id="ho12__multicopy_plasmid_cohort_consensus",
            short_name="multicopy_plasmid_cohort",
            wave="wave_2",
            scientific_domain="Computational Genomics / Multi-Isolate Cohort & Copy-Number Audit",
            wormhole_family="fam_biomystery_genomic_homology",
            declared_artifacts=("/holdout_task/plasmid_audit.py",),
            instruction_text=(
                "Write `/holdout_task/plasmid_audit.py` exposing:\n"
                "  `audit_outbreak_cohort(cohort_dir: str) -> dict`\n"
                "`cohort_dir` contains `N` isolate JSON bundles (`isolate_01.json` .. `isolate_N.json`, `N >= 12`).\n"
                "- **CRITICAL COHORT TRAP**: Isolates `01` and `02` carry a high-copy contaminant bacteriophage contig (`copy ~ 18x`) "
                "that is ABSENT from the rest of the outbreak cohort (`03..N`), whereas ALL `>= 90%` of outbreak isolates share "
                "one conserved **multicopy AMR plasmid** (`circular == True`, `copy_number = contig_depth / median_chromosomal_depth` "
                "in `[3.2, 7.5]`, `length >= 2400 bp`) carrying a conserved `36 bp` resistance core motif with **1 exact non-synonymous "
                "codon substitution** relative to `reference_wildtype_cds` in `cohort_dir/reference_panel.json`!\n"
                "Programmatically inspect ALL `1..N` isolates (`never infer from isolates 01-02 alone!`) and return:\n"
                "  `{'shared_plasmid_family_tag': str, 'median_plasmid_copy_number': float, 'amino_acid_substitution': str}`\n"
                "(e.g. `'S83L'`, 1-indexed codon position in `reference_wildtype_cds` using standard genetic code).\n"
                "Verify on `/holdout_task/public_cohort`."
            ),
            public_setup_py=r'''
import json, os, numpy as np

CODON = {
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S","AGT":"S","AGC":"S","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "TAT":"Y","TAC":"Y","CAT":"H","CAC":"H","CAA":"Q","CAG":"Q","AAT":"N","AAC":"N","AAA":"K","AAG":"K",
    "GAT":"D","GAC":"D","GAA":"E","GAG":"E","TGT":"C","TGC":"C","TGG":"W","CGT":"R","CGC":"R","CGA":"R","CGG":"R","AGA":"R","AGG":"R",
    "GGT":"G","GGC":"G","GGA":"G","GGG":"G","TAA":"*","TAG":"*","TGA":"*",
}

def build_cohort(out_dir, seed, mut_codon_idx_0, wt_codon, mut_codon, true_tag, true_cn):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    codons = ["ATG", "GCT", "AAAG"[:3], "GAA", "TTC", "GGT", "TCT", "ATC", "GAC", "TCT", "GCA", "TAC", "CTG", "CCG", "CGT", "GTG", "AAC", "TAA"]
    codons[2] = "AAG"
    codons[mut_codon_idx_0] = wt_codon
    wt_cds = "".join(codons)
    mut_codons = list(codons); mut_codons[mut_codon_idx_0] = mut_codon
    mut_cds = "".join(mut_codons)
    with open(os.path.join(out_dir, "reference_panel.json"), "w") as fh:
        json.dump({"reference_wildtype_cds": wt_cds}, fh)

    bases = np.array(list("ACGT"))
    for k in range(1, 15):
        chrom_d = float(rng.uniform(38.0, 62.0))
        contigs = []
        # 5 chromosomal contigs (~1x depth, circular=False)
        for c_i in range(5):
            seq = "".join(rng.choice(bases, size=4200))
            contigs.append({"contig_id": f"ctg_chr_{c_i}", "circular": False, "mean_depth": round(chrom_d * float(rng.uniform(0.94, 1.06)), 2), "family_tag": f"CHR_SEG_{c_i}", "sequence": seq})
        # Isolates 1 & 2 have passenger phage!
        if k in (1, 2):
            p_seq = "".join(rng.choice(bases, size=3100))
            contigs.append({"contig_id": "ctg_phage_x", "circular": True, "mean_depth": round(chrom_d * 17.5, 2), "family_tag": "PHAGE_CONTAM_PHI9", "sequence": p_seq})
        # Shared outbreak AMR plasmid in 13 of 14 isolates (reverse-complemented in half!)
        if k != 9:
            flank_l = "".join(rng.choice(bases, size=1300))
            flank_r = "".join(rng.choice(bases, size=1300))
            full_p = flank_l + mut_cds + flank_r
            if k % 2 == 0:
                comp = str.maketrans("ACGT", "TGCA")
                full_p = full_p.translate(comp)[::-1]
            cn = true_cn * float(rng.uniform(0.95, 1.05))
            contigs.append({"contig_id": f"ctg_u_{k}", "circular": True, "mean_depth": round(chrom_d * cn, 2), "family_tag": true_tag, "sequence": full_p})
        with open(os.path.join(out_dir, f"isolate_{k:02d}.json"), "w") as fh:
            json.dump({"isolate_id": f"ISO_{k:02d}", "contigs": contigs}, fh)

build_cohort("/holdout_task/public_cohort", 1201, 6, "TCT", "TTG", "IncX4_CTX_PLASMID", 4.6)
''',
            private_verifier_py=r'''
import json, os, sys, tempfile, numpy as np
sys.path.insert(0, "/holdout_task")
import plasmid_audit

rng = np.random.default_rng(91201)
with tempfile.TemporaryDirectory() as td:
    codons = ["ATG", "GCT", "AAG", "GAA", "TTC", "GGT", "TCT", "ATC", "GAC", "GAT", "GCA", "TAC", "CTG", "CCG", "CGT", "GTG", "AAC", "TAA"]
    wt_cds = "".join(codons)
    mut_codons = list(codons); mut_codons[9] = "GGT"  # Pos 10 (1-indexed): GAT (D) -> GGT (G) => 'D10G'!
    mut_cds = "".join(mut_codons)
    with open(os.path.join(td, "reference_panel.json"), "w") as fh:
        json.dump({"reference_wildtype_cds": wt_cds}, fh)
    bases = np.array(list("ACGT"))
    cns = []
    for k in range(1, 16):
        chrom_d = float(rng.uniform(40.0, 65.0))
        contigs = [{"contig_id": f"c_{i}", "circular": False, "mean_depth": round(chrom_d * float(rng.uniform(0.95, 1.05)), 2), "family_tag": f"CHR_{i}", "sequence": "".join(rng.choice(bases, size=4000))} for i in range(5)]
        if k in (1, 2):
            contigs.append({"contig_id": "c_ph", "circular": True, "mean_depth": round(chrom_d * 19.0, 2), "family_tag": "PHAGE_DECOY_77", "sequence": "".join(rng.choice(bases, size=3200))})
        if k != 11:
            full_p = "".join(rng.choice(bases, size=1250)) + mut_cds + "".join(rng.choice(bases, size=1250))
            if k % 2 == 1:
                full_p = full_p.translate(str.maketrans("ACGT", "TGCA"))[::-1]
            cn = 5.4 * float(rng.uniform(0.96, 1.04))
            cns.append(cn)
            contigs.append({"contig_id": f"c_pl_{k}", "circular": True, "mean_depth": round(chrom_d * cn, 2), "family_tag": "IncFII_NDM_PLASMID", "sequence": full_p})
        with open(os.path.join(td, f"isolate_{k:02d}.json"), "w") as fh:
            json.dump({"isolate_id": f"ISO_{k:02d}", "contigs": contigs}, fh)

    res = plasmid_audit.audit_outbreak_cohort(td)
    tag_ok = str(res["shared_plasmid_family_tag"]) == "IncFII_NDM_PLASMID"
    sub_ok = str(res["amino_acid_substitution"]) == "D10G"
    cn_ok = abs(float(res["median_plasmid_copy_number"]) - float(np.median(cns))) <= 0.25
    passed = bool(tag_ok and sub_ok and cn_ok)
    print(json.dumps({"strict_pass": passed, "tag_ok": tag_ok, "sub_ok": sub_ok, "cn_ok": cn_ok, "reported": res}))
''',
        ),
    ]


def compute_holdout_suite_preregistration_sha256() -> str:
    """Compute canonical SHA-256 over all 12 pre-registered Hold-out OOD task definitions + verifiers."""
    suite = build_holdout_ood_12_tasks()
    serial = [
        {
            "task_id": t.task_id,
            "wave": t.wave,
            "declared_artifacts": list(t.declared_artifacts),
            "instruction_text": t.instruction_text,
            "public_setup_py": t.public_setup_py,
            "private_verifier_py": t.private_verifier_py,
        }
        for t in suite
    ]
    return hashlib.sha256(json.dumps(serial, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def parse_authenticated_grader_stdout(stdout: str, sentinel_prefix: str) -> dict[str, Any]:
    """Extract ONLY the parent-authenticated sentinel JSON line, ignoring any fake candidate stdout (`M7-Audit` Fix)."""
    for ln in reversed(stdout.splitlines()):
        s = ln.strip()
        if s.startswith(sentinel_prefix):
            payload = s[len(sentinel_prefix):].strip()
            try:
                obj = json.loads(payload)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    return {"strict_pass": False, "auth_sentinel_missing": True}


def run_holdout_private_verifier(task: HoldoutOODTask, submitted_tar_path: Path) -> dict[str, Any]:
    """Grade `submitted_tar_path` inside an air-gapped (`network_mode='none'`) `hegel_clean_scipy_sandbox:v1` container.

    Hardened against M7-Audit boundary controls:
      1. `network_mode='none'` blocks all outbound network exfiltration/lookup.
      2. `/tmp/_verify.py` is read into memory and unlinked BEFORE executing user code so candidates cannot read `/tmp/_verify.py`.
      3. Candidate stdout is redirected during execution, and ONLY the cryptographic `sentinel_prefix` line emitted
         by the outer wrapper is trusted (`parse_authenticated_grader_stdout`).
      Boundary note: Candidate module import still executes inside the same Python process within the fresh container
      (so full hostile-adversary stack-frame isolation would additionally require a child worker subprocess + IPC pipe).
    """
    import secrets

    ensure_clean_scipy_sandbox_image()
    t0 = monotonic()
    vcid = m6f.start_detached_container(
        CLEAN_SCIPY_SANDBOX_IMAGE,
        memory_limit="3072m",
        cpus="2",
        network_mode="none",
    )
    sentinel_prefix = f"__HEGEL_VERIFIER_AUTH_{secrets.token_hex(16)}__:"
    try:
        m6f.exec_container_bash(vcid, "mkdir -p /holdout_task /tmp", timeout_sec=10)
        wrapper_py = (
            "import contextlib, io, json, os, sys\n"
            "v_src = open('/tmp/_verify.py', 'r', encoding='utf-8').read()\n"
            "os.unlink('/tmp/_verify.py')\n"
            f"sentinel = {sentinel_prefix!r}\n"
            "_real_stdout = sys.stdout\n"
            "_cap = io.StringIO()\n"
            "with contextlib.redirect_stdout(_cap):\n"
            "    exec(compile(v_src, '<isolated_holdout_verifier>', 'exec'), {'__name__': '__main__'})\n"
            "lines = [ln.strip() for ln in _cap.getvalue().splitlines() if ln.strip().startswith('{')]\n"
            "ver_json = lines[-1] if lines else json.dumps({'strict_pass': False})\n"
            "_real_stdout.write(sentinel + ver_json + '\\n')\n"
            "_real_stdout.flush()\n"
        )
        with tempfile.TemporaryDirectory(prefix="hegel_ho_ver_") as td:
            td_p = Path(td)
            (td_p / "_setup.py").write_text(task.public_setup_py, encoding="utf-8")
            (td_p / "_verify.py").write_text(task.private_verifier_py, encoding="utf-8")
            (td_p / "_run_ver.py").write_text(wrapper_py, encoding="utf-8")
            subprocess.run(["docker", "cp", str(td_p / "_setup.py"), f"{vcid}:/tmp/_setup.py"], check=True, timeout=20)
            subprocess.run(["docker", "cp", str(td_p / "_verify.py"), f"{vcid}:/tmp/_verify.py"], check=True, timeout=20)
            subprocess.run(["docker", "cp", str(td_p / "_run_ver.py"), f"{vcid}:/tmp/_run_ver.py"], check=True, timeout=20)
        m6f.exec_container_bash(vcid, "python3 /tmp/_setup.py && rm -f /tmp/_setup.py", timeout_sec=60)
        if submitted_tar_path.is_file():
            m6f.import_declared_artifacts_tar(vcid, task.declared_artifacts, submitted_tar_path)
        art_chk = m6f.check_declared_artifacts_in_container(vcid, task.declared_artifacts)
        if not art_chk["all_present"]:
            return {
                "benchmark": "Hegel-OOD-Holdout-12",
                "strict_pass": False,
                "reward": 0.0,
                "artifact_audit": art_chk,
                "verifier_stdout": "",
                "verifier_stderr": f"Missing declared artifacts: {art_chk['missing_or_empty']}",
                "verifier_wall_ms": round((monotonic() - t0) * 1000.0, 2),
                "failure_taxonomy": "EMPTY_OR_MISSING_DECLARED_ARTIFACT",
            }
        ex = m6f.exec_container_bash(
            vcid,
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I -c "
            "\"import os; s=open('/tmp/_run_ver.py').read(); os.unlink('/tmp/_run_ver.py'); exec(compile(s, '<runner>', 'exec'))\"",
            timeout_sec=120,
        )
        parsed_v = parse_authenticated_grader_stdout(ex["stdout"], sentinel_prefix)
        sp = bool(parsed_v.get("strict_pass")) and int(ex["exit_code"]) == 0
        return {
            "benchmark": "Hegel-OOD-Holdout-12",
            "strict_pass": sp,
            "reward": 1.0 if sp else 0.0,
            "artifact_audit": art_chk,
            "verifier_metrics": parsed_v,
            "verifier_exit_code": ex["exit_code"],
            "verifier_stdout": ex["stdout"][-2000:],
            "verifier_stderr": ex["stderr"][-1500:],
            "verifier_wall_ms": round((monotonic() - t0) * 1000.0, 2),
            "failure_taxonomy": "PASSED_ALL_CHECKS" if sp else "NUMERICAL_ACCURACY_OR_MECHANISTIC_MODEL_DEFECT",
        }
    finally:
        m6f.stop_container_safely(vcid)


def holdout_tasks_by_id() -> dict[str, HoldoutOODTask]:
    return {t.task_id: t for t in build_holdout_ood_12_tasks()}


def holdout_tasks_as_frontier_waves() -> dict[str, list[m6f.FrontierBenchmarkTask]]:
    """Convert `Hegel-OOD-Holdout-12` into `FrontierBenchmarkTask` waves (`private_verifier_py` physically excluded!)."""
    ensure_clean_scipy_sandbox_image()
    waves: dict[str, list[m6f.FrontierBenchmarkTask]] = {"wave_1": [], "wave_2": []}
    for t in build_holdout_ood_12_tasks():
        waves[t.wave].append(
            m6f.FrontierBenchmarkTask(
                task_id=t.task_id,
                short_name=t.short_name,
                benchmark="Hegel-OOD-Holdout-12",
                wave=t.wave,
                scientific_domain=t.scientific_domain,
                wormhole_family=t.wormhole_family,
                instruction_text=t.instruction_text,
                agent_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
                verifier_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
                declared_artifacts=t.declared_artifacts,
                verifier_timeout_sec=120,
                agent_cpus=2,
                agent_memory_mb=3072,
                verifier_cpus=2,
                verifier_memory_mb=3072,
                holdout_public_setup_py=t.public_setup_py,
            )
        )
    return waves


def run_holdout_private_verifier_by_id(task_id: str, submitted_tar_path: Path) -> dict[str, Any]:
    ho_task = holdout_tasks_by_id()[task_id]
    return run_holdout_private_verifier(ho_task, submitted_tar_path)


def run_holdout_private_verifier_for_container(task_id: str, cid_with_artifacts: str) -> dict[str, Any]:
    """Export declared artifacts from `cid_with_artifacts` and grade inside a separate air-gapped container."""
    ho_task = holdout_tasks_by_id()[task_id]
    with tempfile.TemporaryDirectory(prefix="hegel_ho_cnt_ver_") as td:
        tar_fp = Path(td) / "submitted_artifacts.tar"
        m6f.export_declared_artifacts_tar(cid_with_artifacts, ho_task.declared_artifacts, tar_fp)
        return run_holdout_private_verifier(ho_task, tar_fp)

