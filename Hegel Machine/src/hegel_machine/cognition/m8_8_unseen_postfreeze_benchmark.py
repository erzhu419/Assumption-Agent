"""Hegel M8.8 Post-Freeze Unseen Out-of-Distribution Scientific Benchmark (`Hegel-Unseen-PostFreeze-4`).

Created strictly AFTER fixing the `M8.8` L1 compiler, capability retrieval, atomic dependency-closure packager,
and post-package verification contracts (`0` exposure during `M8.6`/`M8.7` trajectory analysis or prompt design):
  1. `uf01__electrical_resistivity_electrode_contact_tomography`: Borehole/cross-line 2D dipole potential kernel
     inversion with unknown dual TX/RX zero-sum electrode contact offsets + 5% faulty-channel spikes.
  2. `uf02__batch_drifted_substrate_inhibition_kinetics`: Multi-plate 96-well Haldane substrate-inhibition kinetics
     with unknown 2D quadratic optical illumination gain per plate + pipetting bubble outliers.
  3. `uf03__orthotropic_robin_heat_sink_inversion`: Steady 2D variable-conductivity finite-volume diffusion with
     Robin boundary dissipation, sparse interior thermal source, reference-junction offset, and stuck sensors.
  4. `uf04__certified_box_l0_sensor_selection`: Exact box-constrained L0-regularized ridge regression requiring
     both global optimal weights and a provably valid dual/branch-and-bound lower bound.
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


def build_unseen_postfreeze_4_tasks() -> list[HoldoutOODTask]:
    """Return the 4 post-freeze unseen scientific generalization tasks (`uf01..uf04`)."""
    return [
        HoldoutOODTask(
            task_id="uf01__electrical_resistivity_electrode_contact_tomography",
            short_name="ert_electrode_contact_tomography",
            wave="wave_1",
            scientific_domain="Geophysics / Borehole Electrical Potential Tomography",
            wormhole_family="fam_unseen_geoelectrical_tomography",
            declared_artifacts=("/holdout_task/ert_solver.py",),
            instruction_text=(
                "Write `/holdout_task/ert_solver.py` exposing:\n"
                "  `invert_ert_anomaly(survey: dict) -> dict`\n"
                "A borehole-to-borehole electrical potential survey injects unit current at `N_tx` electrodes `(x=0, z_tx)` "
                "across `x in [0, L_m]` to `N_rx` potential electrodes at `(x=L_m, z_rx)` (`z in [0, D_m]`).\n"
                "- Observed voltage matrix `V_obs` (`N_tx x N_rx`, in Volts) obeys:\n"
                "  `V_obs[i, j] = c0 / R_ij + c1 * (z_rx[j] - z_tx[i])**2 / R_ij**3 + tx_offset_v[i] + rx_offset_v[j] + amp * K(x_a, z_a; i, j) + noise`\n"
                "  where `R_ij = hypot(L_m, z_rx[j] - z_tx[i])`, `sum(tx_offset_v) == 0.0`, `sum(rx_offset_v) == 0.0`, "
                "`amp > 0`, and the regularized 2D conductive-body kernel (`a0 = survey['a0_m']`, default `6.0`) is:\n"
                "  `K(x_a, z_a; i, j) = 1.0 / (sqrt(x_a**2 + (z_a - z_tx[i])**2 + a0**2) * sqrt((L_m - x_a)**2 + (z_rx[j] - z_a)**2 + a0**2))`.\n"
                "- Roughly `5%` of channels suffer electrode-polarization spike outliers (`+-0.06` V)!\n"
                "Inspect `/holdout_task/public_ert_surveys.pkl` and return:\n"
                "  `{'anomaly_xz_m': [float(x_a), float(z_a)], 'rx_contact_offset_v': list[float]}`.\n"
                "Target: Anomaly position error `<= 2.0 m` and `rx_contact_offset_v` RMSE `<= 0.0012 V` on held-out surveys (`< 15s`)."
            ),
            public_setup_py=r'''
import pickle, numpy as np

def make_ert(seed, xa, za, c0, c1, amp):
    rng = np.random.default_rng(seed)
    L, D, a0 = 36.0, 72.0, 6.0
    z_tx = np.linspace(6.0, 66.0, 14)
    z_rx = np.linspace(5.0, 67.0, 15)
    tx_off = rng.uniform(-0.012, 0.012, size=len(z_tx)); tx_off -= np.mean(tx_off)
    rx_off = rng.uniform(-0.012, 0.012, size=len(z_rx)); rx_off -= np.mean(rx_off)
    dz = z_rx[None, :] - z_tx[:, None]
    R = np.hypot(L, dz)
    d1 = np.sqrt(xa**2 + (za - z_tx[:, None])**2 + a0**2)
    d2 = np.sqrt((L - xa)**2 + (z_rx[None, :] - za)**2 + a0**2)
    K = 1.0 / (d1 * d2)
    V = c0 / R + c1 * (dz**2) / (R**3) + tx_off[:, None] + rx_off[None, :] + amp * K + rng.normal(0.0, 2.5e-4, size=R.shape)
    for fi in rng.choice(V.size, size=10, replace=False):
        V.flat[int(fi)] += float(rng.choice([-0.06, 0.06]))
    return {"L_m": L, "D_m": D, "a0_m": a0, "z_tx_m": z_tx, "z_rx_m": z_rx, "V_obs": V}, {"anomaly_xz_m": [xa, za], "rx_contact_offset_v": rx_off}

pub = [make_ert(31, 13.5, 25.0, 1.8, -0.35, 6.4), make_ert(32, 23.0, 48.5, 2.1, 0.28, 7.2)]
with open("/holdout_task/public_ert_surveys.pkl", "wb") as fh:
    pickle.dump([{"survey": s, "gt": g} for s, g in pub], fh)
''',
            private_verifier_py=r'''
import json, sys, numpy as np
sys.path.insert(0, "/holdout_task")
import ert_solver

def make_ert(seed, xa, za, c0, c1, amp):
    rng = np.random.default_rng(seed)
    L, D, a0 = 40.0, 78.0, 6.0
    z_tx = np.linspace(6.0, 72.0, 15)
    z_rx = np.linspace(5.0, 73.0, 16)
    tx_off = rng.uniform(-0.014, 0.014, size=len(z_tx)); tx_off -= np.mean(tx_off)
    rx_off = rng.uniform(-0.014, 0.014, size=len(z_rx)); rx_off -= np.mean(rx_off)
    dz = z_rx[None, :] - z_tx[:, None]
    R = np.hypot(L, dz)
    d1 = np.sqrt(xa**2 + (za - z_tx[:, None])**2 + a0**2)
    d2 = np.sqrt((L - xa)**2 + (z_rx[None, :] - za)**2 + a0**2)
    V = c0 / R + c1 * (dz**2) / (R**3) + tx_off[:, None] + rx_off[None, :] + amp * (1.0 / (d1 * d2)) + rng.normal(0.0, 2.8e-4, size=R.shape)
    for fi in rng.choice(V.size, size=12, replace=False):
        V.flat[int(fi)] += float(rng.choice([-0.065, 0.065]))
    return {"L_m": L, "D_m": D, "a0_m": a0, "z_tx_m": z_tx, "z_rx_m": z_rx, "V_obs": V}, np.array([xa, za]), rx_off

pos_errs, rx_errs = [], []
all_valid = True
for sd, xa, za, c0, c1, amp in ((901, 14.5, 54.0, 1.95, -0.42, 6.8), (902, 27.0, 24.5, 1.65, 0.31, 7.6), (903, 19.5, 39.0, 2.25, -0.18, 6.1)):
    sv, gt_xz, gt_rx = make_ert(sd, xa, za, c0, c1, amp)
    out = ert_solver.invert_ert_anomaly(sv)
    if not isinstance(out, dict) or "anomaly_xz_m" not in out or "rx_contact_offset_v" not in out:
        all_valid = False; break
    xz_arr = np.asarray(out["anomaly_xz_m"], dtype=float)
    rx_raw = np.asarray(out["rx_contact_offset_v"], dtype=float)
    if xz_arr.shape != (2,) or rx_raw.shape != gt_rx.shape or not (np.all(np.isfinite(xz_arr)) and np.all(np.isfinite(rx_raw))):
        all_valid = False; break
    pos_errs.append(float(np.linalg.norm(xz_arr - gt_xz)))
    rx_hat = rx_raw - np.mean(rx_raw)
    rx_errs.append(float(np.sqrt(np.mean((rx_hat - gt_rx) ** 2))))
passed = bool(all_valid and len(pos_errs) == 3 and max(pos_errs) <= 2.0 and max(rx_errs) <= 0.0012)
print(json.dumps({"strict_pass": passed, "max_pos_err_m": (round(max(pos_errs), 4) if pos_errs and all_valid else None), "max_rx_rmse_v": (round(max(rx_errs), 6) if rx_errs and all_valid else None)}, allow_nan=False))
''',
        ),
        HoldoutOODTask(
            task_id="uf02__batch_drifted_substrate_inhibition_kinetics",
            short_name="substrate_inhibition_plate_kinetics",
            wave="wave_1",
            scientific_domain="Systems Enzymology / Spatial Plate Deconfounding",
            wormhole_family="fam_unseen_enzymology_deconfounding",
            declared_artifacts=("/holdout_task/enzyme_plate_solver.py",),
            instruction_text=(
                "Write `/holdout_task/enzyme_plate_solver.py` exposing:\n"
                "  `estimate_variant_haldane_parameters(plates: list[dict]) -> dict[str, dict[str, float]]`\n"
                "High-throughput 96-well (`8 x 12`, `row in 0..7`, `col in 0..11`) fluorometric enzyme initial-rate assays "
                "measure enzyme variants across multiple plates.\n"
                "- Each plate suffers an unknown additive background `B_p` (measured on `BLANK` wells where `substrate_mM == 0`) "
                "AND a smooth 2D optical illumination gain `G_p(row, col)` (quadratic in `(row, col)`), plus `~4%` pipetting-bubble gross outlier wells!\n"
                "- `REF_WT` control wells on every plate all use wild-type reference rate `v_ref = 1.0` so `rate_raw = B_p + G_p(row, col) * 1.0 + noise`.\n"
                "- `VARIANT` wells follow Haldane substrate-inhibition kinetics:\n"
                "  `v_true(S) = V_max * S / (K_m + S + (S ** 2) / K_i)` (`V_max > 0, K_m > 0, K_i > 0`),\n"
                "  with `rate_raw = B_p + G_p(row, col) * v_true(S) + noise`.\n"
                "Inspect `/holdout_task/public_enzyme_plates.json` and return:\n"
                "  `{variant_id: {'V_max': float, 'K_m': float, 'K_i': float}}`.\n"
                "Target: Max relative parameter error `<= 0.12` across all variants on held-out multi-plate batches (`< 15s`)."
            ),
            public_setup_py=r'''
import json, numpy as np

def sim_batches(seed, variants):
    rng = np.random.default_rng(seed)
    concs = np.geomspace(0.06, 36.0, 12)
    plates = []
    v_ids = sorted(variants.keys())
    for p_idx in range(3):
        bg = float(rng.uniform(12.0, 24.0))
        c0 = float(rng.uniform(85.0, 115.0))
        cr, cc = float(rng.uniform(-14.0, 14.0)), float(rng.uniform(-18.0, 18.0))
        crr, ccc, crc = float(rng.uniform(-9.0, 9.0)), float(rng.uniform(-9.0, 9.0)), float(rng.uniform(-6.0, 6.0))
        coords = [(r, c) for r in range(8) for c in range(12)]
        rng.shuffle(coords)
        wells = []
        for idx, (r, c) in enumerate(coords):
            u, v = (r - 3.5) / 3.5, (c - 5.5) / 5.5
            gain = c0 + cr * u + cc * v + crr * (u**2) + ccc * (v**2) + crc * u * v
            if idx < 8:
                wtype, vid, s_val, rate_true = "BLANK", None, 0.0, 0.0
            elif idx < 36:
                wtype, vid, s_val, rate_true = "REF_WT", None, 1.0, 1.0
            else:
                vid = v_ids[(idx - 36) % len(v_ids)]
                s_val = float(concs[((idx - 36) // len(v_ids) + 4 * p_idx) % len(concs)])
                vm, km, ki = variants[vid]["V_max"], variants[vid]["K_m"], variants[vid]["K_i"]
                rate_true = vm * s_val / (km + s_val + (s_val**2) / ki)
                wtype = "VARIANT"
            obs = bg + gain * rate_true + float(rng.normal(0.0, 0.9))
            if wtype != "BLANK" and rng.random() < 0.04:
                obs += float(rng.choice([-38.0, 42.0]))
            wells.append({"row": r, "col": c, "well_type": wtype, "variant_id": vid, "substrate_mM": round(s_val, 5), "raw_rate": round(obs, 4)})
        plates.append({"plate_id": f"P{p_idx+1}", "wells": wells})
    return plates

gt_pub = {
    "VAR_A": {"V_max": 2.4, "K_m": 0.45, "K_i": 8.5},
    "VAR_B": {"V_max": 1.6, "K_m": 1.15, "K_i": 14.0},
    "VAR_C": {"V_max": 3.1, "K_m": 0.75, "K_i": 5.2},
}
with open("/holdout_task/public_enzyme_plates.json", "w") as fh:
    json.dump({"plates": sim_batches(41, gt_pub), "ground_truth": gt_pub}, fh, indent=2)
''',
            private_verifier_py=r'''
import json, sys, numpy as np
sys.path.insert(0, "/holdout_task")
import enzyme_plate_solver

def sim_batches(seed, variants):
    rng = np.random.default_rng(seed)
    concs = np.geomspace(0.05, 40.0, 12)
    plates = []
    v_ids = sorted(variants.keys())
    for p_idx in range(4):
        bg = float(rng.uniform(10.0, 28.0))
        c0 = float(rng.uniform(80.0, 125.0))
        cr, cc = float(rng.uniform(-16.0, 16.0)), float(rng.uniform(-20.0, 20.0))
        crr, ccc, crc = float(rng.uniform(-11.0, 11.0)), float(rng.uniform(-11.0, 11.0)), float(rng.uniform(-7.0, 7.0))
        coords = [(r, c) for r in range(8) for c in range(12)]
        rng.shuffle(coords)
        wells = []
        for idx, (r, c) in enumerate(coords):
            u, v = (r - 3.5) / 3.5, (c - 5.5) / 5.5
            gain = c0 + cr * u + cc * v + crr * (u**2) + ccc * (v**2) + crc * u * v
            if idx < 8:
                wtype, vid, s_val, rate_true = "BLANK", None, 0.0, 0.0
            elif idx < 36:
                wtype, vid, s_val, rate_true = "REF_WT", None, 1.0, 1.0
            else:
                vid = v_ids[(idx - 36) % len(v_ids)]
                s_val = float(concs[((idx - 36) // len(v_ids) + 3 * p_idx) % len(concs)])
                vm, km, ki = variants[vid]["V_max"], variants[vid]["K_m"], variants[vid]["K_i"]
                rate_true = vm * s_val / (km + s_val + (s_val**2) / ki)
                wtype = "VARIANT"
            obs = bg + gain * rate_true + float(rng.normal(0.0, 0.95))
            if wtype != "BLANK" and rng.random() < 0.04:
                obs += float(rng.choice([-40.0, 45.0]))
            wells.append({"row": r, "col": c, "well_type": wtype, "variant_id": vid, "substrate_mM": round(s_val, 5), "raw_rate": round(obs, 4)})
        plates.append({"plate_id": f"H{p_idx+1}", "wells": wells})
    return plates

gt_priv = {
    "ENZ_101": {"V_max": 2.75, "K_m": 0.55, "K_i": 7.2},
    "ENZ_102": {"V_max": 1.45, "K_m": 1.35, "K_i": 16.5},
    "ENZ_103": {"V_max": 3.40, "K_m": 0.85, "K_i": 4.8},
    "ENZ_104": {"V_max": 2.05, "K_m": 0.35, "K_i": 11.0},
}
rel_errs = []
all_valid = True
for sd in (911, 912):
    est = enzyme_plate_solver.estimate_variant_haldane_parameters(sim_batches(sd, gt_priv))
    if not isinstance(est, dict) or set(est.keys()) != set(gt_priv.keys()):
        all_valid = False; break
    for vid, g in gt_priv.items():
        if not isinstance(est[vid], dict):
            all_valid = False; break
        for k in ("V_max", "K_m", "K_i"):
            v_hat = est[vid].get(k)
            if isinstance(v_hat, bool) or not isinstance(v_hat, (int, float, np.floating, np.integer)) or not np.isfinite(float(v_hat)):
                all_valid = False; break
            rel_errs.append(abs(float(v_hat) - g[k]) / g[k])
        if not all_valid:
            break
    if not all_valid:
        break
mx = max(rel_errs) if (all_valid and len(rel_errs) == 2 * len(gt_priv) * 3) else None
print(json.dumps({"strict_pass": bool(mx is not None and mx <= 0.12), "max_rel_param_err": (round(mx, 5) if mx is not None else None)}, allow_nan=False))
''',
        ),
        HoldoutOODTask(
            task_id="uf03__orthotropic_robin_heat_sink_inversion",
            short_name="robin_heat_sink_pde_inversion",
            wave="wave_2",
            scientific_domain="Thermal Physics / Conservative Finite-Volume PDE Inversion",
            wormhole_family="fam_unseen_elliptic_robin_inversion",
            declared_artifacts=("/holdout_task/thermal_sink_solver.py",),
            instruction_text=(
                "Write `/holdout_task/thermal_sink_solver.py` exposing:\n"
                "  `localize_thermal_source(measurement: dict) -> dict`\n"
                "A 2D plate `[0, 1] x [0, 1]` is discretized into `N x N` uniform cells (`h = 1.0 / N`, cell centers "
                "`x_i = (i + 0.5) * h`, `y_j = (j + 0.5) * h`) with known cell conductivity `kappa[i, j] > 0`.\n"
                "- Interior face conductances use harmonic averaging `2 * k_a * k_b / (k_a + k_b)` (flux balance `(A_flux + A_robin) u = q_cell * h**2`).\n"
                "- Every exterior boundary cell face dissipates heat to ambient `0` via Robin conductance `gamma_robin * h` "
                "(so each boundary face of cell `(i, j)` adds `gamma_robin * h` to the diagonal of `A`).\n"
                "- Volumetric source at cell center `(x_i, y_j)` is `q_cell[i, j] = amplitude * exp(-0.5 * ((x_i - x_s)**2 + (y_j - y_s)**2) / sigma_m**2)` (`amplitude > 0`).\n"
                "- `M` thermistors at cell indices `sensor_ij` record `T_obs[k] = u[i_k, j_k] + junction_offset + noise`, "
                "with `~6%` stuck-thermistor gross outliers!\n"
                "Inspect `/holdout_task/public_thermal_cases.pkl` and return:\n"
                "  `{'source_xy': [float(x_s), float(y_s)], 'amplitude': float, 'junction_offset': float}`.\n"
                "Target: `source_xy` error `<= 0.035`, `amplitude` relative error `<= 0.08`, and `junction_offset` error `<= 0.008` (`< 15s`)."
            ),
            public_setup_py=r'''
import pickle, numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def build_robin_fvm(kappa, gamma_robin):
    N = kappa.shape[0]; h = 1.0 / N
    rows, cols, vals = [], [], []
    diag = np.zeros(N * N, dtype=float)
    for i in range(N):
        for j in range(N):
            u = i * N + j
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    v = ni * N + nj
                    g = 2.0 * kappa[i, j] * kappa[ni, nj] / (kappa[i, j] + kappa[ni, nj])
                    diag[u] += g
                    rows.append(u); cols.append(v); vals.append(-g)
                else:
                    diag[u] += gamma_robin * h
    idx = np.arange(N * N)
    return sparse.csr_matrix((np.r_[vals, diag], (np.r_[rows, idx], np.r_[cols, idx])), shape=(N * N, N * N))

def make_case(seed, xs, ys, amp, j_off):
    rng = np.random.default_rng(seed)
    N = 22; h = 1.0 / N; sigma = 0.085; gamma = 2.2
    xc = (np.arange(N) + 0.5) * h
    gx, gy = np.meshgrid(xc, xc, indexing="ij")
    kappa = np.exp(0.55 * np.sin(2 * np.pi * gx) * np.cos(2 * np.pi * gy) + 0.25 * gx)
    A = build_robin_fvm(kappa, gamma)
    q = amp * np.exp(-0.5 * ((gx - xs)**2 + (gy - ys)**2) / (sigma**2))
    u = spsolve(A, (q * (h**2)).ravel()).reshape(N, N)
    flat_cells = rng.choice(N * N, size=52, replace=False)
    s_ij = np.column_stack([flat_cells // N, flat_cells % N])
    T_obs = u[s_ij[:, 0], s_ij[:, 1]] + j_off + rng.normal(0.0, 0.0012, size=len(s_ij))
    for oi in rng.choice(len(T_obs), size=3, replace=False):
        T_obs[int(oi)] += float(rng.choice([-0.22, 0.25]))
    return {
        "N": N, "kappa": kappa, "gamma_robin": gamma, "sigma_m": sigma,
        "sensor_ij": s_ij, "T_obs": T_obs,
    }, {"source_xy": [xs, ys], "amplitude": amp, "junction_offset": j_off}

pub = [make_case(51, 0.34, 0.62, 18.0, 0.045), make_case(52, 0.68, 0.29, 24.5, -0.032)]
with open("/holdout_task/public_thermal_cases.pkl", "wb") as fh:
    pickle.dump([{"measurement": m, "gt": g} for m, g in pub], fh)
''',
            private_verifier_py=r'''
import json, sys, numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
sys.path.insert(0, "/holdout_task")
import thermal_sink_solver

def build_robin_fvm(kappa, gamma_robin):
    N = kappa.shape[0]; h = 1.0 / N
    rows, cols, vals = [], [], []
    diag = np.zeros(N * N, dtype=float)
    for i in range(N):
        for j in range(N):
            u = i * N + j
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    v = ni * N + nj
                    g = 2.0 * kappa[i, j] * kappa[ni, nj] / (kappa[i, j] + kappa[ni, nj])
                    diag[u] += g
                    rows.append(u); cols.append(v); vals.append(-g)
                else:
                    diag[u] += gamma_robin * h
    idx = np.arange(N * N)
    return sparse.csr_matrix((np.r_[vals, diag], (np.r_[rows, idx], np.r_[cols, idx])), shape=(N * N, N * N))

def make_case(seed, xs, ys, amp, j_off):
    rng = np.random.default_rng(seed)
    N = 24; h = 1.0 / N; sigma = 0.085; gamma = 2.5
    xc = (np.arange(N) + 0.5) * h
    gx, gy = np.meshgrid(xc, xc, indexing="ij")
    kappa = np.exp(0.6 * np.cos(2 * np.pi * gx) * np.sin(2 * np.pi * gy) + 0.3 * gy)
    A = build_robin_fvm(kappa, gamma)
    q = amp * np.exp(-0.5 * ((gx - xs)**2 + (gy - ys)**2) / (sigma**2))
    u = spsolve(A, (q * (h**2)).ravel()).reshape(N, N)
    flat_cells = rng.choice(N * N, size=56, replace=False)
    s_ij = np.column_stack([flat_cells // N, flat_cells % N])
    T_obs = u[s_ij[:, 0], s_ij[:, 1]] + j_off + rng.normal(0.0, 0.0012, size=len(s_ij))
    for oi in rng.choice(len(T_obs), size=4, replace=False):
        T_obs[int(oi)] += float(rng.choice([-0.24, 0.26]))
    return {"N": N, "kappa": kappa, "gamma_robin": gamma, "sigma_m": sigma, "sensor_ij": s_ij, "T_obs": T_obs}

xy_e, amp_e, off_e = [], [], []
all_valid = True
for sd, xs, ys, amp, j_off in ((921, 0.27, 0.71, 21.0, -0.041), (922, 0.73, 0.38, 16.5, 0.052), (923, 0.46, 0.48, 26.0, -0.019)):
    out = thermal_sink_solver.localize_thermal_source(make_case(sd, xs, ys, amp, j_off))
    if not isinstance(out, dict) or not all(k in out for k in ("source_xy", "amplitude", "junction_offset")):
        all_valid = False; break
    xy_arr = np.asarray(out["source_xy"], dtype=float)
    amp_raw, off_raw = out["amplitude"], out["junction_offset"]
    if (
        xy_arr.shape != (2,)
        or not np.all(np.isfinite(xy_arr))
        or isinstance(amp_raw, bool)
        or isinstance(off_raw, bool)
        or not isinstance(amp_raw, (int, float, np.floating, np.integer))
        or not isinstance(off_raw, (int, float, np.floating, np.integer))
        or not (np.isfinite(float(amp_raw)) and np.isfinite(float(off_raw)))
    ):
        all_valid = False; break
    xy_e.append(float(np.linalg.norm(xy_arr - np.array([xs, ys]))))
    amp_e.append(abs(float(amp_raw) - amp) / amp)
    off_e.append(abs(float(off_raw) - j_off))
passed = bool(all_valid and len(xy_e) == 3 and max(xy_e) <= 0.035 and max(amp_e) <= 0.08 and max(off_e) <= 0.008)
print(json.dumps({"strict_pass": passed, "max_xy_err": (round(max(xy_e), 5) if all_valid and xy_e else None), "max_amp_rel_err": (round(max(amp_e), 5) if all_valid and amp_e else None), "max_offset_err": (round(max(off_e), 5) if all_valid and off_e else None)}, allow_nan=False))
''',
        ),
        HoldoutOODTask(
            task_id="uf04__certified_box_l0_sensor_selection",
            short_name="certified_box_l0_selection",
            wave="wave_2",
            scientific_domain="Optimization / Exact Certified Box-L0 Regression",
            wormhole_family="fam_unseen_certified_box_l0",
            declared_artifacts=("/holdout_task/certified_l0_solver.py",),
            instruction_text=(
                "Write `/holdout_task/certified_l0_solver.py` exposing:\n"
                "  `solve_certified_box_l0(instance: dict) -> dict`\n"
                "Given `X` (`n x p`, `p <= 15`), `y` (`n,`), `ridge > 0`, `box > 0`, and `l0_penalty > 0`, find `w in [-box, box]^p` "
                "minimizing exact objective:\n"
                "  `J(w) = 0.5 * sum((X @ w - y)**2) + 0.5 * ridge * sum(w**2) + l0_penalty * count(|w_j| > 1e-9)`\n"
                "AND compute a mathematically valid global lower bound `certified_lower_bound <= J_opt` with duality/branch-and-bound "
                "certificate gap `J(w) - certified_lower_bound <= 1e-5` (using `<= 600` branch-and-bound nodes, NOT brute-force `2^p` enumeration!).\n"
                "Inspect `/holdout_task/public_box_l0_instances.pkl` and return:\n"
                "  `{'weights': list[float], 'objective': float, 'certified_lower_bound': float, 'nodes_explored': int}`."
            ),
            public_setup_py=r'''
import pickle, numpy as np

def make_inst(seed, n=42, p=13):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, p))
    X[:, 1] = 0.75 * X[:, 0] + 0.65 * X[:, 1]
    w_true = np.zeros(p)
    w_true[[1, 4, 8, 11]] = [1.1, -0.85, 0.95, -1.25]
    y = X @ w_true + rng.normal(0.0, 0.25, size=n)
    return {"X": X, "y": y, "ridge": 0.35, "box": 1.5, "l0_penalty": 0.85}

with open("/holdout_task/public_box_l0_instances.pkl", "wb") as fh:
    pickle.dump([make_inst(61), make_inst(62, p=14)], fh)
''',
            private_verifier_py=r'''
import json, sys, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "/holdout_task")
import certified_l0_solver

def exact_brute_opt(X, y, ridge, box, lam0):
    p = X.shape[1]
    G = X.T @ X + ridge * np.eye(p)
    rhs = X.T @ y
    best = 0.5 * float(np.dot(y, y))
    for mask in range(1, 1 << p):
        idx = [j for j in range(p) if (mask >> j) & 1]
        Gs = G[np.ix_(idx, idx)]; rs = rhs[idx]
        w_unc = np.linalg.solve(Gs, rs)
        if np.all(np.abs(w_unc) <= box):
            val = 0.5 * float(np.dot(y, y) - np.dot(rs, w_unc)) + lam0 * len(idx)
        else:
            res = minimize(lambda z: 0.5 * float(z @ Gs @ z - 2.0 * np.dot(rs, z)), np.clip(w_unc, -box, box),
                           jac=lambda z: Gs @ z - rs, bounds=[(-box, box)] * len(idx), method="L-BFGS-B")
            val = 0.5 * float(np.dot(y, y)) + float(res.fun) + lam0 * len(idx)
        if val < best:
            best = val
    return best

def make_inst(seed, n=46, p=14):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, p))
    X[:, 2] = 0.8 * X[:, 0] + 0.6 * X[:, 2]
    w_true = np.zeros(p)
    w_true[[0, 3, 7, 10, 12]] = [1.2, -0.9, 0.75, -1.35, 0.8]
    y = X @ w_true + rng.normal(0.0, 0.22, size=n)
    return {"X": X, "y": y, "ridge": 0.4, "box": 1.4, "l0_penalty": 0.9}

all_ok = True
max_gap = 0.0
max_nodes = 0
checked_count = 0
for sd in (931, 932, 933):
    inst = make_inst(sd)
    X, y, ridge, box, lam0 = inst["X"], inst["y"], inst["ridge"], inst["box"], inst["l0_penalty"]
    j_star = exact_brute_opt(X, y, ridge, box, lam0)
    out = certified_l0_solver.solve_certified_box_l0(inst)
    if not isinstance(out, dict) or not all(k in out for k in ("weights", "objective", "certified_lower_bound", "nodes_explored")):
        all_ok = False; break
    w_raw, obj_raw, lb_raw, nodes_raw = out["weights"], out["objective"], out["certified_lower_bound"], out["nodes_explored"]
    if (
        isinstance(obj_raw, bool)
        or isinstance(lb_raw, bool)
        or isinstance(nodes_raw, bool)
        or not isinstance(obj_raw, (int, float, np.floating, np.integer))
        or not isinstance(lb_raw, (int, float, np.floating, np.integer))
        or not isinstance(nodes_raw, (int, np.integer))
    ):
        all_ok = False; break
    try:
        w_list = list(w_raw)
    except TypeError:
        all_ok = False; break
    if any(isinstance(el, bool) or not isinstance(el, (int, float, np.floating, np.integer)) for el in w_list):
        all_ok = False; break
    w = np.asarray(w_list, dtype=float)
    obj_rep = float(obj_raw)
    lb = float(lb_raw)
    nodes = int(nodes_raw)
    if w.shape != (X.shape[1],) or not np.all(np.isfinite(w)) or not np.isfinite(obj_rep) or not np.isfinite(lb):
        all_ok = False; break
    if np.max(np.abs(w)) > box + 1e-7 or nodes <= 0 or nodes > 600:
        all_ok = False; break
    j_w = 0.5 * float(np.sum((X @ w - y) ** 2) + ridge * np.sum(w ** 2)) + lam0 * int(np.sum(np.abs(w) > 1e-9))
    gap = j_w - lb
    if not (np.isfinite(j_w) and np.isfinite(gap)):
        all_ok = False; break
    # Match exact task specification (`gap <= 1e-5`, consistent `objective == J(w)`, and valid lower bound `lb <= J_opt`)!
    if abs(obj_rep - j_w) > 1e-6 or lb > j_star + 1e-5 or j_w > j_star + 1e-5 or gap < -1e-5 or gap > 1e-5:
        all_ok = False; break
    max_gap = max(max_gap, abs(gap), abs(j_w - j_star))
    max_nodes = max(max_nodes, nodes)
    checked_count += 1
if checked_count != 3:
    all_ok = False
print(json.dumps({"strict_pass": bool(all_ok), "max_gap": (round(max_gap, 7) if all_ok else None), "max_nodes_explored": (max_nodes if all_ok else None)}, allow_nan=False))
''',
        ),
    ]


def unseen_postfreeze_tasks_by_id() -> dict[str, HoldoutOODTask]:
    return {t.task_id: t for t in build_unseen_postfreeze_4_tasks()}


def unseen_postfreeze_tasks_as_frontier_list() -> list[m6f.FrontierBenchmarkTask]:
    """Convert `Hegel-Unseen-PostFreeze-4` into `FrontierBenchmarkTask` objects (`private_verifier_py` physically excluded!)."""
    ensure_clean_scipy_sandbox_image()
    out: list[m6f.FrontierBenchmarkTask] = []
    for t in build_unseen_postfreeze_4_tasks():
        out.append(
            m6f.FrontierBenchmarkTask(
                task_id=t.task_id,
                short_name=t.short_name,
                benchmark="Hegel-Unseen-PostFreeze-4",
                wave=t.wave,
                scientific_domain=t.scientific_domain,
                wormhole_family=t.wormhole_family,
                instruction_text=t.instruction_text,
                agent_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
                verifier_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
                declared_artifacts=t.declared_artifacts,
                verifier_timeout_sec=120,
                agent_cpus=2,
                agent_memory_mb=2048,
                verifier_cpus=2,
                verifier_memory_mb=2048,
                holdout_public_setup_py=t.public_setup_py,
            )
        )
    return out


def run_unseen_postfreeze_private_verifier_by_id(task_id: str, submitted_tar_path: Path) -> dict[str, Any]:
    uf_task = unseen_postfreeze_tasks_by_id()[task_id]
    res = run_holdout_private_verifier(uf_task, submitted_tar_path)
    res["benchmark"] = "Hegel-Unseen-PostFreeze-4"
    return res


def run_unseen_postfreeze_private_verifier_for_container(task_id: str, cid_with_artifacts: str) -> dict[str, Any]:
    uf_task = unseen_postfreeze_tasks_by_id()[task_id]
    with tempfile.TemporaryDirectory(prefix="hegel_uf_cnt_ver_") as td:
        tar_fp = Path(td) / "submitted_artifacts.tar"
        m6f.export_declared_artifacts_tar(cid_with_artifacts, uf_task.declared_artifacts, tar_fp)
        res = run_holdout_private_verifier(uf_task, tar_fp)
        res["benchmark"] = "Hegel-Unseen-PostFreeze-4"
        return res
