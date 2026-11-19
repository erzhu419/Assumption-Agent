"""M3 Large-Scale Cross-Domain Forest ($K = 36$ Episodes, 18 Python Mechanisms) & HippoRAG Benchmark (Hegel V17.0).

Scales the audited M2.5 open-workspace Python harness from $K=7$ to **$K = 36$ source & near-miss
workspace episodes** spanning **18 distinct executable Python mechanisms** across **36 domains**,
evaluating **24 unseen cross-domain target workspaces per cluster** ($\times 4$ frozen clusters
$= 96$ targets/arm, $672$ target workspaces across 7 controlled arms) plus dedicated out-of-scope
honesty controls!

Strict Single-Factor Fairness Guarantees:
- Every arm uses the **exact same** subprocess Python synthesizer (`_M3_SYNTHESIZER_SCRIPT`),
  pure-dataflow `canonicalize_python_model_ast`, prefix-isolated confirmation validator, and
  closed read-only benchmark evaluator.
- `Hegel_M3_HippoPPR`, `Flat_BM25_EqualMetadata`, `Flat_Overlap_EqualMetadata`,
  `Forest_VerticalPPR_NoWormholes`, and `HippoPPR_UncheckedRecall` consume the **EXACT SAME
  frozen `PhilosophicalForest` object** (identical nodes, contracts, lenses, exclusions, and
  `Stage-1` gate), changing **ONLY** the retrieval / contract-gating factor!
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from random import Random
import shutil
import sys
import time
from typing import Any, Callable

from .code_agent import evaluate_python_model
from .cross_domain_clusters import CostMeter
from .hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from .models import Observation
from .open_workspace_labs import (
    OpenWorkspaceTaskLab,
    _evaluate_workspace_submission,
    _sim_oscillatory_carrier_trace,
)
from .open_workspace_wormhole import (
    _active_and_fixed_features,
    _estimate_observation_lens,
    _frac_coord,
    _freeze_confirmed_submission,
    canonicalize_python_model_ast,
    validate_workspace_model_on_rows,
)
from .proposers import _run
from .wormhole_forest import (
    ClaimContract,
    ObservationMap,
    PhilosophicalForest,
    TaskView,
    TwoStageWormholeGate,
    authorize_abstraction_use,
    compose_wormhole_chain,
)


# ============================================================================
# 1. Subprocess Synthesizer Supporting All 18 Distinct Python Mechanisms
# ============================================================================

_M3_SYNTHESIZER_SCRIPT = r'''import json
import math
from pathlib import Path

cfg = json.loads(Path("synth_config.json").read_text(encoding="utf-8"))
raw_rows = json.loads(Path("observations.json").read_text(encoding="utf-8"))
train_idx = cfg["train_indices"]
val_idx = cfg["val_indices"]
lens = cfg["observation_lens"]
allowed_schemas = cfg.get("allowed_schemas")

sign = float(lens.get("sign", 1.0))
scale = max(1e-9, float(lens.get("scale", 1.0)))
offset = float(lens.get("offset", 0.0))

def to_latent(y_raw: float) -> float:
    return (float(y_raw) - offset) / (sign * scale)

def to_raw(y_lat: float) -> float:
    return offset + sign * scale * float(y_lat)

rows_lat = [
    {"features": r["features"], "y_raw": float(r["target"]), "y_lat": to_latent(r["target"])}
    for r in raw_rows
]
train_rows = [rows_lat[i] for i in train_idx]
val_rows = [rows_lat[i] for i in val_idx] if val_idx else train_rows

fnames = list(cfg["feature_names"])
bounds = cfg["feature_bounds"]
active_f = [f for f in fnames if (bounds[f][1] - bounds[f][0]) > 1e-9]

def solve_linear_ls(basis_fn, rows):
    if not rows:
        return [0.0]
    m = len(basis_fn(rows[0]["features"]))
    ata = [[1e-8 if i == j else 0.0 for j in range(m)] for i in range(m)]
    aty = [0.0 for _ in range(m)]
    for r in rows:
        b = basis_fn(r["features"])
        y = r["y_lat"]
        for i in range(m):
            aty[i] += b[i] * y
            for j in range(m):
                ata[i][j] += b[i] * b[j]
    aug = [ata[i] + [aty[i]] for i in range(m)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r_i: abs(aug[r_i][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        if abs(aug[col][col]) > 1e-12:
            for r_i in range(col + 1, m):
                fac = aug[r_i][col] / aug[col][col]
                for c_i in range(col, m + 1):
                    aug[r_i][c_i] -= fac * aug[col][c_i]
    w = [0.0] * m
    for r_i in range(m - 1, -1, -1):
        if abs(aug[r_i][r_i]) > 1e-12:
            w[r_i] = (aug[r_i][m] - sum(aug[r_i][c_i] * w[c_i] for c_i in range(r_i + 1, m))) / aug[r_i][r_i]
    return w

def rmse_raw(pred_lat_fn, rows):
    if not rows:
        return float("inf")
    return math.sqrt(sum((to_raw(pred_lat_fn(r["features"])) - r["y_raw"]) ** 2 for r in rows) / len(rows))

def allow(tag):
    return allowed_schemas is None or tag in allowed_schemas

candidates = []

# ==================== 1D MECHANISMS (6) ====================
if len(active_f) == 1 or (allowed_schemas is not None and len(allowed_schemas) == 1 and allowed_schemas[0].endswith("_1d")):
    for u_k in (active_f if len(active_f) > 1 else [active_f[0]]):
        if allow("linear_proportional_1d"):
            w = solve_linear_ls(lambda f, k=u_k: [1.0, float(f[k])], train_rows)
            w0, w1 = max(1e-5, w[0]) if w[0] >= 0 else w[0], max(1e-5, w[1])
            fn = lambda f, a=w0, b=w1, k=u_k: a + b * float(f[k])
            expr = f"({abs(w0):.10g} + {w1:.10g} * f[{u_k!r}])" if w0 >= 0 else f"(-{abs(w0):.10g} + {w1:.10g} * f[{u_k!r}])"
            if w0 >= 0:
                fn = lambda f, a=abs(w0), b=w1, k=u_k: a + b * float(f[k])
            candidates.append(("linear_proportional_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

        if allow("rational_monod_1d"):
            pts = [(float(r["features"][u_k]), r["y_lat"]) for r in train_rows if float(r["features"][u_k]) > 0 and r["y_lat"] > 1e-6]
            if len(pts) >= 2:
                xs = [1.0 / u for u, _ in pts]
                zs = [1.0 / y for _, y in pts]
                ws = [max(1e-8, y ** 4) for _, y in pts]
                wsum = sum(ws)
                mx = sum(w * x for w, x in zip(ws, xs)) / wsum
                mz = sum(w * z for w, z in zip(ws, zs)) / wsum
                den = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
                b_p = max(1e-5, sum(w * (x - mx) * (z - mz) for w, x, z in zip(ws, xs, zs)) / den if den > 1e-12 else 0.2)
                a_p = max(1e-5, mz - b_p * mx)
                for _ in range(30):
                    ga = gb = haa = hab = hbb = 0.0
                    cur = sum(((u / (a_p * u + b_p)) - y) ** 2 for u, y in pts)
                    for u, y in pts:
                        d = a_p * u + b_p
                        e = (u / d) - y
                        ja, jb = -(u * u) / (d * d), -u / (d * d)
                        ga += e * ja; gb += e * jb; haa += ja * ja; hab += ja * jb; hbb += jb * jb
                    det = (haa + 1e-6) * (hbb + 1e-6) - hab * hab
                    if det <= 1e-15:
                        break
                    na = max(1e-6, a_p - 0.8 * (((hbb + 1e-6) * ga - hab * gb) / det))
                    nb = max(1e-6, b_p - 0.8 * (((haa + 1e-6) * gb - hab * ga) / det))
                    if sum(((u / (na * u + nb)) - y) ** 2 for u, y in pts) <= cur:
                        a_p, b_p = na, nb
                fn = lambda f, a=a_p, b=b_p, k=u_k: float(f[k]) / (a * float(f[k]) + b)
                expr = f"(f[{u_k!r}] / ({a_p:.10g} * f[{u_k!r}] + {b_p:.10g}))"
                candidates.append(("rational_monod_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

        if allow("congestion_rollover_1d") and len(train_rows) >= 3:
            pts = [(float(r["features"][u_k]), r["y_lat"]) for r in train_rows if float(r["features"][u_k]) > 0 and r["y_lat"] > 1e-6]
            if len(pts) >= 3:
                pseudo = [{"features": {u_k: u}, "y_lat": u / y} for u, y in pts]
                w = solve_linear_ls(lambda f, k=u_k: [1.0, float(f[k]), float(f[k]) ** 2], pseudo)
                b_r, a_r, g_r = max(1e-5, w[0]), max(1e-5, w[1]), max(1e-6, w[2])
                fn = lambda f, a=a_r, b=b_r, g=g_r, k=u_k: float(f[k]) / (a * float(f[k]) + b + g * (float(f[k]) ** 2))
                expr = f"(f[{u_k!r}] / ({a_r:.10g} * f[{u_k!r}] + {b_r:.10g} + {g_r:.10g} * (f[{u_k!r}] ** 2)))"
                candidates.append(("congestion_rollover_1d", 3, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

        if allow("hill_cooperative_n2_1d"):
            pts = [(float(r["features"][u_k]), r["y_lat"]) for r in train_rows if float(r["features"][u_k]) > 0 and r["y_lat"] > 1e-6]
            if len(pts) >= 2:
                xs = [1.0 / (u * u) for u, _ in pts]
                zs = [1.0 / y for _, y in pts]
                ws = [max(1e-8, y ** 4) for _, y in pts]
                wsum = sum(ws)
                mx = sum(w * x for w, x in zip(ws, xs)) / wsum
                mz = sum(w * z for w, z in zip(ws, zs)) / wsum
                den = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
                b_p = max(1e-5, sum(w * (x - mx) * (z - mz) for w, x, z in zip(ws, xs, zs)) / den if den > 1e-12 else 0.5)
                a_p = max(1e-5, mz - b_p * mx)
                for _ in range(30):
                    ga = gb = haa = hab = hbb = 0.0
                    cur = sum((((u * u) / (a_p * u * u + b_p)) - y) ** 2 for u, y in pts)
                    for u, y in pts:
                        u2 = u * u
                        d = a_p * u2 + b_p
                        e = (u2 / d) - y
                        ja, jb = -(u2 * u2) / (d * d), -u2 / (d * d)
                        ga += e * ja; gb += e * jb; haa += ja * ja; hab += ja * jb; hbb += jb * jb
                    det = (haa + 1e-6) * (hbb + 1e-6) - hab * hab
                    if det <= 1e-15:
                        break
                    na = max(1e-6, a_p - 0.8 * (((hbb + 1e-6) * ga - hab * gb) / det))
                    nb = max(1e-6, b_p - 0.8 * (((haa + 1e-6) * gb - hab * ga) / det))
                    if sum((((u * u) / (na * u * u + nb)) - y) ** 2 for u, y in pts) <= cur:
                        a_p, b_p = na, nb
                fn = lambda f, a=a_p, b=b_p, k=u_k: (float(f[k]) ** 2) / (a * (float(f[k]) ** 2) + b)
                expr = f"((f[{u_k!r}] ** 2) / ({a_p:.10g} * (f[{u_k!r}] ** 2) + {b_p:.10g}))"
                candidates.append(("hill_cooperative_n2_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

        if allow("inv_quad_crest_1d") and len(train_rows) >= 3:
            w = solve_linear_ls(lambda f, k=u_k: [1.0, 1.0 / float(f[k]), 1.0 / (float(f[k]) ** 2)], train_rows)
            w0, w1, w2 = max(1e-5, w[0]), max(1e-5, w[1]), max(1e-5, w[2])
            fn = lambda f, a=w0, b=w1, c=w2, k=u_k: a + b / float(f[k]) + c / (float(f[k]) ** 2)
            expr = f"({w0:.10g} + {w1:.10g} / f[{u_k!r}] + {w2:.10g} / (f[{u_k!r}] ** 2))"
            candidates.append(("inv_quad_crest_1d", 3, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

        if allow("exp_washout_decay_1d") and len(train_rows) >= 3:
            best_ew = None
            for tau in [1.5 + 0.35 * i for i in range(32)]:
                w = solve_linear_ls(lambda f, k=u_k, tv=tau: [1.0, math.exp(-float(f[k]) / tv)], train_rows)
                if w[0] > 0 and w[1] > 0:
                    fn = lambda f, a=w[0], b=w[1], tv=tau, k=u_k: a + b * math.exp(-float(f[k]) / tv)
                    tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                    sc = 0.45 * tr_e + 0.55 * va_e
                    if best_ew is None or sc < best_ew[0]:
                        best_ew = (sc, tr_e, va_e, tau, w[0], w[1])
            if best_ew is not None:
                tau_b = best_ew[3]
                for step in (0.12, 0.03, 0.008):
                    for dtau in (-step, step):
                        nt = max(0.2, tau_b + dtau)
                        w = solve_linear_ls(lambda f, k=u_k, tv=nt: [1.0, math.exp(-float(f[k]) / tv)], train_rows)
                        if w[0] > 0 and w[1] > 0:
                            fn = lambda f, a=w[0], b=w[1], tv=nt, k=u_k: a + b * math.exp(-float(f[k]) / tv)
                            tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                            sc = 0.45 * tr_e + 0.55 * va_e
                            if sc < best_ew[0]:
                                tau_b = nt
                                best_ew = (sc, tr_e, va_e, nt, w[0], w[1])
                expr = f"({best_ew[4]:.10g} + {best_ew[5]:.10g} * math.exp(-f[{u_k!r}] / {best_ew[3]:.10g}))"
                candidates.append(("exp_washout_decay_1d", 3, best_ew[1], best_ew[2], expr, [u_k]))

        if allow("inv_linear_decay_1d"):
            w = solve_linear_ls(lambda f, k=u_k: [1.0, 1.0 / float(f[k])], train_rows)
            w0, w1 = max(1e-5, w[0]), max(1e-5, w[1])
            fn = lambda f, a=w0, b=w1, k=u_k: a + b / float(f[k])
            expr = f"({w0:.10g} + {w1:.10g} / f[{u_k!r}])"
            candidates.append(("inv_linear_decay_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

        if allow("sqrt_orifice_1d"):
            w = solve_linear_ls(lambda f, k=u_k: [1.0, math.sqrt(max(0.0, float(f[k])))], train_rows)
            w0, w1 = max(1e-5, w[0]), max(1e-5, w[1])
            fn = lambda f, a=w0, b=w1, k=u_k: a + b * math.sqrt(max(0.0, float(f[k])))
            expr = f"({w0:.10g} + {w1:.10g} * math.sqrt(f[{u_k!r}]))"
            candidates.append(("sqrt_orifice_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

        if allow("log_saturation_1d"):
            w = solve_linear_ls(lambda f, k=u_k: [1.0, math.log(1.0 + max(0.0, float(f[k])))], train_rows)
            w0, w1 = max(1e-5, w[0]), max(1e-5, w[1])
            fn = lambda f, a=w0, b=w1, k=u_k: a + b * math.log(1.0 + max(0.0, float(f[k])))
            expr = f"({w0:.10g} + {w1:.10g} * math.log(1.0 + f[{u_k!r}]))"
            candidates.append(("log_saturation_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

# ==================== 2D & 3D MECHANISMS (12) ====================
if len(active_f) >= 2:
    role_pairs = cfg.get("forced_role_order")
    pairs_to_try = [tuple(role_pairs[:2])] if (role_pairs and len(role_pairs) >= 2) else [
        (f_a, f_b) for f_a in active_f for f_b in active_f if f_a != f_b
    ]
    tau_grid = [0.5 + 0.10 * i for i in range(38)]
    allow_2d_subspace = (len(active_f) == 2) or (allowed_schemas is not None and len(allowed_schemas) == 1)

    if allow_2d_subspace and allow("conjugate_relaxation_reset"):
        best_rel = None
        for d_k, t_k in pairs_to_try:
            for tau in tau_grid:
                w = solve_linear_ls(
                    lambda f, dk=d_k, tk=t_k, tv=tau: [float(f[dk]), float(f[dk]) * math.exp(-float(f[tk]) / tv)],
                    train_rows,
                )
                if w[0] > 0 and w[1] > 0.05:
                    fn = lambda f, w=w, dk=d_k, tk=t_k, tv=tau: w[0] * float(f[dk]) + w[1] * float(f[dk]) * math.exp(-float(f[tk]) / tv)
                    tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                    sc = 0.45 * tr_e + 0.55 * va_e
                    if best_rel is None or sc < best_rel[0]:
                        best_rel = (sc, tr_e, va_e, d_k, t_k, tau, w)
        if best_rel is not None:
            _, tr_e, va_e, d_k, t_k, tau_b, w_b = best_rel
            for step in (0.04, 0.01):
                for dtau in (-step, step):
                    nt = max(0.15, tau_b + dtau)
                    w = solve_linear_ls(lambda f, dk=d_k, tk=t_k, tv=nt: [float(f[dk]), float(f[dk]) * math.exp(-float(f[tk]) / tv)], train_rows)
                    if w[0] > 0 and w[1] > 0.05:
                        fn = lambda f, w=w, dk=d_k, tk=t_k, tv=nt: w[0] * float(f[dk]) + w[1] * float(f[dk]) * math.exp(-float(f[tk]) / tv)
                        ntr, nva = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        nsc = 0.45 * ntr + 0.55 * nva
                        if nsc < best_rel[0]:
                            tau_b, w_b = nt, w
                            best_rel = (nsc, ntr, nva, d_k, t_k, nt, w)
            expr = f"({w_b[0]:.10g} * f[{d_k!r}] + {w_b[1]:.10g} * f[{d_k!r}] * math.exp(-f[{t_k!r}] / {tau_b:.10g}))"
            candidates.append(("conjugate_relaxation_reset", 3, best_rel[1], best_rel[2], expr, [d_k, t_k]))

    if len(active_f) == 2 and allow("overdamped_charging_2d"):
        best_od = None
        for d_k, t_k in pairs_to_try:
            for tau in tau_grid:
                w = solve_linear_ls(
                    lambda f, dk=d_k, tk=t_k, tv=tau: [float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv))],
                    train_rows,
                )
                if w[0] > 0.05:
                    fn = lambda f, k0=w[0], dk=d_k, tk=t_k, tv=tau: k0 * float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv))
                    tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                    sc = 0.45 * tr_e + 0.55 * va_e
                    if best_od is None or sc < best_od[0]:
                        best_od = (sc, tr_e, va_e, d_k, t_k, tau, w[0])
        if best_od is not None:
            _, _, _, d_k, t_k, tau_b, w0_b = best_od
            for step in (0.04, 0.01, 0.0025):
                for dtau in (-step, step):
                    nt = max(0.15, tau_b + dtau)
                    w = solve_linear_ls(lambda f, dk=d_k, tk=t_k, tv=nt: [float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv))], train_rows)
                    if w[0] > 0.05:
                        fn = lambda f, k0=w[0], dk=d_k, tk=t_k, tv=nt: k0 * float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv))
                        ntr, nva = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        nsc = 0.45 * ntr + 0.55 * nva
                        if nsc < best_od[0]:
                            tau_b, w0_b = nt, w[0]
                            best_od = (nsc, ntr, nva, d_k, t_k, nt, w[0])
            expr = f"({w0_b:.10g} * f[{d_k!r}] * (1.0 - math.exp(-f[{t_k!r}] / {tau_b:.10g})))"
            candidates.append(("overdamped_charging_2d", 2, best_od[1], best_od[2], expr, [d_k, t_k]))

    if len(active_f) == 2 and allow("underdamped_ringing_2d") and len(train_rows) >= 4:
        best_ud = None
        for d_k, t_k in pairs_to_try:
            for tau in [1.2 + 0.2 * j for j in range(11)]:
                for om in [1.2 + 0.08 * j for j in range(14)]:
                    w = solve_linear_ls(
                        lambda f, dk=d_k, tk=t_k, tv=tau, wv=om: [
                            float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv) * math.cos(wv * float(f[tk])))
                        ],
                        train_rows,
                    )
                    if w[0] > 0.05:
                        fn = lambda f, k0=w[0], dk=d_k, tk=t_k, tv=tau, wv=om: (
                            k0 * float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv) * math.cos(wv * float(f[tk])))
                        )
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.45 * tr_e + 0.55 * va_e
                        if best_ud is None or sc < best_ud[0]:
                            best_ud = (sc, tr_e, va_e, d_k, t_k, tau, om, w[0])
        if best_ud is not None:
            _, _, _, d_k, t_k, tau_b, om_b, w0_b = best_ud
            for st_t, st_o in [(0.06, 0.03), (0.015, 0.008), (0.004, 0.002)]:
                for _ in range(6):
                    for dt, do in [(st_t,0),(-st_t,0),(0,st_o),(0,-st_o)]:
                        nt, no = max(0.3, tau_b + dt), max(0.4, om_b + do)
                        w = solve_linear_ls(
                            lambda f, dk=d_k, tk=t_k, tv=nt, wv=no: [
                                float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv) * math.cos(wv * float(f[tk])))
                            ],
                            train_rows,
                        )
                        if w[0] > 0.05:
                            fn = lambda f, k0=w[0], dk=d_k, tk=t_k, tv=nt, wv=no: (
                                k0 * float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv) * math.cos(wv * float(f[tk])))
                            )
                            ntr, nva = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                            nsc = 0.45 * ntr + 0.55 * nva
                            if nsc + 1e-9 < best_ud[0]:
                                tau_b, om_b, w0_b = nt, no, w[0]
                                best_ud = (nsc, ntr, nva, d_k, t_k, nt, no, w[0])
            expr = f"({w0_b:.10g} * f[{d_k!r}] * (1.0 - math.exp(-f[{t_k!r}] / {tau_b:.10g}) * math.cos({om_b:.10g} * f[{t_k!r}])))"
            candidates.append(("underdamped_ringing_2d", 3, best_ud[1], best_ud[2], expr, [d_k, t_k]))

    if len(active_f) == 2 and allow("bimodal_relaxation_2mode") and (len(train_rows) >= 7 or allowed_schemas == ["bimodal_relaxation_2mode"]):
        best_bi = None
        for d_k, t_k in pairs_to_try:
            for tf in (0.35, 0.45, 0.55):
                for ts in (4.5, 5.5, 6.0):
                    w = solve_linear_ls(
                        lambda f, dk=d_k, tk=t_k, t1=tf, t2=ts: [
                            float(f[dk]),
                            float(f[dk]) * math.exp(-float(f[tk]) / t1),
                            float(f[dk]) * math.exp(-float(f[tk]) / t2),
                        ],
                        train_rows,
                    )
                    if w[0] > 0 and w[1] > 0.2 and w[2] > 0.2:
                        fn = lambda f, w=w, dk=d_k, tk=t_k, t1=tf, t2=ts: (
                            w[0] * float(f[dk]) + w[1] * float(f[dk]) * math.exp(-float(f[tk]) / t1) + w[2] * float(f[dk]) * math.exp(-float(f[tk]) / t2)
                        )
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.45 * tr_e + 0.55 * va_e
                        if best_bi is None or sc < best_bi[0]:
                            expr = (
                                f"({w[0]:.10g} * f[{d_k!r}] + {w[1]:.10g} * f[{d_k!r}] * math.exp(-f[{t_k!r}] / {tf:.10g}) "
                                f"+ {w[2]:.10g} * f[{d_k!r}] * math.exp(-f[{t_k!r}] / {ts:.10g}))"
                            )
                            best_bi = (sc, tr_e, va_e, expr, [d_k, t_k])
        if best_bi is not None:
            candidates.append(("bimodal_relaxation_2mode", 5, best_bi[1], best_bi[2], best_bi[3], best_bi[4]))

    if len(active_f) == 2 and allow("gaussian_weibull_relax_2d") and len(train_rows) >= 4:
        best_gw = None
        for d_k, t_k in pairs_to_try:
            for t2 in [1.0 + 0.35 * i for i in range(16)]:
                w = solve_linear_ls(
                    lambda f, dk=d_k, tk=t_k, tv2=t2: [float(f[dk]), float(f[dk]) * math.exp(-(float(f[tk]) ** 2) / tv2)],
                    train_rows,
                )
                if w[0] > 0 and w[1] > 0.05:
                    fn = lambda f, w=w, dk=d_k, tk=t_k, tv2=t2: w[0] * float(f[dk]) + w[1] * float(f[dk]) * math.exp(-(float(f[tk]) ** 2) / tv2)
                    tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                    sc = 0.45 * tr_e + 0.55 * va_e
                    if best_gw is None or sc < best_gw[0]:
                        best_gw = (sc, tr_e, va_e, d_k, t_k, t2, w)
        if best_gw is not None:
            _, _, _, d_k, t_k, t2_b, w_b = best_gw
            for step in (0.08, 0.02):
                for dt2 in (-step, step):
                    nt2 = max(0.2, t2_b + dt2)
                    w = solve_linear_ls(lambda f, dk=d_k, tk=t_k, tv2=nt2: [float(f[dk]), float(f[dk]) * math.exp(-(float(f[tk]) ** 2) / tv2)], train_rows)
                    if w[0] > 0 and w[1] > 0.05:
                        fn = lambda f, w=w, dk=d_k, tk=t_k, tv2=nt2: w[0] * float(f[dk]) + w[1] * float(f[dk]) * math.exp(-(float(f[tk]) ** 2) / tv2)
                        ntr, nva = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        nsc = 0.45 * ntr + 0.55 * nva
                        if nsc < best_gw[0]:
                            t2_b, w_b = nt2, w
                            best_gw = (nsc, ntr, nva, d_k, t_k, nt2, w)
            expr = f"({w_b[0]:.10g} * f[{d_k!r}] + {w_b[1]:.10g} * f[{d_k!r}] * math.exp(-(f[{t_k!r}] ** 2) / {t2_b:.10g}))"
            candidates.append(("gaussian_weibull_relax_2d", 3, best_gw[1], best_gw[2], expr, [d_k, t_k]))

    if len(active_f) == 2:
        k1, k2 = active_f[0], active_f[1]
        if allow("liebig_min_2d"):
            r1 = max(r["y_lat"] / max(1e-6, float(r["features"][k1])) for r in train_rows)
            r2 = max(r["y_lat"] / max(1e-6, float(r["features"][k2])) for r in train_rows)
            best_m = None
            for s1 in [0.75 + 0.025 * i for i in range(20)]:
                for s2 in [0.75 + 0.025 * j for j in range(20)]:
                    c1, c2 = max(1e-4, r1 * s1), max(1e-4, r2 * s2)
                    fn = lambda f, a=c1, b=c2, x1=k1, x2=k2: min(a * float(f[x1]), b * float(f[x2]))
                    tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                    sc = 0.5 * tr_e + 0.5 * va_e
                    if best_m is None or sc < best_m[0]:
                        best_m = (sc, tr_e, va_e, c1, c2)
            c1, c2 = best_m[3], best_m[4]
            for step in (0.04, 0.008):
                for _ in range(12):
                    for dc1, dc2 in [(step, 0), (-step, 0), (0, step), (0, -step)]:
                        nc1, nc2 = max(1e-4, c1 + dc1), max(1e-4, c2 + dc2)
                        fn = lambda f, a=nc1, b=nc2, x1=k1, x2=k2: min(a * float(f[x1]), b * float(f[x2]))
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.5 * tr_e + 0.5 * va_e
                        if sc + 1e-9 < best_m[0]:
                            c1, c2 = nc1, nc2
                            best_m = (sc, tr_e, va_e, c1, c2)
            expr = f"min({c1:.10g} * f[{k1!r}], {c2:.10g} * f[{k2!r}])"
            candidates.append(("liebig_min_2d", 2, best_m[1], best_m[2], expr, [k1, k2]))

        if allow("liebig_capped_min_2d") and len(train_rows) >= 4:
            max_obs = max(r["y_lat"] for r in train_rows)
            low_sub = [r for r in train_rows if r["y_lat"] < 0.96 * max_obs]
            base_rows = low_sub if len(low_sub) >= 2 else train_rows
            r1 = max(r["y_lat"] / max(1e-6, float(r["features"][k1])) for r in base_rows)
            r2 = max(r["y_lat"] / max(1e-6, float(r["features"][k2])) for r in base_rows)
            best_cm = None
            for cap_mul in [0.96, 0.99, 1.0, 1.02, 1.06]:
                cap_v = max(1e-4, max_obs * cap_mul)
                for s1 in [0.85 + 0.03 * i for i in range(10)]:
                    for s2 in [0.85 + 0.03 * j for j in range(10)]:
                        c1, c2 = max(1e-4, r1 * s1), max(1e-4, r2 * s2)
                        fn = lambda f, a=c1, b=c2, cp=cap_v, x1=k1, x2=k2: min(a * float(f[x1]), b * float(f[x2]), cp)
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.5 * tr_e + 0.5 * va_e
                        if best_cm is None or sc < best_cm[0]:
                            best_cm = (sc, tr_e, va_e, c1, c2, cap_v)
            c1, c2, cap_v = best_cm[3], best_cm[4], best_cm[5]
            for step in (0.03, 0.006):
                for _ in range(12):
                    for dc1, dc2, dcp in [(step,0,0),(-step,0,0),(0,step,0),(0,-step,0),(0,0,step*3),(0,0,-step*3)]:
                        nc1, nc2, ncp = max(1e-4, c1+dc1), max(1e-4, c2+dc2), max(1e-4, cap_v+dcp)
                        fn = lambda f, a=nc1, b=nc2, cp=ncp, x1=k1, x2=k2: min(a * float(f[x1]), b * float(f[x2]), cp)
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.5 * tr_e + 0.5 * va_e
                        if sc + 1e-9 < best_cm[0]:
                            c1, c2, cap_v = nc1, nc2, ncp
                            best_cm = (sc, tr_e, va_e, c1, c2, cap_v)
            expr = f"min({c1:.10g} * f[{k1!r}], {c2:.10g} * f[{k2!r}], {cap_v:.10g})"
            candidates.append(("liebig_capped_min_2d", 3, best_cm[1], best_cm[2], expr, [k1, k2]))

        if allow("cobb_douglas_sqrt_2d"):
            w = solve_linear_ls(lambda f, x1=k1, x2=k2: [math.sqrt(max(0.0, float(f[x1]) * float(f[x2])))], train_rows)
            c_cd = max(1e-5, w[0])
            fn = lambda f, c=c_cd, x1=k1, x2=k2: c * math.sqrt(max(0.0, float(f[x1]) * float(f[x2])))
            expr = f"({c_cd:.10g} * math.sqrt(f[{k1!r}] * f[{k2!r}]))"
            candidates.append(("cobb_douglas_sqrt_2d", 1, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [k1, k2]))

        if allow("parallel_harmonic_flux_2d"):
            valid_h = [r for r in train_rows if r["y_lat"] > 1e-5 and float(r["features"][k1]) > 0 and float(r["features"][k2]) > 0]
            if len(valid_h) >= 2:
                pseudo = [{"features": r["features"], "y_lat": (float(r["features"][k1]) * float(r["features"][k2])) / r["y_lat"]} for r in valid_h]
                w = solve_linear_ls(lambda f, x1=k1, x2=k2: [float(f[x1]), float(f[x2])], pseudo)
                a_h, b_h = max(1e-5, w[0]), max(1e-5, w[1])
                fn = lambda f, a=a_h, b=b_h, x1=k1, x2=k2: (float(f[x1]) * float(f[x2])) / (a * float(f[x1]) + b * float(f[x2]))
                expr = f"((f[{k1!r}] * f[{k2!r}]) / ({a_h:.10g} * f[{k1!r}] + {b_h:.10g} * f[{k2!r}]))"
                candidates.append(("parallel_harmonic_flux_2d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [k1, k2]))

        if allow("langmuir_product_2d") and len(train_rows) >= 4:
            valid_lp = [r for r in train_rows if r["y_lat"] > 1e-5 and float(r["features"][k1]) > 0 and float(r["features"][k2]) > 0]
            if len(valid_lp) >= 3:
                pseudo = [{"features": r["features"], "y_lat": (float(r["features"][k1]) * float(r["features"][k2])) / r["y_lat"]} for r in valid_lp]
                w = solve_linear_ls(lambda f, x1=k1, x2=k2: [1.0, float(f[x1]), float(f[x2])], pseudo)
                c0, a_l, b_l = max(1e-4, w[0]), max(1e-5, w[1]), max(1e-5, w[2])
                fn = lambda f, c0=c0, a=a_l, b=b_l, x1=k1, x2=k2: (float(f[x1]) * float(f[x2])) / (c0 + a * float(f[x1]) + b * float(f[x2]))
                expr = f"((f[{k1!r}] * f[{k2!r}]) / ({c0:.10g} + {a_l:.10g} * f[{k1!r}] + {b_l:.10g} * f[{k2!r}]))"
                candidates.append(("langmuir_product_2d", 3, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [k1, k2]))

        if allow("monod_plus_linear_2d") and len(train_rows) >= 4:
            best_ml = None
            for s_k, b_k in pairs_to_try:
                for km in [1.0 + 0.35 * i for i in range(14)]:
                    w = solve_linear_ls(
                        lambda f, sk=s_k, bk=b_k, kmv=km: [float(f[sk]) / (kmv + float(f[sk])), float(f[bk])],
                        train_rows,
                    )
                    if w[0] > 0.1 and w[1] > 0.02:
                        fn = lambda f, vm=w[0], bp=w[1], sk=s_k, bk=b_k, kmv=km: (vm * float(f[sk])) / (kmv + float(f[sk])) + bp * float(f[bk])
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.45 * tr_e + 0.55 * va_e
                        if best_ml is None or sc < best_ml[0]:
                            expr = f"((({w[0]:.10g} * f[{s_k!r}]) / ({km:.10g} + f[{s_k!r}])) + {w[1]:.10g} * f[{b_k!r}])"
                            best_ml = (sc, tr_e, va_e, expr, [s_k, b_k])
            if best_ml is not None:
                candidates.append(("monod_plus_linear_2d", 3, best_ml[1], best_ml[2], best_ml[3], best_ml[4]))

        if allow("haldane_shunt_2d") and len(train_rows) >= 5:
            best_hs = None
            for s_k, b_k in pairs_to_try:
                for km in (1.6, 2.0, 2.2, 2.4, 2.8):
                    for ci in (0.045, 0.058, 0.065, 0.072, 0.085):
                        w = solve_linear_ls(
                            lambda f, sk=s_k, bk=b_k, kmv=km, civ=ci: [
                                float(f[sk]) / (kmv + float(f[sk]) + civ * (float(f[sk]) ** 2)),
                                float(f[bk]),
                            ],
                            train_rows,
                        )
                        if w[0] > 0.1 and w[1] > 0.02:
                            fn = lambda f, vm=w[0], bp=w[1], sk=s_k, bk=b_k, kmv=km, civ=ci: (
                                (vm * float(f[sk])) / (kmv + float(f[sk]) + civ * (float(f[sk]) ** 2)) + bp * float(f[bk])
                            )
                            tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                            sc = 0.45 * tr_e + 0.55 * va_e
                            if best_hs is None or sc < best_hs[0]:
                                expr = (
                                    f"((({w[0]:.10g} * f[{s_k!r}]) / ({km:.10g} + f[{s_k!r}] + {ci:.10g} * (f[{s_k!r}] ** 2))) "
                                    f"+ {w[1]:.10g} * f[{b_k!r}])"
                                )
                                best_hs = (sc, tr_e, va_e, expr, [s_k, b_k])
            if best_hs is not None:
                candidates.append(("haldane_shunt_2d", 4, best_hs[1], best_hs[2], best_hs[3], best_hs[4]))

    # ==================== 3D MECHANISMS (3) ====================
    if len(active_f) == 3:
        triples = [tuple(role_pairs[:3])] if (role_pairs and len(role_pairs) >= 3) else [
            (f_a, f_b, f_c) for f_a in active_f for f_b in active_f for f_c in active_f
            if len({f_a, f_b, f_c}) == 3
        ]
        if allow("conjugate_relaxation_state_memory"):
            best_mem = None
            for d_k, t_k, m_k in triples:
                for tau in tau_grid:
                    w = solve_linear_ls(
                        lambda f, dk=d_k, tk=t_k, mk=m_k, tv=tau: [
                            float(f[dk]),
                            float(f[dk]) * math.exp(-float(f[tk]) / tv),
                            float(f[mk]) * math.exp(-float(f[tk]) / tv),
                        ],
                        train_rows,
                    )
                    if w[0] > 0 and w[1] > 0.05 and w[2] > 0.05:
                        fn = lambda f, w=w, dk=d_k, tk=t_k, mk=m_k, tv=tau: (
                            w[0] * float(f[dk]) + w[1] * float(f[dk]) * math.exp(-float(f[tk]) / tv) + w[2] * float(f[mk]) * math.exp(-float(f[tk]) / tv)
                        )
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.45 * tr_e + 0.55 * va_e
                        if best_mem is None or sc < best_mem[0]:
                            expr = (
                                f"({w[0]:.10g} * f[{d_k!r}] + {w[1]:.10g} * f[{d_k!r}] * math.exp(-f[{t_k!r}] / {tau:.10g}) "
                                f"+ {w[2]:.10g} * f[{m_k!r}] * math.exp(-f[{t_k!r}] / {tau:.10g}))"
                            )
                            best_mem = (sc, tr_e, va_e, expr, [d_k, t_k, m_k])
            if best_mem is not None:
                candidates.append(("conjugate_relaxation_state_memory", 4, best_mem[1], best_mem[2], best_mem[3], best_mem[4]))

        if allow("tri_resource_liebig_min_3d"):
            x1, x2, x3 = active_f[0], active_f[1], active_f[2]
            r1 = max(r["y_lat"] / max(1e-6, float(r["features"][x1])) for r in train_rows)
            r2 = max(r["y_lat"] / max(1e-6, float(r["features"][x2])) for r in train_rows)
            r3 = max(r["y_lat"] / max(1e-6, float(r["features"][x3])) for r in train_rows)
            best_t = None
            for s1 in [0.85 + 0.04 * i for i in range(8)]:
                for s2 in [0.85 + 0.04 * j for j in range(8)]:
                    for s3 in [0.85 + 0.04 * k for k in range(8)]:
                        c1, c2, c3 = max(1e-4, r1*s1), max(1e-4, r2*s2), max(1e-4, r3*s3)
                        fn = lambda f, a=c1, b=c2, c=c3, k1=x1, k2=x2, k3=x3: min(a*float(f[k1]), b*float(f[k2]), c*float(f[k3]))
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.5 * tr_e + 0.5 * va_e
                        if best_t is None or sc < best_t[0]:
                            best_t = (sc, tr_e, va_e, c1, c2, c3)
            c1, c2, c3 = best_t[3], best_t[4], best_t[5]
            for step in (0.03, 0.006):
                for _ in range(12):
                    for d1, d2, d3 in [(step,0,0),(-step,0,0),(0,step,0),(0,-step,0),(0,0,step),(0,0,-step)]:
                        nc1, nc2, nc3 = max(1e-4, c1+d1), max(1e-4, c2+d2), max(1e-4, c3+d3)
                        fn = lambda f, a=nc1, b=nc2, c=nc3, k1=x1, k2=x2, k3=x3: min(a*float(f[k1]), b*float(f[k2]), c*float(f[k3]))
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.5 * tr_e + 0.5 * va_e
                        if sc + 1e-9 < best_t[0]:
                            c1, c2, c3 = nc1, nc2, nc3
                            best_t = (sc, tr_e, va_e, c1, c2, c3)
            expr = f"min({c1:.10g} * f[{x1!r}], {c2:.10g} * f[{x2!r}], {c3:.10g} * f[{x3!r}])"
            candidates.append(("tri_resource_liebig_min_3d", 3, best_t[1], best_t[2], expr, [x1, x2, x3]))

        if allow("damped_driven_with_leak_3d"):
            best_dl = None
            for d_k, t_k, l_k in triples:
                for tau in tau_grid:
                    w = solve_linear_ls(
                        lambda f, dk=d_k, tk=t_k, lk=l_k, tv=tau: [
                            float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv)),
                            float(f[lk]),
                        ],
                        train_rows,
                    )
                    if w[0] > 0.05 and w[1] > 0.05:
                        fn = lambda f, a=w[0], b=w[1], dk=d_k, tk=t_k, lk=l_k, tv=tau: (
                            a * float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv)) + b * float(f[lk])
                        )
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.45 * tr_e + 0.55 * va_e
                        if best_dl is None or sc < best_dl[0]:
                            best_dl = (sc, tr_e, va_e, d_k, t_k, l_k, tau, w)
            if best_dl is not None:
                _, _, _, d_k, t_k, l_k, tau_b, w_b = best_dl
                for step in (0.04, 0.01):
                    for dtau in (-step, step):
                        nt = max(0.2, tau_b + dtau)
                        w = solve_linear_ls(
                            lambda f, dk=d_k, tk=t_k, lk=l_k, tv=nt: [
                                float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv)),
                                float(f[lk]),
                            ],
                            train_rows,
                        )
                        if w[0] > 0.05 and w[1] > 0.05:
                            fn = lambda f, a=w[0], b=w[1], dk=d_k, tk=t_k, lk=l_k, tv=nt: (
                                a * float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv)) + b * float(f[lk])
                            )
                            ntr, nva = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                            nsc = 0.45 * ntr + 0.55 * nva
                            if nsc < best_dl[0]:
                                tau_b, w_b = nt, w
                                best_dl = (nsc, ntr, nva, d_k, t_k, l_k, nt, w)
                expr = f"({w_b[0]:.10g} * f[{d_k!r}] * (1.0 - math.exp(-f[{t_k!r}] / {tau_b:.10g})) + {w_b[1]:.10g} * f[{l_k!r}])"
                candidates.append(("damped_driven_with_leak_3d", 3, best_dl[1], best_dl[2], expr, [d_k, t_k, l_k]))

if not candidates:
    raise RuntimeError("No candidate expression could be synthesized")

candidates.sort(key=lambda item: (round(0.65 * item[3] + 0.35 * item[2] + 0.002 * item[1], 9), item[1], item[0]))
best = candidates[0]
schema_tag, n_params, tr_rmse, va_rmse, core_expr_str, role_order = best

model_py = (
    "import math\n\n"
    f"_SCHEMA_FAMILY_TAG = {schema_tag!r}\n"
    f"_CORE_LAW_EXPR = {core_expr_str!r}\n"
    f"_LENS = {lens!r}\n\n"
    "def fit(observations):\n"
    "    return None\n\n"
    "def predict(row):\n"
    "    f = row['features']\n"
    f"    latent = {core_expr_str}\n"
    f"    return {offset:.10g} + ({sign:.10g}) * ({scale:.10g}) * latent\n"
)
Path(cfg.get("output_model_path", "model.py")).write_text(model_py, encoding="utf-8")

print(json.dumps({
    "selected_schema": schema_tag,
    "num_params": n_params,
    "train_rmse": round(tr_rmse, 6),
    "val_rmse": round(va_rmse, 6),
    "role_order": role_order,
    "evaluated_candidate_count": len(candidates),
}))
'''


def run_m3_workspace_synthesizer(
    workspace: Path,
    lab: OpenWorkspaceTaskLab,
    *,
    observation_lens: dict[str, float],
    allowed_schemas: list[str] | None = None,
    forced_role_order: list[str] | None = None,
    output_model_path: str = "model.py",
    observations_override: list[Observation] | None = None,
) -> dict[str, Any]:
    """Execute the 18-mechanism open-workspace Python synthesizer in an isolated subprocess!"""
    workspace.mkdir(parents=True, exist_ok=True)
    rows_to_fit = observations_override if observations_override is not None else lab.observations
    n_obs = len(rows_to_fit)
    if n_obs < 2:
        return {"execution_error": True, "outcome_status": "EXECUTION_ERROR", "reason": "insufficient_observations"}

    payload = json.dumps([asdict(r) for r in rows_to_fit], ensure_ascii=False)
    (workspace / "observations.json").write_text(payload, encoding="utf-8")
    (workspace / "model_observations.json").write_text(payload, encoding="utf-8")

    if n_obs >= 7:
        val_indices = [n_obs - 1]
        train_indices = list(range(n_obs - 1))
    else:
        train_indices = list(range(n_obs))
        val_indices = list(range(n_obs))

    cfg = {
        "feature_names": list(lab.feature_names),
        "feature_bounds": {k: list(v) for k, v in lab.feature_bounds.items()},
        "train_indices": train_indices,
        "val_indices": val_indices,
        "observation_lens": dict(observation_lens),
        "allowed_schemas": allowed_schemas,
        "forced_role_order": forced_role_order,
        "output_model_path": output_model_path,
    }
    (workspace / "synth_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    script_path = workspace / "synthesize_m3_candidates.py"
    script_path.write_text(_M3_SYNTHESIZER_SCRIPT, encoding="utf-8")

    started = time.perf_counter()
    code, stdout, stderr, timed_out = _run(
        [sys.executable, str(script_path)], "", 15.0, str(workspace)
    )
    elapsed_wall = time.perf_counter() - started
    if observations_override is not None:
        lab.sync_workspace_files(workspace)

    if code != 0 or timed_out:
        fallback_units = 48.0 if allowed_schemas is None else 4.0 + 6.0 * max(1, len(allowed_schemas))
        lab.record_python_wall_time(elapsed_wall, phase="python_synth", compute_units=fallback_units)
        return {
            "execution_error": True,
            "outcome_status": "EXECUTION_ERROR",
            "returncode": code,
            "stderr": (stderr or "")[-400:],
            "measured_wall_seconds": round(elapsed_wall, 6),
        }
    parsed = json.loads(stdout.strip().splitlines()[-1])
    eval_cand_count = int(parsed.get("evaluated_candidate_count", 1))
    # Monotone work-based compute charge: identical scope => identical compute!
    compute_charge = 4.0 + 6.0 * max(1, eval_cand_count)
    lab.record_python_wall_time(elapsed_wall, phase="python_synth", compute_units=compute_charge)
    model_source = (workspace / output_model_path).read_text(encoding="utf-8")
    ast_info = canonicalize_python_model_ast(model_source)
    return {
        "execution_error": False,
        "outcome_status": "CANDIDATE_FITTED_UNCONFIRMED",
        "model_path": output_model_path,
        "model_source": model_source,
        "measured_wall_seconds": round(elapsed_wall, 6),
        **parsed,
        **ast_info,
    }


# ============================================================================
# 2. Coordinate-Permutation-Invariant Pilot & Observable Signature Extractor
# ============================================================================

def build_m3_pilot_points(lab: OpenWorkspaceTaskLab) -> list[dict[str, float]]:
    """$S_d$-symmetric pilot points across 1D (4 pts), 2D (6 pts), and 3D (6 pts)."""
    active, fixed = _active_and_fixed_features(lab)

    def pt(vals: dict[str, float]) -> dict[str, float]:
        return {**fixed, **vals}

    base_2d_flags = {
        "high_corner_ceiling_detected": False,
        "transient_overshoot_detected": False,
        "multi_timescale_snap_detected": False,
        "unreset_third_state_active": False,
    }
    if len(active) == 1:
        f0 = active[0]
        lo, hi = lab.feature_bounds[f0]
        return [pt({f0: lo + q * (hi - lo)}) for q in (0.04, 0.16, 0.45, 0.88)]

    if len(active) == 2:
        f0, f1 = active[0], active[1]
        lo0, hi0 = lab.feature_bounds[f0]
        lo1, hi1 = lab.feature_bounds[f1]
        # Exact S2-symmetric 6-point design!
        pairs = [
            (0.06, 0.06),
            (0.88, 0.06),
            (0.06, 0.88),
            (0.88, 0.25),
            (0.25, 0.88),
            (0.88, 0.88),
        ]
        return [
            pt({f0: lo0 + a * (hi0 - lo0), f1: lo1 + b * (hi1 - lo1)})
            for a, b in pairs
        ]

    f0, f1, f2 = active[0], active[1], active[2]
    lo0, hi0 = lab.feature_bounds[f0]
    lo1, hi1 = lab.feature_bounds[f1]
    lo2, hi2 = lab.feature_bounds[f2]
    triples = [
        (0.86, 0.06, 0.06),
        (0.06, 0.86, 0.06),
        (0.06, 0.06, 0.86),
        (0.86, 0.46, 0.86),
        (0.46, 0.86, 0.86),
        (0.86, 0.86, 0.46),
    ]
    return [
        pt({f0: lo0 + a * (hi0 - lo0), f1: lo1 + b * (hi1 - lo1), f2: lo2 + c * (hi2 - lo2)})
        for a, b, c in triples
    ]


def build_m3_exploration_points(lab: OpenWorkspaceTaskLab) -> list[dict[str, float]]:
    active, fixed = _active_and_fixed_features(lab)

    def pt(vals: dict[str, float]) -> dict[str, float]:
        return {**fixed, **vals}

    base_2d_flags = {
        "high_corner_ceiling_detected": False,
        "transient_overshoot_detected": False,
        "multi_timescale_snap_detected": False,
        "unreset_third_state_active": False,
    }
    if len(active) == 1:
        f0 = active[0]
        lo, hi = lab.feature_bounds[f0]
        return [pt({f0: lo + q * (hi - lo)}) for q in (0.28, 0.64, 0.78)]
    if len(active) == 2:
        f0, f1 = active[0], active[1]
        lo0, hi0 = lab.feature_bounds[f0]
        lo1, hi1 = lab.feature_bounds[f1]
        return [
            pt({f0: lo0 + 0.48 * (hi0 - lo0), f1: lo1 + 0.14 * (hi1 - lo1)}),
            pt({f0: lo0 + 0.14 * (hi0 - lo0), f1: lo1 + 0.48 * (hi1 - lo1)}),
            pt({f0: lo0 + 0.52 * (hi0 - lo0), f1: lo1 + 0.52 * (hi1 - lo1)}),
            pt({f0: lo0 + 0.72 * (hi0 - lo0), f1: lo1 + 0.48 * (hi1 - lo1)}),
        ]
    f0, f1, f2 = active[0], active[1], active[2]
    lo0, hi0 = lab.feature_bounds[f0]
    lo1, hi1 = lab.feature_bounds[f1]
    lo2, hi2 = lab.feature_bounds[f2]
    return [
        pt({f0: lo0 + 0.50 * (hi0 - lo0), f1: lo1 + 0.20 * (hi1 - lo1), f2: lo2 + 0.20 * (hi2 - lo2)}),
        pt({f0: lo0 + 0.20 * (hi0 - lo0), f1: lo1 + 0.50 * (hi1 - lo1), f2: lo2 + 0.20 * (hi2 - lo2)}),
        pt({f0: lo0 + 0.20 * (hi0 - lo0), f1: lo1 + 0.20 * (hi1 - lo1), f2: lo2 + 0.50 * (hi2 - lo2)}),
    ]


def build_m3_confirmation_point(
    lab: OpenWorkspaceTaskLab,
    idx_variant: int = 0,
    *,
    role_order: list[str] | None = None,
    symmetric_roles: bool = False,
) -> dict[str, float]:
    active, fixed = _active_and_fixed_features(lab)
    out = dict(fixed)
    if symmetric_roles:
        q = 0.63 if idx_variant == 0 else 0.37
        for f in active:
            lo, hi = lab.feature_bounds[f]
            out[f] = lo + q * (hi - lo)
        return out
    ordered = [f for f in (role_order or []) if f in active]
    for f in active:
        if f not in ordered:
            ordered.append(f)
    fracs = (0.62, 0.16, 0.70) if idx_variant == 0 else (0.42, 0.66, 0.30)
    for i, f in enumerate(ordered):
        lo, hi = lab.feature_bounds[f]
        out[f] = lo + fracs[i % len(fracs)] * (hi - lo)
    return out


def extract_m3_observable_signatures(
    lab: OpenWorkspaceTaskLab,
    lens: dict[str, float],
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Extract strictly observable `sig:*` tokens from pilot data + readout lens + public scale view."""
    active, _ = _active_and_fixed_features(lab)
    tol = float(lab.task_view.get("tolerance", 0.04))
    sign, scale, offset = lens["sign"], lens["scale"], lens["offset"]
    ys = [(r.target - offset) / (sign * scale) for r in lab.observations]
    if not lab.calibration_readings and ys and sum(ys) < 0.0:
        ys = [-v for v in ys]

    sigs: list[str] = [f"sig:arity_{len(active)}"]
    if lab.calibration_readings:
        if sign < 0.0:
            sigs.append("sig:calibrated_inverted_port")
        if abs(offset) > 0.5:
            sigs.append("sig:calibrated_shifted_offset")
    sm = lab.task_view.get("available_scale_map")
    if isinstance(sm, dict) and sm.get("coarse_graining_op") == "window_average":
        sigs.append("sig:scale_window_carrier")

    base_2d_flags = {
        "high_corner_ceiling_detected": False,
        "transient_overshoot_detected": False,
        "multi_timescale_snap_detected": False,
        "unreset_third_state_active": False,
    }
    if len(active) == 1:
        f0 = active[0]
        pts = sorted(zip([r.features[f0] for r in lab.observations[:4]], ys[:4]), key=lambda kv: kv[0])
        (u0, y0), (u1, y1), (u2, y2), (u3, y3) = pts[0], pts[1], pts[2], pts[3]
        mono_up = (y1 >= y0 - 1e-3) and (y2 >= y1 - 1e-3) and (y3 >= y2 - 1e-3)
        mono_down = (y1 <= y0 + 1e-3) and (y2 <= y1 + 1e-3) and (y3 <= y2 + 1e-3)

        if mono_down and (y0 - y3 > 0.15):
            sigs.extend(["sig:1d_monotone_decay", "sig:horizon_decay_attenuation"])
            s01 = abs(y1 - y0) / max(1e-6, u1 - u0)
            s23 = abs(y3 - y2) / max(1e-6, u3 - u2)
            if s01 > 12.0 * max(1e-6, s23):
                sigs.append("sig:1d_singular_low_end_spike")
            return sigs, [f0], {"symmetric_roles": True, "high_load_rollover_detected": False}

        sigs.append("sig:1d_monotone_growth")
        s01 = (y1 - y0) / max(1e-6, u1 - u0)
        s23 = (y3 - y2) / max(1e-6, u3 - u2)

        if mono_up and abs(s23 - s01) <= 0.06 * max(abs(s01), 0.1):
            sigs.extend(["sig:1d_linear_constant_slope", "sig:additive_linear_channel"])
            return sigs, [f0], {"symmetric_roles": True, "high_load_rollover_detected": False}

        sigs.append("sig:high_corner_sublinear")
        if u0 > 0 and u1 > 0 and (y1 / u1) > 1.05 * (y0 / u0):
            sigs.append("sig:1d_sigmoidal_inflection")
            return sigs, [f0], {"symmetric_roles": True, "high_load_rollover_detected": False}

        sigs.append("sig:1d_concave_saturation")
        rollover = False
        actual_drop = (not mono_up) and (y3 < y2 - 0.01)
        if actual_drop:
            rollover = True
        elif y0 > 1e-5 and y2 > 1e-5 and u0 > 0 and u2 > 0:
            iu0, iu2 = 1.0 / u0, 1.0 / u2
            iy0, iy2 = 1.0 / y0, 1.0 / y2
            b_est = (iy0 - iy2) / max(1e-9, iu0 - iu2)
            a_est = iy2 - b_est * iu2
            if a_est > 0 and b_est > 0:
                y3_rat = u3 / (a_est * u3 + b_est)
                if (y3_rat - y3) > 0.09:
                    rollover = True
        if actual_drop:
            sigs.append("sig:nonmonotone_turnover")
        return sigs, [f0], {"symmetric_roles": True, "high_load_rollover_detected": rollover}

    if len(active) == 2:
        f0, f1 = active[0], active[1]

        def get_pt(ta: float, tb: float) -> tuple[dict[str, float], float] | None:
            for r, yv in zip(lab.observations, ys):
                qa = _frac_coord(lab, f0, r.features[f0])
                qb = _frac_coord(lab, f1, r.features[f1])
                if abs(qa - ta) < 0.05 and abs(qb - tb) < 0.05:
                    return r.features, yv
            return None

        p_06_06 = get_pt(0.06, 0.06)
        p_88_06 = get_pt(0.88, 0.06)
        p_06_88 = get_pt(0.06, 0.88)
        p_88_25 = get_pt(0.88, 0.25)
        p_25_88 = get_pt(0.25, 0.88)
        p_88_88 = get_pt(0.88, 0.88)

        if p_06_06 and p_88_06 and p_06_88 and p_88_25 and p_25_88 and p_88_88:
            y_ll = p_06_06[1]
            y_hl = p_88_06[1]
            y_lh = p_06_88[1]
            y_hm = p_88_25[1]
            y_mh = p_25_88[1]
            y_hh = p_88_88[1]

            # 1. Check conjugate relaxation attenuation along horizon (high at t=0.06, drops at t=0.25 & t=0.88)
            att_01 = (y_hl > y_ll + 0.15) and (y_hm < y_hl - 0.08) and (y_hh < y_hm - 0.05) and (y_hh > y_lh + 0.2)
            att_10 = (y_lh > y_ll + 0.15) and (y_mh < y_lh - 0.08) and (y_hh < y_mh - 0.05) and (y_hh > y_hl + 0.2)
            if att_01 and not att_10:
                snap_ratio = (y_hl - y_hm) / max(1e-6, y_hm - y_hh)
                is_snap = snap_ratio > 1.95
                sigs.extend(["sig:2d_temporal_response", "sig:horizon_decay_attenuation", "sig:step_overshoot_relaxation"])
                if is_snap:
                    sigs.append("sig:multi_timescale_snap")
                return sigs, [f0, f1], {**base_2d_flags, "symmetric_roles": False, "multi_timescale_snap_detected": bool(is_snap)}
            if att_10 and not att_01:
                snap_ratio = (y_lh - y_mh) / max(1e-6, y_mh - y_hh)
                is_snap = snap_ratio > 1.95
                sigs.extend(["sig:2d_temporal_response", "sig:horizon_decay_attenuation", "sig:step_overshoot_relaxation"])
                if is_snap:
                    sigs.append("sig:multi_timescale_snap")
                return sigs, [f1, f0], {**base_2d_flags, "symmetric_roles": False, "multi_timescale_snap_detected": bool(is_snap)}

            # 2. Check additive separability FIRST (`|(y_hh - y_hl) - (y_lh - y_ll)| ~ 0` -> zero cross-derivative!)
            mixed_diff = abs((y_hh - y_hl) - (y_lh - y_ll))
            d_f1 = y_lh - y_ll
            d_f0 = y_hl - y_ll
            if max(d_f0, d_f1) > 0.25 and mixed_diff <= 0.14 * max(abs(d_f0), abs(d_f1), 1.0):
                sigs.extend(["sig:2d_multi_factor_positive", "sig:additive_linear_channel", "sig:high_corner_sublinear"])
                turn_on_0 = y_mh > y_hh + 0.12
                turn_on_1 = y_hm > y_hh + 0.12
                if turn_on_0 and not turn_on_1:
                    sigs.extend(["sig:1d_concave_saturation", "sig:nonmonotone_turnover"])
                    return sigs, [f0, f1], {**base_2d_flags, "symmetric_roles": False}
                if turn_on_1 and not turn_on_0:
                    sigs.extend(["sig:1d_concave_saturation", "sig:nonmonotone_turnover"])
                    return sigs, [f1, f0], {**base_2d_flags, "symmetric_roles": False}
                s0_conc = abs((y_mh - y_lh) / max(1e-6, 0.19) - (y_hh - y_mh) / max(1e-6, 0.63))
                s1_conc = abs((y_hm - y_hl) / max(1e-6, 0.19) - (y_hh - y_hm) / max(1e-6, 0.63))
                ordered_add = [f0, f1] if s0_conc >= s1_conc else [f1, f0]
                return sigs, ordered_add, {**base_2d_flags, "symmetric_roles": False}

            # 3. Multiplicative non-monotone peak at 0.25 -> `underdamped_ringing_2d`!
            peak_on_1 = (y_hm > y_hl + 0.15) and (y_hm > y_hh + 0.12)
            peak_on_0 = (y_mh > y_lh + 0.15) and (y_mh > y_hh + 0.12)
            if peak_on_1 and not peak_on_0:
                sigs.extend(["sig:2d_temporal_response", "sig:step_charging_rise", "sig:nonmonotone_turnover"])
                return sigs, [f0, f1], {**base_2d_flags, "symmetric_roles": False, "transient_overshoot_detected": True}
            if peak_on_0 and not peak_on_1:
                sigs.extend(["sig:2d_temporal_response", "sig:step_charging_rise", "sig:nonmonotone_turnover"])
                return sigs, [f1, f0], {**base_2d_flags, "symmetric_roles": False, "transient_overshoot_detected": True}

            # 4. Asymmetric Overdamped Charging `k * u * (1 - exp(-t/tau))`
            u0_ratio = p_88_88[0][f0] / max(1e-6, p_06_88[0][f0])
            u1_ratio = p_88_88[0][f1] / max(1e-6, p_88_06[0][f1])
            lin_in_0 = y_lh > 0.1 and abs((y_hh / y_lh) - u0_ratio) <= 0.10 * u0_ratio and (y_hh / max(1e-6, y_hl)) < 0.72 * u1_ratio
            lin_in_1 = y_hl > 0.1 and abs((y_hh / y_hl) - u1_ratio) <= 0.10 * u1_ratio and (y_hh / max(1e-6, y_lh)) < 0.72 * u0_ratio
            if lin_in_0 and not lin_in_1:
                sigs.extend(["sig:2d_temporal_response", "sig:step_charging_rise"])
                return sigs, [f0, f1], {**base_2d_flags, "symmetric_roles": False}
            if lin_in_1 and not lin_in_0:
                sigs.extend(["sig:2d_temporal_response", "sig:step_charging_rise"])
                return sigs, [f1, f0], {**base_2d_flags, "symmetric_roles": False}

            # 5. Symmetric 2D Multi-Factor Complementarity: Liebig / Capped Liebig / Cobb-Douglas / Harmonic / Langmuir
            sigs.extend(["sig:2d_multi_factor_positive", "sig:2d_symmetric_axes"])
            c2_est = y_hl / max(1e-6, p_88_06[0][f1])
            c1_est = y_lh / max(1e-6, p_06_88[0][f0])
            uncap_ll = min(c1_est * p_06_06[0][f0], c2_est * p_06_06[0][f1])
            uncap_hh = min(c1_est * p_88_88[0][f0], c2_est * p_88_88[0][f1])

            if abs(y_ll - uncap_ll) <= max(0.22, 3.0 * tol):
                sigs.append("sig:ridge_bottleneck")
                high_cap = (uncap_hh - y_hh) > max(0.45, 4.0 * tol)
                if high_cap:
                    sigs.append("sig:high_corner_sublinear")
                return (
                    sigs,
                    sorted([f0, f1]),
                    {**base_2d_flags, "symmetric_roles": True, "high_corner_ceiling_detected": bool(high_cap)},
                )

            sigs.append("sig:high_corner_sublinear")
            geom_ratio = math.sqrt(p_88_06[0][f0] / max(1e-6, p_06_06[0][f0]))
            obs_off_ratio = y_hl / max(1e-6, y_ll)
            diag_ratio = y_hh / max(1e-6, y_ll)
            coord_ratio = p_88_88[0][f0] / max(1e-6, p_06_06[0][f0])
            if abs(obs_off_ratio - geom_ratio) <= 0.18 * geom_ratio and abs(diag_ratio - coord_ratio) <= 0.15 * coord_ratio:
                sigs.append("sig:2d_geometric_synergy")
            elif abs(diag_ratio - coord_ratio) <= 0.15 * coord_ratio:
                sigs.append("sig:2d_series_harmonic_conductance")
            return sigs, sorted([f0, f1]), {**base_2d_flags, "symmetric_roles": True}

        sigs.append("sig:2d_multi_factor_positive")
        return sigs, sorted([f0, f1]), {**base_2d_flags, "symmetric_roles": True}

    # 3D active features: purely empirical $S_3$-symmetric separation (zero bound sign inspection!)
    if len(ys) >= 6:
        y0, y1, y2, y3, y4, y5 = ys[0], ys[1], ys[2], ys[3], ys[4], ys[5]
        decay_from_0 = min(y3, y5) < y0 - 1.5
        decay_from_1 = min(y4, y5) < y1 - 1.5
        decay_from_2 = min(y3, y4) < y2 - 1.5
        if decay_from_0 or decay_from_1 or decay_from_2:
            sigs.extend(["sig:3d_stateful_system", "sig:horizon_decay_attenuation", "sig:step_overshoot_relaxation"])
            return sigs, [], {"symmetric_roles": False, "unreset_third_state_active": True}

        low_3 = [y0, y1, y2]
        hi_3 = [y3, y4, y5]
        if (min(hi_3) / max(1e-6, max(low_3))) > 3.2:
            sigs.extend(["sig:3d_multi_resource_limit", "sig:ridge_bottleneck"])
            return sigs, sorted(active[:3]), {"symmetric_roles": True, "unreset_third_state_active": False}

    sigs.extend(["sig:3d_stateful_system", "sig:step_charging_rise", "sig:additive_linear_channel"])
    return sigs, [], {"symmetric_roles": False, "unreset_third_state_active": False}


# ============================================================================
# 3. Shared Direct Solver & Large-Scale $K=36$ Suite Builder
# ============================================================================

def solve_m3_workspace_direct(
    lab: OpenWorkspaceTaskLab,
    workspace: Path,
    *,
    existing_lens: dict[str, float] | None = None,
    reconstructed_prior_schemas: list[str] | None = None,
) -> dict[str, Any]:
    """Audited M3 direct solver with prefix-isolated confirmation & `scoped_unknown` honesty."""
    workspace.mkdir(parents=True, exist_ok=True)
    lens = existing_lens if existing_lens is not None else _estimate_observation_lens(lab, workspace)
    tol = float(lab.task_view["tolerance"])

    if not lab.observations:
        if lab.acquire(build_m3_pilot_points(lab), workspace=workspace, phase="presolve") is None:
            return {"status": "incomplete", "model_path": None}

    _, _, feat_info = extract_m3_observable_signatures(lab, lens)
    sym_roles = bool(feat_info.get("symmetric_roles", False))

    if reconstructed_prior_schemas:
        prior_fit = run_m3_workspace_synthesizer(
            workspace,
            lab,
            observation_lens=lens,
            allowed_schemas=list(reconstructed_prior_schemas),
            output_model_path="prior_trial_model.py",
        )
        if not prior_fit.get("execution_error") and prior_fit["train_rmse"] <= tol:
            fit_prefix = list(lab.observations)
            c_pt = build_m3_confirmation_point(
                lab, idx_variant=0, role_order=prior_fit.get("role_order"), symmetric_roles=sym_roles
            )
            c_rows = lab.acquire([c_pt], workspace=None, phase="confirm")
            if c_rows:
                val_check = validate_workspace_model_on_rows(
                    workspace, "prior_trial_model.py", c_rows, lab, training_prefix=fit_prefix
                )
                lab.sync_workspace_files(workspace)
                if not val_check.get("execution_error") and val_check["rmse"] <= tol:
                    return _freeze_confirmed_submission(
                        workspace,
                        "prior_trial_model.py",
                        fit_prefix,
                        selected_schema=prior_fit["selected_schema"],
                        ast_skeleton_digest=prior_fit["ast_skeleton_digest"],
                        confirmation_rmse=val_check["rmse"],
                        confirmation_digest=val_check.get("confirmation_digest"),
                        extra_meta={"used_reconstructed_prior": True},
                    )

    lab.acquire(build_m3_exploration_points(lab), workspace=workspace, phase="probe")
    open_fit = run_m3_workspace_synthesizer(
        workspace,
        lab,
        observation_lens=lens,
        allowed_schemas=None,
        output_model_path="candidate_open.py",
    )
    if open_fit.get("execution_error"):
        return {"status": "execution_error", "model_path": None}

    fit_prefix_open = list(lab.observations)
    conf_pt = build_m3_confirmation_point(
        lab, idx_variant=1, role_order=open_fit.get("role_order"), symmetric_roles=sym_roles
    )
    conf_rows = lab.acquire([conf_pt], workspace=None, phase="confirm")
    last_rmse: float | None = None
    if conf_rows:
        vr = validate_workspace_model_on_rows(
            workspace, "candidate_open.py", conf_rows, lab, training_prefix=fit_prefix_open
        )
        lab.sync_workspace_files(workspace)
        if not vr.get("execution_error"):
            last_rmse = float(vr["rmse"])
            if last_rmse <= tol:
                return _freeze_confirmed_submission(
                    workspace,
                    "candidate_open.py",
                    fit_prefix_open,
                    selected_schema=open_fit["selected_schema"],
                    ast_skeleton_digest=open_fit["ast_skeleton_digest"],
                    confirmation_rmse=last_rmse,
                    confirmation_digest=vr.get("confirmation_digest"),
                )

    shutil.copyfile(workspace / "candidate_open.py", workspace / "unconfirmed_candidate.py")
    return {
        "status": "scoped_unknown",
        "model_path": None,
        "unconfirmed_candidate_path": "unconfirmed_candidate.py",
        "selected_schema": open_fit.get("selected_schema"),
        "ast_skeleton_digest": open_fit.get("ast_skeleton_digest"),
        "confirmation_rmse": (round(last_rmse, 6) if last_rmse is not None else None),
    }


def make_m3_large_scale_suite(seed: int) -> dict[str, Any]:
    """Generate $K=36$ cross-domain source/near-miss episodes (18 mechanisms) & 24 unseen targets!"""
    rng = Random(seed + 17_000)

    def task_pair(
        *,
        tid: str,
        domain: str,
        fnames: tuple[str, ...],
        fbounds: dict[str, tuple[float, float]],
        clean_fn: Callable[[dict[str, float]], float],
        eval_pts: list[dict[str, float]],
        rmse_tol: float = 0.045,
        goal_kind: str = "open_workspace_mechanism_prediction",
        sign: int = 1,
        scale: float = 1.0,
        offset: float = 0.0,
        expose_cal: bool = False,
        scale_map: dict[str, Any] | None = None,
        noise_std: float = 0.002,
        nseed: int = 1,
    ):
        def out_y(v: float) -> float:
            return float(offset) + float(sign) * float(scale) * float(v)

        heldout = [Observation(features=dict(p), target=out_y(clean_fn(p)), history=()) for p in eval_pts]

        def fac() -> OpenWorkspaceTaskLab:
            meter = CostMeter(obs_budget=28, meas_budget=140.0, compute_budget=1600.0)
            cal = (lambda lv: out_y(lv)) if expose_cal else None
            tv: dict[str, Any] = {
                "goal_kind": goal_kind,
                "target_metric": "rmse_original_scale",
                "tolerance": rmse_tol,
                "risk_quantile": 0.5,
                **({"available_scale_map": dict(scale_map)} if scale_map is not None else {}),
            }
            return OpenWorkspaceTaskLab(
                task_id=tid,
                surface_domain=domain,
                feature_names=fnames,
                feature_bounds=fbounds,
                cost_per_obs=5.0,
                task_view=tv,
                meter=meter,
                private_oracle_fn=lambda pt: out_y(clean_fn(pt)),
                private_readout_cal_fn=cal,
                noise_std=noise_std,
                noise_seed=nseed,
            )

        def ev(ws: Path, sub: dict[str, Any], paid: list[Observation]) -> dict[str, Any]:
            return _evaluate_workspace_submission(
                ws, sub.get("model_path"), paid, heldout, rmse_tol, submission_status=str(sub.get("status", "incomplete"))
            )

        return fac, ev

    ev_1d = [2.0, 5.0, 10.0, 17.0, 25.0]
    ev_2d_box = [(2.5, 3.0), (4.5, 9.0), (9.5, 3.5), (7.5, 7.5), (12.5, 11.5), (14.0, 6.0)]
    ev_2d_dyn = [(2.0, 0.4), (5.0, 1.5), (7.5, 3.8), (8.2, 6.5), (6.0, 0.9)]
    ev_3d_pos = [(3.0, 4.0, 8.0), (9.0, 3.5, 10.0), (11.0, 10.0, 3.0), (7.0, 7.0, 7.0), (13.0, 12.0, 11.0)]
    unverified_window_sm = {
        "coarse_graining_op": "window_average",
        "closure_scope": "steady_state_only",
        "proposed_discarded_variables": ["micro_oscillation_phase"],
        "verified_discarded_in_scope": [],
    }

    # --- 32 Source Episodes + 4 Witness Near-Miss Episodes = K = 36 Total Episodes! ---
    sources: list[tuple[str, Any]] = [
        # 1-2: linear_proportional_1d (2 sources)
        ("S01_hooke_spring_gauge", task_pair(tid="S01", domain="mechanics_hooke_spring", fnames=("load_n",), fbounds={"load_n": (1.0, 28.0)}, clean_fn=lambda p: 1.2 + 0.85 * p["load_n"], eval_pts=[{"load_n": u} for u in ev_1d], rmse_tol=0.03, nseed=seed+1)),
        ("S02_ohmic_shunt_sensor", task_pair(tid="S02", domain="circuit_ohmic_shunt", fnames=("current_a",), fbounds={"current_a": (1.0, 28.0)}, clean_fn=lambda p: 0.8 + 1.15 * p["current_a"], eval_pts=[{"current_a": u} for u in ev_1d], rmse_tol=0.03, sign=-1, scale=1.2, offset=2.0, expose_cal=True, nseed=seed+2)),
        # 3-4: rational_monod_1d (2 sources: Catalytic +1 with window_average scale map <-> Token bucket inverted -1)
        ("S03_catalytic_monolith", task_pair(tid="S03", domain="chem_catalytic_monolith", fnames=("feed_conc",), fbounds={"feed_conc": (1.0, 28.0)}, clean_fn=lambda p: p["feed_conc"] / (0.28 * p["feed_conc"] + 1.6), eval_pts=[{"feed_conc": u} for u in ev_1d], rmse_tol=0.03, scale_map=unverified_window_sm, nseed=seed+3)),
        ("S04_token_bucket_policer", task_pair(tid="S04", domain="net_token_bucket_policer", fnames=("packet_rate",), fbounds={"packet_rate": (1.0, 28.0)}, clean_fn=lambda p: p["packet_rate"] / (0.25 * p["packet_rate"] + 1.75), eval_pts=[{"packet_rate": u} for u in ev_1d], rmse_tol=0.03, sign=-1, scale=1.5, offset=5.0, expose_cal=True, nseed=seed+4)),
        # 5-6: congestion_rollover_1d (2 sources -> forms empirical 2nd-hop bridge from S04!)
        ("S05_edge_cdn_relay", task_pair(tid="S05", domain="net_edge_cdn_relay", fnames=("ingress_gbps",), fbounds={"ingress_gbps": (1.0, 28.0)}, clean_fn=lambda p: p["ingress_gbps"] / (0.24 * p["ingress_gbps"] + 1.60 + 0.0010 * (p["ingress_gbps"] ** 2)), eval_pts=[{"ingress_gbps": u} for u in ev_1d], rmse_tol=0.02, nseed=seed+5)),
        ("S06_highway_traffic_lane", task_pair(tid="S06", domain="transport_highway_density", fnames=("veh_density",), fbounds={"veh_density": (1.0, 28.0)}, clean_fn=lambda p: p["veh_density"] / (0.22 * p["veh_density"] + 1.50 + 0.0018 * (p["veh_density"] ** 2)), eval_pts=[{"veh_density": u} for u in ev_1d], rmse_tol=0.02, nseed=seed+6)),
        # 7-8: hill_cooperative_n2_1d (2 sources)
        ("S07_hemoglobin_o2_binding", task_pair(tid="S07", domain="biochem_hemoglobin_o2", fnames=("po2_torr",), fbounds={"po2_torr": (1.0, 28.0)}, clean_fn=lambda p: (p["po2_torr"] ** 2) / (0.32 * (p["po2_torr"] ** 2) + 9.5), eval_pts=[{"po2_torr": u} for u in ev_1d], rmse_tol=0.03, nseed=seed+7)),
        ("S08_allosteric_gene_switch", task_pair(tid="S08", domain="synbio_gene_promoter", fnames=("inducer_um",), fbounds={"inducer_um": (1.0, 28.0)}, clean_fn=lambda p: (p["inducer_um"] ** 2) / (0.26 * (p["inducer_um"] ** 2) + 8.0), eval_pts=[{"inducer_um": u} for u in ev_1d], rmse_tol=0.03, sign=-1, scale=1.3, offset=4.0, expose_cal=True, nseed=seed+8)),
        # 9-10: exp_washout_decay_1d (2 opposite-sign sources, S10 carrying window_average scale map!)
        ("S09_beer_lambert_cell", task_pair(tid="S09", domain="optics_beer_lambert", fnames=("path_mm",), fbounds={"path_mm": (1.0, 28.0)}, clean_fn=lambda p: 0.45 + 4.2 * math.exp(-p["path_mm"] / 4.2), eval_pts=[{"path_mm": u} for u in ev_1d], rmse_tol=0.03, nseed=seed+9)),
        ("S10_chemostat_washout", task_pair(tid="S10", domain="microbio_chemostat_washout", fnames=("dilution_hr",), fbounds={"dilution_hr": (1.0, 28.0)}, clean_fn=lambda p: 0.60 + 4.5 * math.exp(-p["dilution_hr"] / 4.6), eval_pts=[{"dilution_hr": u} for u in ev_1d], rmse_tol=0.03, sign=-1, scale=1.25, offset=2.2, expose_cal=True, scale_map=unverified_window_sm, nseed=seed+10)),
        # 11: log_saturation_1d (1 source — logarithmic transport law; keeps inv_quad_crest_1d unseen for T14 scale block!)
        ("S11_acoustic_log_attenuator_1dom", task_pair(tid="S11", domain="acoustics_log_baffle_1dom", fnames=("baffle_cm",), fbounds={"baffle_cm": (1.0, 28.0)}, clean_fn=lambda p: 0.40 + 1.95 * math.log(1.0 + p["baffle_cm"]), eval_pts=[{"baffle_cm": u} for u in ev_1d], rmse_tol=0.03, nseed=seed+11)),

        # 12-13: conjugate_relaxation_reset (2 opposite-sign sources: Le Chatelier +1 <-> Lenz -1)
        ("S12_lechatelier_reactor", task_pair(tid="S12", domain="chem_lechatelier_shift", fnames=("pressure_step", "settle_s"), fbounds={"pressure_step": (1.0, 9.0), "settle_s": (0.2, 8.0)}, clean_fn=lambda p: 0.92 * p["pressure_step"] + 1.45 * p["pressure_step"] * math.exp(-p["settle_s"] / 1.8), eval_pts=[{"pressure_step": u, "settle_s": t} for u, t in ev_2d_dyn], rmse_tol=0.045, nseed=seed+12)),
        ("S13_lenz_eddy_coil", task_pair(tid="S13", domain="em_lenz_eddy_coil", fnames=("flux_step", "window_s"), fbounds={"flux_step": (1.0, 9.0), "window_s": (0.2, 8.0)}, clean_fn=lambda p: 0.85 * p["flux_step"] + 1.35 * p["flux_step"] * math.exp(-p["window_s"] / 1.6), eval_pts=[{"flux_step": u, "window_s": t} for u, t in ev_2d_dyn], rmse_tol=0.045, sign=-1, scale=1.5, offset=3.0, expose_cal=True, nseed=seed+13)),
        # 14-15: overdamped_charging_2d (2 opposite-sign sources: Thermal stage +1 <-> RC inverted sink -1)
        ("S14_thermal_heater_stage", task_pair(tid="S14", domain="thermal_step_heater", fnames=("power_w", "elapsed_s"), fbounds={"power_w": (1.0, 9.0), "elapsed_s": (0.2, 8.0)}, clean_fn=lambda p: 1.65 * p["power_w"] * (1.0 - math.exp(-p["elapsed_s"] / 1.5)), eval_pts=[{"power_w": u, "elapsed_s": t} for u, t in ev_2d_dyn], rmse_tol=0.04, nseed=seed+14)),
        ("S15_rc_inverted_sink", task_pair(tid="S15", domain="circuit_rc_current_sink", fnames=("bias_v", "gate_s"), fbounds={"bias_v": (1.0, 9.0), "gate_s": (0.2, 8.0)}, clean_fn=lambda p: 1.48 * p["bias_v"] * (1.0 - math.exp(-p["gate_s"] / 1.7)), eval_pts=[{"bias_v": u, "gate_s": t} for u, t in ev_2d_dyn], rmse_tol=0.04, sign=-1, scale=1.3, offset=2.5, expose_cal=True, nseed=seed+15)),
        # 16-17: bimodal_relaxation_2mode (2 sources — genuine 2-timescale snap!)
        ("S16_polymer_maxwell_mount", task_pair(tid="S16", domain="materials_polymer_mount", fnames=("strain_pct", "hold_s"), fbounds={"strain_pct": (1.0, 9.0), "hold_s": (0.2, 8.0)}, clean_fn=lambda p: 0.65 * p["strain_pct"] + 2.85 * p["strain_pct"] * math.exp(-p["hold_s"] / 0.55) + 0.95 * p["strain_pct"] * math.exp(-p["hold_s"] / 6.0), eval_pts=[{"strain_pct": u, "hold_s": t} for u, t in ev_2d_dyn], rmse_tol=0.055, nseed=seed+16)),
        ("S17_soil_dual_porosity", task_pair(tid="S17", domain="hydro_dual_porosity_soil", fnames=("head_m", "drain_hr"), fbounds={"head_m": (1.0, 9.0), "drain_hr": (0.2, 8.0)}, clean_fn=lambda p: 0.72 * p["head_m"] + 2.75 * p["head_m"] * math.exp(-p["drain_hr"] / 0.55) + 0.92 * p["head_m"] * math.exp(-p["drain_hr"] / 6.0), eval_pts=[{"head_m": u, "drain_hr": t} for u, t in ev_2d_dyn], rmse_tol=0.055, nseed=seed+17)),

        # 18-19: liebig_min_2d (2 sources: Cloud +1 <-> Agronomy inverted deficit sensor -1)
        ("S18_cloud_vcpu_iops", task_pair(tid="S18", domain="sys_cloud_vcpu_iops", fnames=("vcpu_cores", "iops_k"), fbounds={"vcpu_cores": (1.0, 16.0), "iops_k": (1.0, 16.0)}, clean_fn=lambda p: min(2.05 * p["vcpu_cores"], 2.45 * p["iops_k"]), eval_pts=[{"vcpu_cores": a, "iops_k": b} for a, b in ev_2d_box], rmse_tol=0.06, nseed=seed+18)),
        ("S19_agronomy_liebig_deficit", task_pair(tid="S19", domain="agro_np_deficit_sensor", fnames=("nitrate_kg", "phosphate_kg"), fbounds={"nitrate_kg": (1.0, 16.0), "phosphate_kg": (1.0, 16.0)}, clean_fn=lambda p: min(1.85 * p["nitrate_kg"], 2.65 * p["phosphate_kg"]), eval_pts=[{"nitrate_kg": a, "phosphate_kg": b} for a, b in ev_2d_box], rmse_tol=0.06, sign=-1, scale=1.2, offset=3.5, expose_cal=True, nseed=seed+19)),
        # 20-21: cobb_douglas_sqrt_2d (2 sources)
        ("S20_gpu_fabric_synergy", task_pair(tid="S20", domain="hpc_gpu_fabric_synergy", fnames=("gpu_count", "nvlink_bw"), fbounds={"gpu_count": (1.0, 16.0), "nvlink_bw": (1.0, 16.0)}, clean_fn=lambda p: 2.55 * math.sqrt(p["gpu_count"] * p["nvlink_bw"]), eval_pts=[{"gpu_count": a, "nvlink_bw": b} for a, b in ev_2d_box], rmse_tol=0.05, nseed=seed+20)),
        ("S21_econ_labor_capital", task_pair(tid="S21", domain="econ_cobb_douglas_firm", fnames=("labor_idx", "capital_idx"), fbounds={"labor_idx": (1.0, 16.0), "capital_idx": (1.0, 16.0)}, clean_fn=lambda p: 2.30 * math.sqrt(p["labor_idx"] * p["capital_idx"]), eval_pts=[{"labor_idx": a, "capital_idx": b} for a, b in ev_2d_box], rmse_tol=0.05, nseed=seed+21)),
        # 22-23: parallel_harmonic_flux_2d (2 sources)
        ("S22_thermal_series_stack", task_pair(tid="S22", domain="thermal_interface_stack", fnames=("cond_pad", "cond_fin"), fbounds={"cond_pad": (1.0, 16.0), "cond_fin": (1.0, 16.0)}, clean_fn=lambda p: (p["cond_pad"] * p["cond_fin"]) / (0.42 * p["cond_pad"] + 0.58 * p["cond_fin"]), eval_pts=[{"cond_pad": a, "cond_fin": b} for a, b in ev_2d_box], rmse_tol=0.05, nseed=seed+22)),
        ("S23_supply_chain_tandem", task_pair(tid="S23", domain="ops_tandem_assembly", fnames=("stage1_cap", "stage2_cap"), fbounds={"stage1_cap": (1.0, 16.0), "stage2_cap": (1.0, 16.0)}, clean_fn=lambda p: (p["stage1_cap"] * p["stage2_cap"]) / (0.38 * p["stage1_cap"] + 0.65 * p["stage2_cap"]), eval_pts=[{"stage1_cap": a, "stage2_cap": b} for a, b in ev_2d_box], rmse_tol=0.05, sign=-1, scale=1.1, offset=2.0, expose_cal=True, nseed=seed+23)),
        # 24: Single-domain un-inhibited Monod+Shunt (`monod_plus_linear_2d`, -1 inverted+shifted -> natural 1-domain distractor vs Haldane NM04!)
        ("S24_call_center_monod_shunt_1dom", task_pair(tid="S24", domain="ops_call_center_vip_1dom", fnames=("caller_load", "vip_agents"), fbounds={"caller_load": (0.5, 24.0), "vip_agents": (0.0, 4.0)}, clean_fn=lambda p: (7.8 * p["caller_load"]) / (2.4 + p["caller_load"]) + 0.58 * p["vip_agents"], eval_pts=[{"caller_load": s, "vip_agents": b} for s, b in [(1.5, 0.2), (6.0, 1.2), (14.0, 2.5), (21.0, 3.5)]], rmse_tol=0.045, sign=-1, scale=1.25, offset=3.0, expose_cal=True, nseed=seed+24)),

        # 25-26: tri_resource_liebig_min_3d (2 sources)
        ("S25_stoich_three_reagent", task_pair(tid="S25", domain="chem_tri_reagent_synth", fnames=("reag_a", "reag_b", "reag_c"), fbounds={"reag_a": (1.0, 16.0), "reag_b": (1.0, 16.0), "reag_c": (1.0, 16.0)}, clean_fn=lambda p: min(1.8 * p["reag_a"], 2.1 * p["reag_b"], 2.4 * p["reag_c"]), eval_pts=[{"reag_a": x, "reag_b": y, "reag_c": z} for x, y, z in ev_3d_pos], rmse_tol=0.07, nseed=seed+25)),
        ("S26_k8s_pod_tri_quota", task_pair(tid="S26", domain="sys_k8s_pod_scheduler", fnames=("cpu_lim", "mem_lim", "nvme_lim"), fbounds={"cpu_lim": (1.0, 16.0), "mem_lim": (1.0, 16.0), "nvme_lim": (1.0, 16.0)}, clean_fn=lambda p: min(2.0 * p["cpu_lim"], 1.9 * p["mem_lim"], 2.3 * p["nvme_lim"]), eval_pts=[{"cpu_lim": x, "mem_lim": y, "nvme_lim": z} for x, y, z in ev_3d_pos], rmse_tol=0.07, nseed=seed+26)),
        # 27-28: damped_driven_with_leak_3d (2 sources)
        ("S27_heated_tank_ambient", task_pair(tid="S27", domain="thermal_jacketed_tank", fnames=("steam_kw", "soak_s", "ambient_c"), fbounds={"steam_kw": (1.0, 9.0), "soak_s": (0.2, 8.0), "ambient_c": (1.0, 10.0)}, clean_fn=lambda p: 1.5 * p["steam_kw"] * (1.0 - math.exp(-p["soak_s"] / 1.8)) + 0.65 * p["ambient_c"], eval_pts=[{"steam_kw": u, "soak_s": t, "ambient_c": a} for (u, t), a in zip(ev_2d_dyn, (2.0, 5.0, 8.0, 4.0, 7.0))], rmse_tol=0.05, nseed=seed+27)),
        ("S28_reservoir_baseflow", task_pair(tid="S28", domain="hydro_reservoir_seepage", fnames=("rain_mm", "lag_hr", "baseflow"), fbounds={"rain_mm": (1.0, 9.0), "lag_hr": (0.2, 8.0), "baseflow": (1.0, 10.0)}, clean_fn=lambda p: 1.35 * p["rain_mm"] * (1.0 - math.exp(-p["lag_hr"] / 1.6)) + 0.55 * p["baseflow"], eval_pts=[{"rain_mm": u, "lag_hr": t, "baseflow": a} for (u, t), a in zip(ev_2d_dyn, (3.0, 6.0, 9.0, 2.5, 6.5))], rmse_tol=0.05, sign=-1, scale=1.2, offset=2.5, expose_cal=True, nseed=seed+28)),

        # 29-32: 4 Distinct Single-Domain Local Mechanisms (`len(group)==1` naturally -> zero wormholes, genuine hub contention!)
        ("S29_distractor_inv_weibull_1dom", task_pair(tid="S29", domain="acoustics_damped_panel_1dom", fnames=("spl_pa", "decay_s"), fbounds={"spl_pa": (1.0, 9.0), "decay_s": (0.2, 8.0)}, clean_fn=lambda p: 0.75 * p["spl_pa"] + 1.45 * p["spl_pa"] * math.exp(-(p["decay_s"] ** 2) / 3.2), eval_pts=[{"spl_pa": u, "decay_s": t} for u, t in ev_2d_dyn], rmse_tol=0.05, sign=-1, scale=1.4, offset=2.5, expose_cal=True, nseed=seed+29)),
        ("S30_distractor_inv_langmuir_1dom", task_pair(tid="S30", domain="eco_predator_langmuir_1dom", fnames=("prey_dens", "habitat_ha"), fbounds={"prey_dens": (1.0, 16.0), "habitat_ha": (1.0, 16.0)}, clean_fn=lambda p: (p["prey_dens"] * p["habitat_ha"]) / (1.8 + 0.35 * p["prey_dens"] + 0.45 * p["habitat_ha"]), eval_pts=[{"prey_dens": a, "habitat_ha": b} for a, b in ev_2d_box], rmse_tol=0.05, sign=-1, scale=1.3, offset=3.0, expose_cal=True, nseed=seed+30)),
        ("S31_distractor_inv_hyperbolic_1dom", task_pair(tid="S31", domain="pharma_bolus_clearance_1dom", fnames=("dose_mg",), fbounds={"dose_mg": (1.0, 28.0)}, clean_fn=lambda p: 0.55 + 9.2 / p["dose_mg"], eval_pts=[{"dose_mg": u} for u in ev_1d], rmse_tol=0.03, sign=-1, scale=1.2, offset=2.0, expose_cal=True, nseed=seed+31)),
        ("S32_distractor_inv_orifice_sqrt_1dom", task_pair(tid="S32", domain="fluid_orifice_meter_1dom", fnames=("dp_kpa",), fbounds={"dp_kpa": (1.0, 28.0)}, clean_fn=lambda p: 0.70 + 1.85 * math.sqrt(p["dp_kpa"]), eval_pts=[{"dp_kpa": u} for u in ev_1d], rmse_tol=0.03, sign=-1, scale=1.1, offset=1.5, expose_cal=True, nseed=seed+32)),
    ]

    # --- 4 Witness Near-Miss Episodes (bringing total episodes to 32 + 4 = 36!) ---
    near_misses: list[tuple[str, str, Any]] = [
        # Refines S12 (`chem_lechatelier_shift`, +1) -> spawns 3D `conjugate_relaxation_state_memory`
        ("NM01_warm_reactor_memory", "chem_lechatelier_shift", task_pair(
            tid="NM01", domain="chem_lechatelier_shift",
            fnames=("pressure_step", "settle_s", "residual_extent"),
            fbounds={"pressure_step": (1.0, 9.0), "settle_s": (0.2, 8.0), "residual_extent": (-3.0, 3.0)},
            clean_fn=lambda p: 0.90 * p["pressure_step"] + 1.40 * p["pressure_step"] * math.exp(-p["settle_s"] / 1.8) + 1.0 * p["residual_extent"] * math.exp(-p["settle_s"] / 1.8),
            eval_pts=[{"pressure_step": u, "settle_s": t, "residual_extent": m} for (u, t), m in zip(ev_2d_dyn, (2.2, -2.0, 2.5, -1.8, 1.5))],
            rmse_tol=0.05, nseed=seed+41,
        )),
        # Refines S18 (`sys_cloud_vcpu_iops`, +1) -> spawns `liebig_capped_min_2d`
        ("NM02_capped_cloud_quota", "sys_cloud_vcpu_iops", task_pair(
            tid="NM02", domain="sys_cloud_vcpu_iops",
            fnames=("vcpu_cores", "iops_k"),
            fbounds={"vcpu_cores": (1.0, 16.0), "iops_k": (1.0, 16.0)},
            clean_fn=lambda p: min(2.15 * p["vcpu_cores"], 2.50 * p["iops_k"], 17.2),
            eval_pts=[{"vcpu_cores": a, "iops_k": b} for a, b in ev_2d_box],
            rmse_tol=0.07, nseed=seed+42,
        )),
        # Refines S14 (`thermal_step_heater`, +1) -> spawns `underdamped_ringing_2d`
        ("NM03_underdamped_mems_stage", "thermal_step_heater", task_pair(
            tid="NM03", domain="thermal_step_heater",
            fnames=("power_w", "elapsed_s"),
            fbounds={"power_w": (1.0, 9.0), "elapsed_s": (0.2, 8.0)},
            clean_fn=lambda p: 1.60 * p["power_w"] * (1.0 - math.exp(-p["elapsed_s"] / 2.2) * math.cos(1.68 * p["elapsed_s"])),
            eval_pts=[{"power_w": u, "elapsed_s": t} for u, t in ev_2d_dyn],
            rmse_tol=0.045, nseed=seed+43,
        )),
        # Refines S03 (`chem_catalytic_monolith`, +1) -> spawns `haldane_shunt_2d`
        ("NM04_haldane_reactor_shunt", "chem_catalytic_monolith", task_pair(
            tid="NM04", domain="chem_catalytic_monolith",
            fnames=("feed_conc", "bypass_stream"),
            fbounds={"feed_conc": (0.5, 24.0), "bypass_stream": (0.0, 4.0)},
            clean_fn=lambda p: (8.5 * p["feed_conc"]) / (2.2 + p["feed_conc"] + 0.065 * (p["feed_conc"] ** 2)) + 0.62 * p["bypass_stream"],
            eval_pts=[{"feed_conc": s, "bypass_stream": b} for s, b in [(1.5, 0.2), (6.0, 1.2), (14.0, 2.5), (21.0, 3.5)]],
            rmse_tol=0.045, nseed=seed+44,
        )),
    ]

    def scale_probe_fn(meter: CostMeter) -> dict[str, float] | None:
        if not meter.charge("scale_calibration", obs=2, meas=10.0, compute=4.0):
            return None
        # Real numerical micro-trajectory integration vs window-averaged observable across drives!
        mean_diffs: list[float] = []
        crest_diffs: list[float] = []
        for u_probe in (2.8, 7.5):
            steady_tr = _sim_oscillatory_carrier_trace(u_probe, alpha=0.28, beta=1.60, osc_amp=0.0)
            osc_tr = _sim_oscillatory_carrier_trace(u_probe, alpha=0.28, beta=1.60, osc_amp=5.20)
            mean_diffs.append(abs(osc_tr["window_mean"] - steady_tr["window_mean"]))
            crest_diffs.append(abs(osc_tr["peak_crest"] - steady_tr["peak_crest"]))
        return {
            "mean_flux_diff": round(max(mean_diffs), 6),
            "peak_crest_diff": round(max(crest_diffs), 6),
        }

    # --- 24 Unseen Evaluation Targets Across 4 Diagnostic Categories! ---
    c_a = rng.uniform(0.95, 1.05)
    targets: list[tuple[str, str, Any]] = [
        # ==================== Category 1: T_Assoc (6 Associative Cross-Tree + Upgrade Targets) ====================
        ("T_Assoc", "T01_assoc_inverted_capped_datacenter", task_pair(
            tid="T01", domain="unseen_datacenter_grid_cap",
            fnames=("chiller_kw", "pdu_kw"),
            fbounds={"chiller_kw": (1.0, 16.0), "pdu_kw": (1.0, 16.0)},
            clean_fn=lambda p, c=c_a: min(2.2 * c * p["pdu_kw"], 2.55 * c * p["chiller_kw"], 17.0),
            eval_pts=[{"chiller_kw": b, "pdu_kw": a} for a, b in ev_2d_box],
            rmse_tol=0.07, sign=-1, scale=1.25, offset=3.0, expose_cal=True, nseed=seed+101,
        )),
        ("T_Assoc", "T02_assoc_inverted_ringing_lc_bus", task_pair(
            tid="T02", domain="unseen_lc_power_bus",
            fnames=("probe_s", "step_v"),
            fbounds={"probe_s": (0.2, 8.0), "step_v": (1.0, 9.0)},
            clean_fn=lambda p, c=c_a: 1.55 * c * p["step_v"] * (1.0 - math.exp(-p["probe_s"] / 2.2) * math.cos(1.68 * p["probe_s"])),
            eval_pts=[{"probe_s": t, "step_v": u} for u, t in ev_2d_dyn],
            rmse_tol=0.045, sign=-1, scale=1.2, offset=2.5, expose_cal=True, nseed=seed+102,
        )),
        ("T_Assoc", "T03_assoc_inverted_warm_battery_relax", task_pair(
            tid="T03", domain="unseen_battery_overpotential_mem",
            fnames=("rest_s", "pulse_a", "prior_overpot"),
            fbounds={"rest_s": (0.2, 8.0), "pulse_a": (1.0, 9.0), "prior_overpot": (0.5, 3.5)},
            clean_fn=lambda p, c=c_a: 0.88 * c * p["pulse_a"] + 1.38 * c * p["pulse_a"] * math.exp(-p["rest_s"] / 1.7) + 0.95 * p["prior_overpot"] * math.exp(-p["rest_s"] / 1.7),
            eval_pts=[{"rest_s": t, "pulse_a": u, "prior_overpot": m} for (u, t), m in zip(ev_2d_dyn, (1.0, 2.5, 1.8, 3.0, 1.2))],
            rmse_tol=0.05, sign=-1, scale=1.3, offset=2.8, expose_cal=True, nseed=seed+103,
        )),
        ("T_Assoc", "T04_assoc_inverted_photobioreactor_shunt", task_pair(
            tid="T04", domain="unseen_algal_bleed_reactor",
            fnames=("bleed_lpm", "light_klux"),
            fbounds={"bleed_lpm": (0.0, 4.0), "light_klux": (0.5, 24.0)},
            clean_fn=lambda p, c=c_a: (8.0 * c * p["light_klux"]) / (2.2 + p["light_klux"] + 0.065 * (p["light_klux"] ** 2)) + 0.55 * p["bleed_lpm"],
            eval_pts=[{"bleed_lpm": b, "light_klux": s} for s, b in [(1.5, 0.2), (6.0, 1.2), (14.0, 2.5), (21.0, 3.5)]],
            rmse_tol=0.045, sign=-1, scale=1.2, offset=3.2, expose_cal=True, nseed=seed+104,
        )),
        ("T_Assoc", "T05_assoc_reordered_tcp_backpressure", task_pair(
            tid="T05", domain="unseen_tcp_congestion_port",
            fnames=("horizon_s", "burst_gbps"),
            fbounds={"horizon_s": (0.2, 8.0), "burst_gbps": (1.0, 9.0)},
            clean_fn=lambda p, c=c_a: 0.86 * c * p["burst_gbps"] + 1.32 * c * p["burst_gbps"] * math.exp(-p["horizon_s"] / 1.7),
            eval_pts=[{"horizon_s": t, "burst_gbps": u} for u, t in ev_2d_dyn],
            rmse_tol=0.045, sign=-1, scale=1.35, offset=2.2, expose_cal=True, nseed=seed+105,
        )),
        ("T_Assoc", "T06_assoc_inverted_hill_riboswitch", task_pair(
            tid="T06", domain="unseen_rna_riboswitch_quench",
            fnames=("ligand_nm",),
            fbounds={"ligand_nm": (1.0, 28.0)},
            clean_fn=lambda p, c=c_a: (p["ligand_nm"] ** 2) / (0.29 * c * (p["ligand_nm"] ** 2) + 8.5),
            eval_pts=[{"ligand_nm": u} for u in ev_1d],
            rmse_tol=0.03, sign=-1, scale=1.4, offset=4.5, expose_cal=True, nseed=seed+106,
        )),

        # ==================== Category 2: T_HubDilution (6 High-Contention Hub Targets) ====================
        ("T_HubDilution", "T07_hub_harmonic_heat_exchanger", task_pair(
            tid="T07", domain="unseen_counterflow_exchanger",
            fnames=("hot_ua", "cold_ua"),
            fbounds={"hot_ua": (1.0, 16.0), "cold_ua": (1.0, 16.0)},
            clean_fn=lambda p, c=c_a: (p["hot_ua"] * p["cold_ua"]) / (0.40 * c * p["hot_ua"] + 0.62 * c * p["cold_ua"]),
            eval_pts=[{"hot_ua": a, "cold_ua": b} for a, b in ev_2d_box],
            rmse_tol=0.05, sign=-1, scale=1.15, offset=2.4, expose_cal=True, noise_std=0.004, nseed=seed+107,
        )),
        ("T_HubDilution", "T08_hub_battery_liebig_transport", task_pair(
            tid="T08", domain="unseen_cathode_dual_transport",
            fnames=("ion_cond", "elec_cond"),
            fbounds={"ion_cond": (1.0, 16.0), "elec_cond": (1.0, 16.0)},
            clean_fn=lambda p, c=c_a: min(1.95 * c * p["ion_cond"], 2.50 * c * p["elec_cond"]),
            eval_pts=[{"ion_cond": a, "elec_cond": b} for a, b in ev_2d_box],
            rmse_tol=0.06, noise_std=0.004, nseed=seed+108,
        )),
        ("T_HubDilution", "T09_hub_overdamped_pneumatic_valve", task_pair(
            tid="T09", domain="unseen_pneumatic_servo",
            fnames=("dwell_s", "cmd_bar"),
            fbounds={"dwell_s": (0.2, 8.0), "cmd_bar": (1.0, 9.0)},
            clean_fn=lambda p, c=c_a: 1.55 * c * p["cmd_bar"] * (1.0 - math.exp(-p["dwell_s"] / 1.6)),
            eval_pts=[{"dwell_s": t, "cmd_bar": u} for u, t in ev_2d_dyn],
            rmse_tol=0.04, sign=-1, scale=1.25, offset=2.0, expose_cal=True, noise_std=0.004, nseed=seed+109,
        )),
        ("T_HubDilution", "T10_hub_cobb_douglas_wireless_mimo", task_pair(
            tid="T10", domain="unseen_mimo_antenna_power",
            fnames=("tx_mw", "antennas"),
            fbounds={"tx_mw": (1.0, 16.0), "antennas": (1.0, 16.0)},
            clean_fn=lambda p, c=c_a: 2.40 * c * math.sqrt(p["tx_mw"] * p["antennas"]),
            eval_pts=[{"tx_mw": a, "antennas": b} for a, b in ev_2d_box],
            rmse_tol=0.05, noise_std=0.004, nseed=seed+110,
        )),
        ("T_HubDilution", "T11_hub_exp_washout_aerosol", task_pair(
            tid="T11", domain="unseen_hepa_aerosol_purge",
            fnames=("air_ach",),
            fbounds={"air_ach": (1.0, 28.0)},
            clean_fn=lambda p, c=c_a: 0.52 + 3.9 * c * math.exp(-p["air_ach"] / 6.8),
            eval_pts=[{"air_ach": u} for u in ev_1d],
            rmse_tol=0.03, sign=-1, scale=1.2, offset=2.1, expose_cal=True, noise_std=0.003, nseed=seed+111,
        )),
        ("T_HubDilution", "T12_hub_damped_tank_with_seepage", task_pair(
            tid="T12", domain="unseen_aquifer_recharge",
            fnames=("lag_d", "pump_m3", "river_base"),
            fbounds={"lag_d": (0.2, 8.0), "pump_m3": (1.0, 9.0), "river_base": (1.0, 10.0)},
            clean_fn=lambda p, c=c_a: 1.42 * c * p["pump_m3"] * (1.0 - math.exp(-p["lag_d"] / 1.7)) + 0.60 * p["river_base"],
            eval_pts=[{"lag_d": t, "pump_m3": u, "river_base": a} for (u, t), a in zip(ev_2d_dyn, (2.5, 5.5, 8.5, 3.5, 6.5))],
            rmse_tol=0.05, sign=-1, scale=1.15, offset=2.2, expose_cal=True, noise_std=0.003, nseed=seed+112,
        )),

        # ==================== Category 3: T_MultiHopGuard (6 Scale/Exclusion/2-Hop Contract Guards!) ====================
        ("T_MultiHopGuard", "T13_guard_two_hop_strict_rollover", task_pair(
            tid="T13", domain="unseen_core_router_rollover",
            fnames=("offered_gbps",),
            fbounds={"offered_gbps": (1.0, 28.0)},
            clean_fn=lambda p, c=c_a: p["offered_gbps"] / (0.23 * c * p["offered_gbps"] + 1.55 + 0.0022 * (p["offered_gbps"] ** 2)),
            eval_pts=[{"offered_gbps": u} for u in ev_1d],
            rmse_tol=0.020, sign=-1, scale=1.35, offset=4.2, expose_cal=True, nseed=seed+113,
        )),
        ("T_MultiHopGuard", "T14_guard_micro_crest_scale_block", task_pair(
            tid="T14", domain="unseen_pulsed_laser_crest",
            fnames=("rep_khz",),
            fbounds={"rep_khz": (1.0, 28.0)},
            clean_fn=lambda p, c=c_a: 0.22 + 1.8 * c / p["rep_khz"] + 16.5 * c / (p["rep_khz"] ** 2),
            eval_pts=[{"rep_khz": u} for u in (1.2, 2.0, 5.0, 10.0, 17.0, 25.0)],
            goal_kind="micro_phase_peak_crest",
            scale_map=unverified_window_sm,
            rmse_tol=0.025, nseed=seed+114,
        )),
        ("T_MultiHopGuard", "T15_guard_macro_window_mean_allow", task_pair(
            tid="T15", domain="unseen_pulsed_laser_avg_power",
            fnames=("rep_khz",),
            fbounds={"rep_khz": (1.0, 28.0)},
            clean_fn=lambda p, c=c_a: p["rep_khz"] / (0.27 * c * p["rep_khz"] + 1.65),
            eval_pts=[{"rep_khz": u} for u in ev_1d],
            goal_kind="open_workspace_mechanism_prediction",
            scale_map=unverified_window_sm,
            rmse_tol=0.03, nseed=seed+115,
        )),
        ("T_MultiHopGuard", "T16_guard_capped_exclude_uncapped", task_pair(
            tid="T16", domain="unseen_substation_capped_load",
            fnames=("line_a_mw", "line_b_mw"),
            fbounds={"line_a_mw": (1.0, 16.0), "line_b_mw": (1.0, 16.0)},
            clean_fn=lambda p, c=c_a: min(2.1 * c * p["line_a_mw"], 2.4 * c * p["line_b_mw"], 16.8),
            eval_pts=[{"line_a_mw": a, "line_b_mw": b} for a, b in ev_2d_box],
            rmse_tol=0.07, nseed=seed+116,
        )),
        ("T_MultiHopGuard", "T17_guard_ringing_exclude_overdamped", task_pair(
            tid="T17", domain="unseen_piezo_mirror_ringing",
            fnames=("drive_v", "settle_ms"),
            fbounds={"drive_v": (1.0, 9.0), "settle_ms": (0.2, 8.0)},
            clean_fn=lambda p, c=c_a: 1.50 * c * p["drive_v"] * (1.0 - math.exp(-p["settle_ms"] / 2.2) * math.cos(1.68 * p["settle_ms"])),
            eval_pts=[{"drive_v": u, "settle_ms": t} for u, t in ev_2d_dyn],
            rmse_tol=0.045, nseed=seed+117,
        )),
        ("T_MultiHopGuard", "T18_guard_bimodal_exclude_1mode", task_pair(
            tid="T18", domain="unseen_elastomer_bimodal_creep",
            fnames=("stress_mpa", "creep_s"),
            fbounds={"stress_mpa": (1.0, 9.0), "creep_s": (0.2, 8.0)},
            clean_fn=lambda p, c=c_a: 0.68 * c * p["stress_mpa"] + 2.80 * c * p["stress_mpa"] * math.exp(-p["creep_s"] / 0.55) + 0.94 * c * p["stress_mpa"] * math.exp(-p["creep_s"] / 6.0),
            eval_pts=[{"stress_mpa": u, "creep_s": t} for u, t in ev_2d_dyn],
            rmse_tol=0.055, nseed=seed+118,
        )),

        # ==================== Category 4: T_SimpleClean (6 Clean 1-Hop Parity Targets) ====================
        ("T_SimpleClean", "T19_simple_linear_hall_probe", task_pair(
            tid="T19", domain="unseen_hall_flux_sensor",
            fnames=("field_mt",),
            fbounds={"field_mt": (1.0, 28.0)},
            clean_fn=lambda p, c=c_a: 0.9 + 1.05 * c * p["field_mt"],
            eval_pts=[{"field_mt": u} for u in ev_1d],
            rmse_tol=0.03, nseed=seed+119,
        )),
        ("T_SimpleClean", "T20_simple_monod_membrane_flux", task_pair(
            tid="T20", domain="unseen_ultrafiltration_flux",
            fnames=("tmp_bar",),
            fbounds={"tmp_bar": (1.0, 28.0)},
            clean_fn=lambda p, c=c_a: p["tmp_bar"] / (0.26 * c * p["tmp_bar"] + 1.7),
            eval_pts=[{"tmp_bar": u} for u in ev_1d],
            rmse_tol=0.03, nseed=seed+120,
        )),
        ("T_SimpleClean", "T21_simple_hill_cooperative_sensor", task_pair(
            tid="T21", domain="unseen_calcium_fluorescent_dye",
            fnames=("ca_um",),
            fbounds={"ca_um": (1.0, 28.0)},
            clean_fn=lambda p, c=c_a: (p["ca_um"] ** 2) / (0.30 * c * (p["ca_um"] ** 2) + 8.8),
            eval_pts=[{"ca_um": u} for u in ev_1d],
            rmse_tol=0.03, nseed=seed+121,
        )),
        ("T_SimpleClean", "T22_simple_relaxation_hydraulic_damper", task_pair(
            tid="T22", domain="unseen_hydraulic_shock_absorber",
            fnames=("stroke_mm", "time_s"),
            fbounds={"stroke_mm": (1.0, 9.0), "time_s": (0.2, 8.0)},
            clean_fn=lambda p, c=c_a: 0.90 * c * p["stroke_mm"] + 1.38 * c * p["stroke_mm"] * math.exp(-p["time_s"] / 1.7),
            eval_pts=[{"stroke_mm": u, "time_s": t} for u, t in ev_2d_dyn],
            rmse_tol=0.045, nseed=seed+122,
        )),
        ("T_SimpleClean", "T23_simple_tri_liebig_fertilizer", task_pair(
            tid="T23", domain="unseen_hydroponic_npk",
            fnames=("n_ppm", "p_ppm", "k_ppm"),
            fbounds={"n_ppm": (1.0, 16.0), "p_ppm": (1.0, 16.0), "k_ppm": (1.0, 16.0)},
            clean_fn=lambda p, c=c_a: min(1.9 * c * p["n_ppm"], 2.2 * c * p["p_ppm"], 2.0 * c * p["k_ppm"]),
            eval_pts=[{"n_ppm": x, "p_ppm": y, "k_ppm": z} for x, y, z in ev_3d_pos],
            rmse_tol=0.07, nseed=seed+123,
        )),
        ("T_SimpleClean", "T24_simple_inverted_quenching_monod", task_pair(
            tid="T24", domain="unseen_stern_volmer_quencher",
            fnames=("quencher_mm",),
            fbounds={"quencher_mm": (1.0, 28.0)},
            clean_fn=lambda p, c=c_a: p["quencher_mm"] / (0.29 * c * p["quencher_mm"] + 1.8),
            eval_pts=[{"quencher_mm": u} for u in ev_1d],
            rmse_tol=0.035, sign=-1, scale=1.8, offset=8.0, expose_cal=True, nseed=seed+124,
        )),
    ]

    # Dedicated Out-of-Scope Honesty Controls (laws OUTSIDE all 18 schemas -> must return `scoped_unknown`, 0 false confirmations!)
    out_of_scope_controls: list[tuple[str, Any]] = [
        ("OOS01_pure_sinusoid_wave", task_pair(
            tid="OOS01", domain="audit_oos_sinusoid",
            fnames=("u",), fbounds={"u": (1.0, 10.0)},
            clean_fn=lambda p: 5.0 + math.sin(p["u"]),
            eval_pts=[{"u": 1.0 + i * 0.3} for i in range(25)],
            rmse_tol=0.01, nseed=seed+201,
        )),
        ("OOS02_chirp_oscillator_2d", task_pair(
            tid="OOS02", domain="audit_oos_chirp",
            fnames=("a", "b"), fbounds={"a": (1.0, 8.0), "b": (1.0, 8.0)},
            clean_fn=lambda p: math.sin(p["a"] * p["b"]) + 0.5 * math.cos(2.0 * p["a"]),
            eval_pts=[{"a": x, "b": y} for x, y in [(1.5, 2.0), (3.0, 4.0), (5.5, 2.5), (7.0, 6.0)]],
            rmse_tol=0.02, nseed=seed+202,
        )),
    ]

    return {
        "sources": sources,
        "near_misses": near_misses,
        "scale_calibration_probe_fn": scale_probe_fn,
        "targets": targets,
        "out_of_scope_controls": out_of_scope_controls,
    }


# ============================================================================
# 4. M3 Forest Formation & Unified Retriever-Parameterized Target Executor
# ============================================================================

class M3ForestResearcher:
    """Forms the $K=36$ `PhilosophicalForest` and executes target workspaces under pluggable retrievers!"""

    def __init__(
        self,
        *,
        gate: TwoStageWormholeGate | None = None,
        retrieval_mode: str = "hipporag_ppr",
        max_recalled_trials: int = 2,
        max_fit_trials: int | None = None,
    ) -> None:
        self.forest = PhilosophicalForest()
        self.gate = gate if gate is not None else TwoStageWormholeGate(margin=1.0, uncovered_exploration_budget_calls=8)
        self.retrieval_mode = retrieval_mode
        self.max_recalled_trials = int(max_recalled_trials)
        self.max_fit_trials = int(max_fit_trials) if max_fit_trials is not None else max(4, int(max_recalled_trials) * 2)
        self.learned_scale_certificates: dict[str, dict[str, Any]] = {}
        self.index: HippoRAGWormholeIndex | FlatEqualMetadataIndex | None = None
        self.index_build_compute_cost: float = 0.0
        self.index_build_wall_seconds: float = 0.0
        self.formation_events: list[dict[str, Any]] = []
        self.revision_events: list[dict[str, Any]] = []
        self.retrieval_logs: list[dict[str, Any]] = []

    def build_or_refresh_index(self) -> float:
        if self.retrieval_mode in {"hipporag_ppr", "hipporag_unchecked"}:
            idx: HippoRAGWormholeIndex | FlatEqualMetadataIndex = HippoRAGWormholeIndex(
                self.forest, allow_horizontal_wormholes=True, wormhole_equivalence_pooling=True, damping_alpha=0.80
            )
        elif self.retrieval_mode == "hipporag_alpha0_pool":
            idx = HippoRAGWormholeIndex(
                self.forest, allow_horizontal_wormholes=True, wormhole_equivalence_pooling=True, damping_alpha=0.0
            )
        elif self.retrieval_mode == "hipporag_alpha0_nopool":
            idx = HippoRAGWormholeIndex(
                self.forest, allow_horizontal_wormholes=True, wormhole_equivalence_pooling=False, damping_alpha=0.0
            )
        elif self.retrieval_mode == "hipporag_walk_nopool":
            idx = HippoRAGWormholeIndex(
                self.forest, allow_horizontal_wormholes=True, wormhole_equivalence_pooling=False, damping_alpha=0.80
            )
        elif self.retrieval_mode == "vertical_ppr_no_wh":
            idx = HippoRAGWormholeIndex(
                self.forest, allow_horizontal_wormholes=False, wormhole_equivalence_pooling=False, damping_alpha=0.80
            )
        elif self.retrieval_mode == "flat_bm25_wormhole_pool":
            idx = FlatEqualMetadataIndex(self.forest, mode="bm25", wormhole_equivalence_pooling=True)
        elif self.retrieval_mode == "flat_bm25":
            idx = FlatEqualMetadataIndex(self.forest, mode="bm25", wormhole_equivalence_pooling=False)
        elif self.retrieval_mode == "flat_cosine":
            idx = FlatEqualMetadataIndex(self.forest, mode="idf_cosine", wormhole_equivalence_pooling=False)
        else:
            raise ValueError(f"Unknown retrieval_mode: {self.retrieval_mode}")
        self.index = idx
        self.index_build_compute_cost = float(idx.index_build_compute_cost)
        self.index_build_wall_seconds = float(idx.build_wall_seconds)
        return self.index_build_compute_cost

    def clone_with_mode(self, retrieval_mode: str) -> "M3ForestResearcher":
        """Create an exact equal-content clone changing ONLY the retrieval / contract mode!"""
        cloned = M3ForestResearcher(
            gate=deepcopy(self.gate),
            retrieval_mode=retrieval_mode,
            max_recalled_trials=self.max_recalled_trials,
            max_fit_trials=self.max_fit_trials,
        )
        cloned.forest = deepcopy(self.forest)
        cloned.learned_scale_certificates = deepcopy(self.learned_scale_certificates)
        cloned.build_or_refresh_index()
        return cloned

    def form_and_revise_36_episodes(
        self,
        cluster_id: str,
        suite: dict[str, Any],
        root_workspace: Path,
    ) -> list[dict[str, Any]]:
        """Solve all 32 source + 4 witness near-miss episodes in real workspaces & wire forest + wormholes!"""
        logs: list[dict[str, Any]] = []
        by_ast_digest: dict[str, list[dict[str, Any]]] = {}
        node_by_domain: dict[str, str] = {}
        one_dim_entries: list[dict[str, Any]] = []

        for ep_name, (fac, ev_fn) in suite["sources"]:
            ws = root_workspace / f"M3_src_{ep_name}"
            lab = fac()
            lens = _estimate_observation_lens(lab, ws)
            sub = solve_m3_workspace_direct(lab, ws, existing_lens=lens)
            lab.meter.charge("ast_formation", compute=6.0)
            ev = ev_fn(ws, sub, lab.observations)
            sigs, roles, _ = extract_m3_observable_signatures(lab, lens)
            ast_digest = sub.get("ast_skeleton_digest")
            schema_tag = sub.get("selected_schema")
            ar = len(_active_and_fixed_features(lab)[0])
            model_path = sub.get("model_path")
            ast_meta = (
                canonicalize_python_model_ast((ws / model_path).read_text(encoding="utf-8"))
                if (model_path and (ws / model_path).is_file())
                else {"skeleton_status": "UNVERIFIED"}
            )
            skel_status = str(ast_meta.get("skeleton_status", "UNVERIFIED"))

            rec = {
                "phase": "source",
                "arm": "Hegel_M3",
                "cluster_id": cluster_id,
                "episode_name": ep_name,
                "surface_domain": lab.surface_domain,
                "active_arity": ar,
                "strict_pass": ev["strict_pass"],
                "selected_schema": schema_tag,
                "ast_skeleton_digest": ast_digest,
                "skeleton_status": skel_status,
                "costs": {
                    **lab.meter.snapshot(),
                    "measured_python_wall_seconds": round(lab.measured_python_wall_seconds, 4),
                },
            }
            self.forest.record_raw_episode(rec)
            logs.append(rec)

            if not ev["strict_pass"] or not ast_digest or not schema_tag:
                continue

            entry = {
                "episode_name": ep_name,
                "workspace": ws,
                "surface_domain": lab.surface_domain,
                "active_arity": ar,
                "schema_tag": schema_tag,
                "ast_digest": ast_digest,
                "skeleton_status": skel_status,
                "sigs": sigs,
                "roles": roles,
                "lens": lens,
                "tolerance": float(lab.task_view["tolerance"]),
                "confirmation_rmse": float(sub.get("confirmation_rmse") or ev.get("rmse") or 0.01),
                "eval_rmse": float(ev.get("rmse") or 0.01),
                "paid_observations": list(lab.observations),
            }
            # Only group across trees when AST is strictly VERIFIED_PURE_AST; unverified nodes stay singleton-local!
            group_key = ast_digest if skel_status == "VERIFIED_PURE_AST" else f"local_unverified:{ep_name}:{ast_digest}"
            by_ast_digest.setdefault(group_key, []).append(entry)
            if ar == 1 and skel_status == "VERIFIED_PURE_AST":
                one_dim_entries.append(entry)

        primary_monod_wh: str | None = None
        node_by_ep: dict[str, str] = {}

        for ast_digest, group in by_ast_digest.items():
            nids: list[str] = []
            for item in group:
                dom = item["surface_domain"]
                nid = f"mid_{dom}_{ast_digest[:6]}"
                tid = f"tree_{dom}"
                reg = self.forest.register_node(
                    node_id=nid,
                    tree_id=tid,
                    layer_role="mid",
                    task_view=TaskView("open_workspace_mechanism_prediction", "rmse_original_scale", item["tolerance"]),
                    observation_map=ObservationMap(
                        raw_channel=f"{dom}_readout",
                        unit_scale=item["lens"]["scale"],
                        sign_convention=int(item["lens"]["sign"]),
                        parameter_semantics={
                            "source_surface_domain": dom,
                            "active_arity": item["active_arity"],
                            "ast_skeleton_digest": ast_digest,
                            "schema_tag": item["schema_tag"],
                        },
                    ),
                    claim_contract=ClaimContract(
                        claim_id=f"m3_law_{ast_digest[:8]}_{dom}",
                        family_plugin="m3_open_workspace_ast",
                        preconditions=[],
                        conclusion_or_action={"schema_tag": item["schema_tag"], "ast_skeleton_digest": ast_digest},
                        protocol_spec={"subprocess_python_verified": True},
                        tolerance=item["tolerance"],
                        evidence_status="SUPPORTED_IN_SCOPE",
                        permission_tier="local_solver_guidance",
                    ),
                    probe_signatures=item["sigs"],
                    program_steps=[
                        {"op": "calibrate_observation_lens"},
                        {"op": "synthesize_scoped_python_schema", "schema_tag": item["schema_tag"]},
                        {"op": "subprocess_confirm_and_refit"},
                    ],
                    source_cluster_ids=[cluster_id],
                )
                nids.append(reg["node_id"])
                node_by_domain[dom] = reg["node_id"]
                node_by_ep[item["episode_name"]] = reg["node_id"]
                self.formation_events.append({
                    "cluster_id": cluster_id,
                    "event": "register_m3_node",
                    "node_id": reg["node_id"],
                    "schema_tag": item["schema_tag"],
                    "probe_signatures": item["sigs"],
                })

            # Wire cross-tree HorizontalWormholeEdges across all corroborated AST pairs!
            if len(nids) >= 2:
                for i_pair in range(len(nids) - 1):
                    g_a, g_b = group[i_pair], group[i_pair + 1]
                    wh_id = f"wh_m3_{ast_digest[:8]}_{i_pair}"
                    emp_err = max(g_a["eval_rmse"], g_b["eval_rmse"], g_a["confirmation_rmse"], g_b["confirmation_rmse"])
                    self.forest.add_horizontal_wormhole(
                        wormhole_id=wh_id,
                        node_a_id=nids[i_pair],
                        node_b_id=nids[i_pair + 1],
                        probe_signature_refs=sorted(set(g_a["sigs"]) | set(g_b["sigs"])),
                        role_and_observation_map={
                            "ast_skeleton_digest": ast_digest,
                            "schema_tag": g_a["schema_tag"],
                            "empirically_calibrated": True,
                        },
                        hop_error_budget=round(float(emp_err), 5),
                        lipschitz_factor=1.0,
                        permission_tier="local_solver_guidance",
                        compatible_goal_kinds=["open_workspace_mechanism_prediction"],
                        source_episode_refs=[g_a["episode_name"], g_b["episode_name"]],
                    )
                    if g_a["schema_tag"] == "rational_monod_1d":
                        primary_monod_wh = wh_id

        # Empirical 2nd-hop approximate bridge between S04 (`rational_monod_1d`) and S05 (`congestion_rollover_1d`)!
        monod_eps = [e for e in one_dim_entries if e["schema_tag"] == "rational_monod_1d"]
        roll_eps = [e for e in one_dim_entries if e["schema_tag"] == "congestion_rollover_1d"]
        if monod_eps and roll_eps and primary_monod_wh is not None:
            b_src, r_src = monod_eps[-1], roll_eps[0]
            probe_lab = OpenWorkspaceTaskLab(
                task_id="m3_bridge_fit",
                surface_domain=r_src["surface_domain"],
                feature_names=(r_src["roles"][0],),
                feature_bounds={r_src["roles"][0]: (1.0, 28.0)},
                cost_per_obs=0.0,
                task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.03},
                meter=CostMeter(obs_budget=50, meas_budget=100.0, compute_budget=500.0),
                private_oracle_fn=lambda _p: 0.0,
            )
            bfit = run_m3_workspace_synthesizer(
                r_src["workspace"] / "_m3_bridge",
                probe_lab,
                observation_lens=r_src["lens"],
                allowed_schemas=["rational_monod_1d"],
                output_model_path="bridge_monod.py",
                observations_override=r_src["paid_observations"],
            )
            bridge_rec = {
                "phase": "source_bridge_calibration",
                "arm": "Hegel_M3",
                "cluster_id": cluster_id,
                "episode_name": "m3_bridge_fit_rollover",
                "surface_domain": r_src["surface_domain"],
                "strict_pass": not bool(bfit.get("execution_error")),
                "costs": {
                    **probe_lab.meter.snapshot(),
                    "measured_python_wall_seconds": round(probe_lab.measured_python_wall_seconds, 4),
                },
            }
            self.forest.record_raw_episode(bridge_rec)
            logs.append(bridge_rec)
            emp_23 = float(bfit.get("train_rmse", 0.04))
            nb, nr = node_by_ep.get(b_src["episode_name"]), node_by_ep.get(r_src["episode_name"])
            if nb and nr and emp_23 <= 0.035:
                self.forest.add_horizontal_wormhole(
                    wormhole_id="wh_m3_emp_rollover_hop2",
                    node_a_id=nb,
                    node_b_id=nr,
                    probe_signature_refs=list(r_src["sigs"]),
                    role_and_observation_map={
                        "empirical_approximate_bridge": True,
                        "chain_parent_wormhole": primary_monod_wh,
                        "bridge_from_schema": "rational_monod_1d",
                        "bridge_to_schema": "congestion_rollover_1d",
                    },
                    hop_error_budget=round(emp_23, 5),
                    lipschitz_factor=1.0,
                    permission_tier="local_solver_guidance",
                    compatible_goal_kinds=["open_workspace_mechanism_prediction"],
                    source_episode_refs=[b_src["episode_name"], r_src["episode_name"]],
                )

        # --- 4 Witness-Driven Intra-Tree Revisions (wiring REAL vertical `local_implication_edges` + ALIGN exclusions!) ---
        for nm_name, parent_domain, (nm_fac, nm_ev) in suite["near_misses"]:
            ws = root_workspace / f"M3_nm_{nm_name}"
            lab = nm_fac()
            lens = _estimate_observation_lens(lab, ws)
            lab.acquire(build_m3_pilot_points(lab), workspace=ws, phase="presolve")

            parent_nid = node_by_domain.get(parent_domain)
            parent_schema = (
                self.forest.nodes[parent_nid]["claim_contract"]["conclusion_or_action"]["schema_tag"]
                if parent_nid and parent_nid in self.forest.nodes
                else None
            )
            witness_rmse = float("inf")
            if parent_schema:
                inc_res = run_m3_workspace_synthesizer(
                    ws, lab, observation_lens=lens, allowed_schemas=[parent_schema], output_model_path="incumbent_check.py"
                )
                witness_rmse = float(inc_res.get("train_rmse", float("inf")))

            sub = solve_m3_workspace_direct(lab, ws, existing_lens=lens)
            lab.meter.charge("ast_revision", compute=8.0)
            ev = nm_ev(ws, sub, lab.observations)
            sigs, _, feat_info = extract_m3_observable_signatures(lab, lens)
            nm_schema = sub.get("selected_schema")
            nm_digest = sub.get("ast_skeleton_digest")
            nm_mpath = sub.get("model_path")
            nm_ast_meta = (
                canonicalize_python_model_ast((ws / nm_mpath).read_text(encoding="utf-8"))
                if (nm_mpath and (ws / nm_mpath).is_file())
                else {"skeleton_status": "UNVERIFIED"}
            )
            nm_skel_status = str(nm_ast_meta.get("skeleton_status", "UNVERIFIED"))
            ar = len(_active_and_fixed_features(lab)[0])

            rec_nm = {
                "phase": "source_nearmiss",
                "arm": "Hegel_M3",
                "cluster_id": cluster_id,
                "episode_name": nm_name,
                "surface_domain": lab.surface_domain,
                "active_arity": ar,
                "strict_pass": ev["strict_pass"],
                "incumbent_parent_schema": parent_schema,
                "incumbent_witness_rmse": round(witness_rmse, 6) if math.isfinite(witness_rmse) else None,
                "selected_schema": nm_schema,
                "ast_skeleton_digest": nm_digest,
                "skeleton_status": nm_skel_status,
                "costs": {
                    **lab.meter.snapshot(),
                    "measured_python_wall_seconds": round(lab.measured_python_wall_seconds, 4),
                },
            }
            self.forest.record_raw_episode(rec_nm)
            logs.append(rec_nm)

            if not ev["strict_pass"] or not nm_schema or not nm_digest or not parent_nid:
                continue

            # Determine exclusion predicate added onto parent's wormhole!
            excl_pred: dict[str, Any] | None = None
            if nm_schema == "liebig_capped_min_2d":
                excl_pred = {"field": "high_corner_ceiling_detected", "op": "==", "value": True}
            elif nm_schema == "underdamped_ringing_2d":
                excl_pred = {"field": "transient_overshoot_detected", "op": "==", "value": True}
            elif nm_schema == "conjugate_relaxation_state_memory":
                excl_pred = {"field": "unreset_third_state_active", "op": "==", "value": True}
            elif nm_schema == "haldane_shunt_2d":
                excl_pred = {"field": "high_substrate_turnover_detected", "op": "==", "value": True}
            elif nm_schema == "congestion_rollover_1d":
                excl_pred = {"field": "high_load_rollover_detected", "op": "==", "value": True}

            if excl_pred is not None:
                neg_prec = {"field": excl_pred["field"], "op": "!=", "value": excl_pred["value"]}
                self.forest.nodes[parent_nid]["claim_contract"].setdefault("preconditions", []).append(neg_prec)
                for wid, wh in list(self.forest.horizontal_wormhole_edges.items()):
                    if wh["node_a_id"] == parent_nid or wh["node_b_id"] == parent_nid:
                        self.forest.refine_wormhole_with_near_miss(
                            wid,
                            near_miss_episode_id=nm_name,
                            exclusion_predicate=excl_pred,
                            reason=f"Witness {nm_name} refuted {parent_schema} (rmse={witness_rmse:.4f}) -> route to child {nm_schema}!",
                        )
                        peer_id = wh["node_b_id"] if wh["node_a_id"] == parent_nid else wh["node_a_id"]
                        if peer_id in self.forest.nodes:
                            self.forest.nodes[peer_id]["claim_contract"].setdefault("preconditions", []).append(dict(neg_prec))

            child_sigs = list(lab.task_view.get("witness_only_signatures") or sigs)
            child_nid = f"mid_rev_{parent_domain}_{nm_digest[:6]}"
            parent_tree_id = self.forest.nodes[parent_nid]["tree_id"]
            self.forest.register_node(
                node_id=child_nid,
                tree_id=parent_tree_id,  # Same domain tree -> true vertical intra-tree refinement edge!
                layer_role="mid",
                task_view=TaskView("open_workspace_mechanism_prediction", "rmse_original_scale", float(lab.task_view["tolerance"])),
                observation_map=ObservationMap(
                    raw_channel=f"{parent_domain}_revised_readout",
                    unit_scale=lens["scale"],
                    sign_convention=int(lens["sign"]),
                    parameter_semantics={
                        "source_surface_domain": parent_domain,
                        "active_arity": ar,
                        "ast_skeleton_digest": nm_digest,
                        "schema_tag": nm_schema,
                    },
                ),
                claim_contract=ClaimContract(
                    claim_id=f"m3_rev_{nm_digest[:8]}_{nm_name}",
                    family_plugin="m3_open_workspace_ast",
                    preconditions=([excl_pred] if excl_pred else []),
                    conclusion_or_action={"schema_tag": nm_schema, "ast_skeleton_digest": nm_digest},
                    protocol_spec={"subprocess_python_verified": True},
                    tolerance=float(lab.task_view["tolerance"]),
                    evidence_status="SUPPORTED_IN_SCOPE",
                    permission_tier="local_solver_guidance",
                ),
                probe_signatures=child_sigs,
                program_steps=[
                    {"op": "calibrate_observation_lens"},
                    {"op": "synthesize_scoped_python_schema", "schema_tag": nm_schema},
                    {"op": "subprocess_confirm_and_refit"},
                ],
                source_cluster_ids=[cluster_id],
            )
            # Wire intra-tree vertical implication edge `child -> parent`!
            self.forest.add_local_implication_edge(
                stronger_node_id=child_nid,
                weaker_node_id=parent_nid,
                preserved_contract={"zero_limit_reduction_to_parent": parent_schema},
                scope={"witness_episode": nm_name},
                evidence_status="SUPPORTED_IN_SCOPE",
            )
            # Only wire cross-tree wormhole if BOTH child and peer have VERIFIED_PURE_AST!
            peer_same_ast = (
                [
                    nid for nid, n in self.forest.nodes.items()
                    if nid != child_nid and n["tree_id"] != parent_tree_id
                    and n["observation_map"].get("parameter_semantics", {}).get("ast_skeleton_digest") == nm_digest
                ]
                if nm_skel_status == "VERIFIED_PURE_AST"
                else []
            )
            if peer_same_ast:
                self.forest.add_horizontal_wormhole(
                    wormhole_id=f"wh_m3_rev_{nm_digest[:8]}",
                    node_a_id=peer_same_ast[0],
                    node_b_id=child_nid,
                    probe_signature_refs=child_sigs,
                    role_and_observation_map={"ast_skeleton_digest": nm_digest, "schema_tag": nm_schema},
                    hop_error_budget=round(float(sub.get("confirmation_rmse") or 0.015), 5),
                    permission_tier="local_solver_guidance",
                    compatible_goal_kinds=["open_workspace_mechanism_prediction"],
                    source_episode_refs=[nm_name],
                )
            self.revision_events.append({
                "cluster_id": cluster_id,
                "witness_episode": nm_name,
                "parent_node_id": parent_nid,
                "child_node_id": child_nid,
                "synthesized_schema": nm_schema,
                "witness_rmse_before": round(witness_rmse, 6) if math.isfinite(witness_rmse) else None,
            })

        # Also attach empirical 2-timescale snap exclusion onto 1-mode `conjugate_relaxation_reset` when 2-mode bimodal sources exist
        has_bimodal = any(
            n["claim_contract"]["conclusion_or_action"].get("schema_tag") == "bimodal_relaxation_2mode"
            for n in self.forest.nodes.values()
        )
        if has_bimodal:
            snap_excl = {"field": "multi_timescale_snap_detected", "op": "==", "value": True}
            snap_neg = {"field": "multi_timescale_snap_detected", "op": "!=", "value": True}
            for nid_c, node_c in self.forest.nodes.items():
                st_c = node_c["claim_contract"]["conclusion_or_action"].get("schema_tag")
                if st_c == "conjugate_relaxation_reset":
                    node_c["claim_contract"].setdefault("preconditions", []).append(dict(snap_neg))
                elif st_c == "bimodal_relaxation_2mode":
                    node_c["claim_contract"].setdefault("preconditions", []).append(dict(snap_excl))
            for wid, wh in list(self.forest.horizontal_wormhole_edges.items()):
                if wh.get("role_and_observation_map", {}).get("schema_tag") == "conjugate_relaxation_reset":
                    self.forest.refine_wormhole_with_near_miss(
                        wid,
                        near_miss_episode_id="S16_bimodal_contrast",
                        exclusion_predicate=snap_excl,
                        reason="2-timescale initial snap exceeds 1-mode exponential envelope",
                    )

        # --- Empirical ScaleMap Calibration ---
        if callable(suite.get("scale_calibration_probe_fn")):
            cal_meter = CostMeter(obs_budget=10, meas_budget=60.0, compute_budget=100.0)
            s_res = suite["scale_calibration_probe_fn"](cal_meter)
            if s_res is not None:
                preserved = []
                verified = []
                if s_res["mean_flux_diff"] <= 0.03:
                    preserved.append("open_workspace_mechanism_prediction")
                    verified.append({
                        "variable": "micro_oscillation_phase",
                        "task_view_goal": "open_workspace_mechanism_prediction",
                        "error_bound": round(float(s_res["mean_flux_diff"]), 6),
                        "evidence_ref": f"m3_scale_probe:{cluster_id}",
                    })
                if s_res["peak_crest_diff"] <= 0.08:
                    preserved.append("micro_phase_peak_crest")
                self.learned_scale_certificates["window_average"] = {
                    "preserved_queries": preserved,
                    "verified_discarded_in_scope": verified,
                }
                rec_sc = {
                    "phase": "source_scale_calibration",
                    "arm": "Hegel_M3",
                    "cluster_id": cluster_id,
                    "episode_name": "M3_scale_probe",
                    "surface_domain": "scale_calibration",
                    "strict_pass": True,
                    "costs": cal_meter.snapshot(),
                }
                logs.append(rec_sc)

        idx_cost = self.build_or_refresh_index()
        logs.append({
            "phase": "index_build",
            "arm": f"Hegel_M3_{self.retrieval_mode}",
            "cluster_id": cluster_id,
            "episode_name": f"index_build_{self.retrieval_mode}",
            "surface_domain": "forest_index",
            "strict_pass": True,
            "costs": {
                "observation_cost": 0,
                "measurement_cost": 0.0,
                "compute_cost": idx_cost,
                "composite_cost": round(0.05 * idx_cost, 4),
                "measured_python_wall_seconds": round(self.index_build_wall_seconds, 5),
            },
        })
        return logs

    def _effective_scale_map(self, raw_sm: dict[str, Any] | None) -> dict[str, Any] | None:
        if raw_sm is None:
            return None
        merged = deepcopy(raw_sm)
        op = merged.get("coarse_graining_op")
        if op in self.learned_scale_certificates:
            cert = self.learned_scale_certificates[op]
            merged["preserved_queries"] = list(dict.fromkeys([*merged.get("preserved_queries", ()), *cert["preserved_queries"]]))
            merged["verified_discarded_in_scope"] = [*merged.get("verified_discarded_in_scope", ()), *deepcopy(cert["verified_discarded_in_scope"])]
        return merged

    def run_paired_post_pilot_rollout(
        self,
        task_factory,
        evaluator_fn,
        dev_workspace: Path,
    ) -> dict[str, Any] | None:
        ws_pilot = dev_workspace / "pilot_shared"
        lab_base = task_factory()
        lens = _estimate_observation_lens(lab_base, ws_pilot)
        if lab_base.acquire(build_m3_pilot_points(lab_base), workspace=ws_pilot, phase="presolve") is None:
            return None
        sigs, role_order, feat_info = extract_m3_observable_signatures(lab_base, lens)
        sig_key = "|".join(s for s in sigs if not s.startswith(("sig:calibrated_", "sig:scale_")))
        pilot_cost = lab_base.meter.snapshot()["composite_cost"]

        lab_t = deepcopy(lab_base)
        lab_d = deepcopy(lab_base)
        sub_t = self._continue_after_pilot("dev_fork", lab_t, dev_workspace / "fork_t", lens=lens, sigs=sigs, role_order=role_order, feat_info=feat_info, bypass_stage1=True)
        ev_t = evaluator_fn(dev_workspace / "fork_t", sub_t, lab_t.observations)
        sub_d = solve_m3_workspace_direct(lab_d, dev_workspace / "fork_d", existing_lens=lens)
        ev_d = evaluator_fn(dev_workspace / "fork_d", sub_d, lab_d.observations)
        return {
            "signature_key": sig_key,
            "pilot_composite_cost": round(pilot_cost, 4),
            "transfer_remaining_cost": round(lab_t.meter.snapshot()["composite_cost"] - pilot_cost, 4),
            "direct_remaining_cost": round(lab_d.meter.snapshot()["composite_cost"] - pilot_cost, 4),
            "transfer_quality_ok": bool(ev_t["strict_pass"]),
            "direct_quality_ok": bool(ev_d["strict_pass"]),
        }

    def solve_target(
        self,
        name: str,
        lab: OpenWorkspaceTaskLab,
        workspace: Path,
        *,
        ground_truth_schema_for_audit_log_only: str | None = None,
    ) -> dict[str, Any]:
        """Unified M3 target execution loop — IDENTICAL across PPR, BM25, Cosine, VerticalPPR, & Unchecked!"""
        workspace.mkdir(parents=True, exist_ok=True)
        enforce_contracts = self.retrieval_mode != "hipporag_unchecked"
        eff_scale = self._effective_scale_map(lab.task_view.get("available_scale_map"))

        if enforce_contracts and eff_scale is not None:
            ref_cc = ClaimContract("scale_gate", "m3_open_workspace_ast", [], {}, {}, float(lab.task_view["tolerance"]), "SUPPORTED_IN_SCOPE", "local_solver_guidance").to_dict()
            s_auth = authorize_abstraction_use(
                requested_tier="local_solver_guidance",
                task_view=lab.task_view,
                target_features={},
                claim_contract=ref_cc,
                scale_map=eff_scale,
                heldout_confirmed=True,
            )
            if not s_auth["granted"] and s_auth["reason"] == "scale_map_does_not_preserve_task_view":
                self.retrieval_logs.append({
                    "target": name,
                    "mode": self.retrieval_mode,
                    "blocked_by_scale_map_pre_pilot": True,
                    "wrong_candidate_trials": 0,
                })
                return solve_m3_workspace_direct(lab, workspace)

        lens = _estimate_observation_lens(lab, workspace)
        if lab.acquire(build_m3_pilot_points(lab), workspace=workspace, phase="presolve") is None:
            return {"status": "incomplete", "model_path": None}
        sigs, role_order, feat_info = extract_m3_observable_signatures(lab, lens)
        return self._continue_after_pilot(
            name,
            lab,
            workspace,
            lens=lens,
            sigs=sigs,
            role_order=role_order,
            feat_info=feat_info,
            bypass_stage1=False,
            ground_truth_schema_for_audit_log_only=ground_truth_schema_for_audit_log_only,
        )

    def _continue_after_pilot(
        self,
        name: str,
        lab: OpenWorkspaceTaskLab,
        workspace: Path,
        *,
        lens: dict[str, float],
        sigs: list[str],
        role_order: list[str],
        feat_info: dict[str, Any],
        bypass_stage1: bool = False,
        ground_truth_schema_for_audit_log_only: str | None = None,
    ) -> dict[str, Any]:
        if self.index is None:
            self.build_or_refresh_index()
        assert self.index is not None

        enforce_contracts = self.retrieval_mode != "hipporag_unchecked"
        eff_scale = self._effective_scale_map(lab.task_view.get("available_scale_map"))
        tol = float(lab.task_view["tolerance"])
        active_ar = len(_active_and_fixed_features(lab)[0])

        ret_tv = (
            {**lab.task_view, "active_arity": active_ar}
            if enforce_contracts
            else {"active_arity": active_ar}
        )
        ret = self.index.retrieve(sigs, task_view=ret_tv, top_k=6)
        lab.meter.charge("retrieve", compute=float(ret["query_compute_cost"]))

        raw_ranked = ret["candidates"]
        recalled_schemas_raw = [
            c["target_node"]["claim_contract"]["conclusion_or_action"].get("schema_tag")
            for c in raw_ranked
        ]

        # Filter candidates via Hegel contracts (`compose_wormhole_chain` + `ALIGN` preconditions/exclusions) BEFORE spending subprocess synthesis!
        authorized_candidates: list[dict[str, Any]] = []
        contract_filtered_count = 0
        for cand in raw_ranked:
            c_schema = cand["target_node"]["claim_contract"]["conclusion_or_action"].get("schema_tag")
            wh_chain = list(cand.get("wormhole_path_edges", ()))
            wh_edge = cand.get("wormhole_edge")

            if enforce_contracts:
                # 2-hop non-transitive chain check!
                if feat_info.get("high_load_rollover_detected") and c_schema == "rational_monod_1d":
                    hop2 = [
                        e for e in self.forest.horizontal_wormhole_edges.values()
                        if e.get("role_and_observation_map", {}).get("bridge_from_schema") == "rational_monod_1d"
                    ]
                    if wh_edge is not None and hop2:
                        cc_res = compose_wormhole_chain([wh_edge, hop2[0]], target_tolerance=tol)
                        if not cc_res["within_tolerance"]:
                            contract_filtered_count += 1
                            continue
                elif len(wh_chain) >= 2 and c_schema == wh_chain[-1].get("role_and_observation_map", {}).get("bridge_from_schema"):
                    cc_res = compose_wormhole_chain(wh_chain, target_tolerance=tol)
                    if not cc_res["within_tolerance"]:
                        contract_filtered_count += 1
                        continue

                t_auth = authorize_abstraction_use(
                    requested_tier="bounded_trial",
                    task_view=lab.task_view,
                    target_features={**dict(lab.domain_regime_attrs), **feat_info},
                    claim_contract=cand["target_node"]["claim_contract"],
                    scale_map=eff_scale,
                    wormhole_edge=wh_edge,
                )
                if not t_auth["granted"]:
                    contract_filtered_count += 1
                    continue

            authorized_candidates.append(cand)

        sig_key = "|".join(s for s in sigs if not s.startswith(("sig:calibrated_", "sig:scale_")))
        gate_dec = (
            {"admit_transfer": True, "mode": "unchecked_or_dev"}
            if (bypass_stage1 or not enforce_contracts)
            else self.gate.decide_stage1(
                sig_key,
                remaining_budget=lab.meter.remaining_meas(),
                estimated_probe_and_bind_cost=lab.cost_per_obs,
            )
        )

        wrong_trials = 0
        confirm_purchases = 0
        confirmed_via_recall = False
        chosen_schema_final: str | None = None
        non_direct_walk_recalled = int(ret.get("non_direct_walk_recalled_count", 0))

        if authorized_candidates and gate_dec["admit_transfer"]:
            for trial_idx, cand in enumerate(authorized_candidates[: self.max_fit_trials]):
                if confirm_purchases >= self.max_recalled_trials:
                    break
                node = cand["target_node"]
                wh_edge = cand.get("wormhole_edge")
                # Genuinely interpret declarative `program_steps`! Deleting ops changes execution!
                prog_steps = [s for s in node.get("program_steps", ()) if isinstance(s, dict)]
                prog_ops = {s.get("op"): s for s in prog_steps}
                c_schema = (
                    prog_ops.get("synthesize_scoped_python_schema", {}).get("schema_tag")
                    if not enforce_contracts
                    else prog_ops.get("synthesize_scoped_python_schema", {}).get("schema_tag")
                )
                if not enforce_contracts and not c_schema:
                    c_schema = node["claim_contract"]["conclusion_or_action"].get("schema_tag")
                if not c_schema:
                    continue

                exec_lens = (
                    lens
                    if ("calibrate_observation_lens" in prog_ops or not enforce_contracts)
                    else {"sign": 1.0, "scale": 1.0, "offset": 0.0}
                )

                cand_rel = f"recalled_cand_{trial_idx}.py"
                t_fit = run_m3_workspace_synthesizer(
                    workspace,
                    lab,
                    observation_lens=exec_lens,
                    allowed_schemas=[c_schema],
                    forced_role_order=(role_order if role_order else None),
                    output_model_path=cand_rel,
                )
                if t_fit.get("execution_error") or t_fit["train_rmse"] > tol:
                    wrong_trials += 1
                    continue

                if enforce_contracts and "subprocess_confirm_and_refit" not in prog_ops:
                    wrong_trials += 1
                    continue

                # Candidate fit pilot within tolerance -> buy fresh prefix-isolated confirmation point!
                fit_prefix = list(lab.observations)
                c_pt = build_m3_confirmation_point(
                    lab,
                    idx_variant=confirm_purchases,
                    role_order=(t_fit.get("role_order") or role_order),
                    symmetric_roles=bool(feat_info.get("symmetric_roles", False)),
                )
                c_rows = lab.acquire([c_pt], workspace=None, phase="confirm")
                if not c_rows:
                    break
                confirm_purchases += 1
                vr = validate_workspace_model_on_rows(workspace, cand_rel, c_rows, lab, training_prefix=fit_prefix)
                lab.sync_workspace_files(workspace)
                obs_err = vr["rmse"] if not vr.get("execution_error") else float("inf")
                if not math.isfinite(obs_err) or obs_err > tol:
                    wrong_trials += 1
                    # If an earlier coarse candidate failed at high-load confirm, update ONLY the single observable regime flag matching this onset class!
                    if lab.task_view.get("pilot_regime") == "safe_low_load":
                        updated_flag: dict[str, Any] = {}
                        if active_ar == 1:
                            updated_flag["high_load_rollover_detected"] = True
                        elif bool(feat_info.get("symmetric_roles", False)):
                            updated_flag["high_corner_ceiling_detected"] = True
                        elif "sig:low_regime_shunt_onset" in sigs:
                            updated_flag["high_substrate_turnover_detected"] = True
                        elif "sig:low_regime_charging_onset" in sigs:
                            updated_flag["transient_overshoot_detected"] = True
                        feat_info = {**feat_info, **updated_flag}
                    continue

                cand_sha = hashlib.sha256((workspace / cand_rel).read_bytes()).hexdigest()
                conf_dig = vr.get("confirmation_digest") or hashlib.sha256(
                    json.dumps([asdict(r) for r in c_rows], sort_keys=True).encode("utf-8")
                ).hexdigest()[:16]
                receipt = {
                    "target_id": name,
                    "claim_id": node["claim_contract"]["claim_id"],
                    "node_version": node["version"],
                    "model_sha256": cand_sha,
                    "confirmation_digest": conf_dig,
                    "observed_error": obs_err,
                    "passed": True,
                    "upgrade_to_tier": "local_solver_guidance",
                }
                if enforce_contracts:
                    g_auth = authorize_abstraction_use(
                        requested_tier="local_solver_guidance",
                        task_view={
                            **lab.task_view,
                            "target_id": name,
                            "node_version": node["version"],
                            "expected_model_sha256": cand_sha,
                            "expected_confirmation_digest": conf_dig,
                        },
                        target_features={**dict(lab.domain_regime_attrs), **feat_info},
                        claim_contract=node["claim_contract"],
                        scale_map=eff_scale,
                        wormhole_edge=wh_edge,
                        evidence_receipt=receipt,
                    )
                    if not g_auth["granted"]:
                        wrong_trials += 1
                        continue

                confirmed_via_recall = True
                chosen_schema_final = t_fit["selected_schema"]
                self.retrieval_logs.append({
                    "target": name,
                    "mode": self.retrieval_mode,
                    "observed_signatures": sigs,
                    "raw_top_schemas": recalled_schemas_raw[:4],
                    "authorized_top_schemas": [
                        x["target_node"]["claim_contract"]["conclusion_or_action"].get("schema_tag")
                        for x in authorized_candidates[:4]
                    ],
                    "non_direct_walk_recalled_count": non_direct_walk_recalled,
                    "selected_reached_only_by_walk": bool(cand.get("reached_only_by_graph_walk", False)),
                    "contract_filtered_count": contract_filtered_count,
                    "wrong_candidate_trials": wrong_trials,
                    "confirm_purchases_in_recall": confirm_purchases,
                    "confirmed_via_recall": True,
                    "truth_schema_audit_only": ground_truth_schema_for_audit_log_only,
                })
                return _freeze_confirmed_submission(
                    workspace,
                    cand_rel,
                    fit_prefix,
                    selected_schema=chosen_schema_final,
                    ast_skeleton_digest=t_fit["ast_skeleton_digest"],
                    confirmation_rmse=obs_err,
                    confirmation_digest=conf_dig,
                    extra_meta={
                        "retrieval_mode": self.retrieval_mode,
                        "wrong_candidate_trials": wrong_trials,
                        "selected_reached_only_by_walk": bool(cand.get("reached_only_by_graph_walk", False)),
                    },
                )

        self.retrieval_logs.append({
            "target": name,
            "mode": self.retrieval_mode,
            "observed_signatures": sigs,
            "raw_top_schemas": recalled_schemas_raw[:4],
            "authorized_top_schemas": [
                x["target_node"]["claim_contract"]["conclusion_or_action"].get("schema_tag")
                for x in authorized_candidates[:4]
            ],
            "non_direct_walk_recalled_count": non_direct_walk_recalled,
            "selected_reached_only_by_walk": False,
            "contract_filtered_count": contract_filtered_count,
            "wrong_candidate_trials": wrong_trials,
            "confirm_purchases_in_recall": confirm_purchases,
            "confirmed_via_recall": confirmed_via_recall,
            "truth_schema_audit_only": ground_truth_schema_for_audit_log_only,
        })
        return solve_m3_workspace_direct(lab, workspace, existing_lens=lens)


# ============================================================================
# 5. Non-Forest Baselines Scaled to $K=36$ (`O_strong_AllValidatedPriors` & `O_HExA`)
# ============================================================================

class M3StrongOrdinaryAllPriors:
    """Empowered ordinary baseline retaining ALL validated same-arity historical schemas from $K=36$ episodes."""

    def __init__(self) -> None:
        self.raw_archive: list[dict[str, Any]] = []
        self.shared_python_files: dict[str, str] = {}
        self.rebuilt_cache: dict[str, Any] = {}

    def run_36_episode_source_phase(self, suite: dict[str, Any], root_workspace: Path) -> list[dict[str, Any]]:
        logs: list[dict[str, Any]] = []
        all_items = [(n, p, "source") for n, p in suite["sources"]] + [(n, p, "source_nearmiss") for n, _dom, p in suite["near_misses"]]
        for ep_name, (fac, ev_fn), phase_tag in all_items:
            ws = root_workspace / f"O_strong_{ep_name}"
            lab = fac()
            sub = solve_m3_workspace_direct(lab, ws)
            ev = ev_fn(ws, sub, lab.observations)
            if ev["strict_pass"] and sub.get("status") == "completed" and (ws / "model.py").is_file():
                self.shared_python_files[f"shared/{ep_name}.py"] = (ws / "model.py").read_text(encoding="utf-8")
            rec = {
                "phase": phase_tag,
                "arm": "O_strong_AllValidatedPriors",
                "episode_name": ep_name,
                "surface_domain": lab.surface_domain,
                "active_arity": len(_active_and_fixed_features(lab)[0]),
                "strict_pass": ev["strict_pass"],
                "selected_schema": sub.get("selected_schema"),
                "costs": {
                    **lab.meter.snapshot(),
                    "measured_python_wall_seconds": round(lab.measured_python_wall_seconds, 4),
                },
            }
            self.raw_archive.append(deepcopy(rec))
            logs.append(rec)
        return logs

    def solve_target(self, name: str, lab: OpenWorkspaceTaskLab, workspace: Path) -> dict[str, Any]:
        workspace.mkdir(parents=True, exist_ok=True)
        if self.shared_python_files:
            sdir = workspace / "shared"
            sdir.mkdir(parents=True, exist_ok=True)
            for rel, code in sorted(self.shared_python_files.items()):
                (workspace / rel).write_text(code, encoding="utf-8")
        ar = len(_active_and_fixed_features(lab)[0])
        if self.rebuilt_cache.get("archive_len") != len(self.raw_archive):
            lab.meter.charge("on_demand_reconstruction", compute=36.0)
            by_ar: dict[int, set[str]] = {}
            for ep in self.raw_archive:
                if ep.get("strict_pass") and ep.get("selected_schema"):
                    by_ar.setdefault(int(ep["active_arity"]), set()).add(str(ep["selected_schema"]))
            self.rebuilt_cache = {"archive_len": len(self.raw_archive), "priors": {k: sorted(v) for k, v in by_ar.items()}}
        lab.meter.charge("retrieve", compute=4.0)
        priors = self.rebuilt_cache["priors"].get(ar)
        return solve_m3_workspace_direct(lab, workspace, reconstructed_prior_schemas=priors)


class M3HExACodeSkills:
    """HExA-inspired conditional skill + mistake-recovery bank across $K=36$ episodes (condition-matched!)."""

    def __init__(self, max_skill_trials: int = 2, max_fit_trials: int | None = None) -> None:
        self.skill_bank: dict[int, list[str]] = {}
        self.skill_sigs: dict[tuple[int, str], set[str]] = {}
        self.mistake_bank: dict[int, list[str]] = {}
        self.mistake_sigs: dict[tuple[int, str], set[str]] = {}
        self.max_skill_trials = int(max_skill_trials)
        self.max_fit_trials = int(max_fit_trials) if max_fit_trials is not None else max(4, int(max_skill_trials) * 2)

    def run_36_episode_source_phase(self, suite: dict[str, Any], root_workspace: Path) -> list[dict[str, Any]]:
        logs: list[dict[str, Any]] = []
        counts_by_ar: dict[int, list[str]] = {}
        for ep_name, (fac, ev_fn) in suite["sources"]:
            ws = root_workspace / f"HExA_{ep_name}"
            lab = fac()
            lens = _estimate_observation_lens(lab, ws)
            sub = solve_m3_workspace_direct(lab, ws, existing_lens=lens)
            lab.meter.charge("distill_skill", compute=6.0)
            ev = ev_fn(ws, sub, lab.observations)
            ar = len(_active_and_fixed_features(lab)[0])
            sigs, _, _ = extract_m3_observable_signatures(lab, lens)
            if ev["strict_pass"] and sub.get("selected_schema"):
                st = str(sub["selected_schema"])
                counts_by_ar.setdefault(ar, []).append(st)
                self.skill_sigs.setdefault((ar, st), set()).update(
                    s for s in sigs if not s.startswith(("sig:arity_", "sig:calibrated_"))
                )
            logs.append({
                "phase": "source",
                "arm": "O_HExA_CodeSkills",
                "episode_name": ep_name,
                "surface_domain": lab.surface_domain,
                "active_arity": ar,
                "strict_pass": ev["strict_pass"],
                "selected_schema": sub.get("selected_schema"),
                "costs": {
                    **lab.meter.snapshot(),
                    "measured_python_wall_seconds": round(lab.measured_python_wall_seconds, 4),
                },
            })

        for ar, tags in sorted(counts_by_ar.items()):
            self.skill_bank[ar] = sorted(set(tags), key=lambda s: (-tags.count(s), s))

        for nm_name, _dom, (nm_fac, nm_ev) in suite["near_misses"]:
            ws = root_workspace / f"HExA_{nm_name}"
            lab = nm_fac()
            lens = _estimate_observation_lens(lab, ws)
            sub = solve_m3_workspace_direct(lab, ws, existing_lens=lens)
            lab.meter.charge("rewrite_skill", compute=8.0)
            ev = nm_ev(ws, sub, lab.observations)
            ar = len(_active_and_fixed_features(lab)[0])
            sigs, _, _ = extract_m3_observable_signatures(lab, lens)
            if ev["strict_pass"] and sub.get("selected_schema"):
                st = str(sub["selected_schema"])
                if st not in self.mistake_bank.setdefault(ar, []):
                    self.mistake_bank[ar].append(st)
                wit_cues = list(lab.task_view.get("witness_only_signatures") or sigs)
                self.mistake_sigs.setdefault((ar, st), set()).update(
                    s for s in wit_cues if not s.startswith(("sig:arity_", "sig:calibrated_"))
                )
            logs.append({
                "phase": "source_nearmiss",
                "arm": "O_HExA_CodeSkills",
                "episode_name": nm_name,
                "surface_domain": lab.surface_domain,
                "active_arity": ar,
                "strict_pass": ev["strict_pass"],
                "selected_schema": sub.get("selected_schema"),
                "costs": {
                    **lab.meter.snapshot(),
                    "measured_python_wall_seconds": round(lab.measured_python_wall_seconds, 4),
                },
            })
        return logs

    def solve_target(self, name: str, lab: OpenWorkspaceTaskLab, workspace: Path) -> dict[str, Any]:
        workspace.mkdir(parents=True, exist_ok=True)
        lab.meter.charge("retrieve", compute=5.0)
        ar = len(_active_and_fixed_features(lab)[0])
        base_skills = list(self.skill_bank.get(ar, ()))
        recovery_skills = list(self.mistake_bank.get(ar, ()))
        if not base_skills and not recovery_skills:
            return solve_m3_workspace_direct(lab, workspace)

        lens = _estimate_observation_lens(lab, workspace)
        tol = float(lab.task_view["tolerance"])
        if lab.acquire(build_m3_pilot_points(lab), workspace=workspace, phase="presolve") is None:
            return {"status": "incomplete", "model_path": None}
        sigs, _, feat_info = extract_m3_observable_signatures(lab, lens)
        sym_roles = bool(feat_info.get("symmetric_roles", False))
        obs_cues = {s for s in sigs if not s.startswith(("sig:arity_", "sig:calibrated_"))}

        # Condition-aware ranking: mistake-recovery skills whose witness signatures match `obs_cues` rank ahead!
        matched_recovery = [
            st for st in recovery_skills
            if self.mistake_sigs.get((ar, st)) and self.mistake_sigs[(ar, st)].issubset(obs_cues)
        ]
        ranked_base = sorted(
            base_skills,
            key=lambda st: (-len(obs_cues & self.skill_sigs.get((ar, st), set())), base_skills.index(st)),
        )
        ordered = list(dict.fromkeys([*matched_recovery, *ranked_base, *recovery_skills]))

        confirm_purchases = 0
        for idx, cand_schema in enumerate(ordered[: self.max_fit_trials]):
            if confirm_purchases >= self.max_skill_trials:
                break
            c_rel = f"hexa_trial_{idx}.py"
            t_fit = run_m3_workspace_synthesizer(
                workspace, lab, observation_lens=lens, allowed_schemas=[cand_schema], output_model_path=c_rel
            )
            if t_fit.get("execution_error") or t_fit["train_rmse"] > tol:
                continue
            fit_prefix = list(lab.observations)
            c_pt = build_m3_confirmation_point(
                lab, idx_variant=confirm_purchases, role_order=t_fit.get("role_order"), symmetric_roles=sym_roles
            )
            c_rows = lab.acquire([c_pt], workspace=None, phase="confirm")
            if not c_rows:
                break
            confirm_purchases += 1
            vr = validate_workspace_model_on_rows(workspace, c_rel, c_rows, lab, training_prefix=fit_prefix)
            lab.sync_workspace_files(workspace)
            if not vr.get("execution_error") and vr["rmse"] <= tol:
                return _freeze_confirmed_submission(
                    workspace, c_rel, fit_prefix,
                    selected_schema=t_fit["selected_schema"],
                    ast_skeleton_digest=t_fit["ast_skeleton_digest"],
                    confirmation_rmse=vr["rmse"],
                    confirmation_digest=vr.get("confirmation_digest"),
                )
        return solve_m3_workspace_direct(lab, workspace, existing_lens=lens)
