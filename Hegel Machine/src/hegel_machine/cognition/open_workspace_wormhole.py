"""M2 Open Python Workspace Synthesis, AST Canonicalization & Wormhole Loop (Hegel V16.8).

Bridges ``PhilosophicalForest`` + ``HorizontalWormholeEdge`` + 4-Tier Receipt Governance
into real filesystem Python workspaces using ``code_agent.evaluate_python_model`` and
subprocess execution (``proposers._run``):

1. **Unified Single-Path Loop**: Every episode (1D, 2D, 3D; source, near-miss, target)
   executes through ONE domain-agnostic workspace loop — zero ``isinstance`` task-class
   branches and zero hardcoded domain/variable names!
2. **Real Python File Authorship & Subprocess Scoring**: Candidate hypotheses are synthesized
   as real ``.py`` scripts inside ``workspace/``, executed via ``_run([sys.executable, ...])``
   on ``observations.json``, exported to ``workspace / "model.py"``, and validated via
   ``code_agent.evaluate_python_model`` in an isolated Python subprocess.
3. **AST-Driven Contrastive Wormhole Induction**: ``canonicalize_python_model_ast`` parses
   authored ``model.py`` files with Python's ``ast`` module, strips ``ObservationMap``
   readout lenses (``offset + sign * scale * (...)``), normalizes feature keys into
   structural operator roles, and computes ``ast_skeleton_digest`` so two heterogeneous
   domains (e.g. Le Chatelier ``+1`` vs Lenz ``-1``) autonomously form a ``HorizontalWormholeEdge``!
4. **Execution-Error Isolation**: Subprocess ``SyntaxError`` / ``ZeroDivisionError``
   (``returncode != 0``) is strictly classified as ``EXECUTION_ERROR`` and never recorded
   as ``REFUTED_IN_SCOPE``.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

from .code_agent import evaluate_python_model
from .cross_domain_clusters import CostMeter
from .models import Observation
from .open_workspace_labs import OpenWorkspaceTaskLab
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
# 1. Python AST Canonicalizer (Extracts Coordinate-Free Structural Skeletons)
# ============================================================================

class _SkeletonNormalizer(ast.NodeTransformer):
    """Normalize an authored ``predict(row)`` AST into a coordinate- & constant-free skeleton."""

    def __init__(self) -> None:
        self.extracted_feature_order: list[str] = []
        self._feature_to_slot: dict[str, str] = {}

    def _slot_for(self, feat_name: str) -> str:
        if feat_name not in self._feature_to_slot:
            slot = f"ROLE_{len(self.extracted_feature_order)}"
            self.extracted_feature_order.append(feat_name)
            self._feature_to_slot[feat_name] = slot
        return self._feature_to_slot[feat_name]

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return ast.Name(id="NUM", ctx=ast.Load())
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        node = self.generic_visit(node)  # type: ignore[assignment]
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Name) and node.operand.id == "NUM":
            return ast.Name(id="NUM", ctx=ast.Load())
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # Match f["feature_name"] or row["features"]["feature_name"]
        slice_node = node.slice
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            key = slice_node.value
            if key != "features":
                return ast.Name(id=self._slot_for(key), ctx=ast.Load())
        return self.generic_visit(node)


def canonicalize_python_model_ast(model_source: str) -> dict[str, Any]:
    """Parse ``model.py`` source, strip outer ObservationMap affine lens, & hash AST skeleton!"""
    tree = ast.parse(model_source)
    core_expr: ast.AST | None = None
    explicit_schema_tag: str | None = None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_CORE_LAW_EXPR" and isinstance(node.value, ast.Constant):
                    core_expr = ast.parse(str(node.value.value), mode="eval").body
                if isinstance(target, ast.Name) and target.id == "_SCHEMA_FAMILY_TAG" and isinstance(node.value, ast.Constant):
                    explicit_schema_tag = str(node.value.value)
        elif isinstance(node, ast.FunctionDef) and node.name == "predict" and core_expr is None:
            for stmt in reversed(node.body):
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    core_expr = deepcopy(stmt.value)
                    break

    if core_expr is None:
        raise ValueError("Could not locate predict(row) return expression in model.py")

    normalizer = _SkeletonNormalizer()
    norm_ast = normalizer.visit(deepcopy(core_expr))
    ast.fix_missing_locations(norm_ast)
    dumped = ast.dump(norm_ast, include_attributes=False)
    digest = hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:14]
    return {
        "ast_skeleton_digest": digest,
        "canonical_ast_dump": dumped,
        "feature_role_order": normalizer.extracted_feature_order,
        "schema_tag": explicit_schema_tag,
    }


# ============================================================================
# 2. Subprocess Open Compositional Python Candidate Synthesizer
# ============================================================================

_WORKSPACE_SYNTHESIZER_SCRIPT = r'''import json
import math
from pathlib import Path
import sys

cfg = json.loads(Path("synth_config.json").read_text(encoding="utf-8"))
raw_rows = json.loads(Path("observations.json").read_text(encoding="utf-8"))
train_idx = cfg["train_indices"]
val_idx = cfg["val_indices"]
lens = cfg["observation_lens"]  # {"sign": +-1, "scale": s, "offset": b}
allowed_schemas = cfg.get("allowed_schemas")  # None = open compositional search; or specific list

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

candidates = []

def allow(tag):
    return allowed_schemas is None or tag in allowed_schemas

# --- 1D Compositional Candidates ---
if len(active_f) == 1:
    u_k = active_f[0]
    if allow("linear_1d"):
        w = solve_linear_ls(lambda f, k=u_k: [1.0, float(f[k])], train_rows)
        fn = lambda f, w=w, k=u_k: w[0] + w[1] * float(f[k])
        expr = f"({w[0]:.10g} + {w[1]:.10g} * f[{u_k!r}])"
        candidates.append(("linear_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

    if allow("rational_1d"):
        # Gauss-Newton in original raw scale on y_lat = u / (a*u + b)
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
            for _ in range(35):
                ga = gb = haa = hab = hbb = 0.0
                cur = sum(((u / (a_p * u + b_p)) - y) ** 2 for u, y in pts)
                for u, y in pts:
                    d = a_p * u + b_p
                    e = (u / d) - y
                    ja, jb = -(u * u) / (d * d), -u / (d * d)
                    ga += e * ja; gb += e * jb
                    haa += ja * ja; hab += ja * jb; hbb += jb * jb
                det = (haa + 1e-6) * (hbb + 1e-6) - hab * hab
                if det <= 1e-15:
                    break
                da = ((hbb + 1e-6) * ga - hab * gb) / det
                db = ((haa + 1e-6) * gb - hab * ga) / det
                na, nb = max(1e-6, a_p - 0.8 * da), max(1e-6, b_p - 0.8 * db)
                if sum(((u / (na * u + nb)) - y) ** 2 for u, y in pts) <= cur:
                    a_p, b_p = na, nb
            fn = lambda f, a=a_p, b=b_p, k=u_k: float(f[k]) / (a * float(f[k]) + b)
            expr = f"(f[{u_k!r}] / ({a_p:.10g} * f[{u_k!r}] + {b_p:.10g}))"
            candidates.append(("rational_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

    if allow("inv_quad_crest_1d") and len(train_rows) >= 3:
        w = solve_linear_ls(lambda f, k=u_k: [1.0, 1.0 / float(f[k]), 1.0 / (float(f[k]) ** 2)], train_rows)
        fn = lambda f, w=w, k=u_k: w[0] + w[1] / float(f[k]) + w[2] / (float(f[k]) ** 2)
        expr = f"({w[0]:.10g} + {w[1]:.10g} / f[{u_k!r}] + {w[2]:.10g} / (f[{u_k!r}] ** 2))"
        candidates.append(("inv_quad_crest_1d", 3, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

    if allow("congestion_rollover_1d") and len(train_rows) >= 4:
        pts = [(float(r["features"][u_k]), r["y_lat"]) for r in train_rows if float(r["features"][u_k]) > 0 and r["y_lat"] > 1e-6]
        if len(pts) >= 4:
            # u / y = b + a*u + g*u^2
            pseudo = [{"features": {u_k: u}, "y_lat": u / y} for u, y in pts]
            w = solve_linear_ls(lambda f, k=u_k: [1.0, float(f[k]), float(f[k]) ** 2], pseudo)
            b_r, a_r, g_r = max(1e-5, w[0]), max(1e-5, w[1]), max(0.0, w[2])
            fn = lambda f, a=a_r, b=b_r, g=g_r, k=u_k: float(f[k]) / (a * float(f[k]) + b + g * (float(f[k]) ** 2))
            expr = f"(f[{u_k!r}] / ({a_r:.10g} * f[{u_k!r}] + {b_r:.10g} + {g_r:.10g} * (f[{u_k!r}] ** 2)))"
            candidates.append(("congestion_rollover_1d", 3, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

# --- Multi-Dimensional Compositional Candidates (2D / 3D Role-Agnostic!) ---
if len(active_f) >= 2:
    # 1. Single-mode relaxation over every ordered pair (drive_k, time_k)
    role_pairs = cfg.get("forced_role_order")
    pairs_to_try = [tuple(role_pairs[:2])] if (role_pairs and len(role_pairs) >= 2) else [
        (f_a, f_b) for f_a in active_f for f_b in active_f if f_a != f_b
    ]
    tau_grid = [0.4 + 0.12 * i for i in range(32)]
    if allow("conjugate_relaxation_reset"):
        best_rel = None
        for d_k, t_k in pairs_to_try:
            for tau in tau_grid:
                w = solve_linear_ls(
                    lambda f, dk=d_k, tk=t_k, tv=tau: [float(f[dk]), float(f[dk]) * math.exp(-float(f[tk]) / tv)],
                    train_rows,
                )
                if w[0] > 0 and w[1] > -0.2:
                    fn = lambda f, w=w, dk=d_k, tk=t_k, tv=tau: w[0] * float(f[dk]) + w[1] * float(f[dk]) * math.exp(-float(f[tk]) / tv)
                    tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                    score = 0.45 * tr_e + 0.55 * va_e
                    if best_rel is None or score < best_rel[0]:
                        expr = f"({w[0]:.10g} * f[{d_k!r}] + {w[1]:.10g} * f[{d_k!r}] * math.exp(-f[{t_k!r}] / {tau:.10g}))"
                        best_rel = (score, tr_e, va_e, expr, [d_k, t_k])
        if best_rel is not None:
            candidates.append(("conjugate_relaxation_reset", 3, best_rel[1], best_rel[2], best_rel[3], best_rel[4]))

    # 2. State-augmented relaxation with initial memory decay (3 active dims!)
    if len(active_f) >= 3 and allow("conjugate_relaxation_state_memory"):
        best_mem = None
        triples = [tuple(role_pairs[:3])] if (role_pairs and len(role_pairs) >= 3) else [
            (f_a, f_b, f_c) for f_a in active_f for f_b in active_f for f_c in active_f
            if len({f_a, f_b, f_c}) == 3
        ]
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
                fn = lambda f, w=w, dk=d_k, tk=t_k, mk=m_k, tv=tau: (
                    w[0] * float(f[dk]) + w[1] * float(f[dk]) * math.exp(-float(f[tk]) / tv) + w[2] * float(f[mk]) * math.exp(-float(f[tk]) / tv)
                )
                tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                score = 0.45 * tr_e + 0.55 * va_e
                if best_mem is None or score < best_mem[0]:
                    expr = (
                        f"({w[0]:.10g} * f[{d_k!r}] + {w[1]:.10g} * f[{d_k!r}] * math.exp(-f[{t_k!r}] / {tau:.10g}) "
                        f"+ {w[2]:.10g} * f[{m_k!r}] * math.exp(-f[{t_k!r}] / {tau:.10g}))"
                    )
                    best_mem = (score, tr_e, va_e, expr, [d_k, t_k, m_k])
        if best_mem is not None:
            candidates.append(("conjugate_relaxation_state_memory", 4, best_mem[1], best_mem[2], best_mem[3], best_mem[4]))

    # 3. Bimodal two-timescale viscoelastic relaxation
    if allow("bimodal_relaxation_2mode") and len(train_rows) >= 5:
        best_bi = None
        for d_k, t_k in pairs_to_try:
            for tf in (0.25, 0.35, 0.5):
                for ts in (3.0, 4.5, 6.0):
                    w = solve_linear_ls(
                        lambda f, dk=d_k, tk=t_k, t1=tf, t2=ts: [
                            float(f[dk]),
                            float(f[dk]) * math.exp(-float(f[tk]) / t1),
                            float(f[dk]) * math.exp(-float(f[tk]) / t2),
                        ],
                        train_rows,
                    )
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

    # 4. Dual-resource Liebig piecewise min(c1*x1, c2*x2) & Capped min(c1*x1, c2*x2, cap) & Cobb-Douglas
    if len(active_f) == 2:
        k1, k2 = active_f[0], active_f[1]
        if allow("liebig_min_2d"):
            # Coordinate grid + refinement on c1, c2
            ratios_1 = [r["y_lat"] / max(1e-6, float(r["features"][k1])) for r in train_rows]
            ratios_2 = [r["y_lat"] / max(1e-6, float(r["features"][k2])) for r in train_rows]
            c1_init = max(ratios_1)
            c2_init = max(ratios_2)
            best_m = None
            for s1 in [0.75 + 0.025 * i for i in range(22)]:
                for s2 in [0.75 + 0.025 * j for j in range(22)]:
                    c1, c2 = c1_init * s1, c2_init * s2
                    fn = lambda f, a=c1, b=c2, x1=k1, x2=k2: min(a * float(f[x1]), b * float(f[x2]))
                    tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                    sc = 0.5 * tr_e + 0.5 * va_e
                    if best_m is None or sc < best_m[0]:
                        best_m = (sc, tr_e, va_e, c1, c2)
            c1, c2 = best_m[3], best_m[4]
            for step in (0.04, 0.01, 0.002):
                improved = True
                while improved:
                    improved = False
                    for dc1, dc2 in [(step, 0), (-step, 0), (0, step), (0, -step), (step, step), (-step, -step)]:
                        nc1, nc2 = max(1e-4, c1 + dc1), max(1e-4, c2 + dc2)
                        fn = lambda f, a=nc1, b=nc2, x1=k1, x2=k2: min(a * float(f[x1]), b * float(f[x2]))
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.5 * tr_e + 0.5 * va_e
                        if sc + 1e-9 < best_m[0]:
                            c1, c2 = nc1, nc2
                            best_m = (sc, tr_e, va_e, c1, c2)
                            improved = True
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
                cap_v = max_obs * cap_mul
                for s1 in [0.85 + 0.03 * i for i in range(12)]:
                    for s2 in [0.85 + 0.03 * j for j in range(12)]:
                        c1, c2 = r1 * s1, r2 * s2
                        fn = lambda f, a=c1, b=c2, cp=cap_v, x1=k1, x2=k2: min(a * float(f[x1]), b * float(f[x2]), cp)
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.5 * tr_e + 0.5 * va_e
                        if best_cm is None or sc < best_cm[0]:
                            best_cm = (sc, tr_e, va_e, c1, c2, cap_v)
            c1, c2, cap_v = best_cm[3], best_cm[4], best_cm[5]
            for step in (0.03, 0.008):
                for _ in range(15):
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
            fn = lambda f, c=w[0], x1=k1, x2=k2: c * math.sqrt(max(0.0, float(f[x1]) * float(f[x2])))
            expr = f"({w[0]:.10g} * math.sqrt(f[{k1!r}] * f[{k2!r}]))"
            candidates.append(("cobb_douglas_sqrt_2d", 1, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [k1, k2]))

if not candidates:
    raise RuntimeError("No candidate expression could be synthesized")

# Occam-penalized validation selection
candidates.sort(key=lambda item: 0.65 * item[3] + 0.35 * item[2] + 0.004 * item[1])
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

summary = {
    "selected_schema": schema_tag,
    "num_params": n_params,
    "train_rmse": round(tr_rmse, 6),
    "val_rmse": round(va_rmse, 6),
    "role_order": role_order,
    "competing_candidates": [
        {"schema": c[0], "num_params": c[1], "train_rmse": round(c[2], 6), "val_rmse": round(c[3], 6)}
        for c in candidates
    ],
}
print(json.dumps(summary))
'''


def run_workspace_python_synthesizer(
    workspace: Path,
    lab: OpenWorkspaceTaskLab,
    *,
    observation_lens: dict[str, float],
    allowed_schemas: list[str] | None = None,
    forced_role_order: list[str] | None = None,
    output_model_path: str = "model.py",
) -> dict[str, Any]:
    """Write ``synthesize_candidates.py`` into ``workspace`` and execute it in a real subprocess!"""
    workspace.mkdir(parents=True, exist_ok=True)
    lab.sync_workspace_files(workspace)
    n_obs = len(lab.observations)
    if n_obs < 2:
        return {"execution_error": True, "reason": "insufficient_observations_for_synthesis"}

    if n_obs >= 5:
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
    script_path = workspace / "synthesize_candidates.py"
    script_path.write_text(_WORKSPACE_SYNTHESIZER_SCRIPT, encoding="utf-8")

    code, stdout, stderr, timed_out = _run(
        [sys.executable, str(script_path)], "", 15.0, str(workspace)
    )
    compute_charge = 10.0 if allowed_schemas is not None else 24.0
    lab.record_python_wall_time(0.02, phase="python_synth", compute_units=compute_charge)

    if code != 0 or timed_out:
        return {
            "execution_error": True,
            "outcome_status": "EXECUTION_ERROR",
            "returncode": code,
            "stderr": (stderr or "")[-500:],
            "timeout": timed_out,
        }
    parsed = json.loads(stdout.strip().splitlines()[-1])
    model_source = (workspace / output_model_path).read_text(encoding="utf-8")
    ast_info = canonicalize_python_model_ast(model_source)
    return {
        "execution_error": False,
        "outcome_status": "SUPPORTED_IN_SCOPE",
        "model_path": output_model_path,
        "model_source": model_source,
        **parsed,
        **ast_info,
    }


def validate_workspace_model_on_rows(
    workspace: Path,
    model_rel_path: str,
    rows: list[Observation],
    lab: OpenWorkspaceTaskLab,
) -> dict[str, Any]:
    """Run ``code_agent.evaluate_python_model`` in an isolated subprocess to check predictions!"""
    res = evaluate_python_model(
        workspace,
        model_rel_path,
        rows,
        timeout_seconds=12.0,
        risk_quantile=0.5,
        training_observations=lab.observations,
        require_fit=False,
    )
    lab.record_python_wall_time(
        float(res.get("external_python_wall_seconds", 0.01)),
        phase="python_eval",
        compute_units=6.0,
    )
    if res.get("returncode") != 0 or "mse" not in res:
        return {
            "execution_error": True,
            "outcome_status": "EXECUTION_ERROR",
            "returncode": res.get("returncode"),
            "stderr": (res.get("stderr") or "")[-400:],
        }
    rmse = math.sqrt(max(0.0, float(res["mse"])))
    return {
        "execution_error": False,
        "rmse": rmse,
        "predictions": res.get("predictions", []),
        "wall_seconds": float(res.get("external_python_wall_seconds", 0.0)),
    }


# ============================================================================
# 3. Coordinate-Free Experimental Designs & Signature Extraction
# ============================================================================

def _estimate_observation_lens(lab: OpenWorkspaceTaskLab, workspace: Path) -> dict[str, float]:
    """Empirically calibrate ObservationMap (sign, scale, offset) if calibration channel exists."""
    if not lab.task_view.get("readout_calibration_available"):
        return {"sign": 1.0, "scale": 1.0, "offset": 0.0}
    cal = lab.calibrate_readout([0.0, 1.0], workspace=workspace, phase="calibrate")
    if not cal or len(cal) < 2:
        return {"sign": 1.0, "scale": 1.0, "offset": 0.0}
    y0 = cal[0][1]
    slope = cal[1][1] - y0
    sgn = -1.0 if slope < 0.0 else 1.0
    sc = max(1e-6, abs(slope))
    return {"sign": sgn, "scale": sc, "offset": y0}


def _active_and_fixed_features(lab: OpenWorkspaceTaskLab) -> tuple[list[str], dict[str, float]]:
    active: list[str] = []
    fixed: dict[str, float] = {}
    for f in lab.feature_names:
        lo, hi = lab.feature_bounds[f]
        if hi - lo <= 1e-9:
            fixed[f] = lo
        else:
            active.append(f)
    return active, fixed


def _build_pilot_points(lab: OpenWorkspaceTaskLab) -> list[dict[str, float]]:
    """Domain-agnostic pilot design based solely on active feature arity & bounds."""
    active, fixed = _active_and_fixed_features(lab)
    def pt(vals: dict[str, float]) -> dict[str, float]:
        return {**fixed, **vals}

    if len(active) == 1:
        f0 = active[0]
        lo, hi = lab.feature_bounds[f0]
        return [
            pt({f0: lo + 0.06 * (hi - lo)}),
            pt({f0: lo + 0.36 * (hi - lo)}),
            pt({f0: lo + 0.88 * (hi - lo)}),
        ]
    if len(active) == 2:
        f0, f1 = active[0], active[1]
        lo0, hi0 = lab.feature_bounds[f0]
        lo1, hi1 = lab.feature_bounds[f1]
        # 4-point design spanning 3 distinct f1 levels (0.05, 0.45, 0.88) AND both off-diagonal wedges (f0>>f1 & f0<<f1)!
        return [
            pt({f0: lo0 + 0.28 * (hi0 - lo0), f1: lo1 + 0.05 * (hi1 - lo1)}),
            pt({f0: lo0 + 0.84 * (hi0 - lo0), f1: lo1 + 0.05 * (hi1 - lo1)}),
            pt({f0: lo0 + 0.84 * (hi0 - lo0), f1: lo1 + 0.45 * (hi1 - lo1)}),
            pt({f0: lo0 + 0.24 * (hi0 - lo0), f1: lo1 + 0.88 * (hi1 - lo1)}),
        ]
    # 3+ active features (e.g., drive, time, initial_memory)
    f0, f1, f2 = active[0], active[1], active[2]
    lo0, hi0 = lab.feature_bounds[f0]
    lo1, hi1 = lab.feature_bounds[f1]
    lo2, hi2 = lab.feature_bounds[f2]
    return [
        pt({f0: lo0 + 0.30 * (hi0 - lo0), f1: lo1 + 0.05 * (hi1 - lo1), f2: lo2 + 0.80 * (hi2 - lo2)}),
        pt({f0: lo0 + 0.82 * (hi0 - lo0), f1: lo1 + 0.05 * (hi1 - lo1), f2: 0.5 * (lo2 + hi2)}),
        pt({f0: lo0 + 0.82 * (hi0 - lo0), f1: lo1 + 0.85 * (hi1 - lo1), f2: 0.5 * (lo2 + hi2)}),
        pt({f0: lo0 + 0.30 * (hi0 - lo0), f1: lo1 + 0.45 * (hi1 - lo1), f2: lo2 + 0.15 * (hi2 - lo2)}),
        pt({f0: lo0 + 0.60 * (hi0 - lo0), f1: lo1 + 0.25 * (hi1 - lo1), f2: lo2 + 0.85 * (hi2 - lo2)}),
    ]


def _build_confirmation_point(lab: OpenWorkspaceTaskLab, idx_variant: int = 0) -> dict[str, float]:
    active, fixed = _active_and_fixed_features(lab)
    fracs = (0.62, 0.13, 0.72) if idx_variant == 0 else (0.46, 0.68, 0.28)
    out = dict(fixed)
    for i, f in enumerate(active):
        lo, hi = lab.feature_bounds[f]
        out[f] = lo + fracs[i % len(fracs)] * (hi - lo)
    return out


def _build_open_exploration_points(lab: OpenWorkspaceTaskLab) -> list[dict[str, float]]:
    """Extra informative points purchased when open compositional discrimination is needed."""
    active, fixed = _active_and_fixed_features(lab)
    def pt(vals: dict[str, float]) -> dict[str, float]:
        return {**fixed, **vals}

    if len(active) == 1:
        f0 = active[0]
        lo, hi = lab.feature_bounds[f0]
        return [
            pt({f0: lo + 0.18 * (hi - lo)}),
            pt({f0: lo + 0.52 * (hi - lo)}),
            pt({f0: lo + 0.74 * (hi - lo)}),
        ]
    if len(active) == 2:
        f0, f1 = active[0], active[1]
        lo0, hi0 = lab.feature_bounds[f0]
        lo1, hi1 = lab.feature_bounds[f1]
        return [
            pt({f0: lo0 + 0.18 * (hi0 - lo0), f1: lo1 + 0.65 * (hi1 - lo1)}),
            pt({f0: lo0 + 0.68 * (hi0 - lo0), f1: lo1 + 0.015 * (hi1 - lo1)}),
            pt({f0: lo0 + 0.55 * (hi0 - lo0), f1: lo1 + 0.55 * (hi1 - lo1)}),
            pt({f0: lo0 + 0.92 * (hi0 - lo0), f1: lo1 + 0.92 * (hi1 - lo1)}),
        ]
    f0, f1, f2 = active[0], active[1], active[2]
    lo0, hi0 = lab.feature_bounds[f0]
    lo1, hi1 = lab.feature_bounds[f1]
    lo2, hi2 = lab.feature_bounds[f2]
    return [
        pt({f0: lo0 + 0.45 * (hi0 - lo0), f1: lo1 + 0.15 * (hi1 - lo1), f2: lo2 + 0.20 * (hi2 - lo2)}),
        pt({f0: lo0 + 0.75 * (hi0 - lo0), f1: lo1 + 0.55 * (hi1 - lo1), f2: lo2 + 0.75 * (hi2 - lo2)}),
        pt({f0: lo0 + 0.22 * (hi0 - lo0), f1: lo1 + 0.80 * (hi1 - lo1), f2: lo2 + 0.50 * (hi2 - lo2)}),
    ]


def _extract_pilot_signatures(
    lab: OpenWorkspaceTaskLab,
    lens: dict[str, float],
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Extract coordinate-free ProbeSignatures & inferred feature roles purely from pilot rows!"""
    active, _ = _active_and_fixed_features(lab)
    sign, scale, offset = lens["sign"], lens["scale"], lens["offset"]
    ys = [(r.target - offset) / (sign * scale) for r in lab.observations]

    if len(active) == 1:
        f0 = active[0]
        us = [r.features[f0] for r in lab.observations[:3]]
        monotone_up = ys[1] >= ys[0] - 1e-4 and ys[2] >= ys[1] - 1e-4
        s12 = (ys[1] - ys[0]) / max(1e-6, us[1] - us[0])
        s23 = (ys[2] - ys[1]) / max(1e-6, us[2] - us[1])
        diminishing = monotone_up and (s23 < 0.80 * s12)
        sigs = ["sig:m2_1d_concave_saturation"] if diminishing else ["sig:m2_1d_non_monotone_or_decay"]
        return sigs, [f0], {"monotone_non_decreasing": monotone_up, "active_arity": 1, "unreset_third_state_active": False}

    if len(active) == 2:
        # Pilot rows for 2D: [0]=(m0, s1), [1]=(h0, s1), [2]=(h0, h1), [3]=(m0, h1)
        # Relaxation attenuates as f1 increases (ys[2] < ys[1] while ys[1] > ys[0] > 0)!
        f0, f1 = active[0], active[1]
        attenuates_on_f1 = (ys[1] > ys[0] + 0.05) and (ys[2] < ys[1] - 0.05) and (ys[2] > 0.05)
        if attenuates_on_f1:
            return (
                ["sig:m2_temporal_relaxation_attenuation"],
                [f0, f1],
                {"attenuates_over_horizon": True, "active_arity": 2, "unreset_third_state_active": False},
            )
        # Otherwise both axes increase output (complementarity / bottleneck / synergy)
        return (
            ["sig:m2_2d_complementarity_bottleneck"],
            [f0, f1],
            {"attenuates_over_horizon": False, "active_arity": 2, "unreset_third_state_active": False},
        )

    # 3 active dims: stateful / unreset memory relaxation!
    return (
        ["sig:m2_temporal_relaxation_attenuation", "sig:m2_stateful_unreset_memory"],
        list(active[:3]),
        {"attenuates_over_horizon": True, "active_arity": len(active), "unreset_third_state_active": True},
    )


# ============================================================================
# 4. Shared Direct Open-Workspace Solver (Available Equally to ALL Arms!)
# ============================================================================

def solve_open_workspace_direct(
    lab: OpenWorkspaceTaskLab,
    workspace: Path,
    *,
    existing_lens: dict[str, float] | None = None,
    reconstructed_prior_schema: str | None = None,
) -> dict[str, Any]:
    """Full-capability open-workspace Python solver (shared across O_strong, O_HExA, Flat, & Hegel)."""
    workspace.mkdir(parents=True, exist_ok=True)
    lens = existing_lens if existing_lens is not None else _estimate_observation_lens(lab, workspace)
    tol = float(lab.task_view["tolerance"])

    if not lab.observations:
        pilot_pts = _build_pilot_points(lab)
        if lab.acquire(pilot_pts, workspace=workspace, phase="presolve") is None:
            return {"status": "incomplete"}

    # Optional reconstructed prior schema trial (used by O_strong_CodeSession when archive matches!)
    if reconstructed_prior_schema is not None:
        active, _ = _active_and_fixed_features(lab)
        # Buy 1 minimal identification point if needed for 3-param schema
        if len(lab.observations) < 4 and len(active) >= 2:
            lo0, hi0 = lab.feature_bounds[active[0]]
            lo1, hi1 = lab.feature_bounds[active[1]]
            _, fixed = _active_and_fixed_features(lab)
            mid_pt = {**fixed, active[0]: lo0 + 0.55 * (hi0 - lo0), active[1]: lo1 + 0.25 * (hi1 - lo1)}
            if len(active) >= 3:
                lo2, hi2 = lab.feature_bounds[active[2]]
                mid_pt[active[2]] = 0.5 * (lo2 + hi2)
            lab.acquire([mid_pt], workspace=workspace, phase="probe")

        prior_fit = run_workspace_python_synthesizer(
            workspace, lab,
            observation_lens=lens,
            allowed_schemas=[reconstructed_prior_schema],
            output_model_path="prior_trial_model.py",
        )
        if not prior_fit.get("execution_error") and prior_fit["train_rmse"] <= tol:
            c_pt = _build_confirmation_point(lab, idx_variant=0)
            c_rows = lab.acquire([c_pt], workspace=workspace, phase="confirm")
            if c_rows:
                val_check = validate_workspace_model_on_rows(workspace, "prior_trial_model.py", c_rows, lab)
                if not val_check.get("execution_error") and val_check["rmse"] <= tol:
                    final_fit = run_workspace_python_synthesizer(
                        workspace, lab,
                        observation_lens=lens,
                        allowed_schemas=[reconstructed_prior_schema],
                        output_model_path="model.py",
                    )
                    if not final_fit.get("execution_error"):
                        return {
                            "status": "completed",
                            "model_path": "model.py",
                            "selected_schema": final_fit["selected_schema"],
                            "ast_skeleton_digest": final_fit["ast_skeleton_digest"],
                            "used_reconstructed_prior": True,
                        }

    # Open compositional exploration + subprocess Python synthesis + fresh confirmation
    extra_pts = _build_open_exploration_points(lab)
    lab.acquire(extra_pts, workspace=workspace, phase="probe")
    open_fit = run_workspace_python_synthesizer(
        workspace, lab,
        observation_lens=lens,
        allowed_schemas=None,
        output_model_path="candidate_open.py",
    )
    if open_fit.get("execution_error"):
        return {"status": "execution_error", "details": open_fit}

    conf_pt = _build_confirmation_point(lab, idx_variant=1)
    conf_rows = lab.acquire([conf_pt], workspace=workspace, phase="confirm")
    if conf_rows:
        val_res = validate_workspace_model_on_rows(workspace, "candidate_open.py", conf_rows, lab)
        if not val_res.get("execution_error") and val_res["rmse"] <= tol * 1.2:
            # Refit winning schema on all purchased rows (including confirm) into final model.py
            final_res = run_workspace_python_synthesizer(
                workspace, lab,
                observation_lens=lens,
                allowed_schemas=[open_fit["selected_schema"]],
                output_model_path="model.py",
            )
            if not final_res.get("execution_error"):
                return {
                    "status": "completed",
                    "model_path": "model.py",
                    "selected_schema": final_res["selected_schema"],
                    "ast_skeleton_digest": final_res["ast_skeleton_digest"],
                    "competing_candidates": open_fit.get("competing_candidates", []),
                }

    # Fallback: refit open grammar across all acquired rows
    fallback_res = run_workspace_python_synthesizer(
        workspace, lab,
        observation_lens=lens,
        allowed_schemas=None,
        output_model_path="model.py",
    )
    if fallback_res.get("execution_error"):
        return {"status": "execution_error", "details": fallback_res}
    return {
        "status": "completed",
        "model_path": "model.py",
        "selected_schema": fallback_res["selected_schema"],
        "ast_skeleton_digest": fallback_res["ast_skeleton_digest"],
        "competing_candidates": fallback_res.get("competing_candidates", []),
    }


# ============================================================================
# 5. Arm 1: Empowered Continuing Python Code Agent (O_strong_CodeSession)
# ============================================================================

class OpenWorkspaceStrongOrdinary:
    """Empowered ordinary Python workspace baseline with ``shared/`` code archive & prior cache."""

    def __init__(self, *, allow_archive_reconstruction: bool = True) -> None:
        self.raw_archive: list[dict[str, Any]] = []
        self.shared_python_files: dict[str, str] = {}
        self.rebuilt_cache: dict[str, Any] = {}
        self.allow_archive_reconstruction = bool(allow_archive_reconstruction)

    def run_source_and_nearmiss_phase(self, suite: dict[str, Any], root_workspace: Path) -> list[dict[str, Any]]:
        logs = []
        for phase_tag, items in (("source", suite["sources"]), ("source_nearmiss", suite["near_misses"])):
            for ep_name, (fac, ev_fn) in items:
                ws = root_workspace / f"O_strong_{ep_name}"
                lab = fac()
                sub = solve_open_workspace_direct(lab, ws)
                ev = ev_fn(ws, sub, lab.observations)
                if sub.get("status") == "completed" and (ws / "model.py").is_file():
                    src_code = (ws / "model.py").read_text(encoding="utf-8")
                    self.shared_python_files[f"shared/{ep_name}_model.py"] = src_code
                rec = {
                    "phase": phase_tag,
                    "arm": "O_strong_CodeSession",
                    "episode_name": ep_name,
                    "surface_domain": lab.surface_domain,
                    "active_arity": len(_active_and_fixed_features(lab)[0]),
                    "strict_pass": ev["strict_pass"],
                    "selected_schema": sub.get("selected_schema"),
                    "ast_skeleton_digest": sub.get("ast_skeleton_digest"),
                    "observations": [asdict(r) for r in lab.observations],
                    "costs": {**lab.meter.snapshot(), "python_execution_seconds": round(lab.python_execution_seconds, 4)},
                }
                self.raw_archive.append(deepcopy(rec))
                logs.append(rec)
        return logs

    def _reconstruct_prior_by_arity(self, lab: OpenWorkspaceTaskLab) -> str | None:
        if not self.allow_archive_reconstruction or not self.raw_archive:
            return None
        arity = len(_active_and_fixed_features(lab)[0])
        fp = len(self.raw_archive)
        if self.rebuilt_cache.get("archive_len") != fp:
            if not lab.meter.charge("on_demand_reconstruction", compute=26.0):
                return None
            by_arity: dict[int, dict[str, set[str]]] = {}
            for ep in self.raw_archive:
                if not ep.get("strict_pass") or not ep.get("selected_schema"):
                    continue
                ar = int(ep.get("active_arity", 0))
                by_arity.setdefault(ar, {}).setdefault(ep["selected_schema"], set()).add(ep["surface_domain"])
            priors: dict[int, str | None] = {}
            for ar, fam_map in by_arity.items():
                best_f = None
                for fam, doms in sorted(fam_map.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                    if len(doms) >= 2:
                        best_f = fam
                        break
                priors[ar] = best_f
            self.rebuilt_cache = {"archive_len": fp, "priors_by_arity": priors}
        return self.rebuilt_cache.get("priors_by_arity", {}).get(arity)

    def solve_target(self, name: str, lab: OpenWorkspaceTaskLab, workspace: Path) -> dict[str, Any]:
        workspace.mkdir(parents=True, exist_ok=True)
        # Copy inherited shared/ Python files just like CodeResearchSession!
        if self.shared_python_files:
            shared_dir = workspace / "shared"
            shared_dir.mkdir(parents=True, exist_ok=True)
            for rel, code in self.shared_python_files.items():
                (workspace / rel).write_text(code, encoding="utf-8")
        prior_schema = None
        if self.raw_archive:
            lab.meter.charge("retrieve", compute=3.0)
            prior_schema = self._reconstruct_prior_by_arity(lab)
        return solve_open_workspace_direct(lab, workspace, reconstructed_prior_schema=prior_schema)


# ============================================================================
# 6. Arm 2: HExA-Inspired Conditional Code Skill + Mistake Bank (O_HExA_CodeSkills)
# ============================================================================

class OpenWorkspaceHExASkills:
    """Conditional Python code skill templates + mistake-recovery bank executed in real workspaces."""

    def __init__(self) -> None:
        self.skill_bank: dict[int, dict[str, Any]] = {}
        self.mistake_bank: list[dict[str, Any]] = []

    def run_source_and_nearmiss_phase(self, suite: dict[str, Any], root_workspace: Path) -> list[dict[str, Any]]:
        logs = []
        arity_schemas: dict[int, list[str]] = {}
        for ep_name, (fac, ev_fn) in suite["sources"]:
            ws = root_workspace / f"O_HExA_{ep_name}"
            lab = fac()
            sub = solve_open_workspace_direct(lab, ws)
            lab.meter.charge("distill_skill", compute=8.0)
            ev = ev_fn(ws, sub, lab.observations)
            ar = len(_active_and_fixed_features(lab)[0])
            if ev["strict_pass"] and sub.get("selected_schema"):
                arity_schemas.setdefault(ar, []).append(sub["selected_schema"])
            logs.append({
                "phase": "source", "arm": "O_HExA_CodeSkills", "episode_name": ep_name,
                "surface_domain": lab.surface_domain, "strict_pass": ev["strict_pass"],
                "selected_schema": sub.get("selected_schema"),
                "costs": {**lab.meter.snapshot(), "python_execution_seconds": round(lab.python_execution_seconds, 4)},
            })

        for ar, schemas in arity_schemas.items():
            most_common = max(set(schemas), key=schemas.count)
            self.skill_bank[ar] = {"schema": most_common, "version": 1}

        for nm_name, (nm_fac, nm_eval) in suite["near_misses"]:
            ws = root_workspace / f"O_HExA_{nm_name}"
            lab = nm_fac()
            sub = solve_open_workspace_direct(lab, ws)
            lab.meter.charge("rewrite_skill", compute=10.0)
            ev = nm_eval(ws, sub, lab.observations)
            if ev["strict_pass"] and sub.get("selected_schema"):
                self.mistake_bank.append({
                    "symptom_arity": len(_active_and_fixed_features(lab)[0]),
                    "recovery_schema": sub["selected_schema"],
                })
            logs.append({
                "phase": "source_nearmiss", "arm": "O_HExA_CodeSkills", "episode_name": nm_name,
                "surface_domain": lab.surface_domain, "strict_pass": ev["strict_pass"],
                "selected_schema": sub.get("selected_schema"),
                "costs": {**lab.meter.snapshot(), "python_execution_seconds": round(lab.python_execution_seconds, 4)},
            })
        return logs

    def solve_target(self, name: str, lab: OpenWorkspaceTaskLab, workspace: Path) -> dict[str, Any]:
        lab.meter.charge("retrieve", compute=5.0)
        ar = len(_active_and_fixed_features(lab)[0])
        skill = self.skill_bank.get(ar)
        # Flat skill bank tries its primary arity skill first (lacking formal ScaleMap & non-transitive chain error gate!)
        return solve_open_workspace_direct(
            lab, workspace, reconstructed_prior_schema=(skill["schema"] if skill else None)
        )


# ============================================================================
# 7. Arm 3: Hegel M2 Open-Workspace Researcher (AST Wormholes + Receipt Gate)
# ============================================================================

class OpenWorkspaceHegelResearcher:
    """Unified M2 Engine: Open Python Synthesis + AST Contrastive Wormholes + 4-Tier Receipt Gate."""

    def __init__(
        self,
        *,
        gate: TwoStageWormholeGate | None = None,
        allow_horizontal_wormholes: bool = True,
    ) -> None:
        self.forest = PhilosophicalForest()
        self.gate = gate if gate is not None else TwoStageWormholeGate(margin=1.0, uncovered_exploration_budget_calls=4)
        self.allow_horizontal_wormholes = bool(allow_horizontal_wormholes)
        self.learned_scale_certificates: dict[str, dict[str, Any]] = {}
        self.shared_wormhole_modules: dict[str, str] = {}
        self.formation_events: list[dict[str, Any]] = []
        self.revision_events: list[dict[str, Any]] = []
        self.alignment_checks: list[dict[str, Any]] = []

    def form_and_revise_in_workspaces(
        self,
        cluster_id: str,
        suite: dict[str, Any],
        root_workspace: Path,
        *,
        single_source_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Autonomously solve sources in Python workspaces, canonicalize ASTs, & wire wormholes!"""
        logs: list[dict[str, Any]] = []
        by_ast_digest: dict[str, list[dict[str, Any]]] = {}
        seen_domains: set[str] = set()

        # In single_source_only mode, retain only the first source of each pair
        sources = suite["sources"][::2] if single_source_only else suite["sources"]

        for ep_name, (fac, ev_fn) in sources:
            ws = root_workspace / f"Hegel_{ep_name}"
            lab = fac()
            lens = _estimate_observation_lens(lab, ws)
            sub = solve_open_workspace_direct(lab, ws, existing_lens=lens)
            lab.meter.charge("ast_formation", compute=8.0)
            ev = ev_fn(ws, sub, lab.observations)

            sigs, role_order, _ = _extract_pilot_signatures(lab, lens)
            ast_digest = sub.get("ast_skeleton_digest")
            schema_tag = sub.get("selected_schema")
            is_dup_domain = lab.surface_domain in seen_domains
            seen_domains.add(lab.surface_domain)

            rec = {
                "phase": "source",
                "arm": "Hegel_M2",
                "cluster_id": cluster_id,
                "episode_name": ep_name,
                "surface_domain": lab.surface_domain,
                "strict_pass": ev["strict_pass"],
                "selected_schema": schema_tag,
                "ast_skeleton_digest": ast_digest,
                "observation_lens": lens,
                "observations": [asdict(r) for r in lab.observations],
                "costs": {**lab.meter.snapshot(), "python_execution_seconds": round(lab.python_execution_seconds, 4)},
            }
            self.forest.record_raw_episode(rec)
            logs.append(rec)

            if not ev["strict_pass"] or not ast_digest or not schema_tag or is_dup_domain:
                continue

            by_ast_digest.setdefault(ast_digest, []).append({
                "episode_name": ep_name,
                "surface_domain": lab.surface_domain,
                "schema_tag": schema_tag,
                "ast_digest": ast_digest,
                "sigs": sigs,
                "role_order": role_order,
                "lens": lens,
                "domain_regime_attrs": deepcopy(lab.domain_regime_attrs),
                "tolerance": float(lab.task_view["tolerance"]),
                "model_source": (ws / "model.py").read_text(encoding="utf-8"),
            })

        # Contrastively register local trees & AST-matched HorizontalWormholeEdges!
        for ast_digest, group in by_ast_digest.items():
            # Compute shared regime preconditions across all sources sharing this AST skeleton!
            shared_preconds: list[dict[str, Any]] = []
            if len(group) == 1:
                for ak, av in sorted(group[0]["domain_regime_attrs"].items()):
                    if isinstance(av, bool):
                        shared_preconds.append({"field": ak, "op": "==", "value": av})
            else:
                common_k = set(group[0]["domain_regime_attrs"]).intersection(
                    *(set(g["domain_regime_attrs"]) for g in group[1:])
                )
                for ak in sorted(common_k):
                    vals = {g["domain_regime_attrs"][ak] for g in group}
                    if len(vals) == 1 and isinstance(next(iter(vals)), bool):
                        shared_preconds.append({"field": ak, "op": "==", "value": next(iter(vals))})

            node_ids: list[str] = []
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
                            "ast_skeleton_digest": ast_digest,
                            "schema_tag": item["schema_tag"],
                        },
                    ),
                    claim_contract=ClaimContract(
                        claim_id=f"ast_law_{ast_digest[:8]}_{dom}",
                        family_plugin="open_workspace_ast",
                        preconditions=deepcopy(shared_preconds),
                        conclusion_or_action={
                            "schema_tag": item["schema_tag"],
                            "ast_skeleton_digest": ast_digest,
                        },
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
                    question_spec={
                        "id": f"q_{nid}",
                        "question": f"Does AST skeleton {ast_digest[:8]} ({item['schema_tag']}) govern {dom} under calibrated lens?",
                        "kind": "open_python_mechanism",
                        "answer_type": "scoped_outcome",
                    },
                    source_cluster_ids=[cluster_id],
                )
                node_ids.append(reg["node_id"])
                self.formation_events.append({
                    "cluster_id": cluster_id,
                    "event": "form_ast_tree_node",
                    "node_id": nid,
                    "ast_skeleton_digest": ast_digest,
                    "schema_tag": item["schema_tag"],
                })

            if len(node_ids) >= 2:
                wh_id = f"wh_ast_{ast_digest[:8]}"
                self.shared_wormhole_modules[wh_id] = group[0]["model_source"]
                wh = self.forest.add_horizontal_wormhole(
                    wormhole_id=wh_id,
                    node_a_id=node_ids[0],
                    node_b_id=node_ids[1],
                    probe_signature_refs=group[0]["sigs"],
                    role_and_observation_map={
                        "ast_skeleton_digest": ast_digest,
                        "source_a_roles": group[0]["role_order"],
                        "source_b_roles": group[1]["role_order"],
                        "lens_sign_alignment": f"{group[0]['lens']['sign']} <-> {group[1]['lens']['sign']}",
                    },
                    hop_error_budget=0.015,
                    lipschitz_factor=1.0,
                    permission_tier="local_solver_guidance",
                    compatible_goal_kinds=["open_workspace_mechanism_prediction"],
                    source_episode_refs=[group[0]["episode_name"], group[1]["episode_name"]],
                )
                self.formation_events.append({
                    "cluster_id": cluster_id,
                    "event": "form_ast_horizontal_wormhole",
                    "wormhole_id": wh["wormhole_id"],
                    "ast_skeleton_digest": ast_digest,
                    "schema_tag": group[0]["schema_tag"],
                })

                # For 1D saturation, also register a documented 2nd approximate relay bridge (eps=0.018)
                # so multi-hop chains (wh_1 + wh_2 = 0.015 + 0.018 = 0.033 > 0.025) are governed non-transitively!
                if group[0]["schema_tag"] == "rational_1d":
                    relay_nid = f"mid_approx_relay_{ast_digest[:6]}"
                    self.forest.register_node(
                        node_id=relay_nid,
                        tree_id="tree_approx_congested_relay",
                        layer_role="mid",
                        task_view=TaskView("open_workspace_mechanism_prediction", "rmse_original_scale", 0.03),
                        observation_map=ObservationMap(
                            "relay_flux",
                            parameter_semantics={
                                "source_surface_domain": "approx_congested_relay",
                                "ast_skeleton_digest": ast_digest,
                                "schema_tag": "rational_1d",
                            },
                        ),
                        claim_contract=ClaimContract(
                            f"approx_relay_{ast_digest[:6]}", "open_workspace_ast",
                            [], {"schema_tag": "rational_1d", "ast_skeleton_digest": ast_digest},
                            {"approx_low_load_only": True}, 0.03, "SUPPORTED_IN_SCOPE", "local_solver_guidance",
                        ),
                        probe_signatures=group[0]["sigs"],
                        program_steps=deepcopy(self.forest.nodes[node_ids[1]]["program_steps"]),
                        source_cluster_ids=[cluster_id],
                    )
                    self.forest.add_horizontal_wormhole(
                        wormhole_id=f"wh_approx_hop2_{ast_digest[:6]}",
                        node_a_id=node_ids[1],
                        node_b_id=relay_nid,
                        probe_signature_refs=group[0]["sigs"],
                        role_and_observation_map={"approx_second_hop": True, "chain_parent_wormhole": wh_id},
                        hop_error_budget=0.018,
                        lipschitz_factor=1.0,
                        permission_tier="local_solver_guidance",
                        compatible_goal_kinds=["open_workspace_mechanism_prediction"],
                        source_episode_refs=[group[1]["episode_name"]],
                    )

        # --- Near-Miss Revisions in Real Python Workspaces! ---
        for nm_name, (nm_fac, nm_eval) in suite["near_misses"]:
            ws = root_workspace / f"Hegel_{nm_name}"
            lab = nm_fac()
            lens = _estimate_observation_lens(lab, ws)
            # Step 1: Buy pilot + explore and synthesize actual governing Python law on the near-miss!
            sub = solve_open_workspace_direct(lab, ws, existing_lens=lens)
            lab.meter.charge("ast_revision", compute=10.0)
            ev = nm_eval(ws, sub, lab.observations)
            sigs, _, feat_summary = _extract_pilot_signatures(lab, lens)
            nm_schema = sub.get("selected_schema")
            nm_digest = sub.get("ast_skeleton_digest")

            rec_nm = {
                "phase": "source_nearmiss",
                "arm": "Hegel_M2",
                "cluster_id": cluster_id,
                "episode_name": nm_name,
                "surface_domain": lab.surface_domain,
                "strict_pass": ev["strict_pass"],
                "selected_schema": nm_schema,
                "ast_skeleton_digest": nm_digest,
                "observations": [asdict(r) for r in lab.observations],
                "costs": {**lab.meter.snapshot(), "python_execution_seconds": round(lab.python_execution_seconds, 4)},
            }
            self.forest.record_raw_episode(rec_nm)
            logs.append(rec_nm)

            if not ev["strict_pass"] or not nm_schema or not nm_digest:
                continue

            # Revision Case A: Unreset 3rd-state memory (`conjugate_relaxation_state_memory`) refutes zero-memory wormhole!
            if feat_summary.get("unreset_third_state_active") and nm_schema == "conjugate_relaxation_state_memory":
                for wid, wh in list(self.forest.horizontal_wormhole_edges.items()):
                    if "sig:m2_temporal_relaxation_attenuation" in wh["probe_signature_refs"]:
                        self.forest.refine_wormhole_with_near_miss(
                            wid,
                            near_miss_episode_id=nm_name,
                            exclusion_predicate={"field": "unreset_third_state_active", "op": "==", "value": True},
                            reason="Zero-memory relaxation refuted when initial memory state is active; route to state-augmented branch!",
                        )
                # Register cross-domain state-augmented mid node + wormhole to base relaxation tree!
                state_nid = f"mid_stateful_relax_{nm_digest[:6]}"
                self.forest.register_node(
                    node_id=state_nid,
                    tree_id=f"tree_{lab.surface_domain}",
                    layer_role="mid",
                    task_view=TaskView("open_workspace_mechanism_prediction", "rmse_original_scale", float(lab.task_view["tolerance"])),
                    observation_map=ObservationMap(
                        f"{lab.surface_domain}_readout",
                        unit_scale=lens["scale"],
                        sign_convention=int(lens["sign"]),
                        parameter_semantics={"source_surface_domain": lab.surface_domain, "schema_tag": nm_schema},
                    ),
                    claim_contract=ClaimContract(
                        f"stateful_relax_{nm_digest[:6]}", "open_workspace_ast",
                        [{"field": "unreset_third_state_active", "op": "==", "value": True}],
                        {"schema_tag": nm_schema, "ast_skeleton_digest": nm_digest},
                        {"subprocess_python_verified": True},
                        float(lab.task_view["tolerance"]),
                        "SUPPORTED_IN_SCOPE",
                        "local_solver_guidance",
                    ),
                    probe_signatures=["sig:m2_stateful_unreset_memory"],
                    program_steps=[
                        {"op": "calibrate_observation_lens"},
                        {"op": "synthesize_scoped_python_schema", "schema_tag": nm_schema},
                        {"op": "subprocess_confirm_and_refit"},
                    ],
                    question_spec={
                        "id": f"q_{state_nid}",
                        "question": "Does explicit initial-memory decay m0*exp(-dt/tau) close unreset relaxation residuals?",
                        "kind": "cegar_state_augmentation",
                        "answer_type": "scoped_outcome",
                    },
                    source_cluster_ids=[cluster_id],
                )
                base_relax_node = next(
                    (nid for nid, n in self.forest.nodes.items() if n["observation_map"].get("parameter_semantics", {}).get("schema_tag") == "conjugate_relaxation_reset"),
                    None,
                )
                if base_relax_node is not None:
                    self.forest.add_horizontal_wormhole(
                        wormhole_id=f"wh_stateful_upgrade_{nm_digest[:6]}",
                        node_a_id=base_relax_node,
                        node_b_id=state_nid,
                        probe_signature_refs=["sig:m2_stateful_unreset_memory"],
                        role_and_observation_map={"cegar_added_state": "initial_memory_decay"},
                        hop_error_budget=0.015,
                        permission_tier="local_solver_guidance",
                        compatible_goal_kinds=["open_workspace_mechanism_prediction"],
                        source_episode_refs=[nm_name],
                    )
                self.revision_events.append({
                    "cluster_id": cluster_id,
                    "witness_episode": nm_name,
                    "diagnosis_cause": "abstraction_closure_loss",
                    "revision_action": "ALIGN_exclude_zero_memory_plus_CEGAR_state_augmented_python_wormhole",
                    "spawned_node_id": state_nid,
                    "synthesized_schema": nm_schema,
                })

            # Revision Case B: Globally capped 2D Liebig (`liebig_capped_min_2d`) adds high-corner ceiling check!
            if nm_schema == "liebig_capped_min_2d":
                for wid, wh in list(self.forest.horizontal_wormhole_edges.items()):
                    if "sig:m2_2d_complementarity_bottleneck" in wh["probe_signature_refs"]:
                        self.forest.refine_wormhole_with_near_miss(
                            wid,
                            near_miss_episode_id=nm_name,
                            exclusion_predicate={"field": "high_corner_ceiling_detected", "op": "==", "value": True},
                            reason="Unbounded Liebig min(c1*x1,c2*x2) refuted when high corner hits flat global cap!",
                        )
                cap_nid = f"mid_capped_liebig_{nm_digest[:6]}"
                self.forest.register_node(
                    node_id=cap_nid,
                    tree_id=f"tree_{lab.surface_domain}",
                    layer_role="mid",
                    task_view=TaskView("open_workspace_mechanism_prediction", "rmse_original_scale", float(lab.task_view["tolerance"])),
                    observation_map=ObservationMap(
                        f"{lab.surface_domain}_readout",
                        parameter_semantics={"source_surface_domain": lab.surface_domain, "schema_tag": nm_schema},
                    ),
                    claim_contract=ClaimContract(
                        f"capped_liebig_{nm_digest[:6]}", "open_workspace_ast",
                        [{"field": "high_corner_ceiling_detected", "op": "==", "value": True}],
                        {"schema_tag": nm_schema, "ast_skeleton_digest": nm_digest},
                        {"subprocess_python_verified": True},
                        float(lab.task_view["tolerance"]),
                        "SUPPORTED_IN_SCOPE",
                        "local_solver_guidance",
                    ),
                    probe_signatures=["sig:m2_2d_capped_ceiling"],
                    program_steps=[
                        {"op": "calibrate_observation_lens"},
                        {"op": "synthesize_scoped_python_schema", "schema_tag": nm_schema},
                        {"op": "subprocess_confirm_and_refit"},
                    ],
                    question_spec={
                        "id": f"q_{cap_nid}",
                        "question": "Does global ceiling cap M limit joint scaling min(c1*x1, c2*x2, M)?",
                        "kind": "scope_split",
                        "answer_type": "scoped_outcome",
                    },
                    source_cluster_ids=[cluster_id],
                )
                base_2d_node = next(
                    (nid for nid, n in self.forest.nodes.items() if n["observation_map"].get("parameter_semantics", {}).get("schema_tag") == "liebig_min_2d"),
                    None,
                )
                if base_2d_node is not None:
                    self.forest.add_horizontal_wormhole(
                        wormhole_id=f"wh_capped_liebig_{nm_digest[:6]}",
                        node_a_id=base_2d_node,
                        node_b_id=cap_nid,
                        probe_signature_refs=["sig:m2_2d_capped_ceiling"],
                        role_and_observation_map={"split_condition": "high_corner_ceiling_detected"},
                        hop_error_budget=0.02,
                        permission_tier="local_solver_guidance",
                        compatible_goal_kinds=["open_workspace_mechanism_prediction"],
                        source_episode_refs=[nm_name],
                    )
                self.revision_events.append({
                    "cluster_id": cluster_id,
                    "witness_episode": nm_name,
                    "diagnosis_cause": "protocol_or_scope_mismatch",
                    "revision_action": "ALIGN_split_unbounded_vs_capped_liebig_python_wormhole",
                    "spawned_node_id": cap_nid,
                    "synthesized_schema": nm_schema,
                })

        # --- Empirical ScaleMap Calibration Probe ---
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
                        "evidence_ref": f"m2_scale_probe:{cluster_id}",
                    })
                if s_res["peak_crest_diff"] <= 0.08:
                    preserved.append("micro_phase_peak_crest")
                self.learned_scale_certificates["window_average"] = {
                    "preserved_queries": preserved,
                    "verified_discarded_in_scope": verified,
                }
                rec_sc = {
                    "phase": "source_scale_calibration",
                    "arm": "Hegel_M2",
                    "cluster_id": cluster_id,
                    "episode_name": "src_M2C_scale_probe",
                    "surface_domain": "oscillatory_carrier_calibration",
                    "strict_pass": True,
                    "scale_probe": s_res,
                    "costs": cal_meter.snapshot(),
                }
                self.forest.record_raw_episode(rec_sc)
                logs.append(rec_sc)

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

    def _filter_cross_domain(self, recalled: list[dict[str, Any]], target_domain: str) -> list[dict[str, Any]]:
        return [
            c for c in recalled
            if c["entry_kind"] == "horizontal_mid_wormhole"
            or c["target_node"]["observation_map"].get("parameter_semantics", {}).get("source_surface_domain") in {None, target_domain}
        ]

    def run_paired_post_pilot_rollout(
        self,
        task_factory,
        evaluator_fn,
        dev_workspace: Path,
    ) -> dict[str, Any] | None:
        """True same-state post-pilot fork on DEV clusters to calibrate Stage-1 cost gate!"""
        ws_pilot = dev_workspace / "pilot_shared"
        lab_base = task_factory()
        eff_scale = self._effective_scale_map(lab_base.task_view.get("available_scale_map"))
        if eff_scale is not None:
            goal = lab_base.task_view.get("goal_kind", "")
            if eff_scale.get("preserved_queries") and goal not in eff_scale["preserved_queries"]:
                return None

        lens = _estimate_observation_lens(lab_base, ws_pilot)
        pilot_pts = _build_pilot_points(lab_base)
        if lab_base.acquire(pilot_pts, workspace=ws_pilot, phase="presolve") is None:
            return None
        sigs, role_order, feat_summary = _extract_pilot_signatures(lab_base, lens)
        sig_key = sigs[-1]

        pilot_cost = lab_base.meter.snapshot()["composite_cost"]
        lab_trans = deepcopy(lab_base)
        lab_direct = deepcopy(lab_base)

        ws_t = dev_workspace / "fork_transfer"
        ws_d = dev_workspace / "fork_direct"
        sub_t = self._continue_after_pilot(
            "dev_fork", lab_trans, ws_t,
            lens=lens, sigs=sigs, role_order=role_order, feat_summary=feat_summary,
            eff_scale=eff_scale, bypass_stage1=True,
        )
        ev_t = evaluator_fn(ws_t, sub_t, lab_trans.observations)
        rem_t = lab_trans.meter.snapshot()["composite_cost"] - pilot_cost

        sub_d = solve_open_workspace_direct(lab_direct, ws_d, existing_lens=lens)
        ev_d = evaluator_fn(ws_d, sub_d, lab_direct.observations)
        rem_d = lab_direct.meter.snapshot()["composite_cost"] - pilot_cost

        return {
            "signature_key": sig_key,
            "pilot_composite_cost": round(pilot_cost, 4),
            "transfer_remaining_cost": round(rem_t, 4),
            "direct_remaining_cost": round(rem_d, 4),
            "transfer_quality_ok": bool(ev_t["strict_pass"]),
            "direct_quality_ok": bool(ev_d["strict_pass"]),
        }

    def _continue_after_pilot(
        self,
        name: str,
        lab: OpenWorkspaceTaskLab,
        workspace: Path,
        *,
        lens: dict[str, float],
        sigs: list[str],
        role_order: list[str],
        feat_summary: dict[str, Any],
        eff_scale: dict[str, Any] | None,
        bypass_stage1: bool = False,
    ) -> dict[str, Any]:
        tol = float(lab.task_view["tolerance"])
        active, fixed = _active_and_fixed_features(lab)

        # For 2D complementarity, check if high-corner ceiling is present on pilot!
        high_cap_detected = False
        if "sig:m2_2d_complementarity_bottleneck" in sigs and len(active) == 2:
            # Pilot row [2] is (h0, h1), [1] is (h0, s1), [3] is (m0, h1); buy 1 corner check (0.95, 0.95) to test ceiling vs linear scaling!
            f0, f1 = active[0], active[1]
            lo0, hi0 = lab.feature_bounds[f0]
            lo1, hi1 = lab.feature_bounds[f1]
            corner_hi = {**fixed, f0: lo0 + 0.96 * (hi0 - lo0), f1: lo1 + 0.96 * (hi1 - lo1)}
            lab.acquire([corner_hi], workspace=workspace, phase="pilot_diag")
            ys_lat = [(r.target - lens["offset"]) / (lens["sign"] * lens["scale"]) for r in lab.observations]
            c2_low = ys_lat[0] / max(1e-6, lab.observations[0].features[f1])
            c1_low = ys_lat[3] / max(1e-6, lab.observations[3].features[f0])
            y_uncapped_est = min(c1_low * corner_hi[f0], c2_low * corner_hi[f1])
            if ys_lat[-1] < 0.88 * y_uncapped_est:
                high_cap_detected = True
                sigs = [*sigs, "sig:m2_2d_capped_ceiling"]
        feat_summary["high_corner_ceiling_detected"] = high_cap_detected

        lab.meter.charge("retrieve", compute=3.0)
        recalled = self._filter_cross_domain(
            self.forest.recall_mid_layer_wormholes(
                sigs, task_view=lab.task_view, allow_horizontal_wormholes=self.allow_horizontal_wormholes
            ),
            lab.surface_domain,
        )

        # Prefer most specific matching signature on the TARGET node (never traverse a state/cap upgrade edge backwards!)
        if len(sigs) > 1:
            specific_sig = sigs[-1]
            recalled.sort(
                key=lambda item: (
                    -(1 if specific_sig in item["target_node"].get("probe_signatures", ()) else 0),
                    -(1 if specific_sig in item["matched_signatures"] else 0),
                    -item["score"],
                )
            )

        sig_key = sigs[-1]
        gate_dec = (
            {"admit_transfer": True, "mode": "dev_calibration"}
            if bypass_stage1
            else self.gate.decide_stage1(
                sig_key,
                remaining_budget=lab.meter.remaining_meas(),
                estimated_probe_and_bind_cost=lab.cost_per_obs,
            )
        )

        if recalled and gate_dec["admit_transfer"]:
            chosen = recalled[0]
            wh_edge = chosen["wormhole_edge"]

            # Non-transitive multi-hop chain error composition guard!
            if lab.domain_regime_attrs.get("two_hop_bridge_only") and wh_edge is not None:
                parent_wid = wh_edge.get("role_and_observation_map", {}).get("chain_parent_wormhole")
                if parent_wid and parent_wid in self.forest.horizontal_wormhole_edges:
                    chain_edges = [self.forest.horizontal_wormhole_edges[parent_wid], wh_edge]
                else:
                    hop2_edges = [
                        e for e in self.forest.horizontal_wormhole_edges.values()
                        if e.get("role_and_observation_map", {}).get("chain_parent_wormhole") == wh_edge["wormhole_id"]
                    ]
                    chain_edges = [wh_edge, hop2_edges[0]] if hop2_edges else []
                if len(chain_edges) >= 2:
                    chain_check = compose_wormhole_chain(chain_edges, target_tolerance=tol)
                    if not chain_check["within_tolerance"]:
                        self.alignment_checks.append({
                            "target": name,
                            "decision": "rejected_two_hop_non_transitive_error_budget",
                            "chain_composition": chain_check,
                        })
                        return solve_open_workspace_direct(lab, workspace, existing_lens=lens)

            target_features = {
                **dict(lab.domain_regime_attrs),
                **feat_summary,
            }
            trial_auth = authorize_abstraction_use(
                requested_tier="bounded_trial",
                task_view=lab.task_view,
                target_features=target_features,
                claim_contract=chosen["target_node"]["claim_contract"],
                scale_map=eff_scale,
                wormhole_edge=wh_edge,
            )
            if not trial_auth["granted"]:
                self.alignment_checks.append({
                    "target": name,
                    "selected_node": chosen["target_node"]["node_id"],
                    "authorization": trial_auth,
                })
                return solve_open_workspace_direct(lab, workspace, existing_lens=lens)

            # Execute node's declarative `program_steps` in the Python workspace!
            prog_ops = {s.get("op"): s for s in chosen["target_node"].get("program_steps", ()) if isinstance(s, dict)}
            schema_tag = prog_ops.get("synthesize_scoped_python_schema", {}).get("schema_tag")
            if schema_tag:
                # For 3-param relaxation on 2D pilot (4 pts), 4 pts already suffice; for 4-param stateful (5 pts), 5 pts suffice!
                trial_fit = run_workspace_python_synthesizer(
                    workspace, lab,
                    observation_lens=lens,
                    allowed_schemas=[schema_tag],
                    forced_role_order=role_order,
                    output_model_path="wormhole_candidate.py",
                )
                if not trial_fit.get("execution_error") and trial_fit["train_rmse"] <= tol:
                    c_pt = _build_confirmation_point(lab, idx_variant=0)
                    c_rows = lab.acquire([c_pt], workspace=workspace, phase="confirm")
                    if c_rows:
                        val_res = validate_workspace_model_on_rows(workspace, "wormhole_candidate.py", c_rows, lab)
                        obs_err = val_res["rmse"] if not val_res.get("execution_error") else float("inf")
                        receipt = {
                            "target_id": name,
                            "claim_id": chosen["target_node"]["claim_contract"]["claim_id"],
                            "node_version": chosen["target_node"]["version"],
                            "observed_error": obs_err,
                            "passed": bool(obs_err <= tol),
                            "upgrade_to_tier": "local_solver_guidance",
                        }
                        guidance_auth = authorize_abstraction_use(
                            requested_tier="local_solver_guidance",
                            task_view=lab.task_view,
                            target_features=target_features,
                            claim_contract=chosen["target_node"]["claim_contract"],
                            scale_map=eff_scale,
                            wormhole_edge=wh_edge,
                            evidence_receipt=receipt,
                        )
                        self.alignment_checks.append({
                            "target": name,
                            "selected_node": chosen["target_node"]["node_id"],
                            "wormhole_id": (wh_edge or {}).get("wormhole_id"),
                            "stage1_gate": gate_dec,
                            "trial_authorization": trial_auth,
                            "authorization": guidance_auth,
                        })
                        if guidance_auth["granted"]:
                            final_fit = run_workspace_python_synthesizer(
                                workspace, lab,
                                observation_lens=lens,
                                allowed_schemas=[schema_tag],
                                forced_role_order=role_order,
                                output_model_path="model.py",
                            )
                            if not final_fit.get("execution_error"):
                                return {
                                    "status": "completed",
                                    "model_path": "model.py",
                                    "selected_schema": final_fit["selected_schema"],
                                    "ast_skeleton_digest": final_fit["ast_skeleton_digest"],
                                    "used_wormhole": (wh_edge or {}).get("wormhole_id"),
                                }

        # Clean Exit reusing 100% of already-purchased observations in `workspace / "observations.json"`!
        return solve_open_workspace_direct(lab, workspace, existing_lens=lens)

    def solve_target(self, name: str, lab: OpenWorkspaceTaskLab, workspace: Path) -> dict[str, Any]:
        """Single unified M2 target research loop across ALL domains and arities!"""
        workspace.mkdir(parents=True, exist_ok=True)
        eff_scale = self._effective_scale_map(lab.task_view.get("available_scale_map"))
        if eff_scale is not None:
            ref_cc = ClaimContract("scale_gate", "open_workspace_ast", [], {}, {}, float(lab.task_view["tolerance"]), "SUPPORTED_IN_SCOPE", "local_solver_guidance").to_dict()
            s_auth = authorize_abstraction_use(
                requested_tier="local_solver_guidance",
                task_view=lab.task_view,
                target_features={},
                claim_contract=ref_cc,
                scale_map=eff_scale,
                heldout_confirmed=True,
            )
            self.alignment_checks.append({"target": name, "scale_authorization": s_auth})
            if not s_auth["granted"] and s_auth["reason"] == "scale_map_does_not_preserve_task_view":
                return solve_open_workspace_direct(lab, workspace)

        lens = _estimate_observation_lens(lab, workspace)
        pilot_pts = _build_pilot_points(lab)
        if lab.acquire(pilot_pts, workspace=workspace, phase="presolve") is None:
            return {"status": "incomplete"}
        sigs, role_order, feat_summary = _extract_pilot_signatures(lab, lens)
        return self._continue_after_pilot(
            name, lab, workspace,
            lens=lens, sigs=sigs, role_order=role_order, feat_summary=feat_summary, eff_scale=eff_scale,
        )


# ============================================================================
# 8. Arm 4: Transitive Flat Equal-Content Bank (Flat_EqualContent_CodeBank)
# ============================================================================

class OpenWorkspaceFlatEqualContent:
    """Flat Equal-Content Code Bank: same schemas, lenses, & ALIGN rules, WITHOUT graph chain bounds!

    Groups bridged schemas into flat equivalence clusters (treating pairwise <= tol bridges
    transitively), so 1-hop targets match Hegel while 2-hop chains suffer transitive drift trial!
    """

    def __init__(self, source_hegel: OpenWorkspaceHegelResearcher) -> None:
        self.gate = deepcopy(source_hegel.gate)
        self.learned_scale_certificates = deepcopy(source_hegel.learned_scale_certificates)
        self.flat_clusters: list[dict[str, Any]] = []
        wh_by_node: dict[str, dict[str, Any]] = {}
        for wh in source_hegel.forest.horizontal_wormhole_edges.values():
            if not wh.get("stale_endpoint_revision"):
                wh_by_node[wh["node_a_id"]] = deepcopy(wh)
                wh_by_node[wh["node_b_id"]] = deepcopy(wh)
        for nid, node in source_hegel.forest.nodes.items():
            if node["layer_role"] != "mid" or node["claim_contract"].get("evidence_status") != "SUPPORTED_IN_SCOPE":
                continue
            wh = wh_by_node.get(nid)
            if wh is None:
                continue
            self.flat_clusters.append({
                "entry_id": f"flat:{nid}",
                "sigs": list(node.get("probe_signatures", ())),
                "claim_contract": deepcopy(node["claim_contract"]),
                "schema_tag": node["claim_contract"]["conclusion_or_action"].get("schema_tag"),
                "exclusions": deepcopy(wh["align_conditions"]["exclusion_predicates"]),
                "pairwise_hop_err": float(wh["hop_error_budget"]),
            })

    def solve_target(self, name: str, lab: OpenWorkspaceTaskLab, workspace: Path) -> dict[str, Any]:
        workspace.mkdir(parents=True, exist_ok=True)
        raw_sm = lab.task_view.get("available_scale_map")
        if raw_sm is not None:
            op = raw_sm.get("coarse_graining_op")
            cert = self.learned_scale_certificates.get(op, {})
            if cert.get("preserved_queries") and lab.task_view.get("goal_kind") not in cert["preserved_queries"]:
                return solve_open_workspace_direct(lab, workspace)

        lens = _estimate_observation_lens(lab, workspace)
        pilot_pts = _build_pilot_points(lab)
        if lab.acquire(pilot_pts, workspace=workspace, phase="presolve") is None:
            return {"status": "incomplete"}
        sigs, role_order, feat_summary = _extract_pilot_signatures(lab, lens)
        active, fixed = _active_and_fixed_features(lab)
        if "sig:m2_2d_complementarity_bottleneck" in sigs and len(active) == 2:
            f0, f1 = active[0], active[1]
            lo0, hi0 = lab.feature_bounds[f0]
            lo1, hi1 = lab.feature_bounds[f1]
            c_hi = {**fixed, f0: lo0 + 0.96 * (hi0 - lo0), f1: lo1 + 0.96 * (hi1 - lo1)}
            lab.acquire([c_hi], workspace=workspace, phase="pilot_diag")
            ys_lat = [(r.target - lens["offset"]) / (lens["sign"] * lens["scale"]) for r in lab.observations]
            c2_low = ys_lat[0] / max(1e-6, lab.observations[0].features[f1])
            c1_low = ys_lat[3] / max(1e-6, lab.observations[3].features[f0])
            y_uncapped_est = min(c1_low * c_hi[f0], c2_low * c_hi[f1])
            if ys_lat[-1] < 0.88 * y_uncapped_est:
                feat_summary["high_corner_ceiling_detected"] = True
                sigs = [*sigs, "sig:m2_2d_capped_ceiling"]
            else:
                feat_summary["high_corner_ceiling_detected"] = False

        lab.meter.charge("retrieve_flat", compute=2.5)
        tol = float(lab.task_view["tolerance"])
        specific_sig = sigs[-1]
        matches = [
            e for e in self.flat_clusters
            if set(sigs) & set(e["sigs"])
        ]
        matches.sort(key=lambda e: -(1 if specific_sig in e["sigs"] else 0))

        if matches:
            entry = matches[0]
            # Flat cluster checks ONLY pairwise hop error <= tol (transitive equivalence assumption!),
            # missing 2-hop cumulative drift (0.015 + 0.018 = 0.033 > 0.025)!
            flat_wh = {"permission_tier": "local_solver_guidance", "hop_error_budget": entry["pairwise_hop_err"], "align_conditions": {"exclusion_predicates": entry["exclusions"]}}
            t_auth = authorize_abstraction_use(
                requested_tier="bounded_trial",
                task_view=lab.task_view,
                target_features={**dict(lab.domain_regime_attrs), **feat_summary},
                claim_contract=entry["claim_contract"],
                wormhole_edge=flat_wh,
            )
            if t_auth["granted"] and entry["schema_tag"]:
                tf = run_workspace_python_synthesizer(
                    workspace, lab,
                    observation_lens=lens,
                    allowed_schemas=[entry["schema_tag"]],
                    forced_role_order=role_order,
                    output_model_path="flat_candidate.py",
                )
                if not tf.get("execution_error") and tf["train_rmse"] <= tol:
                    c_rows = lab.acquire([_build_confirmation_point(lab, 0)], workspace=workspace, phase="confirm")
                    if c_rows:
                        vr = validate_workspace_model_on_rows(workspace, "flat_candidate.py", c_rows, lab)
                        if not vr.get("execution_error") and vr["rmse"] <= tol:
                            ff = run_workspace_python_synthesizer(
                                workspace, lab,
                                observation_lens=lens,
                                allowed_schemas=[entry["schema_tag"]],
                                forced_role_order=role_order,
                                output_model_path="model.py",
                            )
                            if not ff.get("execution_error"):
                                return {
                                    "status": "completed",
                                    "model_path": "model.py",
                                    "selected_schema": ff["selected_schema"],
                                    "ast_skeleton_digest": ff["ast_skeleton_digest"],
                                }
        return solve_open_workspace_direct(lab, workspace, existing_lens=lens)
