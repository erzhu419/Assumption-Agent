"""M2.5 Open Python Workspace Synthesis, Pure-Dataflow AST Canonicalizer & Wormhole Loop (Hegel V16.9).

Upgrades V16.8 after independent red-team audit (`Hegel_M2_Audit`):
1. **Pure-Dataflow AST Canonicalizer (`F2`)**: Inlines local pure bindings inside ``predict(row)``,
   strips only verified non-zero outer ``ObservationMap`` affine lenses, preserves structural exponents
   (``x**2`` != ``x**-2``) and signs, cross-verifies ``_CORE_LAW_EXPR`` against executable ``predict``
   (flagging mismatches via unique ``model_sha256``), and rejects impure/memorizer globals.
2. **Prefix-Isolated Confirmation & Honest Status (`F5`, `F7`, `F11`)**: Freezes ``training_prefix``
   strictly BEFORE purchasing fresh confirmation rows so stateful ``fit(observations)`` never sees
   confirmation labels; binds ``evidence_receipt`` to ``target_id``, ``node_version``, ``model_sha256``,
   and finite ``observed_error <= tolerance``; returns ``status="scoped_unknown"`` (never ``completed``)
   whenever confirmation fails.
3. **Coordinate-Permutation Invariance & Interpreted `program_steps` (`F6`, `F8`, `F9`, `F10`)**:
   Uses $S_2$- and $S_3$-symmetric pilot designs, data-driven role alignment, real ``program_steps``
   execution, measured subprocess wall-clock timing, and empirical 2-hop source bridge residuals.
4. **Fair Equal-History Baselines**: ``O_strong_CodeSession`` retains ALL validated historical schemas
   per arity (including 1-domain & near-miss) and executes ``shared/*.py``; ``O_HExA_CodeSkills``
   uses deterministic tie-breaking and consumes ``mistake_bank``; ``Flat_EqualContent`` shares the
   exact ``Stage-1`` gate and supports both equal-metadata path-aware and transitive-cluster modes.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
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
# 1. Pure-Dataflow Python AST Canonicalizer (Self-Contained for AST Probes!)
# ============================================================================

class _SkeletonNormalizer(ast.NodeTransformer):
    """Normalize a pure core expression AST into a coordinate-free skeleton while preserving exponents."""

    _ALLOWED_CALLS = {
        "min", "max", "abs", "round",
        "math.exp", "math.log", "math.sqrt", "math.sin", "math.cos", "math.tanh",
        "exp", "log", "sqrt", "sin", "cos", "tanh",
    }

    def __init__(self, feature_aliases: set[str] | None = None, param_dicts: set[str] | None = None) -> None:
        self.feature_aliases = set(feature_aliases or {"f", "features"})
        self.param_dicts = set(param_dicts or {"_PARAMS", "PARAMS"})
        self.extracted_feature_order: list[str] = []
        self._feature_to_slot: dict[str, str] = {}
        self.unsupported_reason: str | None = None

    def _slot_for(self, feat_name: str) -> str:
        if feat_name not in self._feature_to_slot:
            slot = f"ROLE_{len(self.extracted_feature_order)}"
            self.extracted_feature_order.append(feat_name)
            self._feature_to_slot[feat_name] = slot
        return self._feature_to_slot[feat_name]

    @staticmethod
    def eval_numeric_literal(node: ast.AST) -> float | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            sub = _SkeletonNormalizer.eval_numeric_literal(node.operand)
            if sub is not None:
                return sub if isinstance(node.op, ast.UAdd) else -sub
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            lv = _SkeletonNormalizer.eval_numeric_literal(node.left)
            rv = _SkeletonNormalizer.eval_numeric_literal(node.right)
            if lv is not None and rv is not None:
                try:
                    if isinstance(node.op, ast.Add):
                        return lv + rv
                    if isinstance(node.op, ast.Sub):
                        return lv - rv
                    if isinstance(node.op, ast.Mult):
                        return lv * rv
                    if isinstance(node.op, ast.Div) and abs(rv) > 1e-15:
                        return lv / rv
                    if isinstance(node.op, ast.Pow):
                        return float(lv ** rv)
                except Exception:
                    return None
        return None

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            val = float(node.value)
            if val == 0.0:
                return ast.Name(id="ZERO", ctx=ast.Load())
            if val < 0.0:
                return ast.Name(id="NEG_NUM", ctx=ast.Load())
            return ast.Name(id="POS_NUM", ctx=ast.Load())
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        num = self.eval_numeric_literal(node)
        if num is not None:
            if num == 0.0:
                return ast.Name(id="ZERO", ctx=ast.Load())
            return ast.Name(id=("NEG_NUM" if num < 0.0 else "POS_NUM"), ctx=ast.Load())
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if isinstance(node.op, ast.Pow):
            left_norm = self.visit(node.left)
            exp_val = self.eval_numeric_literal(node.right)
            if exp_val is not None:
                iv = int(round(exp_val))
                tok = f"EXP_{iv}" if abs(exp_val - iv) < 1e-9 else f"EXP_{exp_val:.6g}"
                return ast.BinOp(left=left_norm, op=ast.Pow(), right=ast.Name(id=tok, ctx=ast.Load()))
            return ast.BinOp(left=left_norm, op=ast.Pow(), right=self.visit(node.right))
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        fname = None
        if isinstance(node.func, ast.Name):
            fname = node.func.id
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            fname = f"{node.func.value.id}.{node.func.attr}"
        if fname not in self._ALLOWED_CALLS:
            self.unsupported_reason = f"unsupported_call:{fname}"
            return node
        new_args = [self.visit(a) for a in node.args]
        short_fn = fname.split(".")[-1]
        return ast.Call(func=ast.Name(id=f"FN_{short_fn}", ctx=ast.Load()), args=new_args, keywords=[])

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        slice_node = node.slice
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            key = slice_node.value
            # Direct row['features']['x']
            if (
                isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "row"
                and isinstance(node.value.slice, ast.Constant)
                and node.value.slice.value == "features"
            ):
                return ast.Name(id=self._slot_for(key), ctx=ast.Load())
            # Alias f['x']
            if isinstance(node.value, ast.Name) and node.value.id in self.feature_aliases and key != "features":
                return ast.Name(id=self._slot_for(key), ctx=ast.Load())
            # Fitted parameter dict _PARAMS['theta_0']
            if isinstance(node.value, ast.Name) and node.value.id in self.param_dicts:
                return ast.Name(id="POS_NUM", ctx=ast.Load())
        self.unsupported_reason = "unsupported_subscript"
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id.startswith(("ROLE_", "POS_NUM", "NEG_NUM", "ZERO", "EXP_", "FN_")):
            return node
        self.unsupported_reason = f"unresolved_symbol:{node.id}"
        return node


def canonicalize_python_model_ast(model_source: str) -> dict[str, Any]:
    """Parse ``model.py`` source, inline pure ``predict(row)`` dataflow, verify annotations, & hash skeleton!"""
    model_sha256 = hashlib.sha256(model_source.encode("utf-8")).hexdigest()
    tree = ast.parse(model_source)

    annot_expr: ast.AST | None = None
    explicit_schema_tag: str | None = None
    predict_fn: ast.FunctionDef | None = None
    module_consts: dict[str, ast.AST] = {}
    param_dicts: set[str] = {"_PARAMS", "PARAMS"}

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            tname = node.targets[0].id
            if tname == "_CORE_LAW_EXPR" and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                annot_expr = ast.parse(node.value.value, mode="eval").body
            elif tname == "_SCHEMA_FAMILY_TAG" and isinstance(node.value, ast.Constant):
                explicit_schema_tag = str(node.value.value)
            elif _SkeletonNormalizer.eval_numeric_literal(node.value) is not None:
                module_consts[tname] = deepcopy(node.value)
            elif isinstance(node.value, ast.Dict):
                # Distinguish parameter dicts from empty/arbitrary memorizer caches:
                # only dicts named _PARAMS / PARAMS are treated as fitted scalar coefficients.
                if tname not in param_dicts:
                    module_consts[tname] = deepcopy(node.value)
        elif isinstance(node, ast.FunctionDef) and node.name == "predict":
            predict_fn = node

    class _EnvInliner(ast.NodeTransformer):
        def __init__(self, env: dict[str, ast.AST], row_arg: str, feat_aliases: set[str]) -> None:
            self.env = env
            self.row_arg = row_arg
            self.feat_aliases = feat_aliases

        def visit_Name(self, n: ast.Name) -> ast.AST:
            if n.id in self.env:
                return deepcopy(self.env[n.id])
            return n

    def _is_row_features_expr(expr: ast.AST, row_arg: str) -> bool:
        if (
            isinstance(expr, ast.Subscript)
            and isinstance(expr.value, ast.Name)
            and expr.value.id == row_arg
            and isinstance(expr.slice, ast.Constant)
            and expr.slice.value == "features"
        ):
            return True
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Attribute)
            and isinstance(expr.func.value, ast.Name)
            and expr.func.value.id == row_arg
            and expr.func.attr == "get"
            and expr.args
            and isinstance(expr.args[0], ast.Constant)
            and expr.args[0].value == "features"
        ):
            return True
        return False

    def _contains_feature_ref(expr: ast.AST, row_arg: str, feat_aliases: set[str]) -> bool:
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant) and isinstance(sub.slice.value, str):
                k = sub.slice.value
                if isinstance(sub.value, ast.Name) and sub.value.id in feat_aliases and k != "features":
                    return True
                if _is_row_features_expr(sub.value, row_arg):
                    return True
        return False

    def _strip_outer_affine(expr: ast.AST, row_arg: str, feat_aliases: set[str]) -> ast.AST:
        cur = expr
        changed = True
        while changed:
            changed = False
            if isinstance(cur, ast.BinOp) and isinstance(cur.op, (ast.Add, ast.Sub)):
                lv = _SkeletonNormalizer.eval_numeric_literal(cur.left)
                rv = _SkeletonNormalizer.eval_numeric_literal(cur.right)
                if lv is not None and _contains_feature_ref(cur.right, row_arg, feat_aliases):
                    cur = cur.right if isinstance(cur.op, ast.Add) else ast.UnaryOp(op=ast.USub(), operand=cur.right)
                    changed = True
                    continue
                if rv is not None and _contains_feature_ref(cur.left, row_arg, feat_aliases):
                    cur = cur.left
                    changed = True
                    continue
            if isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Mult):
                lv = _SkeletonNormalizer.eval_numeric_literal(cur.left)
                rv = _SkeletonNormalizer.eval_numeric_literal(cur.right)
                if lv is not None and abs(lv) > 1e-12 and _contains_feature_ref(cur.right, row_arg, feat_aliases):
                    cur = cur.right
                    changed = True
                    continue
                if rv is not None and abs(rv) > 1e-12 and _contains_feature_ref(cur.left, row_arg, feat_aliases):
                    cur = cur.left
                    changed = True
                    continue
            if isinstance(cur, ast.UnaryOp) and isinstance(cur.op, (ast.UAdd, ast.USub)) and _contains_feature_ref(cur.operand, row_arg, feat_aliases):
                cur = cur.operand
                changed = True
        return cur

    predict_core: ast.AST | None = None
    feat_aliases: set[str] = {"f", "features"}
    impure_predict = False

    if predict_fn is not None:
        row_arg = predict_fn.args.args[0].arg if predict_fn.args.args else "row"
        env: dict[str, ast.AST] = {k: v for k, v in module_consts.items() if not isinstance(v, ast.Dict)}
        return_nodes: list[ast.AST] = []
        for stmt in predict_fn.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                var_name = stmt.targets[0].id
                if _is_row_features_expr(stmt.value, row_arg):
                    feat_aliases.add(var_name)
                else:
                    inlined_val = _EnvInliner(env, row_arg, feat_aliases).visit(deepcopy(stmt.value))
                    env[var_name] = inlined_val
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                inlined_ret = _EnvInliner(env, row_arg, feat_aliases).visit(deepcopy(stmt.value))
                return_nodes.append(inlined_ret)
            else:
                impure_predict = True
                break

        if not impure_predict and len(return_nodes) == 1:
            raw_ret = return_nodes[0]
            if _contains_feature_ref(raw_ret, row_arg, feat_aliases):
                predict_core = _strip_outer_affine(raw_ret, row_arg, feat_aliases)
            else:
                # Constant-only return (e.g. `return 100000.0`) — preserve exact constant identity!
                cval = _SkeletonNormalizer.eval_numeric_literal(raw_ret)
                if cval is not None:
                    predict_core = ast.Name(id=f"POS_NUM_{cval:.8g}", ctx=ast.Load())
                else:
                    impure_predict = True
        else:
            impure_predict = True

    if predict_fn is None and annot_expr is None:
        raise ValueError("Could not locate predict(row) or _CORE_LAW_EXPR in model.py")

    if predict_fn is not None and impure_predict:
        return {
            "ast_skeleton_digest": f"impure:{model_sha256[:12]}",
            "canonical_ast_dump": f"IMPURE_OR_STATEFUL:{model_sha256}",
            "feature_role_order": [],
            "schema_tag": explicit_schema_tag,
            "skeleton_status": "UNSUPPORTED_IMPURE_PYTHON",
            "model_sha256": model_sha256,
        }

    if predict_core is not None:
        norm_pred = _SkeletonNormalizer(feature_aliases=feat_aliases, param_dicts=param_dicts)
        pred_tree = norm_pred.visit(deepcopy(predict_core))
        ast.fix_missing_locations(pred_tree)
        if norm_pred.unsupported_reason is not None:
            return {
                "ast_skeleton_digest": f"unknown:{model_sha256[:12]}",
                "canonical_ast_dump": f"UNSUPPORTED:{norm_pred.unsupported_reason}:{model_sha256}",
                "feature_role_order": [],
                "schema_tag": explicit_schema_tag,
                "skeleton_status": "UNSUPPORTED_AST_CONSTRUCT",
                "model_sha256": model_sha256,
            }
        pred_dump = ast.dump(pred_tree, include_attributes=False)

        if annot_expr is not None:
            norm_ann = _SkeletonNormalizer(feature_aliases=feat_aliases | {"f"}, param_dicts=param_dicts)
            ann_tree = norm_ann.visit(deepcopy(annot_expr))
            ast.fix_missing_locations(ann_tree)
            ann_dump = ast.dump(ann_tree, include_attributes=False)
            if norm_ann.unsupported_reason is not None or ann_dump != pred_dump:
                return {
                    "ast_skeleton_digest": f"mismatch:{model_sha256[:12]}",
                    "canonical_ast_dump": f"ANNOTATION_MISMATCH:{pred_dump}!={ann_dump}",
                    "feature_role_order": norm_pred.extracted_feature_order,
                    "schema_tag": explicit_schema_tag,
                    "skeleton_status": "ANNOTATION_CODE_MISMATCH",
                    "model_sha256": model_sha256,
                }

        digest = hashlib.sha256(pred_dump.encode("utf-8")).hexdigest()[:14]
        return {
            "ast_skeleton_digest": digest,
            "canonical_ast_dump": pred_dump,
            "feature_role_order": norm_pred.extracted_feature_order,
            "schema_tag": explicit_schema_tag,
            "skeleton_status": "VERIFIED_PURE_AST",
            "model_sha256": model_sha256,
        }

    # Annotation-only snippet (e.g. unit test passing `_CORE_LAW_EXPR = ...` without `def predict`)
    norm_only = _SkeletonNormalizer(feature_aliases={"f", "features"}, param_dicts=param_dicts)
    only_tree = norm_only.visit(deepcopy(annot_expr))  # type: ignore[arg-type]
    ast.fix_missing_locations(only_tree)
    only_dump = ast.dump(only_tree, include_attributes=False)
    digest = hashlib.sha256(only_dump.encode("utf-8")).hexdigest()[:14]
    return {
        "ast_skeleton_digest": digest,
        "canonical_ast_dump": only_dump,
        "feature_role_order": norm_only.extracted_feature_order,
        "schema_tag": explicit_schema_tag,
        "skeleton_status": "ANNOTATION_ONLY_UNBOUND",
        "model_sha256": model_sha256,
    }


# ============================================================================
# 2. Subprocess Workspace Python Synthesizer (Controlled Grammar Tool)
# ============================================================================

_WORKSPACE_SYNTHESIZER_SCRIPT = r'''import json
import math
from pathlib import Path
import sys

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

candidates = []

def allow(tag):
    return allowed_schemas is None or tag in allowed_schemas

# --- 1D Candidates ---
if len(active_f) == 1:
    u_k = active_f[0]
    if allow("linear_1d"):
        w = solve_linear_ls(lambda f, k=u_k: [1.0, float(f[k])], train_rows)
        fn = lambda f, w=w, k=u_k: w[0] + w[1] * float(f[k])
        expr = f"({w[0]:.10g} + {w[1]:.10g} * f[{u_k!r}])"
        candidates.append(("linear_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

    if allow("rational_1d"):
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

    if allow("congestion_rollover_1d") and len(train_rows) >= 3:
        pts = [(float(r["features"][u_k]), r["y_lat"]) for r in train_rows if float(r["features"][u_k]) > 0 and r["y_lat"] > 1e-6]
        if len(pts) >= 3:
            pseudo = [{"features": {u_k: u}, "y_lat": u / y} for u, y in pts]
            w = solve_linear_ls(lambda f, k=u_k: [1.0, float(f[k]), float(f[k]) ** 2], pseudo)
            b_r, a_r, g_r = max(1e-5, w[0]), max(1e-5, w[1]), max(0.0, w[2])
            fn = lambda f, a=a_r, b=b_r, g=g_r, k=u_k: float(f[k]) / (a * float(f[k]) + b + g * (float(f[k]) ** 2))
            expr = f"(f[{u_k!r}] / ({a_r:.10g} * f[{u_k!r}] + {b_r:.10g} + {g_r:.10g} * (f[{u_k!r}] ** 2)))"
            candidates.append(("congestion_rollover_1d", 3, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

# --- Multi-Dimensional Role-Permutation-Invariant Candidates ---
if len(active_f) >= 2:
    role_pairs = cfg.get("forced_role_order")
    pairs_to_try = [tuple(role_pairs[:2])] if (role_pairs and len(role_pairs) >= 2) else [
        (f_a, f_b) for f_a in active_f for f_b in active_f if f_a != f_b
    ]
    tau_grid = [0.4 + 0.10 * i for i in range(38)]
    if allow("conjugate_relaxation_reset"):
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
                    score = 0.45 * tr_e + 0.55 * va_e
                    if best_rel is None or score < best_rel[0]:
                        expr = f"({w[0]:.10g} * f[{d_k!r}] + {w[1]:.10g} * f[{d_k!r}] * math.exp(-f[{t_k!r}] / {tau:.10g}))"
                        best_rel = (score, tr_e, va_e, expr, [d_k, t_k])
        if best_rel is not None:
            candidates.append(("conjugate_relaxation_reset", 3, best_rel[1], best_rel[2], best_rel[3], best_rel[4]))

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
                if w[0] > 0 and w[1] > 0.05 and abs(w[2]) > 0.05:
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
                    if w[0] > 0 and w[1] > 0 and w[2] > 0:
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

    if len(active_f) == 2:
        k1, k2 = active_f[0], active_f[1]
        if allow("liebig_min_2d"):
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

candidates.sort(key=lambda item: (round(0.65 * item[3] + 0.35 * item[2] + 0.004 * item[1], 9), item[1], item[0]))
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
    observations_override: list[Observation] | None = None,
) -> dict[str, Any]:
    """Write ``synthesize_candidates.py`` into ``workspace`` and execute it in a real measured subprocess!"""
    workspace.mkdir(parents=True, exist_ok=True)
    rows_to_fit = observations_override if observations_override is not None else lab.observations
    n_obs = len(rows_to_fit)
    if n_obs < 2:
        return {"execution_error": True, "outcome_status": "EXECUTION_ERROR", "reason": "insufficient_observations_for_synthesis"}

    payload = json.dumps([asdict(r) for r in rows_to_fit], ensure_ascii=False)
    (workspace / "observations.json").write_text(payload, encoding="utf-8")
    (workspace / "model_observations.json").write_text(payload, encoding="utf-8")

    if n_obs >= 6:
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

    started = time.perf_counter()
    code, stdout, stderr, timed_out = _run(
        [sys.executable, str(script_path)], "", 15.0, str(workspace)
    )
    elapsed_wall = time.perf_counter() - started
    if allowed_schemas is None:
        compute_charge = 24.0
    else:
        compute_charge = 10.0 * max(1, len(allowed_schemas))
    lab.record_python_wall_time(elapsed_wall, phase="python_synth", compute_units=compute_charge)
    # Restore canonical lab workspace files if an override view was used
    if observations_override is not None:
        lab.sync_workspace_files(workspace)

    if code != 0 or timed_out:
        return {
            "execution_error": True,
            "outcome_status": "EXECUTION_ERROR",
            "returncode": code,
            "stderr": (stderr or "")[-500:],
            "timeout": timed_out,
            "measured_wall_seconds": round(elapsed_wall, 6),
        }
    parsed = json.loads(stdout.strip().splitlines()[-1])
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


def validate_workspace_model_on_rows(
    workspace: Path,
    model_rel_path: str,
    rows: list[Observation],
    lab: OpenWorkspaceTaskLab,
    *,
    training_prefix: list[Observation] | None = None,
) -> dict[str, Any]:
    """Run ``code_agent.evaluate_python_model`` with strictly isolated pre-confirmation ``training_prefix``!"""
    if training_prefix is None:
        n_eval = len(rows)
        if n_eval > 0 and len(lab.observations) >= n_eval and lab.observations[-n_eval:] == rows:
            isolated_prefix = list(lab.observations[:-n_eval])
        else:
            eval_ids = {id(r) for r in rows}
            isolated_prefix = [r for r in lab.observations if id(r) not in eval_ids]
    else:
        isolated_prefix = list(training_prefix)

    res = evaluate_python_model(
        workspace,
        model_rel_path,
        rows,
        timeout_seconds=12.0,
        risk_quantile=0.5,
        training_observations=isolated_prefix,
        require_fit=False,
    )
    lab.record_python_wall_time(
        float(res.get("external_python_wall_seconds", 0.0)),
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
        "training_prefix_count": len(isolated_prefix),
    }


def _freeze_confirmed_submission(
    workspace: Path,
    confirmed_candidate_rel: str,
    training_prefix: list[Observation],
    *,
    selected_schema: str | None,
    ast_skeleton_digest: str | None,
    confirmation_rmse: float,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze exact confirmed ``model.py`` bytes and ``confirmed_training_observations.json`` without unconfirmed refit!"""
    src_path = workspace / confirmed_candidate_rel
    dst_path = workspace / "model.py"
    if src_path.resolve() != dst_path.resolve():
        shutil.copyfile(src_path, dst_path)
    model_bytes = dst_path.read_bytes()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    (workspace / "confirmed_training_observations.json").write_text(
        json.dumps([asdict(r) for r in training_prefix], ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "status": "completed",
        "model_path": "model.py",
        "model_sha256": model_sha256,
        "selected_schema": selected_schema,
        "ast_skeleton_digest": ast_skeleton_digest,
        "confirmation_rmse": round(float(confirmation_rmse), 6),
        "confirmed_training_prefix_count": len(training_prefix),
        **(extra_meta or {}),
    }


# ============================================================================
# 3. Coordinate-Permutation-Invariant Designs & Signature Extraction
# ============================================================================

def _estimate_observation_lens(lab: OpenWorkspaceTaskLab, workspace: Path) -> dict[str, float]:
    """Empirically calibrate ObservationMap (sign, scale, offset) if calibration channel exists."""
    if not lab.task_view.get("readout_calibration_available"):
        return {"sign": 1.0, "scale": 1.0, "offset": 0.0}
    if len(lab.calibration_readings) >= 2:
        cal = lab.calibration_readings[:2]
    else:
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


def _frac_coord(lab: OpenWorkspaceTaskLab, feat: str, val: float) -> float:
    lo, hi = lab.feature_bounds[feat]
    return (float(val) - lo) / max(1e-9, hi - lo)


def _build_pilot_points(lab: OpenWorkspaceTaskLab) -> list[dict[str, float]]:
    """Coordinate-permutation-invariant ($S_d$-symmetric) pilot design!"""
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
        # Exact S2-symmetric 5-point design: swapping (f0, f1) produces the EXACT SAME set of points!
        frac_pairs = [
            (0.06, 0.06),
            (0.86, 0.06),
            (0.06, 0.86),
            (0.86, 0.45),
            (0.45, 0.86),
        ]
        return [
            pt({f0: lo0 + a * (hi0 - lo0), f1: lo1 + b * (hi1 - lo1)})
            for a, b in frac_pairs
        ]
    # Exact S3-symmetric 6-point design across 3 active features!
    f0, f1, f2 = active[0], active[1], active[2]
    lo0, hi0 = lab.feature_bounds[f0]
    lo1, hi1 = lab.feature_bounds[f1]
    lo2, hi2 = lab.feature_bounds[f2]
    frac_triples = [
        (0.86, 0.06, 0.06),
        (0.06, 0.86, 0.06),
        (0.06, 0.06, 0.86),
        (0.86, 0.46, 0.86),
        (0.46, 0.86, 0.86),
        (0.86, 0.86, 0.46),
    ]
    return [
        pt({f0: lo0 + a * (hi0 - lo0), f1: lo1 + b * (hi1 - lo1), f2: lo2 + c * (hi2 - lo2)})
        for a, b, c in frac_triples
    ]


def _build_confirmation_point(
    lab: OpenWorkspaceTaskLab,
    idx_variant: int = 0,
    *,
    role_order: list[str] | None = None,
    symmetric_roles: bool = False,
) -> dict[str, float]:
    """Role-aligned or coordinate-symmetric fresh confirmation point."""
    active, fixed = _active_and_fixed_features(lab)
    out = dict(fixed)
    if symmetric_roles:
        sym_frac = 0.64 if idx_variant == 0 else 0.34
        for f in active:
            lo, hi = lab.feature_bounds[f]
            out[f] = lo + sym_frac * (hi - lo)
        return out

    ordered = [f for f in (role_order or []) if f in active]
    for f in active:
        if f not in ordered:
            ordered.append(f)
    fracs = (0.62, 0.13, 0.72) if idx_variant == 0 else (0.46, 0.68, 0.28)
    for i, f in enumerate(ordered):
        lo, hi = lab.feature_bounds[f]
        out[f] = lo + fracs[i % len(fracs)] * (hi - lo)
    return out


def _build_open_exploration_points(lab: OpenWorkspaceTaskLab) -> list[dict[str, float]]:
    """Coordinate-symmetric open exploration points."""
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
            pt({f0: lo0 + 0.22 * (hi0 - lo0), f1: lo1 + 0.68 * (hi1 - lo1)}),
            pt({f0: lo0 + 0.68 * (hi0 - lo0), f1: lo1 + 0.22 * (hi1 - lo1)}),
            pt({f0: lo0 + 0.55 * (hi0 - lo0), f1: lo1 + 0.55 * (hi1 - lo1)}),
            pt({f0: lo0 + 0.94 * (hi0 - lo0), f1: lo1 + 0.94 * (hi1 - lo1)}),
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


def _extract_pilot_signatures(
    lab: OpenWorkspaceTaskLab,
    lens: dict[str, float],
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Extract permutation-invariant ProbeSignatures & data-verified feature roles from pilot rows!"""
    active, _ = _active_and_fixed_features(lab)
    tol = float(lab.task_view.get("tolerance", 0.045))
    sign, scale, offset = lens["sign"], lens["scale"], lens["offset"]
    # If readout calibration was not yet executed, normalize orientation by high-vs-low magnitude trend
    raw_ys = [(r.target - offset) / (sign * scale) for r in lab.observations]
    if not lab.calibration_readings and raw_ys and sum(raw_ys) < 0.0:
        ys = [-v for v in raw_ys]
    else:
        ys = raw_ys

    if len(active) == 1:
        f0 = active[0]
        pts = sorted(zip([r.features[f0] for r in lab.observations[:3]], ys[:3]), key=lambda kv: kv[0])
        (u1, y1), (u2, y2), (u3, y3) = pts[0], pts[1], pts[2]
        monotone_up = (y2 >= y1 - 1e-4) and (y3 >= y2 - 1e-4)
        s12 = (y2 - y1) / max(1e-6, u2 - u1)
        s23 = (y3 - y2) / max(1e-6, u3 - u2)
        diminishing = monotone_up and (s23 < 0.80 * s12)
        high_load_rollover = False
        if diminishing and y1 > 1e-5 and y2 > 1e-5 and u1 > 0 and u2 > 0:
            inv_u1, inv_u2 = 1.0 / u1, 1.0 / u2
            inv_y1, inv_y2 = 1.0 / y1, 1.0 / y2
            b_2pt = (inv_y1 - inv_y2) / max(1e-9, inv_u1 - inv_u2)
            a_2pt = inv_y2 - b_2pt * inv_u2
            if a_2pt > 0 and b_2pt > 0:
                y3_rat_est = u3 / (a_2pt * u3 + b_2pt)
                if (y3_rat_est - y3) > 0.14:
                    high_load_rollover = True
        if diminishing:
            sigs = ["sig:m2_1d_concave_saturation"]
            if high_load_rollover:
                sigs.append("sig:m2_1d_high_load_rollover")
        else:
            sigs = ["sig:m2_1d_non_monotone_or_decay"]
        return (
            sigs,
            [f0],
            {
                "monotone_non_decreasing": monotone_up,
                "high_load_rollover_detected": high_load_rollover,
                "active_arity": 1,
                "unreset_third_state_active": False,
                "symmetric_roles": True,
            },
        )

    if len(active) == 2:
        f0, f1 = active[0], active[1]

        def lookup_2d(ta: float, tb: float) -> tuple[dict[str, float], float] | None:
            for obs, yv in zip(lab.observations, ys):
                qa = _frac_coord(lab, f0, obs.features[f0])
                qb = _frac_coord(lab, f1, obs.features[f1])
                if abs(qa - ta) < 0.05 and abs(qb - tb) < 0.05:
                    return obs.features, yv
            return None

        p_ll = lookup_2d(0.06, 0.06)
        p_hl = lookup_2d(0.86, 0.06)
        p_lh = lookup_2d(0.06, 0.86)
        p_hm = lookup_2d(0.86, 0.45)
        p_mh = lookup_2d(0.45, 0.86)

        if p_ll and p_hl and p_lh and p_hm and p_mh:
            # Hypothesis 0: f0 is drive, f1 is horizon (grows with f0 at low f1, attenuates with f1 at high f0)
            rel_01 = (p_hl[1] > p_ll[1] + 0.08) and (p_hm[1] < p_hl[1] - 0.08)
            # Hypothesis 1: f1 is drive, f0 is horizon (exact mirror!)
            rel_10 = (p_lh[1] > p_ll[1] + 0.08) and (p_mh[1] < p_lh[1] - 0.08)

            if rel_01 and not rel_10:
                return (
                    ["sig:m2_temporal_relaxation_attenuation"],
                    [f0, f1],
                    {"attenuates_over_horizon": True, "active_arity": 2, "unreset_third_state_active": False, "symmetric_roles": False},
                )
            if rel_10 and not rel_01:
                return (
                    ["sig:m2_temporal_relaxation_attenuation"],
                    [f1, f0],
                    {"attenuates_over_horizon": True, "active_arity": 2, "unreset_third_state_active": False, "symmetric_roles": False},
                )

            # Complementarity / bottleneck check + zero-extra-probe ceiling detection!
            c2_low = p_hl[1] / max(1e-6, p_hl[0][f1])
            c1_low = p_lh[1] / max(1e-6, p_lh[0][f0])
            uncap_hm = min(c1_low * p_hm[0][f0], c2_low * p_hm[0][f1])
            uncap_mh = min(c1_low * p_mh[0][f0], c2_low * p_mh[0][f1])
            # Also check smooth synergy vs piecewise linear min: at (ll), Liebig predicts min(c1*ll0, c2*ll1)
            uncap_ll = min(c1_low * p_ll[0][f0], c2_low * p_ll[0][f1])
            matches_low_wedge = abs(p_ll[1] - uncap_ll) <= max(0.25, 3.0 * tol)
            high_cap = matches_low_wedge and (
                (uncap_hm - p_hm[1] > max(0.35, 3.5 * tol)) or (uncap_mh - p_mh[1] > max(0.35, 3.5 * tol))
            )
            sigs = ["sig:m2_2d_complementarity_bottleneck"]
            if high_cap:
                sigs.append("sig:m2_2d_capped_ceiling")
            return (
                sigs,
                sorted([f0, f1]),
                {
                    "attenuates_over_horizon": False,
                    "high_corner_ceiling_detected": bool(high_cap),
                    "active_arity": 2,
                    "unreset_third_state_active": False,
                    "symmetric_roles": True,
                },
            )

        return (
            ["sig:m2_2d_complementarity_bottleneck"],
            sorted([f0, f1]),
            {"attenuates_over_horizon": False, "high_corner_ceiling_detected": False, "active_arity": 2, "unreset_third_state_active": False, "symmetric_roles": True},
        )

    # 3+ active dimensions: empirically verify that all 3 coordinates cause non-trivial output variation!
    coord_effects: dict[str, float] = {}
    for fk in active[:3]:
        hi_vals = [yv for r, yv in zip(lab.observations, ys) if _frac_coord(lab, fk, r.features[fk]) > 0.7]
        lo_vals = [yv for r, yv in zip(lab.observations, ys) if _frac_coord(lab, fk, r.features[fk]) < 0.2]
        if hi_vals and lo_vals:
            coord_effects[fk] = abs((sum(hi_vals) / len(hi_vals)) - (sum(lo_vals) / len(lo_vals)))
        else:
            coord_effects[fk] = 0.0
    third_active = sum(1 for v in coord_effects.values() if v > 2.0 * tol) >= 3
    sigs_3d = ["sig:m2_temporal_relaxation_attenuation"]
    if third_active:
        sigs_3d.append("sig:m2_stateful_unreset_memory")
    return (
        sigs_3d,
        [],  # Let synthesizer determine exact (drive, horizon, memory) permutation from data!
        {"attenuates_over_horizon": True, "active_arity": len(active), "unreset_third_state_active": bool(third_active), "symmetric_roles": False},
    )


# ============================================================================
# 4. Shared Direct Open-Workspace Solver (Honest Confirmation Contract!)
# ============================================================================

def solve_open_workspace_direct(
    lab: OpenWorkspaceTaskLab,
    workspace: Path,
    *,
    existing_lens: dict[str, float] | None = None,
    reconstructed_prior_schema: str | list[str] | None = None,
) -> dict[str, Any]:
    """Full-capability open-workspace solver — NEVER returns ``completed`` without passing fresh confirmation!"""
    workspace.mkdir(parents=True, exist_ok=True)
    lens = existing_lens if existing_lens is not None else _estimate_observation_lens(lab, workspace)
    tol = float(lab.task_view["tolerance"])

    if not lab.observations:
        pilot_pts = _build_pilot_points(lab)
        if lab.acquire(pilot_pts, workspace=workspace, phase="presolve") is None:
            return {"status": "incomplete", "model_path": None}

    _, _, feat_info = _extract_pilot_signatures(lab, lens)
    sym_roles = bool(feat_info.get("symmetric_roles", False))

    # Optional reconstructed prior schema(s) trial (supports single schema OR full validated schema list!)
    prior_list: list[str] = []
    if isinstance(reconstructed_prior_schema, str):
        prior_list = [reconstructed_prior_schema]
    elif isinstance(reconstructed_prior_schema, (list, tuple)):
        prior_list = [str(s) for s in reconstructed_prior_schema if s]

    if prior_list:
        prior_fit = run_workspace_python_synthesizer(
            workspace,
            lab,
            observation_lens=lens,
            allowed_schemas=prior_list,
            output_model_path="prior_trial_model.py",
        )
        if not prior_fit.get("execution_error") and prior_fit["train_rmse"] <= tol:
            fit_prefix = list(lab.observations)
            c_pt = _build_confirmation_point(
                lab,
                idx_variant=0,
                role_order=prior_fit.get("role_order"),
                symmetric_roles=sym_roles,
            )
            c_rows = lab.acquire([c_pt], workspace=workspace, phase="confirm")
            if c_rows:
                val_check = validate_workspace_model_on_rows(
                    workspace, "prior_trial_model.py", c_rows, lab, training_prefix=fit_prefix
                )
                if not val_check.get("execution_error") and val_check["rmse"] <= tol:
                    return _freeze_confirmed_submission(
                        workspace,
                        "prior_trial_model.py",
                        fit_prefix,
                        selected_schema=prior_fit["selected_schema"],
                        ast_skeleton_digest=prior_fit["ast_skeleton_digest"],
                        confirmation_rmse=val_check["rmse"],
                        extra_meta={"used_reconstructed_prior": True},
                    )

    # Open exploration + candidate synthesis + strictly prefix-isolated fresh confirmation!
    extra_pts = _build_open_exploration_points(lab)
    lab.acquire(extra_pts, workspace=workspace, phase="probe")
    open_fit = run_workspace_python_synthesizer(
        workspace,
        lab,
        observation_lens=lens,
        allowed_schemas=None,
        output_model_path="candidate_open.py",
    )
    if open_fit.get("execution_error"):
        return {"status": "execution_error", "model_path": None, "details": open_fit}

    fit_prefix_open = list(lab.observations)
    conf_pt = _build_confirmation_point(
        lab,
        idx_variant=1,
        role_order=open_fit.get("role_order"),
        symmetric_roles=sym_roles,
    )
    conf_rows = lab.acquire([conf_pt], workspace=workspace, phase="confirm")
    last_conf_rmse: float | None = None
    if conf_rows:
        val_res = validate_workspace_model_on_rows(
            workspace, "candidate_open.py", conf_rows, lab, training_prefix=fit_prefix_open
        )
        if not val_res.get("execution_error"):
            last_conf_rmse = float(val_res["rmse"])
            if last_conf_rmse <= tol:
                return _freeze_confirmed_submission(
                    workspace,
                    "candidate_open.py",
                    fit_prefix_open,
                    selected_schema=open_fit["selected_schema"],
                    ast_skeleton_digest=open_fit["ast_skeleton_digest"],
                    confirmation_rmse=last_conf_rmse,
                    extra_meta={"competing_candidates": open_fit.get("competing_candidates", [])},
                )

    # Honest exit: retain paid observations & unconfirmed candidate WITHOUT claiming `completed`!
    shutil.copyfile(workspace / "candidate_open.py", workspace / "unconfirmed_candidate.py")
    return {
        "status": "scoped_unknown",
        "model_path": None,
        "unconfirmed_candidate_path": "unconfirmed_candidate.py",
        "selected_schema": open_fit.get("selected_schema"),
        "ast_skeleton_digest": open_fit.get("ast_skeleton_digest"),
        "confirmation_rmse": (round(last_conf_rmse, 6) if last_conf_rmse is not None else None),
        "reason": "fresh_confirmation_failed_or_out_of_hypothesis_scope",
    }


# ============================================================================
# 5. Arm 1: Empowered Continuing Python Code Agent (O_strong_CodeSession)
# ============================================================================

class OpenWorkspaceStrongOrdinary:
    """Empowered ordinary Python workspace baseline retaining ALL validated historical schemas & shared/*.py!"""

    def __init__(
        self,
        *,
        allow_archive_reconstruction: bool = True,
        retain_all_validated_priors: bool = True,
    ) -> None:
        self.raw_archive: list[dict[str, Any]] = []
        self.shared_python_files: dict[str, str] = {}
        self.rebuilt_cache: dict[str, Any] = {}
        self.allow_archive_reconstruction = bool(allow_archive_reconstruction)
        self.retain_all_validated_priors = bool(retain_all_validated_priors)

    def run_source_and_nearmiss_phase(self, suite: dict[str, Any], root_workspace: Path) -> list[dict[str, Any]]:
        logs = []
        for phase_tag, items in (("source", suite["sources"]), ("source_nearmiss", suite["near_misses"])):
            for ep_name, (fac, ev_fn) in items:
                ws = root_workspace / f"O_strong_{ep_name}"
                lab = fac()
                sub = solve_open_workspace_direct(lab, ws)
                ev = ev_fn(ws, sub, lab.observations)
                if ev["strict_pass"] and sub.get("status") == "completed" and (ws / "model.py").is_file():
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
                    "costs": {
                        **lab.meter.snapshot(),
                        "python_execution_seconds": round(lab.python_execution_seconds, 4),
                        "measured_python_wall_seconds": round(lab.measured_python_wall_seconds, 4),
                    },
                }
                self.raw_archive.append(deepcopy(rec))
                logs.append(rec)
        return logs

    def _inspect_shared_python_schemas(self, workspace: Path) -> list[str]:
        """Actually parse & import inherited ``shared/*.py`` modules in ``workspace`` to extract schemas!"""
        shared_dir = workspace / "shared"
        if not shared_dir.is_dir():
            return []
        discovered: set[str] = set()
        for py_file in sorted(shared_dir.glob("*.py")):
            info = canonicalize_python_model_ast(py_file.read_text(encoding="utf-8"))
            if info.get("schema_tag") and info.get("skeleton_status") == "VERIFIED_PURE_AST":
                discovered.add(str(info["schema_tag"]))
        return sorted(discovered)

    def _reconstruct_prior_by_arity(self, lab: OpenWorkspaceTaskLab, workspace: Path) -> list[str] | None:
        if not self.allow_archive_reconstruction or not self.raw_archive:
            return None
        arity = len(_active_and_fixed_features(lab)[0])
        fp = len(self.raw_archive)
        if self.rebuilt_cache.get("archive_len") != fp:
            if not lab.meter.charge("on_demand_reconstruction", compute=26.0):
                return None
            shared_tags = set(self._inspect_shared_python_schemas(workspace))
            by_arity: dict[int, set[str]] = {}
            doms_by_arity: dict[int, dict[str, set[str]]] = {}
            for ep in self.raw_archive:
                if not ep.get("strict_pass") or not ep.get("selected_schema"):
                    continue
                ar = int(ep.get("active_arity", 0))
                tag = str(ep["selected_schema"])
                if not shared_tags or tag in shared_tags:
                    by_arity.setdefault(ar, set()).add(tag)
                    doms_by_arity.setdefault(ar, {}).setdefault(tag, set()).add(str(ep["surface_domain"]))
            if self.retain_all_validated_priors:
                priors = {ar: sorted(tags) for ar, tags in by_arity.items()}
            else:
                priors = {}
                for ar, fmap in doms_by_arity.items():
                    best = [k for k, ds in sorted(fmap.items(), key=lambda kv: (-len(kv[1]), kv[0])) if len(ds) >= 2]
                    priors[ar] = best[:1]
            self.rebuilt_cache = {"archive_len": fp, "priors_by_arity": priors}
        return self.rebuilt_cache.get("priors_by_arity", {}).get(arity) or None

    def solve_target(self, name: str, lab: OpenWorkspaceTaskLab, workspace: Path) -> dict[str, Any]:
        workspace.mkdir(parents=True, exist_ok=True)
        if self.shared_python_files:
            shared_dir = workspace / "shared"
            shared_dir.mkdir(parents=True, exist_ok=True)
            for rel, code in sorted(self.shared_python_files.items()):
                (workspace / rel).write_text(code, encoding="utf-8")
        prior_schemas = None
        if self.raw_archive:
            lab.meter.charge("retrieve", compute=3.0)
            prior_schemas = self._reconstruct_prior_by_arity(lab, workspace)
        return solve_open_workspace_direct(lab, workspace, reconstructed_prior_schema=prior_schemas)


# ============================================================================
# 6. Arm 2: HExA-Inspired Conditional Skill + Active Mistake-Bank Recovery
# ============================================================================

class OpenWorkspaceHExASkills:
    """HExA-inspired conditional Python skill bank + active mistake-recovery bank with deterministic tie-break."""

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
                arity_schemas.setdefault(ar, []).append(str(sub["selected_schema"]))
            logs.append({
                "phase": "source", "arm": "O_HExA_CodeSkills", "episode_name": ep_name,
                "surface_domain": lab.surface_domain, "strict_pass": ev["strict_pass"],
                "selected_schema": sub.get("selected_schema"),
                "costs": {
                    **lab.meter.snapshot(),
                    "python_execution_seconds": round(lab.python_execution_seconds, 4),
                    "measured_python_wall_seconds": round(lab.measured_python_wall_seconds, 4),
                },
            })

        for ar, schemas in sorted(arity_schemas.items()):
            # Deterministic tie-break independent of PYTHONHASHSEED!
            ordered_unique = sorted(set(schemas), key=lambda s: (-schemas.count(s), s))
            self.skill_bank[ar] = {
                "primary_schema": ordered_unique[0],
                "secondary_schemas": ordered_unique[1:],
                "version": 1,
            }

        for nm_name, (nm_fac, nm_eval) in suite["near_misses"]:
            ws = root_workspace / f"O_HExA_{nm_name}"
            lab = nm_fac()
            sub = solve_open_workspace_direct(lab, ws)
            lab.meter.charge("rewrite_skill", compute=10.0)
            ev = nm_eval(ws, sub, lab.observations)
            if ev["strict_pass"] and sub.get("selected_schema"):
                self.mistake_bank.append({
                    "symptom_arity": len(_active_and_fixed_features(lab)[0]),
                    "recovery_schema": str(sub["selected_schema"]),
                    "source_nearmiss": nm_name,
                })
            logs.append({
                "phase": "source_nearmiss", "arm": "O_HExA_CodeSkills", "episode_name": nm_name,
                "surface_domain": lab.surface_domain, "strict_pass": ev["strict_pass"],
                "selected_schema": sub.get("selected_schema"),
                "costs": {
                    **lab.meter.snapshot(),
                    "python_execution_seconds": round(lab.python_execution_seconds, 4),
                    "measured_python_wall_seconds": round(lab.measured_python_wall_seconds, 4),
                },
            })
        return logs

    def solve_target(self, name: str, lab: OpenWorkspaceTaskLab, workspace: Path) -> dict[str, Any]:
        workspace.mkdir(parents=True, exist_ok=True)
        lab.meter.charge("retrieve", compute=5.0)
        ar = len(_active_and_fixed_features(lab)[0])
        skill = self.skill_bank.get(ar)
        recovery_schemas = [
            m["recovery_schema"] for m in self.mistake_bank if int(m["symptom_arity"]) == ar
        ]
        ordered_priors: list[str] = []
        if skill:
            ordered_priors.append(skill["primary_schema"])
            ordered_priors.extend(skill.get("secondary_schemas", ()))
        for r_s in recovery_schemas:
            if r_s not in ordered_priors:
                ordered_priors.append(r_s)

        # Try primary skill first; on pilot residual failure, redirect via mistake_bank before open fallback!
        if ordered_priors:
            lens = _estimate_observation_lens(lab, workspace)
            tol = float(lab.task_view["tolerance"])
            if not lab.observations:
                if lab.acquire(_build_pilot_points(lab), workspace=workspace, phase="presolve") is None:
                    return {"status": "incomplete", "model_path": None}
            _, _, feat_info = _extract_pilot_signatures(lab, lens)
            sym_roles = bool(feat_info.get("symmetric_roles", False))

            for cand_idx, cand_schema in enumerate(ordered_priors):
                fit_res = run_workspace_python_synthesizer(
                    workspace,
                    lab,
                    observation_lens=lens,
                    allowed_schemas=[cand_schema],
                    output_model_path=f"hexa_cand_{cand_idx}.py",
                )
                if not fit_res.get("execution_error") and fit_res["train_rmse"] <= tol:
                    fit_prefix = list(lab.observations)
                    c_pt = _build_confirmation_point(
                        lab, idx_variant=0, role_order=fit_res.get("role_order"), symmetric_roles=sym_roles
                    )
                    c_rows = lab.acquire([c_pt], workspace=workspace, phase="confirm")
                    if c_rows:
                        vr = validate_workspace_model_on_rows(
                            workspace, f"hexa_cand_{cand_idx}.py", c_rows, lab, training_prefix=fit_prefix
                        )
                        if not vr.get("execution_error") and vr["rmse"] <= tol:
                            return _freeze_confirmed_submission(
                                workspace,
                                f"hexa_cand_{cand_idx}.py",
                                fit_prefix,
                                selected_schema=fit_res["selected_schema"],
                                ast_skeleton_digest=fit_res["ast_skeleton_digest"],
                                confirmation_rmse=vr["rmse"],
                                extra_meta={"used_mistake_recovery": cand_schema in recovery_schemas and cand_idx > 0},
                            )
                    break  # Paid confirmation failed -> proceed to open exploration reusing paid rows!
            return solve_open_workspace_direct(
            lab, workspace, existing_lens=(lens if lab.calibration_readings or not lab.task_view.get("readout_calibration_available") else None)
        )

        return solve_open_workspace_direct(lab, workspace)


# ============================================================================
# 7. Arm 3: Hegel M2.5 Open-Workspace Researcher (AST Wormholes + Bound Receipts)
# ============================================================================

def _empirical_cross_fit_rmse(
    workspace: Path,
    model_rel_path: str,
    source_rows: list[Observation],
    target_rows: list[Observation],
    *,
    source_roles: list[str] | None = None,
    target_roles: list[str] | None = None,
) -> float:
    """Measure empirical cross-episode discrepancy after role alignment & optimal linear readout scaling."""
    mapped_targets = target_rows
    if source_roles and target_roles and len(source_roles) == len(target_roles):
        mapped_targets = [
            Observation(
                features={
                    **dict(r.features),
                    **{sf: r.features[tf] for sf, tf in zip(source_roles, target_roles) if tf in r.features},
                },
                target=r.target,
                history=r.history,
            )
            for r in target_rows
        ]
    res = evaluate_python_model(
        workspace,
        model_rel_path,
        mapped_targets,
        timeout_seconds=10.0,
        risk_quantile=0.5,
        training_observations=source_rows,
        require_fit=False,
    )
    preds = [float(p) for p in res.get("predictions", ())]
    ys = [float(r.target) for r in target_rows]
    if len(preds) != len(ys) or len(ys) < 2:
        return 0.05
    mp = sum(preds) / len(preds)
    my = sum(ys) / len(ys)
    vp = sum((p - mp) ** 2 for p in preds)
    slope = sum((p - mp) * (y - my) for p, y in zip(preds, ys)) / vp if vp > 1e-12 else 1.0
    inter = my - slope * mp
    aligned_err = math.sqrt(sum(((inter + slope * p) - y) ** 2 for p, y in zip(preds, ys)) / len(ys))
    return round(float(aligned_err), 5)


class OpenWorkspaceHegelResearcher:
    """Unified M2.5 Engine: Pure-AST Wormholes + Witness Revision + Executed `program_steps` + Bound Receipts."""

    def __init__(
        self,
        *,
        gate: TwoStageWormholeGate | None = None,
        allow_horizontal_wormholes: bool = True,
        allow_cross_domain_local_without_wormhole: bool = False,
    ) -> None:
        self.forest = PhilosophicalForest()
        self.gate = gate if gate is not None else TwoStageWormholeGate(margin=1.0, uncovered_exploration_budget_calls=4)
        self.allow_horizontal_wormholes = bool(allow_horizontal_wormholes)
        self.allow_cross_domain_local_without_wormhole = bool(allow_cross_domain_local_without_wormhole)
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
        """Solve sources in real workspaces, measure empirical bridge errors, & execute witness-driven revisions!"""
        logs: list[dict[str, Any]] = []
        by_ast_digest: dict[str, list[dict[str, Any]]] = {}
        one_dim_validated: list[dict[str, Any]] = []
        seen_domains: set[str] = set()

        sources = suite["sources"][::2] if single_source_only else suite["sources"]

        for ep_name, (fac, ev_fn) in sources:
            ws = root_workspace / f"Hegel_{ep_name}"
            lab = fac()
            lens = _estimate_observation_lens(lab, ws)
            sub = solve_open_workspace_direct(lab, ws, existing_lens=lens)
            lab.meter.charge("ast_formation", compute=8.0)
            ev = ev_fn(ws, sub, lab.observations)

            sigs, sig_roles, _ = _extract_pilot_signatures(lab, lens)
            ast_digest = sub.get("ast_skeleton_digest")
            schema_tag = sub.get("selected_schema")
            is_dup_domain = lab.surface_domain in seen_domains
            seen_domains.add(lab.surface_domain)

            model_src = (ws / "model.py").read_text(encoding="utf-8") if (ws / "model.py").is_file() else ""
            ast_meta = canonicalize_python_model_ast(model_src) if model_src else {}
            verified_roles = ast_meta.get("feature_role_order") or sig_roles

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
                "costs": {
                    **lab.meter.snapshot(),
                    "python_execution_seconds": round(lab.python_execution_seconds, 4),
                    "measured_python_wall_seconds": round(lab.measured_python_wall_seconds, 4),
                },
            }
            self.forest.record_raw_episode(rec)
            logs.append(rec)

            if (
                not ev["strict_pass"]
                or not ast_digest
                or ast_meta.get("skeleton_status") != "VERIFIED_PURE_AST"
                or not schema_tag
                or is_dup_domain
            ):
                continue

            entry = {
                "episode_name": ep_name,
                "workspace": ws,
                "surface_domain": lab.surface_domain,
                "schema_tag": schema_tag,
                "ast_digest": ast_digest,
                "sigs": sigs,
                "role_order": verified_roles,
                "lens": lens,
                "domain_regime_attrs": deepcopy(lab.domain_regime_attrs),
                "tolerance": float(lab.task_view["tolerance"]),
                "confirmation_rmse": float(sub.get("confirmation_rmse") or ev.get("rmse") or 0.01),
                "eval_rmse": float(ev.get("rmse") or 0.01),
                "paid_observations": list(lab.observations),
                "model_source": model_src,
                "active_arity": len(_active_and_fixed_features(lab)[0]),
            }
            by_ast_digest.setdefault(ast_digest, []).append(entry)
            if entry["active_arity"] == 1:
                one_dim_validated.append(entry)

        # Register AST-verified nodes & empirically measured HorizontalWormholeEdges!
        node_by_episode: dict[str, str] = {}
        primary_1d_wh_id: str | None = None
        for ast_digest, group in by_ast_digest.items():
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
                node_by_episode[item["episode_name"]] = reg["node_id"]
                self.formation_events.append({
                    "cluster_id": cluster_id,
                    "event": "form_ast_tree_node",
                    "node_id": nid,
                    "ast_skeleton_digest": ast_digest,
                    "schema_tag": item["schema_tag"],
                    "verified_roles": item["role_order"],
                })

            if len(node_ids) >= 2:
                wh_id = f"wh_ast_{ast_digest[:8]}"
                self.shared_wormhole_modules[wh_id] = group[0]["model_source"]
                # Empirically measured schema-fit hop error from actual paid source observations!
                emp_hop_err = max(
                    group[0]["eval_rmse"],
                    group[1]["eval_rmse"],
                    group[0]["confirmation_rmse"],
                    group[1]["confirmation_rmse"],
                )
                wh = self.forest.add_horizontal_wormhole(
                    wormhole_id=wh_id,
                    node_a_id=node_ids[0],
                    node_b_id=node_ids[1],
                    probe_signature_refs=sorted({s for g in group[:2] for s in g["sigs"]}),
                    role_and_observation_map={
                        "ast_skeleton_digest": ast_digest,
                        "source_a_roles": group[0]["role_order"],
                        "source_b_roles": group[1]["role_order"],
                        "lens_sign_alignment": f"{group[0]['lens']['sign']} <-> {group[1]['lens']['sign']}",
                        "empirically_calibrated": True,
                    },
                    hop_error_budget=round(float(emp_hop_err), 5),
                    lipschitz_factor=1.0,
                    permission_tier="local_solver_guidance",
                    compatible_goal_kinds=["open_workspace_mechanism_prediction"],
                    source_episode_refs=[group[0]["episode_name"], group[1]["episode_name"]],
                )
                if group[0]["schema_tag"] == "rational_1d":
                    primary_1d_wh_id = wh_id
                self.formation_events.append({
                    "cluster_id": cluster_id,
                    "event": "form_ast_horizontal_wormhole",
                    "wormhole_id": wh["wormhole_id"],
                    "ast_skeleton_digest": ast_digest,
                    "schema_tag": group[0]["schema_tag"],
                    "empirical_hop_error_budget": round(float(emp_hop_err), 5),
                })

        # Empirical 2nd-hop bridge discovery across real 1D source episodes (re-fitting bridge schema on roll_src's paid rows)!
        if not single_source_only and len(one_dim_validated) >= 3 and primary_1d_wh_id is not None:
            rat_eps = [e for e in one_dim_validated if e["schema_tag"] == "rational_1d"]
            roll_eps = [e for e in one_dim_validated if e["schema_tag"] != "rational_1d"]
            if rat_eps and roll_eps:
                bridge_src = rat_eps[-1]
                roll_src = roll_eps[0]
                probe_lab = OpenWorkspaceTaskLab(
                    task_id="bridge_check",
                    surface_domain=roll_src["surface_domain"],
                    feature_names=tuple(roll_src["role_order"]),
                    feature_bounds={roll_src["role_order"][0]: (1.0, 28.0)},
                    cost_per_obs=0.0,
                    task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.03},
                    meter=CostMeter(obs_budget=50, meas_budget=100.0, compute_budget=500.0),
                    private_oracle_fn=lambda _p: 0.0,
                )
                bridge_fit = run_workspace_python_synthesizer(
                    roll_src["workspace"] / "_bridge_fit",
                    probe_lab,
                    observation_lens=roll_src["lens"],
                    allowed_schemas=[bridge_src["schema_tag"]],
                    output_model_path="bridge_approx.py",
                    observations_override=roll_src["paid_observations"],
                )
                emp_hop_23 = float(bridge_fit.get("train_rmse", 0.05))
                nid_b = node_by_episode.get(bridge_src["episode_name"])
                nid_r = node_by_episode.get(roll_src["episode_name"])
                if nid_b and nid_r and emp_hop_23 <= 0.035:
                    self.forest.add_horizontal_wormhole(
                        wormhole_id=f"wh_emp_relay_{roll_src['ast_digest'][:6]}",
                        node_a_id=nid_b,
                        node_b_id=nid_r,
                        probe_signature_refs=list(roll_src["sigs"]),
                        role_and_observation_map={
                            "empirical_approximate_bridge": True,
                            "chain_parent_wormhole": primary_1d_wh_id,
                            "bridge_from_schema": bridge_src["schema_tag"],
                            "bridge_to_schema": roll_src["schema_tag"],
                        },
                        hop_error_budget=round(float(emp_hop_23), 5),
                        lipschitz_factor=1.0,
                        permission_tier="local_solver_guidance",
                        compatible_goal_kinds=["open_workspace_mechanism_prediction"],
                        source_episode_refs=[bridge_src["episode_name"], roll_src["episode_name"]],
                    )

        # --- Genuine Witness-Driven Near-Miss Revisions in Real Python Workspaces (`F3`) ---
        for nm_name, (nm_fac, nm_eval) in suite["near_misses"]:
            ws = root_workspace / f"Hegel_{nm_name}"
            lab = nm_fac()
            lens = _estimate_observation_lens(lab, ws)
            # Step 1: Acquire pilot rows & test incumbent parent schema FIRST to record concrete witness failure!
            lab.acquire(_build_pilot_points(lab), workspace=ws, phase="presolve")
            sigs, _, feat_summary = _extract_pilot_signatures(lab, lens)
            incumbent_schema = (
                "conjugate_relaxation_reset"
                if "sig:m2_temporal_relaxation_attenuation" in sigs
                else "liebig_min_2d"
            )
            incumbent_trial = run_workspace_python_synthesizer(
                ws,
                lab,
                observation_lens=lens,
                allowed_schemas=[incumbent_schema],
                output_model_path="incumbent_witness_check.py",
            )
            witness_rmse = float(incumbent_trial.get("train_rmse", float("inf")))

            # Step 2: Run full synthesis + fresh confirmation on remaining budget to obtain repaired mechanism!
            sub = solve_open_workspace_direct(lab, ws, existing_lens=lens)
            lab.meter.charge("ast_revision", compute=10.0)
            ev = nm_eval(ws, sub, lab.observations)
            nm_schema = sub.get("selected_schema")
            nm_digest = sub.get("ast_skeleton_digest")

            rec_nm = {
                "phase": "source_nearmiss",
                "arm": "Hegel_M2",
                "cluster_id": cluster_id,
                "episode_name": nm_name,
                "surface_domain": lab.surface_domain,
                "strict_pass": ev["strict_pass"],
                "incumbent_schema_tested": incumbent_schema,
                "incumbent_witness_rmse": round(witness_rmse, 6),
                "selected_schema": nm_schema,
                "ast_skeleton_digest": nm_digest,
                "observations": [asdict(r) for r in lab.observations],
                "costs": {
                    **lab.meter.snapshot(),
                    "python_execution_seconds": round(lab.python_execution_seconds, 4),
                    "measured_python_wall_seconds": round(lab.measured_python_wall_seconds, 4),
                },
            }
            self.forest.record_raw_episode(rec_nm)
            logs.append(rec_nm)

            if not ev["strict_pass"] or not nm_schema or not nm_digest or witness_rmse <= float(lab.task_view["tolerance"]):
                continue

            if feat_summary.get("unreset_third_state_active") and nm_schema == "conjugate_relaxation_state_memory":
                for wid, wh in list(self.forest.horizontal_wormhole_edges.items()):
                    if "sig:m2_temporal_relaxation_attenuation" in wh["probe_signature_refs"]:
                        self.forest.refine_wormhole_with_near_miss(
                            wid,
                            near_miss_episode_id=nm_name,
                            exclusion_predicate={"field": "unreset_third_state_active", "op": "==", "value": True},
                            reason=f"Incumbent {incumbent_schema} failed witness (rmse={witness_rmse:.4f}); route active 3rd-state memory to state-augmented branch!",
                        )
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
                        role_and_observation_map={"cegar_added_state": "initial_memory_decay", "witness_rmse_before": round(witness_rmse, 5)},
                        hop_error_budget=round(float(sub.get("confirmation_rmse") or 0.012), 5),
                        permission_tier="local_solver_guidance",
                        compatible_goal_kinds=["open_workspace_mechanism_prediction"],
                        source_episode_refs=[nm_name],
                    )
                self.revision_events.append({
                    "cluster_id": cluster_id,
                    "witness_episode": nm_name,
                    "incumbent_witness_rmse": round(witness_rmse, 6),
                    "diagnosis_cause": "abstraction_closure_loss",
                    "revision_action": "ALIGN_exclude_zero_memory_plus_CEGAR_state_augmented_python_wormhole",
                    "spawned_node_id": state_nid,
                    "synthesized_schema": nm_schema,
                })

            if nm_schema == "liebig_capped_min_2d":
                for wid, wh in list(self.forest.horizontal_wormhole_edges.items()):
                    if "sig:m2_2d_complementarity_bottleneck" in wh["probe_signature_refs"]:
                        self.forest.refine_wormhole_with_near_miss(
                            wid,
                            near_miss_episode_id=nm_name,
                            exclusion_predicate={"field": "high_corner_ceiling_detected", "op": "==", "value": True},
                            reason=f"Incumbent {incumbent_schema} failed witness (rmse={witness_rmse:.4f}); route high-load ceiling to capped branch!",
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
                        role_and_observation_map={"split_condition": "high_corner_ceiling_detected", "witness_rmse_before": round(witness_rmse, 5)},
                        hop_error_budget=round(float(sub.get("confirmation_rmse") or 0.015), 5),
                        permission_tier="local_solver_guidance",
                        compatible_goal_kinds=["open_workspace_mechanism_prediction"],
                        source_episode_refs=[nm_name],
                    )
                self.revision_events.append({
                    "cluster_id": cluster_id,
                    "witness_episode": nm_name,
                    "incumbent_witness_rmse": round(witness_rmse, 6),
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
        if self.allow_cross_domain_local_without_wormhole:
            return recalled
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
        """True independent-RNG post-pilot fork on DEV clusters calibrating exact final decision signature!"""
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
            "dev_fork",
            lab_trans,
            ws_t,
            lens=lens,
            sigs=sigs,
            role_order=role_order,
            feat_summary=feat_summary,
            eff_scale=eff_scale,
            bypass_stage1=True,
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

        lab.meter.charge("retrieve", compute=3.0)
        recalled = self._filter_cross_domain(
            self.forest.recall_mid_layer_wormholes(
                sigs, task_view=lab.task_view, allow_horizontal_wormholes=self.allow_horizontal_wormholes
            ),
            lab.surface_domain,
        )

        # Check multi-hop chain certification from real empirical source edges (no target hint flags!)
        valid_recalled: list[dict[str, Any]] = []
        for cand in recalled:
            wh_e = cand.get("wormhole_edge")
            if wh_e is not None:
                parent_wid = wh_e.get("role_and_observation_map", {}).get("chain_parent_wormhole")
                if parent_wid and parent_wid in self.forest.horizontal_wormhole_edges:
                    # Transporting root schema across 2-hop chain requires composed error bound <= tol;
                    # if target_node itself holds the direct 1-hop rollover schema, only 1 hop is used!
                    target_schema = cand["target_node"]["claim_contract"]["conclusion_or_action"].get("schema_tag")
                    bridge_from = wh_e.get("role_and_observation_map", {}).get("bridge_from_schema")
                    if target_schema == bridge_from:
                        chain_check = compose_wormhole_chain(
                            [self.forest.horizontal_wormhole_edges[parent_wid], wh_e],
                            target_tolerance=tol,
                        )
                        if not chain_check["within_tolerance"]:
                            self.alignment_checks.append({
                                "target": name,
                                "decision": "rejected_uncertified_two_hop_transport",
                                "chain_composition": chain_check,
                            })
                            continue
                elif "sig:m2_1d_high_load_rollover" in sigs and cand["target_node"]["claim_contract"]["conclusion_or_action"].get("schema_tag") == "rational_1d":
                    # Trying to apply pure rational_1d to a rollover target requires traversing the 2-hop bridge!
                    hop2 = [
                        e for e in self.forest.horizontal_wormhole_edges.values()
                        if e.get("role_and_observation_map", {}).get("chain_parent_wormhole") == wh_e["wormhole_id"]
                    ]
                    if hop2:
                        chain_check = compose_wormhole_chain([wh_e, hop2[0]], target_tolerance=tol)
                        if not chain_check["within_tolerance"]:
                            self.alignment_checks.append({
                                "target": name,
                                "decision": "rejected_uncertified_two_hop_simplification",
                                "chain_composition": chain_check,
                            })
                            continue
            valid_recalled.append(cand)

        if len(sigs) > 1:
            specific_sig = sigs[-1]
            valid_recalled.sort(
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

        if valid_recalled and gate_dec["admit_transfer"]:
            chosen = valid_recalled[0]
            wh_edge = chosen["wormhole_edge"]
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
                return solve_open_workspace_direct(
            lab, workspace, existing_lens=(lens if lab.calibration_readings or not lab.task_view.get("readout_calibration_available") else None)
        )

            # Genuinely interpret `program_steps`! Deleting calibration or confirm steps changes behavior!
            prog_steps = [s for s in chosen["target_node"].get("program_steps", ()) if isinstance(s, dict)]
            prog_ops = {s.get("op"): s for s in prog_steps}
            exec_lens = (
                _estimate_observation_lens(lab, workspace)
                if "calibrate_observation_lens" in prog_ops
                else {"sign": 1.0, "scale": 1.0, "offset": 0.0}
            )
            schema_tag = prog_ops.get("synthesize_scoped_python_schema", {}).get("schema_tag")
            if schema_tag:
                trial_fit = run_workspace_python_synthesizer(
                    workspace,
                    lab,
                    observation_lens=exec_lens,
                    allowed_schemas=[schema_tag],
                    forced_role_order=(role_order if role_order else None),
                    output_model_path="wormhole_candidate.py",
                )
                if not trial_fit.get("execution_error") and trial_fit["train_rmse"] <= tol:
                    receipt = None
                    val_res = None
                    fit_prefix = list(lab.observations)
                    if "subprocess_confirm_and_refit" in prog_ops:
                        c_pt = _build_confirmation_point(
                            lab,
                            idx_variant=0,
                            role_order=(trial_fit.get("role_order") or role_order),
                            symmetric_roles=bool(feat_summary.get("symmetric_roles", False)),
                        )
                        c_rows = lab.acquire([c_pt], workspace=workspace, phase="confirm")
                        if c_rows:
                            val_res = validate_workspace_model_on_rows(
                                workspace, "wormhole_candidate.py", c_rows, lab, training_prefix=fit_prefix
                            )
                            obs_err = val_res["rmse"] if not val_res.get("execution_error") else float("inf")
                            cand_sha = hashlib.sha256((workspace / "wormhole_candidate.py").read_bytes()).hexdigest()
                            conf_digest = hashlib.sha256(
                                json.dumps([asdict(r) for r in c_rows], sort_keys=True).encode("utf-8")
                            ).hexdigest()[:16]
                            receipt = {
                                "target_id": name,
                                "claim_id": chosen["target_node"]["claim_contract"]["claim_id"],
                                "node_version": chosen["target_node"]["version"],
                                "model_sha256": cand_sha,
                                "confirmation_digest": conf_digest,
                                "observed_error": obs_err,
                                "passed": bool(math.isfinite(obs_err) and obs_err <= tol),
                                "upgrade_to_tier": "local_solver_guidance",
                            }
                    bound_tv = {
                        **lab.task_view,
                        "target_id": name,
                        "node_version": chosen["target_node"]["version"],
                        **(
                            {"expected_model_sha256": receipt["model_sha256"], "expected_confirmation_digest": receipt["confirmation_digest"]}
                            if receipt is not None
                            else {}
                        ),
                    }
                    guidance_auth = authorize_abstraction_use(
                        requested_tier="local_solver_guidance",
                        task_view=bound_tv,
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
                        "executed_program_ops": list(prog_ops.keys()),
                        "stage1_gate": gate_dec,
                        "trial_authorization": trial_auth,
                        "authorization": guidance_auth,
                    })
                    if guidance_auth["granted"] and val_res is not None:
                        return _freeze_confirmed_submission(
                            workspace,
                            "wormhole_candidate.py",
                            fit_prefix,
                            selected_schema=trial_fit["selected_schema"],
                            ast_skeleton_digest=trial_fit["ast_skeleton_digest"],
                            confirmation_rmse=val_res["rmse"],
                            extra_meta={"used_wormhole": (wh_edge or {}).get("wormhole_id")},
                        )

        return solve_open_workspace_direct(
            lab, workspace, existing_lens=(lens if lab.calibration_readings or not lab.task_view.get("readout_calibration_available") else None)
        )

    def solve_target(self, name: str, lab: OpenWorkspaceTaskLab, workspace: Path) -> dict[str, Any]:
        """Single unified M2.5 target research loop across ALL domains and arities!"""
        workspace.mkdir(parents=True, exist_ok=True)
        eff_scale = self._effective_scale_map(lab.task_view.get("available_scale_map"))
        if eff_scale is not None:
            ref_cc = ClaimContract(
                "scale_gate", "open_workspace_ast", [], {}, {}, float(lab.task_view["tolerance"]), "SUPPORTED_IN_SCOPE", "local_solver_guidance"
            ).to_dict()
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

        pilot_pts = _build_pilot_points(lab)
        if lab.acquire(pilot_pts, workspace=workspace, phase="presolve") is None:
            return {"status": "incomplete", "model_path": None}
        # Note: readout lens is calibrated inside `_continue_after_pilot` ONLY if `calibrate_observation_lens` is in `program_steps`,
        # while pilot signatures normalize overall trend orientation without spending extra calibration if deleted!
        prelim_lens = (
            _estimate_observation_lens(lab, workspace)
            if any(
                any(s.get("op") == "calibrate_observation_lens" for s in n.get("program_steps", ()) if isinstance(s, dict))
                for n in self.forest.nodes.values()
            )
            else {"sign": 1.0, "scale": 1.0, "offset": 0.0}
        )
        sigs, role_order, feat_summary = _extract_pilot_signatures(lab, prelim_lens)
        return self._continue_after_pilot(
            name,
            lab,
            workspace,
            lens=prelim_lens,
            sigs=sigs,
            role_order=role_order,
            feat_summary=feat_summary,
            eff_scale=eff_scale,
        )


# ============================================================================
# 8. Arm 4: Equal-Content Flat Banks (Path-Aware Equal-Metadata & Transitive)
# ============================================================================

class OpenWorkspaceFlatEqualContent:
    """Flat Equal-Content Code Bank sharing the EXACT same Stage-1 gate, schemas, & conditions!

    - ``transitive_equivalence=False`` (default strict equal-metadata control): preserves exact
      per-entry signatures, ALIGN exclusions, Stage-1 gate, AND cumulative path error budgets in a flat table!
    - ``transitive_equivalence=True``: collapses pairwise <= 0.025 bridges into transitive equivalence classes
      headed by the lowest-complexity canonical representative.
    """

    def __init__(
        self,
        source_hegel: OpenWorkspaceHegelResearcher,
        *,
        transitive_equivalence: bool = False,
    ) -> None:
        self.gate = deepcopy(source_hegel.gate)
        self.learned_scale_certificates = deepcopy(source_hegel.learned_scale_certificates)
        self.transitive_equivalence = bool(transitive_equivalence)
        self.flat_clusters: list[dict[str, Any]] = []

        wh_by_node: dict[str, list[dict[str, Any]]] = {}
        all_edges = {
            wid: deepcopy(wh)
            for wid, wh in source_hegel.forest.horizontal_wormhole_edges.items()
            if not wh.get("stale_endpoint_revision")
        }
        for wh in all_edges.values():
            wh_by_node.setdefault(wh["node_a_id"], []).append(wh)
            wh_by_node.setdefault(wh["node_b_id"], []).append(wh)

        for nid, node in source_hegel.forest.nodes.items():
            if node["layer_role"] != "mid" or node["claim_contract"].get("evidence_status") != "SUPPORTED_IN_SCOPE":
                continue
            edges = wh_by_node.get(nid, [])
            if not edges:
                continue
            wh = edges[0]
            parent_wid = wh.get("role_and_observation_map", {}).get("chain_parent_wormhole")
            cum_err = float(wh["hop_error_budget"])
            if parent_wid and parent_wid in all_edges:
                cum_err = float(all_edges[parent_wid]["hop_error_budget"]) * float(wh.get("lipschitz_factor", 1.0)) + cum_err
            canonical_schema = node["claim_contract"]["conclusion_or_action"].get("schema_tag")
            if self.transitive_equivalence and wh.get("role_and_observation_map", {}).get("bridge_from_schema"):
                canonical_schema = wh["role_and_observation_map"]["bridge_from_schema"]

            self.flat_clusters.append({
                "entry_id": f"flat:{nid}",
                "sigs": list(node.get("probe_signatures", ())),
                "claim_contract": deepcopy(node["claim_contract"]),
                "schema_tag": canonical_schema,
                "exclusions": deepcopy(wh["align_conditions"]["exclusion_predicates"]),
                "pairwise_hop_err": float(wh["hop_error_budget"]),
                "cumulative_path_err": round(cum_err, 5),
                "is_multi_hop_simplification": bool(parent_wid),
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
            return {"status": "incomplete", "model_path": None}
        sigs, role_order, feat_summary = _extract_pilot_signatures(lab, lens)

        # Flat table linear scan over entries + same Stage-1 gate!
        lab.meter.charge("retrieve_flat", compute=3.0 + 0.25 * len(self.flat_clusters))
        tol = float(lab.task_view["tolerance"])
        specific_sig = sigs[-1]
        matches = [
            e for e in self.flat_clusters
            if set(sigs) & set(e["sigs"])
            and (
                self.transitive_equivalence
                or not e["is_multi_hop_simplification"]
                or e["cumulative_path_err"] <= tol
                or e["schema_tag"] == e["claim_contract"]["conclusion_or_action"].get("schema_tag")
            )
        ]
        matches.sort(key=lambda e: -(1 if specific_sig in e["sigs"] else 0))

        gate_dec = self.gate.decide_stage1(
            specific_sig,
            remaining_budget=lab.meter.remaining_meas(),
            estimated_probe_and_bind_cost=lab.cost_per_obs,
        )

        if matches and gate_dec["admit_transfer"]:
            entry = matches[0]
            eff_err = entry["pairwise_hop_err"] if self.transitive_equivalence else entry["cumulative_path_err"]
            flat_wh = {
                "permission_tier": "local_solver_guidance",
                "hop_error_budget": eff_err,
                "align_conditions": {"exclusion_predicates": entry["exclusions"]},
            }
            t_auth = authorize_abstraction_use(
                requested_tier="bounded_trial",
                task_view=lab.task_view,
                target_features={**dict(lab.domain_regime_attrs), **feat_summary},
                claim_contract=entry["claim_contract"],
                wormhole_edge=flat_wh,
            )
            if t_auth["granted"] and entry["schema_tag"]:
                tf = run_workspace_python_synthesizer(
                    workspace,
                    lab,
                    observation_lens=lens,
                    allowed_schemas=[entry["schema_tag"]],
                    forced_role_order=(role_order if role_order else None),
                    output_model_path="flat_candidate.py",
                )
                if not tf.get("execution_error") and tf["train_rmse"] <= tol:
                    fit_prefix = list(lab.observations)
                    c_pt = _build_confirmation_point(
                        lab,
                        idx_variant=0,
                        role_order=(tf.get("role_order") or role_order),
                        symmetric_roles=bool(feat_summary.get("symmetric_roles", False)),
                    )
                    c_rows = lab.acquire([c_pt], workspace=workspace, phase="confirm")
                    if c_rows:
                        vr = validate_workspace_model_on_rows(
                            workspace, "flat_candidate.py", c_rows, lab, training_prefix=fit_prefix
                        )
                        if not vr.get("execution_error") and vr["rmse"] <= tol:
                            return _freeze_confirmed_submission(
                                workspace,
                                "flat_candidate.py",
                                fit_prefix,
                                selected_schema=tf["selected_schema"],
                                ast_skeleton_digest=tf["ast_skeleton_digest"],
                                confirmation_rmse=vr["rmse"],
                            )
        return solve_open_workspace_direct(
            lab, workspace, existing_lens=(lens if lab.calibration_readings or not lab.task_view.get("readout_calibration_available") else None)
        )
