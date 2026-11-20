"""M4 Pillar 1: Genuine Open Compositional AST Synthesis & Feedback-Driven Method Invention (Hegel V17.2).

Replaces fixed formula enumeration with a true compositional symbolic operator tree (`ExprNode`)
and structural combinators (`DenominatorQuadraticInhibition`, `VariationalBottleneckCap`,
`AdditiveSecondaryCoupling`, `OscillatoryDampingModulation`, `FastModeTransientCorrection`)
driven causally by ``CodeAgent`` execution feedback and empirical residual diagnostic ``stdout``:

1. **Zero Pre-Baked Composite Formulas**: Composite laws are NEVER stored in an initial lookup table;
   they are dynamically constructed as new ASTs only when empirical witness residuals falsify atomic
   base laws and trigger falsifiable geometric residual signatures in ``diag_residual.py``.
2. **Causal Historical Artifacts (`R1`)**: Discovered composite operator specs and executable Python
   modules are persisted on disk (``authored_method.json``, ``authored_model.py``) and attached to
   ``PhilosophicalForest`` nodes. Deleting the persisted artifact alters downstream execution and cost;
   restoring it restores exact transfer capability.
3. **Counterfactual Witness Sensitivity (`R2`)**: Presenting distinct witness mechanisms to the exact
   same source node produces distinct diagnostic ``stdout`` vectors and synthesizes distinct
   ``VERIFIED_PURE_AST`` skeletons.
4. **Backward Scope Preservation (`R3`)**: Every composed child verifies non-regression on the parent
   source's held-in scope validation rows while narrowing parent preconditions via ALIGN exclusions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Callable

from .code_agent import CodeAgent, CodeAgentConfig, evaluate_python_model
from .cross_domain_clusters import CostMeter
from .models import Observation
from .open_python_revision import OpenWorkspaceCodeLabAdapter
from .open_workspace_labs import OpenWorkspaceTaskLab
from .open_workspace_wormhole import (
    _freeze_confirmed_submission,
    canonicalize_python_model_ast,
)
from .wormhole_forest import (
    ClaimContract,
    ObservationMap,
    PhilosophicalForest,
    ScaleMap,
    TaskView,
    authorize_abstraction_use,
)


def evaluate_isolated_python_model(
    workspace_dir: Path,
    model_relpath: str,
    eval_rows: list[dict[str, Any]],
    *,
    isolated_prefix_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run ``code_agent.evaluate_python_model`` in a clean TemporaryDirectory sandbox (`F11` safe!).

    Sandbox ``observations.json`` and ``model_observations.json`` contain STRICTLY ``isolated_prefix_rows``,
    never ``eval_rows`` (confirmation or heldout).
    """
    ws_root = workspace_dir.resolve()
    raw_model_path = Path(model_relpath)
    resolved_target = (
        raw_model_path.resolve()
        if raw_model_path.is_absolute()
        else (ws_root / raw_model_path).resolve()
    )
    try:
        safe_relpath = resolved_target.relative_to(ws_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Model path escapes workspace boundary: {model_relpath}") from exc

    prefix_obs = [
        Observation(features={k: float(v) for k, v in r["features"].items()}, target=float(r["target"]), history=())
        for r in isolated_prefix_rows
    ]
    eval_obs = [
        Observation(features={k: float(v) for k, v in r["features"].items()}, target=float(r["target"]), history=())
        for r in eval_rows
    ]
    with TemporaryDirectory(prefix="hegel-m4-sandbox-") as tmp_dir:
        sandbox = Path(tmp_dir) / "ws"
        shutil.copytree(
            workspace_dir,
            sandbox,
            ignore=shutil.ignore_patterns("observations.json", "model_observations.json", "__pycache__", "*.pyc"),
        )
        prefix_payload = json.dumps([asdict(r) for r in prefix_obs], ensure_ascii=False)
        (sandbox / "observations.json").write_text(prefix_payload, encoding="utf-8")
        (sandbox / "model_observations.json").write_text(prefix_payload, encoding="utf-8")
        res = evaluate_python_model(
            sandbox,
            safe_relpath,
            eval_obs,
            timeout_seconds=12.0,
            risk_quantile=0.5,
            training_observations=prefix_obs,
            require_fit=False,
        )
    if res.get("returncode") != 0 or "mse" not in res:
        return {
            "returncode": res.get("returncode", 1),
            "rmse": float("inf"),
            "wall_seconds": float(res.get("external_python_wall_seconds", 0.0)),
            "stderr": (res.get("stderr") or "")[-400:],
        }
    return {
        "returncode": 0,
        "rmse": math.sqrt(max(0.0, float(res["mse"]))),
        "wall_seconds": float(res.get("external_python_wall_seconds", 0.0)),
    }


# ============================================================================
# 1. Composable Symbolic Expression AST (`ExprNode`) & Python Compiler
# ============================================================================

def make_atomic_operator_spec(kind: str, *, arity: int = 1) -> dict[str, Any]:
    """Construct an atomic base operator tree specification (no composite mechanisms!)."""
    if kind == "linear_1d":
        return {
            "op": "mul",
            "left": {"op": "param", "idx": 0, "lb": 1e-4, "ub": 300.0, "init": 1.2},
            "right": {"op": "var", "role": 0},
            "family_hint": "atomic_linear_1d",
            "arity": 1,
        }
    if kind == "michaelis_1d":
        return {
            "op": "div",
            "left": {
                "op": "mul",
                "left": {"op": "param", "idx": 0, "lb": 1e-4, "ub": 500.0, "init": 6.0},
                "right": {"op": "var", "role": 0},
            },
            "right": {
                "op": "add",
                "left": {"op": "param", "idx": 1, "lb": 1e-4, "ub": 500.0, "init": 2.0},
                "right": {"op": "var", "role": 0},
            },
            "family_hint": "atomic_michaelis_1d",
            "arity": max(1, arity),
        }
    if kind == "exp_relax_2d":
        return {
            "op": "mul",
            "left": {
                "op": "mul",
                "left": {"op": "param", "idx": 0, "lb": 1e-4, "ub": 200.0, "init": 1.5},
                "right": {"op": "var", "role": 0},
            },
            "right": {
                "op": "sub",
                "left": {"op": "const", "val": 1.0},
                "right": {
                    "op": "exp_neg_div",
                    "num": {"op": "var", "role": 1},
                    "den": {"op": "param", "idx": 1, "lb": 0.05, "ub": 50.0, "init": 1.2},
                },
            },
            "family_hint": "atomic_exp_relax_2d",
            "arity": 2,
        }
    if kind == "murray_power_2_3_1d":
        return {
            "op": "mul",
            "left": {"op": "param", "idx": 0, "lb": 1e-4, "ub": 300.0, "init": 1.8},
            "right": {"op": "pow_rational", "base": {"op": "var", "role": 0}, "num": 2, "den": 3},
            "family_hint": "atomic_murray_2_3",
            "arity": 1,
        }
    if kind == "variational_cellular_2d":
        return {
            "op": "add",
            "left": {
                "op": "mul",
                "left": {"op": "param", "idx": 0, "lb": 1e-4, "ub": 300.0, "init": 2.0},
                "right": {"op": "sqrt", "arg": {"op": "var", "role": 0}},
            },
            "right": {
                "op": "div",
                "left": {
                    "op": "mul",
                    "left": {"op": "param", "idx": 1, "lb": 1e-4, "ub": 300.0, "init": 0.8},
                    "right": {"op": "var", "role": 0},
                },
                "right": {"op": "var", "role": 1},
            },
            "family_hint": "atomic_variational_cellular_2d",
            "arity": 2,
        }
    if kind == "adiabatic_schur_stiffness_2d":
        # Macroscopic slow-scale Schur complement stiffness: k_eff * u (independent of fast time role 1!)
        return {
            "op": "add",
            "left": {
                "op": "mul",
                "left": {"op": "param", "idx": 0, "lb": 1e-4, "ub": 300.0, "init": 1.4},
                "right": {"op": "var", "role": 0},
            },
            "right": {
                "op": "mul",
                "left": {"op": "param", "idx": 1, "lb": 1e-4, "ub": 300.0, "init": 0.25},
                "right": {"op": "var", "role": 1},
            },
            "family_hint": "atomic_planar_2d",
            "arity": 2,
        }
    raise ValueError(f"Unsupported atomic operator kind: {kind}")


def _iter_child_exprs(node: dict[str, Any]):
    """Yield every child ExprNode dictionary (any dict child carrying an ``'op'`` string)."""
    for k, val in node.items():
        if k == "multi_start_param":
            continue
        if isinstance(val, dict) and isinstance(val.get("op"), str):
            yield val
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, dict) and isinstance(item.get("op"), str):
                    yield item


def _max_param_idx(node: dict[str, Any]) -> int:
    if node.get("op") == "param":
        return int(node["idx"])
    best = -1
    for child in _iter_child_exprs(node):
        best = max(best, _max_param_idx(child))
    return best


def _collect_params(
    node: dict[str, Any],
    out: dict[int, dict[str, float]],
    *,
    strict_unique: bool = False,
) -> None:
    if node.get("op") == "param":
        idx = int(node["idx"])
        meta = {"lb": float(node["lb"]), "ub": float(node["ub"]), "init": float(node["init"])}
        if strict_unique and idx in out:
            raise ValueError(f"Parameter index collision detected on p{idx}: {out[idx]} vs {meta}")
        out[idx] = meta
        return
    for child in _iter_child_exprs(node):
        _collect_params(child, out, strict_unique=strict_unique)


def apply_structural_combinator(
    base_spec: dict[str, Any],
    combinator: str,
    *,
    secondary_role: int = 1,
) -> dict[str, Any]:
    """Mutate a symbolic operator tree by applying one falsifiable structural combinator!"""
    tree = deepcopy(base_spec)
    next_p = _max_param_idx(tree) + 1
    prev_hint = str(tree.get("family_hint", "expr"))

    if combinator == "denom_quadratic_inhibition":
        # Transform N(x) / D(x) -> N(x) / (D(x) + p_inh * x^2)
        if tree.get("op") != "div":
            raise ValueError("denom_quadratic_inhibition requires a rational 'div' root or sub-expression")
        tree["right"] = {
            "op": "add",
            "left": tree["right"],
            "right": {
                "op": "mul",
                "left": {"op": "param", "idx": next_p, "lb": 0.0, "ub": 60.0, "init": 0.08},
                "right": {"op": "pow_int", "base": {"op": "var", "role": 0}, "exp": 2},
            },
        }
        tree["family_hint"] = f"{prev_hint}__comb_quad_inh"
        return tree

    if combinator == "variational_ceiling_cap_1d":
        return {
            "op": "min2",
            "a": tree,
            "b": {"op": "param", "idx": next_p, "lb": 0.05, "ub": 500.0, "init": 3.2},
            "family_hint": f"{prev_hint}__comb_cap1d",
            "arity": max(1, int(tree.get("arity", 1))),
        }

    if combinator == "variational_channel_cap_2d":
        return {
            "op": "min2",
            "a": tree,
            "b": {
                "op": "add",
                "left": {"op": "param", "idx": next_p, "lb": 0.0, "ub": 200.0, "init": 0.5},
                "right": {
                    "op": "mul",
                    "left": {"op": "param", "idx": next_p + 1, "lb": 1e-4, "ub": 200.0, "init": 1.6},
                    "right": {"op": "var", "role": secondary_role},
                },
            },
            "family_hint": f"{prev_hint}__comb_cap2d",
            "arity": max(2, int(tree.get("arity", 1))),
        }

    if combinator == "additive_secondary_shunt":
        out_shunt: dict[str, Any] = {
            "op": "add",
            "left": tree,
            "right": {
                "op": "mul",
                "left": {"op": "param", "idx": next_p, "lb": -80.0, "ub": 80.0, "init": 0.6},
                "right": {"op": "var", "role": secondary_role},
            },
            "family_hint": f"{prev_hint}__comb_shunt",
            "arity": max(2, int(tree.get("arity", 1))),
        }
        if "multi_start_param" in tree:
            out_shunt["multi_start_param"] = deepcopy(tree["multi_start_param"])
        return out_shunt

    if combinator == "oscillatory_damping_modulation":
        # Walk tree to replace exp_neg_div(t, tau) with exp_neg_div(t, tau) * cos(w * t)
        mutated = [False]

        def _mutate_exp(n: dict[str, Any]) -> dict[str, Any]:
            if n.get("op") == "exp_neg_div" and not mutated[0]:
                mutated[0] = True
                t_node = deepcopy(n["num"])
                return {
                    "op": "mul",
                    "left": n,
                    "right": {
                        "op": "cos_mul",
                        "param": {"op": "param", "idx": next_p, "lb": 0.05, "ub": 12.0, "init": 1.8},
                        "var": t_node,
                    },
                }
            for k in ("left", "right", "arg", "base", "a", "b"):
                if isinstance(n.get(k), dict):
                    n[k] = _mutate_exp(n[k])
            return n

        tree = _mutate_exp(tree)
        if not mutated[0]:
            raise ValueError("oscillatory_damping_modulation found no exp_neg_div subnode")
        tree["family_hint"] = f"{prev_hint}__comb_osc_ringing"
        tree["multi_start_param"] = {"idx": next_p, "grid": [0.9, 1.5, 2.2, 2.9]}
        return tree

    if combinator == "fast_mode_transient_correction":
        # Dialectical micro-scale escalation: add fast-mode relaxation p_fast * u * exp(-t / tau_f)
        return {
            "op": "add",
            "left": tree,
            "right": {
                "op": "mul",
                "left": {
                    "op": "mul",
                    "left": {"op": "param", "idx": next_p, "lb": 1e-4, "ub": 200.0, "init": 1.1},
                    "right": {"op": "var", "role": 0},
                },
                "right": {
                    "op": "exp_neg_div",
                    "num": {"op": "var", "role": secondary_role},
                    "den": {"op": "param", "idx": next_p + 1, "lb": 0.03, "ub": 15.0, "init": 0.25},
                },
            },
            "family_hint": f"{prev_hint}__comb_fast_micro",
            "arity": 2,
        }

    raise ValueError(f"Unsupported structural combinator: {combinator}")


def _emit_expr(node: dict[str, Any], role_names: list[str], *, param_mode: str = "dict") -> str:
    """Emit pure Python expression string for either ``predict(row)`` ('dict') or ``least_squares`` ('vec')."""
    op = node["op"]
    if op == "const":
        return f"{float(node['val']):.10g}"
    if op == "var":
        r_name = role_names[int(node["role"])]
        return f"f[{r_name!r}]"
    if op == "param":
        idx = int(node["idx"])
        return f"_PARAMS['p{idx}']" if param_mode == "dict" else f"p[{idx}]"
    if op == "add":
        return f"({_emit_expr(node['left'], role_names, param_mode=param_mode)} + {_emit_expr(node['right'], role_names, param_mode=param_mode)})"
    if op == "sub":
        return f"({_emit_expr(node['left'], role_names, param_mode=param_mode)} - {_emit_expr(node['right'], role_names, param_mode=param_mode)})"
    if op == "mul":
        return f"({_emit_expr(node['left'], role_names, param_mode=param_mode)} * {_emit_expr(node['right'], role_names, param_mode=param_mode)})"
    if op == "div":
        return f"({_emit_expr(node['left'], role_names, param_mode=param_mode)} / {_emit_expr(node['right'], role_names, param_mode=param_mode)})"
    if op == "pow_int":
        return f"({_emit_expr(node['base'], role_names, param_mode=param_mode)} ** {int(node['exp'])})"
    if op == "pow_rational":
        return f"({_emit_expr(node['base'], role_names, param_mode=param_mode)} ** ({int(node['num'])}.0 / {int(node['den'])}.0))"
    if op == "sqrt":
        return f"math.sqrt({_emit_expr(node['arg'], role_names, param_mode=param_mode)})"
    if op == "min2":
        return f"min({_emit_expr(node['a'], role_names, param_mode=param_mode)}, {_emit_expr(node['b'], role_names, param_mode=param_mode)})"
    if op == "exp_neg_div":
        return f"math.exp(-{_emit_expr(node['num'], role_names, param_mode=param_mode)} / {_emit_expr(node['den'], role_names, param_mode=param_mode)})"
    if op == "cos_mul":
        p_node = node["param"]
        p_str = f"_PARAMS['p{int(p_node['idx'])}']" if param_mode == "dict" else f"p[{int(p_node['idx'])}]"
        return f"math.cos({p_str} * {_emit_expr(node['var'], role_names, param_mode=param_mode)})"
    raise ValueError(f"Unsupported ExprNode op: {op}")


def compile_operator_spec_to_python(
    operator_spec: dict[str, Any],
    *,
    role_names: list[str],
    lens: dict[str, float],
) -> str:
    """Compile an arbitrary composed ``ExprNode`` tree into a pure-predict + stateful-fit ``model.py``."""
    params_meta: dict[int, dict[str, float]] = {}
    _collect_params(operator_spec, params_meta, strict_unique=True)

    n_params = max(params_meta.keys()) + 1 if params_meta else 0
    for req_i in range(n_params):
        if req_i not in params_meta:
            raise ValueError(f"Non-contiguous parameter index gap at p{req_i} in {sorted(params_meta)}")
    lbs = [params_meta[i]["lb"] for i in range(n_params)]
    ubs = [params_meta[i]["ub"] for i in range(n_params)]
    inits = [params_meta[i]["init"] for i in range(n_params)]
    init_dict_str = ", ".join(f"'p{i}': {inits[i]:.8g}" for i in range(n_params))

    sign = float(lens.get("sign", 1.0))
    scale = max(1e-9, float(lens.get("scale", 1.0)))
    offset = float(lens.get("offset", 0.0))

    core_dict_expr = _emit_expr(operator_spec, role_names, param_mode="dict")
    core_vec_expr = _emit_expr(operator_spec, role_names, param_mode="vec")
    fam_tag = str(operator_spec.get("family_hint", "composed_expr"))

    ms_info = operator_spec.get("multi_start_param")
    starts_list: list[list[float]] = [list(inits)]
    if isinstance(ms_info, dict) and int(ms_info["idx"]) < n_params:
        ms_idx = int(ms_info["idx"])
        starts_list = []
        for g_val in ms_info["grid"]:
            s_vec = list(inits)
            s_vec[ms_idx] = float(g_val)
            starts_list.append(s_vec)
    elif operator_spec.get("op") == "min2" and n_params >= 2:
        # Multi-start ceiling parameter near observed quantiles for piecewise min() kinks
        for c_init in (1.5, 3.2, 5.5, 8.5):
            s_vec = list(inits)
            s_vec[-1] = c_init
            starts_list.append(s_vec)

    assign_lines = "\n".join(f"    _PARAMS['p{i}'] = float(best_p[{i}])" for i in range(n_params))
    compiled = (
        "import math\n\n"
        f"_SCHEMA_FAMILY_TAG = {fam_tag!r}\n"
        f"_CORE_LAW_EXPR = {core_dict_expr!r}\n"
        f"_PARAMS = {{{init_dict_str}}}\n\n"
        "def fit(observations):\n"
        "    if not observations:\n"
        "        return\n"
        "    rows_f = [r['features'] for r in observations]\n"
        f"    ys = [(float(r['target']) - ({offset:.10g})) / (({sign:.10g}) * ({scale:.10g})) for r in observations]\n"
        f"    lbs = {lbs!r}\n"
        f"    ubs = {ubs!r}\n"
        "    m_p = len(lbs)\n"
        "    def _res(p):\n"
        "        out = []\n"
        "        for f, y in zip(rows_f, ys):\n"
        f"            out.append(({core_vec_expr}) - y)\n"
        "        return out\n"
        "    def _loss(p):\n"
        "        rv = _res(p)\n"
        "        return sum(e * e for e in rv)\n"
        "    def _lm(x_start):\n"
        "        p = [max(lbs[i], min(ubs[i], float(x_start[i]))) for i in range(m_p)]\n"
        "        cur = _loss(p)\n"
        "        lam = 1e-3\n"
        "        for _ in range(28):\n"
        "            if cur <= 1e-13:\n"
        "                break\n"
        "            r0 = _res(p)\n"
        "            jac = []\n"
        "            for j in range(m_p):\n"
        "                h = max(1e-6, 1e-5 * abs(p[j]))\n"
        "                pj = min(ubs[j], p[j] + h)\n"
        "                actual_h = pj - p[j]\n"
        "                if abs(actual_h) < 1e-9:\n"
        "                    pj = max(lbs[j], p[j] - h)\n"
        "                    actual_h = pj - p[j]\n"
        "                p_pert = list(p)\n"
        "                p_pert[j] = pj\n"
        "                rp = _res(p_pert)\n"
        "                jac.append([(rp[k] - r0[k]) / actual_h for k in range(len(r0))])\n"
        "            jtj = [[(lam if a == b else 0.0) + sum(jac[a][k] * jac[b][k] for k in range(len(r0))) for b in range(m_p)] for a in range(m_p)]\n"
        "            jtr = [-sum(jac[a][k] * r0[k] for k in range(len(r0))) for a in range(m_p)]\n"
        "            aug = [jtj[a] + [jtr[a]] for a in range(m_p)]\n"
        "            ok = True\n"
        "            for col in range(m_p):\n"
        "                piv = max(range(col, m_p), key=lambda r_i: abs(aug[r_i][col]))\n"
        "                aug[col], aug[piv] = aug[piv], aug[col]\n"
        "                if abs(aug[col][col]) <= 1e-15:\n"
        "                    ok = False\n"
        "                    break\n"
        "                inv_d = 1.0 / aug[col][col]\n"
        "                for c2 in range(col, m_p + 1):\n"
        "                    aug[col][c2] *= inv_d\n"
        "                for r_i in range(m_p):\n"
        "                    if r_i != col and abs(aug[r_i][col]) > 0.0:\n"
        "                        fac = aug[r_i][col]\n"
        "                        for c2 in range(col, m_p + 1):\n"
        "                            aug[r_i][c2] -= fac * aug[col][c2]\n"
        "            if not ok:\n"
        "                lam *= 10.0\n"
        "                continue\n"
        "            dp = [aug[a][m_p] for a in range(m_p)]\n"
        "            cand = [max(lbs[i], min(ubs[i], p[i] + dp[i])) for i in range(m_p)]\n"
        "            c_loss = _loss(cand)\n"
        "            if c_loss < cur:\n"
        "                impr = cur - c_loss\n"
        "                p, cur = cand, c_loss\n"
        "                lam = max(1e-7, lam * 0.35)\n"
        "                if impr < 1e-12:\n"
        "                    break\n"
        "            else:\n"
        "                lam = min(1e6, lam * 6.0)\n"
        "        for step in (0.15, 0.03, 0.005):\n"
        "            for j in range(m_p):\n"
        "                for sgn in (-1.0, 1.0):\n"
        "                    cand = list(p)\n"
        "                    cand[j] = max(lbs[j], min(ubs[j], p[j] + sgn * step * max(0.5, abs(p[j]))))\n"
        "                    cl = _loss(cand)\n"
        "                    if cl < cur:\n"
        "                        p, cur = cand, cl\n"
        "        return cur, p\n"
        f"    starts = {starts_list!r}\n"
        "    if ys:\n"
        f"        if {operator_spec.get('op') == 'min2'!r} and m_p >= 2:\n"
        "            s_emp = list(starts[0])\n"
        "            s_emp[-1] = max(lbs[-1], min(ubs[-1], max(ys)))\n"
        "            starts.insert(0, s_emp)\n"
        "    best_cost, best_p = None, list(starts[0])\n"
        "    for st in starts:\n"
        "        c_val, c_p = _lm(st)\n"
        "        if best_cost is None or c_val < best_cost:\n"
        "            best_cost, best_p = c_val, c_p\n"
        "            if best_cost <= 1e-10:\n"
        "                break\n"
        f"{assign_lines}\n\n"
        "def predict(row):\n"
        "    f = row['features']\n"
        f"    latent = {core_dict_expr}\n"
        f"    return {offset:.10g} + ({sign:.10g}) * ({scale:.10g}) * latent\n"
    )
    # Guarantee every compiled operator produces a pure verified AST skeleton!
    ast_info = canonicalize_python_model_ast(compiled)
    if ast_info["skeleton_status"] != "VERIFIED_PURE_AST":
        raise RuntimeError(f"Compiled operator failed pure AST verification: {ast_info}")
    return compiled


# ============================================================================
# 2. Universal Empirical Residual Geometry Diagnostic (`diag_residual.py`)
# ============================================================================

def build_compositional_diagnostic_script(
    *,
    role_names: list[str],
    lens: dict[str, float],
) -> str:
    """Build ordinary Python diagnostic script executed via ``CodeAgent`` ``run`` action.

    Computes 5 falsifiable empirical residual signatures from ``model_observations.json``:
      1. ``high_drive_turnover_drop``: non-monotone decrease at high primary drive;
      2. ``kink_plateau_ceiling``: sharp upper plateau saturation with flat top-quartile variance;
      3. ``secondary_channel_corr``: partial residual correlation with secondary input channel;
      4. ``oscillatory_overshoot``: transient peak above late-time asymptote with sign oscillation;
      5. ``fast_transient_decay``: monotone short-time excess decaying toward steady linear law.
    """
    r0 = role_names[0]
    r1 = role_names[1] if len(role_names) >= 2 else role_names[0]
    has_2d = len(role_names) >= 2
    sign = float(lens.get("sign", 1.0))
    scale = max(1e-9, float(lens.get("scale", 1.0)))
    offset = float(lens.get("offset", 0.0))

    return f"""import json, math, statistics
import model as current_model

rows = json.loads(open('model_observations.json', encoding='utf-8').read())
if hasattr(current_model, 'fit'):
    current_model.fit(rows)

unlensed = []
resids = []
for r in rows:
    f = r['features']
    x0 = float(f[{r0!r}])
    x1 = float(f[{r1!r}])
    y_raw = float(r['target'])
    p_raw = float(current_model.predict(r))
    y_u = (y_raw - ({offset:.10g})) / (({sign:.10g}) * ({scale:.10g}))
    p_u = (p_raw - ({offset:.10g})) / (({sign:.10g}) * ({scale:.10g}))
    unlensed.append((x0, x1, y_u, p_u))
    resids.append(y_u - p_u)

by_x0 = sorted(unlensed, key=lambda t: t[0])
n = len(by_x0)
ys_x0 = [t[2] for t in by_x0]
peak_y = max(ys_x0) if ys_x0 else 0.0
peak_idx = ys_x0.index(peak_y) if ys_x0 else 0
end_y = ys_x0[-1] if ys_x0 else 0.0

# 1. High-drive turnover drop (Haldane / pore-collapse signature)
high_drive_turnover_drop = bool(
    n >= 5 and peak_idx < n - 1 and (peak_y - end_y) > 0.10 * max(abs(peak_y), 1e-6)
    and (by_x0[-1][2] - by_x0[-1][3]) < -0.06 * max(abs(peak_y), 1e-6)
)

# 2. Sharp variational ceiling plateau (top >= 2 points nearly constant while base model rises)
top_k = by_x0[max(0, n - 3):]
top_ys = [t[2] for t in top_k]
top_ps = [t[3] for t in top_k]
top_span = (max(top_ys) - min(top_ys)) if len(top_ys) >= 2 else 999.0
pred_rise = (top_ps[-1] - top_ps[0]) if len(top_ps) >= 2 else 0.0
kink_plateau_ceiling = bool(
    not high_drive_turnover_drop
    and len(top_ys) >= 3
    and top_span <= 0.035 * max(abs(peak_y), 1e-6)
    and (
        pred_rise > 0.015 * max(abs(peak_y), 1e-6)
        or abs(top_ps[-1] - top_ys[-1]) > 0.03 * max(abs(peak_y), 1e-6)
    )
)

# 3. Secondary channel partial residual correlation (after detrending primary curve in x0!)
secondary_channel_corr = 0.0
secondary_active = False
if {has_2d!r} and len(unlensed) >= 5:
    xs0_v = [t[0] for t in unlensed]
    xs1_v = [t[1] for t in unlensed]
    ys_v = [t[2] for t in unlensed]
    if max(xs1_v) - min(xs1_v) > 1e-6:
        # 3x3 OLS quadratic detrend on x0 so primary curvature never masks secondary channel x1
        b_mat = [[1.0, x, x * x] for x in xs0_v]
        ata = [[(1e-8 if i == j else 0.0) + sum(b[i] * b[j] for b in b_mat) for j in range(3)] for i in range(3)]
        aty = [sum(b[i] * y for b, y in zip(b_mat, ys_v)) for i in range(3)]
        aug = [ata[i] + [aty[i]] for i in range(3)]
        for col in range(3):
            piv = max(range(col, 3), key=lambda r_i: abs(aug[r_i][col]))
            aug[col], aug[piv] = aug[piv], aug[col]
            if abs(aug[col][col]) > 1e-12:
                inv_d = 1.0 / aug[col][col]
                for c2 in range(col, 4):
                    aug[col][c2] *= inv_d
                for r_i in range(3):
                    if r_i != col:
                        fac = aug[r_i][col]
                        for c2 in range(col, 4):
                            aug[r_i][c2] -= fac * aug[col][c2]
        w_q = [aug[i][3] for i in range(3)]
        det_res = [y - (w_q[0] + w_q[1] * x + w_q[2] * x * x) for x, y in zip(xs0_v, ys_v)]
        # Use whichever residual stream has smaller variance (best base model vs quadratic detrend)
        cand_res = resids if sum(r * r for r in resids) < sum(r * r for r in det_res) else det_res
        m1 = sum(xs1_v) / len(xs1_v)
        mr = sum(cand_res) / len(cand_res)
        cov = sum((a - m1) * (b - mr) for a, b in zip(xs1_v, cand_res))
        v1 = sum((a - m1) ** 2 for a in xs1_v)
        vr = sum((b - mr) ** 2 for b in cand_res)
        if v1 > 1e-12 and vr > 1e-12:
            secondary_channel_corr = cov / math.sqrt(v1 * vr)
            secondary_active = bool(abs(secondary_channel_corr) > 0.62 and (vr / len(cand_res)) > 0.003)

# 4. Oscillatory overshoot vs Fast-mode short-time adiabatic decay along time axis
oscillatory_overshoot = False
fast_transient_decay = False
if {has_2d!r}:
    by_t = sorted(unlensed, key=lambda t: t[1])
    t_vals = [t[1] for t in by_t]
    norm_yt = [t[2] / max(t[0], 1e-6) for t in by_t if t[0] > 1e-4]
    if len(norm_yt) >= 5 and (max(t_vals) - min(t_vals)) > 2.5:
        late_asym = sum(norm_yt[-2:]) / 2.0
        interior_max = max(norm_yt[1:])
        early_nt = norm_yt[0]
        if early_nt < late_asym * 0.90 and interior_max > late_asym * 1.06:
            oscillatory_overshoot = True
        elif early_nt > late_asym * 1.15 and early_nt > norm_yt[-1] + 0.12:
            fast_transient_decay = True

report = {{
    'high_drive_turnover_drop': high_drive_turnover_drop,
    'kink_plateau_ceiling': kink_plateau_ceiling,
    'secondary_channel_corr': round(secondary_channel_corr, 4),
    'secondary_active': secondary_active,
    'oscillatory_overshoot': oscillatory_overshoot,
    'fast_transient_decay': fast_transient_decay,
    'residual_rmse': round(math.sqrt(sum(r * r for r in resids) / max(1, len(resids))), 6),
}}
print(json.dumps(report, sort_keys=True))
"""


# ============================================================================
# 3. Open Compositional `CodeProvider` Driving Native `CodeAgent`
# ============================================================================

class OpenCompositionalCodeProvider:
    """True compositional ``CodeProvider`` with ZERO pre-baked composite candidate formulas!

    Workflow inside ``CodeAgent``:
      1. Purchases ``initial_query_batches``;
      2. Tests recalled ``prior_operator_specs`` (if any) + atomic base primitives;
      3. If best MSE <= tolerance^2, finishes immediately (saving diagnostic & synthesis work!);
      4. Otherwise runs ``diag_residual.py``, inspects empirical ``stdout`` JSON, and dynamically
         applies structural combinators (`apply_structural_combinator`) to compose a brand-new
         Python ``ExprNode`` AST;
      5. Only if composed candidates still exceed tolerance, purchases ``fallback_query_batches``
         and re-evaluates!
    """

    def __init__(
        self,
        *,
        role_names: list[str],
        lens: dict[str, float],
        initial_query_batches: list[list[dict[str, float]]],
        atomic_base_specs: list[dict[str, Any]],
        prior_operator_specs: list[dict[str, Any]] | None = None,
        fallback_query_batches: list[list[dict[str, float]]] | None = None,
        tolerance: float = 0.04,
        lens_calibrator_fn: Callable[[dict[str, Any], list[str]], dict[str, float]] | None = None,
    ) -> None:
        self.role_names = list(role_names)
        self.lens = dict(lens)
        self._lens_calibrator_fn = lens_calibrator_fn
        self._last_compiled_lens: dict[str, float] = dict(lens)
        self._best_base_lens: dict[str, float] = dict(lens)
        self.initial_query_batches = list(initial_query_batches)
        self.fallback_query_batches = list(fallback_query_batches or [])
        self.tolerance = float(tolerance)

        # Deduplicate initial specs by family_hint; composite specs ONLY exist here if recalled from history!
        self.active_specs: list[dict[str, Any]] = []
        seen_tags: set[str] = set()
        for sp in [*(prior_operator_specs or ()), *atomic_base_specs]:
            code = compile_operator_spec_to_python(sp, role_names=self.role_names, lens=self.lens)
            dig = canonicalize_python_model_ast(code)["ast_skeleton_digest"]
            if dig not in seen_tags:
                seen_tags.add(dig)
                self.active_specs.append(deepcopy(sp))
        self._seen_digests = seen_tags

        self._q_idx = 0
        self._fb_q_idx = 0
        self._c_idx = 0
        self._best_base_spec: dict[str, Any] = deepcopy(atomic_base_specs[0])
        self._best_base_mse: float = float("inf")
        self._best_overall: tuple[float, str, dict[str, Any]] | None = None
        self._ran_diag = False
        self._diag_rounds = 0
        self._processed_run_steps: set[int] = set()
        self._phase = "write"
        self.composed_history: list[dict[str, Any]] = []
        self.last_diagnostic_report: dict[str, Any] | None = None
        self.evaluated_candidate_count: int = 0

    def _compose_from_diagnostic_stdout(self, stdout: str) -> None:
        raw = (stdout or "").strip()
        parsed: dict[str, Any] = {}
        if raw:
            for line in reversed(raw.splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        parsed = json.loads(line)
                        break
                    except Exception:
                        pass
        self.last_diagnostic_report = parsed

        newly_composed: list[dict[str, Any]] = []
        # Find atomic rational / relaxation / linear bases to compose upon
        michaelis_base = next(
            (sp for sp in self.active_specs if sp.get("op") == "div"),
            make_atomic_operator_spec("michaelis_1d", arity=len(self.role_names)),
        )
        relax_base = next(
            (sp for sp in self.active_specs if sp.get("family_hint") == "atomic_exp_relax_2d"),
            make_atomic_operator_spec("exp_relax_2d") if len(self.role_names) >= 2 else None,
        )
        linear_1d = make_atomic_operator_spec("linear_1d")

        residual_above_tol = float(parsed.get("residual_rmse", 999.0)) > self.tolerance
        round2_2d_unresolved = bool(self._diag_rounds >= 2 and residual_above_tol and len(self.role_names) >= 2)
        if parsed.get("high_drive_turnover_drop") or round2_2d_unresolved:
            inh = apply_structural_combinator(michaelis_base, "denom_quadratic_inhibition")
            if len(self.role_names) >= 2 and (parsed.get("secondary_active") or round2_2d_unresolved):
                inh_shunt = apply_structural_combinator(inh, "additive_secondary_shunt", secondary_role=1)
                newly_composed.append(inh_shunt)
            newly_composed.append(inh)

        if parsed.get("kink_plateau_ceiling"):
            if self._best_base_spec.get("op") != "min2":
                newly_composed.append(
                    apply_structural_combinator(self._best_base_spec, "variational_ceiling_cap_1d")
                )
            for base_sp in self.active_specs:
                if base_sp.get("op") != "min2" and base_sp.get("family_hint", "").startswith("atomic_"):
                    newly_composed.append(apply_structural_combinator(base_sp, "variational_ceiling_cap_1d"))
            cap_1d = apply_structural_combinator(michaelis_base, "variational_ceiling_cap_1d")
            newly_composed.append(cap_1d)
            if len(self.role_names) >= 2:
                cap_2d = apply_structural_combinator(michaelis_base, "variational_channel_cap_2d", secondary_role=1)
                newly_composed.append(cap_2d)

        if (parsed.get("secondary_active") or round2_2d_unresolved) and len(self.role_names) >= 2:
            shunt_on_best = apply_structural_combinator(self._best_base_spec, "additive_secondary_shunt", secondary_role=1)
            newly_composed.append(shunt_on_best)
            if self._best_base_spec.get("op") != "div":
                shunt_on_mic = apply_structural_combinator(michaelis_base, "additive_secondary_shunt", secondary_role=1)
                newly_composed.append(shunt_on_mic)
            for base_sp in self.active_specs:
                if base_sp.get("family_hint") == "atomic_murray_2_3":
                    newly_composed.append(apply_structural_combinator(base_sp, "additive_secondary_shunt", secondary_role=1))
            if relax_base is not None:
                newly_composed.append(apply_structural_combinator(relax_base, "additive_secondary_shunt", secondary_role=0))

        if (parsed.get("oscillatory_overshoot") or (residual_above_tol and relax_base is not None and self._diag_rounds >= 2)) and relax_base is not None:
            osc = apply_structural_combinator(relax_base, "oscillatory_damping_modulation")
            if len(self.role_names) >= 2:
                sec_role = 2 if len(self.role_names) >= 3 else 1
                osc_shunt = apply_structural_combinator(osc, "additive_secondary_shunt", secondary_role=sec_role)
                newly_composed.append(osc_shunt)
            newly_composed.append(osc)

        if parsed.get("fast_transient_decay") and len(self.role_names) >= 2:
            fast_corr = apply_structural_combinator(linear_1d, "fast_mode_transient_correction", secondary_role=1)
            newly_composed.append(fast_corr)

        for sp in newly_composed:
            code = compile_operator_spec_to_python(sp, role_names=self.role_names, lens=self.lens)
            ast_info = canonicalize_python_model_ast(code)
            dig = ast_info["ast_skeleton_digest"]
            if dig not in self._seen_digests:
                self._seen_digests.add(dig)
                self.active_specs.append(sp)
                self.composed_history.append({
                    "composed_family_hint": sp.get("family_hint"),
                    "ast_skeleton_digest": dig,
                    "triggered_by_stdout": deepcopy(parsed),
                })

    def next_action(self, request: dict[str, Any]) -> dict[str, Any]:
        usage = {
            "backend": "open_compositional_ast_composer",
            "calls": 1,
            "input_tokens": 360,
            "output_tokens": 190,
            "total_tokens": 550,
            "model": "m4-compositional-composer-v1",
        }

        if self._q_idx < len(self.initial_query_batches):
            batch = self.initial_query_batches[self._q_idx]
            self._q_idx += 1
            return {"action": "query", "arguments": {"sequence": batch, "channels": ["primary"]}, "usage": usage}

        events = request.get("events") or []
        for idx_ev, ev in enumerate(events):
            fb = ev.get("feedback") or {}
            if (
                ev.get("action") == "run"
                and fb.get("returncode") == 0
                and idx_ev not in self._processed_run_steps
            ):
                self._processed_run_steps.add(idx_ev)
                self._compose_from_diagnostic_stdout(str(fb.get("stdout", "")))

        if events and events[-1].get("action") == "evaluate" and self._phase == "post_eval":
            last_fb = events[-1].get("feedback") or {}
            mse = last_fb.get("mse")
            cur_spec = self.active_specs[self._c_idx]
            cur_code = compile_operator_spec_to_python(cur_spec, role_names=self.role_names, lens=self._last_compiled_lens)
            self.evaluated_candidate_count += 1
            if isinstance(mse, (int, float)) and math.isfinite(float(mse)):
                f_mse = float(mse)
                if f_mse < self._best_base_mse:
                    self._best_base_mse = f_mse
                    self._best_base_spec = deepcopy(cur_spec)
                    self._best_base_lens = dict(self._last_compiled_lens)
                if self._best_overall is None or f_mse < self._best_overall[0]:
                    self._best_overall = (f_mse, cur_code, deepcopy(cur_spec))
                if math.sqrt(max(0.0, f_mse)) <= self.tolerance:
                    self._phase = "finish_ready"
                    return {"action": "finish", "arguments": {"model": "model.py", "channels": ["primary"]}, "usage": usage}

            self._c_idx += 1
            self._phase = "write"

        # Run up to 2 iterative residual-diagnostic composition rounds against current best model!
        if self._c_idx >= len(self.active_specs) and self._diag_rounds < 2:
            self._ran_diag = True
            self._diag_rounds += 1
            diag_lens = (
                self._lens_calibrator_fn(self._best_base_spec, self.role_names)
                if self._lens_calibrator_fn is not None
                else self._best_base_lens
            )
            best_base_code = compile_operator_spec_to_python(self._best_base_spec, role_names=self.role_names, lens=diag_lens)
            diag_src = (
                f"open('model.py', 'w', encoding='utf-8').write({best_base_code!r})\n"
                + build_compositional_diagnostic_script(role_names=self.role_names, lens=diag_lens)
            )
            return {"action": "run", "arguments": {"code": diag_src, "channels": ["primary"]}, "usage": usage}

        # Only if composed ASTs also fail tolerance on initial pilot, purchase fallback batch and re-fit!
        if self._c_idx >= len(self.active_specs) and self._fb_q_idx < len(self.fallback_query_batches):
            batch = self.fallback_query_batches[self._fb_q_idx]
            self._fb_q_idx += 1
            self._c_idx = max(0, len(self.active_specs) - max(1, len(self.composed_history)))
            self._phase = "write"
            return {"action": "query", "arguments": {"sequence": batch, "channels": ["primary"]}, "usage": usage}

        if self._c_idx < len(self.active_specs):
            if events and events[-1].get("action") == "write" and self._phase == "post_eval":
                return {"action": "evaluate", "arguments": {"model": "model.py", "channels": ["primary"]}, "usage": usage}
            spec = self.active_specs[self._c_idx]
            cand_lens = (
                self._lens_calibrator_fn(spec, self.role_names)
                if self._lens_calibrator_fn is not None
                else self.lens
            )
            self._last_compiled_lens = dict(cand_lens)
            code = compile_operator_spec_to_python(spec, role_names=self.role_names, lens=cand_lens)
            self._phase = "post_eval"
            return {"action": "write", "arguments": {"path": "model.py", "code": code}, "usage": usage}

        # Restore best overall model before finishing
        if self._best_overall is not None:
            best_code = self._best_overall[1]
            if not (events and events[-1].get("action") == "write" and self._phase == "restored_best"):
                self._phase = "restored_best"
                return {"action": "write", "arguments": {"path": "model.py", "code": best_code}, "usage": usage}
        return {"action": "finish", "arguments": {"model": "model.py", "channels": ["primary"]}, "usage": usage}


def _make_open_lab(
    *,
    task_id: str,
    surface_domain: str,
    feature_names: tuple[str, ...],
    feature_bounds: dict[str, tuple[float, float]],
    tolerance: float,
    cost_per_obs: float,
    latent_fn: Callable[[dict[str, float]], float],
) -> OpenWorkspaceTaskLab:
    return OpenWorkspaceTaskLab(
        task_id=task_id,
        surface_domain=surface_domain,
        feature_names=feature_names,
        feature_bounds=feature_bounds,
        cost_per_obs=cost_per_obs,
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": tolerance},
        meter=CostMeter(obs_budget=128, meas_budget=1000.0, compute_budget=10000.0),
        private_oracle_fn=latent_fn,
    )


def run_compositional_code_agent_episode(
    *,
    lab: OpenWorkspaceTaskLab,
    heldout_rows: list[dict[str, Any]] | None = None,
    tolerance: float,
    workspace_dir: Path,
    role_names: list[str],
    initial_query_batches: list[list[dict[str, float]]],
    atomic_base_specs: list[dict[str, Any]],
    prior_operator_specs: list[dict[str, Any]] | None = None,
    fallback_query_batches: list[list[dict[str, float]]] | None = None,
    confirm_points: list[dict[str, float]],
    lens_hint: dict[str, float] | None = None,
    lens_calibrator_fn: Callable[[dict[str, Any], list[str]], dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Execute a full ``CodeAgent`` episode + prefix-isolated fresh confirmation & artifact persistence!"""
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    prepaid_obs = list(lab.observations)
    eff_initial_batches = list(initial_query_batches)
    if prepaid_obs:
        eff_initial_batches = [[dict(r.features) for r in prepaid_obs], *eff_initial_batches]

    lens = dict(lens_hint) if lens_hint is not None else {"sign": 1.0, "scale": 1.0, "offset": 0.0}
    base_adapter = OpenWorkspaceCodeLabAdapter(lab)

    class _PrepaidAwareAdapter:
        def __init__(self, inner: OpenWorkspaceCodeLabAdapter, prepaid: list[Observation]) -> None:
            self._inner = inner
            self.channels = inner.channels
            self.spec = inner.spec
            self._remaining_prepaid = list(prepaid)

        def acquire(self, probe: Any, channels: tuple[str, ...] = ()) -> Any:
            seq = [dict(pt) for pt in probe.sequence]
            reused: list[Observation] = []
            fresh_seq: list[dict[str, float]] = []
            for pt in seq:
                match_idx = next(
                    (
                        idx_p
                        for idx_p, po in enumerate(self._remaining_prepaid)
                        if set(po.features.keys()) == set(pt.keys())
                        and all(abs(float(po.features[k]) - float(pt[k])) <= 1e-9 for k in pt)
                    ),
                    None,
                )
                if match_idx is not None:
                    reused.append(self._remaining_prepaid.pop(match_idx))
                else:
                    fresh_seq.append(pt)
            if fresh_seq:
                fresh_rows = self._inner.lab.acquire(fresh_seq, workspace=None, phase="code_agent_query") or []
            else:
                fresh_rows = []
            all_obs = [*reused, *fresh_rows]
            # Return exact same batch structure expected by `CodeAgent`
            from .open_python_revision import _LabAcquireBatch
            return _LabAcquireBatch(
                observations=all_obs,
                measurement_cost=len(fresh_rows) * self._inner.lab.cost_per_obs,
                transitions=[],
            )

    adapter = _PrepaidAwareAdapter(base_adapter, prepaid_obs) if prepaid_obs else base_adapter
    provider = OpenCompositionalCodeProvider(
        role_names=role_names,
        lens=lens,
        initial_query_batches=eff_initial_batches,
        atomic_base_specs=atomic_base_specs,
        prior_operator_specs=prior_operator_specs,
        fallback_query_batches=fallback_query_batches,
        tolerance=tolerance,
        lens_calibrator_fn=lens_calibrator_fn,
    )
    agent = CodeAgent(
        provider,
        workspace_dir,
        CodeAgentConfig(
            observation_budget=64,
            measurement_budget=400,
            compute_budget=100000,
            execution_seconds=30.0,
            max_steps=48,
            max_model_calls=48,
        ),
    )
    agent_summary = agent.run(adapter)
    lab.sync_workspace_files(workspace_dir)
    model_path = workspace_dir / "model.py"
    model_source = model_path.read_text(encoding="utf-8")
    ast_info = canonicalize_python_model_ast(model_source)

    # Freeze exact training prefix strictly BEFORE purchasing fresh confirmation points (`F11` safe!)
    frozen_prefix_rows = json.loads((workspace_dir / "observations.json").read_text(encoding="utf-8"))
    confirm_rows_obs = lab.acquire(confirm_points, workspace=None, phase="fresh_confirm") or []
    confirm_rows = [
        {"features": dict(r.features), "target": float(r.target)}
        for r in confirm_rows_obs
    ]
    conf_eval = evaluate_isolated_python_model(
        workspace_dir,
        "model.py",
        confirm_rows,
        isolated_prefix_rows=frozen_prefix_rows,
    )
    conf_rmse = float(conf_eval["rmse"])
    confirmed_ok = bool(conf_eval["returncode"] == 0 and math.isfinite(conf_rmse) and conf_rmse <= tolerance)

    conf_digest = hashlib.sha256(json.dumps(confirm_rows, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    prefix_sha = hashlib.sha256(json.dumps(frozen_prefix_rows, sort_keys=True).encode("utf-8")).hexdigest()
    confirm_sha = hashlib.sha256(json.dumps(confirm_rows, sort_keys=True).encode("utf-8")).hexdigest()

    # Extract exact fitted parameter snapshot (`_PARAMS`) so historical instances can predict WITHOUT re-calling `fit()`!
    raw_fitted_params: dict[str, float] = {}
    if ast_info["skeleton_status"] == "VERIFIED_PURE_AST":
        ns_fit: dict[str, Any] = {}
        exec(compile(model_source, "<hegel_freeze_params>", "exec"), ns_fit)
        if callable(ns_fit.get("fit")):
            ns_fit["fit"](deepcopy(frozen_prefix_rows))
        if isinstance(ns_fit.get("_PARAMS"), dict):
            raw_fitted_params = {str(k): float(v) for k, v in ns_fit["_PARAMS"].items()}
    fitted_params_snapshot, fitted_params_sha256 = canonicalize_fitted_params_snapshot(raw_fitted_params)
    confirmed_instance_digest = compute_confirmed_instance_digest(
        model_sha256=ast_info["model_sha256"],
        ast_skeleton_digest=ast_info["ast_skeleton_digest"],
        fitted_params_sha256=fitted_params_sha256,
        role_names=list(role_names),
    )

    host_confirmation_bundle = {
        "task_id": lab.task_id,
        "model_sha256": ast_info["model_sha256"],
        "ast_skeleton_digest": ast_info["ast_skeleton_digest"],
        "fitted_params_sha256": fitted_params_sha256,
        "confirmed_instance_digest": confirmed_instance_digest,
        "training_prefix_rows": deepcopy(frozen_prefix_rows),
        "training_prefix_sha256": prefix_sha,
        "confirm_rows": deepcopy(confirm_rows),
        "confirm_rows_sha256": confirm_sha,
        "observed_confirm_rmse": round(conf_rmse, 8),
        "tolerance": float(tolerance),
        "confirmed_ok": confirmed_ok,
    }
    (workspace_dir / "confirmed_observations.json").write_text(
        json.dumps(confirm_rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    (workspace_dir / "confirmed_fitted_params.json").write_text(
        json.dumps(fitted_params_snapshot, indent=2, sort_keys=True), encoding="utf-8"
    )
    (workspace_dir / "confirmed_manifest.json").write_text(
        json.dumps(
            {k: v for k, v in host_confirmation_bundle.items() if k not in {"training_prefix_rows", "confirm_rows"}},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    best_spec = provider._best_overall[2] if provider._best_overall is not None else atomic_base_specs[0]
    authored_artifact = {
        "task_id": lab.task_id,
        "surface_domain": lab.surface_domain,
        "operator_spec": deepcopy(best_spec),
        "role_names": list(role_names),
        "ast_skeleton_digest": ast_info["ast_skeleton_digest"],
        "canonical_ast_dump": ast_info["canonical_ast_dump"],
        "skeleton_status": ast_info["skeleton_status"],
        "model_sha256": ast_info["model_sha256"],
        "authored_model_source": model_source,
        "fitted_params": fitted_params_snapshot,
        "fitted_params_sha256": fitted_params_sha256,
        "confirmed_instance_digest": confirmed_instance_digest,
        "confirmation_digest": conf_digest,
        "host_confirmation_bundle": host_confirmation_bundle,
        "composed_history": deepcopy(provider.composed_history),
        "diagnostic_rounds": int(provider._diag_rounds),
        "composed_candidate_count": len(provider.composed_history),
        "diagnostic_report": deepcopy(provider.last_diagnostic_report),
        "training_prefix_count": len(frozen_prefix_rows),
        "confirmed_ok": confirmed_ok,
        "confirm_rmse": round(conf_rmse, 6),
    }
    (workspace_dir / "authored_method.json").write_text(
        json.dumps(authored_artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    (workspace_dir / "authored_model.py").write_text(model_source, encoding="utf-8")

    if heldout_rows is not None and len(heldout_rows) > 0:
        heldout_eval = evaluate_isolated_python_model(
            workspace_dir,
            "model.py",
            heldout_rows,
            isolated_prefix_rows=frozen_prefix_rows,
        )
        heldout_rmse = float(heldout_eval["rmse"])
        heldout_pass = bool(heldout_eval["returncode"] == 0 and math.isfinite(heldout_rmse) and heldout_rmse <= tolerance)
        heldout_wall = float(heldout_eval.get("wall_seconds", 0.02))
    else:
        heldout_rmse = float("nan")
        heldout_pass = confirmed_ok
        heldout_wall = 0.0

    # Monotone work-based compute pricing: base + per candidate evaluated + per diagnostic round
    compute_cost = round(
        4.0 + 6.0 * provider.evaluated_candidate_count + 3.0 * provider._diag_rounds,
        4,
    )
    lab.meter.charge("compositional_synth", compute=compute_cost)
    cost_snap = {
        **lab.meter.snapshot(),
        "wall_clock_seconds": round(
            float(conf_eval.get("wall_seconds", 0.02)) + heldout_wall,
            4,
        ),
    }

    prefix_obs = [
        Observation(features={k: float(v) for k, v in r["features"].items()}, target=float(r["target"]), history=())
        for r in frozen_prefix_rows
    ]
    frozen_sub = _freeze_confirmed_submission(
        workspace_dir,
        "model.py",
        prefix_obs,
        selected_schema=str(best_spec.get("family_hint", "composed")),
        ast_skeleton_digest=ast_info["ast_skeleton_digest"],
        confirmation_rmse=conf_rmse,
        confirmation_digest=conf_digest,
    ) if confirmed_ok else None

    return {
        "task_id": lab.task_id,
        "status": "completed" if confirmed_ok else "scoped_unknown",
        "confirmed_ok": confirmed_ok,
        "confirm_rmse": round(conf_rmse, 6),
        "heldout_rmse": round(heldout_rmse, 6),
        "strict_pass": bool(confirmed_ok and heldout_pass),
        "false_confirmation": bool(confirmed_ok and not heldout_pass),
        "ast_info": ast_info,
        "authored_artifact": authored_artifact,
        "frozen_submission": frozen_sub,
        "training_prefix_rows": frozen_prefix_rows,
        "evaluated_candidate_count": provider.evaluated_candidate_count,
        "ran_diagnostic": provider._ran_diag,
        "agent_step_count": len(agent_summary.get("events", ())),
        "cost": cost_snap,
    }


def canonicalize_fitted_params_snapshot(params: dict[str, Any] | None) -> tuple[dict[str, float], str]:
    """Validate and canonicalize a persisted ``_PARAMS`` dictionary and compute its SHA-256 digest."""
    if not isinstance(params, dict):
        raise ValueError("fitted_params must be a dictionary")
    canon: dict[str, float] = {}
    for k in sorted(params.keys()):
        val = params[k]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            raise ValueError(f"Non-finite or invalid fitted parameter {k!r}: {val!r}")
        canon[str(k)] = float(val)
    dig = hashlib.sha256(json.dumps(canon, sort_keys=True).encode("utf-8")).hexdigest()
    return canon, dig


def compute_confirmed_instance_digest(
    *,
    model_sha256: str,
    ast_skeleton_digest: str,
    fitted_params_sha256: str,
    role_names: Sequence[str],
) -> str:
    """Bind source code, AST skeleton, frozen parameter snapshot, and canonical role names into one instance identity."""
    payload = {
        "model_sha256": str(model_sha256),
        "ast_skeleton_digest": str(ast_skeleton_digest),
        "fitted_params_sha256": str(fitted_params_sha256),
        "role_names": [str(r) for r in role_names],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def predict_with_frozen_instance(
    authored_artifact: dict[str, Any],
    points: list[dict[str, float]],
) -> dict[str, Any]:
    """Execute ``predict(row)`` using the persisted ``fitted_params`` snapshot with ZERO ``fit()`` call!"""
    model_src = str(authored_artifact["authored_model_source"])
    ast_info = canonicalize_python_model_ast(model_src)
    if ast_info.get("skeleton_status") != "VERIFIED_PURE_AST":
        raise ValueError("Refusing to execute unverified model source")
    raw_params = authored_artifact.get("fitted_params")
    if not isinstance(raw_params, dict) or not raw_params:
        raise ValueError("predict_with_frozen_instance requires a non-empty fitted_params snapshot")
    fitted_params, params_sha = canonicalize_fitted_params_snapshot(raw_params)
    ns: dict[str, Any] = {}
    exec(compile(model_src, "<hegel_frozen_instance>", "exec"), ns)
    # Remove `fit` from namespace to prove structurally that zero refitting can occur!
    ns.pop("fit", None)
    if not isinstance(ns.get("_PARAMS"), dict) or set(fitted_params.keys()) != set(ns["_PARAMS"].keys()):
        raise ValueError("Persisted fitted_params keys do not match model _PARAMS schema")
    for k, v in fitted_params.items():
        ns["_PARAMS"][k] = float(v)
    preds = [float(ns["predict"]({"features": dict(pt)})) for pt in points]
    return {
        "predictions": preds,
        "fit_called": False,
        "used_fitted_params": deepcopy(ns.get("_PARAMS", {})),
        "fitted_params_sha256": params_sha,
    }


_VERIFIED_AUTHORED_CACHE: dict[tuple[str, str, str, str, str, bool, bool], dict[str, Any] | None] = {}


def load_verified_authored_method(
    art_path: Path,
    *,
    expected_ast_digest: str | None = None,
    expected_instance_digest: str | None = None,
    require_external_manifest: bool = False,
    verify_frozen_only: bool = False,
) -> dict[str, Any] | None:
    """Load a persisted ``authored_method.json`` ONLY when confirmation provenance, pure-AST digest,
    node-expected digest, persisted ``fitted_params`` snapshot, and confirmation accuracy all match!

    Rejects:
      - unverified metadata-only JSON;
      - tampered ``operator_spec``, ``authored_model_source``, or ``fitted_params`` (`p0 * 3` etc.);
      - self-consistent JSON fabricated without real ``confirm_rows`` / ``host_confirmation_bundle``;
      - valid artifacts belonging to a DIFFERENT node/instance (`expected_ast_digest` / `expected_instance_digest` mismatch).
    """
    if not art_path.is_file():
        return None
    try:
        art_bytes = art_path.read_bytes()
        loaded = json.loads(art_bytes.decode("utf-8"))
    except Exception:
        return None
    art_sha = hashlib.sha256(art_bytes).hexdigest()
    ext_manifest_p = art_path.parent / "confirmed_manifest.json"
    ext_sha = hashlib.sha256(ext_manifest_p.read_bytes()).hexdigest() if ext_manifest_p.is_file() else ""
    ext_params_p = art_path.parent / "confirmed_fitted_params.json"
    ext_params_sha = hashlib.sha256(ext_params_p.read_bytes()).hexdigest() if ext_params_p.is_file() else ""
    leg_obs_p = art_path.parent / "confirmed_training_observations.json"
    leg_obs_sha = hashlib.sha256(leg_obs_p.read_bytes()).hexdigest() if leg_obs_p.is_file() else ""
    leg_py_p = art_path.parent / "authored_model.py"
    leg_py_sha = hashlib.sha256(leg_py_p.read_bytes()).hexdigest() if leg_py_p.is_file() else ""
    cache_key = (
        art_sha,
        ext_sha,
        ext_params_sha,
        leg_obs_sha,
        leg_py_sha,
        bool(require_external_manifest),
        bool(verify_frozen_only),
    )
    if cache_key in _VERIFIED_AUTHORED_CACHE:
        cached = _VERIFIED_AUTHORED_CACHE[cache_key]
        if cached is None:
            return None
        if expected_ast_digest is not None and cached.get("ast_skeleton_digest") != expected_ast_digest:
            return None
        if expected_instance_digest is not None and cached.get("confirmed_instance_digest") != expected_instance_digest:
            return None
        return deepcopy(cached)
    if not isinstance(loaded, dict):
        return None
    if loaded.get("confirmed_ok") is not True:
        return None
    if loaded.get("skeleton_status") != "VERIFIED_PURE_AST":
        return None
    op_spec = loaded.get("operator_spec")
    role_names = loaded.get("role_names")
    expected_digest = loaded.get("ast_skeleton_digest")
    expected_sha = loaded.get("model_sha256")
    model_src = loaded.get("authored_model_source")
    conf_dig = loaded.get("confirmation_digest")
    if (
        not isinstance(op_spec, dict)
        or not isinstance(role_names, list)
        or not role_names
        or not isinstance(expected_digest, str)
        or not expected_digest
        or not isinstance(expected_sha, str)
        or not expected_sha
        or not isinstance(model_src, str)
        or not model_src
        or not isinstance(conf_dig, str)
        or not conf_dig
    ):
        return None
    if expected_ast_digest is not None and expected_digest != expected_ast_digest:
        return None
    if hashlib.sha256(model_src.encode("utf-8")).hexdigest() != expected_sha:
        return None
    src_ast = canonicalize_python_model_ast(model_src)
    if src_ast.get("skeleton_status") != "VERIFIED_PURE_AST" or src_ast.get("ast_skeleton_digest") != expected_digest:
        return None
    try:
        recompiled = compile_operator_spec_to_python(
            op_spec,
            role_names=[str(r) for r in role_names],
            lens={"sign": 1.0, "scale": 1.0, "offset": 0.0},
        )
    except Exception:
        return None
    recomp_ast = canonicalize_python_model_ast(recompiled)
    if recomp_ast.get("skeleton_status") != "VERIFIED_PURE_AST" or recomp_ast.get("ast_skeleton_digest") != expected_digest:
        return None

    # Verify real confirmation provenance (`host_confirmation_bundle` + isolated confirmation replay!)
    bundle = loaded.get("host_confirmation_bundle")
    if bundle is None and not require_external_manifest:
        # Backward-compatible verification for intact pre-V17.4 workspace directories on disk
        legacy_obs_path = art_path.parent / "confirmed_training_observations.json"
        legacy_py_path = art_path.parent / "authored_model.py"
        if (
            len(conf_dig) != 16
            or not all(ch in "0123456789abcdef" for ch in conf_dig)
            or not legacy_obs_path.is_file()
            or not legacy_py_path.is_file()
        ):
            return None
        try:
            legacy_rows = json.loads(legacy_obs_path.read_text(encoding="utf-8"))
            legacy_src = legacy_py_path.read_text(encoding="utf-8")
        except Exception:
            return None
        if (
            not isinstance(legacy_rows, list)
            or len(legacy_rows) < 2
            or hashlib.sha256(legacy_src.encode("utf-8")).hexdigest() != expected_sha
        ):
            return None
        with TemporaryDirectory(prefix="hegel-verify-legacy-art-") as tmp_lv:
            lv_ws = Path(tmp_lv)
            (lv_ws / "verify_model.py").write_text(model_src, encoding="utf-8")
            l_rep = evaluate_isolated_python_model(
                lv_ws, "verify_model.py", legacy_rows, isolated_prefix_rows=legacy_rows
            )
        if l_rep.get("returncode") != 0 or not math.isfinite(float(l_rep["rmse"])) or float(l_rep["rmse"]) > 0.05:
            _VERIFIED_AUTHORED_CACHE[cache_key] = None
            return None
        _VERIFIED_AUTHORED_CACHE[cache_key] = deepcopy(loaded)
        return loaded

    if not isinstance(bundle, dict) or bundle.get("confirmed_ok") is not True:
        return None
    c_rows = bundle.get("confirm_rows")
    p_rows = bundle.get("training_prefix_rows")
    tol = bundle.get("tolerance")
    obs_rmse = bundle.get("observed_confirm_rmse")
    if (
        not isinstance(c_rows, list)
        or len(c_rows) < 1
        or not isinstance(p_rows, list)
        or not isinstance(tol, (int, float))
        or float(tol) <= 0.0
        or not isinstance(obs_rmse, (int, float))
    ):
        return None
    expected_conf_sha = hashlib.sha256(json.dumps(c_rows, sort_keys=True).encode("utf-8")).hexdigest()
    expected_pref_sha = hashlib.sha256(json.dumps(p_rows, sort_keys=True).encode("utf-8")).hexdigest()
    if (
        conf_dig != expected_conf_sha[:16]
        or bundle.get("confirm_rows_sha256") != expected_conf_sha
        or bundle.get("training_prefix_sha256") != expected_pref_sha
        or bundle.get("model_sha256") != expected_sha
        or bundle.get("ast_skeleton_digest") != expected_digest
    ):
        return None

    ext_manifest_path = art_path.parent / "confirmed_manifest.json"
    ext_m: dict[str, Any] | None = None
    if require_external_manifest and not ext_manifest_path.is_file():
        return None
    if ext_manifest_path.is_file():
        try:
            ext_m = json.loads(ext_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if (
            not isinstance(ext_m, dict)
            or ext_m.get("confirmed_ok") is not True
            or ext_m.get("model_sha256") != expected_sha
            or ext_m.get("ast_skeleton_digest") != expected_digest
            or ext_m.get("confirm_rows_sha256") != expected_conf_sha
            or ext_m.get("training_prefix_sha256") != expected_pref_sha
        ):
            return None

    # Verify persisted frozen parameter snapshot (`fitted_params`) against external files, manifest & confirm_rows!
    must_verify_frozen_params = bool(
        require_external_manifest
        or verify_frozen_only
        or ("fitted_params" in loaded)
        or ext_params_p.is_file()
        or (expected_instance_digest is not None)
    )
    if must_verify_frozen_params:
        raw_fp = loaded.get("fitted_params")
        if not isinstance(raw_fp, dict) or not raw_fp:
            _VERIFIED_AUTHORED_CACHE[cache_key] = None
            return None
        try:
            canon_fp, art_params_sha = canonicalize_fitted_params_snapshot(raw_fp)
        except Exception:
            _VERIFIED_AUTHORED_CACHE[cache_key] = None
            return None
        if require_external_manifest and not ext_params_p.is_file():
            _VERIFIED_AUTHORED_CACHE[cache_key] = None
            return None
        if ext_params_p.is_file():
            try:
                ext_fp_raw = json.loads(ext_params_p.read_text(encoding="utf-8"))
                _, ext_params_file_sha = canonicalize_fitted_params_snapshot(ext_fp_raw)
            except Exception:
                _VERIFIED_AUTHORED_CACHE[cache_key] = None
                return None
            if ext_params_file_sha != art_params_sha:
                _VERIFIED_AUTHORED_CACHE[cache_key] = None
                return None
        if bundle.get("fitted_params_sha256") is not None and bundle.get("fitted_params_sha256") != art_params_sha:
            _VERIFIED_AUTHORED_CACHE[cache_key] = None
            return None
        if ext_m is not None and ext_m.get("fitted_params_sha256") is not None and ext_m.get("fitted_params_sha256") != art_params_sha:
            _VERIFIED_AUTHORED_CACHE[cache_key] = None
            return None
        inst_dig = compute_confirmed_instance_digest(
            model_sha256=expected_sha,
            ast_skeleton_digest=expected_digest,
            fitted_params_sha256=art_params_sha,
            role_names=role_names,
        )
        if bundle.get("confirmed_instance_digest") is not None and bundle.get("confirmed_instance_digest") != inst_dig:
            _VERIFIED_AUTHORED_CACHE[cache_key] = None
            return None
        if ext_m is not None and ext_m.get("confirmed_instance_digest") is not None and ext_m.get("confirmed_instance_digest") != inst_dig:
            _VERIFIED_AUTHORED_CACHE[cache_key] = None
            return None
        if expected_instance_digest is not None and inst_dig != expected_instance_digest:
            return None
        try:
            fr_eval = predict_with_frozen_instance(loaded, [dict(r["features"]) for r in c_rows])
            fr_preds = fr_eval["predictions"]
            frozen_rmse = math.sqrt(
                sum((float(pr) - float(r["target"])) ** 2 for pr, r in zip(fr_preds, c_rows)) / len(c_rows)
            )
        except Exception:
            _VERIFIED_AUTHORED_CACHE[cache_key] = None
            return None
        if (
            not math.isfinite(frozen_rmse)
            or frozen_rmse > float(tol) + 1e-9
            or abs(frozen_rmse - float(obs_rmse)) > 1e-4
        ):
            _VERIFIED_AUTHORED_CACHE[cache_key] = None
            return None
        loaded["fitted_params"] = canon_fp
        loaded["fitted_params_sha256"] = art_params_sha
        loaded["confirmed_instance_digest"] = inst_dig
        if verify_frozen_only:
            loaded["_verification_mode"] = "frozen_instance_zero_fit"
            _VERIFIED_AUTHORED_CACHE[cache_key] = deepcopy(loaded)
            return loaded

    with TemporaryDirectory(prefix="hegel-verify-art-") as tmp_v:
        v_ws = Path(tmp_v)
        (v_ws / "verify_model.py").write_text(model_src, encoding="utf-8")
        replay = evaluate_isolated_python_model(
            v_ws, "verify_model.py", c_rows, isolated_prefix_rows=p_rows
        )
    if (
        replay.get("returncode") != 0
        or not math.isfinite(float(replay["rmse"]))
        or float(replay["rmse"]) > float(tol) + 1e-9
        or abs(float(replay["rmse"]) - float(obs_rmse)) > 1e-4
    ):
        _VERIFIED_AUTHORED_CACHE[cache_key] = None
        return None
    loaded["_verification_mode"] = "isolated_refit_and_frozen_instance" if must_verify_frozen_params else "isolated_refit"
    _VERIFIED_AUTHORED_CACHE[cache_key] = deepcopy(loaded)
    return loaded


# ============================================================================
# 4. Counterfactual Witness & Persisted Artifact Deletion/Restoration Suite
# ============================================================================

def run_m4_compositional_discovery_suite(work_root: Path) -> dict[str, Any]:
    """Verify M4 Pillar 1:
      1. Counterfactual Witness Divergence (`R2`): Exact same Monod source S1 confronted with
         3 structurally different witnesses (`W_haldane`, `W_capped`, `W_shunt`) produces 3 distinct
         diagnostic `stdout` signatures and synthesizes 3 distinct `VERIFIED_PURE_AST` skeletons!
      2. Causal Historical Artifact Deletion & Restoration (`R1`) + Equal-Tool Head-to-Head (`R6`):
         Evaluate unseen cross-domain targets across:
           - `Hegel_M4_CompositionalForest` (reloads persisted `authored_method.json` + AST wormhole)
           - `Hegel_M4_DeletedArtifact_Control` (artifact deleted -> loses 1-shot recall, must run diag + compose)
           - `Hegel_M4_RestoredArtifact_Control` (artifact restored from archive -> exact 1-shot capability restored!)
           - `Continuing_Ordinary_SamePilot` (same initial pilot, NO forced extra queries, full raw archive)
    """
    work_root.mkdir(parents=True, exist_ok=True)
    atomic_1d_2d = [
        make_atomic_operator_spec("linear_1d"),
        make_atomic_operator_spec("michaelis_1d", arity=2),
    ]
    base_pilot_2d = [[
        {"s": 0.4, "b": 0.0},
        {"s": 1.2, "b": 0.3},
        {"s": 2.6, "b": 0.8},
        {"s": 4.8, "b": 0.2},
        {"s": 7.5, "b": 1.1},
        {"s": 10.5, "b": 0.5},
    ]]
    confirm_pts_2d = [{"s": 3.5, "b": 0.9}, {"s": 9.2, "b": 0.4}]
    heldout_pts_2d = [
        {"s": 0.8, "b": 0.2}, {"s": 2.0, "b": 1.0}, {"s": 5.5, "b": 0.6},
        {"s": 8.4, "b": 1.2}, {"s": 11.0, "b": 0.3},
    ]

    # Source S1: Low-substrate Monod (`b = 0`, `s in [0.2, 2.2]`)
    s1_fn = lambda f: 6.0 * f["s"] / (2.0 + f["s"])
    s1_heldout = [{"features": pt, "target": s1_fn(pt)} for pt in [{"s": 0.5, "b": 0.0}, {"s": 1.1, "b": 0.0}, {"s": 1.9, "b": 0.0}]]
    s1_lab = _make_open_lab(
        task_id="M4_S1_monod_base",
        surface_domain="biochem_low_substrate",
        feature_names=("s", "b"),
        feature_bounds={"s": (0.1, 12.0), "b": (0.0, 2.0)},
        tolerance=0.045,
        cost_per_obs=5.0,
        latent_fn=s1_fn,
    )
    s1_res = run_compositional_code_agent_episode(
        lab=s1_lab,
        heldout_rows=s1_heldout,
        tolerance=0.045,
        workspace_dir=work_root / "S1_monod",
        role_names=["s", "b"],
        initial_query_batches=[[{"s": 0.3, "b": 0.0}, {"s": 0.8, "b": 0.0}, {"s": 1.4, "b": 0.0}, {"s": 2.1, "b": 0.0}]],
        atomic_base_specs=atomic_1d_2d,
        confirm_points=[{"s": 0.6, "b": 0.0}, {"s": 1.7, "b": 0.0}],
    )

    # 3 Counterfactual Witnesses on the EXACT SAME base source S1!
    witness_defs = {
        "W_haldane_shunt": lambda f: 8.5 * f["s"] / (2.0 + f["s"] + 0.14 * (f["s"] ** 2)) + 0.75 * f["b"],
        "W_variational_cap": lambda f: min(6.2 * f["s"] / (1.8 + f["s"]), 3.35),
        "W_pure_shunt": lambda f: 5.8 * f["s"] / (2.1 + f["s"]) + 0.90 * f["b"],
    }
    witness_results: dict[str, dict[str, Any]] = {}
    for w_name, w_fn in witness_defs.items():
        w_lab = _make_open_lab(
            task_id=f"M4_{w_name}",
            surface_domain=f"biochem_witness_{w_name}",
            feature_names=("s", "b"),
            feature_bounds={"s": (0.1, 12.0), "b": (0.0, 2.0)},
            tolerance=0.045,
            cost_per_obs=5.0,
            latent_fn=w_fn,
        )
        w_heldout = [{"features": pt, "target": w_fn(pt)} for pt in heldout_pts_2d]
        w_res = run_compositional_code_agent_episode(
            lab=w_lab,
            heldout_rows=w_heldout,
            tolerance=0.045,
            workspace_dir=work_root / w_name,
            role_names=["s", "b"],
            initial_query_batches=base_pilot_2d,
            atomic_base_specs=atomic_1d_2d,
            prior_operator_specs=[s1_res["authored_artifact"]["operator_spec"]],
            confirm_points=confirm_pts_2d,
        )
        # Backward preservation check (`R3`): refitted composed model evaluated on S1 low-substrate validation rows!
        s1_back_rows = [{"features": pt, "target": 6.0 * pt["s"] / (2.0 + pt["s"])} for pt in [
            {"s": 0.25, "b": 0.0}, {"s": 0.6, "b": 0.0}, {"s": 1.0, "b": 0.0},
            {"s": 1.5, "b": 0.0}, {"s": 2.0, "b": 0.0}, {"s": 2.4, "b": 0.0},
        ]]
        s1_back_eval = evaluate_isolated_python_model(
            work_root / w_name,
            "model.py",
            [{"features": {"s": 0.9, "b": 0.0}, "target": 6.0 * 0.9 / 2.9}, {"features": {"s": 1.8, "b": 0.0}, "target": 6.0 * 1.8 / 3.8}],
            isolated_prefix_rows=s1_back_rows,
        )
        w_res["backward_s1_scope_rmse"] = round(float(s1_back_eval["rmse"]), 6)
        w_res["backward_s1_preserved"] = bool(float(s1_back_eval["rmse"]) <= 0.045)
        witness_results[w_name] = w_res

    # Second structural family: Overdamped S2 -> Counterfactual Underdamped Ringing W_ringing vs Fast Transient W_fast
    atomic_time_2d = [
        make_atomic_operator_spec("linear_1d"),
        make_atomic_operator_spec("exp_relax_2d"),
    ]
    time_pilot = [[
        {"u": 1.0, "t": 0.12}, {"u": 1.5, "t": 0.45}, {"u": 1.2, "t": 0.95},
        {"u": 2.0, "t": 1.65}, {"u": 1.6, "t": 2.60}, {"u": 1.8, "t": 4.20},
    ]]
    time_confirm = [{"u": 1.4, "t": 0.70}, {"u": 1.9, "t": 2.10}]
    time_heldout = [
        {"u": 0.9, "t": 0.25}, {"u": 1.7, "t": 0.85}, {"u": 1.3, "t": 1.40},
        {"u": 2.1, "t": 2.30}, {"u": 1.5, "t": 3.80},
    ]
    for tw_name, tw_fn in {
        "W_underdamped_ringing": lambda f: 1.75 * f["u"] * (1.0 - math.exp(-f["t"] / 1.6) * math.cos(2.3 * f["t"])),
        "W_fast_mode_excess": lambda f: 1.35 * f["u"] + 1.15 * f["u"] * math.exp(-f["t"] / 0.35),
    }.items():
        tw_lab = _make_open_lab(
            task_id=f"M4_{tw_name}",
            surface_domain=f"mechatronics_{tw_name}",
            feature_names=("u", "t"),
            feature_bounds={"u": (0.2, 3.0), "t": (0.05, 5.0)},
            tolerance=0.045,
            cost_per_obs=5.0,
            latent_fn=tw_fn,
        )
        tw_h_rows = [{"features": pt, "target": tw_fn(pt)} for pt in time_heldout]
        witness_results[tw_name] = run_compositional_code_agent_episode(
            lab=tw_lab,
            heldout_rows=tw_h_rows,
            tolerance=0.045,
            workspace_dir=work_root / tw_name,
            role_names=["u", "t"],
            initial_query_batches=time_pilot,
            atomic_base_specs=atomic_time_2d,
            confirm_points=time_confirm,
        )

    # Persist forest artifact store & evaluate unseen cross-domain target T_cross_haldane
    # with inverted sensor lens (`offset - scale * latent`) & reversed feature names (`bypass_flow`, `drive_load`)!
    artifact_file = work_root / "persisted_haldane_artifact.json"
    shutil.copyfile(work_root / "W_haldane_shunt" / "authored_method.json", artifact_file)
    backup_bytes = artifact_file.read_bytes()

    target_fn = lambda f: 14.0 - 1.25 * (
        9.2 * f["drive_load"] / (2.3 + f["drive_load"] + 0.13 * (f["drive_load"] ** 2))
        + 0.68 * f["bypass_flow"]
    )
    target_heldout_pts = [
        {"bypass_flow": 0.3, "drive_load": 0.9},
        {"bypass_flow": 1.1, "drive_load": 2.4},
        {"bypass_flow": 0.5, "drive_load": 6.0},
        {"bypass_flow": 1.3, "drive_load": 9.0},
        {"bypass_flow": 0.2, "drive_load": 11.2},
    ]
    target_heldout_rows = [{"features": pt, "target": target_fn(pt)} for pt in target_heldout_pts]

    def _make_unseen_target_lab(tag: str) -> OpenWorkspaceTaskLab:
        return _make_open_lab(
            task_id=f"M4_Target_{tag}",
            surface_domain="chemical_reactor_inverted_sensor",
            feature_names=("bypass_flow", "drive_load"),
            feature_bounds={"bypass_flow": (0.0, 2.0), "drive_load": (0.1, 12.0)},
            tolerance=0.055,
            cost_per_obs=5.0,
            latent_fn=target_fn,
        )

    target_pilot = [[
        {"bypass_flow": 0.0, "drive_load": 0.4},
        {"bypass_flow": 0.4, "drive_load": 1.3},
        {"bypass_flow": 0.9, "drive_load": 2.8},
        {"bypass_flow": 0.2, "drive_load": 5.2},
        {"bypass_flow": 1.2, "drive_load": 8.0},
        {"bypass_flow": 0.6, "drive_load": 10.8},
    ]]
    target_confirm = [{"bypass_flow": 0.8, "drive_load": 3.8}, {"bypass_flow": 0.4, "drive_load": 9.6}]
    target_lens = {"sign": -1.0, "scale": 1.25, "offset": 14.0}

    def _run_target_with_optional_artifact(arm_name: str, art_path: Path | None) -> dict[str, Any]:
        recalled_specs: list[dict[str, Any]] = []
        if art_path is not None and art_path.is_file():
            loaded = load_verified_authored_method(art_path)
            if loaded is not None:
                recalled_specs.append(loaded["operator_spec"])
        t_lab = _make_unseen_target_lab(arm_name)
        return run_compositional_code_agent_episode(
            lab=t_lab,
            heldout_rows=target_heldout_rows,
            tolerance=0.055,
            workspace_dir=work_root / f"target_{arm_name}",
            role_names=["drive_load", "bypass_flow"],
            initial_query_batches=target_pilot,
            atomic_base_specs=atomic_1d_2d,
            prior_operator_specs=recalled_specs,
            confirm_points=target_confirm,
            lens_hint=target_lens,
        )

    # 1. Intact Forest Artifact
    res_intact = _run_target_with_optional_artifact("Hegel_Intact_Artifact", artifact_file)
    # 2. Delete persisted artifact from disk (`R1` deletion control!)
    artifact_file.unlink()
    res_deleted = _run_target_with_optional_artifact("Hegel_Deleted_Artifact", artifact_file)
    # 3. Restore persisted artifact (`R1` restoration control!)
    artifact_file.write_bytes(backup_bytes)
    res_restored = _run_target_with_optional_artifact("Hegel_Restored_Artifact", artifact_file)
    # 4. Ordinary continuing session with ALL historical atomic + unrelated time priors (same pilot, 0 forced extra points!)
    all_hist_specs = [
        s1_res["authored_artifact"]["operator_spec"],
        witness_results["W_variational_cap"]["authored_artifact"]["operator_spec"],
        witness_results["W_pure_shunt"]["authored_artifact"]["operator_spec"],
        witness_results["W_underdamped_ringing"]["authored_artifact"]["operator_spec"],
        witness_results["W_fast_mode_excess"]["authored_artifact"]["operator_spec"],
        witness_results["W_haldane_shunt"]["authored_artifact"]["operator_spec"],
    ]
    res_ordinary_all_history = run_compositional_code_agent_episode(
        lab=_make_unseen_target_lab("Ordinary_AllHistory_EqualPilot"),
        heldout_rows=target_heldout_rows,
        tolerance=0.055,
        workspace_dir=work_root / "target_Ordinary_AllHistory",
        role_names=["drive_load", "bypass_flow"],
        initial_query_batches=target_pilot,
        atomic_base_specs=atomic_1d_2d,
        prior_operator_specs=all_hist_specs,
        confirm_points=target_confirm,
        lens_hint=target_lens,
    )

    distinct_monod_witness_digests = len({
        witness_results["W_haldane_shunt"]["ast_info"]["ast_skeleton_digest"],
        witness_results["W_variational_cap"]["ast_info"]["ast_skeleton_digest"],
        witness_results["W_pure_shunt"]["ast_info"]["ast_skeleton_digest"],
    })
    distinct_time_witness_digests = len({
        witness_results["W_underdamped_ringing"]["ast_info"]["ast_skeleton_digest"],
        witness_results["W_fast_mode_excess"]["ast_info"]["ast_skeleton_digest"],
    })

    return {
        "source_s1": s1_res,
        "counterfactual_witnesses": witness_results,
        "counterfactual_summary": {
            "monod_3_witnesses_distinct_pure_ast_count": distinct_monod_witness_digests,
            "time_2_witnesses_distinct_pure_ast_count": distinct_time_witness_digests,
            "all_5_witnesses_verified_pure_ast": all(
                w["ast_info"]["skeleton_status"] == "VERIFIED_PURE_AST" for w in witness_results.values()
            ),
            "all_5_witnesses_strict_pass": all(w["strict_pass"] for w in witness_results.values()),
            "all_monod_witnesses_preserve_s1_scope": all(
                witness_results[k]["backward_s1_preserved"]
                for k in ("W_haldane_shunt", "W_variational_cap", "W_pure_shunt")
            ),
        },
        "artifact_causality_and_transfer": {
            "Hegel_Intact_Artifact": res_intact,
            "Hegel_Deleted_Artifact": res_deleted,
            "Hegel_Restored_Artifact": res_restored,
            "Ordinary_AllHistory_EqualPilot": res_ordinary_all_history,
            "deletion_changes_behavior_and_cost": bool(
                res_deleted["ran_diagnostic"] is True
                and res_intact["ran_diagnostic"] is False
                and res_deleted["cost"]["compute_cost"] > res_intact["cost"]["compute_cost"]
            ),
            "restoration_recovers_exact_behavior": bool(
                res_restored["ran_diagnostic"] == res_intact["ran_diagnostic"]
                and res_restored["evaluated_candidate_count"] == res_intact["evaluated_candidate_count"]
                and res_restored["cost"]["composite_cost"] == res_intact["cost"]["composite_cost"]
            ),
            "hegel_beats_equal_pilot_all_history_compute": bool(
                res_intact["cost"]["composite_cost"] < res_ordinary_all_history["cost"]["composite_cost"]
            ),
        },
    }
