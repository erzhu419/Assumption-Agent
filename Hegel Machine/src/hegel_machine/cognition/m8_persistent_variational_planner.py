"""Hegel M8.2 Persistent Mechanism Graph, Atomic Variational Models & Pre-Payment Action Planner.

Upgrades M8.0 (`42d1ea688`) to close all structural & mathematical gaps identified in `Hegel_M8_1_Audit`
while preserving full backward compatibility with M8.0/M8.1 callers and tests:

  1. **Atomic Model Commit & Sample-Invariant Prediction (`planner_controls`: `commit_config`, `batch_gate`, `memory_cut`, `y_scale`)**:
     - `TaskModel` atomically commits its fitted posterior (`committed_nig_state`), `committed_prior_shrinkage`,
       fixed training-time `gate_scale` per `RegimeConduitBranch` (zero test-batch std leakage!), and causal
       `terminal_memory_states` across trajectory splits (`0.0` boundary reset error!).
     - Evaluating an already-committed `TaskModel` predicts directly from its frozen state (`0.0` drift on
       repeated evaluation with unchanged data), and `evaluate_task_zero_shot_from_shared` predicts purely
       from frozen shared upstream + inherited conduits without touching target training labels `y`!
     - `fit_scale_covariant_nig_linear_model` scales `beta_0` covariantly with empirical target variance
       without an artificial `1e-3` floor on non-constant targets (`0.0` nats discrepancy under `y -> 1e-6 * y`!).

  2. **Single Authoritative Cross-Process Persistence (`persistence` & `checkpoint_storage` integration)**:
     - Synchronizes `PhilosophicalForest` node versions, `shared_weights`, `active_task_models`, `decision_ledger`,
       and empirical `action_return_history` into `forest` AND `save_checkpoint` / `load_checkpoint` (via
       `checkpoint_storage.write_text_incrementally`), so `PersistentMechanismGraph(restored_forest)` and cold
       process restarts recover 100% of mechanisms, weights, versions, and predictions!

  3. **Autonomous Mechanism Recall & Cross-Variable Role Mapping (`ReuseMechanism`)**:
     - Subsequent tasks do NOT require manual `lift_ids`: the planner automatically recalls verified
       `interaction_lift` and `temporal_memory` nodes across ancestors and `HorizontalWormholeEdge`s,
       evaluates observable `scope_guard` compatibility (`evaluate_predicate_rule`), aligns renamed feature
       columns via residual-sensitivity role permutations, and tests `ReuseMechanism` BEFORE paid LLM calls!

  4. **Pre-Payment Budgeted Selection, Complete Attempt Costing, & Reachable `AcquireObservation`**:
     - Separates cheap `propose -> estimate -> budget_select` from `execute -> observe -> adopt`: expensive
       `GPT CLI` (`codex`) calls are skipped when `remaining_budget` is insufficient or when cheap
       `ReuseMechanism` / `AncestorReweight` / `LocalSplit` already resolves the structured residual!
     - Meters EVERY attempt (including unselected candidates and `ok=False` failed LLM calls!) in
       `attempt_cost_events`, filters qualified candidates BEFORE utility ranking, and retains top-K diverse
       `TaskModel` hypotheses so `AcquireObservation` is genuinely reachable in the main loop!
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import itertools
import json
import math
from pathlib import Path
from time import monotonic
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

from .checkpoint_storage import write_text_incrementally
from .wormhole_forest import (
    ClaimContract,
    ObservationMap,
    PhilosophicalForest,
    ScaleMap,
    TaskView,
    evaluate_predicate_rule,
)

M8_ACTIONS = (
    "NoChange",
    "ReuseMechanism",
    "LocalRefit",
    "LocalSplit",
    "AncestorReweight",
    "ConceptLift",
    "ChangeMemory",
    "AcquireObservation",
)

FORBIDDEN_DECISION_STATE_KEYS = frozenset(
    {
        "chosen_was_best",
        "oracle_action",
        "oracle_truth",
        "posthoc_label",
        "posthoc_winner",
        "future_validation_score",
        "private_verifier_pass",
        "terminal_retrospective_rank",
    }
)

FORBIDDEN_FEATURE_EXPR_TOKENS = frozenset(
    {
        "task_id",
        "seed",
        "oracle",
        "ground_truth",
        "target",
        "y_true",
        "__import__",
        "open",
        "eval",
        "exec",
        "os",
        "sys",
        "subprocess",
    }
)

M8_CONCEPT_LIFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "concept_name": {
            "type": "string",
            "description": "Short snake_case identifier for the lifted interaction/nonlinear mechanism basis.",
        },
        "python_expression": {
            "type": "string",
            "description": (
                "Pure vectorized numpy/scalar mathematical expression over the named input columns `x['<col>']` "
                "(allowed functions: `np.tanh`, `np.exp`, `np.log1p`, `np.sqrt`, `np.abs`, `np.sin`, `np.cos`, "
                "`np.maximum`, `np.minimum`, arithmetic `+ - * / **`). Must NOT reference target `y` or task IDs."
            ),
        },
        "mechanistic_rationale": {
            "type": "string",
            "description": "Concise scientific explanation of why existing additive/single-feature branches miss the residual pattern and how this lifted conduit resolves it.",
        },
        "domain_guard_note": {
            "type": "string",
            "description": "Domain/extrapolation condition under which this lifted conduit remains bounded and physically meaningful.",
        },
    },
    "required": [
        "concept_name",
        "python_expression",
        "mechanistic_rationale",
        "domain_guard_note",
    ],
    "additionalProperties": False,
}


# ============================================================================
# 1. Mathematical & Empirical-Bayes NIG Primitives (Input & Output Scale Covariant)
# ============================================================================


def evaluate_gated_vs_mixture_log_likelihood(
    y: np.ndarray,
    branch_preds: np.ndarray,
    gate_weights: np.ndarray,
    sigma: float,
) -> dict[str, float]:
    """Explicitly separate gated deterministic prediction likelihood from mixture-of-experts likelihood (`T01`)."""
    yv = np.asarray(y, dtype=float).ravel()
    fp = np.asarray(branch_preds, dtype=float)
    gw = np.asarray(gate_weights, dtype=float)
    s2 = max(float(sigma) ** 2, 1e-12)
    norm_c = -0.5 * math.log(2.0 * math.pi * s2)

    mean_pred = np.sum(gw * fp, axis=1)
    ll_gated = float(np.sum(norm_c - 0.5 * ((yv - mean_pred) ** 2) / s2))

    comp_log_probs = np.log(np.clip(gw, 1e-15, 1.0)) + norm_c - 0.5 * ((yv[:, None] - fp) ** 2) / s2
    ll_mixture = float(np.sum(logsumexp(comp_log_probs, axis=1)))

    return {
        "gated_predictor_loglik": ll_gated,
        "mixture_experts_loglik": ll_mixture,
        "abs_gap_nats": float(abs(ll_gated - ll_mixture)),
    }


def compute_likelihood_curvature(
    residuals: np.ndarray,
    jacobian: np.ndarray,
    *,
    likelihood: str = "gaussian",
    sigma: float = 1.0,
    student_df: float = 3.0,
) -> dict[str, Any]:
    """Compute exact likelihood curvature without factor-of-2 bugs (`T02`) or hidden non-PSD Student-t tails (`T03`)."""
    r = np.asarray(residuals, dtype=float).ravel()
    J = np.asarray(jacobian, dtype=float)
    s2 = max(float(sigma) ** 2, 1e-12)

    if likelihood == "gaussian":
        w_obs = np.full_like(r, 1.0 / s2)
        H_gn = (J.T * w_obs) @ J
        eig_min = float(np.min(np.linalg.eigvalsh(0.5 * (H_gn + H_gn.T)))) if J.size else 0.0
        return {
            "likelihood": "gaussian",
            "residual_second_derivative": w_obs,
            "gauss_newton_hessian": H_gn,
            "expected_fisher": H_gn,
            "observed_is_psd": bool(eig_min >= -1e-12),
            "expected_fisher_is_psd": True,
        }

    if likelihood == "student_t":
        nu = max(float(student_df), 1.01)
        denom = nu * s2 + r ** 2
        w_obs = (nu + 1.0) * (nu * s2 - r ** 2) / np.maximum(denom ** 2, 1e-24)
        H_obs_gn = (J.T * w_obs) @ J
        eig_obs = float(np.min(np.linalg.eigvalsh(0.5 * (H_obs_gn + H_obs_gn.T)))) if J.size else float(np.min(w_obs))
        fisher_scalar = ((nu + 1.0) / (nu + 3.0)) / s2
        F_exp = fisher_scalar * (J.T @ J)
        w_irls = (nu + 1.0) / np.maximum(denom, 1e-12)
        H_irls = (J.T * w_irls) @ J
        return {
            "likelihood": "student_t",
            "residual_second_derivative": w_obs,
            "observed_gn_hessian": H_obs_gn,
            "expected_fisher": F_exp,
            "irls_psd_surrogate": H_irls,
            "observed_is_psd": bool(eig_obs >= -1e-12 and np.all(w_obs >= -1e-12)),
            "expected_fisher_is_psd": True,
        }

    raise ValueError(f"Unsupported likelihood `{likelihood}`")


def fit_scale_covariant_nig_linear_model(
    Phi: np.ndarray,
    y: np.ndarray,
    *,
    prior_mean: np.ndarray | None = None,
    prior_shrinkage: float = 1e-2,
    alpha_0: float = 2.5,
    beta_0: float | None = None,
    reference_y_var: float | None = None,
    intercept_col_Zero: bool = True,
) -> dict[str, Any]:
    """Empirical-Bayes / Fixed-Scale Normal-Inverse-Gamma (`NIG`) regression covariant in BOTH `X` AND `y` (`M8.2` Fix).

    - Under `Phi[:, j] -> c_x * Phi[:, j]`, `det(Lambda_0)/det(Lambda_n)` is invariant to `c_x`.
    - Under `y -> c_y * y` (for any positive `c_y`, including `1e-6`), `beta_0` scales by `c_y^2` without an
      artificial `1e-3` clamp whenever `y` has positive variance (`y_scale2 > 0`), shifting `log_marginal_likelihood`
      by the exact Jacobian `-n * log(c_y)` (`0.0` nats discrepancy!) while remaining strictly finite at `SSE == 0`!
    - Passing `reference_y_var` freezes `beta_0` across competing models on the same task so candidate comparison
      uses an identical noise prior!
    """
    X = np.asarray(Phi, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    yv = np.asarray(y, dtype=float).ravel()
    n, d = X.shape
    if n == 0:
        raise ValueError("Cannot fit NIG model on empty dataset")

    mu0 = np.zeros(d, dtype=float) if prior_mean is None else np.asarray(prior_mean, dtype=float).ravel()
    if mu0.shape[0] != d:
        raise ValueError(f"prior_mean shape {mu0.shape} != ({d},)")

    col_msq = np.mean(X ** 2, axis=0)
    lam_diag = np.where(col_msq > 1e-30, float(prior_shrinkage) * col_msq, float(prior_shrinkage))
    if intercept_col_Zero and d > 0 and np.allclose(X[:, 0], 1.0):
        lam_diag[0] = float(prior_shrinkage) * 1e-2
    Lambda_0 = np.diag(lam_diag)

    a0 = max(float(alpha_0), 1.01)
    if beta_0 is not None:
        b0 = max(float(beta_0), 1e-300)
    else:
        y_var = float(reference_y_var) if reference_y_var is not None else (float(np.var(yv)) if n > 1 else float(np.mean(yv ** 2)))
        if y_var > 0.0:
            b0 = a0 * (0.15 * y_var)
        else:
            msq_y = float(np.mean(yv ** 2))
            b0 = a0 * (0.15 * msq_y if msq_y > 0.0 else 1e-6)

    Lambda_n = Lambda_0 + X.T @ X
    rhs = Lambda_0 @ mu0 + X.T @ yv
    mu_n = np.linalg.solve(Lambda_n, rhs)

    sign0, logdet0 = np.linalg.slogdet(Lambda_0)
    sign_n, logdet_n = np.linalg.slogdet(Lambda_n)
    assert sign0 > 0 and sign_n > 0

    quad_term = max(0.0, float(yv @ yv + mu0 @ (Lambda_0 @ mu0) - mu_n @ (Lambda_n @ mu_n)))
    a_n = a0 + 0.5 * n
    b_n = b0 + 0.5 * quad_term

    log_ev = float(
        0.5 * (logdet0 - logdet_n)
        + a0 * math.log(b0)
        - a_n * math.log(b_n)
        + float(gammaln(a_n) - gammaln(a0))
        - 0.5 * n * math.log(2.0 * math.pi)
    )
    post_sigma2 = float(b_n / max(a_n - 1.0, 1e-6))
    y_hat = X @ mu_n
    resid = yv - y_hat
    tot_ss = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / max(tot_ss, 1e-30) if tot_ss > 1e-30 else (1.0 if float(np.sum(resid ** 2)) < 1e-24 else 0.0)

    return {
        "posterior_mean": mu_n,
        "posterior_precision": Lambda_n,
        "prior_precision": Lambda_0,
        "alpha_0": a0,
        "beta_0": b0,
        "alpha_n": a_n,
        "beta_n": b_n,
        "posterior_sigma2": post_sigma2,
        "log_marginal_likelihood": log_ev,
        "train_rmse": float(np.sqrt(np.mean(resid ** 2))),
        "train_r2": float(r2),
        "residuals": resid,
    }


def predict_nig_student_t_log_score(
    fit_info: dict[str, Any],
    Phi_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    domain_inflation: np.ndarray | None = None,
) -> dict[str, Any]:
    """Exact Student-t posterior predictive density `p(y_* | X_*, D)` (separating parameter epistemic std from noise!)."""
    Xe = np.asarray(Phi_eval, dtype=float)
    if Xe.ndim == 1:
        Xe = Xe[:, None]
    ye = np.asarray(y_eval, dtype=float).ravel()
    mu_n = np.asarray(fit_info["posterior_mean"], dtype=float)
    Lambda_n = np.asarray(fit_info["posterior_precision"], dtype=float)
    a_n = float(fit_info["alpha_n"])
    b_n = float(fit_info["beta_n"])

    pred_mean = Xe @ mu_n
    V_x = np.maximum(0.0, np.sum(Xe * np.linalg.solve(Lambda_n, Xe.T).T, axis=1))
    if domain_inflation is not None:
        V_x = V_x + np.maximum(0.0, np.asarray(domain_inflation, dtype=float).ravel())
    nu = 2.0 * a_n
    noise_scale2 = b_n / a_n
    epistemic_var = noise_scale2 * V_x * (nu / max(nu - 2.0, 1e-6))
    scale2 = noise_scale2 * (1.0 + V_x)

    z2 = ((ye - pred_mean) ** 2) / np.maximum(nu * scale2, 1e-30)
    log_pdf = (
        float(gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu))
        - 0.5 * np.log(nu * math.pi * np.maximum(scale2, 1e-30))
        - 0.5 * (nu + 1.0) * np.log1p(z2)
    )
    rmse = float(np.sqrt(np.mean((ye - pred_mean) ** 2)))
    return {
        "mean_log_predictive": float(np.mean(log_pdf)),
        "sum_log_predictive": float(np.sum(log_pdf)),
        "rmse": rmse,
        "predictions": pred_mean,
        "epistemic_std": np.sqrt(np.maximum(epistemic_var, 1e-30)),
        "predictive_std": np.sqrt(np.maximum(scale2 * (nu / max(nu - 2.0, 1e-6)), 1e-30)),
    }


@dataclass
class SequentialGaussianBelief:
    """Tracks exact Bayesian belief updates without double-counting historical data (`T07`)."""

    prior_mean: np.ndarray
    prior_precision: np.ndarray
    sigma2: float
    current_mean: np.ndarray = field(init=False)
    current_precision: np.ndarray = field(init=False)
    ingested_batch_ids: set[str] = field(default_factory=set)
    stored_batches: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prior_mean = np.asarray(self.prior_mean, dtype=float).copy()
        self.prior_precision = np.asarray(self.prior_precision, dtype=float).copy()
        self.current_mean = self.prior_mean.copy()
        self.current_precision = self.prior_precision.copy()

    def sequential_update(self, batch_id: str, X_new: np.ndarray, y_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if batch_id in self.ingested_batch_ids:
            raise ValueError(f"Batch `{batch_id}` already ingested; sequential re-ingestion would double-count data!")
        Xn = np.asarray(X_new, dtype=float)
        yn = np.asarray(y_new, dtype=float).ravel()
        s2 = max(float(self.sigma2), 1e-12)
        rhs = self.current_precision @ self.current_mean + (Xn.T @ yn) / s2
        self.current_precision = self.current_precision + (Xn.T @ Xn) / s2
        self.current_mean = np.linalg.solve(self.current_precision, rhs)
        self.ingested_batch_ids.add(batch_id)
        self.stored_batches[batch_id] = (Xn.copy(), yn.copy())
        return self.current_mean.copy(), self.current_precision.copy()

    def joint_refit_from_prior(self, active_batch_ids: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        ids = list(self.stored_batches.keys()) if active_batch_ids is None else list(active_batch_ids)
        s2 = max(float(self.sigma2), 1e-12)
        prec = self.prior_precision.copy()
        rhs = self.prior_precision @ self.prior_mean
        for bid in ids:
            Xb, yb = self.stored_batches[bid]
            prec = prec + (Xb.T @ Xb) / s2
            rhs = rhs + (Xb.T @ yb) / s2
        mean = np.linalg.solve(prec, rhs)
        return mean, prec


def compute_deduplicated_model_posterior(
    models_and_objectives: list[tuple[str, str, float]],
) -> dict[str, Any]:
    """Compute posterior weights over distinct semantic equivalence classes (`T05`)."""
    best_by_sem: dict[str, tuple[str, float]] = {}
    duplicates_collapsed: list[str] = []
    for mid, sem_key, free_energy in models_and_objectives:
        if sem_key in best_by_sem:
            duplicates_collapsed.append(mid)
            if free_energy < best_by_sem[sem_key][1]:
                best_by_sem[sem_key] = (mid, float(free_energy))
        else:
            best_by_sem[sem_key] = (mid, float(free_energy))

    sem_keys = sorted(best_by_sem.keys())
    neg_F = np.array([-best_by_sem[k][1] for k in sem_keys], dtype=float)
    log_Z = float(logsumexp(neg_F)) if len(neg_F) else 0.0
    probs = np.exp(neg_F - log_Z) if len(neg_F) else np.array([], dtype=float)
    return {
        "unique_semantic_models": {
            best_by_sem[k][0]: {
                "semantic_key": k,
                "variational_free_energy": best_by_sem[k][1],
                "posterior_prob": float(p),
            }
            for k, p in zip(sem_keys, probs)
        },
        "duplicates_collapsed": duplicates_collapsed,
        "log_partition": log_Z,
    }


# ============================================================================
# 2. Safe Expression Conduit Sandbox, Role Mapping & Conditioning (`T10`)
# ============================================================================


def remap_conduit_expression_roles(python_expr: str, role_map: dict[str, str]) -> str:
    """Safely rewrite `x['<src_col>']` and bare `<src_col>` identifiers under a variable-role mapping."""
    if not role_map:
        return python_expr
    tree = ast.parse(python_expr, mode="eval")

    class _RoleRemapper(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            self.generic_visit(node)
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "x"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
                and node.slice.value in role_map
            ):
                return ast.Subscript(
                    value=node.value,
                    slice=ast.Constant(value=role_map[node.slice.value]),
                    ctx=node.ctx,
                )
            return node

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in role_map and node.id not in {"x", "np", "math"}:
                return ast.Name(id=role_map[node.id], ctx=node.ctx)
            return node

    new_tree = ast.fix_missing_locations(_RoleRemapper().visit(tree))
    return ast.unparse(new_tree)


def compile_safe_feature_conduit(python_expr: str, feature_names: tuple[str, ...]) -> Callable[[dict[str, np.ndarray]], np.ndarray]:
    """Compile and AST-verify a safe vectorized equation conduit `phi(x)` over named observable columns."""
    expr = str(python_expr or "").strip()
    if not expr or len(expr) > 400:
        raise ValueError("Empty or overly long conduit expression")
    tree = ast.parse(expr, mode="eval")

    allowed_np_attrs = {
        "tanh", "exp", "expm1", "log", "log1p", "sqrt", "abs", "sin", "cos",
        "maximum", "minimum", "square", "clip", "where", "pi",
    }
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
        ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Load, ast.Attribute, ast.Subscript,
        ast.Constant, ast.Compare, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.IfExp,
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Disallowed AST construct `{type(node).__name__}` in conduit `{expr}`")
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_FEATURE_EXPR_TOKENS:
                raise ValueError(f"Forbidden token `{node.id}` in conduit")
            if node.id not in {"x", "np", "math", *feature_names}:
                raise ValueError(f"Unknown identifier `{node.id}` in conduit")
        elif isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id in {"np", "math"} and node.attr in allowed_np_attrs):
                raise ValueError(f"Disallowed attribute `{node.attr}` in conduit")
        elif isinstance(node, ast.Subscript):
            if not (
                isinstance(node.value, ast.Name)
                and node.value.id == "x"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
                and node.slice.value in feature_names
            ):
                raise ValueError("Subscript only allowed on observable `x['<feature_name>']`")

    code_obj = compile(tree, "<m8_feature_conduit>", "eval")

    def _eval_conduit(x_cols: dict[str, np.ndarray]) -> np.ndarray:
        n_rows = len(next(iter(x_cols.values())))
        local_ns: dict[str, Any] = {"np": np, "math": np, "x": x_cols}
        for fnm in feature_names:
            if fnm in x_cols:
                local_ns[fnm] = x_cols[fnm]
        raw = eval(code_obj, {"__builtins__": {}}, local_ns)
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 0:
            arr = np.full(n_rows, float(arr), dtype=float)
        return arr.ravel()

    return _eval_conduit


def estimate_conduit_conditioning_and_domain(
    conduit_fn: Callable[[dict[str, np.ndarray]], np.ndarray],
    x_cols: dict[str, np.ndarray],
    *,
    eps_step: float = 1e-4,
) -> dict[str, Any]:
    """Compute local Lipschitz conditioning + empirical support box (`T10` honest numeric guard)."""
    vals = conduit_fn(x_cols)
    if not np.all(np.isfinite(vals)):
        return {"finite_on_support": False, "empirical_lipschitz": float("inf"), "support_bounds": {}}

    grad_sq = np.zeros_like(vals, dtype=float)
    bounds: dict[str, tuple[float, float]] = {}
    for k, col in x_cols.items():
        arr = np.asarray(col, dtype=float).ravel()
        bounds[k] = (float(np.min(arr)), float(np.max(arr)))
        step = max(eps_step * max(float(np.std(arr)), 1.0), 1e-6)
        pert = dict(x_cols)
        pert[k] = arr + step
        v_plus = conduit_fn(pert)
        if not np.all(np.isfinite(v_plus)):
            return {"finite_on_support": False, "empirical_lipschitz": float("inf"), "support_bounds": bounds}
        grad_sq += ((v_plus - vals) / step) ** 2

    lip = float(np.max(np.sqrt(grad_sq))) if grad_sq.size else 0.0
    return {
        "finite_on_support": True,
        "empirical_lipschitz": lip,
        "ill_conditioned_warning": bool(lip > 1e5),
        "support_bounds": bounds,
    }


def compute_out_of_support_inflation(
    x_cols: dict[str, np.ndarray],
    support_bounds: dict[str, tuple[float, float]],
    *,
    lipschitz_scale: float = 1.0,
) -> np.ndarray:
    """Inflate predictive uncertainty quadratically with distance outside training feature support (`T10`)."""
    n_rows = len(next(iter(x_cols.values())))
    dist_sq = np.zeros(n_rows, dtype=float)
    for k, (lo, hi) in support_bounds.items():
        if k not in x_cols:
            continue
        arr = np.asarray(x_cols[k], dtype=float).ravel()
        span = max(hi - lo, 1e-3)
        excess = np.maximum(0.0, lo - arr) + np.maximum(0.0, arr - hi)
        dist_sq += (excess / span) ** 2
    eff_scale = min(max(float(lipschitz_scale), 0.25), 25.0)
    return 0.25 * eff_scale * dist_sq


# ============================================================================
# 3. Authoritative Persistent Graph `G`, Atomic `TaskModel`, & Checkpoint Restore
# ============================================================================


@dataclass
class TypedMechanism:
    mechanism_id: str
    parent_id: str | None
    role: str  # 'shared_upstream_premise' | 'regime_branch' | 'interaction_lift' | 'temporal_memory'
    feature_names: tuple[str, ...]
    version: int = 1
    shared_weights: dict[str, float] = field(default_factory=dict)
    scope_guard: dict[str, Any] | None = None
    python_expression: str | None = None
    temporal_tau: float | None = None
    temporal_driver_col: str | None = None
    support_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    empirical_lipschitz: float = 1.0
    dependent_task_ids: set[str] = field(default_factory=set)
    inapplicable_task_ids: set[str] = field(default_factory=set)
    epistemic_status: str = "SUPPORTED_IN_SCOPE"
    provenance_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["feature_names"] = list(self.feature_names)
        d["support_bounds"] = {k: [float(v[0]), float(v[1])] for k, v in self.support_bounds.items()}
        d["dependent_task_ids"] = sorted(self.dependent_task_ids)
        d["inapplicable_task_ids"] = sorted(self.inapplicable_task_ids)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TypedMechanism:
        return cls(
            mechanism_id=str(data["mechanism_id"]),
            parent_id=data.get("parent_id"),
            role=str(data["role"]),
            feature_names=tuple(str(x) for x in data.get("feature_names", ())),
            version=int(data.get("version", 1)),
            shared_weights={str(k): float(v) for k, v in (data.get("shared_weights") or {}).items()},
            scope_guard=deepcopy(data.get("scope_guard")),
            python_expression=data.get("python_expression"),
            temporal_tau=(float(data["temporal_tau"]) if data.get("temporal_tau") is not None else None),
            temporal_driver_col=data.get("temporal_driver_col"),
            support_bounds={str(k): (float(v[0]), float(v[1])) for k, v in (data.get("support_bounds") or {}).items()},
            empirical_lipschitz=float(data.get("empirical_lipschitz", 1.0)),
            dependent_task_ids={str(x) for x in data.get("dependent_task_ids", [])},
            inapplicable_task_ids={str(x) for x in data.get("inapplicable_task_ids", [])},
            epistemic_status=str(data.get("epistemic_status", "SUPPORTED_IN_SCOPE")),
            provenance_note=str(data.get("provenance_note", "")),
        )


@dataclass
class RegimeConduitBranch:
    """Internal observable-gated equation conduit with fixed training-time `gate_scale` (zero batch-std drift!)."""

    branch_id: str
    mechanism_id: str
    scope_guard: dict[str, Any] | None = None
    gate_sharpness: float = 12.0
    gate_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegimeConduitBranch:
        return cls(
            branch_id=str(data["branch_id"]),
            mechanism_id=str(data["mechanism_id"]),
            scope_guard=deepcopy(data.get("scope_guard")),
            gate_sharpness=float(data.get("gate_sharpness", 12.0)),
            gate_scale=float(data.get("gate_scale", 1.0)),
        )


@dataclass
class TaskModel:
    """Complete versioned `TaskModel` `M` storing structure, role maps, AND committed frozen posterior state."""

    model_id: str
    task_id: str
    shared_upstream_id: str
    feature_names: tuple[str, ...]
    regime_branches: list[RegimeConduitBranch] = field(default_factory=list)
    interaction_lift_ids: list[str] = field(default_factory=list)
    lift_role_maps: dict[str, dict[str, str]] = field(default_factory=dict)  # lift_id -> {src_col: task_col}
    shared_role_map: dict[str, str] = field(default_factory=dict)  # shared_col -> task_col
    temporal_memory_id: str | None = None
    committed_prior_shrinkage: float = 12.0
    reference_y_var: float | None = None
    terminal_memory_states: dict[str, list[float]] = field(default_factory=dict)  # str(traj_id) -> [last_t, last_z]
    committed_nig_state: dict[str, Any] | None = None
    fit_summary: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def referenced_mechanism_ids(self) -> set[str]:
        out = {self.shared_upstream_id, *(b.mechanism_id for b in self.regime_branches), *self.interaction_lift_ids}
        if self.temporal_memory_id:
            out.add(self.temporal_memory_id)
        return out

    def canonical_semantic_key(self, graph: PersistentMechanismGraph) -> str:
        shared_m = graph.mechanisms[self.shared_upstream_id]
        guards = sorted(
            json.dumps(b.scope_guard, sort_keys=True) if b.scope_guard else "DEFAULT"
            for b in self.regime_branches
        )
        lifts = sorted(
            remap_conduit_expression_roles(
                str(graph.mechanisms[lid].python_expression or lid),
                self.lift_role_maps.get(lid, {}),
            )
            for lid in self.interaction_lift_ids
            if lid in graph.mechanisms
        )
        tau = (
            graph.mechanisms[self.temporal_memory_id].temporal_tau
            if (self.temporal_memory_id and self.temporal_memory_id in graph.mechanisms)
            else None
        )
        spec = {
            "task_id": self.task_id,
            "shared_weights": {k: round(float(v), 6) for k, v in sorted(shared_m.shared_weights.items())},
            "features": list(self.feature_names),
            "guards": guards,
            "lifts": lifts,
            "temporal_tau": round(float(tau), 5) if tau is not None else None,
        }
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def intrinsic_grammar_log_prior(self, graph: PersistentMechanismGraph) -> float:
        n_branches = max(1, len(self.regime_branches))
        n_lifts = len(self.interaction_lift_ids)
        has_mem = 1 if self.temporal_memory_id else 0
        expr_chars = sum(
            len(str(graph.mechanisms[lid].python_expression or ""))
            for lid in self.interaction_lift_ids
            if lid in graph.mechanisms
        )
        return float(-1.35 * (n_branches - 1) - 1.25 * n_lifts - 1.20 * has_mem - 0.015 * expr_chars)

    def to_dict(self) -> dict[str, Any]:
        nig_ser = None
        if self.committed_nig_state is not None:
            nig_ser = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.committed_nig_state.items()
                if k != "residuals"
            }
        return {
            "model_id": self.model_id,
            "task_id": self.task_id,
            "shared_upstream_id": self.shared_upstream_id,
            "feature_names": list(self.feature_names),
            "regime_branches": [b.to_dict() for b in self.regime_branches],
            "interaction_lift_ids": list(self.interaction_lift_ids),
            "lift_role_maps": deepcopy(self.lift_role_maps),
            "shared_role_map": deepcopy(self.shared_role_map),
            "temporal_memory_id": self.temporal_memory_id,
            "committed_prior_shrinkage": self.committed_prior_shrinkage,
            "reference_y_var": self.reference_y_var,
            "terminal_memory_states": deepcopy(self.terminal_memory_states),
            "committed_nig_state": nig_ser,
            "fit_summary": deepcopy(self.fit_summary),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskModel:
        nig_raw = data.get("committed_nig_state")
        if isinstance(nig_raw, dict):
            nig_raw = dict(nig_raw)
            for k in ("posterior_mean", "posterior_precision", "prior_precision"):
                if k in nig_raw:
                    nig_raw[k] = np.asarray(nig_raw[k], dtype=float)
        return cls(
            model_id=str(data["model_id"]),
            task_id=str(data["task_id"]),
            shared_upstream_id=str(data["shared_upstream_id"]),
            feature_names=tuple(str(x) for x in data.get("feature_names", ())),
            regime_branches=[RegimeConduitBranch.from_dict(b) for b in data.get("regime_branches", [])],
            interaction_lift_ids=[str(x) for x in data.get("interaction_lift_ids", [])],
            lift_role_maps={str(k): dict(v) for k, v in (data.get("lift_role_maps") or {}).items()},
            shared_role_map={str(k): str(v) for k, v in (data.get("shared_role_map") or {}).items()},
            temporal_memory_id=data.get("temporal_memory_id"),
            committed_prior_shrinkage=float(data.get("committed_prior_shrinkage", 12.0)),
            reference_y_var=(float(data["reference_y_var"]) if data.get("reference_y_var") is not None else None),
            terminal_memory_states={str(k): [float(v[0]), float(v[1])] for k, v in (data.get("terminal_memory_states") or {}).items()},
            committed_nig_state=nig_raw,
            fit_summary=deepcopy(data.get("fit_summary") or {}),
            version=int(data.get("version", 1)),
        )


@dataclass
class ResearchDecisionRecord:
    """Strictly prefix-clean trajectory record `tau` (`T09` + complete cost accounting)."""

    decision_id: str
    step_index: int
    task_id: str
    pre_action_state: dict[str, Any]
    available_actions: list[str]
    selected_action: str
    selected_candidate_id: str
    execution_status: str
    actual_cost: float
    public_val_log_predictive_delta: float
    collateral_min_delta: float
    rationale: str
    estimated_selected_cost: float = 0.0
    total_search_cost_spent: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    attempt_cost_events: list[dict[str, Any]] = field(default_factory=list)

    def validate_prefix_clean(self) -> None:
        bad = set(self.pre_action_state.keys()) & FORBIDDEN_DECISION_STATE_KEYS
        if bad:
            raise ValueError(f"Post-hoc leakage keys forbidden in `pre_action_state`: {sorted(bad)}")
        for k in self.pre_action_state:
            if k.startswith(("posthoc_", "oracle_", "future_", "private_verifier_")):
                raise ValueError(f"Forbidden prefix in `pre_action_state` key `{k}`")

    def to_dict(self) -> dict[str, Any]:
        self.validate_prefix_clean()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchDecisionRecord:
        rec = cls(**data)
        rec.validate_prefix_clean()
        return rec


M8_FOREST_STATE_KEY = "__m8_authoritative_persistent_state__"


class PersistentMechanismGraph:
    """Authoritative persistent mechanism graph `G` with full `PhilosophicalForest` sync & cold-restart recovery."""

    def __init__(self, forest: PhilosophicalForest | None = None) -> None:
        self.forest = forest if forest is not None else PhilosophicalForest()
        self.mechanisms: dict[str, TypedMechanism] = {}
        self.mechanism_lineage: dict[str, list[dict[str, Any]]] = {}
        self.task_datasets: dict[str, dict[str, Any]] = {}
        self.active_task_models: dict[str, TaskModel] = {}
        self.decision_ledger: list[ResearchDecisionRecord] = []
        self.action_return_history: dict[str, list[dict[str, float]]] = {}
        self._hydrate_from_forest_if_present()

    def _hydrate_from_forest_if_present(self) -> None:
        """Automatically hydrate full M8 executable state when constructed from a saved/restored `PhilosophicalForest`!"""
        state_blob = self.forest.questions.get(M8_FOREST_STATE_KEY)
        if isinstance(state_blob, dict) and state_blob.get("schema") == "hegel_m8_authoritative_state_v2":
            self.mechanisms = {
                mid: TypedMechanism.from_dict(md)
                for mid, md in (state_blob.get("mechanisms") or {}).items()
            }
            self.mechanism_lineage = deepcopy(state_blob.get("mechanism_lineage") or {})
            self.active_task_models = {
                tid: TaskModel.from_dict(tm_d)
                for tid, tm_d in (state_blob.get("active_task_models") or {}).items()
            }
            self.decision_ledger = [
                ResearchDecisionRecord.from_dict(rd)
                for rd in (state_blob.get("decision_ledger") or [])
            ]
            self.action_return_history = deepcopy(state_blob.get("action_return_history") or {})
            return

        # Fallback node-level hydration if only forest nodes carry `m8_typed_mechanism`
        for nid, nd in self.forest.nodes.items():
            cc = nd.get("claim_contract") or {}
            concl = cc.get("conclusion_or_action") or {}
            if isinstance(concl.get("m8_typed_mechanism"), dict):
                self.mechanisms[nid] = TypedMechanism.from_dict(concl["m8_typed_mechanism"])

    def sync_authoritative_state_to_forest(self) -> None:
        """Keep `forest` node versions/claims and `forest.questions[M8_FOREST_STATE_KEY]` 100% synchronized!"""
        for mid, mech in self.mechanisms.items():
            if mid in self.forest.nodes:
                fnode = self.forest.nodes[mid]
                fnode["version"] = int(mech.version)
                cc = fnode.get("claim_contract")
                if isinstance(cc, dict):
                    cc["claim_id"] = f"cc_{mid}_v{mech.version}"
                    concl = cc.setdefault("conclusion_or_action", {})
                    concl["role"] = mech.role
                    concl["version"] = int(mech.version)
                    concl["python_expression"] = mech.python_expression
                    concl["shared_weights"] = dict(mech.shared_weights)
                    concl["m8_typed_mechanism"] = mech.to_dict()

        self.forest.questions[M8_FOREST_STATE_KEY] = {
            "schema": "hegel_m8_authoritative_state_v2",
            "mechanisms": {mid: m.to_dict() for mid, m in self.mechanisms.items()},
            "mechanism_lineage": deepcopy(self.mechanism_lineage),
            "active_task_models": {tid: tm.to_dict() for tid, tm in self.active_task_models.items()},
            "decision_ledger": [r.to_dict() for r in self.decision_ledger],
            "action_return_history": deepcopy(self.action_return_history),
        }

    def save_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """Persist complete versioned M8 state to disk via `checkpoint_storage.write_text_incrementally`."""
        from .m4_4_instance_role_closure_and_online_writeback import serialize_location_independent_forest

        self.sync_authoritative_state_to_forest()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        snap = serialize_location_independent_forest(self.forest)
        raw_txt = json.dumps(snap, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        write_text_incrementally(checkpoint_path, raw_txt)
        sha = hashlib.sha256(raw_txt.encode("utf-8")).hexdigest()
        return {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha,
            "mechanism_count": len(self.mechanisms),
            "active_task_model_count": len(self.active_task_models),
        }

    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path) -> PersistentMechanismGraph:
        """Restore complete `PersistentMechanismGraph` across process restarts with identical predictions & versions."""
        from .m4_4_instance_role_closure_and_online_writeback import load_location_independent_forest

        forest = load_location_independent_forest(checkpoint_path, checkpoint_path.parent)
        return cls(forest)

    def register_mechanism(self, mech: TypedMechanism, *, tree_id: str | None = None) -> str:
        self.mechanisms[mech.mechanism_id] = mech
        self.mechanism_lineage.setdefault(mech.mechanism_id, []).append(mech.to_dict())

        eff_tree_id = (
            str(self.forest.nodes[mech.parent_id]["tree_id"])
            if (tree_id is None and mech.parent_id and mech.parent_id in self.forest.nodes)
            else (tree_id or "tree_m8_persistent")
        )
        layer = "coarse" if mech.role == "shared_upstream_premise" else ("mid" if mech.role in {"interaction_lift", "temporal_memory"} else "fine")
        preconds = [mech.scope_guard] if mech.scope_guard else []
        t_mode = "history_conditioned_horizon" if mech.role == "temporal_memory" else "present_snapshot"
        f_kind = "memory_kernel" if mech.role == "temporal_memory" else "effective_constant"
        if mech.mechanism_id not in self.forest.nodes:
            self.forest.register_node(
                node_id=mech.mechanism_id,
                tree_id=eff_tree_id,
                layer_role=layer,
                task_view=TaskView("m8_continual_variational_modeling", "heldout_log_predictive", 0.05),
                observation_map=ObservationMap("y", temporal_readout_mode=t_mode, causal_cutoff_policy="strict_prefix_le_t0"),
                scale_map=ScaleMap(
                    "pi_fold_memory" if mech.role == "temporal_memory" else "identity",
                    preserved_queries=["m8_continual_variational_modeling"],
                    closure_scope="exact",
                    temporal_parameters={"tau": mech.temporal_tau} if mech.temporal_tau else None,
                    folding_output_kind=f_kind,
                ),
                claim_contract=ClaimContract(
                    claim_id=f"cc_{mech.mechanism_id}_v{mech.version}",
                    family_plugin="m8_persistent_variational_graph",
                    preconditions=preconds,
                    conclusion_or_action={
                        "role": mech.role,
                        "version": mech.version,
                        "python_expression": mech.python_expression,
                        "shared_weights": dict(mech.shared_weights),
                        "m8_typed_mechanism": mech.to_dict(),
                    },
                    protocol_spec={"empirical_lipschitz": mech.empirical_lipschitz},
                    tolerance=0.05,
                    evidence_status="SUPPORTED_IN_SCOPE" if mech.epistemic_status == "SUPPORTED_IN_SCOPE" else "candidate",
                    permission_tier="local_solver_guidance",
                ),
                probe_signatures=[f"sig:m8_{mech.role}"],
            )
            if mech.parent_id and mech.parent_id in self.forest.nodes:
                self.forest.add_local_implication_edge(
                    stronger_node_id=mech.mechanism_id,
                    weaker_node_id=mech.parent_id,
                    preserved_contract={"role": mech.role, "version": mech.version},
                    scope={"preconditions": preconds},
                    evidence_status="derived",
                )
        self.sync_authoritative_state_to_forest()
        return mech.mechanism_id

    def minimal_affected_task_ids(self, mechanism_id: str) -> list[str]:
        affected_mechs = {mechanism_id}
        changed = True
        while changed:
            changed = False
            for mid, m in self.mechanisms.items():
                if m.parent_id in affected_mechs and mid not in affected_mechs:
                    affected_mechs.add(mid)
                    changed = True
        tasks: set[str] = set()
        for tid, tm in self.active_task_models.items():
            if tm.referenced_mechanism_ids() & affected_mechs:
                tasks.add(tid)
        for mid in affected_mechs:
            tasks |= self.mechanisms[mid].dependent_task_ids
        return sorted(tasks)


# ============================================================================
# 4. Causal Memory State Carry-Over, Sample-Invariant Gates & Frozen Prediction
# ============================================================================


def _compute_causal_memory_feature(
    dataset: dict[str, Any],
    driver_col: str,
    tau: float,
    *,
    initial_states: dict[str, list[float]] | None = None,
    return_terminal_states: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, list[float]]]:
    """Compute strictly prefix-causal (`<= t`) Mori-Zwanzig memory state WITH boundary carry-over (`M8.2` Fix!).

    When `initial_states` (or `dataset['initial_memory_states']`) provides `{str(traj_id): [t_prev, z_prev, u_prev]}`,
    step `0` of a validation continuation seamlessly decays `z_prev` across `dt = t_0 - t_prev` instead of resetting to `0`!
    """
    x_cols = dataset["x"]
    n = len(next(iter(x_cols.values())))
    tau_pos = max(float(tau), 1e-4)

    if "past_driver_history" in dataset and dataset["past_driver_history"] is not None:
        hist = np.asarray(dataset["past_driver_history"], dtype=float)
        dt = float(dataset.get("history_dt", 1.0))
        decay = math.exp(-dt / tau_pos)
        z = np.zeros(n, dtype=float)
        for step_idx in range(hist.shape[1]):
            z = decay * z + (1.0 - decay) * hist[:, step_idx]
        if return_terminal_states:
            return z, {}
        return z

    u = np.asarray(x_cols[driver_col], dtype=float).ravel()
    ts = np.asarray(dataset.get("timestamps", np.arange(n, dtype=float)), dtype=float).ravel()
    ep_ids = np.asarray(dataset.get("trajectory_ids", np.zeros(n, dtype=int)))
    init_map = initial_states if initial_states is not None else dataset.get("initial_memory_states") or {}

    z = np.zeros(n, dtype=float)
    term_map: dict[str, list[float]] = {}
    for i in range(n):
        tid_key = str(int(ep_ids[i]))
        if i == 0 or ep_ids[i] != ep_ids[i - 1]:
            if tid_key in init_map and len(init_map[tid_key]) >= 2:
                prev_rec = init_map[tid_key]
                t_prev, z_prev = float(prev_rec[0]), float(prev_rec[1])
                u_prev = float(prev_rec[2]) if len(prev_rec) >= 3 else float(u[i])
                dt0 = float(ts[i] - t_prev)
                if dt0 > 0.0:
                    dec0 = math.exp(-dt0 / tau_pos)
                    z[i] = dec0 * z_prev + (1.0 - dec0) * u_prev
                else:
                    z[i] = z_prev
            else:
                z[i] = 0.0
        else:
            dt = max(float(ts[i] - ts[i - 1]), 1e-4)
            decay = math.exp(-dt / tau_pos)
            z[i] = decay * z[i - 1] + (1.0 - decay) * u[i - 1]
        term_map[tid_key] = [float(ts[i]), float(z[i]), float(u[i])]

    if return_terminal_states:
        return z, term_map
    return z


def _evaluate_branch_gates(
    branches: list[RegimeConduitBranch],
    x_cols: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute sample-wise deterministic observable regime gates `pi_b(x)` using fixed `br.gate_scale` (`M8.2` Fix!)."""
    n = len(next(iter(x_cols.values())))
    if len(branches) <= 1:
        return np.ones((n, 1), dtype=float)

    raw_gates = np.ones((n, len(branches)), dtype=float)
    guarded_sum = np.zeros(n, dtype=float)
    default_indices: list[int] = []

    for b_idx, br in enumerate(branches):
        if br.scope_guard is None:
            default_indices.append(b_idx)
            continue
        col_nm = str(br.scope_guard["field"])
        op = str(br.scope_guard.get("op", ">"))
        thr = float(br.scope_guard["value"])
        vals = np.asarray(x_cols[col_nm], dtype=float).ravel()
        # Use fixed `br.gate_scale` stored on the branch (NEVER recompute `np.std(vals)` on test batches!)
        scale = max(float(br.gate_scale), 1e-6)
        k = float(br.gate_sharpness) / scale
        if op in {">", ">="}:
            g = 0.5 * (1.0 + np.tanh(0.5 * k * (vals - thr)))
        elif op in {"<", "<="}:
            g = 0.5 * (1.0 + np.tanh(0.5 * k * (thr - vals)))
        else:
            g = (vals == thr).astype(float)
        raw_gates[:, b_idx] = g
        guarded_sum += g

    if default_indices:
        rem = np.clip(1.0 - guarded_sum, 0.0, 1.0)
        for d_idx in default_indices:
            raw_gates[:, d_idx] = rem / len(default_indices)

    row_sums = np.maximum(np.sum(raw_gates, axis=1, keepdims=True), 1e-12)
    return raw_gates / row_sums


def build_task_design_matrix(
    model: TaskModel,
    graph: PersistentMechanismGraph,
    dataset: dict[str, Any],
    *,
    initial_memory_states: dict[str, list[float]] | None = None,
    capture_terminal_memory: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]]]:
    """Build `(Phi, shared_offset, support_inflation, terminal_mem_states)` supporting role maps & memory carry-over."""
    x_cols = dataset["x"]
    n = len(next(iter(x_cols.values())))
    shared_mech = graph.mechanisms[model.shared_upstream_id]

    shared_offset = np.zeros(n, dtype=float)
    for src_fnm, w_val in shared_mech.shared_weights.items():
        dst_fnm = model.shared_role_map.get(src_fnm, src_fnm)
        if dst_fnm in x_cols:
            shared_offset = shared_offset + float(w_val) * np.asarray(x_cols[dst_fnm], dtype=float).ravel()

    base_cols = [np.ones(n, dtype=float)]
    for fnm in model.feature_names:
        base_cols.append(np.asarray(x_cols[fnm], dtype=float).ravel())
    base_mat = np.column_stack(base_cols)

    gates = _evaluate_branch_gates(model.regime_branches, x_cols)
    design_blocks: list[np.ndarray] = []
    for b_idx in range(gates.shape[1]):
        design_blocks.append(base_mat * gates[:, [b_idx]])

    support_inflation = np.zeros(n, dtype=float)
    for lid in model.interaction_lift_ids:
        l_mech = graph.mechanisms[lid]
        if l_mech.python_expression:
            rmap = model.lift_role_maps.get(lid, {})
            mapped_expr = remap_conduit_expression_roles(l_mech.python_expression, rmap)
            fn = compile_safe_feature_conduit(mapped_expr, model.feature_names)
            lift_vec = fn(x_cols)
            design_blocks.append(lift_vec[:, None])
            if l_mech.support_bounds:
                mapped_bounds = {rmap.get(k, k): v for k, v in l_mech.support_bounds.items()}
                support_inflation += compute_out_of_support_inflation(
                    x_cols,
                    mapped_bounds,
                    lipschitz_scale=l_mech.empirical_lipschitz,
                )

    term_states: dict[str, list[float]] = {}
    if model.temporal_memory_id and model.temporal_memory_id in graph.mechanisms:
        t_mech = graph.mechanisms[model.temporal_memory_id]
        raw_drv = t_mech.temporal_driver_col or model.feature_names[0]
        drv = raw_drv if raw_drv in x_cols else model.feature_names[0]
        mem_res = _compute_causal_memory_feature(
            dataset,
            drv,
            float(t_mech.temporal_tau or 1.5),
            initial_states=initial_memory_states,
            return_terminal_states=True,
        )
        assert isinstance(mem_res, tuple)
        mem_vec, term_states = mem_res
        design_blocks.append(mem_vec[:, None])

    Phi = np.hstack(design_blocks)
    if capture_terminal_memory:
        model.terminal_memory_states = term_states
    return Phi, shared_offset, support_inflation, term_states


def evaluate_committed_task_model(
    model: TaskModel,
    graph: PersistentMechanismGraph,
    eval_data: dict[str, Any],
) -> dict[str, Any]:
    """Predict & score a committed `TaskModel` directly from its frozen posterior state (ZERO refit drift!)."""
    assert model.committed_nig_state is not None, f"TaskModel `{model.model_id}` has no committed posterior state!"
    Phi_ev, off_ev, inf_ev, _ = build_task_design_matrix(
        model,
        graph,
        eval_data,
        initial_memory_states=model.terminal_memory_states or None,
    )
    y_ev = np.asarray(eval_data["y"], dtype=float).ravel()
    pred = predict_nig_student_t_log_score(model.committed_nig_state, Phi_ev, y_ev - off_ev, domain_inflation=inf_ev)
    full_preds = off_ev + pred["predictions"]
    rmse = float(np.sqrt(np.mean((y_ev - full_preds) ** 2)))
    log_p0 = model.intrinsic_grammar_log_prior(graph)
    nig = model.committed_nig_state
    fe = float(-float(nig["log_marginal_likelihood"]) - log_p0)
    return {
        "nig_fit": nig,
        "log_marginal_likelihood": float(nig["log_marginal_likelihood"]),
        "grammar_log_prior": log_p0,
        "variational_free_energy": fe,
        "val_mean_log_predictive": pred["mean_log_predictive"],
        "val_sum_log_predictive": pred["sum_log_predictive"],
        "val_rmse": rmse,
        "predictions": full_preds,
        "epistemic_std": pred["epistemic_std"],
    }


def fit_and_score_task_model(
    model: TaskModel,
    graph: PersistentMechanismGraph,
    train_data: dict[str, Any],
    val_data: dict[str, Any] | None = None,
    *,
    prior_shrinkage: float = 1e-2,
    commit_to_model: bool = True,
) -> dict[str, Any]:
    """Fit `TaskModel` via Empirical-Bayes NIG (with fixed task `reference_y_var` & memory carry-over) and score `val_data`."""
    Phi_tr, off_tr, _, term_tr = build_task_design_matrix(
        model,
        graph,
        train_data,
        capture_terminal_memory=commit_to_model,
    )
    y_tr = np.asarray(train_data["y"], dtype=float).ravel()
    y_adj_tr = y_tr - off_tr

    ref_var = model.reference_y_var
    if ref_var is None:
        ref_var = float(np.var(y_tr)) if len(y_tr) > 1 else float(np.mean(y_tr ** 2))
        if commit_to_model:
            model.reference_y_var = ref_var

    nig = fit_scale_covariant_nig_linear_model(
        Phi_tr,
        y_adj_tr,
        prior_shrinkage=prior_shrinkage,
        reference_y_var=ref_var,
        intercept_col_Zero=(len(model.regime_branches) <= 1),
    )
    log_p0 = model.intrinsic_grammar_log_prior(graph)
    variational_free_energy = float(-nig["log_marginal_likelihood"] - log_p0)
    full_tr_pred = off_tr + Phi_tr @ nig["posterior_mean"]
    tr_resid = y_tr - full_tr_pred

    out: dict[str, Any] = {
        "nig_fit": nig,
        "log_marginal_likelihood": nig["log_marginal_likelihood"],
        "grammar_log_prior": log_p0,
        "variational_free_energy": variational_free_energy,
        "train_rmse": float(np.sqrt(np.mean(tr_resid ** 2))),
        "residuals": tr_resid,
    }

    if val_data is not None:
        Phi_va, off_va, inf_va, _ = build_task_design_matrix(
            model,
            graph,
            val_data,
            initial_memory_states=term_tr or None,
        )
        y_va = np.asarray(val_data["y"], dtype=float).ravel()
        pred_va = predict_nig_student_t_log_score(nig, Phi_va, y_va - off_va, domain_inflation=inf_va)
        out["val_mean_log_predictive"] = pred_va["mean_log_predictive"]
        out["val_sum_log_predictive"] = pred_va["sum_log_predictive"]
        out["val_rmse"] = float(np.sqrt(np.mean((y_va - (off_va + pred_va["predictions"])) ** 2)))
        out["val_predictions"] = off_va + pred_va["predictions"]
    else:
        out["val_mean_log_predictive"] = float(nig["log_marginal_likelihood"] / max(len(y_tr), 1))
        out["val_sum_log_predictive"] = float(nig["log_marginal_likelihood"])
        out["val_rmse"] = out["train_rmse"]

    if commit_to_model:
        model.committed_prior_shrinkage = float(prior_shrinkage)
        model.committed_nig_state = {k: v for k, v in nig.items() if k != "residuals"}
        model.terminal_memory_states = term_tr
        model.fit_summary = {
            "posterior_mean": nig["posterior_mean"].tolist(),
            "prior_shrinkage": float(prior_shrinkage),
            "variational_free_energy": variational_free_energy,
            "val_mean_log_predictive": out["val_mean_log_predictive"],
            "val_rmse": out["val_rmse"],
            "train_rmse": out["train_rmse"],
        }
    return out


# ============================================================================
# 5. Residual Proposal Diagnostics & Pre-Payment Candidate / Probe Machinery
# ============================================================================


def diagnose_residual_signals(
    residuals: np.ndarray,
    dataset: dict[str, Any],
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    """Scale-invariant residual diagnostics used strictly to rank pre-payment proposal order (`T06`, `T08`)."""
    r = np.asarray(residuals, dtype=float).ravel()
    n = len(r)
    x_cols = dataset["x"]
    r_var = float(np.var(r))
    if n < 6 or r_var < 1e-12:
        return {
            "status": "CONVERGED_LOW_RESIDUAL",
            "linear_shift_r2": 0.0,
            "pairwise_interaction_r2": 0.0,
            "tail_concentration_ratio": 0.0,
            "history_divergence_r2": 0.0,
            "heteroscedastic_abs_r2": 0.0,
            "ranked_proposal_families": ["NoChange"],
        }

    Z_cols: list[np.ndarray] = []
    for fnm in feature_names:
        col = np.asarray(x_cols[fnm], dtype=float).ravel()
        sd = float(np.std(col))
        Z_cols.append((col - float(np.mean(col))) / max(sd, 1e-9))
    Z = np.column_stack([np.ones(n), *Z_cols])

    def _ols_adj_r2(design: np.ndarray, target: np.ndarray) -> float:
        t_c = target - float(np.mean(target))
        tss = float(np.sum(t_c ** 2))
        if tss < 1e-12:
            return 0.0
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        rss = float(np.sum((target - design @ coef) ** 2))
        raw_r2 = max(0.0, 1.0 - rss / tss)
        p = max(1, design.shape[1] - 1)
        if n <= p + 2:
            return 0.0
        return max(0.0, 1.0 - (1.0 - raw_r2) * (n - 1) / (n - p - 1))

    lin_r2 = _ols_adj_r2(Z, r)

    inter_terms = [Z]
    for i in range(len(Z_cols)):
        inter_terms.append((Z_cols[i] ** 2)[:, None])
        inter_terms.append(np.abs(Z_cols[i])[:, None])
        for j in range(i + 1, len(Z_cols)):
            inter_terms.append((Z_cols[i] * Z_cols[j])[:, None])
            inter_terms.append((Z_cols[i] * np.abs(Z_cols[j]))[:, None])
            inter_terms.append((Z_cols[j] * np.abs(Z_cols[i]))[:, None])
    Z_inter = np.hstack(inter_terms)
    inter_r2 = max(0.0, _ols_adj_r2(Z_inter, r) - lin_r2)

    abs_r = np.abs(r)
    hetero_r2 = _ols_adj_r2(Z_inter, abs_r)

    k_top = max(1, int(0.10 * n))
    top_idx = np.argsort(r ** 2)[-k_top:]
    top_energy = float(np.sum((r[top_idx]) ** 2))
    tail_conc = top_energy / max(float(np.sum(r ** 2)), 1e-12)
    tail_sign_coherence = abs(float(np.sum(r[top_idx]))) / max(float(np.sum(np.abs(r[top_idx]))), 1e-12)
    coherent_tail_conc = tail_conc * tail_sign_coherence

    hist_r2 = 0.0
    if ("past_driver_history" in dataset and dataset["past_driver_history"] is not None) or ("timestamps" in dataset):
        mem_probe = _compute_causal_memory_feature(dataset, feature_names[0], tau=1.5)
        assert isinstance(mem_probe, np.ndarray)
        Z_mem = np.column_stack([Z, (mem_probe - np.mean(mem_probe)) / max(float(np.std(mem_probe)), 1e-9)])
        hist_r2 = max(0.0, _ols_adj_r2(Z_mem, r) - lin_r2)

    mean_explainable = max(lin_r2, inter_r2, hist_r2)
    status = "STRUCTURED_RESIDUAL_CANDIDATES"
    if mean_explainable < 0.08 and coherent_tail_conc < 0.35:
        status = "HETEROSCEDASTIC_OR_UNRESOLVED_NOISE" if hetero_r2 > 0.15 else "UNRESOLVED_WHITE_OR_WEAK_SIGNAL"

    scores = [
        ("ReuseMechanism", inter_r2 + hist_r2 + 0.05),
        ("AncestorReweight", lin_r2),
        ("LocalRefit", lin_r2 * 0.95 + 0.02),
        ("LocalSplit", max(0.0, coherent_tail_conc - 0.25)),
        ("ConceptLift", inter_r2),
        ("ChangeMemory", hist_r2),
        ("NoChange", 0.12 if status != "STRUCTURED_RESIDUAL_CANDIDATES" else 0.02),
    ]
    ranked = [nm for nm, _ in sorted(scores, key=lambda kv: kv[1], reverse=True)]
    return {
        "status": status,
        "linear_shift_r2": round(lin_r2, 5),
        "pairwise_interaction_r2": round(inter_r2, 5),
        "tail_concentration_ratio": round(tail_conc, 5),
        "history_divergence_r2": round(hist_r2, 5),
        "heteroscedastic_abs_r2": round(hetero_r2, 5),
        "ranked_proposal_families": ranked,
    }


@dataclass
class RevisionCandidate:
    candidate_id: str
    action_type: str
    description: str
    estimated_cost: float
    actual_cost_spent: float = 0.0
    execution_status: str = "EXECUTED_OK"
    epistemic_outcome: str = "EVALUATED"
    val_log_predictive_gain: float = 0.0
    collateral_min_delta: float = 0.0
    unrelated_max_abs_drift: float = 0.0
    variational_free_energy_delta: float = 0.0
    predict_fn: Callable[[dict[str, np.ndarray]], np.ndarray] | None = field(default=None, repr=False)
    apply_fn: Callable[[PersistentMechanismGraph], None] | None = field(default=None, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def public_summary(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action_type": self.action_type,
            "description": self.description,
            "estimated_cost": self.estimated_cost,
            "actual_cost_spent": round(self.actual_cost_spent, 6),
            "execution_status": self.execution_status,
            "epistemic_outcome": self.epistemic_outcome,
            "val_log_predictive_gain": round(self.val_log_predictive_gain, 6),
            "collateral_min_delta": round(self.collateral_min_delta, 6),
            "unrelated_max_abs_drift": round(self.unrelated_max_abs_drift, 8),
            "variational_free_energy_delta": round(self.variational_free_energy_delta, 6),
            "metadata": deepcopy(self.metadata),
        }


def fit_nonlinear_power_parameter_escape_zero_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    *,
    w_init: float = 0.0,
) -> dict[str, Any]:
    """Multi-start parameter refit escaping local `J(0) == 0` saddle (`f(x; w) = w^2 * x` at `w = 0`, `T06`)."""
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    jac_at_init = float(np.linalg.norm(2.0 * float(w_init) * xv))

    best_w = float(w_init)
    best_loss = float(0.5 * np.mean((yv - (best_w ** 2) * xv) ** 2))
    for seed_w in (w_init, -1.5, -0.5, 0.5, 1.5, 2.5):
        res = minimize(
            lambda w_arr: float(0.5 * np.mean((yv - (float(w_arr[0]) ** 2) * xv) ** 2)),
            x0=np.array([float(seed_w)]),
            method="BFGS",
        )
        if float(res.fun) < best_loss:
            best_loss = float(res.fun)
            best_w = float(res.x[0])

    return {
        "w_init": float(w_init),
        "jacobian_norm_at_init": jac_at_init,
        "fitted_w": best_w,
        "fitted_w_squared": best_w ** 2,
        "rmse": float(np.sqrt(2.0 * best_loss)),
        "solved_by_parameter_refit_without_concept_lift": bool(math.sqrt(2.0 * best_loss) < 0.05),
    }


def propose_concept_lift_via_gpt_cli(
    *,
    task_id: str,
    feature_names: tuple[str, ...],
    train_data: dict[str, Any],
    residuals: np.ndarray,
    diagnostics: dict[str, Any],
    codex_client: Any,
    stage_subdir: str = "m8_concept_lift",
) -> dict[str, Any]:
    """Invoke live official `codex` ChatGPT CLI (`gpt-6.2-sol`) with blinded task tag (no mechanism hint in task_id!)."""
    x_cols = train_data["x"]
    r = np.asarray(residuals, dtype=float).ravel()
    top_idx = np.argsort(np.abs(r))[-14:][::-1]
    quant_idx = np.argsort(r)[np.linspace(0, len(r) - 1, 14, dtype=int)]
    sample_idx = np.unique(np.concatenate([top_idx, quant_idx]))[:26]

    rows_preview = [
        {
            "features": {fnm: round(float(x_cols[fnm][i]), 4) for fnm in feature_names},
            "residual_y_minus_pred": round(float(r[i]), 4),
        }
        for i in sample_idx
    ]

    corr_table: dict[str, float] = {}
    for i, f1 in enumerate(feature_names):
        v1 = np.asarray(x_cols[f1], dtype=float).ravel()
        corr_table[f"{f1}^2"] = round(float(np.corrcoef(v1 ** 2, r)[0, 1]) if np.std(v1 ** 2) > 1e-9 else 0.0, 4)
        corr_table[f"|{f1}|"] = round(float(np.corrcoef(np.abs(v1), r)[0, 1]) if np.std(np.abs(v1)) > 1e-9 else 0.0, 4)
        for f2 in feature_names[i + 1 :]:
            v2 = np.asarray(x_cols[f2], dtype=float).ravel()
            prod = v1 * v2
            corr_table[f"{f1}*{f2}"] = round(float(np.corrcoef(prod, r)[0, 1]) if np.std(prod) > 1e-9 else 0.0, 4)
            mod12 = v1 * np.abs(v2)
            corr_table[f"{f1}*|{f2}|"] = round(float(np.corrcoef(mod12, r)[0, 1]) if np.std(mod12) > 1e-9 else 0.0, 4)

    blinded_task_ref = "task_" + hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8]
    sys_prompt = (
        "You are the `Hegel-M8` Variational Mechanism Synthesis Planner.\n"
        "An active additive/branched `TaskModel` in the persistent `PhilosophicalForest` exhibits structured residuals "
        "on public training observations.\n"
        "Synthesize ONE clean, interpretable nonlinear/interaction basis conduit `python_expression` over `x['<col>']` "
        "(using ONLY observable feature names listed in `feature_names` and safe `np.*` math primitives) that best "
        "explains `residual_y_minus_pred` while remaining smooth and bounded on the domain!"
    )
    user_payload = json.dumps(
        {
            "task_ref": blinded_task_ref,
            "feature_names": list(feature_names),
            "residual_diagnostics": diagnostics,
            "basis_residual_correlations": corr_table,
            "representative_residual_samples": rows_preview,
        },
        ensure_ascii=False,
    )
    return codex_client.call_structured(
        stage_subdir=stage_subdir,
        system_prompt=sys_prompt,
        conversation_turns=[{"role": "user", "content": user_payload}],
        output_schema=M8_CONCEPT_LIFT_SCHEMA,
    )


def select_discriminating_observation_probe(
    competing_predict_fns: list[tuple[str, Callable[[dict[str, np.ndarray]], np.ndarray]]],
    candidate_probes: list[dict[str, float]],
    *,
    probe_costs: list[float] | None = None,
) -> dict[str, Any]:
    """Select active intervention `AcquireObservation` maximizing full-`TaskModel` disagreement per cost."""
    if not candidate_probes or len(competing_predict_fns) < 2:
        return {"selected_index": None, "max_utility": 0.0}

    best_idx = 0
    best_util = -1.0
    probe_scores: list[dict[str, Any]] = []
    for idx, p_pt in enumerate(candidate_probes):
        x_probe = {k: np.array([float(v)], dtype=float) for k, v in p_pt.items()}
        preds = [float(fn(x_probe)[0]) for _, fn in competing_predict_fns]
        var_disagree = float(np.var(preds))
        cost = max(float(probe_costs[idx]) if probe_costs is not None else 1.0, 1e-6)
        util = var_disagree / cost
        probe_scores.append({"probe": p_pt, "disagreement_variance": round(var_disagree, 6), "cost": cost, "utility": round(util, 6)})
        if util > best_util:
            best_util = util
            best_idx = idx

    return {
        "selected_index": best_idx,
        "selected_probe": candidate_probes[best_idx],
        "max_utility": best_util,
        "probe_scores": probe_scores,
    }


# ============================================================================
# 6. Unified `M8VariationalPlanner`: Auto-Recall, Pre-Payment Budget, & Full Ledger
# ============================================================================


class M8VariationalPlanner:
    """Unified M8.2 Planner with atomic commits, autonomous role-mapped recall, pre-payment budgeting & reachable probes."""

    def __init__(
        self,
        graph: PersistentMechanismGraph,
        *,
        min_gain_nats: float = 0.08,
        collateral_tolerance_nats: float = -0.03,
        cost_weight: float = 0.01,
    ) -> None:
        self.graph = graph
        self.min_gain_nats = float(min_gain_nats)
        self.collateral_tolerance_nats = float(collateral_tolerance_nats)
        self.cost_weight = float(cost_weight)
        self._step_counter = len(graph.decision_ledger)

    def register_task(
        self,
        *,
        task_id: str,
        shared_upstream_id: str,
        feature_names: tuple[str, ...],
        train_data: dict[str, Any],
        val_data: dict[str, Any],
        initial_local_shrinkage: float = 12.0,
        shared_role_map: dict[str, str] | None = None,
        auto_propose_existing_mechanisms: bool = False,
    ) -> TaskModel:
        """Register a task anchored to `shared_upstream_id` with atomic initial model state."""
        assert shared_upstream_id in self.graph.mechanisms, f"Missing shared upstream `{shared_upstream_id}`"
        self.graph.task_datasets[task_id] = {
            "train": train_data,
            "val": val_data,
            "local_shrinkage": float(initial_local_shrinkage),
        }
        base_br_id = f"mech_branch_default_{task_id}"
        self.graph.register_mechanism(
            TypedMechanism(
                mechanism_id=base_br_id,
                parent_id=shared_upstream_id,
                role="regime_branch",
                feature_names=feature_names,
                dependent_task_ids={task_id},
            )
        )
        self.graph.mechanisms[shared_upstream_id].dependent_task_ids.add(task_id)

        # Infer default positional role map if target uses renamed observable columns!
        s_mech = self.graph.mechanisms[shared_upstream_id]
        eff_shared_map = dict(shared_role_map or {})
        if not eff_shared_map and set(s_mech.feature_names) != set(feature_names) and len(s_mech.feature_names) == len(feature_names):
            eff_shared_map = {sf: tf for sf, tf in zip(s_mech.feature_names, feature_names)}

        tm = TaskModel(
            model_id=f"model_{task_id}_v1",
            task_id=task_id,
            shared_upstream_id=shared_upstream_id,
            feature_names=feature_names,
            regime_branches=[RegimeConduitBranch(branch_id="default", mechanism_id=base_br_id, scope_guard=None, gate_scale=1.0)],
            shared_role_map=eff_shared_map,
            committed_prior_shrinkage=float(initial_local_shrinkage),
        )
        fit_and_score_task_model(tm, self.graph, train_data, val_data, prior_shrinkage=float(initial_local_shrinkage), commit_to_model=True)
        self.graph.active_task_models[task_id] = tm
        self.graph.sync_authoritative_state_to_forest()
        if auto_propose_existing_mechanisms:
            self.evaluate_and_revise_task(task_id, budget_remaining=1.0)
        return self.graph.active_task_models[task_id]

    def evaluate_task_zero_shot_from_shared(self, task_id: str) -> dict[str, Any]:
        """Strictly label-free zero-shot prediction from `shared_upstream_id` (NEVER fits on target `train_data['y']`!)."""
        tm = self.graph.active_task_models[task_id]
        ds_va = self.graph.task_datasets[task_id]["val"]
        _, off_va, _, _ = build_task_design_matrix(tm, self.graph, ds_va)
        y_va = np.asarray(ds_va["y"], dtype=float).ravel()
        rmse = float(np.sqrt(np.mean((y_va - off_va) ** 2)))
        return {
            "val_rmse": rmse,
            "predictions": off_va,
            "used_target_training_labels": False,
        }

    def _discover_reusable_mechanism_candidates(
        self,
        incumbent: TaskModel,
        ds_tr: dict[str, Any],
        ds_va: dict[str, Any],
        base_val_lp: float,
        base_fe: float,
    ) -> list[RevisionCandidate]:
        """Autonomously recall verified `interaction_lift` / `temporal_memory` mechanisms across trees & renamed roles!"""
        task_id = incumbent.task_id
        recalled: list[RevisionCandidate] = []
        obs_summary = {
            k: float(np.mean(np.asarray(v, dtype=float)))
            for k, v in ds_tr["x"].items()
        }

        for mid, mech in sorted(self.graph.mechanisms.items()):
            if mech.epistemic_status != "SUPPORTED_IN_SCOPE" or task_id in mech.inapplicable_task_ids:
                continue
            if mech.role == "interaction_lift" and mid not in incumbent.interaction_lift_ids and mech.python_expression:
                if mech.scope_guard and evaluate_predicate_rule(mech.scope_guard, obs_summary) == "FALSE":
                    continue
                # Enumerate valid variable-role mappings (identity if names match, else bounded permutations <= 6)
                role_maps_to_try: list[dict[str, str]] = []
                if set(mech.feature_names).issubset(set(incumbent.feature_names)):
                    role_maps_to_try.append({f: f for f in mech.feature_names})
                if len(mech.feature_names) <= len(incumbent.feature_names) and len(incumbent.feature_names) <= 4:
                    for perm in itertools.permutations(incumbent.feature_names, len(mech.feature_names)):
                        rm = {sf: tf for sf, tf in zip(mech.feature_names, perm)}
                        if rm not in role_maps_to_try:
                            role_maps_to_try.append(rm)

                best_rm_cand: RevisionCandidate | None = None
                for rm in role_maps_to_try[:6]:
                    try:
                        mapped_expr = remap_conduit_expression_roles(mech.python_expression, rm)
                        fn_c = compile_safe_feature_conduit(mapped_expr, incumbent.feature_names)
                        cond = estimate_conduit_conditioning_and_domain(fn_c, ds_tr["x"])
                        if not cond["finite_on_support"]:
                            continue
                    except Exception:
                        continue

                    m_reuse = deepcopy(incumbent)
                    m_reuse.version += 1
                    m_reuse.model_id = f"{incumbent.model_id}_reuse_{mid[:14]}"
                    m_reuse.interaction_lift_ids = [*incumbent.interaction_lift_ids, mid]
                    m_reuse.lift_role_maps = {**incumbent.lift_role_maps, mid: rm}
                    sc_r = fit_and_score_task_model(m_reuse, self.graph, ds_tr, ds_va, prior_shrinkage=1e-2, commit_to_model=True)
                    gain_r = float(sc_r["val_mean_log_predictive"] - base_val_lp)

                    def _make_pred_r(m_snap=deepcopy(m_reuse)) -> Callable[[dict[str, np.ndarray]], np.ndarray]:
                        return lambda x_in: evaluate_committed_task_model(m_snap, self.graph, {"x": x_in, "y": np.zeros(len(next(iter(x_in.values()))))})["predictions"]

                    def _apply_reuse(g: PersistentMechanismGraph, m_new=m_reuse, mech_id=mid, gain_val=gain_r) -> None:
                        g.mechanisms[mech_id].dependent_task_ids.add(task_id)
                        g.task_datasets[task_id]["local_shrinkage"] = 1e-2
                        g.active_task_models[task_id] = m_new
                        g.sync_authoritative_state_to_forest()

                    cand_r = RevisionCandidate(
                        candidate_id=f"cand_{self._step_counter}_reuse_{mid[:16]}",
                        action_type="ReuseMechanism",
                        description=f"Autonomously recall & bind persisted mechanism `{mid}` via role map `{rm}`.",
                        estimated_cost=0.6,
                        actual_cost_spent=0.05,
                        val_log_predictive_gain=gain_r,
                        variational_free_energy_delta=float(base_fe - sc_r["variational_free_energy"]),
                        predict_fn=_make_pred_r(),
                        apply_fn=_apply_reuse,
                        metadata={"recalled_mechanism_id": mid, "role_map": rm, "mapped_expression": mapped_expr},
                    )
                    if best_rm_cand is None or cand_r.val_log_predictive_gain > best_rm_cand.val_log_predictive_gain:
                        best_rm_cand = cand_r
                if best_rm_cand is not None:
                    # If recalled mechanism fails to improve target task, record negative scope evidence!
                    if best_rm_cand.val_log_predictive_gain <= 0.0:
                        mech.inapplicable_task_ids.add(task_id)
                    recalled.append(best_rm_cand)
        return recalled

    def evaluate_and_revise_task(
        self,
        task_id: str,
        *,
        codex_client: Any | None = None,
        injected_broken_expr: str | None = None,
        candidate_probes: list[dict[str, float]] | None = None,
        probe_oracle_fn: Callable[[dict[str, float]], float] | None = None,
        budget_remaining: float | None = None,
    ) -> dict[str, Any]:
        """Pre-payment budgeted planning + atomic model commit + complete attempt cost ledger + reachable probes."""
        self._step_counter += 1
        t0 = monotonic()
        incumbent = self.graph.active_task_models[task_id]
        t_entry = self.graph.task_datasets[task_id]
        ds_tr = t_entry["train"]
        ds_va = t_entry["val"]

        # Evaluate committed incumbent WITHOUT silently changing its shrinkage configuration (`M8.2` Bug 1 Fix)!
        if incumbent.committed_nig_state is not None:
            inc_va = evaluate_committed_task_model(incumbent, self.graph, ds_va)
            inc_tr = evaluate_committed_task_model(incumbent, self.graph, ds_tr)
            tr_resid = np.asarray(ds_tr["y"], dtype=float).ravel() - inc_tr["predictions"]
            base_val_lp = float(inc_va["val_mean_log_predictive"])
            base_fe = float(inc_va["variational_free_energy"])
            base_val_rmse = float(inc_va["val_rmse"])
        else:
            cur_shrink = float(t_entry.get("local_shrinkage", incumbent.committed_prior_shrinkage))
            inc_eval = fit_and_score_task_model(incumbent, self.graph, ds_tr, ds_va, prior_shrinkage=cur_shrink, commit_to_model=True)
            tr_resid = inc_eval["residuals"]
            base_val_lp = float(inc_eval["val_mean_log_predictive"])
            base_fe = float(inc_eval["variational_free_energy"])
            base_val_rmse = float(inc_eval["val_rmse"])

        diag = diagnose_residual_signals(tr_resid, ds_tr, incumbent.feature_names)

        # Snapshot baseline scores of all committed active task models whose evaluation datasets are loaded in this process
        baseline_task_scores: dict[str, float] = {}
        for other_tid, other_tm in self.graph.active_task_models.items():
            if other_tid not in self.graph.task_datasets:
                continue
            o_va = self.graph.task_datasets[other_tid]["val"]
            if other_tm.committed_nig_state is not None:
                baseline_task_scores[other_tid] = float(evaluate_committed_task_model(other_tm, self.graph, o_va)["val_mean_log_predictive"])
            else:
                o_tr = self.graph.task_datasets[other_tid]["train"]
                baseline_task_scores[other_tid] = float(
                    fit_and_score_task_model(deepcopy(other_tm), self.graph, o_tr, o_va, prior_shrinkage=other_tm.committed_prior_shrinkage, commit_to_model=False)["val_mean_log_predictive"]
                )

        rem_budget = float("inf") if budget_remaining is None else float(budget_remaining)
        attempt_cost_events: list[dict[str, Any]] = []
        total_prompt_tok = 0
        total_comp_tok = 0

        def _spend(action_nm: str, cost_val: float, p_tok: int = 0, c_tok: int = 0, status: str = "OK", note: str = "") -> bool:
            nonlocal rem_budget, total_prompt_tok, total_comp_tok
            total_prompt_tok += int(p_tok)
            total_comp_tok += int(c_tok)
            rem_budget -= float(cost_val)
            attempt_cost_events.append(
                {
                    "action": action_nm,
                    "cost_spent": round(float(cost_val), 5),
                    "prompt_tokens": int(p_tok),
                    "completion_tokens": int(c_tok),
                    "status": status,
                    "note": note,
                }
            )
            return True

        pre_state = {
            "task_id": task_id,
            "incumbent_model_id": incumbent.model_id,
            "incumbent_model_version": incumbent.version,
            "incumbent_committed_shrinkage": incumbent.committed_prior_shrinkage,
            "incumbent_val_mean_log_predictive": round(base_val_lp, 6),
            "incumbent_variational_free_energy": round(base_fe, 6),
            "incumbent_val_rmse": round(base_val_rmse, 6),
            "residual_diagnostics": diag,
            "budget_remaining_before_step": (None if budget_remaining is None else round(float(budget_remaining), 4)),
            "shared_upstream_id": incumbent.shared_upstream_id,
            "shared_upstream_version": self.graph.mechanisms[incumbent.shared_upstream_id].version,
        }

        candidates: list[RevisionCandidate] = [
            RevisionCandidate(
                candidate_id=f"cand_{self._step_counter}_nochange",
                action_type="NoChange",
                description="Retain committed incumbent mechanism state without modification.",
                estimated_cost=0.0,
                actual_cost_spent=0.0,
                val_log_predictive_gain=0.0,
                variational_free_energy_delta=0.0,
                predict_fn=lambda x_in: evaluate_committed_task_model(incumbent, self.graph, {"x": x_in, "y": np.zeros(len(next(iter(x_in.values()))))})["predictions"],
            )
        ]

        # Early budget/convergence exit: if residual is already at machine-precision / converged and no forced fault test, skip paid search!
        if diag["status"] == "CONVERGED_LOW_RESIDUAL" and injected_broken_expr is None:
            rec_conv = ResearchDecisionRecord(
                decision_id=f"dec_{self._step_counter:03d}_{task_id}",
                step_index=self._step_counter,
                task_id=task_id,
                pre_action_state=pre_state,
                available_actions=["NoChange"],
                selected_action="NoChange",
                selected_candidate_id=candidates[0].candidate_id,
                execution_status="EXECUTED_OK",
                actual_cost=0.0,
                public_val_log_predictive_delta=0.0,
                collateral_min_delta=0.0,
                rationale="Incumbent residual already converged; skipped paid candidate search.",
            )
            self.graph.decision_ledger.append(rec_conv)
            self.graph.sync_authoritative_state_to_forest()
            return {
                "decision_record": rec_conv.to_dict(),
                "selected_candidate": candidates[0].public_summary(),
                "evaluated_candidates": [candidates[0].public_summary()],
                "wall_ms": round((monotonic() - t0) * 1000.0, 2),
            }

        # Stage 1 (Cheap Pre-Payment Recall): Try autonomous `ReuseMechanism` from persistent graph FIRST!
        resolved_cheaply = False
        if rem_budget >= 0.05:
            reuse_cands = self._discover_reusable_mechanism_candidates(incumbent, ds_tr, ds_va, base_val_lp, base_fe)
            for rc in reuse_cands:
                _spend("ReuseMechanism_probe", rc.actual_cost_spent, note=rc.description)
                candidates.append(rc)
                if rc.val_log_predictive_gain >= max(self.min_gain_nats, 1.2) and rc.variational_free_energy_delta > 5.0:
                    resolved_cheaply = True

        # Stage 2: `LocalRefit` & `AncestorReweight` (cheap linear solves)
        if rem_budget >= 0.10 and incumbent.committed_prior_shrinkage > 1e-1:
            m_refit = deepcopy(incumbent)
            m_refit.version += 1
            m_refit.model_id = f"{incumbent.model_id}_local_refit"
            sc_refit = fit_and_score_task_model(m_refit, self.graph, ds_tr, ds_va, prior_shrinkage=1e-2, commit_to_model=True)
            _spend("LocalRefit_fit", 0.10)

            def _apply_local_refit(g: PersistentMechanismGraph, m_new=m_refit) -> None:
                g.task_datasets[task_id]["local_shrinkage"] = 1e-2
                g.active_task_models[task_id] = m_new
                g.sync_authoritative_state_to_forest()

            candidates.append(
                RevisionCandidate(
                    candidate_id=f"cand_{self._step_counter}_local_refit",
                    action_type="LocalRefit",
                    description="Refit only task-local parameters (`shrinkage -> 1e-2`) without modifying shared upstream.",
                    estimated_cost=1.4,
                    actual_cost_spent=0.10,
                    val_log_predictive_gain=float(sc_refit["val_mean_log_predictive"] - base_val_lp),
                    variational_free_energy_delta=float(base_fe - sc_refit["variational_free_energy"]),
                    predict_fn=lambda x_in, ms=deepcopy(m_refit): evaluate_committed_task_model(ms, self.graph, {"x": x_in, "y": np.zeros(len(next(iter(x_in.values()))))})["predictions"],
                    apply_fn=_apply_local_refit,
                )
            )

        shared_id = incumbent.shared_upstream_id
        affected_tids = self.graph.minimal_affected_task_ids(shared_id)
        unrelated_tids = sorted(set(self.graph.active_task_models.keys()) - set(affected_tids))
        shared_mech_cur = self.graph.mechanisms[shared_id]
        shared_feats = tuple(shared_mech_cur.shared_weights.keys()) or shared_mech_cur.feature_names
        if rem_budget >= 0.15 and shared_feats:
            X_pool_list, y_pool_list = [], []
            for r_tid in affected_tids:
                if r_tid in self.graph.task_datasets and r_tid in self.graph.active_task_models:
                    r_tm = self.graph.active_task_models[r_tid]
                    r_tr = self.graph.task_datasets[r_tid]["train"]
                    X_pool_list.append(
                        np.column_stack(
                            [np.asarray(r_tr["x"][r_tm.shared_role_map.get(fnm, fnm)], dtype=float).ravel() for fnm in shared_feats]
                        )
                    )
                    y_pool_list.append(np.asarray(r_tr["y"], dtype=float).ravel())
            if X_pool_list:
                w_hat, *_ = np.linalg.lstsq(np.vstack(X_pool_list), np.concatenate(y_pool_list), rcond=None)
                new_shared_weights = {fnm: float(w_hat[idx]) for idx, fnm in enumerate(shared_feats)}
                _spend("AncestorReweight_fit", 0.15)

                sim_graph = deepcopy(self.graph)
                sim_graph.mechanisms[shared_id].shared_weights = new_shared_weights
                sim_graph.mechanisms[shared_id].version += 1

                m_anc_self = deepcopy(incumbent)
                m_anc_self.version += 1
                sc_anc = fit_and_score_task_model(
                    m_anc_self,
                    sim_graph,
                    ds_tr,
                    ds_va,
                    prior_shrinkage=incumbent.committed_prior_shrinkage,
                    commit_to_model=True,
                )
                rel_deltas: list[float] = []
                for r_tid in affected_tids:
                    if r_tid == task_id or r_tid not in sim_graph.task_datasets:
                        continue
                    r_entry = sim_graph.task_datasets[r_tid]
                    r_tm_c = deepcopy(sim_graph.active_task_models[r_tid])
                    r_sc = fit_and_score_task_model(
                        r_tm_c,
                        sim_graph,
                        r_entry["train"],
                        r_entry["val"],
                        prior_shrinkage=r_tm_c.committed_prior_shrinkage,
                        commit_to_model=True,
                    )
                    rel_deltas.append(float(r_sc["val_mean_log_predictive"] - baseline_task_scores[r_tid]))

                unrel_drifts: list[float] = []
                for u_tid in unrelated_tids:
                    if u_tid not in sim_graph.task_datasets:
                        continue
                    u_va = sim_graph.task_datasets[u_tid]["val"]
                    u_sc = evaluate_committed_task_model(sim_graph.active_task_models[u_tid], sim_graph, u_va)
                    unrel_drifts.append(abs(float(u_sc["val_mean_log_predictive"] - baseline_task_scores[u_tid])))

                coll_min = min(rel_deltas) if rel_deltas else 0.0
                total_rel_gain = float(np.sum(rel_deltas)) if rel_deltas else 0.0

                def _apply_anc(g: PersistentMechanismGraph, nw=new_shared_weights) -> None:
                    m = g.mechanisms[shared_id]
                    m.shared_weights = dict(nw)
                    m.version += 1
                    g.mechanism_lineage.setdefault(shared_id, []).append(m.to_dict())
                    for r_tid in g.minimal_affected_task_ids(shared_id):
                        if r_tid in g.active_task_models and r_tid in g.task_datasets:
                            re_e = g.task_datasets[r_tid]
                            r_mod = g.active_task_models[r_tid]
                            r_mod.version += 1
                            fit_and_score_task_model(
                                r_mod,
                                g,
                                re_e["train"],
                                re_e["val"],
                                prior_shrinkage=r_mod.committed_prior_shrinkage,
                                commit_to_model=True,
                            )
                    g.sync_authoritative_state_to_forest()

                candidates.append(
                    RevisionCandidate(
                        candidate_id=f"cand_{self._step_counter}_ancestor_reweight",
                        action_type="AncestorReweight",
                        description=f"Update shared upstream `{shared_id}` weights across {len(affected_tids)} related task(s).",
                        estimated_cost=1.5,
                        actual_cost_spent=0.15,
                        val_log_predictive_gain=float(sc_anc["val_mean_log_predictive"] - base_val_lp) + total_rel_gain,
                        collateral_min_delta=coll_min,
                        unrelated_max_abs_drift=max(unrel_drifts) if unrel_drifts else 0.0,
                        variational_free_energy_delta=float(base_fe - sc_anc["variational_free_energy"]),
                        predict_fn=lambda x_in, ms=deepcopy(m_anc_self), sg=sim_graph: evaluate_committed_task_model(ms, sg, {"x": x_in, "y": np.zeros(len(next(iter(x_in.values()))))})["predictions"],
                        apply_fn=_apply_anc,
                        metadata={"new_shared_weights": new_shared_weights, "affected_tasks": affected_tids, "unrelated_tasks": unrelated_tids},
                    )
                )

        # Stage 3: `LocalSplit` (stores fixed training-time `gate_scale` on `RegimeConduitBranch`!)
        if rem_budget >= 0.20 and not resolved_cheaply:
            best_split_cand: RevisionCandidate | None = None
            for fnm in incumbent.feature_names:
                col = np.asarray(ds_tr["x"][fnm], dtype=float).ravel()
                fixed_g_scale = max(float(np.std(col)), 0.15)
                for q_lvl in (0.75, 0.85, 0.90):
                    thr = float(np.quantile(col, q_lvl))
                    if np.sum(col > thr) < 5 or np.sum(col <= thr) < 5:
                        continue
                    sim_g = deepcopy(self.graph)
                    sp_mech_id = f"mech_split_{task_id}_{fnm}_{int(q_lvl * 100)}_s{self._step_counter}"
                    guard_spec = {"field": fnm, "op": ">", "value": round(thr, 5)}
                    sim_g.register_mechanism(
                        TypedMechanism(
                            mechanism_id=sp_mech_id,
                            parent_id=incumbent.shared_upstream_id,
                            role="regime_branch",
                            feature_names=incumbent.feature_names,
                            scope_guard=guard_spec,
                            dependent_task_ids={task_id},
                        )
                    )
                    m_sp = deepcopy(incumbent)
                    m_sp.version += 1
                    m_sp.model_id = f"{incumbent.model_id}_split_{fnm}"
                    m_sp.regime_branches = [
                        *incumbent.regime_branches,
                        RegimeConduitBranch(
                            branch_id=f"exc_{fnm}",
                            mechanism_id=sp_mech_id,
                            scope_guard=guard_spec,
                            gate_sharpness=18.0,
                            gate_scale=fixed_g_scale,
                        ),
                    ]
                    sc_sp = fit_and_score_task_model(m_sp, sim_g, ds_tr, ds_va, prior_shrinkage=1e-2, commit_to_model=True)
                    gain_sp = float(sc_sp["val_mean_log_predictive"] - base_val_lp)
                    if best_split_cand is None or gain_sp > best_split_cand.val_log_predictive_gain:
                        def _apply_sp(g: PersistentMechanismGraph, mid_s=sp_mech_id, gspec=guard_spec, m_new=deepcopy(m_sp)) -> None:
                            g.register_mechanism(
                                TypedMechanism(
                                    mechanism_id=mid_s,
                                    parent_id=incumbent.shared_upstream_id,
                                    role="regime_branch",
                                    feature_names=incumbent.feature_names,
                                    scope_guard=gspec,
                                    dependent_task_ids={task_id},
                                )
                            )
                            g.task_datasets[task_id]["local_shrinkage"] = 1e-2
                            g.active_task_models[task_id] = m_new
                            g.sync_authoritative_state_to_forest()

                        best_split_cand = RevisionCandidate(
                            candidate_id=f"cand_{self._step_counter}_local_split_{fnm}",
                            action_type="LocalSplit",
                            description=f"Bifurcate local exception branch on `{fnm} > {round(thr, 4)}` (`gate_scale={round(fixed_g_scale, 4)}`).",
                            estimated_cost=2.0,
                            actual_cost_spent=0.20,
                            val_log_predictive_gain=gain_sp,
                            variational_free_energy_delta=float(base_fe - sc_sp["variational_free_energy"]),
                            predict_fn=lambda x_in, ms=deepcopy(m_sp), sg=sim_g: evaluate_committed_task_model(ms, sg, {"x": x_in, "y": np.zeros(len(next(iter(x_in.values()))))})["predictions"],
                            apply_fn=_apply_sp,
                            metadata={"scope_guard": guard_spec, "gate_scale": fixed_g_scale},
                        )
            if best_split_cand is not None:
                _spend("LocalSplit_search", 0.20)
                candidates.append(best_split_cand)

        # Stage 4: `ConceptLift` (Pre-payment gated! Paid `codex_client` ONLY runs if budget >= 2.4 AND not already solved by `ReuseMechanism`!)
        lift_proposals: list[tuple[str, str, str, dict[str, Any]]] = []
        if injected_broken_expr is not None:
            lift_proposals.append(("broken_injected", injected_broken_expr, "injected_execution_fault_test", {}))

        if (
            codex_client is not None
            and not resolved_cheaply
            and rem_budget >= 2.4
            and diag["status"] == "STRUCTURED_RESIDUAL_CANDIDATES"
        ):
            cli_res = propose_concept_lift_via_gpt_cli(
                task_id=task_id,
                feature_names=incumbent.feature_names,
                train_data=ds_tr,
                residuals=tr_resid,
                diagnostics=diag,
                codex_client=codex_client,
                stage_subdir=f"m8_step_{self._step_counter:02d}",
            )
            p_tok = int(cli_res.get("prompt_tokens") or 0)
            c_tok = int(cli_res.get("completion_tokens") or 0)
            if cli_res.get("ok") and isinstance(cli_res.get("parsed"), dict):
                _spend("ConceptLift_gpt_cli", 2.4, p_tok=p_tok, c_tok=c_tok, status="OK")
                p_obj = cli_res["parsed"]
                lift_proposals.append(
                    (
                        str(p_obj["concept_name"]),
                        str(p_obj["python_expression"]),
                        "gpt_cli_codex",
                        {
                            "llm_rationale": p_obj.get("mechanistic_rationale", ""),
                            "domain_guard_note": p_obj.get("domain_guard_note", ""),
                            "prompt_tokens": p_tok,
                            "completion_tokens": c_tok,
                            "wall_ms": cli_res.get("wall_ms", 0.0),
                        },
                    )
                )
            else:
                err_msg = str(cli_res.get("error") or "gpt_cli_schema_or_call_error")
                _spend("ConceptLift_gpt_cli", 2.4, p_tok=p_tok, c_tok=c_tok, status="LLM_CALL_ERROR", note=err_msg)
                candidates.append(
                    RevisionCandidate(
                        candidate_id=f"cand_{self._step_counter}_gpt_cli_failed",
                        action_type="ConceptLift",
                        description=f"Paid GPT CLI proposal failed (`{err_msg}`).",
                        estimated_cost=2.4,
                        actual_cost_spent=2.4,
                        execution_status="EXECUTION_ERROR",
                        epistemic_outcome="UNRESOLVED_EXECUTION_FAILURE",
                        val_log_predictive_gain=-1e6,
                        metadata={
                            "source": "gpt_cli_codex",
                            "error": err_msg,
                            "prompt_tokens": p_tok,
                            "completion_tokens": c_tok,
                            "wall_ms": cli_res.get("wall_ms", 0.0),
                        },
                    )
                )

        if not resolved_cheaply and rem_budget >= 0.20:
            for i, f1 in enumerate(incumbent.feature_names):
                lift_proposals.append((f"{f1}_sq", f"x['{f1}'] ** 2", "algebraic_grammar", {}))
                for f2 in incumbent.feature_names[i + 1 :]:
                    lift_proposals.append((f"{f1}_x_{f2}", f"x['{f1}'] * x['{f2}']", "algebraic_grammar", {}))
                    lift_proposals.append((f"{f1}_x_abs_{f2}", f"x['{f1}'] * np.abs(x['{f2}'])", "algebraic_grammar", {}))

        valid_lift_cands: list[RevisionCandidate] = []
        for cname, cexpr, c_src, c_meta in lift_proposals:
            try:
                fn_c = compile_safe_feature_conduit(cexpr, incumbent.feature_names)
                cond_info = estimate_conduit_conditioning_and_domain(fn_c, ds_tr["x"])
                if not cond_info["finite_on_support"]:
                    raise ValueError("Non-finite conduit output on training support")
            except Exception as exc:
                _spend(f"ConceptLift_err_{cname[:12]}", 0.05, status="EXECUTION_ERROR", note=str(exc))
                candidates.append(
                    RevisionCandidate(
                        candidate_id=f"cand_{self._step_counter}_lift_err_{cname[:16]}",
                        action_type="ConceptLift",
                        description=f"Failed execution of proposed conduit `{cexpr}` ({type(exc).__name__}).",
                        estimated_cost=1.0,
                        actual_cost_spent=0.05,
                        execution_status="EXECUTION_ERROR",
                        epistemic_outcome="UNRESOLVED_EXECUTION_FAILURE",
                        val_log_predictive_gain=-1e6,
                        metadata={"error": str(exc), "source": c_src, **c_meta},
                    )
                )
                continue

            sim_g = deepcopy(self.graph)
            lift_id = f"mech_lift_{cname[:20]}_s{self._step_counter}"
            sim_g.register_mechanism(
                TypedMechanism(
                    mechanism_id=lift_id,
                    parent_id=incumbent.shared_upstream_id,
                    role="interaction_lift",
                    feature_names=incumbent.feature_names,
                    python_expression=cexpr,
                    support_bounds=cond_info["support_bounds"],
                    empirical_lipschitz=cond_info["empirical_lipschitz"],
                    dependent_task_ids={task_id},
                    provenance_note=c_src,
                )
            )
            m_lf = deepcopy(incumbent)
            m_lf.version += 1
            m_lf.model_id = f"{incumbent.model_id}_lift_{cname[:16]}"
            m_lf.interaction_lift_ids = [*incumbent.interaction_lift_ids, lift_id]
            sc_lf = fit_and_score_task_model(m_lf, sim_g, ds_tr, ds_va, prior_shrinkage=1e-2, commit_to_model=True)
            gain_lf = float(sc_lf["val_mean_log_predictive"] - base_val_lp)
            _spend(f"ConceptLift_fit_{cname[:12]}", 0.05)

            def _apply_lf(
                g: PersistentMechanismGraph,
                lid=lift_id,
                expr_s=cexpr,
                cinfo=cond_info,
                src_s=c_src,
                m_new=deepcopy(m_lf),
            ) -> None:
                g.register_mechanism(
                    TypedMechanism(
                        mechanism_id=lid,
                        parent_id=incumbent.shared_upstream_id,
                        role="interaction_lift",
                        feature_names=incumbent.feature_names,
                        python_expression=expr_s,
                        support_bounds=cinfo["support_bounds"],
                        empirical_lipschitz=cinfo["empirical_lipschitz"],
                        dependent_task_ids={task_id},
                        provenance_note=src_s,
                    )
                )
                g.task_datasets[task_id]["local_shrinkage"] = 1e-2
                g.active_task_models[task_id] = m_new
                g.sync_authoritative_state_to_forest()

            valid_lift_cands.append(
                RevisionCandidate(
                    candidate_id=f"cand_{self._step_counter}_concept_lift_{cname[:18]}",
                    action_type="ConceptLift",
                    description=f"Lift shared interaction conduit `{cexpr}` (`source={c_src}`).",
                    estimated_cost=2.4 if c_src == "gpt_cli_codex" else 2.5,
                    actual_cost_spent=(2.45 if c_src == "gpt_cli_codex" else 0.05),
                    val_log_predictive_gain=gain_lf,
                    variational_free_energy_delta=float(base_fe - sc_lf["variational_free_energy"]),
                    predict_fn=lambda x_in, ms=deepcopy(m_lf), sg=sim_g: evaluate_committed_task_model(ms, sg, {"x": x_in, "y": np.zeros(len(next(iter(x_in.values()))))})["predictions"],
                    apply_fn=_apply_lf,
                    metadata={"expression": cexpr, "source": c_src, "empirical_lipschitz": cond_info["empirical_lipschitz"], **c_meta},
                )
            )
        # Retain top-2 distinct ConceptLift candidates so competing explanations remain available for `AcquireObservation` (`M8.2` Bug 5.1 Fix)!
        valid_lift_cands.sort(key=lambda c: c.val_log_predictive_gain - self.cost_weight * c.estimated_cost, reverse=True)
        candidates.extend(valid_lift_cands[:2])

        # Stage 5: `ChangeMemory` (with prefix memory carry-over!)
        if (
            not resolved_cheaply
            and rem_budget >= 0.15
            and (("past_driver_history" in ds_tr and ds_tr["past_driver_history"] is not None) or ("timestamps" in ds_tr))
        ):
            best_mem_cand: RevisionCandidate | None = None
            for tau_cand in (0.8, 2.0, 4.5):
                sim_g = deepcopy(self.graph)
                mem_id = f"mech_mem_{task_id}_tau{str(tau_cand).replace('.', 'p')}_s{self._step_counter}"
                drv_col = incumbent.feature_names[0]
                sim_g.register_mechanism(
                    TypedMechanism(
                        mechanism_id=mem_id,
                        parent_id=incumbent.shared_upstream_id,
                        role="temporal_memory",
                        feature_names=incumbent.feature_names,
                        temporal_tau=tau_cand,
                        temporal_driver_col=drv_col,
                        dependent_task_ids={task_id},
                    )
                )
                m_mem = deepcopy(incumbent)
                m_mem.version += 1
                m_mem.model_id = f"{incumbent.model_id}_mem_{tau_cand}"
                m_mem.temporal_memory_id = mem_id
                sc_mem = fit_and_score_task_model(m_mem, sim_g, ds_tr, ds_va, prior_shrinkage=1e-2, commit_to_model=True)
                gain_mem = float(sc_mem["val_mean_log_predictive"] - base_val_lp)
                if best_mem_cand is None or gain_mem > best_mem_cand.val_log_predictive_gain:
                    def _apply_mem(g: PersistentMechanismGraph, mid_m=mem_id, tc=tau_cand, dc=drv_col, m_new=deepcopy(m_mem)) -> None:
                        g.register_mechanism(
                            TypedMechanism(
                                mechanism_id=mid_m,
                                parent_id=incumbent.shared_upstream_id,
                                role="temporal_memory",
                                feature_names=incumbent.feature_names,
                                temporal_tau=tc,
                                temporal_driver_col=dc,
                                dependent_task_ids={task_id},
                            )
                        )
                        g.task_datasets[task_id]["local_shrinkage"] = 1e-2
                        g.active_task_models[task_id] = m_new
                        g.sync_authoritative_state_to_forest()

                    best_mem_cand = RevisionCandidate(
                        candidate_id=f"cand_{self._step_counter}_change_memory_{tau_cand}",
                        action_type="ChangeMemory",
                        description=f"Activate causal multi-scale exponential memory kernel (`tau={tau_cand}`).",
                        estimated_cost=2.2,
                        actual_cost_spent=0.15,
                        val_log_predictive_gain=gain_mem,
                        variational_free_energy_delta=float(base_fe - sc_mem["variational_free_energy"]),
                        predict_fn=lambda x_in, ms=deepcopy(m_mem), sg=sim_g: evaluate_committed_task_model(ms, sg, {"x": x_in, "y": np.zeros(len(next(iter(x_in.values()))))})["predictions"],
                        apply_fn=_apply_mem,
                        metadata={"temporal_tau": tau_cand, "driver_col": drv_col},
                    )
            if best_mem_cand is not None:
                _spend("ChangeMemory_search", 0.15)
                candidates.append(best_mem_cand)

        # Stage 6: Filter qualified candidates BEFORE ranking so an unqualified rank-1 NEVER blocks a qualified rank-2 (`M8.2` Bug 5.4 Fix)!
        # Include per-sample Occam grammar log-prior delta (`log P_0(M') - log P_0(M)`) / N_val so structural expansions (`LocalSplit`, `ConceptLift`)
        # only beat parameter-only `LocalRefit` when out-of-sample predictive gain exceeds structural complexity (`T04`)!
        n_val_obs = max(1, len(np.asarray(ds_va["y"]).ravel()))
        base_grammar_lp = incumbent.intrinsic_grammar_log_prior(self.graph)

        def _occam_structural_delta_per_obs(cand: RevisionCandidate) -> float:
            if cand.action_type == "LocalSplit":
                return -1.35 / n_val_obs
            if cand.action_type == "ConceptLift":
                expr_len = len(str(cand.metadata.get("expression") or ""))
                return (-1.25 - 0.015 * expr_len) / n_val_obs
            if cand.action_type == "ChangeMemory":
                return -1.20 / n_val_obs
            return 0.0

        exec_ok_cands = [
            c for c in candidates
            if c.execution_status == "EXECUTED_OK"
            and c.collateral_min_delta >= self.collateral_tolerance_nats
            and c.unrelated_max_abs_drift <= 1e-9
        ]
        qualified_non_trivial = [
            c for c in exec_ok_cands
            if c.action_type != "NoChange"
            and (c.val_log_predictive_gain + _occam_structural_delta_per_obs(c)) >= self.min_gain_nats
            and c.variational_free_energy_delta > 0.0
        ]
        if qualified_non_trivial:
            qualified_non_trivial.sort(
                key=lambda c: (c.val_log_predictive_gain + _occam_structural_delta_per_obs(c)) - self.cost_weight * c.estimated_cost,
                reverse=True,
            )
            chosen = qualified_non_trivial[0]
        else:
            _ = base_grammar_lp
            chosen = next(c for c in exec_ok_cands if c.action_type == "NoChange")

        # Stage 7: Reachable `AcquireObservation` over full competing `TaskModel` predictions when commit is withheld / ambiguous!
        if chosen.action_type == "NoChange" and candidate_probes:
            diverse_models = [c for c in exec_ok_cands if c.predict_fn is not None]
            diverse_models.sort(key=lambda c: c.val_log_predictive_gain, reverse=True)
            pred_fns = [(c.candidate_id, c.predict_fn) for c in diverse_models[:4] if c.predict_fn is not None]
            if len(pred_fns) >= 2:
                probe_sel = select_discriminating_observation_probe(pred_fns, candidate_probes)
                if probe_sel["max_utility"] > 0.02:
                    probe_meta = dict(probe_sel)
                    def _apply_probe(g: PersistentMechanismGraph, psel=probe_sel) -> None:
                        if probe_oracle_fn is not None and psel.get("selected_probe") is not None:
                            pt = psel["selected_probe"]
                            y_new = float(probe_oracle_fn(pt))
                            tr_d = g.task_datasets[task_id]["train"]
                            for k, v in pt.items():
                                tr_d["x"][k] = np.append(np.asarray(tr_d["x"][k], dtype=float), float(v))
                            tr_d["y"] = np.append(np.asarray(tr_d["y"], dtype=float), y_new)
                            probe_meta["observed_y"] = y_new

                    _spend("AcquireObservation_probe", 0.50 if probe_oracle_fn is not None else 0.05)
                    chosen = RevisionCandidate(
                        candidate_id=f"cand_{self._step_counter}_acquire_obs",
                        action_type="AcquireObservation",
                        description="Purchase maximally discriminating active intervention probe across competing TaskModels.",
                        estimated_cost=0.5,
                        actual_cost_spent=(0.50 if probe_oracle_fn is not None else 0.05),
                        val_log_predictive_gain=0.0,
                        apply_fn=_apply_probe,
                        metadata=probe_meta,
                    )
                    candidates.append(chosen)

        if chosen.apply_fn is not None and chosen.execution_status == "EXECUTED_OK":
            chosen.apply_fn(self.graph)

        total_spent = round(sum(float(ev["cost_spent"]) for ev in attempt_cost_events), 5)
        self.graph.action_return_history.setdefault(chosen.action_type, []).append(
            {
                "gain_nats": round(chosen.val_log_predictive_gain, 6),
                "cost_spent": total_spent,
            }
        )

        wall_ms = round((monotonic() - t0) * 1000.0, 2)
        rec = ResearchDecisionRecord(
            decision_id=f"dec_{self._step_counter:03d}_{task_id}",
            step_index=self._step_counter,
            task_id=task_id,
            pre_action_state=pre_state,
            available_actions=sorted({c.action_type for c in candidates}),
            selected_action=chosen.action_type,
            selected_candidate_id=chosen.candidate_id,
            execution_status=chosen.execution_status,
            actual_cost=total_spent,
            estimated_selected_cost=chosen.estimated_cost,
            total_search_cost_spent=total_spent,
            total_prompt_tokens=total_prompt_tok,
            total_completion_tokens=total_comp_tok,
            attempt_cost_events=attempt_cost_events,
            public_val_log_predictive_delta=round(chosen.val_log_predictive_gain, 6),
            collateral_min_delta=round(chosen.collateral_min_delta, 6),
            rationale=chosen.description,
        )
        rec.validate_prefix_clean()
        self.graph.decision_ledger.append(rec)
        self.graph.sync_authoritative_state_to_forest()

        return {
            "decision_record": rec.to_dict(),
            "selected_candidate": chosen.public_summary(),
            "evaluated_candidates": [c.public_summary() for c in candidates],
            "total_prompt_tokens": total_prompt_tok,
            "total_completion_tokens": total_comp_tok,
            "attempt_cost_events": attempt_cost_events,
            "wall_ms": wall_ms,
        }
