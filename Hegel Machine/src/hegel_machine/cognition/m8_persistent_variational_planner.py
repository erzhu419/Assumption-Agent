"""Hegel M8.0 Persistent Mechanism Graph, Variational Conduit Models & Dialectical Action Planner.

Synthesizes the M8 variational-planning architecture directly onto the existing `PhilosophicalForest`
(`wormhole_forest.py`), multi-scale temporal folding (`m5_temporal_multiscale_and_base_subtree.py`),
and official `codex` ChatGPT CLI (`FrontierCodexClient`), while satisfying all 13 mathematical invariants
audited in `Hegel_M7_7_M8_Audit`:

  1. **Four Strictly Separated Core Objects (`T01`, `T09`)**:
     - `TypedMechanism` inside `PersistentMechanismGraph` (`G`, backed by `PhilosophicalForest`):
       persistent shared upstream premises/representations, observable-guarded regime branches,
       lifted interaction conduits, and multi-scale exponential memory kernels, with versioned lineage
       and explicit `dependent_task_ids` dependency tracking.
     - `TaskModel` (`M`): a complete candidate task explanation combining shared upstream parameters,
       internal observable-gated equation conduits (`pi_b(x)`), optional lifted basis conduits, and
       optional causal temporal state `z_t = exp(-dt/tau) z_{t-1} + (1 - exp(-dt/tau)) u_{t-1}`, paired
       with a single normalized observation likelihood (`gated_predictor_nig`).
     - `RevisionCandidate`: an executable competing graph/model revision under one of 7 actions:
       `NoChange`, `LocalRefit`, `LocalSplit`, `AncestorReweight`, `ConceptLift` (algebraic + live
       `GPT CLI` `codex` structured synthesis), `ChangeMemory`, and `AcquireObservation`.
     - `ResearchDecisionRecord` (`tau`): strictly prefix-clean pre-action state, feasible actions,
       chosen action, actual cost, public validation delta, and 3-valued execution/epistemic status
       (`EXECUTED_OK` vs `EXECUTION_ERROR` never conflated with `REFUTED_MECHANISM`, `T11`).

  2. **Proper Scale-Covariant Normal-Inverse-Gamma (`NIG`) Evidence & Sequential Belief Hygiene (`T02-T08`)**:
     - Exact Gaussian curvature `1 / sigma^2` (`T02`) and Student-t Expected Fisher PSD vs non-PSD
       observed tail curvature (`T03`).
     - Proper `sigma^2 ~ InverseGamma(alpha_0, beta_0)` + empirical-variance-covariant `Lambda_0`
       guaranteeing finite evidence at zero residual (`T04`) and exact invariance under invertible
       coordinate rescaling `x_j -> c_j * x_j` (`T08`).
     - Fixed intrinsic grammar complexity prior `log p_0(M)` independent of candidate pool size +
       canonical semantic deduplication before model-posterior normalization (`T05`).
     - Disjoint batch-ID ledger preventing prior/data double counting while matching joint-from-prior
       and sequential-from-posterior updates to machine precision (`T07`).
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from time import monotonic
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

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
# 1. Mathematical & Statistical Primitives (Audited Against T01-T08, T10)
# ============================================================================


def evaluate_gated_vs_mixture_log_likelihood(
    y: np.ndarray,
    branch_preds: np.ndarray,
    gate_weights: np.ndarray,
    sigma: float,
) -> dict[str, float]:
    """Explicitly separate gated deterministic prediction likelihood from mixture-of-experts likelihood (`T01`).

    Args:
        y: shape `(n,)`
        branch_preds: shape `(n, B)` predictions `f_b(x_i)`
        gate_weights: shape `(n, B)` row-normalized non-negative gates `sum_b pi_b(x_i) == 1`
        sigma: positive observation noise scale
    """
    yv = np.asarray(y, dtype=float).ravel()
    fp = np.asarray(branch_preds, dtype=float)
    gw = np.asarray(gate_weights, dtype=float)
    s2 = max(float(sigma) ** 2, 1e-12)
    norm_c = -0.5 * math.log(2.0 * math.pi * s2)

    # 1. Gated conduit predictor: y_i ~ N(sum_b pi_b(x_i) f_b(x_i), sigma^2)
    mean_pred = np.sum(gw * fp, axis=1)
    ll_gated = float(np.sum(norm_c - 0.5 * ((yv - mean_pred) ** 2) / s2))

    # 2. Mixture-of-experts likelihood: p(y_i) = sum_b pi_b(x_i) N(y_i | f_b(x_i), sigma^2)
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
        # NLL = sum_i r_i^2 / (2 * sigma^2)  =>  d^2 NLL / dr_i^2 = 1 / sigma^2 (NOT 2 / sigma^2!)
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
        # NLL_i = ((nu + 1) / 2) * log(1 + r_i^2 / (nu * s2))
        # d^2 NLL_i / dr_i^2 = (nu + 1) * (nu * s2 - r_i^2) / (nu * s2 + r_i^2)^2  (negative when |r_i| > sqrt(nu)*sigma!)
        denom = nu * s2 + r ** 2
        w_obs = (nu + 1.0) * (nu * s2 - r ** 2) / np.maximum(denom ** 2, 1e-24)
        H_obs_gn = (J.T * w_obs) @ J
        eig_obs = float(np.min(np.linalg.eigvalsh(0.5 * (H_obs_gn + H_obs_gn.T)))) if J.size else float(np.min(w_obs))
        # Exact Expected Fisher weight E[d^2 NLL / dr^2] = ((nu + 1) / (nu + 3)) / s2 > 0 (strictly PSD!)
        fisher_scalar = ((nu + 1.0) / (nu + 3.0)) / s2
        F_exp = fisher_scalar * (J.T @ J)
        # IRLS robustness weight (d NLL / dr) / r = (nu + 1) / (nu * s2 + r^2) > 0 (PSD surrogate, labeled honestly)
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
    intercept_col_Zero: bool = True,
) -> dict[str, Any]:
    """Analytic Normal-Inverse-Gamma (`NIG`) Bayesian regression with exact coordinate-scaling invariance (`T04`, `T08`).

    Prior specification:
      - `sigma^2 ~ InverseGamma(alpha_0, beta_0)` with strictly positive `alpha_0 > 0, beta_0 > 0`
        (so even when `SSE == 0`, `beta_n = beta_0 > 0` and marginal log-evidence is strictly finite — no noise collapse!).
      - `theta | sigma^2 ~ Normal(mu_0, sigma^2 * Lambda_0^{-1})` where `Lambda_{0,jj} = prior_shrinkage * col_scale_j^2`,
        with `col_scale_j^2 = mean(Phi[:, j]^2)` for non-constant columns! Under `Phi[:, j] -> c * Phi[:, j]`
        (and `mu_0[j] -> mu_0[j] / c`), `det(Lambda_0) / det(Lambda_n)` and `y_hat` are 100% invariant to `c`!
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
    # If a column is identically zero, give unit prior precision so Lambda_0 stays strictly SPD
    lam_diag = np.where(col_msq > 1e-18, float(prior_shrinkage) * col_msq, float(prior_shrinkage))
    if intercept_col_Zero and d > 0 and np.allclose(X[:, 0], 1.0):
        lam_diag[0] = float(prior_shrinkage) * 1e-2
    Lambda_0 = np.diag(lam_diag)

    y_var = float(np.var(yv)) if n > 1 else 0.0
    b0 = float(beta_0) if beta_0 is not None else max(float(alpha_0) * max(y_var * 0.15, 1e-3), 1e-3)
    a0 = max(float(alpha_0), 1.01)

    Lambda_n = Lambda_0 + X.T @ X
    rhs = Lambda_0 @ mu0 + X.T @ yv
    mu_n = np.linalg.solve(Lambda_n, rhs)

    sign0, logdet0 = np.linalg.slogdet(Lambda_0)
    sign_n, logdet_n = np.linalg.slogdet(Lambda_n)
    assert sign0 > 0 and sign_n > 0

    quad_term = float(yv @ yv + mu0 @ (Lambda_0 @ mu0) - mu_n @ (Lambda_n @ mu_n))
    quad_term = max(0.0, quad_term)
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
    r2 = 1.0 - float(np.sum(resid ** 2)) / max(tot_ss, 1e-12) if tot_ss > 1e-12 else (1.0 if float(np.sum(resid ** 2)) < 1e-12 else 0.0)

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
    """Exact Student-t posterior predictive density `p(y_* | X_*, D)` for out-of-sample / prequential scoring."""
    Xe = np.asarray(Phi_eval, dtype=float)
    if Xe.ndim == 1:
        Xe = Xe[:, None]
    ye = np.asarray(y_eval, dtype=float).ravel()
    mu_n = np.asarray(fit_info["posterior_mean"], dtype=float)
    Lambda_n = np.asarray(fit_info["posterior_precision"], dtype=float)
    a_n = float(fit_info["alpha_n"])
    b_n = float(fit_info["beta_n"])

    pred_mean = Xe @ mu_n
    V_x = np.sum(Xe * np.linalg.solve(Lambda_n, Xe.T).T, axis=1)
    if domain_inflation is not None:
        V_x = V_x + np.maximum(0.0, np.asarray(domain_inflation, dtype=float).ravel())
    nu = 2.0 * a_n
    scale2 = (b_n / a_n) * (1.0 + np.maximum(0.0, V_x))

    z2 = ((ye - pred_mean) ** 2) / np.maximum(nu * scale2, 1e-18)
    log_pdf = (
        float(gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu))
        - 0.5 * np.log(nu * math.pi * np.maximum(scale2, 1e-18))
        - 0.5 * (nu + 1.0) * np.log1p(z2)
    )
    rmse = float(np.sqrt(np.mean((ye - pred_mean) ** 2)))
    return {
        "mean_log_predictive": float(np.mean(log_pdf)),
        "sum_log_predictive": float(np.sum(log_pdf)),
        "rmse": rmse,
        "predictions": pred_mean,
        "predictive_std": np.sqrt(np.maximum(scale2 * (nu / max(nu - 2.0, 1e-6)), 1e-18)),
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
        """Update from current posterior using ONLY unseen `batch_id` (`T07` guard)."""
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
        """Recompute posterior from original `(prior_mean, prior_precision)` across valid historical batches."""
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
    """Compute posterior weights over distinct semantic Equivalence Classes (`T05`).

    Args:
        models_and_objectives: list of `(model_id, canonical_semantic_key, variational_free_energy_F)`
            where `F(M) = -log p(D | M) - log p_0(M)` uses an intrinsic grammar prior `log p_0(M)`
            independent of candidate pool size!
    """
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
# 2. Safe Expression Conduit Sandbox & Lipschitz / Domain Diagnostics (`T10`)
# ============================================================================


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
    """Compute local Lipschitz conditioning + empirical support box (`T10` honest numeric guard, NOT truth filter!)."""
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
# 3. Four Core M8 Objects: `PersistentMechanismGraph`, `TaskModel`, `RevisionCandidate`, `ResearchDecisionRecord`
# ============================================================================


@dataclass
class TypedMechanism:
    mechanism_id: str
    parent_id: str | None
    role: str  # 'shared_upstream_premise' | 'regime_branch' | 'interaction_lift' | 'temporal_memory'
    feature_names: tuple[str, ...]
    version: int = 1
    shared_weights: dict[str, float] = field(default_factory=dict)
    scope_guard: dict[str, Any] | None = None  # {'field': str, 'op': str, 'value': float}
    python_expression: str | None = None
    temporal_tau: float | None = None
    temporal_driver_col: str | None = None
    support_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    empirical_lipschitz: float = 1.0
    dependent_task_ids: set[str] = field(default_factory=set)
    epistemic_status: str = "SUPPORTED_IN_SCOPE"  # 'SUPPORTED_IN_SCOPE' | 'CANDIDATE' | 'UNRESOLVED_EXECUTION_FAILURE' | 'SUPERSEDED'
    provenance_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dependent_task_ids"] = sorted(self.dependent_task_ids)
        return d


class PersistentMechanismGraph:
    """Persistent cross-task mechanism graph `G` synchronized directly with `PhilosophicalForest` (`T01`)."""

    def __init__(self, forest: PhilosophicalForest | None = None) -> None:
        self.forest = forest if forest is not None else PhilosophicalForest()
        self.mechanisms: dict[str, TypedMechanism] = {}
        self.mechanism_lineage: dict[str, list[dict[str, Any]]] = {}
        self.task_datasets: dict[str, dict[str, Any]] = {}
        self.active_task_models: dict[str, TaskModel] = {}
        self.decision_ledger: list[ResearchDecisionRecord] = []

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
        return mech.mechanism_id

    def minimal_affected_task_ids(self, mechanism_id: str) -> list[str]:
        """Return ONLY tasks that actually depend on `mechanism_id` or its descendants (`PremiseDependency` discipline)."""
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


@dataclass
class RegimeConduitBranch:
    """One internal equation-conduit branch inside a `TaskModel` (`pi_b(x)` observable gate + linear/basis weights)."""

    branch_id: str
    mechanism_id: str
    scope_guard: dict[str, Any] | None = None  # None => default fallback regime
    gate_sharpness: float = 12.0


@dataclass
class TaskModel:
    """Complete task explanation `M` (`T01`: internal regime gates != posterior over models!)."""

    model_id: str
    task_id: str
    shared_upstream_id: str
    feature_names: tuple[str, ...]
    regime_branches: list[RegimeConduitBranch] = field(default_factory=list)
    interaction_lift_ids: list[str] = field(default_factory=list)
    temporal_memory_id: str | None = None
    nonlinear_power_exponent: float | None = None  # optional nonlinear conduit e.g. w^2 * x (`T06` zero-Jacobian escape)
    fit_summary: dict[str, Any] = field(default_factory=dict)

    def referenced_mechanism_ids(self) -> set[str]:
        out = {self.shared_upstream_id, *(b.mechanism_id for b in self.regime_branches), *self.interaction_lift_ids}
        if self.temporal_memory_id:
            out.add(self.temporal_memory_id)
        return out

    def canonical_semantic_key(self, graph: PersistentMechanismGraph) -> str:
        """Canonical semantic fingerprint invariant to model ID / duplicate registration (`T05`)."""
        shared_m = graph.mechanisms[self.shared_upstream_id]
        guards = sorted(
            json.dumps(b.scope_guard, sort_keys=True) if b.scope_guard else "DEFAULT"
            for b in self.regime_branches
        )
        lifts = sorted(
            str(graph.mechanisms[lid].python_expression or lid)
            for lid in self.interaction_lift_ids
            if lid in graph.mechanisms
        )
        tau = graph.mechanisms[self.temporal_memory_id].temporal_tau if (self.temporal_memory_id and self.temporal_memory_id in graph.mechanisms) else None
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
        """Fixed grammar prior `log p_0(M)` independent of candidate pool size (`T05`)."""
        n_branches = max(1, len(self.regime_branches))
        n_lifts = len(self.interaction_lift_ids)
        has_mem = 1 if self.temporal_memory_id else 0
        expr_chars = sum(
            len(str(graph.mechanisms[lid].python_expression or ""))
            for lid in self.interaction_lift_ids
            if lid in graph.mechanisms
        )
        # Mild Occam complexity penalty (~1.4 nats per extra split/lift/memory state)
        return float(-1.35 * (n_branches - 1) - 1.25 * n_lifts - 1.20 * has_mem - 0.015 * expr_chars)


def _compute_causal_memory_feature(
    dataset: dict[str, Any],
    driver_col: str,
    tau: float,
) -> np.ndarray:
    """Compute strictly prefix-causal (`<= t`) Mori-Zwanzig exponential history state `z_t` (`Pi_fold` integration).

    Supports both sequential trajectory arrays (`timestamps`, `episode_ids`) and explicit pre-sampled
    history sequences (`past_driver_history`: shape `(n, K)` causal history before present snapshot!).
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
        return z

    u = np.asarray(x_cols[driver_col], dtype=float).ravel()
    ts = np.asarray(dataset.get("timestamps", np.arange(n, dtype=float)), dtype=float).ravel()
    ep_ids = np.asarray(dataset.get("trajectory_ids", np.zeros(n, dtype=int)))
    z = np.zeros(n, dtype=float)
    for i in range(1, n):
        if ep_ids[i] != ep_ids[i - 1]:
            z[i] = 0.0
        else:
            dt = max(float(ts[i] - ts[i - 1]), 1e-4)
            decay = math.exp(-dt / tau_pos)
            z[i] = decay * z[i - 1] + (1.0 - decay) * u[i - 1]
    return z


def _evaluate_branch_gates(
    branches: list[RegimeConduitBranch],
    x_cols: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute row-normalized observable regime gates `pi_b(x)` (`shape = (n, B)`)."""
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
        scale = max(float(np.std(vals)), 0.15)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build `(Phi, shared_offset, support_inflation)` for `TaskModel` on `dataset` (`y - shared_offset ~ Phi @ theta_local`)."""
    x_cols = dataset["x"]
    n = len(next(iter(x_cols.values())))
    shared_mech = graph.mechanisms[model.shared_upstream_id]

    shared_offset = np.zeros(n, dtype=float)
    for fnm, w_val in shared_mech.shared_weights.items():
        if fnm in x_cols:
            shared_offset = shared_offset + float(w_val) * np.asarray(x_cols[fnm], dtype=float).ravel()

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
            fn = compile_safe_feature_conduit(l_mech.python_expression, model.feature_names)
            lift_vec = fn(x_cols)
            design_blocks.append(lift_vec[:, None])
            if l_mech.support_bounds:
                support_inflation += compute_out_of_support_inflation(
                    x_cols,
                    l_mech.support_bounds,
                    lipschitz_scale=l_mech.empirical_lipschitz,
                )

    if model.temporal_memory_id and model.temporal_memory_id in graph.mechanisms:
        t_mech = graph.mechanisms[model.temporal_memory_id]
        drv = t_mech.temporal_driver_col or model.feature_names[0]
        mem_vec = _compute_causal_memory_feature(dataset, drv, float(t_mech.temporal_tau or 1.5))
        design_blocks.append(mem_vec[:, None])

    Phi = np.hstack(design_blocks)
    return Phi, shared_offset, support_inflation


def fit_and_score_task_model(
    model: TaskModel,
    graph: PersistentMechanismGraph,
    train_data: dict[str, Any],
    val_data: dict[str, Any] | None = None,
    *,
    prior_shrinkage: float = 1e-2,
) -> dict[str, Any]:
    """Fit `TaskModel` via exact NIG marginal evidence + evaluate out-of-sample Student-t log predictive score."""
    Phi_tr, off_tr, _ = build_task_design_matrix(model, graph, train_data)
    y_tr = np.asarray(train_data["y"], dtype=float).ravel()
    y_adj_tr = y_tr - off_tr

    nig = fit_scale_covariant_nig_linear_model(
        Phi_tr,
        y_adj_tr,
        prior_shrinkage=prior_shrinkage,
        intercept_col_Zero=(len(model.regime_branches) <= 1),
    )
    log_p0 = model.intrinsic_grammar_log_prior(graph)
    variational_free_energy = float(-nig["log_marginal_likelihood"] - log_p0)

    out: dict[str, Any] = {
        "nig_fit": nig,
        "log_marginal_likelihood": nig["log_marginal_likelihood"],
        "grammar_log_prior": log_p0,
        "variational_free_energy": variational_free_energy,
        "train_rmse": float(np.sqrt(np.mean((y_tr - (off_tr + Phi_tr @ nig["posterior_mean"])) ** 2))),
        "residuals": y_tr - (off_tr + Phi_tr @ nig["posterior_mean"]),
    }

    if val_data is not None:
        Phi_va, off_va, inf_va = build_task_design_matrix(model, graph, val_data)
        y_va = np.asarray(val_data["y"], dtype=float).ravel()
        pred_va = predict_nig_student_t_log_score(nig, Phi_va, y_va - off_va, domain_inflation=inf_va)
        out["val_mean_log_predictive"] = pred_va["mean_log_predictive"]
        out["val_sum_log_predictive"] = pred_va["sum_log_predictive"]
        out["val_rmse"] = float(np.sqrt(np.mean((y_va - (off_va + pred_va["predictions"])) ** 2)))
    else:
        out["val_mean_log_predictive"] = float(nig["log_marginal_likelihood"] / max(len(y_tr), 1))
        out["val_sum_log_predictive"] = float(nig["log_marginal_likelihood"])
        out["val_rmse"] = out["train_rmse"]

    model.fit_summary = {
        "posterior_mean": nig["posterior_mean"].tolist(),
        "variational_free_energy": variational_free_energy,
        "val_mean_log_predictive": out["val_mean_log_predictive"],
        "val_rmse": out["val_rmse"],
        "train_rmse": out["train_rmse"],
    }
    return out


# ============================================================================
# 4. Honest Proposal-Ranking Residual Diagnostics (Non-Brittle, Respects T06/T08)
# ============================================================================


def diagnose_residual_signals(
    residuals: np.ndarray,
    dataset: dict[str, Any],
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    """Compute scale-invariant residual diagnostics solely to prioritize `RevisionCandidate` generation (`T06`, `T08`).

    Explicitly separates:
      - `linear_shift_r2`: global feature-aligned linear residual correlation (suggests `AncestorReweight` / `LocalRefit`),
      - `tail_concentration_ratio`: top-10% localized outlier share (suggests `LocalSplit` regime exception),
      - `pairwise_interaction_r2`: quadratic/product basis explainability over additive residual (suggests `ConceptLift`),
      - `history_divergence_r2`: lag/history state explainability on identical-present inputs (suggests `ChangeMemory`),
      - `heteroscedastic_abs_r2`: whether `|r|` depends on `x` while conditional mean signal is near zero (`r = x * eps`,
        classified honestly as `HETEROSCEDASTIC_OR_UNRESOLVED_NOISE`, NEVER misclassified as structural mean failure!).
    """
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

    # Standardize columns so diagnostics are 100% invariant to coordinate rescaling x_j -> c_j * x_j (`T08`)!
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
        Z_mem = np.column_stack([Z, (mem_probe - np.mean(mem_probe)) / max(float(np.std(mem_probe)), 1e-9)])
        hist_r2 = max(0.0, _ols_adj_r2(Z_mem, r) - lin_r2)

    mean_explainable = max(lin_r2, inter_r2, hist_r2)
    status = "STRUCTURED_RESIDUAL_CANDIDATES"
    if mean_explainable < 0.08 and coherent_tail_conc < 0.35:
        status = "HETEROSCEDASTIC_OR_UNRESOLVED_NOISE" if hetero_r2 > 0.15 else "UNRESOLVED_WHITE_OR_WEAK_SIGNAL"

    scores = [
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


# ============================================================================
# 5. Competing Revision Candidates, GPT CLI (`codex`) ConceptLift, & Prefix-Clean Ledger
# ============================================================================


@dataclass
class RevisionCandidate:
    candidate_id: str
    action_type: str  # one of M8_ACTIONS
    description: str
    estimated_cost: float
    execution_status: str = "EXECUTED_OK"  # 'EXECUTED_OK' | 'EXECUTION_ERROR' | 'SKIPPED'
    epistemic_outcome: str = "EVALUATED"  # 'EVALUATED' | 'UNRESOLVED_EXECUTION_FAILURE'
    val_log_predictive_gain: float = 0.0
    collateral_min_delta: float = 0.0
    unrelated_max_abs_drift: float = 0.0
    variational_free_energy_delta: float = 0.0
    apply_fn: Callable[[PersistentMechanismGraph], None] | None = field(default=None, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def public_summary(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action_type": self.action_type,
            "description": self.description,
            "estimated_cost": self.estimated_cost,
            "execution_status": self.execution_status,
            "epistemic_outcome": self.epistemic_outcome,
            "val_log_predictive_gain": round(self.val_log_predictive_gain, 6),
            "collateral_min_delta": round(self.collateral_min_delta, 6),
            "unrelated_max_abs_drift": round(self.unrelated_max_abs_drift, 8),
            "variational_free_energy_delta": round(self.variational_free_energy_delta, 6),
            "metadata": deepcopy(self.metadata),
        }


@dataclass
class ResearchDecisionRecord:
    """Strictly prefix-clean trajectory record `tau` (`T09`: zero post-hoc / oracle fields in `pre_action_state`!)."""

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
    """Invoke live official `codex` ChatGPT CLI (`gpt-6.2-sol`) to synthesize a `ConceptLift` equation conduit!"""
    x_cols = train_data["x"]
    r = np.asarray(residuals, dtype=float).ravel()
    top_idx = np.argsort(np.abs(r))[-14:][::-1]
    quant_idx = np.argsort(r)[np.linspace(0, len(r) - 1, 14, dtype=int)]
    sample_idx = np.unique(np.concatenate([top_idx, quant_idx]))[:26]

    rows_preview = []
    for i in sample_idx:
        rows_preview.append(
            {
                "features": {fnm: round(float(x_cols[fnm][i]), 4) for fnm in feature_names},
                "residual_y_minus_pred": round(float(r[i]), 4),
            }
        )

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
            "task_id": task_id,
            "feature_names": list(feature_names),
            "residual_diagnostics": diagnostics,
            "basis_residual_correlations": corr_table,
            "representative_residual_samples": rows_preview,
        },
        ensure_ascii=False,
    )
    res = codex_client.call_structured(
        stage_subdir=stage_subdir,
        system_prompt=sys_prompt,
        conversation_turns=[{"role": "user", "content": user_payload}],
        output_schema=M8_CONCEPT_LIFT_SCHEMA,
    )
    return res


def select_discriminating_observation_probe(
    competing_predict_fns: list[tuple[str, Callable[[dict[str, np.ndarray]], np.ndarray]]],
    candidate_probes: list[dict[str, float]],
    *,
    probe_costs: list[float] | None = None,
) -> dict[str, Any]:
    """Select active intervention `AcquireObservation` maximizing prediction disagreement per unit cost (`probing` integration)."""
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


class M8VariationalPlanner:
    """Unified M8 Planner operating over ONE `PersistentMechanismGraph`, ONE action space, and ONE prefix-clean ledger."""

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
        self._step_counter = 0

    def register_task(
        self,
        *,
        task_id: str,
        shared_upstream_id: str,
        feature_names: tuple[str, ...],
        train_data: dict[str, Any],
        val_data: dict[str, Any],
        initial_local_shrinkage: float = 12.0,
    ) -> TaskModel:
        """Register a task anchored to `shared_upstream_id` under hierarchical local shrinkage around shared upstream weights."""
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
        tm = TaskModel(
            model_id=f"model_{task_id}_v1",
            task_id=task_id,
            shared_upstream_id=shared_upstream_id,
            feature_names=feature_names,
            regime_branches=[RegimeConduitBranch(branch_id="default", mechanism_id=base_br_id, scope_guard=None)],
        )
        fit_and_score_task_model(tm, self.graph, train_data, val_data, prior_shrinkage=float(initial_local_shrinkage))
        self.graph.active_task_models[task_id] = tm
        return tm

    def evaluate_task_zero_shot_from_shared(self, task_id: str) -> dict[str, Any]:
        """Evaluate `task_id` purely from `shared_upstream_id` (`theta_local -> 0`, `prior_shrinkage=1e8`) to measure forward transfer."""
        tm = deepcopy(self.graph.active_task_models[task_id])
        ds = self.graph.task_datasets[task_id]
        return fit_and_score_task_model(tm, self.graph, ds["train"], ds["val"], prior_shrinkage=1e8)

    def evaluate_and_revise_task(
        self,
        task_id: str,
        *,
        codex_client: Any | None = None,
        injected_broken_expr: str | None = None,
        candidate_probes: list[dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        """Generate & adjudicate all competing `RevisionCandidate`s (`NoChange`, `LocalRefit`, `AncestorReweight`,
        `LocalSplit`, `ConceptLift`, `ChangeMemory`, `AcquireObservation`) on one unified persistent graph `G`!
        """
        self._step_counter += 1
        t0 = monotonic()
        incumbent = self.graph.active_task_models[task_id]
        t_entry = self.graph.task_datasets[task_id]
        ds_tr = t_entry["train"]
        ds_va = t_entry["val"]
        cur_shrink = float(t_entry.get("local_shrinkage", 12.0))

        inc_eval = fit_and_score_task_model(
            incumbent,
            self.graph,
            ds_tr,
            ds_va,
            prior_shrinkage=cur_shrink,
        )
        base_val_lp = float(inc_eval["val_mean_log_predictive"])
        base_fe = float(inc_eval["variational_free_energy"])
        diag = diagnose_residual_signals(inc_eval["residuals"], ds_tr, incumbent.feature_names)

        # Snapshot baseline hierarchical scores across ALL registered tasks for collateral & transfer checks
        baseline_task_scores: dict[str, float] = {}
        for other_tid, other_tm in self.graph.active_task_models.items():
            o_entry = self.graph.task_datasets[other_tid]
            o_sc = fit_and_score_task_model(
                deepcopy(other_tm),
                self.graph,
                o_entry["train"],
                o_entry["val"],
                prior_shrinkage=float(o_entry.get("local_shrinkage", 12.0)),
            )
            baseline_task_scores[other_tid] = float(o_sc["val_mean_log_predictive"])

        # Strictly prefix-clean pre-action state (`T09`)
        pre_state = {
            "task_id": task_id,
            "incumbent_model_id": incumbent.model_id,
            "incumbent_val_mean_log_predictive": round(base_val_lp, 6),
            "incumbent_variational_free_energy": round(base_fe, 6),
            "incumbent_val_rmse": round(float(inc_eval["val_rmse"]), 6),
            "residual_diagnostics": diag,
            "shared_upstream_id": incumbent.shared_upstream_id,
            "shared_upstream_version": self.graph.mechanisms[incumbent.shared_upstream_id].version,
        }

        candidates: list[RevisionCandidate] = []

        # 0. `NoChange` safety baseline
        candidates.append(
            RevisionCandidate(
                candidate_id=f"cand_{self._step_counter}_nochange",
                action_type="NoChange",
                description="Retain incumbent mechanism graph and classify residual as unresolved/noise.",
                estimated_cost=0.0,
                val_log_predictive_gain=0.0,
                collateral_min_delta=0.0,
                unrelated_max_abs_drift=0.0,
                variational_free_energy_delta=0.0,
                apply_fn=None,
            )
        )

        # 1. `LocalRefit` (relax task-local hierarchical shrinkage `12.0 -> 1e-2` without altering shared upstream!)
        m_refit = deepcopy(incumbent)
        m_refit.model_id = f"{incumbent.model_id}_local_refit"
        sc_refit = fit_and_score_task_model(m_refit, self.graph, ds_tr, ds_va, prior_shrinkage=1e-2)

        def _apply_local_refit(g: PersistentMechanismGraph, m_new=m_refit) -> None:
            g.task_datasets[task_id]["local_shrinkage"] = 1e-2
            fit_and_score_task_model(m_new, g, ds_tr, ds_va, prior_shrinkage=1e-2)
            g.active_task_models[task_id] = m_new

        candidates.append(
            RevisionCandidate(
                candidate_id=f"cand_{self._step_counter}_local_refit",
                action_type="LocalRefit",
                description="Refit only task-local leaf parameters without updating shared upstream representation.",
                estimated_cost=1.4,
                val_log_predictive_gain=float(sc_refit["val_mean_log_predictive"] - base_val_lp),
                collateral_min_delta=0.0,
                unrelated_max_abs_drift=0.0,
                variational_free_energy_delta=float(base_fe - sc_refit["variational_free_energy"]),
                apply_fn=_apply_local_refit,
            )
        )

        # 2. `AncestorReweight` (jointly estimate minimal affected shared upstream shift across related tasks!)
        shared_id = incumbent.shared_upstream_id
        affected_tids = self.graph.minimal_affected_task_ids(shared_id)
        unrelated_tids = sorted(set(self.graph.active_task_models.keys()) - set(affected_tids))
        shared_mech_cur = self.graph.mechanisms[shared_id]
        shared_feats = tuple(shared_mech_cur.shared_weights.keys()) or incumbent.feature_names
        X_pool_list, y_pool_list = [], []
        for r_tid in affected_tids:
            if r_tid in self.graph.task_datasets:
                r_tr = self.graph.task_datasets[r_tid]["train"]
                X_pool_list.append(np.column_stack([np.asarray(r_tr["x"][fnm], dtype=float).ravel() for fnm in shared_feats]))
                y_pool_list.append(np.asarray(r_tr["y"], dtype=float).ravel())
        if X_pool_list:
            X_pool = np.vstack(X_pool_list)
            y_pool = np.concatenate(y_pool_list)
            w_hat, *_ = np.linalg.lstsq(X_pool, y_pool, rcond=None)
            new_shared_weights = {fnm: float(w_hat[idx]) for idx, fnm in enumerate(shared_feats)}

            sim_graph = deepcopy(self.graph)
            sim_graph.mechanisms[shared_id].shared_weights = new_shared_weights
            sim_graph.mechanisms[shared_id].version += 1

            sc_anc = fit_and_score_task_model(
                deepcopy(incumbent),
                sim_graph,
                ds_tr,
                ds_va,
                prior_shrinkage=cur_shrink,
            )
            rel_deltas: list[float] = []
            for r_tid in affected_tids:
                if r_tid == task_id:
                    continue
                r_entry = sim_graph.task_datasets[r_tid]
                r_sc = fit_and_score_task_model(
                    deepcopy(sim_graph.active_task_models[r_tid]),
                    sim_graph,
                    r_entry["train"],
                    r_entry["val"],
                    prior_shrinkage=float(r_entry.get("local_shrinkage", 12.0)),
                )
                rel_deltas.append(float(r_sc["val_mean_log_predictive"] - baseline_task_scores[r_tid]))

            unrel_drifts: list[float] = []
            for u_tid in unrelated_tids:
                u_entry = sim_graph.task_datasets[u_tid]
                u_sc = fit_and_score_task_model(
                    deepcopy(sim_graph.active_task_models[u_tid]),
                    sim_graph,
                    u_entry["train"],
                    u_entry["val"],
                    prior_shrinkage=float(u_entry.get("local_shrinkage", 12.0)),
                )
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
                        fit_and_score_task_model(
                            g.active_task_models[r_tid],
                            g,
                            re_e["train"],
                            re_e["val"],
                            prior_shrinkage=float(re_e.get("local_shrinkage", 12.0)),
                        )

            candidates.append(
                RevisionCandidate(
                    candidate_id=f"cand_{self._step_counter}_ancestor_reweight",
                    action_type="AncestorReweight",
                    description=f"Update shared upstream premise `{shared_id}` weights across {len(affected_tids)} related task(s).",
                    estimated_cost=1.5,
                    val_log_predictive_gain=float(sc_anc["val_mean_log_predictive"] - base_val_lp) + total_rel_gain,
                    collateral_min_delta=coll_min,
                    unrelated_max_abs_drift=max(unrel_drifts) if unrel_drifts else 0.0,
                    variational_free_energy_delta=float(base_fe - sc_anc["variational_free_energy"]),
                    apply_fn=_apply_anc,
                    metadata={"new_shared_weights": new_shared_weights, "affected_tasks": affected_tids, "unrelated_tasks": unrelated_tids},
                )
            )

        # 3. `LocalSplit` (search observable feature quantile thresholds for localized sub-regime exception!)
        best_split_cand: RevisionCandidate | None = None
        for fnm in incumbent.feature_names:
            col = np.asarray(ds_tr["x"][fnm], dtype=float).ravel()
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
                m_sp.model_id = f"{incumbent.model_id}_split_{fnm}"
                m_sp.regime_branches = [
                    *incumbent.regime_branches,
                    RegimeConduitBranch(branch_id=f"exc_{fnm}", mechanism_id=sp_mech_id, scope_guard=guard_spec, gate_sharpness=18.0),
                ]
                sc_sp = fit_and_score_task_model(m_sp, sim_g, ds_tr, ds_va)
                gain_sp = float(sc_sp["val_mean_log_predictive"] - base_val_lp)
                if best_split_cand is None or gain_sp > best_split_cand.val_log_predictive_gain:
                    def _apply_sp(g: PersistentMechanismGraph, mid_s=sp_mech_id, gspec=guard_spec, m_new=m_sp) -> None:
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
                        fit_and_score_task_model(m_new, g, ds_tr, ds_va)
                        g.active_task_models[task_id] = m_new

                    best_split_cand = RevisionCandidate(
                        candidate_id=f"cand_{self._step_counter}_local_split_{fnm}",
                        action_type="LocalSplit",
                        description=f"Bifurcate local exception branch on observable guard `{fnm} > {round(thr, 4)}`.",
                        estimated_cost=2.0,
                        val_log_predictive_gain=gain_sp,
                        collateral_min_delta=0.0,
                        unrelated_max_abs_drift=0.0,
                        variational_free_energy_delta=float(base_fe - sc_sp["variational_free_energy"]),
                        apply_fn=_apply_sp,
                        metadata={"scope_guard": guard_spec},
                    )
        if best_split_cand is not None:
            candidates.append(best_split_cand)

        # 4. `ConceptLift` (supports injected broken test `T11`, live `GPT CLI` `codex`, AND algebraic interaction basis!)
        lift_proposals: list[tuple[str, str, str, dict[str, Any]]] = []
        if injected_broken_expr is not None:
            lift_proposals.append(("broken_injected", injected_broken_expr, "injected_execution_fault_test", {}))

        if codex_client is not None and diag["status"] == "STRUCTURED_RESIDUAL_CANDIDATES":
            cli_res = propose_concept_lift_via_gpt_cli(
                task_id=task_id,
                feature_names=incumbent.feature_names,
                train_data=ds_tr,
                residuals=inc_eval["residuals"],
                diagnostics=diag,
                codex_client=codex_client,
                stage_subdir=f"m8_step_{self._step_counter:02d}_{task_id}",
            )
            if cli_res.get("ok") and isinstance(cli_res.get("parsed"), dict):
                p_obj = cli_res["parsed"]
                lift_proposals.append(
                    (
                        str(p_obj["concept_name"]),
                        str(p_obj["python_expression"]),
                        "gpt_cli_codex",
                        {
                            "llm_rationale": p_obj.get("mechanistic_rationale", ""),
                            "domain_guard_note": p_obj.get("domain_guard_note", ""),
                            "prompt_tokens": cli_res.get("prompt_tokens", 0),
                            "completion_tokens": cli_res.get("completion_tokens", 0),
                            "wall_ms": cli_res.get("wall_ms", 0.0),
                        },
                    )
                )

        # Deterministic algebraic pairwise/nonlinear basis fallback/competitor
        for i, f1 in enumerate(incumbent.feature_names):
            for f2 in incumbent.feature_names[i + 1 :]:
                lift_proposals.append((f"{f1}_x_{f2}", f"x['{f1}'] * x['{f2}']", "algebraic_grammar", {}))

        best_lift_cand: RevisionCandidate | None = None
        for cname, cexpr, c_src, c_meta in lift_proposals:
            try:
                fn_c = compile_safe_feature_conduit(cexpr, incumbent.feature_names)
                cond_info = estimate_conduit_conditioning_and_domain(fn_c, ds_tr["x"])
                if not cond_info["finite_on_support"]:
                    raise ValueError("Non-finite conduit output on training support")
            except Exception as exc:
                # `T11`: Execution/syntax failure MUST stay `UNRESOLVED_EXECUTION_FAILURE`, NEVER refuting mechanism!
                candidates.append(
                    RevisionCandidate(
                        candidate_id=f"cand_{self._step_counter}_lift_err_{cname[:16]}",
                        action_type="ConceptLift",
                        description=f"Failed execution of proposed conduit `{cexpr}` ({type(exc).__name__}).",
                        estimated_cost=1.0,
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
            m_lf.model_id = f"{incumbent.model_id}_lift_{cname[:16]}"
            m_lf.interaction_lift_ids = [*incumbent.interaction_lift_ids, lift_id]
            sc_lf = fit_and_score_task_model(m_lf, sim_g, ds_tr, ds_va)
            gain_lf = float(sc_lf["val_mean_log_predictive"] - base_val_lp)

            def _apply_lf(
                g: PersistentMechanismGraph,
                lid=lift_id,
                expr_s=cexpr,
                cinfo=cond_info,
                src_s=c_src,
                m_new=m_lf,
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
                fit_and_score_task_model(m_new, g, ds_tr, ds_va)
                g.active_task_models[task_id] = m_new

            cand_lf = RevisionCandidate(
                candidate_id=f"cand_{self._step_counter}_concept_lift_{cname[:18]}",
                action_type="ConceptLift",
                description=f"Lift shared interaction conduit `{cexpr}` (`source={c_src}`).",
                estimated_cost=2.4 if c_src == "gpt_cli_codex" else 2.5,
                val_log_predictive_gain=gain_lf,
                collateral_min_delta=0.0,
                unrelated_max_abs_drift=0.0,
                variational_free_energy_delta=float(base_fe - sc_lf["variational_free_energy"]),
                apply_fn=_apply_lf,
                metadata={"expression": cexpr, "source": c_src, "empirical_lipschitz": cond_info["empirical_lipschitz"], **c_meta},
            )
            if best_lift_cand is None or (
                cand_lf.val_log_predictive_gain - self.cost_weight * cand_lf.estimated_cost
                > best_lift_cand.val_log_predictive_gain - self.cost_weight * best_lift_cand.estimated_cost
            ):
                best_lift_cand = cand_lf
        if best_lift_cand is not None:
            candidates.append(best_lift_cand)

        # 5. `ChangeMemory` (activate `Pi_fold` causal exponential history kernel when trajectory/history signals exist!)
        if ("past_driver_history" in ds_tr and ds_tr["past_driver_history"] is not None) or ("timestamps" in ds_tr):
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
                m_mem.model_id = f"{incumbent.model_id}_mem_{tau_cand}"
                m_mem.temporal_memory_id = mem_id
                sc_mem = fit_and_score_task_model(m_mem, sim_g, ds_tr, ds_va)
                gain_mem = float(sc_mem["val_mean_log_predictive"] - base_val_lp)
                if best_mem_cand is None or gain_mem > best_mem_cand.val_log_predictive_gain:
                    def _apply_mem(g: PersistentMechanismGraph, mid_m=mem_id, tc=tau_cand, dc=drv_col, m_new=m_mem) -> None:
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
                        fit_and_score_task_model(m_new, g, ds_tr, ds_va)
                        g.active_task_models[task_id] = m_new

                    best_mem_cand = RevisionCandidate(
                        candidate_id=f"cand_{self._step_counter}_change_memory_{tau_cand}",
                        action_type="ChangeMemory",
                        description=f"Activate causal multi-scale exponential memory kernel (`tau={tau_cand}`).",
                        estimated_cost=2.2,
                        val_log_predictive_gain=gain_mem,
                        collateral_min_delta=0.0,
                        unrelated_max_abs_drift=0.0,
                        variational_free_energy_delta=float(base_fe - sc_mem["variational_free_energy"]),
                        apply_fn=_apply_mem,
                        metadata={"temporal_tau": tau_cand, "driver_col": drv_col},
                    )
            if best_mem_cand is not None:
                candidates.append(best_mem_cand)

        # 6. `AcquireObservation` when feasible probes are supplied and top structural candidates are tied/ambiguous
        valid_exec_cands = [
            c for c in candidates
            if c.execution_status == "EXECUTED_OK"
            and c.collateral_min_delta >= self.collateral_tolerance_nats
            and c.unrelated_max_abs_drift <= 1e-9
        ]
        ranked_valid = sorted(
            valid_exec_cands,
            key=lambda c: c.val_log_predictive_gain - self.cost_weight * c.estimated_cost,
            reverse=True,
        )

        chosen = ranked_valid[0]
        if chosen.action_type != "NoChange" and (
            chosen.val_log_predictive_gain < self.min_gain_nats
            or chosen.variational_free_energy_delta <= 0.0
        ):
            chosen = next(c for c in valid_exec_cands if c.action_type == "NoChange")

        if (
            chosen.action_type == "NoChange"
            and candidate_probes
            and len(ranked_valid) >= 2
        ):
            # Active probe selection when current observational data leaves competing mechanisms underdetermined
            pred_fns: list[tuple[str, Callable[[dict[str, np.ndarray]], np.ndarray]]] = []
            for c in ranked_valid[:3]:
                if c.action_type == "ConceptLift" and "expression" in c.metadata:
                    pred_fns.append((c.candidate_id, compile_safe_feature_conduit(c.metadata["expression"], incumbent.feature_names)))
            if len(pred_fns) >= 2:
                probe_sel = select_discriminating_observation_probe(pred_fns, candidate_probes)
                if probe_sel["max_utility"] > 0.05:
                    chosen = RevisionCandidate(
                        candidate_id=f"cand_{self._step_counter}_acquire_obs",
                        action_type="AcquireObservation",
                        description="Purchase maximally discriminating active intervention probe before structural commit.",
                        estimated_cost=0.5,
                        val_log_predictive_gain=0.0,
                        metadata=probe_sel,
                    )
                    candidates.append(chosen)

        if chosen.apply_fn is not None and chosen.execution_status == "EXECUTED_OK":
            chosen.apply_fn(self.graph)

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
            actual_cost=chosen.estimated_cost,
            public_val_log_predictive_delta=round(chosen.val_log_predictive_gain, 6),
            collateral_min_delta=round(chosen.collateral_min_delta, 6),
            rationale=chosen.description,
        )
        rec.validate_prefix_clean()
        self.graph.decision_ledger.append(rec)

        return {
            "decision_record": rec.to_dict(),
            "selected_candidate": chosen.public_summary(),
            "evaluated_candidates": [c.public_summary() for c in candidates],
            "wall_ms": wall_ms,
        }
