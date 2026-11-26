"""Hereditary rheometry and control-referenced cohort calibration."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _stress_design(trace, taus):
    past_t = np.asarray(trace['past_timestamps'], dtype=float)
    past_u = np.asarray(trace['past_strain_history'], dtype=float)
    u = np.asarray(trace['strain_u'], dtype=float)
    t = np.r_[past_t, trace['timestamps']]
    history = np.r_[past_u, u]
    dt_tau = np.diff(t)[:, None] / taus
    decay = np.exp(-dt_tau)
    update = -np.expm1(-dt_tau)
    state = np.full(taus.size, history[0])
    memory = np.empty((u.size, taus.size))
    for j, value in enumerate(history[1:], start=1):
        state = decay[j - 1] * state + update[j - 1] * value
        if j >= past_t.size:
            memory[j - past_t.size] = state
    return np.column_stack((np.ones(u.size), u, memory))


def _robust_fit(design, stress, subsets):
    # Seed the fit from small subsets so large glitches cannot set its scale.
    n = stress.size
    keep = n - n // 5
    best_loss = np.inf
    for rows in subsets:
        coef = np.linalg.lstsq(design[rows], stress[rows], rcond=None)[0]
        squared = (stress - design @ coef)**2
        loss = np.partition(squared, keep - 1)[:keep].sum()
        if loss < best_loss:
            best_loss, best_coef = loss, coef
    for _ in range(4):
        rows = np.argpartition(np.abs(stress - design @ best_coef), keep - 1)[:keep]
        best_coef = np.linalg.lstsq(design[rows], stress[rows], rcond=None)[0]
    resid = stress - design @ best_coef
    center = np.median(resid)
    scale = max(1.4826 * np.median(np.abs(resid - center)), 1e-10)
    for _ in range(8):
        w = 1 / np.hypot(1, (stress - design @ best_coef) / (2.5 * scale))
        best_coef = np.linalg.lstsq(design * w[:, None], stress * w, rcond=None)[0]
    return best_coef


def _extract(block, taus):
    rng = np.random.default_rng(391)
    n = len(block['traces'][0]['timestamps'])
    subsets = np.asarray([rng.choice(n, 9, replace=False) for _ in range(32)])
    moduli = np.array([_robust_fit(_stress_design(tr, taus), np.asarray(tr['stress_obs']), subsets)[1:]
                      for tr in block['traces']])
    moduli[np.asarray(block['missing_channel_mask'], dtype=bool)] = np.nan
    return moduli


def _align(values, controls, centers):
    aligned = np.empty_like(values)
    for c in np.unique(centers):
        rows = centers == c
        panel = controls[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0))
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError(f'Center {c} lacks a usable reference-panel scale')
        aligned[rows] = (values[rows] - offset) / gain
    return aligned


def _cell_weights(center, strata):
    _, inv, counts = np.unique(np.column_stack((center, strata)), axis=0,
                               return_inverse=True, return_counts=True)
    return center.size / (counts.size * counts[inv.reshape(-1)])


def _resolve_missing(train, query, weights):
    available = np.isfinite(train)
    support = weights @ available
    fill = np.divide(weights @ np.where(available, train, 0.), support,
                     out=np.zeros(train.shape[1]), where=support > 0)
    return np.where(available, train, fill), np.where(np.isfinite(query), query, fill)


def _features(z, strata):
    # One-hot stratum columns supply the intercepts, without imposing ordinality.
    return np.column_stack((z, strata[:, None] == np.arange(3)))


def _logistic(x, y, weights, ridge=1 / 3):
    penalty = np.r_[np.full(3, ridge), np.zeros(x.shape[1] - 3)]

    def objective(b):
        eta = x @ b
        loss = weights @ np.logaddexp(0, (1 - 2*y) * eta) + 0.5 * (penalty * b) @ b
        grad = x.T @ (weights * (expit(eta) - y)) + penalty * b
        return loss, grad

    return minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B',
                    options={'gtol': 1e-8, 'ftol': 1e-12}).x


def _risk(design, coef, missing, train_aligned):
    # Mean imputation supplies the logit's location, not certainty about it.
    # Approximate the logistic-normal mean under independent missing moduli.
    observed = np.isfinite(train_aligned).any(axis=0)
    spread = np.zeros(train_aligned.shape[1])
    spread[observed] = np.nanstd(train_aligned[:, observed], axis=0)
    logit_variance = missing @ (coef[:3] * spread)**2
    score = (design @ coef) / np.sqrt(1 + np.pi / 8 * logit_variance)
    return np.clip(expit(score), 1e-15, 1 - 1e-15)


def calibrate_viscoelastic_cohort(cohort: dict) -> dict:
    """Return imputed latent query moduli, failure probabilities, and slopes."""
    train, query = cohort['train'], cohort['query']
    taus = np.asarray(cohort['taus'], dtype=float)
    ct, cq = np.asarray(train['center_id']), np.asarray(query['center_id'])
    st, sq = np.asarray(train['stratum']), np.asarray(query['stratum'])
    zt = _align(_extract(train, taus), np.asarray(train['controls']), ct)
    zq = _align(_extract(query, taus), np.asarray(query['controls']), cq)
    weights = _cell_weights(ct, st)
    train_aligned = zt
    missing = ~np.isfinite(zq)
    zt, zq = _resolve_missing(zt, zq, weights)
    coef = _logistic(_features(zt, st), np.asarray(train['y'], dtype=float), weights)
    return {'aligned_query_moduli': zq,
            'query_probabilities': _risk(_features(zq, sq), coef, missing, train_aligned),
            'modulus_weights': coef[:3]}
