"""Recover two variable-step Prony modes and the coupon-dependent creep drift."""

import numpy as np
from scipy.optimize import minimize


def _basis(split, dt, times, taus):
    remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    w = -np.expm1(-dt[:, None] / taus) * np.exp(-remaining[:, None] / taus)
    gap = split['stress_hold_0'][:, None] - split['past_stress'] @ w
    return gap[:, None, :] * -np.expm1(-times[:, None] / taus), gap / taus


def _calibration(train, test, lab):
    panel = np.concatenate([s['ref_coupons'][s['lab_id'] == lab] for s in (train, test)])
    return np.nanmedian(np.nanmean(panel, axis=0)), np.nanmedian(np.nanstd(panel, axis=0))


def _coupon_design(train, test):
    labs = np.unique(train['lab_id'])
    tr = np.zeros((len(train['lab_id']), 5 * len(labs)))
    te = np.zeros((len(test['lab_id']), 5 * len(labs)))
    transfer = np.zeros((len(test['lab_id']), 5))
    for lab in np.union1d(labs, test['lab_id']):
        offset, gain = _calibration(train, test, lab)
        known = np.flatnonzero(labs == lab)
        for split, design in ((train, tr), (test, te)):
            raw = split['coupon_features_raw']
            rows = (split['lab_id'] == lab) & np.isfinite(raw).all(axis=1)
            aligned = np.c_[np.ones(rows.sum()), (raw[rows] - offset) / gain]
            if known.size:
                j = known[0]
                design[rows, 5*j:5*j+5] = aligned
            elif split is test:
                transfer[rows] = aligned
    return tr, te, transfer


def _fit(train, dt, t, drift):
    y = train['strain_curves_obs'] - train['strain_0'][:, None]
    r = train['instant_rate_obs']
    complete = drift.any(axis=1)
    lo, hi = np.log([.2, 8.])
    rate_weight = 4.

    def profile(p, w):
        curve, rate = _basis(train, dt, t, np.exp(p))
        # Profile one drift per incomplete specimen, while complete specimens
        # share their laboratory's affine coupon-to-drift relation.
        mass = w @ (t*t) + rate_weight
        dx = np.einsum('nk,nkb,k->nb', w, curve, t) + rate_weight * rate
        dy = (w*y) @ t + rate_weight * r
        c = complete
        m = ~c
        a = (np.einsum('nk,nki,nkj->ij', w, curve, curve) + rate_weight * rate.T @ rate
             - dx[m].T @ (dx[m] / mass[m, None]))
        b = np.einsum('nk,nkb->b', w*y, curve) + rate_weight * rate.T @ r - dx[m].T @ (dy[m] / mass[m])
        cross = dx[c].T @ drift[c]
        lower = drift[c].T @ (mass[c, None] * drift[c])
        coef = np.linalg.lstsq(np.block([[a, cross], [cross.T, lower]]),
                               np.r_[b, drift[c].T @ dy[c]], rcond=1e-12)[0]
        tangent = drift @ coef[2:]
        tangent[m] = (dy[m] - dx[m] @ coef[:2]) / mass[m]
        e = curve @ coef[:2] + tangent[:, None] * t - y
        er = rate @ coef[:2] + tangent - r
        return np.sum(w * e**2) + rate_weight * (er @ er), coef, e

    w = np.ones_like(y)
    grid = np.linspace(lo, hi, 9)
    p = min(((a, b) for i, a in enumerate(grid) for b in grid[i+1:]), key=lambda z: profile(z, w)[0])
    for scale in (.35, .15, .07, .035, .025, .025, .025, None):
        result = minimize(lambda z: profile(z, w)[0], p, method='L-BFGS-B',
                          bounds=[(lo, hi)]*2, options={'ftol': 1e-12, 'gtol': 1e-8})
        p = np.sort(result.x)
        _, coef, e = profile(p, w)
        if scale is not None:
            w = 1 / (1 + (e/scale)**2)
    return np.exp(p), coef


def _update_partial_rates(train, test, drift, coef, base_rate, noise_var, rate):
    raw = test['coupon_features_raw']
    observed = np.isfinite(raw)
    for j, lab in enumerate(np.unique(train['lab_id'])):
        donors = drift[:, 5*j] != 0
        rows = np.flatnonzero((test['lab_id'] == lab) & ~observed.all(axis=1))
        if not len(rows) or donors.sum() < 12:
            continue
        offset, gain = _calibration(train, test, lab)
        intercept = coef[2+5*j]
        slope = coef[3+5*j:7+5*j]
        for stratum in np.unique(test['stratum'][rows]):
            local = donors & (train['stratum'] == stratum)
            ref = drift[local if local.sum() >= 12 else donors, 5*j+1:5*j+5]
            mu = ref.mean(axis=0)
            cov = np.cov(ref, rowvar=False)
            cov = .75 * cov + .25 * np.diag(np.diag(cov))
            for i in rows[test['stratum'][rows] == stratum]:
                keep = observed[i]
                miss = ~keep
                cross = cov[np.ix_(miss, keep)]
                conditional = np.linalg.solve(cov[np.ix_(keep, keep)], cross.T).T
                x = mu.copy()
                x[keep] = (raw[i, keep] - offset) / gain
                x[miss] += conditional @ (x[keep] - mu[keep])
                var = slope[miss] @ (cov[np.ix_(miss, miss)] - conditional @ cross.T) @ slope[miss]
                weight = noise_var / (noise_var + var)
                rate[i] += weight * (base_rate[i] + intercept + x @ slope - rate[i])


def solve_hereditary_prony_recovery(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    drift, drift_test, transfer = _coupon_design(train, test)
    taus, coef = _fit(train, dt, t, drift)
    curve, rate_basis = _basis(test, dt, t, taus)
    rate = test['instant_rate_obs'].copy()
    matched = drift_test.any(axis=1)
    model = rate_basis @ coef[:2] + drift_test @ coef[2:]
    rate[matched] = .95 * model[matched] + .05 * rate[matched]
    counts = (drift[:, ::5] != 0).sum(axis=0)
    shared = np.average(coef[2:].reshape(-1, 5), axis=0, weights=counts)
    # In an unseen lab, trust the onset measurement more than transferred calibration.
    unseen = transfer[:, 0] != 0
    rate[unseen] += .35 * ((rate_basis[unseen] @ coef[:2] + transfer[unseen] @ shared) - rate[unseen])
    # Missing coupon coordinates contribute uncertainty, so the onset reading
    # remains the primary anchor for an incomplete specimen.
    _, train_rate = _basis(train, dt, t, taus)
    complete = drift.any(axis=1)
    e = (train['instant_rate_obs'] - train_rate @ coef[:2] - drift @ coef[2:])[complete]
    noise_var = (1.4826 * np.median(np.abs(e - np.median(e))))**2
    _update_partial_rates(train, test, drift, coef, rate_basis @ coef[:2], noise_var, rate)
    # Integrate forward from the same onset rate returned to the caller.
    return {'estimated_taus': taus, 'hereditary_weights': np.r_[coef[:2], shared],
            'predicted_strain': test['strain_0'][:, None] + curve @ coef[:2] +
                                (rate - rate_basis @ coef[:2])[:, None] * t,
            'initial_rate_pred': rate}
