"""Two-branch hereditary response with signed overstress and work softening."""

import numpy as np
from scipy.optimize import minimize


def _aligned_coupons(train, test):
    """Calibrate shared lab transforms on the reference panel, then fill by cell balance."""
    lab = np.concatenate([train['lab_id'], test['lab_id']])
    stratum = np.concatenate([train['stratum'], test['stratum']])
    x = np.vstack([train['coupon_features_raw'], test['coupon_features_raw']])
    ref = np.vstack([train['ref_coupons'], test['ref_coupons']])
    for group in np.unique(lab):
        mask = lab == group
        panel = ref[mask]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0))
        x[mask] = (x[mask] - center) / scale
    n = len(train['lab_id'])
    _, cell, count = np.unique(np.column_stack((lab[:n], stratum[:n])), axis=0,
                               return_inverse=True, return_counts=True)
    weights = (1.0 / count[cell])[:, None]
    observed = np.isfinite(x[:n])
    means = np.sum(weights * np.where(observed, x[:n], 0), axis=0) / np.sum(weights * observed, axis=0)
    x = np.where(np.isfinite(x), x, means)
    return x[:n], x[n:]


def _history(split, dt, threshold):
    h = np.asarray(split['past_stress'], dtype=float)
    s = np.asarray(split['stress_hold_0'], dtype=float)
    ov = np.sign(s) * np.maximum(np.abs(s) - threshold, 0.0)**2
    work = (np.maximum(np.abs(h) - threshold, 0.0)**2 @ dt) * s
    return h, s, ov, work


def _branches(history, dt, t, tau):
    h, s, _, work = history
    decay = np.exp(-dt[:, None] / tau)
    arrival = -np.expm1(-dt[:, None] / tau)
    z = np.zeros((len(s), 2))
    for j in range(len(dt)):
        z = z * decay[j] + h[:, j, None] * arrival[j]
    a = np.column_stack((s[:, None] - z, work))
    phi = -np.expm1(-t[:, None] / tau)
    phi = np.vstack((phi, 1.0 / tau))
    return a[:, None, :] * phi[None, :, [0, 1, 0]]


def _project(A, y, b, w):
    """Weighted least squares with a separate linear drift for every sample."""
    wb = w * b
    norm = wb @ b
    a_slopes = np.einsum('nkp,nk->np', A, wb) / norm[:, None]
    y_slopes = np.sum(y * wb, axis=1) / norm
    Ap = A - b[None, :, None] * a_slopes[:, None, :]
    yp = y - b * y_slopes[:, None]
    gram = np.einsum('nki,nkj,nk->ij', Ap, Ap, w)
    rhs = np.einsum('nki,nk,nk->i', Ap, yp, w)
    coeff = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    drift = y_slopes - a_slopes @ coeff
    return coeff, drift, yp - Ap @ coeff


def _fit_memory(history, dt, t, train):
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs']))
    scale = np.r_[np.full(len(t), .026), .016]
    b = np.r_[t, 1.] / scale
    y = y / scale

    def profile(log_tau, full=False):
        A = _branches(history, dt, t, np.exp(log_tau)) / scale[None, :, None]
        w = np.ones_like(y)
        for _ in range(24):
            coeff, drift, resid = _project(A, y, b, w)
            new_w = 1. / (1. + resid**2)
            if np.max(np.abs(w - new_w)) < 1e-5:
                break
            w = new_w
        if full:
            return coeff, drift
        return np.mean(np.log1p(resid**2))

    # Enforce branch ordering while allowing the complete physical time range.
    bounds = [(np.log(.2), np.log(8.0))] * 2
    ordered = {'type': 'ineq', 'fun': lambda u: u[1] - u[0] - 1e-6}
    fits = [minimize(profile, np.log(start), method='SLSQP', bounds=bounds,
                     constraints=ordered, options={'ftol': 1e-10, 'maxiter': 70})
            for start in ((.45, 3.0), (1.0, 5.5))]
    log_tau = min(fits, key=lambda f: f.fun).x
    coeff, drift = profile(log_tau, full=True)
    return np.exp(log_tau), coeff, drift


def _fit_drift(x, ov, slopes):
    A = np.column_stack((np.ones(len(x)), x, ov))
    w = np.ones(len(x))
    for step in range(20):
        root = np.sqrt(w)
        coef = np.linalg.lstsq(A * root[:, None], slopes * root, rcond=None)[0]
        r = slopes - A @ coef
        if step == 0:
            scale = max(.014, 1.4826 * np.median(np.abs(r - np.median(r))))
        w = 1. / (1. + (r / (2 * scale))**2)
    return coef, scale


def solve_viscoplastic_hereditary_revision(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt = np.asarray(dataset['past_dt'], dtype=float)
    t = np.asarray(dataset['eval_times'], dtype=float)
    yld = float(dataset['yield_stress'])
    train_h, test_h = (_history(split, dt, yld) for split in (train, test))
    train_x, test_x = _aligned_coupons(train, test)
    tau, coeff, slopes = _fit_memory(train_h, dt, t, train)
    drift_coeff, scatter = _fit_drift(train_x, train_h[2], slopes)
    response = _branches(test_h, dt, t, tau) @ coeff
    coupon_drift = np.column_stack((np.ones(len(test_x)), test_x, test_h[2])) @ drift_coeff
    rate_drift = test['instant_rate_obs'] - response[:, -1]
    # The observed onset rate directly identifies drift after subtracting the
    # Prony and damage derivatives; coupons supply a noisy independent estimate.
    correction = coupon_drift - rate_drift
    fraction = .010**2 / (.010**2 + scatter**2)
    drift = rate_drift + fraction * np.clip(correction, -2.5*scatter, 2.5*scatter)
    return {'estimated_taus': tau,
            'viscoplastic_weights': np.array([drift_coeff[-1], coeff[-1]]),
            'predicted_strain': test['strain_0'][:, None] + response[:, :-1] + drift[:, None]*t}
