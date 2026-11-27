"""Estimate shared material relaxation and predict deployment-chamber creep."""

import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    x = np.vstack([train['coupon_features_raw'], test['coupon_features_raw']])
    refs = np.vstack([train['ref_coupons'], test['ref_coupons']])
    labs = np.concatenate([train['lab_id'], test['lab_id']])
    strata = np.concatenate([train['stratum'], test['stratum']])
    for lab in np.unique(labs):
        i = labs == lab
        ref = refs[i]
        center = np.median(ref.mean(axis=0))
        scale = np.median(ref.std(axis=0, ddof=0))
        x[i] = (x[i] - center) / scale
    _, cells, counts = np.unique(np.column_stack([labs, strata]), axis=0,
                                 return_inverse=True, return_counts=True)
    w = 1.0 / counts[cells.reshape(-1), None]
    observed = np.isfinite(x)
    means = (w * np.where(observed, x, 0)).sum(axis=0) / (w * observed).sum(axis=0)
    x = np.where(observed, x, means)
    return x[:len(train['lab_id'])], x[len(train['lab_id']):]


def _stress_map(cal):
    raw = cal['bridge_raw']
    return np.linalg.lstsq(np.column_stack([np.ones(len(raw)), raw]),
                           cal['true_channel_loads'], rcond=None)[0].sum(axis=1)


def _design(group, x, stress, dt, t, taus):
    past = stress[0] + group['past_channel_raw'] @ stress[1:]
    hold = stress[0] + group['hold_channel_raw'] @ stress[1:]
    age = np.r_[np.cumsum(dt[::-1])[::-1][1:], 0.0]
    history_weights = -np.expm1(-dt[:, None] / taus) * np.exp(-age[:, None] / taus)
    gap = hold[:, None] - past @ history_weights
    drift = np.column_stack([np.ones(len(x)), x])
    curves = np.concatenate([
        gap[:, None, :] * -np.expm1(-t[None, :, None] / taus),
        drift[:, None, :] * t[None, :, None]], axis=2)
    rates = np.column_stack([gap / taus, drift])
    return curves, rates


def _mad(residual):
    return 1.4826 * np.median(np.abs(residual - np.median(residual)))


def _fit(train, x, stress, dt, times):
    y_curve = train['strain_curves_obs'] - train['strain_0'][:, None]
    y_rate = train['instant_rate_obs']
    y = np.r_[y_curve.ravel(), y_rate]
    n = y_curve.size

    def design(taus):
        c, r = _design(train, x, stress, dt, times, taus)
        return np.vstack([c.reshape(-1, 7), r])

    def residual(p, scales):
        return (design(np.exp(p[:2])) @ p[2:] - y) / scales

    lower = np.r_[np.log([.05000001] * 2), np.full(7, -np.inf)]
    upper = np.r_[np.log([24.999999] * 2), np.full(7, np.inf)]
    starts = [(0.28, 2.2), (0.11, 1.1), (0.65, 5.6), (1.2, 13.)]
    candidates = []
    for taus in starts:
        theta = np.linalg.lstsq(design(np.array(taus)), y, rcond=None)[0]
        fit = least_squares(residual, np.r_[np.log(taus), theta], args=(1.,),
                            bounds=(lower, upper), loss='soft_l1', f_scale=.035,
                            x_scale='jac', max_nfev=160)
        candidates.append(fit)
    pilot = min(candidates, key=lambda fit: fit.cost).x
    err = residual(pilot, 1.)
    scales = np.r_[np.full(n, max(_mad(err[:n]), 1e-7)),
                   np.full(len(y_rate), max(_mad(err[n:]), 1e-7))]
    fit = least_squares(residual, pilot, args=(scales,), bounds=(lower, upper),
                        loss='cauchy', f_scale=2.5, x_scale='jac', max_nfev=200)
    taus, theta = np.exp(fit.x[:2]), fit.x[2:].copy()
    order = np.argsort(taus)
    theta[:2] = theta[:2][order]
    return taus[order], theta



def _chamber_features(group):
    """Use the chamber's own reference panels and ordinary mean imputation."""
    x = np.array(group['coupon_features_raw'], dtype=float, copy=True)
    refs = group['ref_coupons']
    labs = group['lab_id']
    for lab in np.unique(labs):
        mask = labs == lab
        panel = refs[mask]
        x[mask] = (x[mask] - np.median(panel.mean(axis=0))) / np.median(panel.std(axis=0))
    return np.where(np.isnan(x), np.nanmean(x, axis=0), x)


def _predict_chamber(group, stress, times, taus, theta):
    # Chamber equilibration removes pre-hold memory from the deployment law.
    hold = stress[0] + group['hold_channel_raw'] @ stress[1:]
    amplitude = hold[:, None] * theta[:2]
    drift = theta[2] + _chamber_features(group) @ theta[3:]
    relaxation = -np.expm1(-times[:, None] / taus)
    strain = group['strain_0'][:, None] + amplitude @ relaxation.T
    strain += drift[:, None] * np.log1p(times)
    rate = amplitude @ (1.0 / taus) + drift
    return strain, rate


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x_train, _ = _features(train, test)
    stress = _stress_map(dataset['calibration_load_cells'])
    times = np.asarray(dataset['eval_times'])
    taus, theta = _fit(train, x_train, stress, dataset['past_dt'], times)
    strain, rate = _predict_chamber(test, stress, times, taus, theta)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': strain, 'initial_rate_pred': rate}
