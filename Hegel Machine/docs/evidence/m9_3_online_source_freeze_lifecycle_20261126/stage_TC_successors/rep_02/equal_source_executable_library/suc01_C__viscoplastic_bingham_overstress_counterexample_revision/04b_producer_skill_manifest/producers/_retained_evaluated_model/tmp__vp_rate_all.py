"""Identify a two-branch Prony model with creep and plastic-work softening."""

import numpy as np
from scipy.optimize import least_squares


def _align(train, test):
    raw, refs, labs = (np.concatenate((train[k], test[k]), axis=0) for k in
                       ('coupon_features_raw', 'ref_coupons', 'lab_id'))
    x = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        rows = labs == lab
        panel = refs[rows]
        x[rows] = (raw[rows] - np.nanmedian(np.nanmean(panel, axis=0))) / np.nanmedian(np.nanstd(panel, axis=0))
    n = len(train['lab_id'])
    cells = np.column_stack((train['lab_id'], train['stratum']))
    _, ids, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    weights = n / (len(counts) * counts[ids])
    present = np.isfinite(x[:n])
    denominator = weights @ present
    fill = np.divide(weights @ np.where(present, x[:n], 0), denominator,
                     out=np.zeros(x.shape[1]), where=denominator > 0)
    missing = ~np.isfinite(x)
    return np.where(missing, fill, x), missing, weights


def _design_inputs(split, x, dt, yield_stress):
    s = np.asarray(split['past_stress'])
    h = np.asarray(split['stress_hold_0'])
    work = np.maximum(np.abs(s) - yield_stress, 0)**2 @ dt
    signed_overstress = np.maximum(np.abs(h) - yield_stress, 0)**2 * np.sign(h)
    drift = np.column_stack((np.ones(len(x)), x, signed_overstress))
    return s, h, drift, work * h


def _design(values, dt, age, times, taus):
    s, h, drift, surge = values
    kernels = -np.expm1(-dt[None, :] / taus[:, None]) * np.exp(-age[None, :] / taus[:, None])
    gap = h[:, None] - np.einsum('ij,kj->ik', s, kernels)
    ramps = -np.expm1(-times[:, None] / taus)
    n, k = len(h), len(times)
    strain = np.empty((n, k, 9))
    strain[..., :2] = gap[:, None, :] * ramps
    strain[..., 2:8] = drift[:, None, :] * times[None, :, None]
    strain[..., 8] = surge[:, None] * ramps[:, 0]
    rate = np.column_stack((gap / taus, drift, surge / taus[0]))
    return strain, rate


def _evaluate(design, coef):
    return np.einsum('...j,j->...', design, coef)


def _taus(q):
    return np.array([q[0], q[0] + (8.0 - q[0]) * q[1]])


def _fit(values, dt, age, times, strain_obs, rate_obs, weights):
    n, k = strain_obs.shape
    observed = np.column_stack((strain_obs, rate_obs))
    valid = np.isfinite(observed)
    root_weight = np.sqrt(weights[:, None] * np.ones((1, k + 1)))[valid]
    data = observed[valid]

    def matrix(q):
        curves, rates = _design(values, dt, age, times, _taus(q))
        return np.concatenate((curves, rates[:, None, :]), axis=1)[valid]

    def observation_scales(curve, rate):
        return np.broadcast_to(np.r_[np.full(k, curve), rate], observed.shape)[valid]

    scales = observation_scales(0.055, 0.05)
    lo = np.r_[0.20, 1e-5, np.full(9, -np.inf)]
    hi = np.r_[7.999, 1.0, np.full(9, np.inf)]

    def fit(initial, loss):
        if loss == 'cauchy':
            # Signed square root gives a weighted Cauchy objective without
            # changing the outlier cutoff for large or small cells.
            def residual(p):
                r = (_evaluate(matrix(p[:2]), p[2:]) - data) / scales
                return root_weight * np.sign(r) * np.sqrt(np.log1p(r * r))
        else:
            def residual(p):
                return root_weight * (_evaluate(matrix(p[:2]), p[2:]) - data) / scales
        return least_squares(residual, initial, bounds=(lo, hi), loss=loss if loss != 'cauchy' else 'linear',
                             tr_solver='lsmr', x_scale='jac', max_nfev=110)

    starts = []
    for fast, slow in ((0.30, 2.0), (0.65, 4.0), (1.35, 6.5)):
        q = np.array([fast, (slow - fast) / (8.0 - fast)])
        a = matrix(q)
        weight = root_weight / scales
        coef = np.linalg.lstsq(a * weight[:, None], data * weight, rcond=None)[0]
        starts.append(fit(np.r_[q, coef], 'soft_l1'))
    p = min(starts, key=lambda result: result.cost).x
    r = _evaluate(matrix(p[:2]), p[2:]) - data
    rate_mask = np.broadcast_to(np.arange(k + 1) == k, observed.shape)[valid]

    def mad(a, floor):
        return max(floor, 1.4826 * np.median(np.abs(a - np.median(a))))

    scales = observation_scales(mad(r[~rate_mask], 0.015), mad(r[rate_mask], 0.015))
    result = fit(p, 'cauchy')
    return _taus(result.x[:2]), result.x[2:]


def solve_viscoplastic_hereditary_revision(dataset: dict) -> dict:
    """Fit the constitutive law to observed training curves and initial rates."""
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    dt = np.asarray(dataset['past_dt'], dtype=float)
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    times = np.asarray(dataset['eval_times'], dtype=float)
    yield_stress = float(dataset['yield_stress'])
    x, missing, weights = _align(train, test)
    known = ~missing[:n].any(axis=1)
    training = _design_inputs(train, x[:n], dt, yield_stress)
    taus, coef = _fit(tuple(a[known] for a in training), dt, age, times,
                      (train['strain_curves_obs'] - train['strain_0'][:, None])[known],
                      train['instant_rate_obs'][known], weights[known])
    testing = _design_inputs(test, x[n:], dt, yield_stress)
    curves, rates = _design(testing, dt, age, times, taus)
    prediction = test['strain_0'][:, None] + _evaluate(curves, coef)

    # Missing coupon coordinates enter only the constant drift. The measured
    # initial rate identifies its net correction without imputing each feature.
    observed_rate = np.asarray(test['instant_rate_obs'])
    use_rate = np.isfinite(observed_rate)
    drift_correction = observed_rate[use_rate] - _evaluate(rates[use_rate], coef)
    prediction[use_rate] += drift_correction[:, None] * times
    return {'estimated_taus': taus, 'viscoplastic_weights': coef[[7, 8]],
            'predicted_strain': prediction}
