"""Identify a two-branch Prony law with thresholded viscoplastic flow."""

def __hegel_load_skill_autogenous_hereditary_spectrum_identification():
    import types as __hegel_types
    import numpy as np
    import numpy as np

    def align_reference_panel(raw_features: 'np.ndarray', reference_values: 'np.ndarray', groups: 'np.ndarray', strata: 'np.ndarray') -> 'tuple[np.ndarray, np.ndarray]':
        """Align a common panel under groupwise scalar affine shifts.

    Reference columns have known population mean zero and variance one
    before the shift. Use medians of column means and population standard
    deviations, retaining columns with at least two finite observations.
    Return aligned features and mean-one specimen weights with equal total
    weight in each observed group–stratum cell. Impute missing features by
    the balanced mean, or zero for a wholly unobserved feature. No penalty
    is used for calibration; a group with no usable positive gain is invalid.
    The panel may include unlabeled prediction rows; it uses no outcomes.
    """
        x = np.array(raw_features, dtype=float, copy=True)
        ref = np.asarray(reference_values, dtype=float)
        (group, stratum) = (np.asarray(groups), np.asarray(strata))
        if x.ndim != 2 or ref.ndim != 2 or len(x) == 0 or (len(ref) != len(x)) or (group.shape != (len(x),)) or (stratum.shape != group.shape):
            raise ValueError('panel arrays must share a nonempty specimen axis')
        (_, cell, counts) = np.unique(np.column_stack((group, stratum)), axis=0, return_inverse=True, return_counts=True)
        w = len(x) / (len(counts) * counts[cell.ravel()])
        for label in np.unique(group):
            mask = group == label
            panel = ref[mask]
            finite = np.isfinite(panel)
            keep = finite.sum(axis=0) >= 2
            if not np.any(keep):
                raise ValueError('each group needs a usable reference column')
            panel = np.where(finite[:, keep], panel[:, keep], np.nan)
            offset = np.median(np.nanmean(panel, axis=0))
            gain = np.median(np.nanstd(panel, axis=0, ddof=0))
            if not np.isfinite(gain) or gain <= 0:
                raise ValueError('reference gain must be finite and positive')
            x[mask] = (x[mask] - offset) / gain
        valid = np.isfinite(x)
        mass = (w[:, None] * valid).sum(axis=0)
        total = (w[:, None] * np.where(valid, x, 0.0)).sum(axis=0)
        fill = np.divide(total, mass, out=np.zeros_like(mass), where=mass > 0)
        return (np.where(valid, x, fill), w)

    def hereditary_design(past_stress: 'np.ndarray', past_dt: 'np.ndarray', hold_stress: 'np.ndarray', eval_times: 'np.ndarray', features: 'np.ndarray', taus: 'np.ndarray') -> 'tuple[np.ndarray, np.ndarray]':
        """Return strain-increment and initial-rate design arrays.

    History stress is piecewise constant in chronological order, beginning
    with zero memory. Hold stress is constant and times are measured from
    hold onset. Use common time units for steps, hold times and relaxation
    times. Coefficient order is fast and slow compliance, then drift
    intercept and drift feature slopes in strain per input time unit.
    Add the initial strain separately. The exact operator uses no penalty.
    """
        (h, dt, hold, t, x, tau) = (np.asarray(a, dtype=float) for a in (past_stress, past_dt, hold_stress, eval_times, features, taus))
        if h.ndim != 2 or dt.shape != (h.shape[1],) or (not dt.size) or (hold.shape != (len(h),)) or (t.ndim != 1) or (not t.size) or (x.ndim != 2) or (len(x) != len(h)) or (tau.shape != (2,)):
            raise ValueError('incompatible history, hold, feature or time shapes')
        if not all((np.isfinite(a).all() for a in (h, dt, hold, t, x, tau))) or np.any(dt <= 0) or np.any(t < 0) or (not 0 < tau[0] < tau[1]):
            raise ValueError('use finite arrays, positive steps and ordered taus')
        unit = np.median(dt)
        (dt, t, tau) = (dt / unit, t / unit, tau / unit)
        tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
        memory = -np.expm1(-dt[:, None] / tau) * np.exp(-tail[:, None] / tau)
        gap = hold[:, None] - h @ memory
        creep = -np.expm1(-t[:, None] / tau)
        drift = np.column_stack((np.ones(len(x)), x))
        curve = np.concatenate((gap[:, None, :] * creep[None, :, :], drift[:, None, :] * (t * unit)[None, :, None]), axis=2)
        rate = np.column_stack((gap / tau / unit, drift))
        return (curve, rate)
    return __hegel_types.SimpleNamespace(align_reference_panel=align_reference_panel, hereditary_design=hereditary_design)
__hegel_skill_ns_autogenous_hereditary_spectrum_identification = __hegel_load_skill_autogenous_hereditary_spectrum_identification()
import numpy as np
from scipy.optimize import least_squares
align_reference_panel = __hegel_skill_ns_autogenous_hereditary_spectrum_identification.align_reference_panel
hereditary_design = __hegel_skill_ns_autogenous_hereditary_spectrum_identification.hereditary_design

def _history(split, dt, yield_stress):
    past = np.asarray(split['past_stress'])
    hold = np.asarray(split['stress_hold_0'])
    over = np.sign(hold) * np.maximum(np.abs(hold) - yield_stress, 0.0) ** 2
    work = np.maximum(np.abs(past) - yield_stress, 0.0) ** 2 @ dt
    return (past, hold, over, work * hold)

def _design(history, dt, time, x, taus):
    (past, hold, over, work_stress) = history
    (curve, rate) = hereditary_design(past, dt, hold, time, x, taus)
    rise = -np.expm1(-time / taus[0])
    vp_curve = np.stack((over[:, None] * time, work_stress[:, None] * rise), axis=-1)
    vp_rate = np.column_stack((over, work_stress / taus[0]))
    return (np.concatenate((curve, vp_curve), axis=-1), np.column_stack((rate, vp_rate)))

def solve_viscoplastic_hereditary_revision(dataset: dict) -> dict:
    (train, test) = (dataset['train'], dataset['test'])
    dt = np.asarray(dataset['past_dt'])
    time = np.asarray(dataset['eval_times'])
    threshold = float(dataset['yield_stress'])
    n = len(train['lab_id'])
    panel = [np.concatenate((train[key], test[key]), axis=0) for key in ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum')]
    (x, _) = align_reference_panel(*panel)
    train_history = _history(train, dt, threshold)
    test_history = _history(test, dt, threshold)
    target = np.r_[(train['strain_curves_obs'] - train['strain_0'][:, None]).ravel(), train['instant_rate_obs']]
    scale = np.r_[np.full(n * len(time), 0.03), np.full(n, 0.045)]
    target = target / scale
    (log_low, log_high) = np.log([0.2, 8.0])

    def decode(p):
        fast = p[0]
        return np.exp([fast, fast + p[1] * (log_high - fast)])

    def matrix(p):
        (curve, rate) = _design(train_history, dt, time, x[:n], decode(p))
        return np.vstack((curve.reshape(n * len(time), -1), rate)) / scale[:, None]

    def residual(p):
        return np.einsum('ij,j->i', matrix(p[:2]), p[2:]) - target
    starts = []
    for fast in np.geomspace(0.24, 1.7, 4):
        for slow in np.geomspace(0.95, 7.8, 5):
            if slow <= fast:
                continue
            p = np.array([np.log(fast), np.log(slow / fast) / np.log(8 / fast)])
            a = matrix(p)
            coef = np.linalg.lstsq(a, target, rcond=None)[0]
            for temperature in (5, 3, 2, 1, 1):
                r = np.einsum('ij,j->i', a, coef) - target
                root = 1 / np.hypot(1, r / temperature)
                coef = np.linalg.lstsq(a * root[:, None], target * root, rcond=None)[0]
            r = np.einsum('ij,j->i', a, coef) - target
            starts.append((np.log1p(r * r).sum(), np.r_[p, coef]))
    bounds = (np.r_[log_low, 1e-08, np.full(coef.size, -np.inf)], np.r_[log_high - 1e-08, 1.0, np.full(coef.size, np.inf)])
    best = min((least_squares(residual, p, bounds=bounds, loss='cauchy', x_scale='jac', max_nfev=120) for (_, p) in sorted(starts, key=lambda row: row[0])[:2]), key=lambda fit: fit.cost)
    taus = decode(best.x)
    coef = best.x[2:]
    curve, rate = _design(test_history, dt, time, x[n:], taus)
    prediction = test['strain_0'][:, None] + curve @ coef
    # Imputation affects only steady drift. Recover it from the measured onset
    # rate on incomplete coupons after subtracting the modeled contributions.
    missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
    drift_residual = test['instant_rate_obs'][missing] - rate[missing] @ coef
    prediction[missing] += drift_residual[:, None] * time
    return {
        'estimated_taus': taus,
        'viscoplastic_weights': coef[-2:],
        'predicted_strain': prediction,
    }
