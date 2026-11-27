"""Estimate chamber strain and onset rate from calibrated load histories."""

import numpy as np
from scipy.optimize import least_squares


def _bridge(calibration):
    raw = np.asarray(calibration['bridge_raw'])
    loads = np.asarray(calibration['true_channel_loads'])
    return np.linalg.lstsq(np.c_[np.ones(len(raw)), raw], loads, rcond=None)[0]


def _prepare(cohort, bridge):
    lab = np.asarray(cohort['lab_id'])
    ref = np.asarray(cohort['ref_coupons'])
    x = np.array(cohort['coupon_features_raw'], dtype=float, copy=True)
    for label in np.unique(lab):
        rows = lab == label
        panel = ref[rows]
        center = np.median(panel.mean(axis=0))
        scale = np.median(panel.std(axis=0, ddof=0))
        x[rows] = (x[rows] - center) / (scale if scale > 0 else 1.)

    cells = np.stack([lab, cohort['stratum']], axis=1)
    _, index, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    w = 1. / counts[index, None]
    seen = np.isfinite(x)
    total = (w * seen).sum(axis=0)
    fill = np.divide((w * np.where(seen, x, 0.)).sum(axis=0), total,
                     out=np.zeros(x.shape[1]), where=total > 0)
    x = np.where(seen, x, fill)

    # Summing the calibrated channel outputs commutes with the affine map.
    slope = bridge[1:].sum(axis=1)
    offset = bridge[0].sum()
    return (cohort['past_channel_raw'] @ slope + offset,
            cohort['hold_channel_raw'] @ slope + offset,
            np.c_[np.ones(len(x)), x])


def _design(inputs, elapsed, times, taus):
    past, hold, drift = inputs
    # Each history interval contributes its exact exponentially decayed mass.
    history = np.diff(np.exp(-elapsed[:, None] / taus), axis=0)
    gap = hold[:, None] - past @ history
    memory = gap[:, None, :] * -np.expm1(-times[:, None] / taus)
    curve = np.concatenate((memory, times[None, :, None] * drift[:, None, :]), axis=2)
    rate = np.c_[gap / taus, drift]
    return np.concatenate((curve, rate[:, None, :]), axis=1)


def _fit(inputs, elapsed, times, cohort, pilot):
    y = np.c_[cohort['strain_curves_obs'] - cohort['strain_0'][:, None],
              cohort['instant_rate_obs']]
    scales = (np.r_[np.hypot(.004, .003 * times), .006] if pilot else
              np.r_[np.hypot(.010, .007 * times), .015])
    lower = np.r_[np.full(2, np.log(.05001)), np.full(7, -np.inf)]
    upper = np.r_[np.full(2, np.log(24.999)), np.full(7, np.inf)]

    def residual(z):
        design = _design(inputs, elapsed, times, np.exp(z[:2]))
        return ((design @ z[2:] - y) / scales).ravel()

    best = None
    for start in ([.25, 1.5], [.65, 4.], [1.4, 12.]):
        design = _design(inputs, elapsed, times, np.array(start)) / scales[None, :, None]
        weights = np.linalg.lstsq(design.reshape(-1, 7), (y / scales).ravel(), rcond=None)[0]
        fit = least_squares(residual, np.r_[np.log(start), weights], bounds=(lower, upper),
                            loss='soft_l1', max_nfev=180)
        if best is None or fit.cost < best.cost:
            best = fit
    z = best.x
    order = np.argsort(z[:2])
    taus = np.exp(z[:2][order])
    if taus[0] == taus[1]:
        taus[1] = np.nextafter(taus[1], np.inf)
    return taus, np.r_[z[2:4][order], z[4:]]


def solve_batch(dataset: dict) -> dict:
    """Return the fitted deployment spectrum and predictions for the test coupons."""
    bridge = _bridge(dataset['calibration_load_cells'])
    times = np.asarray(dataset['eval_times'])
    elapsed = np.r_[np.cumsum(np.asarray(dataset['past_dt'])[::-1])[::-1], 0.]
    pilot = dataset.get('chamber_verification_trials')
    fit_cohort = dataset['train'] if pilot is None else pilot
    taus, weights = _fit(_prepare(fit_cohort, bridge), elapsed, times, fit_cohort,
                         pilot is not None)
    test = dataset['test']
    response = _design(_prepare(test, bridge), elapsed, times, taus) @ weights
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'predicted_strain': response[:, :-1] + test['strain_0'][:, None],
            'initial_rate_pred': response[:, -1]}
