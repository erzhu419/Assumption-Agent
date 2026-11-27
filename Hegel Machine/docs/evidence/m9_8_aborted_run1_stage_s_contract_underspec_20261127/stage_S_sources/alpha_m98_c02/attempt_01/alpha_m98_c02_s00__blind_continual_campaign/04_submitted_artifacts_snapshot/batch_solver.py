"""Calibrated two-branch creep model for a characterization campaign."""

import numpy as np
from scipy.optimize import least_squares


def _calibrate(calibration):
    raw = np.asarray(calibration['bridge_raw'], dtype=float)
    true = np.asarray(calibration['true_channel_loads'], dtype=float)
    return np.linalg.lstsq(np.c_[np.ones(len(raw)), raw], true, rcond=None)[0]


def _axial(raw, bridge):
    return (np.asarray(raw) @ bridge[1:] + bridge[0]).sum(axis=-1)


def _prepare(group, bridge, *, reference=None):
    labs = np.asarray(group['lab_id'])
    strata = np.asarray(group['stratum'])
    x = np.array(group['coupon_features_raw'], dtype=float, copy=True)
    _, cells, counts = np.unique(np.column_stack((labs, strata)), axis=0,
                                 return_inverse=True, return_counts=True)
    weights = 1.0 / counts[cells]
    weights /= weights.mean()
    if reference is None:
        ref = np.asarray(group['ref_coupons'], dtype=float)
        reference = {}
        for lab in np.unique(labs):
            r = ref[labs == lab]
            loc = np.median(r.mean(axis=0))
            sd = np.median(r.std(axis=0))
            # A one-coupon lab has no within-column scale measurement.
            if sd <= 0:
                sd = np.median(np.abs(r - np.median(r))) / 0.67449
            reference[lab] = loc, sd if sd > 0 else 1.0
    for lab, (loc, sd) in reference.items():
        use = labs == lab
        x[use] = (x[use] - loc) / sd
    observed = np.isfinite(x)
    mass = observed.T @ weights
    fill = np.divide(np.where(observed, x, 0).T @ weights, mass,
                     out=np.zeros(x.shape[1]), where=mass > 0)
    x = np.where(observed, x, fill)
    start = np.asarray(group['strain_0'], dtype=float)
    y = None
    if 'strain_curves_obs' in group:
        y = np.column_stack((group['instant_rate_obs'],
                             np.asarray(group['strain_curves_obs']) - start[:, None]))
    return (_axial(group['past_channel_raw'], bridge),
            _axial(group['hold_channel_raw'], bridge),
            np.column_stack((np.ones(len(x)), x)), start, y, weights)


def _design(group, taus, duration, times):
    past, hold, drift = group[:3]
    tail = np.cumsum(duration[::-1])[::-1] - duration
    kernel = np.exp(-tail[:, None] / taus) * -np.expm1(-duration[:, None] / taus)
    amplitude = hold[:, None] - past @ kernel
    # Row zero is the onset derivative, not an observation of strain at t=0.
    relaxation = np.vstack((np.ones(2), -taus * np.expm1(-times[:, None] / taus)))
    return np.concatenate((amplitude[:, None, :] * relaxation[None, :, :],
                           drift[:, None, :] * np.r_[1., times][None, :, None]), axis=-1)


def _fit(parts, duration, times, extra_starts=()):
    first, _, _ = parts[0]
    lo, hi = np.log([0.0500001, 24.99999])
    prepared = [(a, a[4], np.sqrt(a[5] * weight)[:, None] / np.asarray(scale))
                for a, scale, weight in parts]

    def unpack(z):
        order = np.argsort(z[:2])
        return np.exp(z[:2][order]), np.r_[z[2:4][order], z[4:]]

    def residual(z):
        tau, theta = unpack(z)
        return np.concatenate([((_design(a, tau, duration, times) @ theta - y) * w).ravel()
                               for a, y, w in prepared])

    candidates = []
    for guess in (*extra_starts, (.25, 1.4), (.65, 4.), (1.2, 12.), (.09, 8.)):
        design = _design(first, np.asarray(guess), duration, times)
        sw = prepared[0][2]
        theta = np.linalg.lstsq((design * sw[:, :, None]).reshape(-1, 7),
                                (first[4] * sw).ravel(), rcond=None)[0]
        result = least_squares(residual, np.r_[np.log(guess), theta],
                               bounds=(np.r_[[lo] * 2, [-np.inf] * 7],
                                       np.r_[[hi] * 2, [np.inf] * 7]),
                               loss='cauchy', max_nfev=150)
        tau, _ = unpack(result.x)
        if tau[0] < tau[1]:
            candidates.append(result)
    if not candidates:
        raise ValueError('Could not identify two ordered relaxation times')
    return unpack(min(candidates, key=lambda result: result.cost).x)


def _estimate(train, pilot, duration, times):
    bench_scale = np.r_[.012, np.full(len(times), .027)]
    bench = _fit([(train, bench_scale, 1.)], duration, times)
    if pilot is None:
        return bench
    pilot_scale = np.r_[.0025, np.full(len(times), .005)]
    chamber = _fit([(pilot, pilot_scale, 1.)], duration, times, (bench[0],))
    # Borrow benchtop information only to the extent that it agrees with the
    # chamber response. Compare both models on the same chamber design.
    difference = _design(pilot, bench[0], duration, times) @ bench[1]
    difference -= _design(pilot, chamber[0], duration, times) @ chamber[1]
    discrepancy = np.median(np.abs(difference) / bench_scale, axis=0).max()
    bench_weight = .2 / max(1., discrepancy)**4
    if bench_weight < .002:
        return chamber
    return _fit([(pilot, pilot_scale, 1.), (train, bench_scale, bench_weight)],
                duration, times, (chamber[0], bench[0]))


def solve_batch(dataset: dict) -> dict:
    """Estimate Prony parameters and predict strain and onset rate in the chamber."""
    bridge = _calibrate(dataset['calibration_load_cells'])
    duration = np.asarray(dataset['past_dt'], dtype=float)
    times = np.asarray(dataset['eval_times'], dtype=float)
    train = _prepare(dataset['train'], bridge)
    trials = dataset.get('chamber_verification_trials')
    pilot = _prepare(trials, bridge) if trials and len(trials['lab_id']) else None
    tau, theta = _estimate(train, pilot, duration, times)
    test = _prepare(dataset['test'], bridge)
    response = _design(test, tau, duration, times) @ theta
    return {'estimated_taus': tau, 'hereditary_weights': theta,
            'predicted_strain': test[3][:, None] + response[:, 1:],
            'initial_rate_pred': response[:, 0]}
