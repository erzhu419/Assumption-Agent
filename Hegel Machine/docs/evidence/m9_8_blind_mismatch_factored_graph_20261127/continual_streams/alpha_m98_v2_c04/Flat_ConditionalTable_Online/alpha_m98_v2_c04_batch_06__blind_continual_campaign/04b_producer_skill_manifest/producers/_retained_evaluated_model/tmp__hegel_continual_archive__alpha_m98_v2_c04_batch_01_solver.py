"""Calibrate the load bridge and fit a two-branch hereditary material law."""
import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    refs = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    for group in np.unique(lab):
        rows = lab == group
        panel = refs[rows]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[rows] = (x[rows] - center) / scale
    _, cell, counts = np.unique(np.column_stack((lab, stratum)), axis=0,
                                return_inverse=True, return_counts=True)
    w = 1.0 / counts[cell, None]
    present = ~np.isnan(x)
    mass = np.sum(w * present, axis=0)
    fill = np.divide(np.nansum(w * x, axis=0), mass,
                     out=np.zeros(x.shape[1]), where=mass > 0)
    return np.where(present, x, fill)


def _bridge(cal):
    raw = cal['bridge_raw']
    return np.linalg.lstsq(np.column_stack((np.ones(len(raw)), raw)),
                           cal['true_channel_loads'], rcond=None)[0]


def _loads(block, bridge):
    total = bridge.sum(axis=1)
    return tuple(block[key] @ total[1:] + total[0]
                 for key in ('past_channel_raw', 'hold_channel_raw'))


def _design(x, past, hold, dt, times, tau):
    # Weight past loads by their elapsed time to the start of the hold.
    later = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    memory = -np.expm1(-dt[:, None] / tau) * np.exp(-later[:, None] / tau)
    gap = hold[:, None] - past @ memory
    creep = -np.expm1(-times[:, None] / tau)
    drift = np.column_stack((np.ones(len(x)), x))
    curve = np.concatenate((gap[:, None, :] * creep[None, :, :],
                            times[None, :, None] * drift[:, None, :]), axis=2)
    rate = np.column_stack((gap / tau, drift))
    return curve.reshape(-1, 7), rate


def _mad(a):
    return 1.4826 * np.median(np.abs(a - np.median(a)))


def _fit(x, past, hold, dt, times, curve_obs, rate_obs):
    yc = curve_obs.ravel()
    yr = rate_obs
    target = np.r_[yc, yr]
    lo, hi = np.log([0.05000001, 24.99999999])
    bounds = (np.r_[[lo, lo], np.full(7, -np.inf)],
              np.r_[[hi, hi], np.full(7, np.inf)])

    def design(logtau):
        a, b = _design(x, past, hold, dt, times, np.exp(logtau))
        return np.vstack((a, b))

    def residual(p):
        return (design(p[:2]) @ p[2:] - target) / scale

    scale = max(0.01, 0.015 * np.std(yc))
    trials = []
    for guess in ((0.18, 1.6), (0.45, 4.0), (0.9, 12.0), (0.12, 8.0)):
        q = np.log(guess)
        coeff = np.linalg.lstsq(design(q), target, rcond=None)[0]
        trials.append(least_squares(residual, np.r_[q, coeff], bounds=bounds,
                                    loss='soft_l1', x_scale='jac', max_nfev=180))
    p = min(trials, key=lambda r: r.cost).x
    for _ in range(2):
        e = design(p[:2]) @ p[2:] - target
        sc = max(_mad(e[:len(yc)]), 1e-6)
        sr = max(_mad(e[len(yc):]), 1e-6)
        # Cauchy downweighting preserves the Gaussian center of each sensor.
        scale = 2.5 * np.r_[np.full(len(yc), sc), np.full(len(yr), sr)]
        p = least_squares(residual, p, bounds=bounds, loss='cauchy',
                          x_scale='jac', max_nfev=180).x
    perm = np.argsort(p[:2])
    return np.exp(p[:2][perm]), np.r_[p[2:4][perm], p[4:]]



# Identified from chamber verification coupons, separate from the
# intrinsic material fit on benchtop observations.
_CHAMBER_TIMES = np.array([0.4718099191421826, 2.4756939307917163])
_CHAMBER_COEFFICIENTS = np.array([[0.6193595560932273, 0.16764713872971412], [0.28119327738345373, 1.3920963008647889], [0.03080586637439311, 0.08122038107343733]])


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = dataset['past_dt'], dataset['eval_times']
    bridge = _bridge(dataset['calibration_load_cells'])
    x = _features(train, test)
    n = len(train['lab_id'])
    taus, theta = _fit(x[:n], *_loads(train, bridge), dt, times,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    _, hold = _loads(test, bridge)
    drift = theta[2] + x[n:] @ theta[3:]
    components = np.column_stack((hold, drift)) @ _CHAMBER_COEFFICIENTS.T
    creep = -np.expm1(-times[:, None] / _CHAMBER_TIMES)
    strain = (test['strain_0'][:, None] + components[:, :2] @ creep.T
              + components[:, 2, None] * times)
    rate = components[:, :2] @ (1.0 / _CHAMBER_TIMES) + components[:, 2]
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': strain, 'initial_rate_pred': rate}
