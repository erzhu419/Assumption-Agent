"""Calibrated piecewise-constant Prony response and chamber drift transfer."""
import numpy as np
from scipy.optimize import least_squares


def _reference(blocks):
    lab = np.concatenate([b['lab_id'] for b in blocks])
    refs = np.concatenate([b['ref_coupons'] for b in blocks])
    stats = {}
    for key in np.unique(lab):
        r = refs[lab == key]
        stats[key] = (np.median(r.mean(axis=0)), np.median(r.std(axis=0)))
    return stats


def _standardize(block, reference, fill=None):
    lab = block['lab_id']
    x = np.array(block['coupon_features_raw'], dtype=float, copy=True)
    for key in np.unique(lab):
        loc, scale = reference[key]
        x[lab == key] = (x[lab == key] - loc) / (scale if scale > 0 else 1.)
    valid = np.isfinite(x)
    if fill is None:
        _, group, counts = np.unique(np.column_stack((lab, block['stratum'])), axis=0,
                                     return_inverse=True, return_counts=True)
        w = 1. / counts[group, None]
        denom = (w * valid).sum(axis=0)
        fill = np.divide((w * np.where(valid, x, 0)).sum(axis=0), denom,
                         out=np.zeros(x.shape[1]), where=denom > 0)
    return np.column_stack((np.ones(len(lab)), np.where(valid, x, fill))), fill


def _bridge(cal):
    raw = cal['bridge_raw']
    return np.linalg.lstsq(np.column_stack((raw, np.ones(len(raw)))),
                           cal['true_channel_loads'], rcond=None)[0].sum(axis=1)


def _loads(block, bridge):
    return (block['past_channel_raw'] @ bridge[:-1] + bridge[-1],
            block['hold_channel_raw'] @ bridge[:-1] + bridge[-1])


def _design(loads, x, tau, age, times):
    past, hold = loads
    interval = np.diff(np.exp(-age[:, None] / tau), axis=0)
    state = hold[:, None] - past @ interval
    kernel = np.vstack((-np.expm1(-times[:, None] / tau), 1. / tau))
    drift = x[:, None, :] * np.r_[times, 1.][None, :, None]
    return np.concatenate((state[:, None, :] * kernel, drift), axis=2)


def _target(block):
    return np.column_stack((block['strain_curves_obs'] - block['strain_0'][:, None],
                            block['instant_rate_obs']))


def _fit(loads, x, y, age, times):
    # Profile linear coefficients at each start, then jointly fit with a robust loss.
    valid = np.isfinite(y)
    scale = np.r_[np.full(len(times), .025), .012] / .025
    def residual(p):
        return ((_design(loads, x, np.exp(p[:2]), age, times) @ p[2:] - y) / scale)[valid]
    bounds = (np.r_[np.log([.050001, .050001]), np.full(7, -np.inf)],
              np.r_[np.log([24.999999, 24.999999]), np.full(7, np.inf)])
    fits = []
    for start in ([.35, 2.5], [.75, 6.], [1.8, 12.]):
        a = (_design(loads, x, np.array(start), age, times) / scale[None, :, None])[valid]
        b = np.linalg.lstsq(a, (y / scale)[valid], rcond=None)[0]
        fits.append(least_squares(residual, np.r_[np.log(start), b], bounds=bounds,
                                  loss='cauchy', f_scale=.025, max_nfev=220))
    p = min(fits, key=lambda f: f.cost).x
    order = np.argsort(p[:2])
    tau = np.exp(p[:2][order])
    return tau, np.r_[p[2:4][order], p[4:]]


def _drift_residual(residual, times):
    z = np.r_[times, 1.]
    weight = np.r_[np.ones(len(times)), 3.]
    valid = np.isfinite(residual)
    clean = np.where(valid, residual, 0.)
    slope = (clean @ (weight * z)) / np.maximum(valid @ (weight * z*z), 1e-15)
    for _ in range(5):
        error = clean - slope[:, None] * z
        w = valid * weight / (1. + (error / .02)**2)
        slope = (w * clean) @ z / np.maximum(w @ (z*z), 1e-15)
    return slope


def _transfer(lab, score, slope, lab_new, score_new):
    if not len(lab):
        return np.zeros(len(score_new))
    keys = np.unique(lab)
    cells = (lab[:, None] == keys).astype(float)
    cells_new = (lab_new[:, None] == keys).astype(float)
    # Pool only the gain on the drift score. Offset corrections need local evidence.
    design = np.column_stack((cells, cells * score[:, None], score))
    design_new = np.column_stack((cells_new, cells_new * score_new[:, None], score_new))
    penalty = np.diag(np.r_[np.full(2 * len(keys), .08), .16])
    coef = np.linalg.lstsq(np.vstack((design, penalty)),
                          np.r_[slope, np.zeros(len(penalty))], rcond=None)[0]
    return design_new @ coef


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    chamber = dataset.get('chamber_verification_trials')
    if chamber is not None and not len(chamber['lab_id']):
        chamber = None
    times = np.asarray(dataset['eval_times'])
    age = np.r_[np.cumsum(np.asarray(dataset['past_dt'])[::-1])[::-1], 0.]
    blocks = [train, test] if chamber is None else [train, test, chamber]
    reference = _reference(blocks)
    bridge = _bridge(dataset['calibration_load_cells'])
    x_train, fill = _standardize(train, reference)
    x_test, _ = _standardize(test, reference, fill)
    tau, theta = _fit(_loads(train, bridge), x_train, _target(train), age, times)
    prediction = _design(_loads(test, bridge), x_test, tau, age, times) @ theta
    if chamber is not None:
        x_ch, _ = _standardize(chamber, reference, fill)
        residual = _target(chamber) - _design(_loads(chamber, bridge), x_ch, tau, age, times) @ theta
        valid = np.isfinite(residual).any(axis=1)
        delta = _transfer(chamber['lab_id'][valid], (x_ch[:, 1:] @ theta[3:])[valid],
                          _drift_residual(residual[valid], times), test['lab_id'],
                          x_test[:, 1:] @ theta[3:])
        prediction += delta[:, None] * np.r_[times, 1.]
    return {'estimated_taus': tau, 'hereditary_weights': theta,
            'predicted_strain': prediction[:, :-1] + test['strain_0'][:, None],
            'initial_rate_pred': prediction[:, -1]}
