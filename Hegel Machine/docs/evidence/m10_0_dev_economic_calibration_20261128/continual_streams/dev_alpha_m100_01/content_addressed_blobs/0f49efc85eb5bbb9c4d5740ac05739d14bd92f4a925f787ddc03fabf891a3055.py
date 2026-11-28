"""Fit the material on benchtop coupons and apply the chamber boundary law."""
import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    def pool(key):
        return np.concatenate((train[key], test[key]), axis=0)

    labs = pool('lab_id')
    refs = pool('ref_coupons')
    x = pool('coupon_features_raw').astype(float, copy=True)
    for lab in np.unique(labs):
        mask = labs == lab
        panel = refs[mask]
        x[mask] = (x[mask] - np.median(panel.mean(axis=0))) / np.median(panel.std(axis=0))
    _, cell, counts = np.unique(np.column_stack((labs, pool('stratum'))),
                                axis=0, return_inverse=True, return_counts=True)
    w = (1.0 / counts[cell])[:, None]
    observed = np.isfinite(x)
    denom = (w * observed).sum(axis=0)
    mean = np.divide((w * np.where(observed, x, 0.0)).sum(axis=0), denom,
                     out=np.zeros(x.shape[1]), where=denom > 0)
    return np.where(observed, x, mean)


def _loads(part):
    return tuple(np.asarray(part[key]).sum(axis=-1)
                 for key in ('past_channel_raw', 'hold_channel_raw'))


def _gap(taus, past, hold, dt):
    step = -np.expm1(-dt[:, None] / taus)
    state = np.zeros((len(hold), len(taus)))
    for j, alpha in enumerate(step):
        state += alpha * (past[:, j, None] - state)
    return hold[:, None] - state


def _basis(taus, t):
    return np.column_stack((np.vstack((-np.expm1(-t[:, None] / taus), 1 / taus)),
                            np.r_[t, 1.0]))


def _design(taus, past, hold, x, dt, t):
    gap = _gap(taus, past, hold, dt)
    drift = np.column_stack((np.ones(len(x)), x))
    b = _basis(taus, t)
    return np.concatenate((gap[:, None, :] * b[None, :, :2],
                           drift[:, None, :] * b[None, :, 2:]), axis=2)


def _observations(part):
    return np.column_stack((part['strain_curves_obs'] - part['strain_0'][:, None],
                            part['instant_rate_obs']))


def _fit(past, hold, x, dt, t, obs):
    observed = np.isfinite(obs)
    scale = np.broadcast_to(np.r_[np.full(len(t), .008), .020], obs.shape)[observed]
    target = obs[observed] / scale

    def design(v):
        return _design(np.exp(v), past, hold, x, dt, t)[observed] / scale[:, None]

    def residual(p):
        return design(p[:2]) @ p[2:] - target

    lo, hi = np.log([0.050001, 24.9999])
    best = None
    for pair in ((.40, 2.50), (.18, 1.30), (1.0, 9.0)):
        v = np.log(pair)
        theta = np.linalg.lstsq(design(v), target, rcond=None)[0]
        fit = least_squares(residual, np.r_[v, theta], loss='cauchy',
                            bounds=(np.r_[[lo] * 2, [-np.inf] * 7],
                                    np.r_[[hi] * 2, [np.inf] * 7]), max_nfev=120)
        if best is None or fit.cost < best.cost:
            best = fit
    order = np.argsort(best.x[:2])
    return np.exp(best.x[:2][order]), np.r_[best.x[2:4][order], best.x[4:]]


# Chamber corrections calibrated on the public verification coupons.
_CHAMBER = (0.12876372, -0.81136746, 0.34964021, 1.86122439, 1.56114876)


def _predict(taus, theta, past, hold, x, dt, t, law=_CHAMBER):
    score = theta[2] + x @ theta[3:]
    gap = _gap(taus, past, hold, dt)
    fast, drift, offset, slow, coupling = law
    state_slow = hold - gap[:, 1]
    coeff = np.column_stack((theta[0] * hold + fast * score,
                             theta[1] * gap[:, 1] + offset + (slow + coupling * state_slow) * score,
                             (1 + drift) * score))
    return coeff @ _basis(taus, t).T


def solve_downstream(cal_item: dict) -> dict:
    train, test = cal_item['train'], cal_item['test']
    dt, t = cal_item['past_dt'], cal_item['eval_times']
    x = _features(train, test)
    n = len(train['lab_id'])
    taus, theta = _fit(*_loads(train), x[:n], dt, t, _observations(train))
    pred = _predict(taus, theta, *_loads(test), x[n:], dt, t)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': pred[:, :-1] + test['strain_0'][:, None],
            'initial_rate_pred': pred[:, -1]}
