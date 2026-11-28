"""Predict the scalar readout and two-mode state trajectory."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import hyp2f1


def _calibration(panel):
    u, v = panel['u'], panel['v']
    rng = np.random.default_rng(452)
    best = (np.inf, None, None)
    for _ in range(100):
        i, j = rng.choice(len(u), 2, replace=False)
        d = u[j] - u[i]
        if np.min(np.abs(d)) < 0.06:
            continue
        gain = (v[j] - v[i]) / d
        offset = v[i] - gain * u[i]
        err = np.linalg.norm(u * gain + offset - v, axis=1)
        score = np.median(err)
        if score < best[0]:
            best = (score, gain, offset)
    _, gain, offset = best
    err = np.linalg.norm(u * gain + offset - v, axis=1)
    keep = err <= max(0.001, 5 * np.median(err))
    uc, vc = u[keep], v[keep]
    a = uc - uc.mean(axis=0)
    gain = np.sum(a * (vc - vc.mean(axis=0)), axis=0) / np.sum(a * a, axis=0)
    return gain, vc.mean(axis=0) - uc.mean(axis=0) * gain


def _basis(times, rates):
    return -np.expm1(-np.exp(rates)[:, None] * times)


def _time_fit(times, traj):
    def residual(rates):
        b = _basis(times, rates)
        return (traj @ np.linalg.pinv(b) @ b - traj).ravel()
    fit = least_squares(residual, [-1.0, 1.0], bounds=(-8.0, 4.0))
    return _basis(times, np.sort(fit.x))


def _affine_features(x):
    return np.column_stack((np.ones(len(x)), x))


def _coordinates(study, x):
    gain, offset = _calibration(study['panel_a'])
    panel = study['panel_b']
    operator = np.linalg.lstsq(_affine_features(panel['u']), panel['v'], rcond=None)[0]
    return _affine_features((x - offset) / gain) @ operator


def _features(x, p):
    a = x * np.exp(p[0]) + p[1:5]
    # Integral of (1 + a**2)**(-p[5]), with a shared shape across coordinates.
    h = a * hyp2f1(0.5, p[5], 1.5, -a*a)
    state = np.column_stack((np.ones(len(x)), h, h*h))
    left, right = np.triu_indices(4, 1)
    readout = np.column_stack((state, h[:, left] * h[:, right]))
    return readout, state


def _spatial_fit(x, targets):
    scale = np.maximum(0.05, targets.std(axis=0))

    normalized = targets / scale

    def profile(d, v):
        penalty = np.diag(np.r_[0., np.full(d.shape[1] - 1, 0.02)])
        w = np.linalg.lstsq(np.vstack((d, penalty)),
                            np.vstack((v, np.zeros((len(penalty), v.shape[1])))),
                            rcond=None)[0]
        return w, (d @ w - v).ravel(), (penalty @ w).ravel()

    def residual(p):
        d, g = _features(x, p)
        _, e1, r1 = profile(d, normalized[:, :1])
        _, e2, r2 = profile(g, normalized[:, 1:])
        return np.r_[e1, e2, r1, r2, 0.07*p[1:5]]

    fit = least_squares(residual, [np.log(3.), 0., 0., 0., 0., 0.5],
                        bounds=([-1.5] + [-0.45]*4 + [0.2], [1.9] + [0.45]*4 + [1.3]),
                        max_nfev=75)
    d, g = _features(x, fit.x)
    w, _, _ = profile(d, normalized[:, :1])
    v, _, _ = profile(g, normalized[:, 1:])
    return fit.x, w[:, 0] * scale[0], v * scale[1:]


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = _coordinates(study, np.concatenate((train['x'], study['test']['x'])))
    n = len(train['x'])
    b = _time_fit(study['eval_times'], train['traj'])
    targets = np.column_stack((train['y'], train['traj'] @ np.linalg.pinv(b)))
    p, w, v = _spatial_fit(x[:n], targets)
    d, g = _features(x[n:], p)
    y, state = d @ w, g @ v @ b
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
