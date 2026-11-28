"""Predict the scalar and trajectory from calibrated panel-C inputs."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=1e-9)[0]


def _bias(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration_basis(x):
    return np.column_stack((_bias(x), x * np.roll(x, -1, axis=1), x**3))


def _calibrate(panel, raw):
    u, v = panel['u'], panel['v']
    affine = _fit(_bias(u), v)
    residual = np.linalg.norm(_bias(u) @ affine - v, axis=1)
    keep = residual < 4 * np.median(residual)
    weights = _fit(_calibration_basis(u[keep]), v[keep])
    initial = np.linalg.solve(affine[1:].T, (raw - affine[0]).T).T
    return np.array([
        least_squares(lambda z: _calibration_basis(z[None])[0] @ weights - r,
                      z0, gtol=1e-12).x
        for r, z0 in zip(raw, initial)
    ]).reshape(raw.shape)


def _bilinear_basis(x):
    a, b, c, d = x.T
    return _bias(np.column_stack((a * b, a * d, b * c, c * d)))


def _cosine_basis(x, frequency):
    return _bias(np.column_stack((x, np.cos(frequency * x))))


def _branches(panel, x):
    u, v1, v2 = (panel[k] for k in ('u', 'v1', 'v2'))
    c = _bilinear_basis(x) @ _fit(_bilinear_basis(u), v1)

    def cosine_error(frequency):
        design = _cosine_basis(u, frequency)
        return np.mean((design @ _fit(design, v2) - v2)**2)

    frequency = minimize_scalar(cosine_error, bounds=(0.7, 1.8), method='bounded').x
    d = _cosine_basis(x, frequency) @ _fit(_cosine_basis(u, frequency), v2)
    return c, d


def _trajectory_design(fast, slow, times, rates):
    exp_rise = -np.expm1(-rates[0] * times)
    rational_rise = times / (1 + rates[1] * times)
    return np.concatenate((fast[:, None, :] * exp_rise[None, :, None],
                           slow[:, None, :] * rational_rise[None, :, None]), axis=2)


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    n = len(train['x'])
    raw = np.concatenate((train['x'], test['x']))
    x = _calibrate(study['panel_a'], raw)
    c, d = _branches(study['panel_c'], x)
    dot = np.sum(c * d, axis=1)
    scalar = _bias(np.column_stack((c, d, dot)))
    y_hat = scalar[n:] @ _fit(scalar[:n], train['y'])

    fast = _bias(np.column_stack((c, dot)))
    slow = _bias(np.column_stack((d, c[:, 0] * d[:, 1])))
    times = study['eval_times']
    target = train['traj'].ravel()
    width = fast.shape[1] + slow.shape[1]

    def design_for_fit(rates):
        return _trajectory_design(fast[:n], slow[:n], times, rates).reshape(-1, width)

    def residual(rates):
        design = design_for_fit(rates)
        return design @ _fit(design, target) - target

    rates = least_squares(residual, (1.8, 0.4), bounds=((0.05, 0.001), (6, 2))).x
    weights = _fit(design_for_fit(rates), target)
    state_pred = _trajectory_design(fast[n:], slow[n:], times, rates) @ weights
    terminal = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
