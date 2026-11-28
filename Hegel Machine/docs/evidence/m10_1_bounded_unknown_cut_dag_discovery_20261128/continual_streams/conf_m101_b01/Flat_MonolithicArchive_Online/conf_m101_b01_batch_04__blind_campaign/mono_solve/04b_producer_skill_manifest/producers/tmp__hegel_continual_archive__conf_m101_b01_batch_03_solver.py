"""Predict the readout and two-mode trajectory from calibrated panel coordinates."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import hyp2f1


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _calibration_basis(x, j):
    a, b = x[:, j], x[:, (j + 1) % x.shape[1]]
    return np.column_stack((np.ones(len(x)), a, b, a*b, a**3))


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    # Use robust affine fits only to initialize outlier rejection and inversion.
    design = _affine(u)
    rng = np.random.default_rng(8)
    best_score = np.inf
    for _ in range(400):
        rows = rng.choice(len(u), design.shape[1], replace=False)
        err = np.linalg.norm(design @ _fit(design[rows], v[rows]) - v, axis=1)
        score = np.median(err)
        if score < best_score:
            best_score, best_err = score, err
    keep = best_err < max(0.001, 5 * best_score)
    for _ in range(3):
        pilot = _fit(design[keep], v[keep])
        err = np.linalg.norm(design @ pilot - v, axis=1)
        keep = err < max(0.001, 5 * np.median(err))
    j = np.arange(u.shape[1])
    k = (j + 1) % len(j)
    features = [_calibration_basis(u, i) for i in j]
    for _ in range(3):
        w = np.column_stack([_fit(f[keep], v[keep, i]) for i, f in enumerate(features)])
        err = np.linalg.norm(np.column_stack([f @ w[:, i] for i, f in enumerate(features)]) - v, axis=1)
        keep = err < max(0.001, 5 * np.median(err))
    z = np.linalg.solve(pilot[1:].T, (x - pilot[0]).T).T
    jac = np.zeros((len(x), len(j), len(j)))
    for _ in range(20):
        residual = w[0] + w[1]*z + w[2]*z[:, k] + w[3]*z*z[:, k] + w[4]*z**3 - x
        if np.max(np.abs(residual), initial=0.) < 1e-12:
            break
        jac[:, j, j] = w[1] + w[3]*z[:, k] + 3*w[4]*z*z
        jac[:, j, k] = w[2] + w[3]*z
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        z -= step / np.maximum(1., np.max(np.abs(step), axis=1, keepdims=True) / .3)
    return z


def _c1_coordinates(panel, x):
    u, v = panel['u'], panel['v1']

    def design(q, log_params):
        scale, power = np.exp(log_params[:2])
        linear = log_params[2]
        return _affine(q * (hyp2f1(.5, power, 1.5, -scale*q*q) + linear))

    def residual(log_params):
        g = design(u, log_params)
        return (g @ _fit(g, v) - v).ravel()

    params = least_squares(residual, [0., 0., 0.], bounds=(-4., 4.), max_nfev=200).x
    return design(x, params) @ _fit(design(u, params), v)


def _state(design, traj, prediction_design, times):
    def modes(log_rates):
        return -np.expm1(-times[:, None] * np.exp(log_rates))

    def residual(log_rates):
        h = modes(log_rates)
        return (h @ _fit(h, traj.T) - traj.T).ravel()

    rates = least_squares(residual, np.log([.4, 1.8]), bounds=(-8., 5.)).x
    h = modes(rates)
    return prediction_design @ _fit(design, _fit(h, traj.T).T) @ h.T


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']))
    calibrated = _calibrate(study['panel_a'], x)
    b = study['panel_b']
    operator_b = _affine(calibrated) @ _fit(_affine(b['u']), b['v'])
    q = _c1_coordinates(study['panel_c'], operator_b)
    state_design = _affine(q)
    readout_design = np.column_stack((state_design, q[:, :2] * q[:, 2:]))
    y = readout_design[n:] @ _fit(readout_design[:n], train['y'])
    state = _state(state_design[:n], train['traj'], state_design[n:], np.asarray(study['eval_times']))
    terminal = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}
