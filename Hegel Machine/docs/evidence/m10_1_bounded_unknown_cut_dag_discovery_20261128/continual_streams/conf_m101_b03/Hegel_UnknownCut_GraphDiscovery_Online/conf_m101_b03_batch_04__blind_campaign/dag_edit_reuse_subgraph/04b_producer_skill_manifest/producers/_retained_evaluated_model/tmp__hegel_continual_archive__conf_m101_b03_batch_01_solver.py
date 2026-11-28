"""Recover the calibration and paired operators, then fit two relaxation modes."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares


def _latent(panel, x):
    u, v = panel['u'], panel['v']
    linear = np.column_stack((np.ones(len(u)), u))
    keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        coef = np.linalg.lstsq(linear[keep], v[keep], rcond=None)[0]
        error = np.max(abs(linear @ coef - v), axis=1)
        keep = error < max(0.001, 3 * np.median(error))
    u, v = u[keep], v[keep]
    i = np.arange(4)
    j = (i + 1) % 4
    coef = np.column_stack([
        np.linalg.lstsq(np.column_stack((np.ones(len(u)), u[:, a], u[:, b],
                                         u[:, a]*u[:, b], u[:, a]**3)),
                        v[:, a], rcond=None)[0] for a, b in zip(i, j)
    ])
    shift, gain, cross, product, cubic = coef
    z = (x-shift)/gain
    for _ in range(20):
        zj = z[:, j]
        error = shift + gain*z + cross*zj + product*z*zj + cubic*z**3 - x
        if np.max(np.abs(error), initial=0.) < 1e-12:
            break
        jac = np.zeros((len(z), 4, 4))
        jac[:, i, i] = gain + product*zj + 3*cubic*z*z
        jac[:, i, j] = cross + product*z
        z -= np.linalg.solve(jac, error[..., None])[..., 0]
    return z


def _operator(panel, key, z):
    u, v = panel['u'], panel[key]
    if key == 'v1':
        def nonlinear(x, h):
            return np.tanh(h[0]*x) + h[1]*x/(1+(h[2]*x)**2)
        starts = [[w, c, f] for w in (1, 2.3, 3) for c in (-.3, .3) for f in (.8, 2)]
    else:
        def nonlinear(x, h):
            return x*x + h[1]*(np.cos(h[0]*x)-1)
        starts = [[w, -.2] for w in (1, 2.3, 3)]

    def design(x, h):
        return np.column_stack((np.ones(len(x)), x, nonlinear(x, h)))

    def residual(h, loo=False):
        q, _ = np.linalg.qr(design(u, h))
        r = v-q@(q.T@v)
        if loo:
            r = r/(1-(q*q).sum(axis=1, keepdims=True))
        return r.ravel()

    fits = (least_squares(residual, start, max_nfev=300) for start in starts)
    h = min(fits, key=lambda fit: np.linalg.norm(residual(fit.x, True))).x
    return design(z, h) @ np.linalg.lstsq(design(u, h), v, rcond=None)[0]


def _time_modes(times, trajectory):
    def basis(log_rates):
        return -np.expm1(-times[:, None]*np.exp(log_rates))

    def amplitudes(h):
        return np.linalg.lstsq(h, trajectory.T, rcond=None)[0].T

    def residual(log_rates):
        h = basis(log_rates)
        return (amplitudes(h) @ h.T - trajectory).ravel()

    result = least_squares(residual, np.log([.45, 1.8]), bounds=(np.log([.05, .9]), np.log([.9, 6.])))
    h = basis(result.x)
    return h, amplitudes(h)


def _features(study, x):
    z = _latent(study['panel_a'], x)
    panel = study['panel_c']
    a, b = (_operator(panel, key, z) for key in ('v1', 'v2'))
    paired = a*b
    scalar = np.column_stack((np.ones(len(z)), a, b, paired.sum(axis=1)))
    state = np.column_stack((scalar[:, :-1], paired))
    return scalar, state, b


def _matern(a, b):
    d = np.sqrt(5*np.sum((a[:, None, :] - b[None, :, :])**2, axis=2))/1.4
    return (1+d+d*d/3)*np.exp(-d)


def _predict_amplitudes(features, coordinates, amplitudes):
    n = len(amplitudes)
    d = features[:n]
    cov = _matern(coordinates[:n], coordinates[:n]) + .001*np.eye(n)
    inverse_design, inverse_targets = np.split(
        cho_solve(cho_factor(cov), np.column_stack((d, amplitudes))),
        [d.shape[1]], axis=1)
    coef = np.linalg.solve(d.T @ inverse_design + .0003*np.eye(d.shape[1]),
                           d.T @ inverse_targets)
    weights = inverse_targets - inverse_design @ coef
    return features[n:] @ coef + _matern(coordinates[n:], coordinates[:n]) @ weights


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    scalar, state, coordinates = _features(study, np.concatenate((train['x'], study['test']['x'])))
    h, amplitudes = _time_modes(study['eval_times'], train['traj'])
    y_hat = scalar[n:] @ np.linalg.lstsq(scalar[:n], train['y'], rcond=None)[0]
    state_pred = _predict_amplitudes(state, coordinates, amplitudes) @ h.T
    endpoint = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), endpoint.mean(), endpoint.std()])}
