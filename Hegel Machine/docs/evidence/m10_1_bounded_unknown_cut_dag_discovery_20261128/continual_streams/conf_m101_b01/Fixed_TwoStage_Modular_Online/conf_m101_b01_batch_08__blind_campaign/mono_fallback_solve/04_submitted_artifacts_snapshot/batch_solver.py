"""Recover calibrated coordinates and predict the two public observables."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _next(x):
    return np.roll(x, -1, axis=1)


def _calibration(panel):
    u, v = panel['u'], panel['v']
    keep = np.argsort(np.linalg.norm(v - _affine(u) @ _fit(_affine(u), v), axis=1))[:45]

    def design(x, j, frequency=None):
        p, q = x[:, j], x[:, (j + 1) % 4]
        base = np.column_stack((np.ones(len(x)), p, q, p*q))
        if frequency is None:
            return base
        return np.column_stack((base, np.sin(frequency*p), np.sin(2*frequency*p)))

    for _ in range(5):
        residual = np.column_stack([
            design(u, j) @ _fit(design(u[keep], j), v[keep, j]) - v[:, j]
            for j in range(4)
        ])
        keep = np.argsort(np.linalg.norm(residual, axis=1))[:45]

    def profile(frequency):
        weights, errors = [], []
        for j in range(4):
            matrix = design(u[keep], j, frequency)
            pinv = np.linalg.pinv(matrix)
            w = _fit(matrix, v[keep, j])
            leverage = np.sum(matrix * pinv.T, axis=1)
            errors.append(np.mean(abs(v[keep, j] - matrix @ w)/(1-leverage)))
            weights.append(w)
        return np.mean(errors), weights

    grid = np.linspace(.35, 8, 32)
    j = np.argmin([profile(f)[0] for f in grid])
    frequency = minimize_scalar(lambda f: profile(f)[0], method='bounded',
                                bounds=(grid[max(0, j-1)], grid[min(len(grid)-1, j+1)])).x
    weights = profile(frequency)[1]

    def forward(x):
        return np.column_stack([design(x, j, frequency) @ weights[j] for j in range(4)])

    def inverse(rows):
        return np.array([least_squares(lambda z: forward(z[None])[0] - row,
                                       row, max_nfev=70).x for row in rows])
    return inverse


def _coordinates(study, x):
    z = _calibration(study['panel_a'])(x)
    panel = study['panel_b']
    u, v = panel['u'], panel['v']

    def warp(x, p):
        f, a, g, h = p
        return _affine(x + a*np.sin(f*x) + g*np.tanh(h*x))

    def residual(p):
        matrix = warp(u, p)
        return (matrix @ _fit(matrix, v) - v).ravel()

    p = least_squares(residual, [3., -.3, -.05, 2.], max_nfev=150).x
    b = warp(z, p) @ _fit(warp(u, p), v)
    panel = study['panel_c']
    u = panel['u']
    c = _affine(b*_next(b)) @ _fit(_affine(u*_next(u)), panel['v1'])
    return z, c


def _readout(z, c, y, traj):
    n = len(y)
    # The scalar and trajectory share a cyclic interaction with tied weights.
    cyclic = (z * np.roll(c, 1, axis=1)).sum(axis=1)
    state_design = _affine(np.column_stack((c, cyclic)))
    scalar_design = np.column_stack((state_design, c[:, 0] * c[:, 2]))
    yhat = scalar_design[n:] @ _fit(scalar_design[:n], y)
    state = state_design[n:] @ _fit(state_design[:n], traj)
    return yhat, state


def solve_batch(study: dict) -> dict:
    train = study['train']
    z, c = _coordinates(study, np.concatenate((train['x'], study['test']['x'])))
    y, state = _readout(z, c, train['y'], train['traj'])
    terminal = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}
