"""Invert the reference calibration, then fit both heads in Panel B coordinates."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _panel_a_basis(x, pair, frequency):
    z = x[:, pair]
    return np.column_stack((_affine(x), z ** 3, np.sin(frequency * z)))


def _calibrate(panel, raw, warm):
    u, v = panel['u'], panel['v']
    f = _affine(u)
    keep = np.arange(len(u))
    for _ in range(8):
        residual = np.linalg.norm(f @ _fit(f[keep], v[keep]) - v, axis=1)
        keep = np.argsort(residual)[:max(10, int(.82 * len(u)))]
    median = np.median(residual)
    mad = np.median(np.abs(residual - median))
    inlier = residual <= max(2.5 * median, median + 6 * mad, 1e-5)
    if np.count_nonzero(inlier) >= len(keep):
        keep = np.flatnonzero(inlier)
    u, v = u[keep], v[keep]
    models = []
    for j in range(u.shape[1]):
        pair = [j, (j + u.shape[1] // 2) % u.shape[1]]

        def objective(freq):
            g = _panel_a_basis(u, pair, freq)
            residual = g @ _fit(g, v[:, j]) - v[:, j]
            return residual @ residual

        freq = minimize_scalar(objective, bounds=(2.0, 3.6), method='bounded').x
        models.append((pair, freq, _fit(_panel_a_basis(u, pair, freq), v[:, j])))

    def forward(z):
        return np.array([(_panel_a_basis(z[None], pair, freq) @ coef)[0]
                         for pair, freq, coef in models])

    def jacobian(z):
        out = np.array([coef[1:1 + u.shape[1]] for _, _, coef in models])
        for j, (pair, freq, coef) in enumerate(models):
            p = 1 + u.shape[1]
            out[j, pair] += 3 * z[pair] ** 2 * coef[p:p + 2]
            out[j, pair] += freq * np.cos(freq * z[pair]) * coef[p + 2:p + 4]
        return out

    bound = np.maximum(2.5, 1.5 * np.max(np.abs(u), axis=0))
    return np.array([least_squares(lambda z: forward(z) - row,
                                   np.clip(start, -.99 * bound, .99 * bound),
                                   jac=jacobian, bounds=(-bound, bound)).x
                     for row, start in zip(raw, warm)])


def _panel_b_coordinates(panel, x):
    u, v = panel['u'], panel['v']

    def basis(z, freq):
        return _affine(np.cos(freq * z))

    def objective(freq):
        g = basis(u, freq)
        return np.sum((g @ _fit(g, v) - v) ** 2)

    grid = np.linspace(0.5, 6.0, 45)
    i = min(range(len(grid)), key=lambda j: objective(grid[j]))
    freq = minimize_scalar(objective, bounds=(grid[max(0, i - 1)],
                                               grid[min(len(grid) - 1, i + 1)]),
                           method='bounded').x
    return basis(x, freq) @ _fit(basis(u, freq), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    upstream = inputs['node_00']
    train, test = study['train'], study['test']
    raw = np.vstack((train['x'], test['x']))
    hybrid, aligned = upstream['hybrid'], upstream['aligned']
    start = aligned.shape[1] + 1
    warm = hybrid[:, start:start + raw.shape[1]]
    latent = _calibrate(study['panel_a'], raw, warm)
    x = _panel_b_coordinates(study['panel_b'], latent)
    f = np.column_stack((_affine(x), *[x[:, i] * x[:, j]
                       for i, j in combinations_with_replacement(range(x.shape[1]), 2)]))
    n = len(train['x'])
    g = f[:n]
    target = np.column_stack((train['y'], train['traj']))
    penalty = 1e-4 * np.eye(g.shape[1])
    penalty[0, 0] = 0.0
    pred = f[n:] @ np.linalg.solve(g.T @ g + penalty, g.T @ target)
    y, state = pred[:, 0], pred[:, 1:]
    diag = np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diag}
