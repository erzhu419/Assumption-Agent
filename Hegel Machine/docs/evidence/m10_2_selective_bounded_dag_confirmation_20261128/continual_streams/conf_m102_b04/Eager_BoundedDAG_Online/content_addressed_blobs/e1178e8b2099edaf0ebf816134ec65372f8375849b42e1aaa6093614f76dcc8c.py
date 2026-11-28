from itertools import combinations_with_replacement

import numpy as np


def _quadratic(x):
    return np.column_stack((np.ones(len(x)), x, *(x[:, i] * x[:, j] for i, j in combinations_with_replacement(range(x.shape[1]), 2))))


def _calibration(panel):
    u, v = (np.asarray(panel[k], float) for k in ('u', 'v'))
    prev = np.roll(np.arange(u.shape[1]), -1)
    features = [np.column_stack((np.ones(len(u)), u[:, j], u[:, prev[j]], u[:, j] * u[:, prev[j]], u[:, j] ** 3)) for j in range(u.shape[1])]
    weights = np.ones(len(u))
    for _ in range(12):
        root = np.sqrt(weights)
        coef = np.array([np.linalg.lstsq(a * root[:, None], v[:, j] * root, rcond=None)[0] for j, a in enumerate(features)])
        errors = np.linalg.norm(np.column_stack([a @ b for a, b in zip(features, coef)]) - v, axis=1)
        scale = max(3.0 * np.median(errors), 1e-4)
        weights = 1 / (1 + (errors / scale) ** 2)
    keep = errors <= max(8 * np.median(errors), 5e-4)
    if keep.sum() >= max(8, len(u) // 2):
        coef = np.array([np.linalg.lstsq(a[keep], v[keep, j], rcond=None)[0] for j, a in enumerate(features)])
    return coef, prev


def _invert(x, warm, coef, prev):
    bias, diagonal, coupling, cross, cubic = coef.T
    axis = np.arange(len(prev))
    def forward(z):
        p = z[:, prev]
        return bias + diagonal * z + coupling * p + cross * z * p + cubic * z ** 3
    linear = np.diag(diagonal)
    linear[axis, prev] = coupling
    affine = np.linalg.solve(linear, (x - bias).T).T
    use_warm = np.sum((forward(warm) - x) ** 2, axis=1) < np.sum((forward(affine) - x) ** 2, axis=1)
    z = np.where(use_warm[:, None], warm, affine)
    jac = np.zeros((len(z), len(axis), len(axis)))
    for _ in range(22):
        residual = forward(z) - x
        if np.max(np.abs(residual)) < 1e-11:
            break
        jac[:, axis, axis] = diagonal + cross * z[:, prev] + 3 * cubic * z * z
        jac[:, axis, prev] = coupling + cross * z
        try:
            step = np.linalg.solve(jac, residual[..., None])[..., 0]
        except np.linalg.LinAlgError:
            step = (np.linalg.pinv(jac) @ residual[..., None])[..., 0]
        best = np.sum(residual * residual, axis=1)
        new = z.copy()
        for damping in (1., .5, .25, .125, .0625, .03125):
            proposal = z - damping * step
            er = np.sum((forward(proposal) - x) ** 2, axis=1)
            better = er < best
            new[better], best[better] = proposal[better], er[better]
        z = new
    return z


def _ridge(a, t, n, penalty):
    train = a[:n]
    reg = penalty * np.eye(a.shape[1])
    reg[0, 0] = 0.
    return a[n:] @ np.linalg.solve(train.T @ train + reg, train.T @ t)


def _scalar(z, y, n, telemetry):
    h = z / np.sqrt(1 + telemetry[0] * z * z)
    p, q, r, s = h.T
    rat1 = q * r / (1 + (q + r) ** 2)
    rat2 = p * s / (1 + telemetry[2] * (p - s) ** 2)
    return _ridge(np.column_stack((_quadratic(h), np.sin(telemetry[2] * h), np.cos(telemetry[2] * h), rat1, rat2)), y, n, 1e-6)


def _trajectory(z, traj, n):
    h = z / np.sqrt(1 + .09 * z * z)
    time = np.linalg.svd(traj, full_matrices=False)[2][:2]
    features = np.column_stack((_quadratic(h), np.sin(1.75 * h), np.cos(1.75 * h)))
    return _ridge(features, traj @ time.T, n, 1e-6) @ time


def run_node(inputs: dict) -> dict:
    study, frozen = inputs['__input__'], inputs['node_00']
    train = study['train']
    n = len(train['x'])
    x = np.row_stack((train['x'], study['test']['x']))
    d = x.shape[1]
    start = np.asarray(frozen['hybrid'])[:, -((d + 1) * (d + 2) * (d + 3) // 6):][:, 1:d + 1]
    coef, prev = _calibration(study['panel_a'])
    z = _invert(x, start, coef, prev)
    y = _scalar(z, np.asarray(train['y']).reshape(n), n, np.asarray(study['telemetry_vec']))
    state = _trajectory(z, np.asarray(train['traj']), n)
    diagnostics = np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
