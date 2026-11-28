"""Infer a calibrated cyclic operator and its scalar and temporal readouts."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _design(*blocks):
    return np.column_stack((np.ones(len(blocks[0])), *blocks))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _calibrate(panel, measured):
    def basis(x, j):
        a, b = x[:, j], x[:, (j + 1) % 4]
        return _design(a, b, a * b, a**3)

    u, v = panel['u'], panel['v']
    designs = [basis(u, j) for j in range(4)]
    # The reference panel has three grossly corrupted rows across its channels.
    rows = np.arange(len(u))
    for _ in range(12):
        weights = [_fit(g[rows], v[rows, j]) for j, g in enumerate(designs)]
        loss = sum((g @ w - v[:, j])**2 for j, (g, w) in enumerate(zip(designs, weights)))
        keep = np.sort(np.argsort(loss)[:len(u) - 3])
        if np.array_equal(keep, rows):
            break
        rows = keep
    weights = np.array([_fit(g[rows], v[rows, j]) for j, g in enumerate(designs)])
    offset, direct, neighbor, cross, cubic = weights.T
    linear = np.diag(direct) + np.diag(neighbor) @ np.roll(np.eye(4), -1, axis=0)
    initial = np.linalg.solve(linear, (measured - offset).T).T

    def inverse(row, start):
        def residual(a):
            b = np.roll(a, -1)
            return offset + direct * a + neighbor * b + cross * a * b + cubic * a**3 - row

        def jacobian(a):
            b = np.roll(a, -1)
            mat = np.diag(direct + cross * b + 3 * cubic * a**2)
            mat[np.arange(4), (np.arange(4) + 1) % 4] = neighbor + cross * a
            return mat

        return least_squares(residual, start, jac=jacobian, gtol=1e-11).x

    return np.array([inverse(row, start) for row, start in zip(measured, initial)])


def _operators(study, a):
    def shrink(x, threshold):
        return _design(np.sign(x) * np.maximum(np.abs(x) - threshold, 0)**1.5)

    panel = study['panel_b']
    u, v = panel['u'], panel['v']

    def panel_loss(threshold):
        g = shrink(u, threshold)
        return np.sum((g @ _fit(g, v) - v)**2)

    threshold = minimize_scalar(panel_loss, bounds=(0.15, 0.55), method='bounded').x
    b = shrink(a, threshold) @ _fit(shrink(u, threshold), v)

    def products(x):
        return _design(x * np.roll(x, -1, axis=1))

    panel = study['panel_c']
    return products(b) @ _fit(products(panel['u']), panel['v1'])


def _time_basis(times, observations):
    def basis(rates):
        return -np.expm1(-times[:, None] * rates)

    def residual(rates):
        g = basis(rates)
        return (g @ _fit(g, observations.T) - observations.T).ravel()

    rates = least_squares(residual, [0.4, 1.8, 5.2],
                          bounds=([0.1, 0.9, 3.0], [0.9, 3.0, 10.0]),
                          gtol=1e-10).x
    return basis(rates)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    measured = np.vstack((train['x'], study['test']['x']))
    a = _calibrate(study['panel_a'], measured)
    c = _operators(study, a)
    diagonal = (a * c).sum(axis=1)
    predecessor = a * np.roll(c, 1, axis=1)

    scalar = _design(a, c, diagonal, predecessor.sum(axis=1))
    y_hat = scalar[n:] @ _fit(scalar[:n], train['y'])

    spatial = _design(a, c, diagonal, predecessor)
    temporal = _time_basis(np.asarray(study['eval_times']), train['traj'])
    amplitudes = _fit(temporal, train['traj'].T).T
    state_pred = (spatial[n:] @ _fit(spatial[:n], amplitudes)) @ temporal.T
    terminal = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
