import itertools

import numpy as np


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _response(x, gain):
    return np.column_stack((_aff(x), np.tanh(gain * x)))


def _calibrate(u, v, z):
    # Profile the shared saturation gain using panel-only leave-one-out error.
    def loss(gain):
        a = _response(u, gain)
        q, _ = np.linalg.qr(a, mode='reduced')
        leverage = np.sum(q*q, axis=1)
        residual = v - q @ (q.T @ v)
        return np.mean((residual / np.maximum(1.0 - leverage[:, None], 1e-10))**2)

    lo, hi = 0.3, 4.0
    for _ in range(3):
        grid = np.linspace(lo, hi, 33)
        i = min(range(len(grid)), key=lambda j: loss(grid[j]))
        lo, hi = grid[max(0, i-1)], grid[min(len(grid)-1, i+1)]
    gain = grid[i]
    return _response(z, gain) @ _fit(_response(u, gain), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    frozen = inputs['node_00']
    b = study['panel_b']
    c = study['panel_c']
    u, v = b['u'], b['v']
    linear = _fit(_aff(u), v)
    # Undo only the mixing in the frozen approximation, retaining its
    # robust sensor calibration, then fit the nonlinear panel response.
    z = np.linalg.solve(linear[1:].T, (frozen['w'] - linear[0]).T).T
    calibrated = _calibrate(u, v, z)
    g = 2.1
    c1 = _fit(_aff(np.arctan(g*c['u'])), c['v1'])
    core = _aff(np.arctan(g*calibrated))
    mixed = core @ c1
    pair = np.column_stack([mixed[:, i]*mixed[:, j]
                            for i, j in itertools.combinations(range(mixed.shape[1]), 2)])
    design = np.column_stack((core, pair))
    tr = study['train']
    n = len(tr['y'])
    pred = design[n:] @ _fit(design[:n], np.column_stack((tr['y'], tr['traj'])))
    y, state = pred[:, 0], pred[:, 1:]
    tail = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}
