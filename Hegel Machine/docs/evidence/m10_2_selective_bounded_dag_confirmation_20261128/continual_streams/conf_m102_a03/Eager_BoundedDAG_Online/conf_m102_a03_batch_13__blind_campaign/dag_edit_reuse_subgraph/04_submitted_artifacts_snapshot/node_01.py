"""Recover calibrated coordinates and fit scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _cosine_basis(u, freq):
    return np.column_stack((_affine(u), np.cos(freq * u)))


def _align(panel, handoff):
    u, v = (np.asarray(panel[key], dtype=float) for key in ('u', 'v'))
    frozen = _fit(_affine(u), v)
    aligned = np.concatenate([np.asarray(handoff[key], dtype=float) for key in ('z', 'zt')])
    raw = np.linalg.solve(frozen[1:].T, (aligned - frozen[0]).T).T

    def loss(freq):
        basis = _cosine_basis(u, freq)
        return np.mean((basis @ _fit(basis, v) - v)**2)

    grid = np.linspace(0.25, 8.0, 64)
    best = int(np.argmin([loss(freq) for freq in grid]))
    freq = minimize_scalar(loss, bounds=(grid[max(0, best-1)], grid[min(len(grid)-1, best+1)]),
                           method='bounded', options={'xatol': 1e-9}).x
    return _cosine_basis(raw, freq) @ _fit(_cosine_basis(u, freq), v)


def _scalar_readout(z, y, n):
    linear = _affine(z)
    pair = z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]
    y = np.asarray(y, dtype=float)

    def predict(p, rows):
        index = linear[rows] @ p[:5]
        return p[5] * np.tanh(index) + p[6] * pair[rows] + p[7] * index

    observed, unseen = slice(0, n), slice(n, None)
    initial = np.r_[_fit(linear[observed], y) / 2.2, 2.2, 0.22, 0.0]
    lower = np.r_[np.full(5, -10.), 0.25, -4., -4.]
    upper = np.r_[np.full(5, 10.), 12., 4., 4.]
    params = least_squares(lambda p: predict(p, observed) - y,
                           np.clip(initial, lower, upper),
                           bounds=(lower, upper), max_nfev=300).x
    return predict(params, unseen)


def run_node(inputs: dict) -> dict:
    study, handoff = inputs['__input__'], inputs['node_00']
    train = study['train']
    n = len(train['y'])
    z = _align(study['panel_b'], handoff)
    yh = _scalar_readout(z, train['y'], n)
    traj = _affine(z)
    sp = traj[n:] @ _fit(traj[:n], train['traj'])
    diag = np.array([yh.mean(), yh.std(), sp[:, -1].mean(), sp[:, -1].std()])
    return {'y_hat': yh, 'state_pred': sp, 'diagnostics_vec': diag}
