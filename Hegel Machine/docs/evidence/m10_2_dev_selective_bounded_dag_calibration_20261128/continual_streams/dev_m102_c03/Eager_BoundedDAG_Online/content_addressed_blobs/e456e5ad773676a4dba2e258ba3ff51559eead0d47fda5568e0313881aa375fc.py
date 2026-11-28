"""Calibrate the two panel operators and refit their observable readout."""
import numpy as np


def _bias(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _coordinates(calibration, x):
    u, v = (np.asarray(calibration[k], dtype=float) for k in ('u', 'v'))
    f = _bias(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(7):
        coeff = _fit(f[keep], v[keep])
        error = np.linalg.norm(f @ coeff - v, axis=1)
        updated = error <= max(1e-3, 4 * np.median(error))
        if np.count_nonzero(updated) < f.shape[1] + 1 or np.array_equal(keep, updated):
            break
        keep = updated
    coeff = _fit(f[keep], v[keep])
    return np.linalg.solve(coeff[1:].T, (x - coeff[0]).T).T


def _panel_design(u, second):
    nonlinear = np.arctan(1.078 * u) if second else u / (1 + 0.836 * np.abs(u))
    cyclic = u * np.roll(u, 1, axis=1)
    return _bias(np.column_stack((u, nonlinear, cyclic)))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    sparse = inputs['node_00']
    tr = study['train']
    x = np.asarray(tr['x'])
    n = len(x)
    z = _coordinates(study['panel_a'], np.vstack((x, study['test']['x'])))
    panel = study['panel_c']
    a, b = (_panel_design(z, second) @ _fit(_panel_design(panel['u'], second), panel[key])
            for second, key in ((False, 'v1'), (True, 'v2')))
    f = np.column_stack((a, b, np.sum(a*b, axis=1), a[:, 0]*b[:, 1]))
    mu = f[:n].mean(axis=0)
    scale = np.maximum(f[:n].std(axis=0), 1e-8)
    train, test = (f[:n]-mu)/scale, (f[n:]-mu)/scale
    traj = np.asarray(tr['traj'])
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    target = np.column_stack((tr['y'], traj @ modes.T))
    offset = target.mean(axis=0)
    coeff = np.linalg.solve(train.T @ train + 0.001 * np.eye(train.shape[1]), train.T @ (target - offset))
    readout = test @ coeff + offset
    y, state = readout[:, 0], readout[:, 1:] @ modes
    # Encode the refitted prediction as a correction to the frozen blend.
    sy, ss = sparse['y_hat'], sparse['state_pred']
    return {'panel_pred': np.column_stack((sy + (y - sy)/0.5, ss + (state - ss)/0.65))}
