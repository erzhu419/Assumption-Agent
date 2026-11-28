"""Calibrate the input, apply the measured mixing operator, and predict two shared temporal modes."""
import numpy as np
from scipy.special import erf


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _coordinates(study, x):
    panel = study['panel_a']
    a, v = _design(panel['u']), np.asarray(panel['v'])
    keep = np.ones(len(a), dtype=bool)
    for _ in range(10):
        coef = np.linalg.lstsq(a[keep], v[keep], rcond=None)[0]
        e = np.linalg.norm(a @ coef - v, axis=1)
        new = e < max(.001, 3 * np.median(e))
        if np.array_equal(new, keep) or new.sum() < a.shape[1]:
            break
        keep = new
    z = np.linalg.solve(coef[1:].T, (x - coef[0]).T).T
    panel = study['panel_b']
    mix = np.linalg.lstsq(_design(panel['u']), panel['v'], rcond=None)[0]
    return _design(z) @ mix


def _mixed_design(b, h):
    # Include every bilinear interaction, but only diagonal pure quadratics.
    return _design(np.column_stack((b, h, b*b, h*h,
                                   (b[:, :, None] * h[:, None, :]).reshape(len(b), 16))))


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, xt = np.asarray(train['x']), np.asarray(study['test']['x'])
    n = len(x)
    b = _coordinates(study, np.concatenate((x, xt)))
    traj = np.asarray(train['traj'])
    basis = np.linalg.svd(traj, full_matrices=False)[2][:2]
    modes = traj @ basis.T
    targets = np.column_stack((train['y'], modes))
    soft = b / np.sqrt(1 + (1.4*b)**2)
    specs = ((soft, .0003, True, True),
             (np.tanh(1.75*b), .0003, True, False),
             (np.arctan(1.75*b), .0003, True, False),
             (b / (1 + b*b), .003, False, True),
             (np.tanh(1.4*b), .003, False, True),
             (erf(1.4*b), .003, False, True))
    y = np.zeros(len(xt))
    predicted_modes = np.zeros((len(xt), len(basis)))
    eye = np.eye(n)
    for h, ridge, scalar, dynamic in specs:
        f = _mixed_design(b, h)
        p = (f[n:] @ f[:n].T) @ np.linalg.solve(f[:n] @ f[:n].T + ridge*eye, targets)
        if scalar:
            y += p[:, 0] / 3
        if dynamic:
            predicted_modes += p[:, 1:] / 4
    state = predicted_modes @ basis
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
