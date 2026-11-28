"""Invert the panel-A calibration, then fit the panel-B readouts."""
import numpy as np
from scipy.optimize import least_squares


def _ls(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration_design(u):
    eye = np.eye(u.shape[1])
    return np.column_stack((np.tile(eye, (len(u), 1)),
                            (u[:, :, None] * eye).reshape(-1, u.shape[1]),
                            u.ravel()**3,
                            ((u - 1) * np.roll(u, -1, axis=1)).ravel()))


def _coordinates(study, x):
    a = study['panel_a']
    u, v = a['u'], a['v']
    # The three corrupted rows are separated from the clean calibration rows.
    keep = np.max(np.abs(v - u), axis=1) < 3
    offset, gain, cubic, coupling = np.split(
        _ls(_calibration_design(u[keep]), v[keep].ravel()), [4, 8, 9])
    w = (x - offset) / gain
    eye = np.eye(4)
    cyclic = np.roll(eye, 1, axis=1)
    for _ in range(15):
        nxt = np.roll(w, -1, axis=1)
        error = offset + gain*w + cubic*w**3 + coupling*(w-1)*nxt - x
        if np.max(np.abs(error)) < 1e-12:
            break
        jac = (gain + 3*cubic*w*w + coupling*nxt)[:, :, None] * eye
        jac += (coupling*(w-1))[:, :, None] * cyclic
        w -= np.clip(np.linalg.solve(jac, error[..., None])[..., 0], -0.5, 0.5)
    b = study['panel_b']
    return _affine(w) @ _ls(_affine(b['u']), b['v'])


def _scalar_design(z):
    return np.column_stack((z, z[:, 0]*z[:, 2] + z[:, 1]*z[:, 3]))


def _trajectory(z, target, query, times):
    fitted = z @ _ls(z, target)

    def basis(rates):
        return -np.expm1(-times[:, None]*rates)

    def residual(rates):
        b = basis(rates)
        return (b @ _ls(b, fitted.T) - fitted.T).ravel()

    rates = least_squares(residual, [0.4, 1.8], bounds=([0.01, 0.8], [0.8, 8])).x
    b = basis(rates)
    loadings = _ls(z, _ls(b, target.T).T)
    return (query @ loadings) @ b.T


def solve_batch(study: dict) -> dict:
    train = study['train']
    z = _coordinates(study, train['x'])
    q = _coordinates(study, study['test']['x'])
    y = _scalar_design(q) @ _ls(_scalar_design(z), train['y'])
    state = _trajectory(z, train['traj'], q, study['eval_times'])
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
