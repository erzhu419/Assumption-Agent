"""Predict the scalar readout and trajectory from calibrated coordinates."""
import numpy as np
from scipy.optimize import least_squares


def _fit(design, response):
    return np.linalg.lstsq(design, response, rcond=None)[0]


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _calibration(panel):
    u = np.asarray(panel['u'], dtype=float)
    v = np.asarray(panel['v'], dtype=float)
    n, d = u.shape
    nxt = np.roll(u, -1, axis=1)
    # Separate channel offsets and gains; the cyclic coupling and cubic
    # correction are shared across channels.
    h = np.zeros((n, d, 2*d + 3))
    j = np.arange(d)
    h[:, j, j] = 1.
    h[:, j, d+j] = u
    h[:, :, -3] = nxt
    h[:, :, -2] = u*nxt
    h[:, :, -1] = u**3
    keep = np.ones(n, dtype=bool)
    for _ in range(6):
        c = _fit(h[keep].reshape(-1, h.shape[-1]), v[keep].ravel())
        err = np.linalg.norm(h @ c - v, axis=1)
        new = err <= max(5*np.median(err), 1.e-3)
        if np.array_equal(new, keep) or new.sum() < n//2:
            break
        keep = new
    return c[:d], c[d:2*d], c[-3:]


def _invert(x, calibration):
    bias, gain, (linear, product, cubic) = calibration
    j = np.arange(len(gain))
    k = (j+1) % len(j)

    def residual(z, target):
        return bias + gain*z + linear*z[k] + product*z*z[k] + cubic*z**3 - target

    def jacobian(z, target):
        jac = np.diag(gain + product*z[k] + 3*cubic*z*z)
        jac[j, k] = linear + product*z
        return jac

    return np.array([least_squares(residual, (target-bias)/gain, jac=jacobian,
                                  args=(target,), gtol=1.e-11).x for target in x]).reshape(x.shape)


def _readout(w):
    # The two opposite-channel products share one coefficient.
    interaction = w[:, 0]*w[:, 2] + w[:, 1]*w[:, 3]
    return np.column_stack((_affine(w), interaction))


def solve_batch(study: dict) -> dict:
    train = study['train']
    calibration = _calibration(study['panel_a'])
    z = _invert(np.asarray(train['x'], dtype=float), calibration)
    query = _invert(np.asarray(study['test']['x'], dtype=float), calibration)
    panel = study['panel_b']
    operator = _fit(_affine(np.asarray(panel['u'], dtype=float)),
                    np.asarray(panel['v'], dtype=float))
    fit_readout = _readout(_affine(z) @ operator)
    query_readout = _readout(_affine(query) @ operator)
    y_hat = query_readout @ _fit(fit_readout, np.asarray(train['y'], dtype=float))
    traj = np.asarray(train['traj'], dtype=float)
    # Learn the two temporal modes using training trajectories only.
    basis = np.linalg.svd(traj, full_matrices=False)[2][:2]
    state_pred = (_affine(query) @ _fit(_affine(z), traj @ basis.T)) @ basis
    terminal = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
