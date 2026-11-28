"""Infer the calibration, affine operator, and two-mode relaxation from the study."""
import numpy as np
from scipy.optimize import least_squares


def _least(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    u, v = (np.asarray(panel[key], dtype=float) for key in ('u', 'v'))
    j = np.arange(u.shape[1]); opposite = (j + 2) % len(j)
    distance = np.linalg.norm(v-u, axis=1)
    keep = distance < np.median(distance) + 8 * np.median(abs(distance-np.median(distance)))

    def fit(k, mask):
        sn = np.sin(k*u)
        designs = np.stack((np.ones_like(u), u, sn, sn[:, opposite]), axis=-1)
        w = np.array([_least(designs[mask, i], v[mask, i]) for i in j])
        return w, np.einsum('nij,ij->ni', designs, w)-v

    for _ in range(3):
        k = least_squares(lambda p: fit(p[0], keep)[1][keep].ravel(), [2.4],
                          bounds=(1.5, 3.5)).x[0]
        w, residual = fit(k, keep)
        error = np.linalg.norm(residual, axis=1)
        mid = np.median(error)
        new_keep = error <= max(1.e-8, mid+7*np.median(abs(error-mid)))
        if np.array_equal(keep, new_keep) or new_keep.sum() < len(u)//2:
            break
        keep = new_keep
    return k, w


def _invert(x, calibration):
    k, w = calibration
    bias, gain, self_amp, cross_amp = w.T
    j = np.arange(x.shape[1]); opposite = (j + 2) % len(j)
    z = (x-bias)/gain
    jac = np.zeros((len(x), len(j), len(j)))
    for _ in range(35):
        sin = np.sin(k*z)
        r = bias+gain*z+self_amp*sin+cross_amp*sin[:, opposite]-x
        if np.max(abs(r), initial=0) < 1e-12:
            break
        cos = k*np.cos(k*z)
        jac[:, j, j] = gain+self_amp*cos
        jac[:, j, opposite] = cross_amp*cos[:, opposite]
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        z -= step/np.maximum(1., np.max(abs(step), axis=1, keepdims=True))
    return z


def _scalar_design(h):
    return np.column_stack((_affine(h), h[:, 0]*h[:, 2], h[:, 1]*h[:, 3]))


def _trajectory(z, z_test, times, trajectory):
    def basis(p):
        return -np.expm1(-times[:, None]*p)

    def residual(p):
        f = basis(p)
        return (f @ _least(f, trajectory.T)-trajectory.T).ravel()

    rates = least_squares(residual, [.4, 1.8], bounds=(.001, 10.)).x
    f = basis(rates)
    amplitudes = _least(f, trajectory.T).T
    return (_affine(z_test) @ _least(_affine(z), amplitudes)) @ f.T


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    calibration = _calibration(study['panel_a'])
    z = _invert(np.asarray(train['x'], dtype=float), calibration)
    z_test = _invert(np.asarray(test['x'], dtype=float), calibration)
    b = study['panel_b']
    operator = _least(_affine(b['u']), b['v'])
    h, h_test = _affine(z) @ operator, _affine(z_test) @ operator
    y = _scalar_design(h_test) @ _least(_scalar_design(h), train['y'])
    t = _trajectory(z, z_test, np.asarray(study['eval_times']), train['traj'])
    return {'y_hat': y, 'state_pred': t,
            'diagnostics_vec': np.array([y.mean(), y.std(), t[:, -1].mean(), t[:, -1].std()])}
