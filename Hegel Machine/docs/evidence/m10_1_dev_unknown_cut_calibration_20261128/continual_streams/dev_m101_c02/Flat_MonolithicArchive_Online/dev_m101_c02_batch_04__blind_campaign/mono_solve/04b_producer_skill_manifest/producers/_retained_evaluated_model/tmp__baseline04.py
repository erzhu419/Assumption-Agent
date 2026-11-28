"""Predict scalar and trajectory readouts from calibrated panel-B coordinates."""
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    good = np.max(np.abs(v - u), axis=1) < 3
    u, v = u[good], v[good]
    eye = np.eye(4)
    next_coord = np.roll(eye, 1, axis=1)
    design = np.column_stack((np.tile(eye, (len(u), 1)),
                              (u[:, :, None] * eye).reshape(-1, 4),
                              u.ravel()**3,
                              ((u - 1) * np.roll(u, -1, axis=1)).ravel()))
    offset, gain, cubic, coupling = np.split(_ls(design, v.ravel()), [4, 8, 9])
    w = (x - offset) / gain
    for _ in range(20):
        neighbor = np.roll(w, -1, axis=1)
        residual = offset + gain*w + cubic*w**3 + coupling*(w-1)*neighbor - x
        jac = ((gain + 3*cubic*w*w + coupling*neighbor)[:, :, None]*eye
               + (coupling*(w-1))[:, :, None]*next_coord)
        w -= np.clip(np.linalg.solve(jac, residual[..., None])[..., 0], -.5, .5)
    return w


def _scalar_design(z):
    return np.column_stack((_affine(z), z[:, 0]*z[:, 2] + z[:, 1]*z[:, 3]))


def _predict(panel, w, q, y, trajectory):
    def basis(x):
        return chebvander(x / 1.25, 15)[..., 1::2]

    def features(b, p):
        return _affine(b[..., 1] + b[..., 0]*p[0] + b[..., 2:] @ p[1:])

    bu, bw, bq = basis(panel['u']), basis(w), basis(q)
    v = panel['v']

    def profile(p):
        a, d = features(bu, p), features(bw, p)
        op = _ls(a, v)
        scalar = _scalar_design(d @ op)
        cy, ct = _ls(scalar, y), _ls(d, trajectory)
        residual = np.r_[(a @ op - v).ravel(), 4*(scalar @ cy - y),
                         (4 / np.sqrt(trajectory.shape[1])) * (d @ ct - trajectory).ravel()]
        return residual, op, cy, ct

    init = [4, -.3, 0, 0, 0, 0, 0]
    p = least_squares(lambda p: profile(p)[0], init, bounds=(-15, 15), max_nfev=100).x
    _, op, cy, ct = profile(p)
    design = features(bq, p)
    return _scalar_design(design @ op) @ cy, design @ ct


def solve_batch(study: dict) -> dict:
    train = study['train']
    w = _calibrate(study['panel_a'], train['x'])
    q = _calibrate(study['panel_a'], study['test']['x'])
    y, state = _predict(study['panel_b'], w, q, train['y'], train['traj'])
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
