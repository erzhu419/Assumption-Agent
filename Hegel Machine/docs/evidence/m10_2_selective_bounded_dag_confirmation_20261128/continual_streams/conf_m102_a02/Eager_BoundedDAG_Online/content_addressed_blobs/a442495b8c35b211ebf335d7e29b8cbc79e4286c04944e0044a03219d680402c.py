"""Recover latent coordinates from the calibration panel and fit readouts."""
from itertools import combinations_with_replacement
import numpy as np


def _calibrate(panel):
    u = np.asarray(panel['u'], dtype=float)
    v = np.asarray(panel['v'], dtype=float)
    # Rowwise rejection: corrupt pairs have a large joint displacement.
    delta = np.max(np.abs(v - u), axis=1)
    good = delta < max(3.0, 4.0 * np.median(delta))
    u, v = u[good], v[good]
    p = np.roll(u, -1, axis=1)
    return np.stack([np.linalg.lstsq(np.column_stack((np.ones(len(u)), u[:, j], p[:, j], u[:, j]*p[:, j], u[:, j]**3)), v[:, j], rcond=None)[0] for j in range(u.shape[1])], axis=1)


def _invert(x, b):
    n, dim = x.shape
    j = np.arange(dim)
    k = (j + 1) % dim
    A = np.diag(b[1])
    A[j, k] += b[2]
    w = np.linalg.solve(A, (x-b[0]).T).T
    J = np.broadcast_to(A, (n, dim, dim)).copy()
    for _ in range(20):
        p = w[:, k]
        err = b[0] + b[1]*w + b[2]*p + b[3]*w*p + b[4]*w**3 - x
        if np.max(np.abs(err), initial=0.) < 1e-12:
            break
        J[:, j, j] = b[1] + b[3]*p + 3*b[4]*w*w
        J[:, j, k] = b[2] + b[3]*w
        w -= np.linalg.solve(J, err[..., None])[..., 0]
    return w


def _design(w):
    affine_sine = np.column_stack((np.ones(len(w)), w, np.sin(3.0*w)))
    products = [w[:, i]*w[:, j] for i, j in combinations_with_replacement(range(w.shape[1]), 2)]
    return np.column_stack((affine_sine, *products))


def _ridge(z, y, penalty):
    return np.linalg.solve(z.T @ z + penalty*np.eye(z.shape[1]), z.T @ y)


def run_node(inputs: dict) -> dict:
    study, raw = inputs['__input__'], inputs['node_00']
    b = _calibrate(study['panel_a'])
    x = _invert(raw['a'][:, 1:], b)
    z = _design(x)
    zt = _design(_invert(raw['a_test'][:, 1:], b))
    train = study['train']
    width = 1 + 2*x.shape[1]
    endpoint = _ridge(z, train['y'], .002)
    trajectory = _ridge(z[:, :width], train['traj'], .001)
    coef = np.zeros((z.shape[1], 1 + trajectory.shape[1]))
    coef[:, 0] = endpoint
    coef[:width, 1:] = trajectory
    return {'test_features': zt, 'coefficients': coef}
