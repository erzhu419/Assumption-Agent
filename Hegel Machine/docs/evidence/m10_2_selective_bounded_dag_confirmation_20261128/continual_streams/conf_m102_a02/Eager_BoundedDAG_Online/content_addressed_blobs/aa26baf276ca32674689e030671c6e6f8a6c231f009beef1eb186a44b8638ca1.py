"""Fit the reference-panel calibration and the two calibrated readouts."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, x):
    u, v = (np.asarray(panel[key], dtype=float) for key in ('u', 'v'))
    n, d = u.shape
    axes = np.arange(d)
    opposite = (axes + d // 2) % d
    w = u[:, opposite]

    def design(freq):
        a = np.zeros((n, d, 2*d + 2))
        a[:, axes, axes] = 1.
        a[:, axes, axes+d] = u
        a[:, :, -2] = np.tanh(freq[0]*u)
        a[:, :, -1] = np.sin(freq[1]*w)
        return a

    def coefficients(a, keep):
        return _fit(a[keep].reshape(-1, a.shape[-1]), v[keep].ravel())

    # Screen the gross, rowwise reference faults before profiling frequency.
    size = np.max(np.abs(v), axis=1)
    med = np.median(size)
    mad = np.median(np.abs(size - med))
    keep = size <= med + 8*max(1.4826*mad, 0.1)
    freq = np.array([1.8, 2.4])
    for _ in range(6):
        def residual(k):
            a = design(k)
            return (a[keep] @ coefficients(a, keep) - v[keep]).ravel()

        freq = least_squares(residual, freq, bounds=([0.7, 1.5], [3.2, 3.4]),
                             max_nfev=60).x
        a = design(freq)
        c = coefficients(a, keep)
        err = np.linalg.norm(a @ c - v, axis=1)
        med = np.median(err[keep])
        mad = np.median(np.abs(err[keep] - med))
        updated = err <= med + 6*max(1.4826*mad, 3e-4)
        if updated.sum() < max(8, n//2) or np.array_equal(updated, keep):
            break
        keep = updated

    offset, scale = c[:d], c[d:2*d]
    amp_self, amp_opp = c[-2:]
    a, b = freq

    def invert(row):
        def residual(z):
            return offset + scale*z + amp_self*np.tanh(a*z) + amp_opp*np.sin(b*z[opposite]) - row

        def jac(z):
            j = np.diag(scale + amp_self*a*(1 - np.tanh(a*z)**2))
            j[axes, opposite] = amp_opp*b*np.cos(b*z[opposite])
            return j

        return least_squares(residual, (row-offset)/scale, jac=jac, max_nfev=60).x

    return np.array([invert(row) for row in x]).reshape(-1, d)


def _panel_coordinates(panel, z):
    u, v = (np.asarray(panel[key], dtype=float) for key in ('u', 'v'))

    def design(x, frequency):
        return np.column_stack((_affine(x), np.cos(frequency*x)))

    def loss(frequency):
        a = design(u, frequency)
        return np.mean((a @ _fit(a, v) - v)**2)

    frequency = minimize_scalar(loss, method='bounded', bounds=(2.1, 3.1),
                                options={'xatol': 1e-9}).x
    return design(z, frequency) @ _fit(design(u, frequency), v)


def _readout_design(g):
    h = g.shape[1] // 2
    return np.column_stack((_affine(g), g[:, :h]*g[:, h:]))


def run_node(inputs: dict) -> dict:
    study, raw = inputs['__input__'], inputs['node_00']
    train_x = raw['a'][:, 1:]
    test_x = raw['a_test'][:, 1:]
    latent = _calibrate(study['panel_a'], np.vstack((train_x, test_x)))
    g = _panel_coordinates(study['panel_b'], latent)
    design = _readout_design(g)
    train_design, test_features = design[:len(train_x)], design[len(train_x):]
    y = np.asarray(study['train']['y'], dtype=float)
    traj = np.asarray(study['train']['traj'], dtype=float)
    p = g.shape[1] + 1
    coefficients = np.zeros((design.shape[1], 1 + traj.shape[1]))
    coefficients[:, 0] = _fit(train_design, y)
    coefficients[:p, 1:] = _fit(train_design[:, :p], traj)
    return {'test_features': test_features, 'coefficients': coefficients}
