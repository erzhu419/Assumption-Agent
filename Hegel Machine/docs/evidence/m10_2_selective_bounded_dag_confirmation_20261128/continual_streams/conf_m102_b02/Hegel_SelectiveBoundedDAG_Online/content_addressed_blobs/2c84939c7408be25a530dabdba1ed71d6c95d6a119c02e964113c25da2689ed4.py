"""Recover latent coordinates and compose the two measured calibration maps."""
import numpy as np
from scipy.optimize import least_squares, root
from scipy.special import erf


def _sensor_coordinates(panel, observed):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    # Gross contamination is separated from the calibration cloud by row norm.
    radius = np.linalg.norm(v - np.median(v, axis=0), axis=1)
    keep = radius < 4 * max(np.median(radius), 1e-8)
    if np.count_nonzero(keep) < 8:
        keep = np.ones(len(u), dtype=bool)

    def design(q):
        neighbor = np.roll(q, -1, axis=-1)
        return np.stack((np.ones_like(q), q, neighbor, q * neighbor, q**3), axis=-1)

    d = design(u)
    coeff = np.column_stack([np.linalg.lstsq(d[keep, j, :], v[keep, j], rcond=None)[0]
                             for j in range(u.shape[1])])
    affine = np.diag(coeff[1]) + np.roll(np.diag(coeff[2]), 1, axis=0)
    initial = np.linalg.solve(affine, (observed - coeff[0]).T).T

    def forward(q):
        return np.sum(design(q) * coeff.T, axis=-1)

    recovered = []
    for row, guess in zip(observed, initial):
        result = root(lambda q: forward(q) - row, guess, tol=1e-10)
        q = result.x
        if np.max(np.abs(forward(q) - row)) > 1e-7:
            q = least_squares(lambda z: forward(z) - row, guess).x
        recovered.append(q)
    return np.asarray(recovered).reshape(observed.shape)


def _profile_map(panel, target, query, kernel, starts):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', target))

    def design(q, params):
        return np.column_stack((np.ones(len(q)), kernel(q, params), q, q**3, q**2))

    def fit(params):
        d = design(u, params)
        w = np.linalg.lstsq(d, v, rcond=None)[0]
        return (d @ w - v).ravel(), w

    best = None
    for start in starts:
        p = least_squares(lambda p: fit(p)[0], start, bounds=(-12, 12), max_nfev=200).x
        residual, w = fit(p)
        error = residual @ residual
        if best is None or error < best[0]:
            best = error, p, w
    _, p, w = best
    return design(query, p) @ w


def _erf_cubic(q, p):
    return erf(np.exp(p[0]) * q + p[1] * q**3)


def _rational(q, p):
    return q / (1.0 + np.exp(p[0]) * q**2)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    raw_train = np.asarray(study['train']['x'], dtype=float)
    raw_test = np.asarray(study['test']['x'], dtype=float)
    n = len(raw_train)
    x = _sensor_coordinates(study['panel_a'], np.concatenate((raw_train, raw_test)))
    b = _profile_map(study['panel_b'], 'v', x, _erf_cubic, ((.6, 4.), (.4, -.27)))
    c = _profile_map(study['panel_c'], 'v1', b, _rational, ((0.,), (-1.,), (1.,)))
    return dict(x_train=x[:n], x_test=x[n:], c_train=c[:n], c_test=c[n:])
