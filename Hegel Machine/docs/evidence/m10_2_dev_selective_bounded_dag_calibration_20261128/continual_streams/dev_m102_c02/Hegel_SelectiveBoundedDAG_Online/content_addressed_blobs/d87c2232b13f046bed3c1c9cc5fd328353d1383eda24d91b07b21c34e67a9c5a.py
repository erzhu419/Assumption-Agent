"""Recover readout coordinates from the two paired calibration panels."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _sensor(u, v):
    nxt = np.roll(u, -1, axis=1)
    # Gross contamination affects complete calibration rows.
    size = np.max(np.abs(v), axis=1)
    center = np.median(size)
    spread = np.median(np.abs(size - center))
    keep = size <= center + 9 * max(spread, 1e-8)
    if keep.sum() < 8:
        keep = np.ones(len(u), dtype=bool)

    def design(j, k):
        x = u[:, j]
        return np.column_stack((np.ones(len(u)), x, nxt[:, j], x*nxt[:, j], (np.sin(k*x)-k*x)/k**3))

    def fit(k):
        ds = [design(j, k)[keep] for j in range(4)]
        ws = [np.linalg.lstsq(d, v[keep, j], rcond=None)[0] for j, d in enumerate(ds)]
        return sum(np.sum((d@w-v[keep, j])**2) for j, (d, w) in enumerate(zip(ds, ws))), np.array(ws)

    k = minimize_scalar(lambda k: fit(k)[0], bounds=(.06, 1.7), method='bounded').x
    return k, fit(k)[1]


def _invert(x, k, w):
    def forward(z):
        nxt = np.roll(z, -1)
        return w[:, 0] + w[:, 1]*z + w[:, 2]*nxt + w[:, 3]*z*nxt + w[:, 4]*(np.sin(k*z)-k*z)/k**3

    def jac(z):
        a = np.diag(w[:, 1]+w[:, 3]*np.roll(z, -1)+w[:, 4]*(np.cos(k*z)-1)/k**2)
        j = np.arange(4)
        a[j, (j+1)%4] = w[:, 2]+w[:, 3]*z
        return a

    return np.array([least_squares(lambda z: forward(z)-row, row, jac=jac, gtol=1e-11, max_nfev=80).x for row in x]).reshape(-1, 4)


def _second(u, v, x):
    def design(a, scale):
        return np.column_stack((np.ones(len(a)), a, np.tanh(scale*a)))

    def fit(scale):
        d = design(u, scale)
        w = np.linalg.lstsq(d, v, rcond=None)[0]
        return np.sum((d@w-v)**2), w

    scale = minimize_scalar(lambda scale: fit(scale)[0], bounds=(.35, 3.), method='bounded').x
    return design(x, scale) @ fit(scale)[1]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a = study['panel_a']
    b = study['panel_b']
    train = np.asarray(study['train']['x'], dtype=float)
    query = np.asarray(study['test']['x'], dtype=float)
    k, w = _sensor(np.asarray(a['u'], dtype=float), np.asarray(a['v'], dtype=float))
    latent = _invert(np.concatenate((train, query)), k, w)
    coord = _second(np.asarray(b['u'], dtype=float), np.asarray(b['v'], dtype=float), latent)
    return {'c': coord[:len(train)], 'query': coord[len(train):]}
