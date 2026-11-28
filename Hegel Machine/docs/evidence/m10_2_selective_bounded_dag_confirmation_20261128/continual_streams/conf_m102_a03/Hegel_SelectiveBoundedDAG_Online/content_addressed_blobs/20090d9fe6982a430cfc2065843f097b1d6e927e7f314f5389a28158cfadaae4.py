"""Recover latent coordinates and build separate state and readout bases."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-11)[0]


def _basis(x, j, scales):
    a = x[:, j]
    b = x[:, (j + 1) % 4]
    c = x[:, (j + 2) % 4]
    return np.column_stack((_affine(x), a / (1 + scales[0] * a*a), np.sin(scales[1] * c), a*b))


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    magnitude = np.linalg.norm(v, axis=1)
    keep = magnitude < 3 * np.median(magnitude)
    if keep.sum() < 12:
        keep = np.ones(len(u), dtype=bool)
    u, v = u[keep], v[keep]

    def coefficients(h):
        return [_fit(_basis(u, j, h), v[:, j]) for j in range(4)]

    def forward(x, h, co):
        return np.column_stack([_basis(x, j, h) @ co[j] for j in range(4)])

    h = least_squares(lambda h: (forward(u, h, coefficients(h)) - v).ravel(),
                      [1.25, 2.15], bounds=([0.1, 0.3], [5, 4])).x
    co = coefficients(h)
    inverse_start = _fit(_affine(v), u)
    return lambda x: forward(x, h, co), inverse_start


def _readout(latent):
    scale = np.sqrt(0.8)
    w = np.arctan(scale * latent) / scale
    columns = [np.ones(len(w))]
    for degree in (1, 2, 3):
        for indices in combinations_with_replacement(range(w.shape[1]), degree):
            if len(set(indices)) <= 2:
                columns.append(np.prod(w[:, indices], axis=1))
    return np.column_stack(columns)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = (np.asarray(study[k]['x'], dtype=float) for k in ('train', 'test'))
    x = np.concatenate((train, test), axis=0)
    forward, inverse_start = _calibration(study['panel_a'])
    latent = np.array([least_squares(lambda z: forward(z[None])[0] - observed,
                                     initial, gtol=1e-12).x
                       for observed, initial in zip(x, _affine(x) @ inverse_start)])
    state, readout = _affine(latent), _readout(latent)
    n = len(train)
    return {'z': state[:n], 'zt': state[n:],
            'readout_train': readout[:n], 'readout_test': readout[n:]}
