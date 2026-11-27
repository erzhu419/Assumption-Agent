import numpy as np
from itertools import combinations_with_replacement

_TERMS = [()] + [j for p in (1, 2, 3) for j in combinations_with_replacement(range(4), p)]


def _fit_bridge(cal):
    loads = cal['true_channel_loads']
    raw = cal['bridge_raw']
    basis = np.column_stack([loads[:, j].prod(axis=1) for j in _TERMS])
    w = np.ones(len(loads))
    for scale in (0.5, 0.1, 0.02, 0.004, 0.001):
        for _ in range(20):
            coef = np.linalg.lstsq(basis * w[:, None], raw * w[:, None], rcond=None)[0]
            error = np.linalg.norm(raw - basis @ coef, axis=1)
            w = 1 / np.hypot(1., error / scale)
    good = error < 0.01
    # The calibration isolates one cyclic cross term and one self cubic per channel.
    support = np.abs(coef) > 0.002
    support[:5] = True
    coef[:] = 0.
    for k in range(4):
        take = support[:, k]
        coef[take, k] = np.linalg.lstsq(basis[good][:, take], raw[good, k], rcond=None)[0]
    terms = [(j, b) for j, b in zip(_TERMS[5:], coef[5:]) if np.any(b)]
    return coef[0], coef[1:5], terms


def _decode(raw, bridge):
    offset, mat, terms = bridge
    obs = np.asarray(raw).reshape(-1, 4) - offset
    x = np.linalg.solve(mat.T, obs.T).T
    for _ in range(20):
        residual = x @ mat - obs
        jac = np.broadcast_to(mat.T, (len(x), 4, 4)).copy()
        for indices, coef in terms:
            residual += x[:, indices].prod(1)[:, None] * coef
            for i, idx in enumerate(indices):
                other = indices[:i] + indices[i+1:]
                jac[:, :, idx] += x[:, other].prod(1)[:, None] * coef
        step = np.linalg.solve(jac, residual[:, :, None])[:, :, 0]
        x -= step
        if np.max(np.abs(step)) < 1e-11:
            break
    return x.reshape(raw.shape)
