"""Invert the reference-panel calibration and apply the panel-B operator."""
from itertools import combinations_with_replacement

import numpy as np


def _design(x, powers):
    return np.prod(x[:, None, :] ** powers[None, :, :], axis=2)


def _invert(x, coef, powers):
    d = x.shape[1]
    linear = coef[1:d+1]
    z = np.linalg.solve(linear.T, (x-coef[0]).T).T
    derivative_powers = [np.maximum(powers - np.eye(d, dtype=int)[j], 0) for j in range(d)]
    derivative_coefs = [powers[:, j, None] * coef for j in range(d)]
    for _ in range(30):
        residual = _design(z, powers) @ coef - x
        err = np.linalg.norm(residual, axis=1)
        if err.max(initial=0) < 1e-11:
            break
        jac = np.stack([_design(z, p) @ c for p, c in zip(derivative_powers, derivative_coefs)], axis=2)
        delta = np.linalg.solve(jac, residual[..., None])[..., 0]
        for j in range(14):
            trial = z - delta * (0.5 ** j)
            e = np.linalg.norm(_design(trial, powers) @ coef - x, axis=1)
            take = e < err
            if np.any(take):
                z[take] = trial[take]
                delta[take] = 0
                err[take] = e[take]
            if not np.any(delta):
                break
    return z


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    u, v = (np.asarray(study['panel_a'][k], float) for k in ('u', 'v'))
    d = u.shape[1]
    powers = np.array([np.bincount(term, minlength=d) for degree in range(4)
                       for term in combinations_with_replacement(range(d), degree)])
    displacement = np.linalg.norm(v-u, axis=1)
    keep = displacement <= max(4.0, 5.0 * np.median(displacement))
    design = _design(u[keep], powers)
    coef = np.linalg.lstsq(design, v[keep], rcond=None)[0]
    # Compact to the shared polynomial support when omitted terms are at noise scale.
    active = np.max(np.abs(coef), axis=1) >= 0.005
    active[:d+1] = True
    compact = np.linalg.lstsq(design[:, active], v[keep], rcond=None)[0]
    if np.max(np.abs(design[:, active] @ compact - v[keep])) < 0.001:
        coef, powers = compact, powers[active]
    x, xt = (np.asarray(study[k]['x'], float) for k in ('train', 'test'))
    z = _invert(np.vstack((x, xt)), coef, powers)
    ub, vb = (np.asarray(study['panel_b'][k], float) for k in ('u', 'v'))
    b = np.linalg.lstsq(np.c_[np.ones(len(ub)), ub], vb, rcond=None)[0]
    z = z @ b[1:] + b[0]
    return {'z': z[:len(x)].copy(), 'zt': z[len(x):].copy()}
