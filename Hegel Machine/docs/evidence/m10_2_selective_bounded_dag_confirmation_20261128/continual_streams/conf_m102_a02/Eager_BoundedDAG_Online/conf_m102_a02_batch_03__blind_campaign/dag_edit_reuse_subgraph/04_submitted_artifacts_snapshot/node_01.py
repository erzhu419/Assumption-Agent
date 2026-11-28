from itertools import combinations_with_replacement
import numpy as np


def _powers(d):
    return [()]+[k for n in range(1, 4) for k in combinations_with_replacement(range(d), n)]


def _design(x, terms):
    return np.column_stack([np.prod(x[:, k], axis=1) for k in terms])


def _inverse(x, coeff, terms):
    d = x.shape[1]
    linear = coeff[1:d+1]
    z = (x-coeff[0]) @ np.linalg.pinv(linear)
    for _ in range(30):
        err = _design(z, terms) @ coeff - x
        if np.max(np.abs(err), initial=0.0) < 1e-10:
            break
        jac = np.zeros((len(x), d, d))
        for j, k in enumerate(terms):
            for l, col in enumerate(k):
                jac[:, :, col] += np.prod(z[:, k[:l]+k[l+1:]], axis=1)[:, None] * coeff[j]
        step = (np.linalg.pinv(jac) @ err[..., None])[..., 0]
        old = np.sum(err**2, axis=1)
        pending = np.ones(len(x), dtype=bool)
        for fraction in (1., .5, .25, .125, .0625, .03125):
            trial = z - fraction*step
            new = np.sum((_design(trial, terms) @ coeff - x)**2, axis=1)
            use = pending & (new < old)
            z[use] = trial[use]
            pending[use] = False
            if not pending.any():
                break
    return z


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    frozen = inputs['node_00']
    x = np.asarray(frozen['a'], dtype=float)[:, 1:]
    u = np.asarray(study['panel_a']['u'], dtype=float)
    v = np.asarray(study['panel_a']['v'], dtype=float)
    terms = _powers(x.shape[1])
    distance = np.linalg.norm(v-u, axis=1)
    med = np.median(distance)
    keep = distance <= med + 8*max(np.median(np.abs(distance-med)), 1e-8)
    if keep.sum() < len(terms):
        keep[np.argsort(distance)[:len(terms)]] = True
    panel_design = _design(u[keep], terms)
    coeff = np.linalg.lstsq(panel_design, v[keep], rcond=None)[0]
    z = _inverse(x, coeff, terms)
    features = _design(z, terms[:1+x.shape[1]+x.shape[1]*(x.shape[1]+1)//2])
    target = np.column_stack((study['train']['y'], study['train']['traj']))
    weights = np.linalg.lstsq(features, target, rcond=None)[0]
    return {'calibration_coeff': coeff, 'readout_coeff': weights}
