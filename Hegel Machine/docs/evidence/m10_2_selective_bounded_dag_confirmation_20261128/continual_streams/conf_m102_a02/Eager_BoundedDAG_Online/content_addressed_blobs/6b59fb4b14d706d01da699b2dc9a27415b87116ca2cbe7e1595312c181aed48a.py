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
    frozen = inputs['node_00']
    fitted = inputs['node_01']
    x = np.asarray(frozen['a_test'], dtype=float)[:, 1:]
    terms = _powers(x.shape[1])
    z = _inverse(x, fitted['calibration_coeff'], terms)
    weights = fitted['readout_coeff']
    pred = _design(z, terms[:weights.shape[0]]) @ weights
    y = pred[:, 0].copy()
    state = pred[:, 1:].copy()
    final = state[:, -1]
    diagnostics = np.array([y.mean(), y.std(), final.mean(), final.std()]) if y.size else np.zeros(4)
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
