"""Predict scalar and temporal readouts from the calibrated panel-B operator."""
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    # Each channel is an independent gain and offset; panel outliers are sparse.
    coefs = []
    for u, v in zip(panel['u'].T, panel['v'].T):
        a = _affine(u[:, None])
        keep = np.arange(len(u))
        for _ in range(6):
            c = np.linalg.lstsq(a[keep], v[keep], rcond=None)[0]
            keep = np.argsort(np.abs(v - a @ c))[:len(u) * 5 // 6]
        coefs.append(np.linalg.lstsq(a[keep], v[keep], rcond=None)[0])
    return np.asarray(coefs).T


def _operator(panel):
    u, v = panel['u'], panel['v']
    candidates = []
    for fun, start, limits in ((np.sin, 1.6, (.25, 3.5)),
                               (np.tanh, 2.5, (.25, 6.0))):
        def design(x, rate):
            return np.column_stack((_affine(x), fun(x * rate)))

        def residual(rate):
            a = design(u, rate)
            return (a @ np.linalg.lstsq(a, v, rcond=None)[0] - v).ravel()

        rate = least_squares(residual, [start], bounds=limits).x[0]
        a = design(u, rate)
        coef = np.linalg.lstsq(a, v, rcond=None)[0]
        q = np.linalg.qr(a)[0]
        loo = (v - a @ coef) / (1.0 - np.sum(q*q, axis=1))[:, None]
        candidates.append((np.mean(loo**2), fun, rate, coef))
    _, fun, rate, coef = min(candidates, key=lambda item: item[0])
    return lambda x: np.column_stack((_affine(x), fun(x * rate))) @ coef


def _scalar_readout(z, zt, y):
    # Select two cross-channel products using only the supplied fit rows.
    pairs = list(combinations(range(z.shape[1]), 2))
    products = np.column_stack([z[:, i] * z[:, j] for i, j in pairs])
    linear = _affine(z)

    def loss(cols):
        design = np.column_stack((linear, products[:, cols]))
        residual = design @ _ridge(design, y, .001) - y
        return np.mean(residual**2)

    cols = min(combinations(range(len(pairs)), 2), key=loss)
    pt = np.column_stack([zt[:, pairs[k][0]] * zt[:, pairs[k][1]] for k in cols])
    p = products[:, cols]
    design = np.column_stack((linear, p, p**2))
    test_design = np.column_stack((_affine(zt), pt, pt**2))
    return test_design @ _ridge(design, y, .015)


def _ridge(a, y, penalty):
    extra = np.diag(np.r_[0., np.full(a.shape[1] - 1, np.sqrt(penalty))])
    return np.linalg.lstsq(np.vstack((a, extra)),
                           np.r_[y, np.zeros((len(extra),) + y.shape[1:])],
                           rcond=None)[0]


def solve_batch(study: dict) -> dict:
    train, xt = study['train'], study['test']['x']
    offset, gain = _calibration(study['panel_a'])
    panel_b = _operator(study['panel_b'])
    z, zt = [panel_b((x - offset) / gain) for x in (train['x'], xt)]
    y_hat = _scalar_readout(z, zt, train['y'])

    times, observed = study['eval_times'], train['traj'].T

    def basis(rates):
        return -np.expm1(-times[:, None] * rates)

    def residual(rates):
        b = basis(rates)
        return (b @ np.linalg.lstsq(b, observed, rcond=None)[0] - observed).ravel()

    rates = least_squares(residual, [1.8, .4], bounds=([.7, .03], [5., .68])).x
    modes = np.linalg.qr(basis(rates))[0]
    state_pred = _affine(zt) @ _ridge(_affine(z), observed.T @ modes, .001) @ modes.T
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])}
