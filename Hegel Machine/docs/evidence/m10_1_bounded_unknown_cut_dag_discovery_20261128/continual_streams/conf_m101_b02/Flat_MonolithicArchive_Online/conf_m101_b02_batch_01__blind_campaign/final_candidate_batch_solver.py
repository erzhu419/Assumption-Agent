"""Predict the response and trajectory from the calibrated panel operators."""

import numpy as np


def _affine_reference(panel):
    u, v = panel['u'], panel['v']
    coeffs = []
    for j in range(u.shape[1]):
        design = np.column_stack((np.ones(len(u)), u[:, j]))
        coeff = np.linalg.lstsq(design, v[:, j], rcond=None)[0]
        for _ in range(8):
            keep = np.argsort(np.abs(design @ coeff - v[:, j]))[:int(0.80 * len(u))]
            coeff = np.linalg.lstsq(design[keep], v[keep, j], rcond=None)[0]
        coeffs.append(coeff)
    return np.asarray(coeffs).T


def _shrink(x):
    return np.sign(x) * np.maximum(np.abs(x) - 0.35, 0.0)**1.5


def _odd(x):
    return -np.sign(x) * np.expm1(-1.8 * np.abs(x))


def _nodes(study, x):
    offset, scale = _affine_reference(study['panel_a'])
    a = (x - offset) / scale
    p = study['panel_b']
    z, v = _shrink(p['u']), p['v']
    z0, v0 = z.mean(axis=0), v.mean(axis=0)
    zc = z - z0
    left, singular, right = np.linalg.svd(zc.T @ (v - v0))
    mixing = (left @ right) * (singular.sum() / np.sum(zc**2))
    b = (_shrink(a) - z0) @ mixing + v0
    p = study['panel_c']
    coeff = np.linalg.lstsq(np.column_stack((np.ones(len(p['u'])), _odd(p['u']))), p['v1'], rcond=None)[0]
    c = _odd(b) @ coeff[1:] + coeff[0]
    return a, b, c


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']))
    a, _, c = _nodes(study, x)
    interaction = np.sum(a * c, axis=1)
    scalar_design = np.column_stack((np.ones(len(x)), a, c, interaction))
    state_design = np.column_stack((scalar_design, a[:, 1] * c[:, 0]))

    scalar_weights = np.linalg.lstsq(scalar_design[:n], train['y'], rcond=None)[0]
    y = scalar_design[n:] @ scalar_weights

    # Two saturating time modes capture the shared trajectory dynamics.
    t = np.asarray(study['eval_times'])
    modes = -np.expm1(-t[:, None] * np.array([1.8, 0.4]))
    amplitudes = np.linalg.lstsq(modes, train['traj'].T, rcond=None)[0].T
    state_weights = np.linalg.lstsq(state_design[:n], amplitudes, rcond=None)[0]
    state = (state_design[n:] @ state_weights) @ modes.T

    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
