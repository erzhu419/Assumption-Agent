"""Fit a shared nonlinear response for the endpoint and trajectory."""
from itertools import combinations_with_replacement
import numpy as np

def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _a_design(x, j):
    a, b = x[:, j], x[:, (j + 1) % 4]
    return np.column_stack((np.ones(len(x)), a, b, a*b, a**3))


def _fit_a(u, v):
    designs = [_a_design(u, j) for j in range(4)]
    weights = np.ones(len(u))
    for _ in range(20):
        coef = np.array([np.linalg.lstsq(d*weights[:, None], v[:, j]*weights, rcond=None)[0]
                         for j, d in enumerate(designs)])
        error = np.linalg.norm(np.column_stack([d@c for d, c in zip(designs, coef)]) - v, axis=1)
        weights = 1/np.hypot(1, error/max(np.median(error), 1e-8))
    keep = np.argsort(error)[:-3]
    coef = np.array([np.linalg.lstsq(d[keep], v[keep, j], rcond=None)[0]
                     for j, d in enumerate(designs)]).T
    inverse_seed = np.linalg.lstsq(_affine(v[keep]), u[keep], rcond=None)[0]
    return coef, inverse_seed


def _invert_a(v, coef, inverse_seed):
    bias, lin, neighbor, pair, cubic = coef
    x = _affine(v) @ inverse_seed
    j = np.arange(4)
    for _ in range(20):
        b = np.roll(x, -1, axis=1)
        residual = bias + lin*x + neighbor*b + pair*x*b + cubic*x**3 - v
        if np.max(np.abs(residual), initial=0) < 1e-11:
            break
        jac = np.zeros((len(x), 4, 4))
        jac[:, j, j] = lin + pair*b + 3*cubic*x*x
        jac[:, j, (j+1)%4] = neighbor + pair*x
        x -= np.clip(np.linalg.solve(jac, residual[..., None])[..., 0], -0.5, 0.5)
    return x


def _basis(x, k, family):
    if family == 'sin':
        h = np.sin(k*x)
    elif family == 'tanh':
        h = np.tanh(k*x)
    else:
        h = x*np.exp(-(k*x)**2)
    w = np.column_stack((x, h))
    return np.column_stack((h, *(w[:, i]*w[:, j]
                                for i, j in combinations_with_replacement(range(w.shape[1]), 2))))


def _readout(a, b, ref, reft, response, scale, families):
    n = len(a)
    baseline = np.linalg.lstsq(a, response, rcond=None)[0]
    residual = response - a @ baseline
    direction = np.linalg.svd(residual, full_matrices=False)[2][0]
    prediction = 0.
    for family, factor, penalty, weight in families:
        features = _basis(np.vstack((ref, reft)), scale*factor, family)
        features /= np.maximum(features[:n].std(axis=0), 1e-3)
        loading = np.linalg.lstsq(a, features[:n], rcond=None)[0]
        fa = features[:n] - a @ loading
        fb = features[n:] - b @ loading
        coef = np.linalg.solve(fa.T @ fa + penalty*np.eye(fa.shape[1]), fa.T @ (residual @ direction))
        prediction += weight*(fb @ coef)[:, None]*direction
    return b @ baseline + prediction


def run_node(inputs: dict) -> dict:
    study, latent = inputs['__input__'], inputs['node_00']
    train, test, panel = study['train'], study['test'], study['panel_a']
    coef, seed = _fit_a(panel['u'], panel['v'])
    ref = _invert_a(train['x'], coef, seed)
    reft = _invert_a(test['x'], coef, seed)
    a = _affine(np.column_stack((latent['z_tr'], ref)))
    b = _affine(np.column_stack((latent['z_te'], reft)))
    response = np.column_stack((train['y'], train['traj']))
    families = [('sin', 1.2, 0.03, 0.5), ('exp', 0.6, 0.015, 0.5)]
    prediction = _readout(a, b, ref, reft, response, study['telemetry_vec'][0], families)
    y_hat, state = prediction[:, 0], prediction[:, 1:]
    final = state[:, -1]
    return {'y_hat': y_hat, 'state_pred': state,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), final.mean(), final.std()])}
