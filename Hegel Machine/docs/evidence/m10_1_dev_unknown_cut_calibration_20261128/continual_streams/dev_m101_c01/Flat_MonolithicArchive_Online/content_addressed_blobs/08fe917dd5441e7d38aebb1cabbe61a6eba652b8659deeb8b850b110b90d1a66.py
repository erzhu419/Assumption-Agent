from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import nnls
from scipy.spatial.distance import cdist


def _poly(x, degree):
    return np.column_stack([np.ones(len(x))] + [
        x[:, inds].prod(axis=1) for d in range(1, degree + 1)
        for inds in combinations_with_replacement(range(x.shape[1]), d)])


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _coordinates(study, x):
    p = study['panel_a']
    u, v = p['u'], p['v']
    keep = np.linalg.norm(v - u, axis=1) < 2
    m = _ls(_poly(u[keep], 1), v[keep])
    a = np.linalg.solve(m[1:].T, (x - m[0]).T).T
    p = study['panel_b']
    m = _ls(_poly(p['u'], 1), p['v'])
    return x, a, _poly(a, 1) @ m, np.linalg.solve(m[1:].T, (a - m[0]).T).T


def _kernels(study, x, n):
    raw, a, b, bi = _coordinates(study, x)
    p = _poly(a, 2)
    base = p @ p.T
    eye = np.eye(len(x))
    result = {'quadratic': base + .002 * eye}
    for name, u in [('raw', raw), ('a', a), ('b', b), ('bi', bi)]:
        d = cdist(u, u, 'sqeuclidean')
        for length in (1.3, 2., 3.):
            r = np.sqrt(5*d) / length
            result[f'{name}-mat-{length}'] = base + (1+r+r*r/3)*np.exp(-r) + .0004*eye
            g = np.exp(-d / (2 * length**2))
            result[f'{name}-dot-{length}'] = base + .2 * g * (1 + u @ u.T) + .001*eye
        q = _poly(u, 2)
        pairs = np.column_stack([u[:, i] + sign*u[:, j] for i in range(4)
                                 for j in range(i+1, 4) for sign in (-1, 1)])
        for stem, v in [('rat', 1/(1+.5*pairs**2)), ('sin', np.sin(pairs)), ('tanh', np.tanh(pairs))]:
            f = (v[:, :, None]*q[:, None, :]).reshape(len(x), -1)
            f /= np.sqrt(np.mean(np.sum(f[:n]**2, axis=1)))
            for scale in ((.5, 5.) if stem == 'rat' else (.05, .5)):
                result[f'{name}-{stem}-{scale}'] = base + scale*(f@f.T) + .001*eye
        q3 = _poly(u, 3)[:, q.shape[1]:]
        result[f'{name}-cubic'] = base + .08*(q3@q3.T) + .002*eye
    return result


def _modes(times, z, telemetry):
    rates = np.asarray(telemetry)[:2]
    basis = -np.expm1(-times[:, None]*rates)
    # The leading empirical mode is kept; project only its noise-like component
    # away from the shared time law when the time law fits the mode closely.
    mode = np.linalg.svd(z, full_matrices=False)[2][0]
    smooth = basis @ _ls(basis, mode)
    if np.linalg.norm(mode - smooth) < .008:
        mode = smooth / np.linalg.norm(smooth)
    return mode


def _bank(study, x, xt, y, z):
    n = len(x)
    mode = _modes(np.asarray(study['eval_times']), z, study['telemetry_vec'])
    target = np.column_stack([y, z @ mode])
    loo, pred = {}, {}
    for name, k in _kernels(study, np.vstack([x, xt]), n).items():
        factor = cho_factor(k[:n, :n])
        coef = cho_solve(factor, target)
        precision_diag = np.diag(cho_solve(factor, np.eye(n)))
        loo[name] = target - coef / precision_diag[:, None]
        pred[name] = k[n:, :n] @ coef
    return target, loo, pred, mode


def _combine(target, loo, pred, j):
    names = list(loo)
    l = np.column_stack([loo[k][:, j] for k in names])
    p = np.column_stack([pred[k][:, j] for k in names])
    e = l - target[:, j, None]
    err = np.mean(np.abs(e), axis=0)
    ref = np.quantile(err, .15)
    prior = np.exp(-12 * (err / ref - 1))
    prior /= prior.sum()
    # Constrain the ensemble to a convex combination, regularized toward the
    # individually competitive predictors to avoid fitting LOO residual noise.
    penalty = ref * np.sqrt(len(target)) * .32
    w = nnls(np.vstack([e, penalty*np.eye(len(names)), np.full((1,len(names)), 10.)]),
             np.r_[np.zeros(len(target)), penalty*prior, 10.])[0]
    return p @ (w/w.sum())


def solve_batch(study: dict) -> dict:
    tr = study['train']
    target, loo, pred, mode = _bank(study, tr['x'], study['test']['x'], tr['y'], tr['traj'])
    y = .7 * pred['a-rat-5.0'][:, 0] + .3 * pred['bi-dot-1.3'][:, 0]
    pair_amp = (.4 * pred['a-tanh-0.5'][:, 1] + .3 * pred['a-sin-0.5'][:, 1]
                + .3 * pred['b-tanh-0.5'][:, 1])
    amp = .55 * _combine(target, loo, pred, 1) + .45 * pair_amp
    state = amp[:, None]*mode
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
