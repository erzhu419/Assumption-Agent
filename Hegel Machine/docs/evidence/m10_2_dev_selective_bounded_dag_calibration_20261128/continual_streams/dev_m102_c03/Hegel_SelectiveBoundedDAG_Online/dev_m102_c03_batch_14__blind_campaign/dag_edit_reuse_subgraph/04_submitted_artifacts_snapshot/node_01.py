"""Calibrate latent coordinates and regress on the paired panel responses."""
from itertools import combinations
import numpy as np
from scipy.optimize import minimize_scalar


def _design(*arrays):
    return np.column_stack((np.ones(len(arrays[0])), *arrays))


def _fit(h, y):
    return np.linalg.lstsq(h, y, rcond=None)[0]


def _latent(panel, x):
    u, v = panel['u'], panel['v']
    # Isolate grossly corrupted calibration records before fitting the map.
    norm = np.linalg.norm(v, axis=1)
    good = norm < max(4., 3. * np.median(norm))
    u, v = u[good], v[good]
    j = np.arange(u.shape[1]); opposite = (j + 2) % len(j)

    def profile(w, coefficients=False):
        sx = np.sin(w * u)
        mats = [_design(u[:, i], sx[:, i], sx[:, k]) for i, k in zip(j, opposite)]
        fits = [_fit(h, v[:, i]) for i, h in enumerate(mats)]
        if coefficients:
            return np.array(fits)
        return sum(np.sum((h @ b - v[:, i])**2) for i, (h, b) in enumerate(zip(mats, fits)))

    w = minimize_scalar(profile, bounds=(1.8, 3.), method='bounded').x
    bias, gain, local, cross = profile(w, True).T
    z = (x - bias) / gain
    for _ in range(30):
        sx = np.sin(w * z)
        cx = w * np.cos(w * z)
        r = bias + gain*z + local*sx + cross*sx[:, opposite] - x
        if np.max(np.abs(r), initial=0.) < 1e-11:
            break
        jac = np.zeros((len(x), len(j), len(j)))
        jac[:, j, j] = gain + local*cx
        jac[:, j, opposite] = cross*cx[:, opposite]
        z -= np.clip(np.linalg.solve(jac, r[..., None])[..., 0], -.35, .35)
    return z


def _branches(panel, z):
    u = panel['u']
    i, j = np.array(list(combinations(range(u.shape[1]), 2))).T
    funcs = (
        lambda x, w: np.sin(w*x), lambda x, w: np.cos(w*x)-1.,
        lambda x, w: np.tanh(w*x), lambda x, w: np.arctan(w*x),
        lambda x, w: np.expm1(-w*x*x), lambda x, w: x/(1.+w*x*x),
        lambda x, w: x*x/(1.+w*x*x),
    )
    grid = np.linspace(.25, 5., 40)
    out = []
    for key in ('v1', 'v2'):
        v = panel[key]
        best = np.inf
        for f in funcs:
            for form in range(3):
                def design(x, w):
                    base = [x]
                    if form:
                        base.append(x[:, i]*x[:, j])
                    if form == 2:
                        base.append(x*x)
                    return _design(*base, f(x, w))

                def score(w):
                    h = design(u, w)
                    p = np.linalg.pinv(h)
                    residual = v - h @ (p @ v)
                    lev = np.einsum('ij,ji->i', h, p)
                    return np.mean((residual / np.maximum(1.-lev[:, None], 1e-8))**2)

                center = grid[np.argmin([score(w) for w in grid])]
                fit = minimize_scalar(score, bounds=(max(.15, center-.13), center+.13), method='bounded')
                if fit.fun < best:
                    best = fit.fun
                    branch = design(z, fit.x) @ _fit(design(u, fit.x), v)
        out.append(branch)
    return np.column_stack(out)


def run_node(inputs: dict) -> dict:
    study, sparse = inputs['__input__'], inputs['node_00']
    tr = study['train']; n = len(tr['x'])
    x = np.concatenate((tr['x'], study['test']['x']))
    f = _branches(study['panel_c'], _latent(study['panel_a'], x))
    f = (f - f[:n].mean(axis=0)) / np.maximum(f[:n].std(axis=0), 1e-8)
    h = f[:n]
    k = (1. + (h @ h.T)/h.shape[1])**2
    g = (1. + (f[n:] @ h.T)/h.shape[1])**2
    targets = np.column_stack((tr['y'], tr['traj']))
    pred = g @ np.linalg.solve(k + .001*np.eye(n), targets)
    prior = np.column_stack((sparse['y_hat'], sparse['state_pred']))
    blend = np.r_[.5, np.full(prior.shape[1]-1, .35)]
    return {'panel_pred': (pred - blend*prior)/(1.-blend)}
