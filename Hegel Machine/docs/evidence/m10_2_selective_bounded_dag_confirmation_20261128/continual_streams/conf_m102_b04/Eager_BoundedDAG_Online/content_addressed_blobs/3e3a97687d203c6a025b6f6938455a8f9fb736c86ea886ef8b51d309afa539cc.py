"""Invert the panel calibration, then fit separate scalar and trajectory heads."""
from itertools import combinations_with_replacement
import numpy as np


def _calibration(u, v, j):
    # The calibration couples each coordinate to its cyclic successor.
    nxt = np.roll(u, -1, axis=1)
    f = np.stack((np.ones_like(u), u, nxt, u * nxt, u**3), axis=-1)
    def fit(keep):
        return np.column_stack([np.linalg.lstsq(f[keep, k], v[keep, k], rcond=None)[0] for k in range(u.shape[1])])
    def residual(a):
        return np.linalg.norm(np.einsum('nkp,pk->nk', f, a) - v, axis=1)
    n = len(u)
    m = max(5, int(.82 * n))
    best = None
    for keep in (np.arange(n), j[:m]):
        for _ in range(10):
            a = fit(keep)
            new = np.sort(np.argsort(residual(a))[:m])
            if np.array_equal(new, keep):
                break
            keep = new
        e = residual(a)
        score = np.mean(np.sort(e)[:m]**2)
        if best is None or score < best[0]:
            best = score, a
    a = best[1]
    return a if np.median(residual(a)) <= .015 * max(np.std(v), 1e-8) else None


def _inverse(x, q, u, v):
    f = np.column_stack((np.ones(len(u)), u))
    affine = np.linalg.lstsq(f, v, rcond=None)[0]
    order = np.argsort(np.linalg.norm(f @ affine - v, axis=1))
    a = _calibration(u, v, order)
    if a is None:
        return None
    d = x.shape[1]
    k = np.arange(d)
    nxt = (k + 1) % d
    def error(w):
        z = w[:, nxt]
        return a[0] + a[1]*w + a[2]*z + a[3]*w*z + a[4]*w**3 - x
    jac = np.zeros((len(x), d, d))
    try:
        jac[:, k, k] = a[1]
        jac[:, k, nxt] = a[2]
        w = np.linalg.solve(jac[0], (x-a[0]).T).T
        use_q = np.linalg.norm(error(q), axis=1) < np.linalg.norm(error(w), axis=1)
        w[use_q] = q[use_q]
        for _ in range(24):
            r = error(w)
            if np.max(np.abs(r)) <= 1e-10:
                return w
            jac[:, k, k] = a[1] + a[3]*w[:, nxt] + 3*a[4]*w*w
            jac[:, k, nxt] = a[2] + a[3]*w
            step = np.linalg.solve(jac, r[..., None])[..., 0]
            norm = np.linalg.norm(r, axis=1)
            for j in range(12):
                trial = w - step * (0.5**j)
                good = np.linalg.norm(error(trial), axis=1) < norm
                if j == 0:
                    update = w.copy()
                    pending = np.ones(len(w), dtype=bool)
                good &= pending
                update[good] = trial[good]
                pending[good] = False
                if not pending.any():
                    break
            w = update
        if np.max(np.abs(error(w))) <= 1e-7:
            return w
    except np.linalg.LinAlgError:
        pass
    return None


def _basis(w, degree, fallback=False):
    cols = [np.ones(len(w))]
    for d in range(1, degree+1):
        for j in combinations_with_replacement(range(w.shape[1]), d):
            weight = 0.5**(len(set(j))-1 + 0.5*max(d-2, 0)) if fallback else 1.0
            cols.append(weight * np.prod(w[:, j], axis=1))
    return np.column_stack(cols)


def _ridge(f, y, penalty):
    n = len(y)
    a, b = f[:n], f[n:]
    if a.shape[1] <= n:
        return b @ np.linalg.solve(a.T @ a + penalty*np.eye(a.shape[1]), a.T @ y)
    return (b @ a.T) @ np.linalg.solve(a @ a.T + penalty*np.eye(n), y)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    upstream = inputs['node_00']
    train, test = study['train'], study['test']
    x = np.vstack((train['x'], test['x']))
    start = upstream['aligned'].shape[1] + 1
    q = upstream['hybrid'][:, start:start+x.shape[1]]
    panel = study['panel_a']
    w = _inverse(x, q, panel['u'], panel['v'])
    traj = train['traj']
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    if w is None:
        y = _ridge(_basis(q / np.sqrt(1 + .60*q*q), 6, True), train['y'], .0003)
        state = _ridge(_basis(q / np.sqrt(1 + .85*q*q), 6, True), traj @ modes.T, .00146) @ modes
    else:
        y = _ridge(_basis(w, 2), train['y'], .0015)
        state = _ridge(_basis(w, 1), traj @ modes.T, .0015) @ modes
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([
        y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
