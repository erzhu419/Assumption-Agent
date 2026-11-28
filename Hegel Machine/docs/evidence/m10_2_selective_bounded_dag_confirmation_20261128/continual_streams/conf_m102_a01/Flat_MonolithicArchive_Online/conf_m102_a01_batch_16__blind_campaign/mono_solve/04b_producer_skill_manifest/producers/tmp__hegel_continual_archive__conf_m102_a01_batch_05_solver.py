"""Panel-calibrated smooth regression with a shared temporal mode."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _coordinates(panel, x):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    ii, jj = np.triu_indices(len(u), 1)
    z = x.copy()
    for j in range(x.shape[1]):
        a, b = u[:, j], v[:, j]
        da = a[ii] - a[jj]
        ok = np.abs(da) > .08
        gain = np.median((b[ii[ok]] - b[jj[ok]]) / da[ok]) if ok.any() else 1.
        offset = np.median(b - gain * a)
        error = np.abs(b - offset - gain * a)
        good = error <= max(.001, 4 * np.median(error))
        if np.count_nonzero(good) >= 2 and np.ptp(a[good]) > 0:
            offset, gain = np.linalg.lstsq(np.column_stack((np.ones(good.sum()), a[good])), b[good], rcond=None)[0]
        z[:, j] = (x[:, j] - offset) / (gain if abs(gain) > 1e-10 else 1.)
    return z


def _fixed(w, target, length, nugget=1e-5):
    n = len(target)
    g = w @ w[:n].T / w.shape[1]
    d = ((w[:, None] - w[None, :n]) ** 2).sum(axis=-1)
    k = 1 + g + .15 * np.exp(-.5 * d / length**2)
    return k[n:] @ cho_solve(cho_factor(k[:n] + nugget * np.eye(n)), target)


def _adaptive(w, target, ard=True):
    n = len(target)
    base = 1 + w @ w[:n].T / w.shape[1]
    d = (w[:, None] - w[None, :n]) ** 2
    di = d[:n]
    eye = np.eye(n)
    target = np.asarray(target)
    scale = max(np.sqrt(np.mean(target**2)), 1e-12)
    y = target / scale

    def loss(p):
        a = np.exp(p[2:])
        exp_arg = -.5 * di * a
        r = np.exp(p[0] + exp_arg.sum(axis=2))
        noise = np.exp(p[1]) * eye
        cf = cho_factor(base[:n] + r + noise)
        v = cho_solve(cf, y)
        ss = max(y @ v / n, 1e-14)
        score = .5 * n * np.log(ss) + np.log(np.diag(cf[0])).sum()
        jac_k = .5 * (cho_solve(cf, eye) - np.outer(v, v) / ss)
        grad_a = np.einsum('ij,ij,ijk->k', jac_k, r, exp_arg)
        if ard:
            delta = p[2:] - p[2:].mean()
            score += .6 * (delta @ delta)
            grad_a += 1.2 * delta
        else:
            grad_a = np.array([grad_a.sum()])
        return score, np.r_[np.sum(jac_k * r), np.sum(jac_k * noise), grad_a]

    m = w.shape[1] if ard else 1
    p0 = [-2., -9.] + [-1.8] * m
    fit = minimize(loss, p0, jac=True, method='L-BFGS-B', bounds=[(-10, 5), (-15, -2)] + [(-5, 2)] * m)
    p = fit.x if np.isfinite(fit.fun) else np.asarray(p0)
    k = base + np.exp(p[0] - .5 * (d * np.exp(p[2:])).sum(axis=2))
    return scale * (k[n:] @ cho_solve(cho_factor(k[:n] + np.exp(p[1]) * eye), y))


def _continuation(z, target, length, weight, nugget):
    n = len(target)
    w = z / max(np.sqrt(np.mean(z[:n]**2)), .1)
    g = w @ w[:n].T / w.shape[1]
    d = ((w[:, None] - w[None, :n])**2).sum(axis=2)
    k = 1 + g + g**2 + weight * np.exp(-.5 * d / length**2)
    return k[n:] @ cho_solve(cho_factor(k[:n] + nugget * np.eye(n)), target)


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    y = np.asarray(train['y'], dtype=float)
    traj = np.asarray(train['traj'], dtype=float)
    x = np.concatenate((train['x'], test['x'])).astype(float)
    z = _coordinates(study['panel_a'], x)
    n = len(y)
    w = (z - z[:n].mean(axis=0)) / np.maximum(z[:n].std(axis=0), 1e-10)
    y_hat = .75 * _fixed(w, y, 2.2) + .25 * _adaptive(w, y)
    _, _, vt = np.linalg.svd(traj, full_matrices=False)
    mode = vt[0]
    a = traj @ mode
    amp = .7 * _adaptive(w, a) + .3 * _fixed(w, a, 5.)
    # Favor the explicit quadratic trend where a query leaves training support.
    excess = np.maximum(w[n:] - w[:n].max(axis=0), w[:n].min(axis=0) - w[n:])
    edge = np.clip(excess.max(axis=1) / .65, 0., 1.)
    y_mix = .08 + .50 * edge
    a_mix = .12 + .40 * edge
    y_hat += y_mix * (_continuation(z, y, 3., .03, 1e-5) - y_hat)
    amp += a_mix * (_continuation(z, a, 2., .3, .0002) - amp)
    state_pred = amp[:, None] * mode
    diagnostics_vec = np.array([y_hat.mean(), y_hat.std(), state_pred[:, -1].mean(), state_pred[:, -1].std()])
    return dict(y_hat=y_hat, state_pred=state_pred, diagnostics_vec=diagnostics_vec)
