import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _profile(tr, times):
    def curve(rate):
        return -np.expm1(-rate * times)

    def error(rate):
        f = curve(rate)
        return np.mean((tr - np.outer(tr @ f / (f @ f), f)) ** 2)

    rate = minimize_scalar(error, bounds=(0.1, 3.0), method='bounded').x
    f = curve(rate)
    return tr @ f / (f @ f), f


def _gp(x, xq, v, vq, target, kind):
    d, dq = _aff(x), _aff(xq)
    tdist = (v[:, None, :] - v[None, :, :]) ** 2
    qdist = (vq[:, None, :] - v[None, :, :]) ** 2
    n, width = v.shape
    p = d.shape[1]
    eye = np.eye(n)
    rhs = np.column_stack((d, target))

    def cov(dist, h):
        q = dist @ np.exp(h[:-1])
        if kind == 'gaussian':
            return np.exp(-q / 2)
        if kind == 'rational':
            return 1 / (1 + q / 2)
        r = np.sqrt(q)
        return (1 + r + q / 3) * np.exp(-r)

    def fitted(h):
        c = cho_factor(cov(tdist, h) + np.exp(h[-1]) * eye)
        f = cho_solve(c, rhs)
        a = d.T @ f[:, :p]
        beta = np.linalg.solve(a, d.T @ f[:, p:])
        alpha = f[:, p:] - f[:, :p] @ beta
        return c, a, beta, alpha

    def loss(h):
        c, a, beta, alpha = fitted(h)
        r = target - d @ beta
        return ((n - p) * np.linalg.slogdet(r.T @ alpha)[1]
                + target.shape[1] * (2 * np.log(np.diag(c[0])).sum() + np.linalg.slogdet(a)[1])
                + width * np.var(h[:-1]))

    h = minimize(loss, np.r_[np.full(width, -.2), -7.5], method='L-BFGS-B',
                 bounds=[(-5, 3)] * width + [(-14, -2)], options={'maxiter': 85}).x
    _, _, beta, alpha = fitted(h)
    return dq @ beta + cov(qdist, h) @ alpha


def solve_batch(study):
    train = study['train']
    x, xq = train['x'], study['test']['x']
    z, f = _profile(train['traj'], study['eval_times'])
    target = np.column_stack((train['y'], z))
    panel = study['panel_b']
    op = np.linalg.lstsq(_aff(panel['u']), panel['v'], rcond=None)[0]
    inv = np.linalg.inv(op[1:])
    v, vq = x @ inv, xq @ inv
    pm = _gp(x, xq, v, vq, target, 'matern')
    pr = _gp(x, xq, v, vq, target, 'rational')
    pg = _gp(x, xq, v, vq, target, 'gaussian')
    y, zq = (0.5 * pm + 0.25 * pr + 0.25 * pg).T
    state = np.outer(zq, f)
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
