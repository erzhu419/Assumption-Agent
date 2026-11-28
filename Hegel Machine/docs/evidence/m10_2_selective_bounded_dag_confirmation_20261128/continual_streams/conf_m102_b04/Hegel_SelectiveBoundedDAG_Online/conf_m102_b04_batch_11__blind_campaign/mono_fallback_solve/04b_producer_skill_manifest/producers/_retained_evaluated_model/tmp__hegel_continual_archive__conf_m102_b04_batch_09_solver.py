"""Nonlinear panel calibration and panel-conditioned trajectory regression."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares, minimize, minimize_scalar


def aff(x):
    return np.column_stack((np.ones(len(x)), x))


def ls(d, y):
    return np.linalg.lstsq(d, y, rcond=None)[0]


def _profile(u, v, design, init, bounds):
    def residual(k):
        f = design(u, k)
        return (f @ ls(f, v) - v).ravel()
    k = least_squares(residual, init, bounds=bounds, max_nfev=120).x
    return k, ls(design(u, k), v)


def _coordinates(study, x):
    a = study['panel_a']
    u, v = a['u'], a['v']
    keep = np.linalg.norm(v-u, axis=1) < 4
    u, v = u[keep], v[keep]
    a_design = lambda p, k: np.c_[aff(p), np.sin(k[0]*p)]
    ka, wa = _profile(u, v, a_design, [2.4], ([1.5], [3.2]))
    w0 = ls(aff(u), v)
    q = (x-w0[0]) @ np.linalg.inv(w0[1:])
    inv = np.linalg.inv(wa[1:5])
    for _ in range(30):
        q = (x-wa[0]-np.sin(ka[0]*q) @ wa[5:]) @ inv
    c = study['panel_c']
    c_design = lambda p, k: aff(np.arctan(k[0]*p+k[1:]))
    kc, wc = _profile(c['u'], c['v1'], c_design, [2.15, 0, 0, 0, 0],
                      ([.5, -1, -1, -1, -1], [5, 1, 1, 1, 1]))
    return np.c_[q, c_design(q, kc) @ wc]


def gp(x, xt, y, additive=False):
    mu, sd = y.mean(), max(y.std(), 1e-12)
    yy = (y-mu)/sd
    n, d = x.shape
    sq = (x[:,None,:]-x[None,:,:])**2
    dsq = (xt[:,None,:]-x[None,:,:])**2
    def kernel(dist, th):
        w = np.exp(th[:d]); e = -.5*dist*w
        per = np.exp(e) if additive else None
        k = np.exp(e.sum(2))
        if additive:
            extra = np.exp(th[d])*per/d
            k = k + extra.sum(2)
        return k, per if not additive else extra, e
    def objective(th):
        k, extra, e = kernel(sq, th)
        noise = np.exp(th[-1]); k.flat[::n+1] += noise
        fac = cho_factor(k, check_finite=False)
        al = cho_solve(fac, yy, check_finite=False)
        g = .5*(cho_solve(fac, np.eye(n), check_finite=False)-np.outer(al,al))
        joint = k.copy(); joint.flat[::n+1] -= noise
        if additive:
            joint -= extra.sum(2)
        grad = np.einsum('ij,ijk->k', g, e*(joint[:,:,None]+(extra if additive else 0.)))
        if additive:
            grad = np.r_[grad,np.sum(g*extra.sum(2))]
        return .5*yy@al+np.log(np.diag(fac[0])).sum(), np.r_[grad, noise*np.trace(g)]
    th = minimize(objective, np.r_[np.zeros(d),[-2.] if additive else [],-9.], jac=True,
                  method='L-BFGS-B', bounds=[(-11,7)]*d+([(-10,5)] if additive else [])+[(-15,-2)]).x
    k = kernel(sq,th)[0]; k.flat[::n+1] += np.exp(th[-1])
    return mu+sd*(kernel(dsq,th)[0]@cho_solve(cho_factor(k),yy))


def solve_batch(study):
    train, test = study['train'], study['test']
    n = len(train['x'])
    features = _coordinates(study, np.vstack((train['x'], test['x'])))
    features = (features-features[:n].mean(axis=0))/np.maximum(features[:n].std(axis=0), 1e-12)
    x, x_test = features[:n], features[n:]
    z = train['traj']
    modes = np.linalg.svd(z, full_matrices=False)[2][:2]
    amplitudes = z @ modes.T
    y_hat = gp(x, x_test, train['y'], True)
    state_pred = np.column_stack([gp(x, x_test, a, True) for a in amplitudes.T]) @ modes
    diagnostics_vec = np.array([y_hat.mean(), y_hat.std(), state_pred[:, -1].mean(), state_pred[:, -1].std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics_vec}
