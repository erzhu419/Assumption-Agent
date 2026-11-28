"""Fit calibrated panel coordinates and two shared trajectory modes."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares, minimize


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-11)[0]


def _coordinates(study, x):
    u, v = (study['panel_a'][k] for k in ('u', 'v'))
    design = [np.column_stack((np.ones(len(u)), ui)) for ui in u.T]
    mask = np.ones(len(u), dtype=bool)
    for _ in range(4):
        shift, scale = np.column_stack([_ls(a[mask], b[mask]) for a, b in zip(design, v.T)])
        error = np.linalg.norm(v - shift - u*scale, axis=1)
        mask = error <= max(0.001, 4*np.median(error))
    u, v = (study['panel_c'][k] for k in ('u', 'v2'))
    def panel(z, k):
        return np.column_stack((np.ones(len(z)), z, np.hypot(1, k*z)-1))
    def resid(k):
        g = panel(u, k[0]); return (g@_ls(g, v)-v).ravel()
    k = least_squares(resid, [1.], bounds=(0.2, 3.)).x[0]
    return panel((x-shift)/scale, k) @ _ls(panel(u, k), v)


def _mode_fit(t, trajectories):
    def basis(rates):
        return -np.expm1(-t[:, None]*rates)
    def resid(rates):
        g = basis(rates); return (g @ _ls(g, trajectories.T) - trajectories.T).ravel()
    g = basis(least_squares(resid, [1.8, .42], bounds=([1., .05], [4., .95])).x)
    return _ls(g, trajectories.T).T, g.T


def _predict(x, xt, y):
    center, scale = x.mean(0), x.std(0)
    x, xt = (x-center)/scale, (xt-center)/scale
    mean, std = y.mean(), y.std()
    y = (y-mean)/std
    n, d = x.shape
    identity = np.eye(n)
    train_dist = (x[:, None]-x)**2
    test_dist = (xt[:, None]-x)**2
    prior = np.log(4.)
    def kernel(p, dist):
        w = np.exp(-2*p[:d]); r2 = dist@w; r = np.sqrt(r2)
        e = np.exp(-r)
        return (1+r+r2/3)*e, (1+r)*e/3, w
    def objective(p):
        k, dg, w = kernel(p, train_dist)
        reg = np.exp(p[-1]); cf = cho_factor(k+reg*identity)
        a = cho_solve(cf, y); variance = y@a/n
        v = (cho_solve(cf, identity)-np.outer(a,a)/variance)/2
        dp = p[:d]-prior
        grad = np.einsum('ij,ij,ijk,k->k', v, dg, train_dist, w)+.3*dp
        loss = n*np.log(variance)/2+np.log(cf[0].diagonal()).sum()+.15*(dp@dp)
        return loss, np.r_[grad, reg*np.trace(v)]
    p = minimize(objective, np.r_[np.full(d, prior), -9.], jac=True, method='L-BFGS-B',
                 bounds=[(-1, 4)]*d+[(-16, -3)], options={'maxiter':100}).x
    k = kernel(p, train_dist)[0]+np.exp(p[-1])*identity
    return mean+std*(kernel(p,test_dist)[0]@cho_solve(cho_factor(k), y))


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    x, xt = np.split(_coordinates(study, np.vstack((train['x'], test['x']))), [len(train['x'])])
    amplitudes, basis = _mode_fit(study['eval_times'], train['traj'])
    pred = np.column_stack([_predict(x, xt, v) for v in np.column_stack((train['y'], amplitudes)).T])
    y, state = pred[:,0], pred[:,1:] @ basis
    last = state[:,-1]
    return dict(y_hat=y, state_pred=state, diagnostics_vec=np.array([y.mean(),y.std(),last.mean(),last.std()]))
