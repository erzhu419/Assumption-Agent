"""Cyclic coordinate calibration and shared two-mode relaxation forecasts."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _next(x):
    return np.roll(x, -1, axis=1)


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    w = []
    for a, b, y in zip(u.T, _next(u).T, v.T):
        f = np.column_stack((np.ones(len(a)), a, b, a*b, a**3))
        # Least-trimmed pilot: the coordinate law is overdetermined even
        # after discarding the largest residuals in each channel.
        p = _ls(f, y)
        k = max(f.shape[1]+1, int(.8*len(a)))
        for _ in range(8):
            keep = np.argsort(abs(f@p-y))[:k]
            p = _ls(f[keep], y[keep])
        r = abs(f@p-y)
        keep = r <= max(7*np.median(r), 1e-6)
        w.append(_ls(f[keep], y[keep]) if keep.sum() >= f.shape[1] else p)
    return np.asarray(w).T


def _inverse(x, w):
    c, a, b, p, d = w
    z = (x-c)/a
    j = np.arange(x.shape[1]); J = np.zeros((len(x), len(j), len(j)))
    for _ in range(20):
        nxt = _next(z)
        r = c+a*z+b*nxt+p*z*nxt+d*z**3-x
        if np.max(abs(r), initial=0.) < 1e-11:
            break
        J[:,j,j] = a+p*nxt+3*d*z*z
        J[:,j,(j+1)%len(j)] = b+p*z
        step = np.linalg.solve(J, r[...,None])[...,0]
        z -= np.clip(step, -.5, .5)
    return z


def _basis(t, traj):
    def basis(k):
        return -np.expm1(-t[:,None]*k)
    def err(k):
        B = basis(k)
        return (B@_ls(B, traj.T)-traj.T).ravel()
    rates = least_squares(err, [1.8,.4], bounds=([.8,.03],[8.,.79])).x
    return basis(rates)


def _design(x, kind):
    i, j = np.triu_indices(x.shape[1])
    s = {'rat': lambda q:q/(1+q*q), 'tanh':np.tanh,
         'gauss':lambda q:q*np.exp(-q*q)}[kind](x)
    return np.column_stack((np.ones(len(x)), x, x[:,i]*x[:,j], s,
                            (x[:,:,None]*s[:,None,:]).reshape(len(x), x.shape[1]**2)))


def _predict(x, q, target, kind, penalty):
    f, g = _design(x, kind), _design(q, kind)
    reg = np.eye(f.shape[1])*np.sqrt(penalty)
    return g@_ls(np.r_[f,reg], np.r_[target, np.zeros((len(reg),target.shape[1]))])


def solve_batch(study: dict) -> dict:
    train = study['train']
    w = _calibration(study['panel_a'])
    x, q = [_inverse(np.asarray(m['x'], dtype=float), w) for m in (train, study['test'])]
    B = _basis(np.asarray(study['eval_times']), np.asarray(train['traj']))
    target = np.column_stack((train['y'], _ls(B,np.asarray(train['traj']).T).T))
    result = (.55*_predict(x,q,target,'rat',.02) + .25*_predict(x,q,target,'gauss',.02)
              + .2*_predict(x,q,target,'tanh',.02))
    y, z = result[:,0], result[:,1:]@B.T
    return {'y_hat':y, 'state_pred':z,
            'diagnostics_vec':np.array([y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()])}
