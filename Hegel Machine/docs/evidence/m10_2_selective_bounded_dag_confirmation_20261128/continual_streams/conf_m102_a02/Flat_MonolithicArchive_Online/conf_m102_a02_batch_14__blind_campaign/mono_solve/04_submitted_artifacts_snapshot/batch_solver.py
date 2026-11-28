"""Calibrate latent inputs, then fit independent scalar and state readouts."""
import numpy as np
from scipy.optimize import least_squares


def _ls(f, y):
    return np.linalg.lstsq(f, y, rcond=None)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    r = np.linalg.norm(v - _aff(u) @ _ls(_aff(u), v), axis=1)
    keep = r < max(5*np.median(r), 2*np.quantile(r, .85))
    u, v = u[keep], v[keep]
    opp = (np.arange(u.shape[1])+2) % u.shape[1]

    def profile(p):
        h, k = np.tanh(p[0]*u), np.sin(p[1]*u[:, opp])
        a = np.stack((v-h*p[2]-k*p[3], u), axis=-1)
        a0, U = a[:, :, 0], a[:, :, 1]
        uc = U-U.mean(axis=0)
        slope = (uc*a0).sum(axis=0)/(uc**2).sum(axis=0)
        bias = np.mean(a0-U*slope, axis=0)
        return bias, slope, bias+U*slope+p[2]*h+p[3]*k-v

    p = least_squares(lambda p: profile(p)[2].ravel(), [1.8,2.4,.18,.32],
                      bounds=([.5,1.,0.,0.],[4.,4.,.7,.7])).x
    bias, slope, _ = profile(p)
    return p, bias, slope


def _inverse(x, calibration):
    p, bias, slope = calibration
    x = np.asarray(x, dtype=float)
    i = np.arange(x.shape[1]); j = (i+2) % len(i)
    z = (x-bias)/slope
    if len(x) == 0:
        return z
    jac = np.zeros((len(z),len(i),len(i)))
    for _ in range(35):
        h = np.tanh(p[0]*z)
        r = bias+slope*z+p[2]*h+p[3]*np.sin(p[1]*z[:,j])-x
        if np.max(abs(r)) < 1e-12:
            break
        jac[:,i,i] = slope+p[0]*p[2]*(1-h*h)
        jac[:,i,j] = p[1]*p[3]*np.cos(p[1]*z[:,j])
        z -= np.clip(np.linalg.solve(jac,r[...,None])[...,0], -.5, .5)
    return z


def _rational_panel(panel):
    u, v = panel['u'], panel['v1']
    def primitive(x, p):
        return x/(1+p[0]*abs(x)**p[1])
    def design(x,p):
        return np.column_stack((_aff(x),primitive(x,p)))
    def residual(p):
        f=design(u,p)
        return (f@_ls(f,v)-v).ravel()
    p=least_squares(residual,[1.,1.1],bounds=([.01,.3],[10.,5.])).x
    return lambda x: primitive(x,p)


def _features(x, op):
    i,j=np.triu_indices(x.shape[1])
    return np.column_stack((_aff(x),x[:,i]*x[:,j],op(x)))


def _operator(panel, z):
    def basis(x):
        return np.c_[_aff(x), x*np.exp(-3.22429526*abs(x)**3), np.tanh(1.64761629*x**3)]
    return basis(z) @ _ls(basis(panel['u']),panel['v'])

def _quadratic(x):
    i,j=np.triu_indices(x.shape[1]); return np.c_[_aff(x),x[:,i]*x[:,j]]

def _predict(a, q, y, penalty):
    scale=np.r_[1,np.maximum(a[:,1:].std(0),.035)]
    d=np.sqrt(penalty)*np.eye(a.shape[1]); d[0,0]=0
    return (q/scale) @ _ls(np.r_[a/scale,d],np.r_[y,np.zeros((len(d),)+y.shape[1:])])

def _dead_zone(panel, z):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    def basis(x, p):
        h = np.sign(x) * np.maximum(np.abs(x) - p[0], 0)**p[1]
        return np.c_[_aff(x), h]
    def residual(p):
        a = basis(u, p)
        return (a @ _ls(a, v) - v).ravel()
    p = least_squares(residual, [.35, 1.5], bounds=([.1, .6], [.7, 2.5])).x
    return basis(z, p) @ _ls(basis(u, p), v)


def _scalar_basis(z, b, mode):
    base = _quadratic(np.c_[z, b])
    zr, br = np.roll(z, -1, axis=1), np.roll(b, -1, axis=1)
    if mode == 'interaction':
        return np.c_[base, z*zr*b, z*b*br, z**3, b**3]
    return np.c_[base, z*z*b, z*b*b, z*z*br, b*b*zr]


def _sparse_predict(a, q, y, penalty):
    scale = np.r_[1, np.maximum(a[:, 1:].std(0), .035)]
    a = a / scale
    p = np.r_[0, np.full(a.shape[1]-1, penalty)]
    h, rhs = a.T @ a, a.T @ y
    w = np.linalg.solve(h + np.diag(p + 1e-10), rhs)
    for _ in range(8):
        w = np.linalg.solve(h + np.diag(p * (.06/(abs(w) + .004))**1.45 + 1e-10), rhs)
    return (q / scale) @ w


def solve_batch(study):
    tr = study['train']
    x = np.r_[tr['x'], study['test']['x']]
    n = len(tr['x'])
    z = _inverse(x, _calibrate(study['panel_a']))
    b = _dead_zone(study['panel_b'], z)
    smooth = _operator(study['panel_b'], z)
    y = np.zeros(len(x)-n)
    for weight, op, mode, penalty in (
        (.40, b, 'interaction', .012),
        (.30, b, 'raw', .16),
        (.30, smooth, 'raw', .16),
    ):
        a = _scalar_basis(z, op, mode)
        y += weight * _sparse_predict(a[:n], a[n:], tr['y'], penalty)
    a = _quadratic(np.c_[z, b])
    t = _predict(a[:n], a[n:], tr['traj'], .003)
    d = np.array([y.mean(), y.std(), t[:, -1].mean(), t[:, -1].std()])
    return {'y_hat': y, 'state_pred': t, 'diagnostics_vec': d}
