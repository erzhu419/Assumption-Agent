"""Calibrated nonlinear spatial readouts with a two-mode relaxation model."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _design(*parts):
    return np.column_stack((np.ones(len(parts[0])), *parts))


def _fit(a, b, penalty=0.):
    if not penalty:
        return np.linalg.lstsq(a, b, rcond=None)[0]
    return np.linalg.solve(a.T @ a + penalty*np.diag(np.r_[0., np.ones(a.shape[1]-1)]), a.T @ b)


def _calibration(panel):
    u, v = (np.asarray(panel[key], float) for key in ('u', 'v'))
    j = np.arange(u.shape[1]); o = np.roll(j, 2)
    d = np.linalg.norm(v-u, axis=1)
    keep = d < 4*np.median(d)
    def matrices(z, k):
        s = np.sin(k*z)
        return [_design(z[:, i], z[:, h], s[:, i], s[:, h]) for i,h in zip(j,o)]
    def fit(k, return_coef=False):
        m = matrices(u[keep], k[0]); t = v[keep]
        b = np.column_stack([_fit(a,t[:,i]) for i,a in enumerate(m)])
        return b if return_coef else (np.column_stack([a@b[:,i] for i,a in enumerate(m)])-t).ravel()
    k = least_squares(fit, [2.4], bounds=(1.5,3.5)).x
    b = fit(k, True); k = k[0]
    def invert(x):
        z = x.copy()
        for _ in range(30):
            f = np.column_stack([a@b[:,i] for i,a in enumerate(matrices(z,k))])-x
            if np.max(np.abs(f), initial=0.) < 1e-11:
                break
            c = k*np.cos(k*z)
            jac = np.zeros((len(z),len(j),len(j)))
            jac[:,j,j] = b[1]+b[3]*c
            jac[:,j,o] = b[2]+b[4]*c[:,o]
            step = np.linalg.solve(jac,f[...,None])[...,0]
            z -= step / np.maximum(1., np.max(abs(step), axis=1, keepdims=True))
        return z
    return invert


def _operator(panel, key, form, bounds):
    u,v = panel['u'], panel[key]
    def f(k, get_coef=False):
        d = _design(u,form(u,k)); b = _fit(d,v)
        if get_coef:
            return b
        lev = (np.linalg.qr(d,mode='reduced')[0]**2).sum(axis=1)
        return np.mean(((d@b-v)/(1.-lev[:,None]))**2)
    k = minimize_scalar(f, bounds=bounds, method='bounded').x
    b = f(k, True)
    return lambda z: _design(z,form(z,k)) @ b


def _scalar_design(panel, z):
    # Profile a single elementwise activation shared by the panel outputs.
    def activation(x, p):
        k, linear = p
        return x/(1.+k*x*x) + linear*x

    u, v = panel['u'], panel['v']
    def residual(p):
        d = _design(activation(u, p))
        return (d @ _fit(d, v) - v).ravel()

    p = least_squares(residual, [.75,-.94], bounds=([.01,-5.],[5.,5.])).x
    h = _design(activation(z, p)) @ _fit(_design(activation(u, p)), v)
    # Opposite-coordinate products occur in symmetric pairs; keep one of each.
    half = h.shape[1]//2
    return _design(h, h[:,:half]*h[:,half:])


def _spatial(study, x):
    z = _calibration(study['panel_a'])(x)
    r = _operator(study['panel_b'], 'v', lambda x,k: x/(1+k*x*x), (.05,4.))(z)
    c = _operator(study['panel_c'], 'v1', lambda x,k: np.arcsinh(k*x), (.05,6.))(z)
    return _scalar_design(study['panel_b'], z), _design(r,c)


def _time_basis(t, trajectory):
    def basis(p):
        return -np.expm1(-t[:,None]*p)
    def residual(p):
        f = basis(p)
        return (f@_fit(f,trajectory.T)-trajectory.T).ravel()
    p = least_squares(residual,[.4,1.8],bounds=(.01,10.)).x
    return basis(p)


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    x=np.asarray(train['x'],float); xx=np.asarray(test['x'],float)
    n=len(x)
    a,b = _spatial(study,np.vstack((x,xx)))
    y = a[n:] @ _fit(a[:n],train['y'],.0015)
    f = _time_basis(np.asarray(study['eval_times']),train['traj'])
    amplitudes = _fit(f,train['traj'].T).T
    s = b[n:] @ _fit(b[:n],amplitudes,.002) @ f.T
    return {'y_hat':y,'state_pred':s,
            'diagnostics_vec':np.array([y.mean(),y.std(),s[:,-1].mean(),s[:,-1].std()])}
