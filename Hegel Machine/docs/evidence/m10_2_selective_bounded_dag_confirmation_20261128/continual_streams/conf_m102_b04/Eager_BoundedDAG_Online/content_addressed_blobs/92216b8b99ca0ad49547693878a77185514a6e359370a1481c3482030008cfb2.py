import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import minimize_scalar


def _panel_fit(panel):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    n, d = u.shape
    a = np.column_stack((np.ones(n), u))
    design = [np.column_stack((a, u[:, j] * u[:, (j+1)%d], u[:, j]**3)) for j in range(d)]
    keep = np.arange(n)
    for _ in range(12):
        b = np.column_stack([np.linalg.lstsq(f[keep], v[keep, j], rcond=None)[0] for j, f in enumerate(design)])
        err = np.linalg.norm(np.column_stack([f @ b[:, j] for j, f in enumerate(design)]) - v, axis=1)
        order = np.argsort(err)
        med = np.median(err)
        clean = np.flatnonzero(err <= max(6*med, 1e-8))
        next_keep = clean if len(clean) >= max(8, int(.8*n)) else np.sort(order[:max(8, int(.85*n))])
        if np.array_equal(keep, next_keep):
            break
        keep = next_keep
    return b


def _inverse(raw, b):
    n, d = raw.shape
    linear = b[1:d+1]
    nxt = np.roll(np.arange(d), -1)
    col = np.arange(d)
    u = (raw-b[0]) @ np.linalg.pinv(linear)
    def residual(p):
        return b[0]+p @ linear+b[-2]*p*p[:, nxt]+b[-1]*p**3-raw
    for _ in range(25):
        r = residual(u)
        if np.max(np.abs(r), initial=0.) < 1e-11:
            break
        jac = np.broadcast_to(linear.T, (n,d,d)).copy()
        jac[:, col, col] += b[-2]*u[:, nxt]+3*b[-1]*u*u
        jac[:, col, nxt] += b[-2]*u
        step = (np.linalg.pinv(jac) @ r[..., None])[..., 0]
        best = u.copy()
        err = np.linalg.norm(r,axis=1)
        for scale in (1., .5, .25, .125, .0625):
            trial = u-scale*step
            trial_err = np.linalg.norm(residual(trial),axis=1)
            take = trial_err < err
            best[take], err[take] = trial[take], trial_err[take]
        u = best
    return u, np.max(np.abs(residual(u)), axis=1)


def _predict(u, y, z):
    n = len(y)
    one = np.ones(len(u))
    square = np.column_stack([u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(u.shape[1]),2)])
    base_y = np.column_stack((one,u,square))
    base_z = np.column_stack((one,u))
    cube = u**3
    def profile(target, base, power, bound):
        def fit(a):
            f = np.column_stack((base,cube/(1+a*np.abs(u)**power)))
            b = np.linalg.lstsq(f[:n], target, rcond=None)[0]
            return f @ b
        def loss(a):
            return np.mean((fit(a)[:n]-target)**2)
        a = minimize_scalar(loss, method='bounded', bounds=(.002,bound)).x
        return fit(a)[n:]
    return profile(y,base_y,3,2.), profile(z,base_z,2,3.)


def _ridge_fallback(f, targets):
    n = len(targets)
    a = f[:n]
    return f[n:] @ np.linalg.solve(a.T @ a + .018*np.eye(a.shape[1]),a.T @ targets)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    frozen = inputs['node_00']
    hybrid = np.asarray(frozen['hybrid'])
    aligned = np.asarray(frozen['aligned'])
    train, test = study['train'], study['test']
    y, z = np.asarray(train['y']), np.asarray(train['traj'])
    raw = np.vstack((train['x'], test['x']))
    u, err = _inverse(raw, _panel_fit(study['panel_a']))
    if hybrid.shape[0] != len(raw) or aligned.shape[0] != len(raw):
        raise ValueError('Frozen feature rows do not match the study')
    bad = (~np.isfinite(u).all(axis=1)) | (err > 1e-4)
    if np.any(bad[:len(y)]):
        yh, state = _ridge_fallback(hybrid,y), _ridge_fallback(aligned,z)
    else:
        yh, state = _predict(np.where(np.isfinite(u),u,0.), y,z)
        fail = bad[len(y):]
        if np.any(fail):
            yh[fail] = _ridge_fallback(hybrid,y)[fail]
            state[fail] = _ridge_fallback(aligned,z)[fail]
    end = state[:, -1]
    return {'y_hat':yh, 'state_pred':state, 'diagnostics_vec':np.array([yh.mean(),yh.std(),end.mean(),end.std()])}
