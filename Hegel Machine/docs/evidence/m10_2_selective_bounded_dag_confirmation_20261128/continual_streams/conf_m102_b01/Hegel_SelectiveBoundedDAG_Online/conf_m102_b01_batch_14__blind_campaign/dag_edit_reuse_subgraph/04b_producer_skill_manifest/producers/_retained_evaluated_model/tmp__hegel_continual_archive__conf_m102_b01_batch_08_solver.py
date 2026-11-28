import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _aff(a):
    return np.column_stack((np.ones(len(a)), a))


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    w = np.roll(u, -1, axis=1)
    coefficients = []
    for j in range(4):
        f = np.column_stack((np.ones(len(u)), u[:, j], w[:, j], u[:, j]*w[:, j], u[:, j]**3))
        target = v[:, j]
        q = _ls(f, target)
        for _ in range(8):
            keep = np.argsort(abs(f@q-target))[:int(.875*len(u))]
            q = _ls(f[keep], target[keep])
        err = abs(f@q-target)
        keep = err < max(.001, 12*np.median(err))
        coefficients.append(_ls(f[keep], target[keep]))
    off, a, b, c, d = np.array(coefficients).T
    z = x.copy()
    j = np.arange(4)
    jac = np.zeros((len(x), 4, 4))
    for _ in range(14):
        w = np.roll(z, -1, axis=1)
        residual = off + a*z + b*w + c*z*w + d*z**3 - x
        jac[:, j, j] = a + c*w + 3*d*z*z
        jac[:, j, (j+1)%4] = b+c*z
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        z -= np.clip(step, -.5, .5)
    return z


def _pairmap(w):
    return _aff(w[:, [0, 0, 1, 2]] * w[:, [1, 3, 2, 3]])


def _project(u, v, x, transform, q0):
    def residual(q):
        A = _aff(transform(u, q))
        return (A @ _ls(A, v) - v).ravel()
    q = least_squares(residual, q0, max_nfev=200).x
    return _aff(transform(x, q)) @ _ls(_aff(transform(u, q)), v)


def _branches(study, x):
    a, b, c = (study[k] for k in ('panel_a', 'panel_b', 'panel_c'))
    z = _calibrate(a, x)
    B = _project(b['u'], b['v'], z,
                 lambda u,q: u+q[0]*np.sin(q[1]*u)+q[2]*np.cbrt(u), [-.4,2.9,0.])
    C = _pairmap(B) @ _ls(_pairmap(c['u']), c['v1'])
    D = _project(c['u'], c['v2'], z,
                 lambda u,q: u+q[0]*u*u+q[1]*(1-np.cos(q[2]*u)), [.35,.065,2.5])
    return B, C, D


def _gram(f, n):
    f = (f-f[:n].mean(0))/np.maximum(f[:n].std(0), 1e-8)
    return (f @ f.T)/f.shape[1]


def _cross(a, b):
    return (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)


def _ridge(f, y, reg):
    n = len(y)
    f = (f - f[:n].mean(0)) / np.maximum(f[:n].std(0), 1e-8)
    mu = y.mean(0)
    a = f[:n]
    return mu + f[n:] @ np.linalg.solve(a.T @ a + reg * np.eye(a.shape[1]), a.T @ (y - mu))


def _kernel(c, d, z, y, reg, local):
    n = len(y)
    l = _gram(np.c_[c, d], n)
    p = _gram(_cross(c, d), n)
    k = 10 + l + p + .4 * _gram(z, n)
    if local:
        v = np.diag(l)
        k += .5 * np.exp(-.25 * np.maximum(v[:, None] + v[None, :] - 2*l, 0))
    else:
        k += .05*l**2 + .03*l**3 + .10*l*p
    mu = y.mean(0)
    return mu + k[n:, :n] @ np.linalg.solve(k[:n, :n] + reg*np.eye(n), y-mu)


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.vstack((train['x'], study['test']['x']))
    gain = float(study['telemetry_vec'][0])
    z = _calibrate(study['panel_a'], x)
    _, c0, d0 = _branches(study, x)
    c = np.tanh(gain*c0)
    d = np.arctan(gain*d0)
    target = train['y']
    # Average compact saturating interactions with a regularized residual model.
    feature = np.c_[c, d, z, np.sin(gain*_cross(c, d))]
    y = .55*_ridge(feature, target, .04)
    y += .30*_kernel(c, d, z, target, .003, False)
    y += .15*_kernel(c, d, z, target, .035, True)
    target = train['traj']
    # A rational interaction complements the local correction at late times.
    cf = c0 / (1 + gain*c0**2)
    df = np.tanh(gain*d0)
    state = .65*_kernel(np.arctan(gain*c0), d, z, target, .003, True)
    state += .35*_ridge(np.c_[c0, df, z, _cross(cf, df)], target, .25)
    tail = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}
