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


def _even(w):
    return _aff(w+.35*w*w+.065*(1-np.cos(2.5*w)))


def _designs(study, x):
    z = _calibrate(study['panel_a'], x)
    panel = study['panel_c']
    cfit = _ls(_pairmap(panel['u']), panel['v1'])
    d = np.tanh(1.25*(_even(z)@_ls(_even(panel['u']), panel['v2'])))
    panel = study['panel_b']
    u, v = panel['u'], panel['v']
    def odd(w, p):
        return _aff(w+p[0]*np.sin(p[1]*w)+p[2]*np.cbrt(w))
    def residual(p):
        f = odd(u, p)
        return (f@_ls(f,v)-v).ravel()
    p = least_squares(residual, [-.4, 2.9, 0], max_nfev=220).x
    b1 = odd(z,p)@_ls(odd(u,p),v)
    def flexible(w):
        return _aff(np.column_stack((w,w*w,w**3,w**5)))
    b2 = flexible(z)@_ls(flexible(u),v)
    for b, transform in ((b1,lambda w: np.arcsinh(1.25*w)), (b2,lambda w:w)):
        c = transform(_pairmap(b)@cfit)
        interaction = (c[:,:,None]*d[:,None,:]).reshape(len(x),16)
        yield np.column_stack((c,d,interaction))



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


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    b,c,d = _branches(study, np.vstack((train['x'],study['test']['x'])))
    d = np.tanh(1.25*d)
    def predict(c, y, scalar):
        main = _gram(np.c_[c,d],n)
        prod = _gram((c[:,:,None]*d[:,None,:]).reshape(len(c),-1),n)
        k = 10+main+prod+.3*_gram(b,n)
        if scalar:
            k += .2*main*prod
        else:
            k += .1*main**2+.07*main**3
        ym = y.mean(0)
        return ym + k[n:,:n] @ np.linalg.solve(k[:n,:n]+.008*np.eye(n), y-ym)
    y = predict(np.arctan(1.25*c), train['y'], True)
    state = predict(np.arcsinh(2*c), train['traj'], False)
    return {'y_hat':y, 'state_pred':state, 'diagnostics_vec':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}
