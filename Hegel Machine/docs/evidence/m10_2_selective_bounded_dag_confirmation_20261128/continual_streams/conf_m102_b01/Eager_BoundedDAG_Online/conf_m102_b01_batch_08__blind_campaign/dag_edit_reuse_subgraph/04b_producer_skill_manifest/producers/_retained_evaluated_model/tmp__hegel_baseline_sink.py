import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _cross(x):
    return np.column_stack([x[:, i]*x[:, j] for i,j in [(0,1),(0,3),(1,2),(2,3)]])


def _cal(s, w):
    b, c = s['panel_b'], s['panel_c']
    a = _fit(_aff(b['u']), b['v'])
    u = _fit(a[1:].T, (w-a[0]).T).T
    def design(x, g):
        return np.column_stack((_aff(x), x**3/(1+g*x*x)))
    def residual(g):
        d = design(b['u'], g)
        return (d @ _fit(d,b['v']) - b['v']).ravel()
    gain = least_squares(residual, np.full(u.shape[1], 1.25), bounds=(.01, 5.)).x
    d = design(b['u'], gain)
    W = design(u, gain) @ _fit(d,b['v'])
    def c1(x):
        return np.column_stack((np.ones(len(x)),_cross(x)))
    def c2(x):
        return np.column_stack((_aff(x),x*x))
    C1 = c1(u) @ _fit(c1(c['u']),c['v1'])
    C2 = c2(u) @ _fit(c2(c['u']),c['v2'])
    return u, W, C1, C2


def _kernel(z, n, length):
    z = (z-z[:n].mean(0))/np.maximum(z[:n].std(0), 1e-6)
    p = z @ z[:n].T / z.shape[1]
    return (1 + p)**3 + np.exp(-cdist(z,z[:n], 'sqeuclidean')/(length*z.shape[1]))


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']
    u,w,c1,c2 = _cal(s, inputs['node_00']['w'])
    tr = s['train']; n = len(tr['y'])
    k = _kernel(w,n,1.5) + .2*_kernel(c1,n,2) + .1*_kernel(c2,n,2)
    target = np.column_stack((tr['y'], tr['traj']))
    p = k[n:] @ np.linalg.solve(k[:n] + .035*np.eye(n), target)
    y, state = p[:,0], p[:,1:]; tail=state[:,-1]
    return {'y_hat':y, 'state_pred':state, 'diagnostics_vec':np.array([y.mean(),y.std(),tail.mean(),tail.std()])}
