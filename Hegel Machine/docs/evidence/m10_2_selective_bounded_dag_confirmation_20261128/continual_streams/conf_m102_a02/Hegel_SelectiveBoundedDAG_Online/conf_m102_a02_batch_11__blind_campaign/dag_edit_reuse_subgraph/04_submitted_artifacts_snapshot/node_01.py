import numpy as np
from scipy.optimize import least_squares, root
from itertools import combinations_with_replacement


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _coords(study):
    u, v = [np.asarray(study['panel_a'][k]) for k in ('u','v')]
    r = np.linalg.norm(v-u, axis=1)
    med = np.median(r)
    keep = r <= med + max(1., 8*np.median(abs(r-med)))
    u, v = u[keep], v[keep]
    def features(q, p, j):
        return np.c_[np.ones(len(q)), q[:,j], np.tanh(p[0]*q[:,j]), np.sin(p[1]*q[:,(j+2)%4])]
    def residual(p):
        return np.column_stack([f@_ls(f,v[:,j])-v[:,j] for j in range(4) for f in [features(u,p,j)]]).ravel()
    p = least_squares(residual, [1.8,2.4], bounds=([.7,1.4],[3.,3.4])).x
    w = np.array([_ls(features(u,p,j),v[:,j]) for j in range(4)])
    bias, gain, amp, cross = w.T
    n = len(study['train']['x'])
    x = np.vstack([study['train']['x'],study['test']['x']])
    def forward(q):
        return bias+gain*q+amp*np.tanh(p[0]*q)+cross*np.sin(p[1]*np.roll(q,2))
    q = np.array([root(lambda q: forward(q)-r,(r-bias)/(gain+amp)).x for r in x])
    return q,n


def _scalar(q, y):
    # Profile the quadratic baseline and affine switching coefficients out of
    # the fit of a single smooth, tilted transition in calibrated coordinates.
    n = len(y)
    affine = np.c_[np.ones(len(q)), q]
    poly = np.c_[affine, *[q[:, i]*q[:, j]
                          for i, j in combinations_with_replacement(range(4), 2)]]
    x_train = affine[:n]
    a_train = poly[:n]

    def design(p, x, a):
        return np.c_[a, x*np.tanh(x@p)[:, None]]

    def residual(p):
        f = design(p, x_train, a_train)
        return f@_ls(f, y)-y

    starts = [np.r_[0., scale*axis] for scale in (6., 2.) for axis in np.eye(4)]
    bounds = ([-3., -15., -15., -15., -15.], [3., 15., 15., 15., 15.])
    fit = min((least_squares(residual, p, bounds=bounds, max_nfev=65)
               for p in starts), key=lambda result: result.cost)
    return design(fit.x, affine[n:], poly[n:])@_ls(design(fit.x, x_train, a_train), y)


def run_node(inputs: dict) -> dict:
    study, upstream = inputs['__input__'], inputs['node_00']
    q,n = _coords(study)
    a,k,e = upstream['p']
    h = q*np.exp(e*np.logaddexp(0,k*(np.log(a)+np.log(np.maximum(abs(q),1e-100)))))
    b = np.c_[np.ones(len(q)),q,h]@upstream['w']
    c = b*np.roll(q,-1,axis=1)
    tr = study['train']
    state = np.c_[np.ones(len(q)),b,c.sum(1)]
    state = state[n:]@_ls(state[:n],tr['traj'])
    y = _scalar(q,tr['y'])
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}
