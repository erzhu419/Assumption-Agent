import itertools
import numpy as np
from scipy.optimize import minimize_scalar


def _terms(x, degree=3):
    return [t for d in range(degree+1) for t in itertools.combinations_with_replacement(range(x.shape[1]), d)]


def _design(x, terms):
    return np.column_stack([np.prod(x[:, t], axis=1) for t in terms])


def _ls(a,b):
    return np.linalg.lstsq(a,b,rcond=1e-10)[0]


def _inverse(panel, observed):
    u,v = panel['u'],panel['v']
    affine = _terms(u,1)
    a = _design(u,affine)
    keep = np.ones(len(u),bool)
    for _ in range(8):
        e = np.linalg.norm(a @ _ls(a[keep],v[keep])-v,axis=1)
        mask = e <= max(4*np.median(e),1e-9)
        if mask.sum() < 12 or np.array_equal(mask,keep):
            break
        keep = mask
    w0 = _ls(a[keep],v[keep])
    terms = _terms(u)
    d = _design(u,terms)
    w = _ls(d[keep],v[keep]); w[np.abs(w)<0.002] = 0
    for j in range(v.shape[1]):
        selected = w[:,j] != 0
        w[selected,j] = _ls(d[keep][:,selected],v[keep,j])
    q = (observed-w0[0]) @ np.linalg.pinv(w0[1:])
    for _ in range(30):
        err = _design(q,terms) @ w - observed
        if np.max(np.abs(err)) < 1e-10:
            break
        deriv = np.zeros((len(q),q.shape[1],observed.shape[1]))
        for i, t in enumerate(terms):
            if not np.any(w[i]):
                continue
            for k in set(t):
                other = list(t); other.remove(k)
                deriv[:,k,:] += (t.count(k)*np.prod(q[:,other],axis=1))[:,None]*w[i]
        step = np.einsum('nij,nj->ni',np.linalg.pinv(deriv.transpose(0,2,1)),err)
        loss = np.sum(err**2,axis=1)
        for _ in range(12):
            bad = np.sum((_design(q-step,terms) @ w-observed)**2,axis=1)>loss
            if not bad.any():
                break
            step[bad] *= .5
        q -= step
    return q


def _ring(x):
    return np.column_stack((np.ones(len(x)), x * np.roll(x, -1, axis=1)))


def _cosine(x, frequency):
    return np.column_stack((np.ones(len(x)), np.cos(frequency*x)))


def _operators(panel, q):
    u = panel['u']
    ring = _ring(u)
    p = _ring(q) @ _ls(ring, panel['v1'])
    target = panel['v2']

    def loss(frequency):
        d = _cosine(u, frequency)
        return np.mean((d @ _ls(d, target) - target)**2)

    frequency = minimize_scalar(loss, bounds=(1., 4.), method='bounded',
                                options={'xatol': 1e-10}).x
    d = _cosine(u, frequency)
    r = _cosine(q, frequency) @ _ls(d, target)
    return p, r


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    tr = study['train']; x = np.concatenate((tr['x'],study['test']['x']))
    q = _inverse(study['panel_a'],x)
    p,r = _operators(study['panel_c'], q)
    common = np.column_stack((p,r,np.sum(p*r,axis=1)))
    offdiag = np.column_stack([p[:,i]*r[:,j] for i in range(p.shape[1]) for j in range(r.shape[1]) if i!=j])
    n = len(tr['x']); a = np.column_stack((np.ones(len(x)),common))
    y = a[n:] @ _ls(a[:n],tr['y'])
    return {'common':common,'offdiag':offdiag,'y_hat':y}
