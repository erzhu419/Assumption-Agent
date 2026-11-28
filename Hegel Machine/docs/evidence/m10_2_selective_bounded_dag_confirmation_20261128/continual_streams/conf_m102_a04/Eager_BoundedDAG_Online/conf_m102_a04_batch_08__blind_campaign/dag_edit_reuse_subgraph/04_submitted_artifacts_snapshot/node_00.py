import numpy as np
from scipy.optimize import least_squares
from itertools import combinations


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _one(x):
    return np.column_stack((np.ones(len(x)), x))


def _pairs(x):
    return np.column_stack([x[:, i]*x[:, j] for i, j in combinations(range(x.shape[1]), 2)])


def _coordinates(study):
    a = study['panel_a']
    u, v = np.asarray(a['u'], float), np.asarray(a['v'], float)
    x = np.vstack((study['train']['x'], study['test']['x'])).astype(float)
    d = u.shape[1]
    coef = np.empty((5, d))
    for j in range(d):
        k = (j+1) % d
        f = np.column_stack((np.ones(len(u)), u[:, j], u[:, k], u[:, j]*u[:, k], u[:, j]**3))
        b = _fit(f, v[:, j])
        for _ in range(10):
            r = abs(f@b-v[:, j]); w = 1/np.sqrt(np.maximum(r, 2e-4))
            b = _fit(f*w[:, None], v[:, j]*w)
        r = abs(f@b-v[:, j]); keep = r <= max(8*np.median(r), .001)
        coef[:, j] = _fit(f[keep], v[keep, j]) if keep.sum() >= 5 else b
    o, a, b, c, e = coef
    z = (x-o)/a
    idx = np.arange(d)
    for _ in range(20):
        q = np.roll(z, -1, axis=1)
        r = o + a*z + b*q + c*z*q + e*z**3 - x
        if np.max(abs(r)) < 1e-10: break
        jac = np.zeros((len(z), d, d))
        jac[:, idx, idx] = a+c*q+3*e*z*z
        jac[:, idx, (idx+1)%d] = b+c*z
        z -= np.clip(np.linalg.solve(jac, r[...,None])[...,0], -.4, .4)
    p = study['panel_b']; rot = _fit(_one(p['u']), p['v'])
    return z, _one(z)@rot, (z-rot[0])@np.linalg.inv(rot[1:])


def _maps(study, xs):
    p = study['panel_c']; u = p['u']; n = u.shape[1]
    coeff = _fit(_one(_pairs(u)), p['v1'])
    def f(x, q):
        x2 = x*x
        return np.column_stack((_one(x), x2, x2/(1+q*x2)))
    v = p['v2']
    def residual(q):
        a = f(u, q)
        return (a@_fit(a,v)-v).ravel()
    q = least_squares(residual, np.full(n,.25), bounds=(.02,2), gtol=1e-12).x
    c = _fit(f(u,q),v)
    return [(_one(_pairs(w))@coeff, f(w,q)@c) for w in xs]




def _dictionary(z, c, d, mode):
    from itertools import combinations_with_replacement
    def mon(x, degree):
        return np.column_stack([np.prod(x[:, j], axis=1) for j in combinations_with_replacement(range(x.shape[1]), degree)])
    base = np.column_stack((z,c,d))
    blocks = [base, mon(base,2)]
    if mode in (1,3):
        blocks.extend([mon(z,3), mon(z,4), np.sin(1.5*z), np.tanh(z), z/(1+.7*abs(z)), z*z/(1+.25*z*z)])
    if mode in (2,3):
        pairs = _pairs(z)
        for x in (z,c,d):
            blocks.append((x[:,:,None]*pairs[:,None,:]).reshape(len(z),-1))
        blocks.extend([mon(c,3), mon(d,3), c/(1+.25*c*c), d/(1+.25*d*d)])
    return np.column_stack(blocks)


def _feature_metric(f, y, n, prior):
    # Multi-response empirical Bayes: feature variances estimated from training rows only.
    from scipy.linalg import cho_factor, cho_solve
    mean = f[:n].mean(0)
    std = np.maximum(f[:n].std(0),1e-7)
    f = np.clip((f-mean)/std,-20,20)
    t = y-y.mean(0)
    t /= np.maximum(t.std(0),1e-7)
    A = f[:n]
    power = np.full(A.shape[1], 1/A.shape[1]); noise = .025
    eye = np.eye(n)
    for _ in range(105):
        cov = (A*power)@A.T+noise*eye
        fac = cho_factor(cov,check_finite=False)
        H = cho_solve(fac,A,check_finite=False)
        T = cho_solve(fac,t,check_finite=False)
        denom = np.sum(A*H,axis=0).clip(1e-10)
        num = np.mean((A.T@T)**2,axis=1)
        power = np.maximum(1e-10, power*np.clip(num/denom,.15,8)**.72)
        dn = np.trace(cho_solve(fac,eye,check_finite=False))
        noise = max(prior,noise*np.clip(np.mean(np.sum(T*T,axis=0))/dn,.2,5)**.72)
    return f, power, noise


def _library(study):
    z = _coordinates(study)[0]
    c,d = _maps(study, (z,))[0]
    return z,c,d


def _representation(study, config, library=None):
    z,c,d = _library(study) if library is None else library
    n = len(study['train']['x'])
    k, a, b, g = config
    if k < 0:
        blocks = [z,c,d,_pairs(z),z*z/(1+.25*z*z)]
        ids = {-1:(0,1,2),-2:(0,1,4),-3:(0,2,3),-4:(0,1,2,3,4)}[k]
        multipliers = [1.,a,b] + ([.45,.4] if len(ids)>3 else [])
        pieces = []
        for i,weight in zip(ids,multipliers):
            q = blocks[i]; pieces.append(weight*(q-q[:n].mean(0))/np.maximum(q[:n].std(0),.1))
        return g*np.column_stack(pieces)
    y = np.column_stack((study['train']['y'],study['train']['traj']))
    y = y[:, list(dict.fromkeys([0,1, min(3,y.shape[1]-1),y.shape[1]-1]))]
    f,p,noise = _feature_metric(_dictionary(z,c,d,k),y,n,a)
    sel = p > max(p.max()*b,1e-9)
    p = p[sel]
    return f[:,sel]*(np.sqrt(p/p.sum())*np.sqrt(len(p))*g)


def run_node(inputs: dict) -> dict:
    study = inputs["__input__"]
    features = _representation(study, (1, 1e-05, 0.01, 0.3))
    return {"features": features, "n_train": len(study["train"]["x"])}
