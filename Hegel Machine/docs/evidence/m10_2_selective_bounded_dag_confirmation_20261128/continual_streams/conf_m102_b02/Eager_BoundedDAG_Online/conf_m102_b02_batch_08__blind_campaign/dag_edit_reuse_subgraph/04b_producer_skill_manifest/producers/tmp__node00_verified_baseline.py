"""Calibrate the sensor and construct regularized prediction operators."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def affine(x):
    return np.c_[np.ones(len(x)), x]


def fit(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def roll(x):
    return np.roll(x, -1, axis=1)


def cubic(x):
    return affine(np.c_[x, x * roll(x), x**3])


def coordinates(panel, x):
    u, v = panel['u'], panel['v']
    f = cubic(u)
    rng = np.random.default_rng(42)
    def error(w):
        return np.linalg.norm(f @ w - v, axis=1)
    w = fit(f, v)
    if np.median(error(w)) > .001:
        w = min((fit(f[i], v[i]) for i in (rng.choice(len(f), 17, replace=False) for _ in range(128))), key=lambda w: np.median(error(w)))
    e = error(w)
    keep = e <= max(.001, 4*np.median(e))
    if keep.sum() >= f.shape[1]:
        w = fit(f[keep], v[keep])
    z = fit(w[1:5].T, (x - w[0]).T).T
    for _ in range(25):
        j = w[1:5] + roll(z)[:, :, None] * w[5:9] + np.roll(z[:, :, None] * w[5:9], 1, axis=1) + 3*z[:, :, None]**2 * w[9:13]
        e = cubic(z) @ w - x
        try:
            dz = np.linalg.solve(j.transpose(0, 2, 1), e[..., None])[..., 0]
        except np.linalg.LinAlgError:
            dz = np.einsum('nij,nj->ni', np.linalg.pinv(j.transpose(0, 2, 1)), e)
        z -= np.clip(dz, -.4, .4)
        if np.max(np.abs(dz), initial=0) < 1e-9:
            break
    return z


def features(s, x):
    z = coordinates(s['panel_a'], x)
    p = s['panel_b']; u, v = p['u'], p['v']
    def dead(x, q):
        return affine(np.sign(x)*np.maximum(abs(x)-q[0], 0)**q[1])
    def residual(q):
        h = dead(u, q)
        return (h @ fit(h, v) - v).ravel()
    q = least_squares(residual, [.22, 1.5], bounds=([0, .5], [.85, 3])).x
    b = dead(z, q) @ fit(dead(u, q), v)
    p = s['panel_c']; u = p['u']; v = p['v2']
    def harmonic(x, k):
        return affine(np.c_[x, np.cos(k*x)])
    def loss(k):
        h = harmonic(u, k)
        return np.mean((h @ fit(h, v) - v)**2)
    k = minimize_scalar(loss, bounds=(.2, 4), method='bounded').x
    c = cubic(z) @ fit(cubic(u), p['v1'])
    d = harmonic(z, k) @ fit(harmonic(u, k), v)
    h = b[:, 2:3]*d[:, 0:1]*d[:, 1:2]
    return z, b, c, d, h


def candidate_kernels(z, b, c, d, h):
    base = affine(np.c_[z, b*roll(b)])
    extra = np.c_[h, d[:, 0]**2, b[:, 2]/(1 + 4*z[:, 0]**2)]
    patterns = {'b': b, 'bd': np.c_[b,d], 'zbd': np.c_[.2*z,b,d], 'mixed': np.c_[.3*z,b,.3*c,d], 'z': z, 'c': c, 'd': d, 'sym': np.c_[b+.3*z,b-.3*z,d]}
    for nm, x in patterns.items():
        g = x @ x.T
        dis = np.maximum(np.diag(g)[:,None]+np.diag(g)-2*g, 0)
        for degree in (2,3,4):
            for scale in (.5, 1.5, 3.):
                k = (1+scale**2*g)**degree
                k /= max(1., np.mean(np.diag(k)))
                yield (nm, degree, scale), base, k
                if degree == 3:
                    yield (nm+'+', degree, scale), np.c_[base,extra], k
        for scale in (.5, 1.5, 3.):
            yield (nm, 0, scale), base, np.exp(-.5*scale**2*dis)
    yield ('sparse',0,0), np.c_[base,extra], np.zeros((len(z),len(z)))


def operators(z,b,c,d,h, target, family='all', fixed=None):
    n = len(target)
    y = np.asarray(target).reshape(n,-1)
    def options():
        for spec, a, k in candidate_kernels(z,b,c,d,h):
            if family != 'all' and spec[0].rstrip('+') not in family.split(','):
                continue
            q = np.linalg.qr(a[:n],mode='complete')[0]
            p = q[:,a.shape[1]:]
            if not p.shape[1]:
                continue
            lin = a[n:] @ np.linalg.pinv(a[:n],rcond=1e-10)
            l = k[:n,:n]
            w,u = np.linalg.eigh(p.T@l@p)
            u = p@u
            w = np.maximum(w,0)
            kt = (k[n:,:n] - lin@l)@u
            uy = u.T@y
            for reg in (.00003,.0004,.005,.04,.25):
                if fixed is not None and (spec,reg) != fixed:
                    continue
                inv = 1/(w+reg)
                r = (u@(uy*inv[:,None]))/np.maximum((u*u)@inv,1e-14)[:,None]
                loss = np.mean(abs(r),axis=0)
                yield (spec,reg), loss, lin + (kt*inv)@u.T
    return options()


FAMILY_Y = 'sparse'
FAMILY_T = 'sym'
FIXED_Y = None
FIXED_T = None


def select(z,b,c,d,h,target,family,fixed):
    cand = list(operators(z,b,c,d,h,target,family,fixed))
    scale = np.maximum(np.std(target,axis=0),1e-6)
    ranked = sorted(cand, key=lambda row: np.mean(row[1]/scale))[:9]
    return np.mean([row[2] for row in ranked],axis=0)


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']; tr = s['train']
    x = np.r_[tr['x'], s['test']['x']]
    z,b,c,d,h = features(s,x)
    y = select(z,b,c,d,h,tr['y'],FAMILY_Y,FIXED_Y) @ tr['y']
    t = select(z,b,c,d,h,tr['traj'],FAMILY_T,FIXED_T)
    return {'y_hat': y, 'trajectory_design': t}
