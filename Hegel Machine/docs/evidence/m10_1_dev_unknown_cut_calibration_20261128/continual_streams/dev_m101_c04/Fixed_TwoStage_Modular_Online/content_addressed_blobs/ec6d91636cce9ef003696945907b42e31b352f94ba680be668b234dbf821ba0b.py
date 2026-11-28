import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    n = np.roll(u, -1, axis=1)
    designs = np.stack((np.ones_like(u), u, n, u*n, u**3), axis=-1)
    weights = np.ones(len(u))
    for _ in range(10):
        w = np.sqrt(weights)
        coef = np.array([np.linalg.lstsq(designs[:, j]*w[:, None], v[:, j]*w, rcond=None)[0] for j in range(u.shape[1])])
        r = np.linalg.norm(np.einsum('nck,ck->nc', designs, coef)-v, axis=1)
        scale = max(np.median(r), 1e-8)
        weights = 1/(1+(r/(2*scale))**2)
    keep = np.argsort(r)[:-3]
    return np.array([np.linalg.lstsq(designs[keep, j], v[keep, j], rcond=None)[0] for j in range(u.shape[1])]).T


def _inverse(x, c):
    bias, linear, cross, product, cubic = c
    def forward(z):
        n = np.roll(z, -1)
        return bias + linear*z + cross*n + product*z*n + cubic*z**3
    return np.array([least_squares(lambda z: forward(z)-row, (row-bias)/linear, gtol=1e-11).x for row in x])




def _quadratic(z):
    i, j = np.triu_indices(z.shape[1])
    return np.c_[_affine(z), z[:, i] * z[:, j]]


def _scalar_coordinate(d, e, y):
    # Identify the scalar index and its response curve using only training rows.
    base = np.linalg.lstsq(d, y, rcond=None)[0]
    best = None
    def fit(kind, init, size, test_kind):
        nonlocal best
        span = d[:, :size]
        ev = e[:, :size]
        def evaluate(p, D, S, jac=False):
            if kind == 'index':
                q = D @ p[:-2]
                k = np.exp(p[-1])
                h = np.tanh(k*q)
                if jac:
                    g = p[-2]*(1-h*h)*k
                    return np.c_[D*(1+g[:, None]), h, g*q]
                return q+p[-2]*h
            q = S @ p[d.shape[1]:-1]
            if kind == 'sin':
                h, g = np.sin(q), np.cos(q)
            else:
                h = np.tanh(q)
                g = 1-h*h
            if jac:
                return np.c_[D, S*(p[-1]*g[:, None]), h]
            return D@p[:d.shape[1]]+p[-1]*h
        lo, hi = np.full(len(init), -20.), np.full(len(init), 20.)
        if kind == 'index':
            lo[-1], hi[-1] = -3, 2.8
        opt = least_squares(lambda p:evaluate(p,d,span)-y, np.clip(init,lo+1e-5,hi-1e-5), jac=lambda p:evaluate(p,d,span,True), bounds=(lo,hi), max_nfev=120, ftol=1e-8)
        err = np.mean(opt.fun**2)
        # Complexity correction favors a single quadratic index over a second index.
        quality = err * (1 + 2*len(init)/(len(y)-len(init)))
        if best is None or quality < best[0]:
            p = opt.x
            if kind == 'index' and test_kind == 'latent':
                z0,z1 = d@p[:-2],e@p[:-2]
            else:
                z0,z1 = evaluate(p,d,span),evaluate(p,e,ev)
            best = quality,z0,z1,kind,err
    for a in (.2, .6, -.25):
        for k in (.6, 1.7, 3.):
            fit('index',np.r_[base/(1+a*.45),a,np.log(k)],d.shape[1],'latent')
    if best[-1] > 0.00014:
        for size in (5, d.shape[1]):
            for kind in ('tanh','sin'):
                for scale in (.6,1.7):
                    fit(kind,np.r_[base*.8,scale*base[:size],.5],size,'prediction')
    return best[1:3]


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']
    train = s['train']
    c = _calibrate(s['panel_a'])
    panel = s['panel_b']
    b = np.linalg.lstsq(_affine(panel['u']), panel['v'], rcond=None)[0]
    def design(x):
        z = np.linalg.solve(b[1:].T, (_inverse(x,c)-b[0]).T).T
        return _quadratic(z)
    d,e = design(train['x']),design(s['test']['x'])
    a0,a1 = _scalar_coordinate(d,e,train['y'])
    # The affine trajectory readout needs the three supported dynamic modes.
    traj = train['traj']
    centered = traj - traj.mean(axis=0)
    _,_,vt = np.linalg.svd(centered, full_matrices=False)
    modes = centered @ vt[:3].T
    w = np.linalg.lstsq(d,modes,rcond=None)[0]
    return {'z_tr':np.c_[a0,d@w], 'z_te':np.c_[a1,e@w], 'y':train['y'], 'traj':traj}
