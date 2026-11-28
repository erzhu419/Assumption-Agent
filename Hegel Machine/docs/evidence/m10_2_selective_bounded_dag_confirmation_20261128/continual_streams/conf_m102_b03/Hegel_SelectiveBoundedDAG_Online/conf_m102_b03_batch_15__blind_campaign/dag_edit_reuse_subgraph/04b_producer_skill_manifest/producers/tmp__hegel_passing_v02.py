"""Calibrate the raw coordinates and construct frozen-predictor features."""
import numpy as np
from scipy.optimize import least_squares
def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _ones(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, x):
    u, v = (np.asarray(panel[k]) for k in ('u', 'v'))
    def basis(z):
        w = np.roll(z, -1, axis=1)
        return np.stack((np.ones_like(z), z, w, z*w, z**3), axis=2)
    a = basis(u)
    def coeff(g):
        return np.stack([_ls(a[g,j], v[g,j]) for j in range(4)])
    rng = np.random.default_rng(12)
    best = np.inf
    for _ in range(160):
        b0 = coeff(rng.choice(len(u), min(9, len(u)), replace=False))
        r = np.linalg.norm(np.einsum('njk,jk->nj', a, b0)-v, axis=1)
        score = np.median(r)
        if score < best:
            best, residual = score, r
    b = coeff(residual <= max(0.001, 6*best))
    def forward(z):
        return np.einsum('njk,jk->nj', basis(z), b)
    z = (x-b[:,0])/b[:,1]
    for _ in range(30):
        r = forward(z)-x
        if np.max(np.abs(r), initial=0.) < 1e-12:
            break
        jac = np.zeros((len(z), 4, 4))
        j = np.arange(4)
        jac[:,j,j] = b[:,1]+b[:,3]*np.roll(z,-1,axis=1)+3*b[:,4]*z*z
        jac[:,j,(j+1)%4] = b[:,2]+b[:,3]*z
        z -= np.linalg.solve(jac,r[...,None])[...,0]
    return z



def _branch(panel, z, k):
    u, v = (np.asarray(panel[j]) for j in ('u', 'v' + str(k)))

    def design(w, p):
        q = w + p[1:]
        h = q / (1 + p[0] * abs(q)) if k == 1 else q*q / (1 + p[0] * q*q)
        return _ones(np.c_[w, h])

    def residual(p):
        a = design(u, p)
        return (a @ _ls(a, v) - v).ravel()

    p = least_squares(residual, [.85 if k == 1 else .165, 0, 0, 0, 0],
                      bounds=([.03, -.2, -.2, -.2, -.2], [3, .2, .2, .2, .2])).x
    return design(z, p) @ _ls(design(u, p), v)


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']
    x = np.vstack((s['train']['x'], s['test']['x']))
    z = _calibrate(s['panel_a'], x)
    b, c = s['panel_b'], s['panel_c']
    q, r = (_branch(c, z, k) for k in (1, 2))
    return dict(z=z, L=_ones(z) @ _ls(_ones(b['u']), b['v']), Q=q, R=r,
                product=q.copy(), cosine=r.copy())
