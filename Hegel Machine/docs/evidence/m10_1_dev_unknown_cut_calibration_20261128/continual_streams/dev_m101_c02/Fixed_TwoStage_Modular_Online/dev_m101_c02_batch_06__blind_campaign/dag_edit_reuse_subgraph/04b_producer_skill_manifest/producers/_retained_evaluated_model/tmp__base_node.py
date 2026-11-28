import numpy as np
from scipy.optimize import least_squares

def fit_a(panel):
    u, v = panel['u'], panel['v']
    n = np.roll(u, -1, axis=1)
    D = np.stack((np.ones_like(u), u, n*(u-1), u**3), axis=-1)
    w = np.ones(len(u))
    for _ in range(24):
        c = np.array([np.linalg.lstsq(D[:,j]*w[:,None], v[:,j]*w,rcond=None)[0] for j in range(4)]).T
        e = np.linalg.norm(np.einsum('njk,kj->nj',D,c)-v, axis=1)
        w = 1/np.sqrt(1+(e/max(0.001, 2*np.median(e)))**2)
    keep = np.argsort(e)[:-3]
    return np.array([np.linalg.lstsq(D[keep,j], v[keep,j],rcond=None)[0] for j in range(4)]).T

def inv_a(x,c):
    b,d,q,g = c
    A=np.diag(d); A[np.arange(4),(np.arange(4)+1)%4]=-q
    u=np.linalg.solve(A,(x-b).T).T
    for i, row in enumerate(x):
        u[i]=least_squares(lambda y:b+d*y+q*(y-1)*np.roll(y,-1)+g*y**3-row,u[i],gtol=1e-12,ftol=1e-12,xtol=1e-12).x
    e=b+d*u+q*(u-1)*np.roll(u,-1,axis=1)+g*u**3-x
    return u


def _basis_b(x, p):
    a, f = p
    w = x + a*np.sin(f*x)
    return np.column_stack((np.ones(len(x)), w, x*np.roll(x,-1,axis=1)))

def _fit_b(panel):
    u, v = panel['u'], panel['v']
    def fit(p):
        d = _basis_b(u,p)
        return d, np.linalg.lstsq(d,v,rcond=None)[0]
    def residual(p):
        d, c = fit(p)
        return (d@c-v).ravel()
    p = least_squares(residual,[-.4,2.8],bounds=([-2,1],[1,5])).x
    return p, fit(p)[1]

def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a = fit_a(study['panel_a'])
    p, b = _fit_b(study['panel_b'])
    return {key: _basis_b(inv_a(study[split]['x'],a),p)@b
            for key, split in [('z_tr','train'),('z_te','test')]}
