"""Recover latent coordinates using two paired calibration panels."""
import numpy as np
from scipy.special import erf


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit_a(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    # The gross errors are coherent across channels; trim on row residuals.
    d = np.column_stack((np.ones(len(u)), u, np.sin(2.4*u)))
    keep = np.ones(len(u), dtype=bool)
    for _ in range(4):
        beta = np.linalg.lstsq(d[keep], v[keep], rcond=None)[0]
        err = np.linalg.norm(d @ beta - v, axis=1)
        med = np.median(err)
        new = err <= med + max(0.03, 5*np.median(np.abs(err-med)))
        if new.sum() < 9 or np.array_equal(new, keep):
            break
        keep = new
    u, v = u[keep], v[keep]
    opp = (np.arange(u.shape[1])+2) % u.shape[1]
    coeff = []
    for j,k in enumerate(opp):
        d = np.column_stack((np.ones(len(u)), u[:,j], np.sin(2.4*u[:,j]), np.sin(2.4*u[:,k])))
        coeff.append(np.linalg.lstsq(d,v[:,j],rcond=None)[0])
    return np.array(coeff).T, opp


def _inverse_a(v, params):
    (b,g,h,k), opp = params
    z = (v-b)/g
    u = z.copy()
    for _ in range(60):
        s = np.sin(2.4*u)
        u += 0.67*(z - (h*s + k*s[:,opp])/g - u)
    return u


def _basis_b(x):
    a,b,tied,(p,q,r) = B_GRAMMAR
    funcs = {'x':lambda k,z:z, 'sin':lambda k,z:np.sin(k*z),
             'cos':lambda k,z:np.cos(k*z), 'tanh':lambda k,z:np.tanh(k*z),
             'rat':lambda k,z:z/(1+k*z*z), 'cube':lambda k,z:z**3,
             'q':lambda k,z:z*z, 'atan':lambda k,z:np.arctan(k*z),
             'erf':lambda k,z:erf(k*z)}
    f,g = funcs[a](p,x),funcs[b](q,x)
    return _affine(f+r*g if tied else np.column_stack((f,g)))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    params = _fit_a(study['panel_a'])
    panel = study['panel_b']
    w = np.linalg.lstsq(_basis_b(np.asarray(panel['u'], dtype=float)), np.asarray(panel['v'], dtype=float), rcond=None)[0]
    def coordinates(split):
        return _basis_b(_inverse_a(np.asarray(study[split]['x'], dtype=float), params)) @ w
    return {'c': coordinates('train'), 'query': coordinates('test')}

B_GRAMMAR = ['sin', 'erf', True, [2.8540933890410667, 0.10090331578150344, -22.19450869429896]]
