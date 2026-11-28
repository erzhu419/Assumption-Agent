"""Calibrated panel transfer and supervised scalar and trajectory readouts."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import least_squares


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _panel(u, panel, key, kind, initial):
    x, y = panel['u'], panel[key]
    def design(t, x):
        return _aff(x + t[0]*kind(t[1]*x))
    def residual(t):
        a = design(t, x)
        return (a@_fit(a,y)-y).ravel()
    t = least_squares(residual, initial, max_nfev=120).x
    return design(t,u) @ _fit(design(t,x),y)


def _coords(st, z):
    b = st['panel_b']
    coef = _fit(_aff(b['u']),b['v'])
    u = np.linalg.solve(coef[1:].T, (z-coef[0]).T).T
    return _panel(u,b,'v',np.sin,[-.4,2.9])


def _quad(z):
    return np.column_stack([z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(z.shape[1]),2)])


def _predict(st,z,zt):
    a,b=np.c_[_aff(z),_quad(z)],np.c_[_aff(zt),_quad(zt)]
    return b@_fit(a,np.column_stack((st['train']['y'],st['train']['traj'])))


def run_node(inputs: dict) -> dict:
    st, enc = inputs['__input__'], inputs['node_00']
    n = len(st['train']['x'])
    z = _coords(st,np.concatenate((enc['z'],enc['zt'])))
    p = _predict(st,z[:n],z[n:])
    y,t = p[:,0],p[:,1:]
    return {'y_hat':y,'state_pred':t,'diagnostics_vec':np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])}
