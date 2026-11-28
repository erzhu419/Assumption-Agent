"""Panel-calibrated bilinear predictor for the scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _cal_terms(u):
    v = np.roll(u, -1, axis=-1)
    return np.stack([np.ones_like(u), u, v, u*v, u**3], axis=-1)


def _calibrate(panel, x):
    a = _cal_terms(panel['u'])
    v = panel['v']
    keep = np.ones(len(v), dtype=bool)
    for _ in range(6):
        c = np.array([_ls(a[keep,j], v[keep,j]) for j in range(4)])
        residual = np.linalg.norm(np.einsum('nij,ij->ni', a,c)-v, axis=1)
        take = residual < max(0.001, 3*np.median(residual))
        if take.sum() < 10 or np.array_equal(take, keep):
            break
        keep = take
    return np.array([least_squares(lambda q: (_cal_terms(q)*c).sum(1)-r, r,
                                  gtol=1e-11).x for r in x]).reshape(-1,4)


def _panel(panel, key, basis, start, bounds):
    u,v = panel['u'], panel[key]
    def design(x,p):
        return np.column_stack([np.ones(len(x)), basis(x,p)])
    def residual(p):
        a = design(u,p)
        return (a@_ls(a,v)-v).ravel()
    q = least_squares(residual, start, bounds=bounds, max_nfev=200).x
    c = _ls(design(u,q),v)
    return lambda x: design(x,q)@c


def _coords(study,x):
    z = _calibrate(study['panel_a'],x)
    def bshape(x,p):
        a, b = p[0]*x, p[1]*x
        return x + p[2]*a*np.exp(-a*a) + p[3]*b/(1+np.abs(b))
    def cshape(x,p):
        return np.c_[np.sin(p[0]*x),np.tanh(p[1]*x)]
    b = _panel(study['panel_b'],'v',bshape,[4.09593,.54798,.04681,-2.22252],
               ([.05,.05,-10,-10],[10,10,10,10]))
    c = _panel(study['panel_c'],'v1',cshape,[2.1,1.75],([.15,.15],[6,6]))
    return z,c(b(z))


def _features(z,h):
    return np.column_stack((np.ones(len(z)), z, h,
                            np.sum(z*h, axis=1), z[:,1]*h[:,0]))


def solve_batch(study: dict) -> dict:
    train=study['train']
    n=len(train['x'])
    d=_features(*_coords(study,np.r_[train['x'],study['test']['x']]))
    pred=d[n:] @ _ls(d[:n], np.c_[train['y'], train['traj']])
    y,state=pred[:,0],pred[:,1:]
    end=state[:,-1]
    diag=np.array([y.mean(),y.std(),end.mean(),end.std()])
    return dict(y_hat=y,state_pred=state,diagnostics_vec=diag)
