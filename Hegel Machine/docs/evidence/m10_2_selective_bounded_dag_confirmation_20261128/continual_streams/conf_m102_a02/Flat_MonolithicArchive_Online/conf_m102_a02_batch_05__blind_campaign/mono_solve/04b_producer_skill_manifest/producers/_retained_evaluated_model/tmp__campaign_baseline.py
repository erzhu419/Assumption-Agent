"""Robust calibration, shared panel response, and separate output regressions."""
import numpy as np
from scipy.optimize import least_squares


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _latents(study, x):
    panel = study['panel_a']
    u, v = panel['u'], panel['v']
    a, b = np.triu_indices(len(u), 1)
    du = u[a] - u[b]
    scale = np.nanmedian(np.divide(v[a]-v[b], du, where=abs(du)>1e-10,
                                  out=np.full_like(du, np.nan)), axis=0)
    shift = np.median(v-u*scale, axis=0)
    for j in range(u.shape[1]):
        e = abs(v[:, j] - shift[j] - u[:, j]*scale[j])
        keep = e <= max(.001, 6*np.median(e))
        shift[j], scale[j] = _ls(_aff(u[keep, j]), v[keep, j])
    panel = study['panel_b']
    u, v = panel['u'], panel['v']

    def basis(z, p):
        return _aff(z*(p[2]+z**2*(1+z**2*(p[0]+p[1]*z**2))))

    def residual(p):
        f = basis(u, p)
        return (f@_ls(f, v)-v).ravel()

    p = least_squares(residual, [-.45,.09,-.09], gtol=1e-11).x
    return basis((x-shift)/scale, p)@_ls(basis(u,p), v)


def _cross(h):
    i,j = np.triu_indices(h.shape[1],1)
    return np.c_[_aff(h), h[:,i]*h[:,j]]


def _scalar(h, q, y):
    f = _cross(h)
    pen = np.diag(np.r_[np.zeros(5), np.full(f.shape[1]-5,.03)])
    return _cross(q) @ np.linalg.solve(f.T@f+pen,f.T@y)


def _state(h, q, traj):
    _, _, right = np.linalg.svd(traj, full_matrices=False)
    time = right[0]
    return np.outer(_aff(q) @ _ls(_aff(h),traj@time), time)


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, q = train['x'], study['test']['x']
    h, hq = np.split(_latents(study,np.vstack((x,q))),[len(x)])
    y = _scalar(h,hq,train['y'])
    state = _state(h,hq,train['traj'])
    end = state[:,-1]
    diag = np.array([y.mean(),y.std(),end.mean(),end.std()]) if len(y) else np.zeros(4)
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':diag}
