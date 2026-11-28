"""Invert cyclic calibration and fit rational scalar and quadratic trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    coeff = []
    for j in range(u.shape[1]):
        a, b = u[:, j], u[:, (j+1) % u.shape[1]]
        design = np.column_stack((np.ones(len(a)), a, b, a*b, a**3))
        target = v[:, j]
        fit = least_squares(lambda c: design @ c - target, [0, 1, .3, .3, .15], loss='soft_l1', f_scale=.001).x
        err = target - design @ fit
        keep = abs(err - np.median(err)) <= max(6*np.median(abs(err - np.median(err))), .0007)
        coeff.append(_ls(design[keep], target[keep]) if keep.sum() >= 8 else fit)
    return np.array(coeff).T


def _inverse(x, coeff):
    x = np.asarray(x, dtype=float)
    bias, slope, cross, bilinear, cubic = coeff
    z = (x - bias) / slope
    j = np.arange(x.shape[1]); k = (j+1) % len(j)
    for _ in range(35):
        nxt = z[:, k]
        err = bias + slope*z + (cross + bilinear*z)*nxt + cubic*z**3 - x
        if not err.size or abs(err).max() < 1e-11:
            break
        jac = np.zeros((len(z), len(j), len(j)))
        jac[:, j, j] = slope + bilinear*nxt + 3*cubic*z*z
        jac[:, j, k] = cross + bilinear*z
        step = np.linalg.solve(jac, err[..., None])[..., 0]
        z -= np.clip(step, -.8, .8)
    return z


def _operator(panel):
    u, v = panel['u'], panel['v']
    def design(x, p):
        f = abs(x)**p[1]
        return np.c_[_affine(x), x*f/(1+p[0]*f)]
    def residual(p):
        a = design(u,p)
        return (a@_ls(a,v)-v).ravel()
    p = least_squares(residual, [1.1,2.5], bounds=([.02,.3],[8,6])).x
    w = _ls(design(u,p),v)
    return lambda x: design(x,p)@w


def _quad(x):
    i,j = np.triu_indices(x.shape[1]); return np.c_[_affine(x),x[:,i]*x[:,j]]


def _features(x,b):
    return np.c_[_quad(x),_quad(b)[:,1:],x*b,x*np.roll(b,-1,axis=1),b*np.roll(x,-1,axis=1)]


def _ridge(a,y,aq,lam):
    scale=np.maximum(np.linalg.norm(a,axis=0),1e-10)
    a=a/scale
    return (aq/scale) @ np.linalg.solve(a.T@a+lam*np.eye(a.shape[1]),a.T@y)


def _rational_readout(x,b,y,q,bq,lam=.001,penalty=.50):
    def numerator(z,h):
        return np.c_[_quad(h),z,z*h,z*np.roll(h,-1,axis=1),h*np.roll(z,-1,axis=1)]
    a,aq=numerator(x,b),numerator(q,bq)
    scale=np.maximum(np.linalg.norm(a,axis=0),1e-10)
    a,aq=a/scale,aq/scale
    sd=np.maximum(b.std(axis=0),.1)
    g,gq=b/sd,bq/sd
    eye=np.eye(a.shape[1])
    def fit(p):
        z=a/(1+(g@p)**2)[:,None]
        w=np.linalg.solve(z.T@z+lam*eye,z.T@y)
        return z@w-y,w
    def objective(p):
        return np.r_[fit(p)[0],penalty*p]
    starts=np.argsort(abs(g.T@fit(np.zeros(g.shape[1]))[0]))[-2:]
    solutions=[least_squares(objective,.25*np.eye(g.shape[1])[j],max_nfev=40) for j in starts]
    p=min(solutions,key=lambda obj:obj.cost).x
    return (aq@fit(p)[1])/(1+(gq@p)**2)


def solve_batch(study: dict) -> dict:
    tr=study['train']; c=_calibration(study['panel_a']); b=_operator(study['panel_b'])
    x=_inverse(tr['x'],c); q=_inverse(study['test']['x'],c)
    a=_features(x,b(x)); aq=_features(q,b(q))
    y=_rational_readout(x,b(x),tr['y'],q,b(q))
    state=_ridge(a,tr['traj'],aq,.0003)
    end=state[:,-1]
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()]) if len(y) else np.zeros(4)}
