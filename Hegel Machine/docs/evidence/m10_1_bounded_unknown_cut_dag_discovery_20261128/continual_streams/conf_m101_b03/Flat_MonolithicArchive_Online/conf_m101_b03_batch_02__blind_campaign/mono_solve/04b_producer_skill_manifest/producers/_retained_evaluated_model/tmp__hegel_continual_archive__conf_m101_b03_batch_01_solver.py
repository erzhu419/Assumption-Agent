"""Panel-calibrated prediction with a shared two-rate trajectory basis."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from sklearn.linear_model import ElasticNet


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x,y):
    return np.linalg.lstsq(x,y,rcond=None)[0]


def _coordinates(study, x):
    panel = study['panel_a']
    u,v = panel['u'],panel['v']
    a = _aff(u)
    b = _fit(a,v)
    for _ in range(20):
        w = np.sqrt(np.minimum(1, 0.05 / np.maximum(np.linalg.norm(a@b-v,axis=1),1e-12)))
        b = _fit(a*w[:,None],v*w[:,None])
    keep = np.linalg.norm(a@b-v,axis=1)<2
    u,v = u[keep],v[keep]
    def ac(q):
        r = np.roll(q,-1,axis=1)
        return np.stack((np.ones_like(q),q,r,q*r,q**3),axis=-1)
    a = ac(u)
    c = np.stack([_fit(a[:,i],v[:,i]) for i in range(4)])
    z = (x-c[:,0])/c[:,1]
    for _ in range(12):
        r = np.roll(z,-1,axis=1)
        jac = np.zeros((len(z),4,4))
        ii = np.arange(4)
        jac[:,ii,ii] = c[:,1]+c[:,3]*r+3*c[:,4]*z*z
        jac[:,ii,(ii+1)%4] = c[:,2]+c[:,3]*z
        z -= np.linalg.solve(jac,(np.sum(ac(z)*c,axis=-1)-x)[...,None])[...,0]
    p = study['panel_c']; u = p['u']; base = _aff(u)
    out = [z]
    for j, fun, bounds in ((1,np.tanh,(0.6,4)),(2,np.cos,(0.6,2))):
        v = p['v'+str(j)]
        def design(f): return np.column_stack((base,fun(f*u)))
        def loss(f):
            d = design(f)
            return np.sum((d@_fit(d,v)-v)**2)
        f = minimize_scalar(loss,bounds=bounds,method='bounded').x
        out.append(np.column_stack((_aff(z),fun(f*z)))@_fit(design(f),v))
    return out


def _quadratic(q):
    i,j = np.triu_indices(q.shape[1])
    return np.column_stack((q,q[:,i]*q[:,j]))


def _modes(times, traj):
    def design(p): return -np.expm1(-times[:,None]*p)
    def resid(p):
        d = design(p)
        return (d@_fit(d,traj.T)-traj.T).ravel()
    d = design(least_squares(resid,[0.4,1.8],bounds=([0.05,0.8],[0.8,5])).x)
    return d, _fit(d,traj.T).T


def _readout(f, y, n, alpha=0.00045):
    loc, scale = f[:n].mean(0), f[:n].std(0)
    z = (f-loc)/np.maximum(scale,1e-12)
    mu, s = y.mean(0),np.maximum(y.std(0),1e-12)
    pred = np.column_stack([ElasticNet(alpha=alpha,l1_ratio=.8,max_iter=10000,tol=5e-5).fit(z[:n],t).predict(z[n:]) for t in ((y-mu)/s).T])
    return pred*s+mu


def solve_batch(study):
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'],study['test']['x']))
    coords = _coordinates(study,x)
    # Include products between calibrated and auxiliary coordinates.
    f = _quadratic(np.column_stack(coords))
    h, modes = _modes(study['eval_times'],train['traj'])
    pred = _readout(f,np.column_stack((train['y'],modes)),n)
    y = pred[:,0]
    state = pred[:,1:]@h.T
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}
