import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from itertools import combinations, combinations_with_replacement


def _cosine_basis(x):
    return np.column_stack((np.ones(len(x)), np.cos(2.4*x)))


def _features(study, panel, b):
    x = panel['x']
    u = panel['u']
    c = _cosine_basis(x) @ np.linalg.lstsq(_cosine_basis(u), panel['v2'], rcond=None)[0]
    pb = study['panel_b']
    d = np.c_[np.ones(len(x)), x] @ np.linalg.lstsq(np.c_[np.ones(len(pb['u'])), pb['u']], pb['v'], rcond=None)[0]
    f = np.cos(2.4*x)
    h = np.sin(2.4*x)
    def cross(a, b):
        return (a[:, :, None]*b[:, None, :]).reshape(len(x), -1)
    def poly(a, k):
        return np.column_stack([np.prod(a[:, ix], axis=1) for ix in combinations_with_replacement(range(a.shape[1]), k)])
    def ratios(a, b):
        return (a[:, :, None]/(1+b[:, None, :]**2)).reshape(len(x), -1)
    groups = [np.ones((len(x),1)), b, c, x, x*x, poly(x,2), poly(x,3), poly(x,4), b*c, cross(b,c), poly(b,2), poly(c,2), cross(x,b), cross(x,c), cross(h,b), cross(h,c), h, f*f, cross(f,b), poly(f,2), cross(x*x,b), cross(x*x,c), cross(b,b*c), cross(c,b*c), cross(b,np.tanh(c)), cross(c,np.tanh(b)), ratios(b,c), ratios(c,b), np.tanh(b+c), np.tanh(b-c), np.tanh(b), np.tanh(c), d*c, d*b, cross(d,c), cross(d,b), np.cos(2.4*(x+np.roll(x,1,axis=1))), np.sin(2.4*(x+np.roll(x,1,axis=1))), np.cos(2.4*(d-d.mean(axis=0))), np.tanh(d), f*x, cross(f,h)]
    return x, groups


def _fit(study, panel, b, mode):
    x, groups = _features(study, panel, b)
    obs = study['train']
    y = np.column_stack((obs['y'], obs['traj']))
    n = len(y)
    means = y.mean(axis=0)
    y = y-means
    if mode in ('base', 'ridge'):
        if mode == 'base':
            a = np.column_stack(groups[:3]); return means+a[n:]@np.linalg.lstsq(a[:n],y,rcond=None)[0]
        groups = groups[:24]
    kernels = []
    columns = []
    for g in groups[1:]:
        sd = g[:n].std(axis=0)
        keep = sd>1e-8
        if not keep.any(): continue
        z = (g[:,keep]-g[:n,keep].mean(axis=0))/sd[keep]
        columns.append(z)
        kernels.append((z@z[:n].T)/z.shape[1])
    dx=x[:,None,:]-x[None,:n,:]; distance=np.sum(dx*dx,axis=-1)
    for ell in (.4,.8,1.5,3.):
        kernels.append(np.exp(-distance/(2*ell**2)))
    kernels=np.array(kernels)
    if mode == 'ridge':
        K = kernels.mean(axis=0)
        return means+K[n:]@np.linalg.solve(K[:n]+.004*np.eye(n),y)
    _, sv, vectors=np.linalg.svd(y,full_matrices=False)
    k=3 if mode=='eb3' else 4
    scale = sv[:k]**(.65 if mode=='eb4' else .9)
    z=(y@vectors[:k].T)/scale
    z=z/np.sqrt(np.mean(z*z))
    stack = np.concatenate((kernels[:,:n,:],np.eye(n)[None]),axis=0)
    flat = stack.reshape(len(stack),-1)
    rand = np.random.default_rng(5)
    def loss(w):
        p=np.exp(w); K=(p@flat).reshape(n,n)
        fac=cho_factor(K,check_finite=False); a=cho_solve(fac,z,check_finite=False)
        v=.5*np.sum(z*a)+k*np.log(np.diag(fac[0])).sum()
        M=k*cho_solve(fac,np.eye(n),check_finite=False)-a@a.T
        grad=.5*p*(flat@M.ravel())
        return v,grad
    best=None
    for i in range(2):
        init=np.r_[np.full(len(kernels),-4.),-5.]
        if i: init+=rand.normal(0,.8,len(init))
        res=minimize(loss,init,jac=True,method='L-BFGS-B',bounds=[(-13.,4.)]*len(init),options={'maxiter':125})
        if best is None or res.fun<best.fun: best=res
    p=np.exp(best.x); K=np.tensordot(p[:-1],kernels,axes=1)
    pred=means+K[n:]@np.linalg.solve(K[:n]+p[-1]*np.eye(n),y)
    return pred


_MODE = 'eb4'


def run_node(inputs: dict) -> dict:
    p=_fit(inputs['__input__'],inputs['node_00'],inputs['node_01']['b'],_MODE)
    y, state = p[:,0], p[:,1:]
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}
