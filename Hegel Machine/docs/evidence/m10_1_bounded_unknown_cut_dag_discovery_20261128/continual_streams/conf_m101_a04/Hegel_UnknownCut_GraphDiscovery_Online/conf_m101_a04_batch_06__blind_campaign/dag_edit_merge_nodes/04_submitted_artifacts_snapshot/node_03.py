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



def _adaptive(study, panel, b, variant):
    rank, power, family, refinement, target = variant
    x, groups = _features(study, panel, b)
    n = len(study['train']['y'])
    raw = np.column_stack((study['train']['y'], study['train']['traj']))
    mean = raw.mean(axis=0)
    Y = raw-mean
    if target == 1:
        source = Y[:,1:]
    elif target == 2:
        source = np.c_[Y[:,0]/Y[:,0].std(), Y[:,1:]/np.sqrt(np.mean(Y[:,1:]**2))]
    else:
        source = Y
    U, sv, V = np.linalg.svd(source, full_matrices=False)
    r = min(rank,len(sv))
    Z = U[:,:r]*sv[:r]**(1-power)
    Z /= np.sqrt(np.mean(Z**2))
    if family:
        c = groups[2]
        f = np.cos(2.4*x)
        h = np.sin(2.4*x)
        def cross(a, q):
            return (a[:,:,None]*q[:,None,:]).reshape(len(a),-1)
        def powers(a, k):
            return np.column_stack([np.prod(a[:,ii],axis=1) for ii in combinations_with_replacement(range(a.shape[1]),k)])
        q = np.column_stack((b,c))
        p = np.column_stack([x[:,i]*x[:,j] for i,j in combinations(range(x.shape[1]),2)])
        groups += [cross(p,f),cross(p,c),cross(p,b),cross(f,c),cross(f,h),cross(f,x),cross(h,x),powers(q,3),cross(x,powers(q,2)),cross(x**3,f),cross(powers(x,2),f),cross(powers(x,2),h),cross(x,f*f),cross(p,f*f),cross(powers(f,2),b),cross(powers(f,2),c),cross(powers(f,2),x)]
        for g in (.5,1.,2.):
            groups += [np.tanh(g*q), np.sin(g*q), np.cos(g*q),cross(np.tanh(g*b),c),cross(np.tanh(g*c),b),cross(np.sin(g*b),c),cross(np.sin(g*c),b)]
        if family==2:
            groups += [cross(np.cos(j*x),p) for j in (1.2,1.8,3.,3.6,4.8)]
            groups += [cross(np.sin(j*x),p) for j in (1.2,1.8,3.,3.6,4.8)]
            groups += [np.column_stack([fun(g*(q[:,i]+d*q[:,j])) for i,j in combinations(range(q.shape[1]),2)]) for fun in (np.sin,np.cos,np.tanh) for d in (-1,1) for g in (1.,2.)]
    cols=[]; kernels=[]
    for g in groups[1:]:
        sd=g[:n].std(axis=0); keep=sd>1e-8
        if not keep.any(): continue
        a=(g[:,keep]-g[:n,keep].mean(axis=0))/sd[keep]; a/=np.sqrt(a.shape[1]);cols.append(a)
        kernels.append(a@a[:n].T)
    dist=np.sum((x[:,None,:]-x[None,:n,:])**2,axis=2)
    for ell in (.4,.8,1.5,3.): kernels.append(np.exp(-dist/(2*ell**2)))
    def learn(K,z,start=None):
        stack=np.concatenate((K[:,:n,:],np.eye(n)[None]),axis=0).reshape(len(K)+1,-1)
        nz=z.shape[1]; eye=np.eye(n)
        def fun(w):
            p=np.exp(w); cov=np.einsum('i,ij->j',p,stack).reshape(n,n)
            fac=cho_factor(cov,check_finite=False); a=cho_solve(fac,z,check_finite=False)
            G=nz*cho_solve(fac,eye,check_finite=False)-a@a.T
            return .5*np.sum(z*a)+nz*np.log(np.diag(fac[0])).sum(), .5*p*np.einsum('ij,j->i',stack,G.ravel())
        if start is None: start=np.r_[np.full(len(K),-4.),-5.]
        opt=minimize(fun,start,jac=True,method='L-BFGS-B',bounds=[(-15.,5.)]*len(start),options={'maxiter':165,'ftol':2e-9})
        return np.exp(opt.x)
    K=np.array(kernels); weights=learn(K,Z)
    if refinement:
        order=np.argsort(weights[:len(cols)])[::-1]; additions=[]
        for j in order[:refinement]:
            a=cols[j]
            additions.extend([np.outer(v,v[:n])*a.shape[1] for v in a.T])
        K=np.concatenate((K,np.array(additions)),axis=0)
        start=np.log(np.r_[weights[:-1],np.full(len(additions),1e-4),weights[-1]])
        weights=learn(K,Z,start)
    cov=np.einsum('i,ijk->jk',weights[:-1],K)
    return mean+cov[n:]@np.linalg.solve(cov[:n]+weights[-1]*np.eye(n),Y)


def run_node(inputs: dict) -> dict:
    s,p,b=inputs['__input__'],inputs['node_00'],inputs['node_01']['b']
    a=_fit(s,p,b,'eb3') if _TRAJ_CONFIG is None or _SCALAR_CONFIG is None else None
    y=a[:,0] if _SCALAR_CONFIG is None else _adaptive(s,p,b,_SCALAR_CONFIG)[:,0]
    if _TRAJ_CONFIG == _SCALAR_CONFIG and _TRAJ_CONFIG is not None:
        pred=_adaptive(s,p,b,_TRAJ_CONFIG)
        y,state=pred[:,0],pred[:,1:]
    elif _TRAJ_CONFIG is None:
        state=a[:,1:]
    elif isinstance(_TRAJ_CONFIG,list):
        state=np.mean([_adaptive(s,p,b,q)[:,1:] if q is not None else _fit(s,p,b,'eb3')[:,1:] for q in _TRAJ_CONFIG],axis=0)
    else:
        state=_adaptive(s,p,b,_TRAJ_CONFIG)[:,1:]
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}


_TRAJ_CONFIG = (2, 0.6, 1, 0, 1)
_SCALAR_CONFIG = None
