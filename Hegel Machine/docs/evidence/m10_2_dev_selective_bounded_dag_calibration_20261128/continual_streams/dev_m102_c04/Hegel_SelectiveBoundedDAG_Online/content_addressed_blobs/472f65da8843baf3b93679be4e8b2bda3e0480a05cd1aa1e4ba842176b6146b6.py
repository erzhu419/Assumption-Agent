"""Calibrated multi-output regression for the public study."""
import numpy as np
from itertools import combinations, combinations_with_replacement
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-9)[0]


def _coordinates(study, x):
    a, b, c = (study[k] for k in ('panel_a','panel_b','panel_c'))
    u, v = a['u'], a['v']
    mask = np.linalg.norm(v-u, axis=1) < 2
    if mask.sum() < 8:
        mask = np.ones(len(u), bool)
    w = np.stack([_ls(_aff(u[mask, j:j+1]), v[mask, j]) for j in range(4)], axis=1)
    q = (x-w[0])/w[1]
    m = _ls(_aff(b['u']), b['v'])
    z = _aff(q) @ m
    def quadratic(p):
        return _aff(np.stack([p[:,i]*p[:,j] for i,j in ((0,1),(0,3),(1,2),(2,3))], axis=1))
    wc = _ls(quadratic(c['u']), c['v1'])
    wd = _ls(_aff(np.cos(2.4*c['u'])), c['v2'])
    return q,z,quadratic(q)@wc,_aff(np.cos(2.4*q))@wd,quadratic(z)@wc,_aff(np.cos(2.4*z))@wd


def _kernels(p, n):
    q,z,c,d,cz,dz = p
    ks = []
    def add(f):
        f = (f - f[:n].mean(axis=0))/np.maximum(f[:n].std(axis=0), 1e-9)
        ks.append(f@f.T/f.shape[1])
    def mon(x, deg):
        return np.stack([np.prod(x[:, ix],axis=1) for ix in combinations_with_replacement(range(4),deg)],axis=1)
    ks.append(np.ones((len(q),len(q))))
    add(q)
    for deg in range(2,5):
        for x in (q,z): add(mon(x,deg))
    add(np.column_stack((c,d)))
    add(np.column_stack((cz,dz)))
    for x in (q,z):
        pair = np.stack([x[:,i]+s*x[:,j] for i,j in combinations(range(4),2) for s in (-1,1)],axis=1)
        for a in (1.25,2.4,4.0):
            h = np.column_stack((x,pair))
            add(np.column_stack((np.sin(a*h),np.cos(a*h))))
    for x in (q,z,c,d):
        delta = x[:,None]-x[None,:]
        dist = ((delta/np.maximum(x[:n].std(0),1e-9))**2).sum(2)
        for scale in (0.12, 0.5, 2.0):
            ks.append(np.exp(-scale*dist))
    return np.asarray(ks)


def _fit(kernels, target, separate=True):
    n = len(target)
    mu = target.mean(0)
    sd = np.maximum(target.std(0),1e-8)
    t = (target-mu)/sd
    K = np.concatenate((kernels[:,:n,:n],np.eye(n)[None]),axis=0).copy()
    Kflat = K.reshape(len(K),-1)
    Qflat = kernels[:,n:,:n].copy().reshape(len(K)-1,-1)
    def fit_one(t):
        m=t.shape[1]
        prior=np.full(len(K),-3.0); prior[-1]=-5.0
        def loss(v):
            a=np.exp(v)
            L=cho_factor((a@Kflat).reshape(n,n),check_finite=False)
            inv=cho_solve(L,np.eye(n),check_finite=False)
            alpha=inv@t
            g=(m*inv-alpha@alpha.T).reshape(-1)
            val=.5*np.sum(alpha*t)+m*np.log(L[0].diagonal()).sum()+.025*np.sum((v-prior)**2)
            return val, .5*a*(Kflat@g)+.05*(v-prior)
        a=np.exp(minimize(loss,prior,jac=True,method='L-BFGS-B',bounds=[(-12,6)]*len(prior),options={'maxiter':180}).x)
        alpha=cho_solve(cho_factor((a@Kflat).reshape(n,n),check_finite=False),t,check_finite=False)
        return (a[:-1]@Qflat).reshape(len(kernels[0])-n,n)@alpha
    if separate:
        out = np.column_stack([fit_one(t[:, i:i+1]) for i in range(t.shape[1])])
    else:
        out = fit_one(t)
    return out*sd+mu


def solve_batch(study):
    train=study['train']; n=len(train['x']); x=np.concatenate((train['x'],study['test']['x']))
    with threadpool_limits(limits=1):
        coords=_coordinates(study,x)
        ks=_kernels(coords,n)
        basis=np.linalg.svd(train['traj'],full_matrices=False)[2][:3]
        target=np.column_stack((train['y'],train['traj']@basis.T))
        ext=np.concatenate((ks,_graph_kernels(study,coords,n)))
        result=(_fit(ks,target)+_fit(ext,target)+_fit(ext,target,separate=False))/3
    y=result[:,0]; state=result[:,1:]@basis
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}

def _graph_kernels(study, coords, n):
    q,z,c,d,cz,dz=coords
    panel=study['panel_c']; pb=study['panel_b']; a0,a1=np.abs(study['telemetry_vec'][:2]); a0=max(a0,.01); a1=max(a1,.001)
    ix=((0,1),(0,3),(1,2),(2,3))
    def quadratic(p): return _aff(np.stack([p[:,i]*p[:,j] for i,j in ix],axis=1))
    w=_ls(quadratic(panel['u']),panel['v1']); v=_ls(_aff(np.cos(2.4*panel['u'])),panel['v2']); b=_ls(_aff(pb['u']),pb['v'])
    B=lambda p:_aff(p)@b
    C=lambda p:quadratic(p)@w
    D=lambda p:_aff(np.cos(2.4*p))@v
    def standard(x): return (x-x[:n].mean(0))/np.maximum(x[:n].std(0),1e-10)
    def gram(x):
        x=standard(x); return x@x.T/x.shape[1]
    def powers(x,k): return np.stack([np.prod(x[:,i],axis=1) for i in combinations_with_replacement(range(4),k)],axis=1)
    groups=[]
    def add(group): groups.append(np.mean(group,axis=0))
    add([gram(powers(x,k)) for x in (q,z) for k in (5,6)])
    fns=(lambda p:np.sin(a0*p),lambda p:np.tanh(a0*p),lambda p:p/(1+a1*p*p),lambda p:p+a1*p*p,lambda p:np.sin(a0*p)+a1*p*p,lambda p:np.cos(a0*p)+a1*p)
    for f in fns:
        p=f(q); u=B(p)
        add([gram(x) for x in (C(p),D(p),C(u),D(u))]);add([gram(powers(x,k)) for x in (p,u,f(z)) for k in (2,3)])
    levels=(c,d,cz,dz,C(d),D(c),C(dz),D(cz))
    for x in (q,z):
        for j in (0,4):
            aug=levels[j:j+4]; gx=gram(x)
            add([gx*gram(p) for p in aug]);add([gram(p*x) for p in aug]);add([gram(x/(1+a1*p*p)) for p in aug]);add([gram(p/(1+a1*x*x)) for p in aug]);add([gram(np.sin(a0*(x+p))) for p in aug]);add([gram(np.cos(a0*(x-p))) for p in aug]);add([gram(powers(p,2)) for p in aug]);add([gram(powers(p,3)) for p in aug]);
    for p,r in ((c,d),(cz,dz),(c,dz),(cz,d),(q,z)):
        add([gram(p)*gram(r),gram(p*r)]);add([gram(p/(1+a1*r*r)),gram(r/(1+a1*p*p))]);add([gram(np.sin(a0*(p+r))),gram(np.cos(a0*(p-r)))])
    for levels0 in ((q,z),(c,d),(cz,dz),(q,c,d),(q,cz,dz)):
        g=sum(gram(p) for p in levels0)/len(levels0); ds=np.maximum(g.diagonal()[:,None]+g.diagonal()[None,:]-2*g,0)
        for k in (0.3,0.8,1.6): groups.append(np.exp(-k*ds));
    return np.asarray(groups)
