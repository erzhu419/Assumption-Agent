"""Panel-calibrated coordinates for the frozen prediction nodes."""
import numpy as np
from scipy.optimize import minimize_scalar


def _design(*x):
    return np.column_stack((np.ones(len(x[0])), *x))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-9)[0]


def _coordinates(s, x):
    p = s['panel_a']
    u, v = p['u'], p['v']
    w = []
    for j in range(u.shape[1]):
        f = _design(u[:, j], np.sin(2.4*u[:, j]), np.sin(2.4*u[:, (j+2)%4]))
        wc = _fit(f,v[:,j]); keep = np.linalg.norm(v,axis=1) < 4
        if keep.sum() < f.shape[1] + 2:
            keep = abs(f@wc-v[:,j]) < 1.5
        w.append(_fit(f[keep],v[keep,j]))
    b,d,h,c = np.array(w).T
    z = (x-b)/d
    for _ in range(35):
        zp = np.roll(z,2,axis=1)
        r = b+d*z+h*np.sin(2.4*z)+c*np.sin(2.4*zp)-x
        a = d+2.4*h*np.cos(2.4*z)
        q = 2.4*c*np.cos(2.4*zp)
        ar,qr,rr = [np.roll(t,2,axis=1) for t in (a,q,r)]
        z -= np.clip((ar*r-q*rr)/(a*ar-q*qr),-.3,.3)
    p = s['panel_b']; f = lambda x: _design(np.cos(2.6*x))
    b = f(z)@_fit(f(p['u']),p['v'])
    p = s['panel_c']; f = lambda x: _design(x*np.roll(x,-1,axis=1))
    c = f(z)@_fit(f(p['u']),p['v1'])
    def r(x,a): return _design(x,x*x/(1+a*x*x))
    u,v = p['u'],p['v2']
    a = minimize_scalar(lambda t: np.linalg.norm(r(u,t)@_fit(r(u,t),v)-v),bounds=(.001,1),method='bounded').x
    d = r(z,a)@_fit(r(u,a),v)
    return z,b,c,d


from itertools import combinations, combinations_with_replacement
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _library(s, x):
    z,b,c,d = _coordinates(s,x)
    h = np.cos(2.6*z); p = z*np.roll(z,-1,axis=1)
    groups = []
    def add(*f):
        g = np.column_stack(f)
        groups.append(g)
    def pairs(u,v):
        return (u[:,:,None]*v[:,None,:]).reshape(len(u),-1)
    def poly(u,j):
        return np.column_stack([u[:,k].prod(1) for k in combinations_with_replacement(range(u.shape[1]),j)])
    def norm(u):
        return np.maximum(u.std(0),1e-6)
    add(z); add(z**2); add(h); add(b,np.tanh(1.25*b),np.sin(2.15*b)); add(p); add(d); add(x)
    add(z**3,z**5,np.tanh(1.25*z),np.sin(2.15*z),np.sin(2.6*z))
    add(poly(z,2)); add(poly(z,3)); add(poly(z,4)); add(poly(h,2)); add(poly(b,2)); add(poly(c,2)); add(poly(d,2))
    add(pairs(z,h)); add(pairs(z,b)); add(pairs(z,c)); add(pairs(z,d)); add(pairs(b,c)); add(pairs(b,d)); add(pairs(c,d))
    add(pairs(z,h*h),pairs(z*z,h)); add(pairs(z,np.sin(2.6*z))); add(pairs(np.sin(2.6*z),h)); add(pairs(x,b))
    for v in (z,b,c,d):
        add(v/(1+.22*v*v),v/(1+1.25*v*v),np.arcsinh(v),np.tanh(2.15*v),np.sin(1.25*v),np.cos(2.15*v))
    for u,v in ((z,z),(z,b),(b,b),(b,c),(b,d)):
        if u is v:
            ij=list(combinations(range(4),2)); aa=u[:,[i for i,j in ij]]; bb=u[:,[j for i,j in ij]]
        else:
            aa=np.repeat(u,4,axis=1); bb=np.tile(v,(1,4))
        add(*[fun(a*(aa+sg*bb)) for a in (1.25,2.15,2.6) for sg in (-1,1) for fun in (np.sin,np.cos,np.tanh)])
    for u,v in ((z,b),(b,z),(z,c),(b,c),(z,d),(b,d),(c,d)):
        a = np.repeat(u,4,axis=1); bb=np.tile(v,(1,4))
        add(a/(1+.22*bb*bb),a/(1+1.25*bb*bb),a/(1+2.15*bb*bb))
    for j in range(4):
        add(z[:,j]); add(h[:,j]); add(p[:,j]); add(z[:,j]**3,np.sin(2.6*z[:,j]),np.tanh(2.15*z[:,j])); add(z[:,j,None]*h)
    return groups, (z,h,b,c,d,x)


def _kernels(groups, coords, n):
    a=[np.ones((len(coords[0]),len(coords[0])))]
    for f in groups:
        f=(f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-7)
        a.append(f@f.T/f.shape[1])
    for u in (coords[0],coords[1],coords[2],np.c_[coords[0],coords[1]],np.c_[coords[0],*coords[2:5]],coords[-1]):
        u=(u-u[:n].mean(0))/np.maximum(u[:n].std(0),1e-7)
        r2=cdist(u,u,'sqeuclidean')/u.shape[1]
        for scale in (.3,1.,2.5):
            r=np.sqrt(scale*r2); a.append((1+r+r*r/3)*np.exp(-r))
    return np.asarray(a)


def _head(ks,n,y):
    mean=y.mean(); sd=max(y.std(),1e-8); y=(y-mean)/sd
    g=np.ascontiguousarray(ks[:,:n,:n]); ff=g.reshape(len(g),-1); eye=np.eye(n)
    base=np.full(len(g)+1,.015); base[0]=.4; base[-1]=.025
    opts=[]
    for penalty in (.035,.25,1.2):
        logbase=np.log(base)
        def score(q):
            w=np.exp(q); k=(w[:-1]@ff).reshape(n,n)+w[-1]*eye
            chol=cho_factor(k,check_finite=False); alpha=cho_solve(chol,y,check_finite=False)
            ki=cho_solve(chol,eye,check_finite=False); dif=q-logbase
            v=.5*y@alpha+np.log(np.diag(chol[0])).sum()+.5*penalty*(dif@dif)
            d=.5*(ki-np.outer(alpha,alpha)); grad=w*np.r_[ff@d.ravel(),np.trace(d)]+penalty*dif
            return v,grad
        sol=minimize(score,logbase,method='L-BFGS-B',jac=True,bounds=[(-13,4)]*(len(base)-1)+[(-11,1)],options={'maxiter':140,'ftol':2e-8})
        w=np.exp(sol.x); k=(w[:-1]@ff).reshape(n,n)+w[-1]*eye
        chol=cho_factor(k,check_finite=False); alpha=cho_solve(chol,y,check_finite=False)
        h=cho_solve(chol,eye,check_finite=False).diagonal(); mse=np.mean((alpha/h)**2)
        pred=np.einsum('k,kij,j->i',w[:-1],ks[:,:,:n],alpha,optimize=True)
        opts.append((mse,pred))
    err=np.asarray([e for e,p in opts]); wt=(min(err)/np.maximum(err,1e-12))**4
    return mean+sd*np.average(np.array([p for e,p in opts]),axis=0,weights=wt)


def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; tr=s['train']; n=len(tr['x']); x=np.vstack((tr['x'],s['test']['x']))
    g,z=_library(s,x); k=_kernels(g,z,n)
    times=np.asarray(s['eval_times']); modes=-np.expm1(-times[:,None]*[1.8,.42])
    train=np.column_stack((tr['y'],_fit(modes,np.asarray(tr['traj']).T).T))
    f=np.column_stack([_head(k,n,y) for y in train.T]); f=(f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-8)
    f*=.52
    return {'train_features':f[:n].copy(),'test_features':f[n:].copy()}
