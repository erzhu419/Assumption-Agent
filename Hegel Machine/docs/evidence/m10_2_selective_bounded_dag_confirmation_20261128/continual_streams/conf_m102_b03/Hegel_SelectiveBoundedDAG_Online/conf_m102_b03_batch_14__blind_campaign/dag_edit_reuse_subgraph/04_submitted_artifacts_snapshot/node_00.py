"""Recover calibrated coordinates and evaluate the two response branches."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _ones(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, y):
    return np.linalg.lstsq(a, y, rcond=None)[0]


def _align(p, x):
    u, v = p['u'], p['v']
    # Gross contamination in the reference readings is shared across channels.
    keep = np.max(abs(v), axis=1) < 3.5
    u, v = u[keep], v[keep]
    best = None
    for typ in ('sin','tanh','rat'):
        def g(a):
            if typ == 'sin': return np.sin(a)
            if typ == 'tanh': return np.tanh(a)
            return a/(1+a*a)
        def basis(w, a): return _ones(np.c_[w,g(a[0]*w),g(a[1]*np.roll(w,2,axis=1))])
        def coeff(w,a,y):
            B=basis(w,a); return np.stack([_ls(B[:,[0,1+j,1+(j+2)%4,5+j,9+j]], y[:,j]) for j in range(4)])
        def calc(w,a,c):
            B=basis(w,a); return np.column_stack([B[:,[0,1+j,1+(j+2)%4,5+j,9+j]]@c[j] for j in range(4)])
        def resid(a): return (calc(u,a,coeff(u,a,v))-v).ravel()
        for a0 in ([2.4,2.4],[1.,1.],[3.2,2.4]):
            a=least_squares(resid,a0,bounds=(.2,4.5)).x
            err=np.linalg.norm(resid(a)); c=coeff(u,a,v)
            if best is None or err < best[0]: best=(err,typ,a,c)
    err,typ,a,c=best
    def g(t):
        if typ=='sin': return np.sin(t)
        if typ=='tanh': return np.tanh(t)
        return t/(1+t*t)
    def dg(t):
        if typ=='sin': return np.cos(t)
        if typ=='tanh': return 1-np.tanh(t)**2
        return (1-t*t)/(1+t*t)**2
    z=(x-c[:,0])/c[:,1]; j=np.arange(4)
    for _ in range(30):
        w=np.roll(z,2,axis=1); r=c[:,0]+c[:,1]*z+c[:,2]*w+c[:,3]*g(a[0]*z)+c[:,4]*g(a[1]*w)-x
        if np.max(abs(r))<1e-12: break
        J=np.zeros((len(z),4,4)); J[:,j,j]=c[:,1]+c[:,3]*a[0]*dg(a[0]*z); J[:,j,(j+2)%4]=c[:,2]+c[:,4]*a[1]*dg(a[1]*w)
        z-=np.clip(np.linalg.solve(J,r[...,None])[...,0],-.4,.4)
    return z


def _pair(w):
    return w[:,[0,0,1,2]]*w[:,[1,3,2,3]]


def _branch(u,v,z,k):
    best=None
    for typ in ('sin','tanh','rat','soft','cos'):
        def g(t):
            if typ=='sin': return np.sin(t)
            if typ=='tanh': return np.tanh(t)
            if typ=='rat': return t/(1+t*t)
            if typ=='cos': return np.cos(t)
            return t/(1+abs(t))
        for structure in (0,1,2):
            def design(w,p):
                h=g(p[0]*w+(p[1:] if len(p)>1 else 0)); b=_pair(w) if k==2 else np.empty((len(w),0))
                if structure==0:return _ones(np.c_[b,w,h])
                if structure==1:return _ones(np.c_[b,w,h,w/(1+.85*abs(w))])
                return _ones(np.c_[b,h])
            for shift in (False,True):
                for start in (.8,1.5,2.4,3.2):
                    def res(p):
                        A=design(u,p); return (A@_ls(A,v)-v).ravel()
                    p=least_squares(res,[start]+([0]*4 if shift else []),bounds=([.09]+([-.65]*4 if shift else []),[4.5]+([.65]*4 if shift else [])),max_nfev=90).x
                    A=design(u,p); P=np.linalg.pinv(A); E=A@(P@v)-v; H=(A*P.T).sum(1)
                    score=np.mean((E/np.maximum(1-H[:,None],1e-5))**2)**.5*(1.15 if shift else 1)
                    if best is None or score<best[0]: best=(score,design(z,p)@(P@v),typ,structure,p)
    return best


def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; z=_align(s['panel_a'],np.vstack((s['train']['x'],s['test']['x'])))
    b,c=s['panel_b'],s['panel_c']; l=_ones(z)@_ls(_ones(b['u']),b['v'])
    q,r=(_branch(c['u'],c['v'+str(k)],z,k)[1] for k in (1,2))
    return dict(z=z,L=l,Q=q,R=r,product=r,cosine=q)
