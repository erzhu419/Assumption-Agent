"""Fit reference-panel coordinates and learn a compact spatial representation."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
from scipy.linalg import solve

def D(x): return np.c_[np.ones(len(x)),x]
def fit(a,b): return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def cyc(z): return z*np.roll(z,-1,axis=1)

def coordinates(s):
    p=s['panel_a']; u,v=p['u'],p['v']; x=np.r_[s['train']['x'],s['test']['x']]
    A=D(np.c_[u,cyc(u),u**3]); keep=np.ones(len(u),bool)
    for _ in range(12):
        w=fit(A[keep],v[keep]); e=np.linalg.norm(A@w-v,axis=1); new=e<max(.001,3*np.median(e))
        if new.sum()<18 or np.array_equal(new,keep): break
        keep=new
    u,v=u[keep],v[keep]
    def form(z,q): return D(np.c_[z,np.sin(q[0]*z),z*z])
    def err(q):
        a=form(u,q); return (a@fit(a,v)-v).ravel()
    q=least_squares(err,[2.15],bounds=(1,4)).x; w=fit(form(u,q),v)
    z=np.linalg.solve(w[1:5].T,(x-w[0]).T).T
    for _ in range(14):
        j=w[1:5]+(q[0]*np.cos(q[0]*z))[:,:,None]*w[5:9]+2*z[:,:,None]*w[9:13]
        z-=np.clip(np.linalg.solve(j.transpose(0,2,1),(form(z,q)@w-x)[...,None])[...,0],-.3,.3)
    p=s['panel_b']; H=lambda a:D(np.sign(a)*np.maximum(abs(a)-.35,0)**1.5); wb=fit(H(p['u']),p['v']); B=lambda a:H(a)@wb
    p=s['panel_c']; u=p['u']; wc=fit(D(cyc(u)),p['v1']); C=lambda a:D(cyc(a))@wc
    def g(a,q): return D(a+q[1]*a*a/(1+q[0]*a*a))
    def e(q):
        a=g(u,q); return (a@fit(a,p['v2'])-p['v2']).ravel()
    q=least_squares(e,[.25,.45],bounds=([0,-3],[8,3])).x; we=fit(g(u,q),p['v2']); E=lambda a:g(a,q)@we
    return z,B,C,E

def features(z,B,C,E):
    b=B(z); c=C(z); e=E(z)
    basic=[z,b,c,e,B(c),B(e),C(b),E(b),C(e),E(c)]
    f=basic.copy()
    for x in [z,b,c,e,z+b,z-b,b-.5*z,c+e,c-e,b+c,b-c,b+e,b-e,z+c,z-c,z+e,z-e]:
        f.extend([x*x,cyc(x),x*np.roll(x,2,axis=1)])
        for a in [.65,1.25,2.15]:
            f.extend([np.sin(a*x)-a*x,np.tanh(a*x)-a*x,x/(1+(a*x)**2)-x,np.arctan(a*x)**2])
    for i,a in enumerate(basic[:4]):
        for b1 in basic[i:4]:
            f.append((a[:,:,None]*b1[:,None,:]).reshape(len(z),-1))
    return np.column_stack(f)

def head(F,Y):
    n=len(Y); A=F[:n]; mean=A.mean(0); scale=np.maximum(A.std(0),.015); X=(F-mean)/scale; A=X[:n]; m=Y.mean(0); y=Y-m
    # Sparse Bayesian regression, with shared scale evidence within each response.
    result=[]
    for k in range(y.shape[1]):
        t=y[:,k]; means=[]
        for power in [0.,.5,1.]:
            a=scale**power; a/=np.median(a); Q=A*a; beta=np.ones(Q.shape[1])/Q.shape[1]; noise=max(.0001*np.var(t),1e-9)
            for _ in range(60):
                W=Q*beta; inv=solve(W@Q.T+noise*np.eye(n),np.c_[t,Q],assume_a='pos'); w=beta*(Q.T@inv[:,0]); lev=np.sum(Q*inv[:,1:],axis=0); proposed=abs(w)/np.sqrt(np.maximum(lev,1e-10)); beta=np.maximum(1e-11,.35*beta+.65*proposed)
            means.append(X@(a*w)+m[k])
        result.append(np.mean(means,axis=0))
    return np.column_stack(result)

def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; z,B,C,E=coordinates(s); tr=s['train']; t=s['eval_times']; T=tr['traj'].T
    def basis(q): return -np.expm1(-t[:,None]*q)
    def err(q):
        a=basis(q); return (a@fit(a,T)-T).ravel()
    a=basis(least_squares(err,[1.8,.42],bounds=([1.,.15],[3.,.85])).x)
    modes=fit(a,T).T; n=len(modes); F=features(z,B,C,E)
    p=head(F,np.c_[tr['y'],modes]); scale=np.maximum(modes.std(0),1e-5)
    return {'y_hat':p[n:,0].copy(),'trajectory_design':np.c_[p[:,1:]/scale,.035*z,.035*B(z)]}
