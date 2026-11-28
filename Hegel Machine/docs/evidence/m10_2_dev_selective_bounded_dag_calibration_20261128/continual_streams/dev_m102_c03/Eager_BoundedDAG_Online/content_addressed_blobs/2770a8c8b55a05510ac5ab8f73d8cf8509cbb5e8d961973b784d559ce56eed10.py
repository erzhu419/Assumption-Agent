import numpy as np
from scipy.optimize import minimize_scalar

def ls(A,Y): return np.linalg.lstsq(A,Y,rcond=None)[0]
def latent(s,x):
 p=s['panel_a']; u,v=p['u'],p['v']; H=np.c_[np.ones(len(u)),u]; keep=np.linalg.norm(v-H@ls(H,v),axis=1)<4; u,v,H=u[keep],v[keep],H[keep]
 def f(k,ret=False):
  w=np.sin(k*u)-k*u; t=u*np.roll(u,-1,axis=1); C=np.stack([ls(np.c_[H,t[:,j],w[:,j]],v[:,j]) for j in range(4)],1); e=H@C[:5]+t*C[5]+w*C[6]-v
  return (C,np.std(e)) if ret else np.mean(e*e)
 k=float(s['telemetry_vec'][0]); c,e=f(k,True)
 z=(x-c[0])@np.linalg.inv(c[1:5]); j=np.arange(4)
 for _ in range(20):
  w=np.roll(z,-1,axis=1); r=c[0]+z@c[1:5]+c[5]*z*w+c[6]*(np.sin(k*z)-k*z)-x; J=np.broadcast_to(c[1:5].T,(len(z),4,4)).copy(); J[:,j,j]+=c[5]*w+c[6]*k*(np.cos(k*z)-1);J[:,j,(j+1)%4]+=c[5]*z; d=np.linalg.solve(J,r[...,None])[...,0];z-=d/np.maximum(1,np.linalg.norm(d,axis=1,keepdims=True))
 return z,(k,e,c)

def maps(s,z):
 p=s['panel_b']; u,v=p['u'],p['v']; h=np.c_[np.ones(len(u)),u]; b=ls(h,v)
 p=s['panel_c'];u=p['u'];h=np.c_[np.ones(len(u)),u]; ones=np.ones(len(z));
 def f(k,fun,key):
  F=np.c_[h,fun(u,k)] if key=='v1' else np.c_[h[:,0],fun(u,k)]; C=ls(F,p[key]);return np.mean((F@C-p[key])**2),C
 fun=lambda x,k:x/(1+k*x*x); k=minimize_scalar(lambda k:f(k,fun,'v1')[0],bounds=(.1,3),method='bounded').x; _,c=f(k,fun,'v1'); _,d=f(2.4,lambda x,k:np.cos(k*x),'v2');return (np.c_[ones,z]@b,np.c_[ones,z,fun(z,k)]@c,np.c_[ones,np.cos(2.4*z)]@d),(b,c,d,np.std(h@b-v) if len(v)==len(h) else 0)

from itertools import combinations, combinations_with_replacement
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.optimize import least_squares


def _poly(x, k):
    return np.column_stack([np.prod(x[:, p], axis=1) for p in combinations_with_replacement(range(x.shape[1]), k)])


def _readout(study):
    tr = study['train']
    n = len(tr['x'])
    raw = np.vstack((tr['x'], study['test']['x']))
    z, _ = latent(study, raw)
    (b,c,d),(bc,_,_,_) = maps(study,z)
    bi = (z-bc[0]) @ np.linalg.pinv(bc[1:])
    t = np.asarray(study['eval_times'])
    traj = np.asarray(tr['traj'])
    def time_basis(r):
        return -np.expm1(-t[:,None]*r)
    def t_resid(r):
        q = time_basis(r)
        return (q @ ls(q,traj.T)-traj.T).ravel()
    q = time_basis(least_squares(t_resid,[.42,1.81],bounds=([.08,.95],[.94,4.])).x)
    y = np.column_stack((tr['y'],ls(q,traj.T).T))
    ym = y.mean(0)
    centered = y-ym
    # Model comparisons use only the incoming training targets. For each
    # ridge fit compute its exact leave-one-out residual via the hat diagonal.
    options = [[],[],[]]
    full_groups = []
    def standard(A):
        A=A-A[:n].mean(0)
        return A/np.maximum(np.linalg.norm(A[:n],axis=0),1e-9)
    def add(p, e):
        mae=np.mean(np.abs(e),axis=0)
        for j in range(3):
            if np.isfinite(mae[j]): options[j].append((mae[j],p[:,j].copy()))
    def kernel(K,regs):
        ev,U=eigh(K[:n],check_finite=False)
        ev=np.maximum(ev,0)
        uy=U.T@centered
        u2=U*U
        V=K[n:]@U
        for r in regs:
            inv=1/(ev+r)
            c=uy*inv[:,None]
            e=(U@c)/np.maximum((u2@inv)[:,None]-1/(n*r),1e-10)
            add(V@c+ym,e)
    def features(A,regs=(.000002,.00008,.001,.012,.10,.6)):
        A=standard(A)
        K=A@A[:n].T
        kernel(K,tuple(r*max(1,A.shape[1]/24) for r in regs))
    def radial(x):
        x=standard(x)*np.sqrt(n)
        D=np.sum((x[:,None]-x[None,:n])**2,axis=2)
        D/=max(np.median(D[:n]),1e-9)
        # Remove the intercept from the kernel before computing the LOO hat.
        for gamma in (.12,.35,1.,2.8,7.):
            K=np.exp(-gamma*D)
            K=K-K[:n].mean(0)-K.mean(1)[:,None]+K[:n].mean()
            kernel(K,(.000002,.00006,.001,.014,.12))
    views=[z,b,bi,c,d,raw]
    w=np.asarray(study['telemetry_vec'])
    for g in (float(w[0]),.55,1.):
        views.extend((z+g*c,z-g*c,z+g*d,z-g*d,b+g*c,b+g*d))
    for i,x in enumerate(views):
        ii,jj=np.array(list(combinations(range(4),2))).T
        l,r=x[:,ii],x[:,jj]
        prod=l*r
        p2=_poly(x,2)
        p3=_poly(x,3)
        H=np.c_[x,p2]
        features(H)
        features(np.c_[H,p3])
        features(np.c_[H,p3,_poly(x,4)])
        radial(x)
        if i>=6: continue
        Z=np.c_[x,l+r,l-r]
        families=[x,p2,p3]
        for k in (.65,1.25,1.65,2.15,2.4,3.1,4.,5.2):
            u=np.c_[np.sin(k*x),np.cos(k*x)]
            v=np.c_[np.sin(k*Z),np.cos(k*Z)]
            a=np.c_[np.tanh(k*x),x/(1+k*x*x)]
            p=prod/(1+k*(l*l+r*r))
            features(np.c_[H,u])
            features(np.c_[x,v])
            features(np.c_[H,u,_poly(u,2)])
            features(np.c_[H,a,_poly(a,2),p])
            families.extend((v,a,p))
        full_groups.append(np.column_stack(families))
    for pair in ([z,c],[z,d],[c,d],[z,b],[z,c,d]):
        X=np.column_stack(pair)
        radial(X)
        features(np.c_[X,_poly(X,2)])
    for g in (full_groups[0],np.column_stack(full_groups[:3]),np.column_stack(full_groups[:5])):
        features(g)
        # Sparse nonlinear terms: iteratively reweight the feature prior.
        A=standard(g); B=A[:n]; C=A[n:]; eye=np.eye(n)
        for j in range(3):
            for lam in (.004,.035,.22):
                weights=np.ones(B.shape[1]); target=centered[:,j]
                for it in range(14):
                    K=(B*weights)@B.T
                    fact=cho_factor(K+lam*eye,check_finite=False)
                    inv=cho_solve(fact,eye,check_finite=False)
                    dual=inv@target
                    coeff=weights*(B.T@dual)
                    if it in (3,7,13):
                        e=dual/np.maximum(inv.diagonal()-1/(n*lam),1e-10)
                        options[j].append((np.mean(abs(e))*1.13,C@coeff+ym[j]))
                    weights=np.maximum(np.abs(coeff),.00015)
                    weights/=weights.mean()
    out=[]
    for opts in options:
        opts.sort(key=lambda p:p[0]); best=opts[0][0]
        selected=[(e,p) for e,p in opts[:18] if e<best*1.18+1e-8]
        if not selected: selected=opts[:1]
        e,p=zip(*selected)
        w=np.exp(-(np.array(e)-best)/max(.037*best,1e-8))
        out.append(np.average(p,axis=0,weights=w))
    return np.column_stack((out[0],np.column_stack(out[1:])@q.T))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    sparse = inputs['node_00']
    p = _readout(study)
    return {'panel_pred': np.column_stack((2*p[:,0]-sparse['y_hat'],(p[:,1:]-.35*sparse['state_pred'])/.65))}
