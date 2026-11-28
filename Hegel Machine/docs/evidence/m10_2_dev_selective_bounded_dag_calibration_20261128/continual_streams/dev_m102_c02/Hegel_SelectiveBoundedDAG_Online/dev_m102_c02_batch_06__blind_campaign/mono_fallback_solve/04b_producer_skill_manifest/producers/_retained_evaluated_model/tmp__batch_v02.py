"""Calibrated nonlinear predictor for the batch readout and state trajectory."""
import numpy as np
from scipy.linalg import solve

def _calibrate(panel, x):
    u,v=(np.asarray(panel[k],float) for k in ('u','v'))
    def features(a,b): return np.column_stack((np.ones(len(a)),a,b,a*b,a**3))
    g=np.ones(len(u),bool)
    for _ in range(4):
        m=np.array([np.linalg.lstsq(features(u[g,j],u[g,(j+1)%4]),v[g,j],rcond=None)[0] for j in range(4)])
        err=np.linalg.norm(np.column_stack([features(u[:,j],u[:,(j+1)%4])@m[j] for j in range(4)])-v,axis=1)
        g=err<=np.partition(err,44)[44]
    z=np.array(x,float,copy=True); j=np.arange(4)
    for _ in range(14):
        w=np.roll(z,-1,axis=1); J=np.zeros((len(z),4,4)); J[:,j,j]=m[:,1]+m[:,3]*w+3*m[:,4]*z*z; J[:,j,(j+1)%4]=m[:,2]+m[:,3]*z
        f=m[:,0]+m[:,1]*z+m[:,2]*w+m[:,3]*z*w+m[:,4]*z**3
        z-=np.linalg.solve(J,(f-x)[...,None])[...,0]
    return z

from scipy.optimize import least_squares
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erf

def _panel_features(panel):
    u,v=(np.asarray(panel[k],float) for k in ('u','v'))
    forms=(lambda a,q:np.tanh(q*a)**3,
           lambda a,q:a*np.tanh(q*a)**2,
           lambda a,q:(a/np.sqrt(1+q*a*a))**3,
           lambda a,q:a**3/(1+q*a*a),
           lambda a,q:a**3/(1+q*abs(a)),
           lambda a,q:np.sin(q*a)**3,
           lambda a,q:erf(q*a)**3,
           lambda a,q:a**3/(1+q*a**4),
           lambda a,q:np.sign(a)*abs(a)**q)
    candidates=[]
    for f in forms:
        for linear in (False,True):
            def design(a,q):
                g=f(a,q)
                return np.c_[np.ones(len(a)),a,g] if linear else np.c_[np.ones(len(a)),g]
            def residual(p):
                H=design(u,p[0]); return (H@np.linalg.lstsq(H,v,rcond=1e-9)[0]-v).ravel()
            for start in (.45,1.3,2.6):
                q=least_squares(residual,[start],bounds=(.03,4.),max_nfev=55).x[0]
                H=design(u,q); pin=np.linalg.pinv(H,rcond=1e-9); B=pin@v
                loo=((H@B-v)/np.maximum(.045,1-np.sum(H*pin.T,axis=1))[:,None]); loss=np.mean(loo**2)
                candidates.append((loss,f,q,B,linear))
    loss,f,q,B,linear=min(candidates,key=lambda c:c[0]); return f,q,B,linear

def _families(x,panel):
    n=len(x); one=np.ones((n,1)); i,j=np.triu_indices(4); k,l=np.triu_indices(4,1)
    f,q,B,lin=_panel_features(panel)
    g=f(x,q); h=(np.c_[one,x,g] if lin else np.c_[one,g])@B
    specs=[(.85,lambda a:a/np.sqrt(1+a*a)),(1.,lambda a:a/np.sqrt(1+a*a)),(1.25,lambda a:a/np.sqrt(1+a*a)),
           (1.3,np.arctan),(1.5,np.arctan),(1.7,np.arctan),(1.25,np.tanh),(1.5,np.tanh),(1.65,np.tanh),(1.85,np.tanh),
           (1.25,erf),(1.5,erf),(1.6,lambda a:np.clip(a,-1,1.)),(1.,lambda a:a/(1+abs(a)))]
    def blocks(z):
        return z[:,i]*z[:,j], (x[:,:,None]*z[:,None,:]).reshape(n,16)
    for index,(q,f) in enumerate(specs):
        z=f(q*x); zz,xz=blocks(z); m=np.c_[one,x,z]; r=np.c_[m,zz,xz]; delta=x-z/(q if index not in (12,13) else max(q,1));
        yield 0,index,r
        yield 1,index,np.c_[m,zz,.35*xz,x*z]
        yield 2,index,np.c_[m,zz,.7*xz,.45*z**3,.35*delta**2,.25*(z[:,k]*z[:,l])*(x[:,k]+x[:,l])]
        yield 3,index,np.c_[m,zz,x*z,.4*f(q*(x[:,k]+x[:,l])),.4*f(q*(x[:,k]-x[:,l]))]
        yield 4,index,np.c_[r, .6*g, .45*(z[:,:,None]*g[:,None,:]).reshape(n,16)]
        yield 5,index,np.c_[r,.65*h,.5*h[:,i]*h[:,j],.4*np.tanh(h),.3*(z[:,:,None]*h[:,None,:]).reshape(n,16)]
        for cut in (1.0,1.25,1.5):
            c=np.clip(x,-cut,cut); w=f(q*c); ww,wx=blocks(w)
            yield 6,index,np.c_[one,c,w,ww,(c[:,:,None]*w[:,None,:]).reshape(n,16)]
    z=np.tanh(1.55*x); zz,xz=blocks(z)
    for power in (1,2,3):
        z=np.sign(g)*abs(g)**(1/power); zz,xz=blocks(z)
        yield 7,power,np.c_[one,x,z,zz,xz]

def _portfolio(study,x):
    train=study['train']; y=np.asarray(train['y'],float); t=np.asarray(train['traj'],float); n=len(y)
    targets=np.c_[y,t]
    groups={}
    for kind,idx,F in _families(x,study['panel_b']):
        K=F@F[:n].T; G=K[:n]; diagonal=np.eye(n)
        for reg in (.0003,.0015,.004,.012,.04):
            C=cho_factor(G+reg*diagonal); a=cho_solve(C,targets); lev=np.diag(cho_solve(C,diagonal)); E=a/lev[:,None]
            oof=targets-E; pred=K[n:]@a
            # Pool multiple nearby feature maps rather than selecting a single unstable extrapolant.
            loss=np.mean(abs(E[:,0])); tail=np.mean(abs(E[:,0])[np.argsort(np.linalg.norm(x[:n],axis=1))[-max(n//4,1):]])
            score=.82*loss+.18*tail
            groups.setdefault(kind,[]).append((score,loss,np.mean(abs(E[:,1:])),pred,oof))
    out={}; loo={}
    for kind,items in groups.items():
        items.sort(key=lambda a:a[0]); subset=items[:min(8,len(items))]; w=np.exp(-(np.array([a[0] for a in subset])-subset[0][0])/.009);w/=w.sum()
        out[kind]=np.einsum('i,ijk->jk',w,np.array([a[3] for a in subset])); loo[kind]=np.einsum('i,ijk->jk',w,np.array([a[4] for a in subset]));
    # Calibrate a low-rank correction between independently estimated output channels.
    state=[]
    for idx,(q,f) in enumerate([(1.03,lambda u:u/np.sqrt(1+u*u)),(1.15,lambda u:u/np.sqrt(1+u*u)),(1.25,np.arctan),(1.32,np.tanh)]):
        z=f(q*x); i,j=np.triu_indices(4); F=np.c_[np.ones(len(x)),x,z,z[:,i]*z[:,j],(x[:,:,None]*z[:,None,:]).reshape(len(x),16)]; K=F@F[:n].T
        for reg in (.0003,.001,.003):
            C=cho_factor(K[:n]+reg*np.eye(n)); a=cho_solve(C,t); E=a/np.diag(cho_solve(C,np.eye(n)))[:,None]; state.append((np.mean(abs(E)),K[n:]@a,t-E))
    state.sort(key=lambda a:a[0]); st=np.mean([a[1] for a in state[:7]],axis=0); os=np.mean([a[2] for a in state[:7]],axis=0)
    return out,loo,st,os

def _readout(study,x):
    from scipy.optimize import nnls
    out,loo,st,os=_portfolio(study,x); tr=study['train']; y=np.asarray(tr['y']); n=len(y)
    ks=sorted(out); oof=np.column_stack([loo[k][:,0] for k in ks]); pred=np.column_stack([out[k][:,0] for k in ks]); costs=np.mean(abs(oof-y[:,None]),axis=0)
    w=np.exp(-(costs-costs.min())/.012);w/=w.sum(); p=pred@w; op=oof@w
    # Predict nonlinear readout using a temporal amplitude as an additional proxy.
    t=tr['traj']; dt=t.mean(0); _,_,V=np.linalg.svd(t-dt,full_matrices=False)
    U=(t-dt)@V[:3].T; Up=(os-dt)@V[:3].T; T=(st-dt)@V[:3].T; scale=U.std(0).clip(.035)
    # A strongly regularized mixture avoids relying on one family's response to tail inputs.
    P=np.c_[oof, op, Up[:,0], np.ones(n)]; Q=np.c_[pred,p,T[:,0],np.ones(len(x)-n)]
    anchor=np.r_[w,0.,0.,0.]; pen=np.diag(np.r_[np.full(len(w)+1,.24),.45,.24]); coef=np.linalg.solve(P.T@P+pen,P.T@y+pen@anchor)
    return .7*p+.3*(Q@coef),st

def solve_batch(study: dict) -> dict:
    tr=study['train']; x=_calibrate(study['panel_a'],np.r_[tr['x'],study['test']['x']]); y,p=_readout(study,x)
    return {'y_hat':y,'state_pred':p,'diagnostics_vec':np.array([y.mean(),y.std(),p[:,-1].mean(),p[:,-1].std()])}
