"""Profile calibration models from the reference panel, then fit joint readouts."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import minimize_scalar, least_squares
from scipy.linalg import solve


def _poly(x, degree):
    return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for d in range(1,degree+1) for i in combinations_with_replacement(range(x.shape[1]),d)])


def _cal_design(x, kind, k, j):
    a=_poly(x,1)
    if kind=='cyclic': return np.c_[a,x[:,j]**3,x[:,j]*x[:,(j+1)%x.shape[1]]]
    if kind=='cube': return np.c_[a,x**3]
    if kind=='quadratic': return _poly(x,2)
    if kind=='cubic': return _poly(x,3)
    if kind=='sin': return np.c_[a,np.sin(k*x)]
    if kind=='cos': return np.c_[a,np.cos(k*x)]
    if kind=='both': return np.c_[a,np.sin(k*x),np.cos(k*x)]
    if kind=='diag': return np.c_[a,np.sin(k*x[:,j]),np.cos(k*x[:,j]),x[:,j]*x[:,(j+1)%x.shape[1]]]
    if kind=='atan': return np.c_[a,np.arctan(k*x)]
    return np.c_[a,np.tanh(k*x)]


def _calibrate(u,v,x,z):
    d=u.shape[1]; n=len(u); ii=np.arange(n)
    def fit(kind,k,t):
        mats=[_cal_design(u,kind,k,j) for j in range(d)]
        keep=t[np.argsort(np.linalg.norm(v[t]-u[t],axis=1))[:max(1,int(.82*len(t)))]]
        for step in range(5):
            w=[np.linalg.lstsq(m[keep],v[keep,j],rcond=1e-9)[0] for j,m in enumerate(mats)]
            r=np.linalg.norm(np.column_stack([m@c for m,c in zip(mats,w)])-v,axis=1)
            nxt=t[np.argsort(r[t])[:len(keep)]]
            if np.array_equal(np.sort(nxt),np.sort(keep)): break
            keep=nxt
        return np.median(r[t]),w,r
    folds=np.array_split(np.random.default_rng(251).permutation(n),4)
    candidates=[]
    for kind in ('cyclic','cube','quadratic','cubic','sin','cos','both','diag','atan','tanh'):
        if kind in ('cyclic','cube','quadratic','cubic'): k=1.
        else:
            gs=np.linspace(.5,5.,37); m=np.argmin([fit(kind,g,ii)[0] for g in gs]); k=minimize_scalar(lambda g:fit(kind,g,ii)[0],bounds=(gs[max(0,m-1)],gs[min(m+1,len(gs)-1)]),method='bounded').x
        err=np.mean([np.median(fit(kind,k,np.setdiff1d(ii,h))[2][h]) for h in folds]); e,w,_=fit(kind,k,ii)
        candidates.append((err,e,kind,k,w))
    out=[('frozen',z)]
    for err,e,kind,k,w in sorted(candidates,key=lambda a:a[0])[:2]:
        def calc(p): return np.array([_cal_design(p[None,:],kind,k,j)[0]@w[j] for j in range(d)])
        zz=[]
        for a,b in zip(z,x):
            fit1=least_squares(lambda p:calc(p)-b,a,max_nfev=35)
            zz.append(fit1.x)
        out.append((kind,np.array(zz)))
    return out


def _frequency(u,v,fun,lo=.25,hi=8.):
    def design(k):return np.c_[np.ones(len(u)),fun(k*u)]
    def err(k):
        a=design(k); return np.mean((a@np.linalg.lstsq(a,v,rcond=None)[0]-v)**2)
    g=np.linspace(lo,hi,125); j=np.argmin([err(k) for k in g]); f=minimize_scalar(err,bounds=(g[max(0,j-1)],g[min(j+1,len(g)-1)]),method='bounded').x
    return f


def _fit(a, t, j, y, p):
    m=a[t].mean(0); sd=np.maximum(a[t].std(0),1e-8); at=(a[t]-m)/sd; bt=(a[j]-m)/sd; mu=y[t].mean(0)
    return mu+bt@solve(at.T@at+p*len(t)*np.eye(at.shape[1]),at.T@(y[t]-mu),assume_a='pos')


def run_node(inputs: dict) -> dict:
    s,z0=inputs['__input__'],inputs['node_00']; tr=s['train']; x=np.vstack((tr['x'],s['test']['x'])); z=np.cbrt(np.vstack((z0['cubic_train'],z0['cubic_test'])))
    b,c,a=s['panel_b'],s['panel_c'],s['panel_a']; f=_frequency(b['u'],b['v'],np.cos); g=_frequency(c['u'],c['v1'],np.arctan)
    y=np.c_[tr['y'],tr['traj']]; n=len(y); ii=np.arange(n); test=np.arange(n,len(x)); folds=np.array_split(np.random.default_rng(232).permutation(n),5); cv=[(np.setdiff1d(ii,h),h) for h in folds]; results=[]
    for name,u in _calibrate(a['u'],a['v'],x,z)+[('raw',x)]:
        cs,ss=np.cos(f*u),np.sin(f*u); at=np.arctan(g*u); cx=np.cos(f*x); sq=_poly(ss,2)[:,1+u.shape[1]:]; cq=_poly(cs,2)[:,1+u.shape[1]:]; j,k=np.triu_indices(u.shape[1],1)
        bases=[cs,np.c_[cs,ss],np.c_[cs,at],np.c_[cs,cx,at],np.c_[cs,at,sq],np.c_[cs,cx,at,sq],np.c_[cs,at,cq],np.c_[cs,at,u,u**3],np.c_[cs,at,np.cos(f*(u[:,j]+u[:,k])),np.cos(f*(u[:,j]-u[:,k]))],np.c_[cs,at,np.cos(f*u**3)],np.c_[cs,at,np.cos(g*u)],np.c_[cs,at,np.sin(g*u)]]
        for bidx,ab in enumerate(bases):
            for penalty in (.0001,.003,.025,.12):
                e=np.array([abs(_fit(ab,t,h,y,penalty)-y[h]).mean(0) for t,h in cv]); v=np.c_[e[:,0],e[:,1:].mean(1)]; metric=.8*v.mean(0)+.2*v.max(0)
                results.append((metric,(name,bidx),ab,penalty))
    out=[]
    for k in range(2):
        top={}
        for e,m,ab,p in sorted(results,key=lambda r:r[0][k]):
            if m not in top: top[m]=(e[k],_fit(ab,ii,test,y,p))
            if len(top)==4:break
        err,pred=zip(*top.values()); err=np.array(err); w=np.exp(-22*(err-err.min())/max(.01,err.min())); out.append(np.einsum('i,ijk->jk',w/w.sum(),np.array(pred)))
    yh=out[0][:,0]; st=out[1][:,1:]
    return {'y_hat':yh,'state_pred':st,'diagnostics_vec':np.array([yh.mean(),yh.std(),st[:,-1].mean(),st[:,-1].std()])}
