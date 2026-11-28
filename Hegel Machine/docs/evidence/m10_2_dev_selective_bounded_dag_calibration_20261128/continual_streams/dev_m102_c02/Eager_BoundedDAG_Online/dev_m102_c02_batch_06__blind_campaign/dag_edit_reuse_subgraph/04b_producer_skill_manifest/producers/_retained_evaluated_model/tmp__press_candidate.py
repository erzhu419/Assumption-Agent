"""Calibrated multi-response readout with training-only PRESS selection."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf
from scipy.spatial.distance import cdist


def _aff(z):
    return np.column_stack((np.ones(len(z)), z))


def _poly(z, degree):
    return np.column_stack([np.prod(z[:, j], axis=1) for d in range(degree + 1)
                            for j in combinations_with_replacement(range(z.shape[1]), d)])


def _latent(panel, raw):
    u, v = (np.asarray(panel[k], float) for k in ('u', 'v'))
    q = np.roll(u, -1, axis=1)
    M = np.stack((np.ones_like(u), u, q, u*q, u**3), -1).transpose(1, 0, 2)
    V = v.T[..., None]
    rng = np.random.default_rng(485)
    best = np.inf
    for _ in range(96):
        i = rng.choice(len(u), min(8, len(u)), replace=False)
        b = np.linalg.pinv(M[:, i]) @ V[:, i]
        err = np.median(np.linalg.norm((M @ b - V)[..., 0], axis=0))
        if err < best:
            best, beta = err, b
    for _ in range(4):
        e = np.linalg.norm((M @ beta - V)[..., 0], axis=0)
        keep = e <= max(6*np.median(e), 1e-8)
        beta = np.linalg.pinv(M[:, keep]) @ V[:, keep]
    o, g, h, m, a = beta[..., 0].T
    k = np.arange(raw.shape[1]); l = (k+1) % len(k)
    J = np.zeros((len(k), len(k))); J[k,k] = g; J[k,l] = h
    w = np.linalg.lstsq(J, (raw-o).T, rcond=None)[0].T
    for _ in range(25):
        z = w[:,l]; r = o+g*w+h*z+m*w*z+a*w**3-raw
        if np.max(np.abs(r), initial=0) < 1e-10: break
        A = np.zeros((len(w),len(k),len(k))); A[:,k,k] = g+m*z+3*a*w*w; A[:,k,l] = h+m*w
        d = (np.linalg.pinv(A) @ r[..., None])[...,0]
        step = np.ones((len(w),1)); old = np.max(abs(r), axis=1)
        for _ in range(16):
            p=w-step*d; z=p[:,l]
            bad=np.max(abs(o+g*p+h*z+m*p*z+a*p**3-raw),axis=1)>old
            if not np.any(bad): break
            step[bad]*=.5
        w=np.where(bad[:,None],w,p)
    return w


def _reference(x, v, kinds):
    out = []
    for f in kinds:
        if f is None:
            feature = _aff
        elif f == 'square':
            feature = lambda z: np.column_stack((_aff(z), z*z))
        else:
            def loss(k):
                A = np.column_stack((_aff(x), f(x,k)))
                return np.mean((A@np.linalg.lstsq(A,v,rcond=None)[0]-v)**2)
            k = minimize_scalar(loss,bounds=(.06,4), method='bounded').x
            feature = lambda z, f=f, k=k: np.column_stack((_aff(z), f(z,k)))
        A=feature(x); beta=np.linalg.lstsq(A,v,rcond=None)[0]
        out.append((np.mean((A@beta-v)**2),feature,beta))
    return sorted(out,key=lambda v:v[0])


def _predict(study, stage, target, debug=False):
    n=len(target); c=np.concatenate((stage['c'],stage['query']))
    raw=np.concatenate((study['train']['x'],study['test']['x']))
    u=_latent(study['panel_a'],raw)
    pc=study['panel_c']; pb=study['panel_b']
    sig=[lambda z,k:z*np.exp(-k*z*z),lambda z,k:np.tanh(k*z),lambda z,k:erf(k*z),lambda z,k:z/(1+k*z*z),lambda z,k:np.sin(k*z)]
    bmaps=_reference(pb['u'],pb['v'],sig)
    maps=[_reference(pc['u'],pc[k],[None,'square',*sig])[:2] for k in ('v1','v2')]
    coords={'c':c,'u':u,'raw':raw}
    for i,(_,f,b) in enumerate(bmaps[:2]):
        coords['B'+str(i)]=f(u)@b
    _,f,b=bmaps[0]
    coords['rot']=(c-b[0])@np.linalg.pinv(b[1:5])
    for j,mm in enumerate(maps):
        for i,(_,f,b) in enumerate(mm):
            for name,z in [('u',u),('c',c),('rot',coords['rot'])]:
                tag=f'C{j}{i}_{name}'
                coords[tag]=f(z)@b
    records=[]
    scale=np.maximum(np.mean(abs(target-target.mean(0)),axis=0),.02)
    weights=np.r_[.55,np.full(target.shape[1]-1,.45/(target.shape[1]-1))]/scale
    def record(E, pred, name):
        if np.all(np.isfinite(pred)):
            records.append((np.mean(abs(E),axis=0),pred,name))
    def lin(F,name,ridges=(0,)):
        mu=F[:n].mean(0); std=np.maximum(F[:n].std(0),1e-10)
        F=(F-mu)/std
        U,S,V=np.linalg.svd(F[:n],full_matrices=False)
        keep=S>max(S[0],1)*1e-8; U=U[:,keep]; S=S[keep]; V=V[keep]
        T=target-target.mean(0); UT=U.T@T
        for a in ridges:
            fac=S*S/(S*S+a)
            E=(T-U@(fac[:,None]*UT))/np.maximum(1-1/n-(U*U)@fac,1e-7)[:,None]
            pred=F@(V.T@((fac/S)[:,None]*UT))+target.mean(0)
            record(E,pred,(name,a))
    def kern(K,name,alphas):
        S,U=np.linalg.eigh((K[:n]+K[:n].T)*.5); S=np.maximum(S,0); UT=U.T@target
        for a in alphas:
            fac=1/(S+a); A=U@(fac[:,None]*UT)
            record(A/((U*U)@fac)[:,None],K@A,(name,a))
    for name,z in coords.items():
        var=np.maximum(z[:n].std(0),.02)
        for d in [1,2,3,4]:
            lin(_poly(z,d), (name,'poly',d), (0,.03,.3,2,10) if d>1 else (0,.3))
        zs=(z-z[:n].mean(0))/var
        for std in (False,True):
            zz=zs if std else (z-z[:n].mean(0))/np.sqrt(np.mean(var**2))
            D=cdist(zz,zz[:n],'sqeuclidean')
            if name in ('u','raw') or name.startswith('C'):
                powers=(.6,1.4,3.0)
            else:
                powers=(.35,.6,1.0,1.7,3.0,5.0)
            A=_aff(zz); L=A@A[:n].T
            for h in powers:
                r=np.sqrt(3*D)/h
                for kn,K in [('rbf',np.exp(-D/(2*h*h))),('matern',(1+r)*np.exp(-r)),('rq',1/(1+D/(2*h*h)))]:
                    kern(K+L,(name,std,h,kn),(.0001,.003,.03,.2))
        if name not in ('c','rot','B0','B1','u') and not name.startswith('C'): continue
        for h in (.5,1,1.5,2,3,5):
            for fn,g in [('r',z/(1+h*z*z)),('e',z*np.exp(-h*z*z)),('t',np.tanh(h*z)),('s',np.sin(h*z))]:
                for typ,F in enumerate([_poly(g,2),_poly(z+g,2),np.column_stack((_poly(z,2),g)),np.column_stack((_aff(z),_poly(g,2)[:,1:]))]):
                    lin(F,(name,fn,h,typ),(0,.1,1))
    # Pairwise signed sums, radial contractions, and sparse products in the calibrated coordinates.
    for name in ('c','rot','B0','u','C00_c','C10_c','C00_u','C10_u'):
        z=coords[name]; D=_aff(z); Q=_poly(z,2); H=target
        v=np.column_stack([z,*[z[:,i]+k*z[:,j] for i in range(4) for j in range(i+1,4) for k in (-2,-1,-.5,.5,1,2)]])
        P=np.column_stack([np.sum(z[:,inds]**2,axis=1) for inds in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3),(0,1,2,3)]])
        G=np.column_stack([_poly(z,4)[:,Q.shape[1]:],*[g for h in (.5,1,2,4,8) for g in (np.sin(h*v), np.cos(h*v),v/(1+h*v*v),1/(1+h*v*v),np.exp(-h*v*v),np.tanh(h*v),np.exp(-h*P),1/(1+h*P))]])
        for base in [D,Q]:
            F=base; J=G.copy()
            for j in range(15):
                U,S,_=np.linalg.svd(F[:n],full_matrices=False); U=U[:,S>S[0]*1e-8]
                E=H-U@(U.T@H); X=J[:n]-U@(U.T@J[:n]); norm=np.linalg.norm(X,axis=0)
                cor=np.linalg.norm((X.T@E)*weights,axis=1)/np.maximum(norm,1e-12)
                cor[norm<1e-7]=-np.inf
                top=np.argsort(cor)[-8:]
                best=None
                for ix in top:
                    F1=np.column_stack((F,J[:,ix])); lin(F1,(name,'sparse',base.shape[1],j)); rec=records[-1]
                    s=rec[0]@weights
                    if best is None or s<best[0]: best=(s,ix,F1)
                _,ix,F=best; J[:,ix]=0
    if debug:
        for key in ('joint','scalar','trajectory'):
            fun={'joint':lambda a:a[0]@weights,'scalar':lambda a:a[0][0],'trajectory':lambda a:a[0][1:].mean()}[key]
            print(key,[(r[2],np.round(r[0][[0,-1]],4).tolist()) for r in sorted(records,key=fun)[:10]],flush=True)
    def combine(a):
        indices=np.argsort(a); a=np.asarray(a); best=a[indices[0]]
        ix=indices[:12]; w=np.exp(-25*(a[ix]/max(best,1e-10)-1)); w/=w.sum()
        return np.einsum('i,ijk->jk',w,np.array([records[i][1] for i in ix]))
    y=combine([r[0][0] for r in records])[:,0]
    t=combine([np.mean(r[0][1:]/scale[1:]) for r in records])[:,1:]
    return y,t


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; stage=inputs['node_00']; train=study['train']
    y=np.asarray(train['y'],float); traj=np.asarray(train['traj'],float)
    y_pred, t_pred=_predict(study,stage,np.column_stack((y,traj)))
    n=len(y)
    # Represent the learned trajectory in the training temporal basis for node_02.
    _,_,modes=np.linalg.svd(traj,full_matrices=False)
    basis=modes[:2].T
    return {'y_hat':y_pred[n:], 'beta':basis.T, 'fitted':(t_pred[:n]@basis)@basis.T, 'new_design':t_pred[n:]@basis}
