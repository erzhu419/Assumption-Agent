"""Calibrated multi-response readout with blocked training-fold selection."""
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
    # Select readouts on repeated blocked folds, not on in-sample PRESS after
    # response-dependent feature selection. Penalize extrapolation explicitly.
    rng=np.random.default_rng(3502)
    folds=[]
    for order in [np.arange(n),rng.permutation(n),np.argsort(c[:n,0]),rng.permutation(n)]:
        folds.extend(np.array_split(order,5))
    scale=np.maximum(np.mean(abs(target-target.mean(0)),axis=0),.02)
    records=[]
    def assess(K, penalties):
        K=(K[:n,:]+K[:n,:].T)*.5 if K.shape[0]==n else K
        S,U=np.linalg.eigh((K[:n]+K[:n].T)*.5)
        S=np.maximum(S,0); UY=U.T@target
        for a in penalties:
            inv=1/(S+a)
            dual=U@(inv[:,None]*UY)
            fold_err=[]
            for ids in folds:
                B=U[ids]; P=(B*inv)@B.T
                e=np.linalg.solve(P,dual[ids]); fold_err.append(np.mean(abs(e),axis=0))
            es=np.asarray(fold_err)/scale
            # Averaged error plus the upper-tail penalty on blocked folds.
            rank=.8*es.mean(0)+.2*np.quantile(es,.85,axis=0)
            records.append((rank,K@dual))
    def linear(F):
        mu=F[:n].mean(0); sigma=np.maximum(F[:n].std(0),.015)
        F=np.column_stack((np.ones(len(F)),(F-mu)/sigma)); return F@F[:n].T
    pairs={key:np.column_stack((c,z)) for key,z in coords.items() if key.startswith('C')}
    pairs['both']=np.column_stack((c,coords['C00_u'],coords['C10_u']))
    for name,z in {**coords,**{'pair_'+key:v for key,v in pairs.items()}}.items():
        for d in (1,2,3) if z.shape[1]<=4 else (1,2):
            assess(linear(_poly(z,d)),(.015,.15,1.5,8) if d>1 else (.005,.2,2))
        w=(z-z[:n].mean(0))/np.maximum(z[:n].std(0),.03)
        L=linear(z); D=cdist(w,w[:n],'sqeuclidean')/z.shape[1]
        if z.shape[1]<=8:
            for h in (.4,.75,1.2,2.):
                r=np.sqrt(3*D)/h
                for K in (np.exp(-D/(2*h*h)),(1+r)*np.exp(-r),1/(1+D/(2*h*h))):
                    assess(L+3*K,(.003,.03,.2))
        if name in ('c','B0','B1','rot'):
            for h in (.5,1.,2.,4.):
                for g in (z/(1+h*z*z),z*np.exp(-h*z*z),np.tanh(h*z)):
                    for F in (_poly(z+g,2),np.column_stack((_poly(z,2),g))):
                        assess(linear(F),(.02,.25,2.))
    def blend(scores, columns):
        order=np.argsort(scores)[:20]; ss=np.asarray(scores)[order]; w=np.exp(-16*(ss/ss[0]-1)); w/=w.sum()
        return sum(wi*records[i][1][:,columns] for wi,i in zip(w,order))
    return blend([r[0][0] for r in records],0),blend([r[0][1:].mean() for r in records],slice(1,None))


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; stage=inputs['node_00']; train=study['train']
    y=np.asarray(train['y'],float); traj=np.asarray(train['traj'],float)
    y_pred, t_pred=_predict(study,stage,np.column_stack((y,traj)))
    n=len(y)
    # Represent the learned trajectory in the training temporal basis for node_02.
    _,_,modes=np.linalg.svd(traj,full_matrices=False)
    basis=modes[:2].T
    return {'y_hat':y_pred[n:], 'beta':basis.T, 'fitted':(t_pred[:n]@basis)@basis.T, 'new_design':t_pred[n:]@basis}
