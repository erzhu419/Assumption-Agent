"""Calibrated multi-output regression for the blind campaign."""
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist


def _alignment(panel, x):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    i, j = np.triu_indices(len(u), 1)
    z = np.empty_like(x, dtype=float)
    for k in range(u.shape[1]):
        dx = u[i,k]-u[j,k]
        good = abs(dx)>1e-9
        slope = np.median((v[i[good],k]-v[j[good],k])/dx[good])
        r = v[:,k]-slope*u[:,k]
        dev = abs(r-np.median(r))
        good = dev <= max(.001, 6*np.median(dev))
        fit = np.linalg.lstsq(np.c_[np.ones(sum(good)),u[good,k]],v[good,k],rcond=None)[0]
        z[:,k] = (x[:,k]-fit[0])/fit[1]
    return z


def _predict(study):
    tr = study['train']
    Y = np.c_[tr['y'],tr['traj']]
    n, d = Y.shape
    Z = _alignment(study['panel_a'],np.r_[tr['x'], study['test']['x']]); Z -= Z[:n].mean(0)
    B = study['panel_b']; bu,bv = B['u'], B['v']
    b = np.linalg.lstsq(np.c_[np.ones(len(bu)),bu],bv,rcond=None)[0][1:]
    ids = np.eye(n)
    groups=[Z, Z@b, Z@b.T, np.tanh(1.25*Z), Z/(1+.22*Z**2)]
    models=[]
    scale = np.maximum(Y.std(0), 1e-8)
    for i,W in enumerate(groups):
        W=W/np.sqrt(np.mean(np.sum(W[:n]**2,axis=1)))
        G=W@W.T; R=cdist(W,W,'sqeuclidean'); diag=[]
        if i<3:
            diag=[sum(np.exp(-a*cdist(W[:,j:j+1],W[:,j:j+1],'sqeuclidean')) for j in range(W.shape[1]))/W.shape[1] for a in (1.,3.)]
        kernels=[1+np.exp(-g*R) for g in (.08,.16,.28,.42,.65,1.)]
        kernels += [(1+g*G)**k for k in (2,3,4,5,7) for g in (.12,.45,1.)]
        kernels += [1+np.exp(-g*R)+w*A for A in diag for g in (.22,.42,.65) for w in (.5,2.,8.)]
        for K in kernels:
            for r in (1e-5,.0002,.001,.005,.02):
                a=solve(K[:n,:n]+r*ids,np.c_[Y,ids],assume_a='pos'); H=a[:,:d]; loo=abs(H/np.diag(a[:,d:])[:,None])/scale
                models.append((np.array([loo[:,0].mean(),loo[:,1:].mean()]),K[n:,:n]@H))
    out=np.empty((len(Z)-n,d))
    for j,cols in enumerate((slice(0,1),slice(1,None))):
        best=sorted(models,key=lambda v:v[0][j])[:16]; es=np.array([e[j] for e,_ in best]); w=np.exp(-(es/es[0]-1)*32); out[:,cols]=np.average([p[:,cols] for _,p in best],axis=0,weights=w)
    return out


def solve_batch(study: dict) -> dict:
    out=_predict(study); y=out[:,0]; state=out[:,1:]; last=state[:,-1]
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),last.mean(),last.std()])}

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _panel_kernels(study,z,n):
    def d(a): return np.c_[np.ones(len(a)),a]
    def fit(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
    def adj(a): return a*np.roll(a,-1,axis=1)
    b,c=study['panel_b'],study['panel_c']
    bm=fit(d(b['u']),b['v']); pm=fit(d(adj(c['u'])),c['v1']); qm=fit(d(np.cos(2.4*c['u'])),c['v2'])
    p=d(adj(z))@pm; q=d(np.cos(2.4*z))@qm
    tb,tc,ta,td=study['telemetry_vec']; wz=d(z)@bm
    def dot(a):
        a=a-a[:n].mean(0); a=a/np.maximum(a[:n].std(0),1e-8); return a@a.T/a.shape[1]
    def sq(a): return cdist(a,a,'sqeuclidean')/a.shape[1]
    z0=(z-z[:n].mean(0))/np.maximum(z[:n].std(0),1e-8)
    zz=sq(z0); kernels=[np.ones_like(zz),dot(z),dot(adj(z)),dot(np.cos(2.4*z))]
    kernels.extend(np.exp(-g*zz) for g in (.08,.25,.55,1.2))
    kernels.extend(zz0**k for zz0 in [dot(z)] for k in (2,3,4))
    kernels.extend(dot(a) for a in (p*q,(p*q).sum(1)[:,None],np.logaddexp(0,tb*p),np.tanh(tc*q),p/(1+ta*q*q),q/(1+td*p*p),np.logaddexp(0,tb*(p+q)),np.logaddexp(0,tb*(p-q)),np.sin(tc*p),np.sin(tc*q)))
    kernels.extend(dot(a) for a in (np.column_stack([p[:,i]*q[:,j] for i in range(4) for j in range(4)]),np.column_stack([z[:,i]*p[:,j] for i in range(4) for j in range(4)]),np.column_stack([z[:,i]*q[:,j] for i in range(4) for j in range(4)]),adj(wz),np.cos(2.4*wz)))
    for a in (z,wz,p,q,p-q):
        a=(a-a[:n].mean(0))/np.maximum(a[:n].std(0),1e-8)
        for g in (.3,1.3):
            kernels.append(sum(np.exp(-g*cdist(a[:,j:j+1],a[:,j:j+1],'sqeuclidean')) for j in range(a.shape[1]))/a.shape[1]); kernels.append(np.exp(-g*sq(a)))
    for a in (z+.22*adj(z), np.tanh(tb*z)+.22*adj(z),z/(1+ta*z*z),np.sin(tb*z)):
        pp=d(adj(a))@pm; qq=d(np.cos(2.4*a))@qm
        kernels.extend([dot(pp),dot(qq),dot(pp*qq)])
    return np.array(kernels)


def _multi_kernel(K,target):
    n=len(target); kt=np.concatenate([K[:,:n,:n],np.eye(n)[None]],axis=0).copy(); flat=kt.reshape(len(kt),-1)
    ks=K[:,n:,:n].copy(); fs=ks.reshape(len(K),-1); identity=np.eye(n)
    target=np.asarray(target); y=target/np.maximum(target.std(0),1e-8); d=y.shape[1]
    guesses=np.full(len(kt),-4.); guesses[0]=0.; guesses[-1]=-5.
    def calc(a):
        w=np.exp(a); F=cho_factor((w@flat).reshape(n,n),check_finite=False); al=cho_solve(F,y,check_finite=False)
        inv=cho_solve(F,identity,check_finite=False)
        return w,F,al,inv
    def obj(a,lam):
        w,F,al,inv=calc(a); dev=a-guesses
        val=.5*np.sum(y*al)+d*np.log(np.diag(F[0])).sum()+.5*lam*np.sum(dev**2)
        grad=.5*w*(flat@(d*inv-al@al.T).ravel())+lam*dev
        return val,grad
    out=[]
    for lam in (.015,.08,.4):
        r=minimize(obj,guesses,args=(lam,),jac=True,method='L-BFGS-B',bounds=[(-14,4)]*len(kt),options={'maxiter':110,'ftol':1e-9}); w,F,al,inv=calc(r.x)
        err=np.abs(al/np.diag(inv)[:,None]).mean(); pred=(w[:-1]@fs).reshape(ks.shape[1:])@al*np.maximum(target.std(0),1e-8); out.append((err,pred))
    return out


def _challenge(study):
    tr=study['train']; n=len(tr['y']); t=np.asarray(study['eval_times']); z=_alignment(study['panel_a'],np.r_[tr['x'],study['test']['x']]); K=_panel_kernels(study,z,n)
    out=[v for _,v in _multi_kernel(K,tr['y'][:,None])]
    T=tr['traj']; _,_,v=np.linalg.svd(T,full_matrices=False)
    # Preserve separate uncertainty scales for the dominant temporal modes.
    pc=[_multi_kernel(K,T@v[i:i+1].T) for i in range(min(3,len(t)))]; states=[np.column_stack([a[j][1][:,0] for a in pc])@v[:len(pc)] for j in range(3)]
    states.extend(p@v[:2] for _,p in _multi_kernel(K,T@v[:2].T))
    return out,states


def solve_batch(study: dict) -> dict:
    base=_predict(study)
    ys,ts=_challenge(study)
    base[:,:1] = 0.0 * base[:,:1] + 1.0 * ys[0]
    base[:,1:] = 0.0 * base[:,1:] + 1.0 * ts[0]
    y=base[:,0]; state=base[:,1:]; last=state[:,-1]
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),last.mean(),last.std()])}
