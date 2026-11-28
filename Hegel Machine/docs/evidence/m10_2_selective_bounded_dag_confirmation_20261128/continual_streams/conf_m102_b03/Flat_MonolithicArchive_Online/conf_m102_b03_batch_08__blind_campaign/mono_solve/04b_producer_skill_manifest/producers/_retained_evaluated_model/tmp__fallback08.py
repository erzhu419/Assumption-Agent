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
