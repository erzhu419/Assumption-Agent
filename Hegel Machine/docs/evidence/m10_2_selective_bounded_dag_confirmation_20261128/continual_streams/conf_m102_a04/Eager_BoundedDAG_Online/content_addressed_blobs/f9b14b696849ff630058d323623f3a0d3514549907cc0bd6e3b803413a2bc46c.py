import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve

def _models(w,n):
    for u in (w, w[:,:4],w[:,4:]):
        uc = u-u[:n].mean(0)
        _, sv, vt = np.linalg.svd(uc[:n], full_matrices=False)
        for frac in ((1., .18, .003) if u.shape[1]==8 else (1.,.15)):
            v = uc @ vt.T / np.sqrt((sv*sv/n+frac)**.65)
            v /= np.sqrt(np.mean(np.sum(v[:n]**2,axis=1)))+1e-12
            g=v @ v[:n].T
            d=cdist(v,v[:n],'sqeuclidean')
            yield g, 'linear'
            yield g+.35*g*g, 'quadratic'
            for t in (.2,.8,2.):
                yield np.exp(-t*d)+.45*g, 'rbf+linear'
                s=np.sqrt(t*d)
                yield (1+s)*np.exp(-s)+.45*g, 'matern+linear'

def _predict(w, target, n, groups):
    center=target.mean(0)
    Y=target-center
    eye=np.eye(n)
    records=[]
    for K,name in _models(w,n):
        for noise in (.02,.09,.28,.75,1.65):
            C=cho_factor(K[:n]+noise*eye+3.)
            ki=cho_solve(C,eye)
            a=ki@Y
            res=np.abs(a/ki.diagonal()[:,None])
            scores=np.array([res[:,j].mean() for j in groups])
            pred=(K[n:]+3.)@a+center
            records.append((scores,pred,name,noise))
    scores=np.stack([z[0] for z in records])
    ans=np.empty((len(w)-n,target.shape[1]))
    for i,g in enumerate(groups):
        idx=np.argsort(scores[:,i])[:18]; s=scores[idx,i]; scale=max(.025*s[0],1e-12)
        weights=np.exp(-(s-s[0])/scale); weights/=weights.sum()
        ans[:,g]=np.tensordot(weights,np.stack([records[j][1][:,g] for j in idx]),axes=1)
    return ans

def run_node(inputs: dict) -> dict:
    feature_data=inputs['node_00']
    w=np.asarray(feature_data['features'],dtype=float)
    n=int(feature_data['n_train'])
    temporal=inputs['node_01']
    y=np.asarray(temporal['targets'],dtype=float)
    basis=np.asarray(temporal['basis'],dtype=float)
    prior=np.asarray(inputs['node_02']['fit'],dtype=float)
    q=np.linalg.qr(basis,mode='reduced')[0]
    proj=q@q.T
    traj=y[:,1:]
    smooth=traj@proj
    tail=traj-smooth
    pred=_predict(w,np.column_stack((y[:,0],smooth,tail)),n,([0],slice(1,1+traj.shape[1]),slice(1+traj.shape[1],None)))
    yh=pred[:,0]
    sp=pred[:,1:1+traj.shape[1]]+pred[:,1+traj.shape[1]:]
    # Limit the influence of the extrapolating upstream fit when accounting for its disagreement.
    radius=np.maximum(np.std(y,axis=0),1e-12)
    delta=prior-np.column_stack((yh,sp))
    correction=.015*radius*np.tanh(delta/radius)
    yh=yh+correction[:,0]
    sp=sp+correction[:,1:]
    diag=np.array([yh.mean(),yh.std(),sp[:,-1].mean(),sp[:,-1].std()]) if len(yh) else np.zeros(4)
    return {'y_hat':yh,'state_pred':sp,'diagnostics_vec':diag}
