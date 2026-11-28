import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import minimize_scalar


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _design(z):
    return np.column_stack((np.ones(len(z)), z))


def _coordinates(study, x):
    p = study['panel_a']
    u, v = np.asarray(p['u']), np.asarray(p['v'])
    keep = np.linalg.norm(v, axis=1) < 7
    u, v = u[keep], v[keep]
    nxt = np.roll(u, -1, axis=1)
    w = np.array([_fit(_design(np.column_stack((u[:,j], nxt[:,j], u[:,j]*nxt[:,j], u[:,j]**3))), v[:,j]) for j in range(4)])
    z = (x-w[:,0])/w[:,1]
    for _ in range(18):
        n = np.roll(z, -1, axis=1)
        e = w[:,0]+w[:,1]*z+w[:,2]*n+w[:,3]*z*n+w[:,4]*z**3-x
        jac = np.zeros((len(z),4,4)); j = np.arange(4)
        jac[:,j,j] = w[:,1]+w[:,3]*n+3*w[:,4]*z*z
        jac[:,j,(j+1)%4] = w[:,2]+w[:,3]*z
        z -= np.clip(np.linalg.solve(jac,e[...,None])[...,0],-.6,.6)
    p=study['panel_b']; b=_design(z)@_fit(_design(p['u']),p['v'])
    p=study['panel_c']; cyc=lambda a:_design(a*np.roll(a,-1,axis=1)); c=cyc(z)@_fit(cyc(p['u']),p['v1'])
    def cv(a,k): return _design(np.cos(k*a)-1)
    def err(k):
        h=cv(p['u'],k); return np.mean((h@_fit(h,p['v2'])-p['v2'])**2)
    k=minimize_scalar(err,bounds=(1.4,3.3),method='bounded').x
    d=cv(z,k)@_fit(cv(p['u'],k),p['v2'])
    return z,b,c,d,np.cos(k*z)


def _predict(study):
    tr,te=study['train'],study['test']; n=len(tr['x'])
    z,_,c,d,g=_coordinates(study,np.vstack((tr['x'],te['x'])))
    p=z*np.roll(z,-1,axis=1)
    y=np.column_stack((tr['y'],tr['traj']))
    def cross(a,b):
        return (a[:,:,None]*b[:,None,:]).reshape(len(z),-1)
    # Distinguish cyclic pair products times linear and even panel coordinates.
    blocks=[z,p,g,cross(z,p),cross(p,g),
            np.column_stack([z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(4),2)]),
            *[z*np.roll(c,k,axis=1) for k in range(4)],
            *[c*np.roll(d,k,axis=1) for k in range(4)]]
    F=np.column_stack(blocks); mu=F[:n].mean(0); sd=np.maximum(F[:n].std(0),1e-10)
    F=(F-mu)/sd
    ids=np.split(np.arange(F.shape[1]),np.cumsum([a.shape[1] for a in blocks])[:-1]); ym=y.mean(0); Y=y-ym
    results={}
    def evaluate(groups):
        if groups in results:return results[groups]
        j=np.concatenate([ids[k] for k in groups]); a=F[:n,j]; b=F[n:,j]
        u,s,v=np.linalg.svd(a,full_matrices=False)
        pos=s>s[0]*1e-10; u,s,v=u[:,pos],s[pos],v[pos]; uy=u.T@Y
        scores=np.full(2,np.inf); preds=[None,None]
        for reg in (0.,.003,.03,.3,2.):
            if reg==0 and len(s)>=n-3:continue
            gain=s*s/(s*s+reg)
            res=(Y-u@(gain[:,None]*uy))/np.maximum(1-1/n-(u*u)@gain,1e-10)[:,None]
            e=np.array([np.mean(abs(res[:,0])),np.mean(abs(res[:,1:]))]); e*=1+.001*len(s)
            for k in range(2):
                if e[k]<scores[k]:
                    scores[k]=e[k]; preds[k]=b@v.T@((gain/s)[:,None]*uy)+ym
        results[groups]=(scores,preds);return results[groups]
    def tuples(names):return tuple(sorted(set(names)))
    base=[(0,1,2,3,4),(0,1,2,3,4,5),(0,1,2),(1,2),(0,1,2,7,10),(0,1,2,7,10,5),(1,2,7,10),(1,2,7,10,5)]
    for groups in base:evaluate(tuples(groups))
    for target in range(2):
        beam=base
        for _ in range(5):
            nxt=set()
            for groups in beam:
                for k in range(len(blocks)):
                    a=tuple(sorted(set(groups)^{k}))
                    if a:nxt.add(a)
            for groups in nxt:evaluate(groups)
            beam=sorted(results,key=lambda g:(results[g][0][target],g))[:7]
    out=[]
    for k,cols in enumerate((slice(0,1),slice(1,None))):
        ranked=sorted(results,key=lambda g:(results[g][0][k],g)); best=results[ranked[0]][0][k]; pred=[]; weights=[]
        for key in ranked[:8]:
            score,val=results[key]
            if score[k] > 1.20*best:break
            pred.append(val[k][:,cols]);weights.append(np.exp(-15*(score[k]/max(best,1e-12)-1)))
        out.append(np.average(pred,axis=0,weights=weights))
    return np.column_stack(out)


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; sparse=inputs['node_00']; p=_predict(study)
    return {'panel_pred':np.column_stack((2*p[:,0]-sparse['y_hat'], (p[:,1:]-.35*sparse['state_pred'])/.65))}
