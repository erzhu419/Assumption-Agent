import numpy as np
from scipy.optimize import minimize_scalar

_PAIRS = [(i,j) for i in range(4) for j in range(i+1,4)]

def _affine(x):
    return np.column_stack((np.ones(len(x)),x))

def _ls(a,b):
    return np.linalg.lstsq(a,b,rcond=1e-10)[0]

def _cross(x):
    return np.column_stack([x[:,i]*x[:,j] for i,j in _PAIRS])

def _calibration(panel,x):
    u,v=panel['u'],panel['v']; a=_affine(u)
    mask=np.ones(len(u),dtype=bool)
    for _ in range(10):
        e=np.linalg.norm(a@_ls(a[mask],v[mask])-v,axis=1)
        new=e<=max(.42,3*np.median(e[mask]))
        if new.sum()<12 or np.array_equal(new,mask): break
        mask=new
    def fit(k):
        d=np.column_stack((a,np.sin(k*u)))[mask]; w=_ls(d,v[mask]); return np.mean((d@w-v[mask])**2),w
    for _ in range(4):
        k=minimize_scalar(lambda k:fit(k)[0],bounds=(1.5,3.2),method='bounded').x
        w=fit(k)[1]; e=np.linalg.norm(a@w[:5]+np.sin(k*u)@w[5:]-v,axis=1)
        new=e<=max(.001,4*np.median(e[mask]))
        if new.sum()<12 or np.array_equal(new,mask): break
        mask=new
    q=np.linalg.solve(w[1:5].T,(x-w[0]).T).T
    for _ in range(18):
        err=w[0]+q@w[1:5]+np.sin(k*q)@w[5:]-x
        jac=w[1:5]+(k*np.cos(k*q))[:,:,None]*w[5:]
        delta=np.linalg.solve(jac.transpose(0,2,1),err[...,None])[...,0]
        q-=np.clip(delta,-.5,.5)
        if np.max(np.abs(delta),initial=0)<1e-11: break
    return q

def _operators(panel,q):
    u=panel['u']; a=np.column_stack((_affine(u),_cross(u)))
    p=np.column_stack((_affine(q),_cross(q)))@_ls(a,panel['v1'])
    def cosine(x,k): return np.column_stack((_affine(x),np.cos(k*x)))
    def fit(k):
        f=cosine(u,k); w=_ls(f,panel['v2']); return np.mean((f@w-panel['v2'])**2),w
    k=minimize_scalar(lambda k:fit(k)[0],bounds=(.6,3.5),method='bounded').x
    return p,cosine(q,k)@fit(k)[1]

def _features(p,r):
    common=np.column_stack((p,r,np.sum(p*r,axis=1)))
    offdiag=np.column_stack([p[:,i]*r[:,j] for i in range(4) for j in range(4) if i!=j])
    return common,offdiag

def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; tr=s['train']; n=len(tr['x']); x=np.concatenate((tr['x'],s['test']['x']))
    q=_calibration(s['panel_a'],x)
    p,r=_operators(s['panel_c'],q)
    common,offdiag=_features(p,r)
    a=_affine(common)
    return {'common':common,'offdiag':offdiag,'y_hat':a[n:]@_ls(a[:n],tr['y'])}
