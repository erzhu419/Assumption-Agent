"""Fit spatial readout and shared trajectory coefficients."""
import numpy as np


def _fit(x,y,blocks):
    scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-8)
    A=x/scale
    p=A.shape[1]; G=A.T@A; B=A.T@y
    best=None
    for penalty in blocks:
        for lam in (.002,.02,.1,.5,2.,8.):
            gram=G+np.diag(lam*penalty)
            inv=np.linalg.solve(gram,A.T)
            c=inv@y
            r=(A@c-y)/(np.maximum(1.-np.sum(A*inv.T,axis=1),1e-7).reshape((-1,)+(1,)*(y.ndim-1)))
            val=np.mean(abs(r))
            if best is None or val<best[0]: best=(val,c)
    return best[1]/scale.reshape((-1,)+(1,)*(y.ndim-1))


def run_node(inputs: dict) -> dict:
    raw=inputs['__input__']; f=inputs['node_00']; A=f['readout_train']; B=f['z']
    y=np.asarray(raw['train']['y']); t=np.asarray(raw['train']['traj']); _,_,q=np.linalg.svd(t,full_matrices=False); q=q[:2]
    ry=[np.r_[np.ones(4), np.full(10,k), np.full(A.shape[1]-18,j), np.full(4,j*3)] for k in (.15,1.,10.) for j in (3.,30.,300.)]
    rs=[np.r_[np.ones(4),np.full(B.shape[1]-8,j),np.full(4,j*k)] for j in (3.,30.,300.) for k in (.3,3.,30.)]
    return {'readout_coef':_fit(A,y,ry),'state_coef':_fit(B,t@q.T,rs)@q}
