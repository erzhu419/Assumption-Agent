import numpy as np
from scipy.optimize import minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _cal(panel):
    u,v=panel['u'],panel['v']; a=_affine(u); mask=np.ones(len(u),bool)
    for _ in range(8):
        w=_ls(a[mask],v[mask]); e=np.linalg.norm(a@w-v,axis=1)
        new=e<=max(1e-9,4*np.median(e))
        if new.sum()<8 or np.array_equal(mask,new): break
        mask=new
    d=np.array([_ls(a[mask][:,[0,j+1]],v[mask,j]) for j in range(4)]).T
    if np.median(np.linalg.norm((u*d[1]+d[0]-v)[mask],axis=1))<=1.35*np.median(e[mask]):
        return d[0],np.diag(d[1])
    return w[0],w[1:]


def _quad(z):
    return z[:,[0,0,1,2]]*z[:,[1,3,2,3]]


def _ops(panel):
    u=panel['u']; v=panel['v2']
    def err(k):
        f=_affine(np.cos(k*u)); return np.linalg.norm(f@_ls(f,v)-v)
    grid=np.linspace(.2,7,69); j=np.argmin([err(k) for k in grid]); k=minimize_scalar(err,bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]),method='bounded').x
    return k,_ls(_affine(_quad(u)),panel['v1']),_ls(_affine(np.cos(k*u)),v)


def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; x=s['train']['x']; xt=s['test']['x']; a,A=_cal(s['panel_a']); b,B=_cal(s['panel_b']); k,wp,wr=_ops(s['panel_c']); Bi=np.linalg.inv(B); Ai=np.linalg.inv(A)
    def feature(x):
        z=((x-b)@Bi@A+a)@B+b
        p=_affine(_quad(z))@wp; r=_affine(np.cos(k*((x-a)@Ai)))@wr
        return _affine(np.c_[x,p,r,p*r])
    f,g=feature(x),feature(xt)
    return {'train_features':f,'test_features':g,'regression':g@_ls(f,s['train']['y'])}
