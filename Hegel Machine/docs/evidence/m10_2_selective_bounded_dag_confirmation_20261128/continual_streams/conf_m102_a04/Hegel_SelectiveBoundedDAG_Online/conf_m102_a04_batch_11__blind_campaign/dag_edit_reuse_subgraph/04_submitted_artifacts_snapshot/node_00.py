"""Calibrate coupled coordinates and operator maps from the supplied panels."""
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator


def _ls(a,b):
    return np.linalg.lstsq(a,b,rcond=1e-10)[0]


def _one(a):
    return np.column_stack((np.ones(len(a)),a))


def _calibrate(panel):
    u,v=map(np.asarray,(panel['u'],panel['v']))
    n,d=u.shape
    j=np.tile(np.eye(d),(n,1))
    a=u.ravel(); b=np.roll(u,-2,axis=1).ravel(); y=v.ravel()
    def design(q):
        return np.c_[j,j*a[:,None],np.tanh(q[0]*a),np.sin(q[1]*b)]
    keep=np.repeat(np.linalg.norm(v,axis=1)<4,d)
    for _ in range(4):
        def residual(q):
            f=design(q)[keep]; return f@_ls(f,y[keep])-y[keep]
        q=least_squares(residual,[1.8,2.4],bounds=([.7,1.4],[4.,3.5]),gtol=1e-12).x
        f=design(q); c=_ls(f[keep],y[keep]); err=np.abs(f@c-y)
        keep=err<max(.0008,6*np.median(err))
    return q,c


def _inverse(x,q,c):
    d=x.shape[1]; off=c[:d]; scale=c[d:2*d]; k,h=c[-2:]
    z=np.zeros_like(x); candidates=[]
    grid=np.linspace(-4,4,4097)
    inv=[PchipInterpolator(off[j]+scale[j]*grid+k*np.tanh(q[0]*grid),grid) for j in range(d)]
    scan=np.linspace(-2.5,2.5,501)
    for i,j in ((0,2),(1,3)):
        def fn(b):
            a=inv[i](x[:,i,None]-h*np.sin(q[1]*b))
            return off[j]+scale[j]*b+k*np.tanh(q[0]*b)+h*np.sin(q[1]*a)-x[:,j,None]
        r=fn(scan[None,:]); mask=r[:,:-1]*r[:,1:]<=0
        ids, ix=np.nonzero(mask); lo=scan[ix].copy(); hi=scan[ix+1].copy(); fl=r[ids,ix]
        for _ in range(25):
            mid=(lo+hi)*.5
            a=inv[i](x[ids,i]-h*np.sin(q[1]*mid))
            fm=off[j]+scale[j]*mid+k*np.tanh(q[0]*mid)+h*np.sin(q[1]*a)-x[ids,j]
            left=fl*fm<=0; lo=np.where(left,lo,mid); hi=np.where(left,mid,hi)
        bb=(lo+hi)*.5; aa=inv[i](x[ids,i]-h*np.sin(q[1]*bb)); sets=[]
        for p in range(len(x)):
            roots=np.c_[aa[ids==p],bb[ids==p]]
            if len(roots)==0:
                b=scan[np.argmin(abs(r[p]))]; roots=np.array([[inv[i](x[p,i]-h*np.sin(q[1]*b)),b]])
            roots=roots[np.argsort(np.sum(roots**2,axis=1))]; z[p,[i,j]]=roots[0]; sets.append(roots)
        candidates.append(sets)
    return z,candidates


def _operators(panel,z):
    u=panel['u']; v1=panel['v1']; v2=panel['v2']
    def f1(a,q): return _one(np.c_[a,a/(1+q[0]*np.abs(a))])
    def f2(a,q): return _one(np.c_[a,a*a,a*a/(1+q[0]*a*a)])
    ans=[]
    for fun,v,start in ((f1,v1,.8),(f2,v2,.3)):
        def resid(q):
            f=fun(u,q); return (f@_ls(f,v)-v).ravel()
        q=least_squares(resid,[start],bounds=([.02],[5]),gtol=1e-12).x
        ans.append(fun(z,q)@_ls(fun(u,q),v))
    return ans


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; tr=study['train']['x']; x=np.vstack((tr,study['test']['x']))
    z,roots=_inverse(x,*_calibrate(study['panel_a'])); c,d=_operators(study['panel_c'],z)
    w=np.c_[c,d]; w=(w-w[:len(tr)].mean(0))/np.maximum(w[:len(tr)].std(0),.1)
    return {'features':w,'coordinates':z,'soft':c,'curvature':d}
