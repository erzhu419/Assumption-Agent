"""Recover latent coordinates from the paired calibration panels."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _a_basis(z):
    w = np.roll(z, -1, axis=-1)
    return np.stack((np.ones_like(z), z, w, z*w, z**3), axis=-1)


def _sensor_inverse(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u','v'))
    a = _a_basis(u).transpose(1,0,2)
    rng = np.random.default_rng(2)
    ids = np.argsort(rng.random((96,len(u))),axis=1)[:,:6]
    coeff = np.linalg.pinv(a[:,ids]) @ v.T[:,ids,None]
    err = np.sqrt(np.sum((np.einsum('dnp,dsp->dsn',a,coeff[...,0])-v.T[:,None,:])**2,axis=0))
    count = max(6,int(.8*len(u)))
    loss = np.mean(np.partition(err,count-1,axis=1)[:,:count]**2,axis=1)
    r = err[np.argmin(loss)]
    keep = r <= max(8*np.median(r),1e-3)
    p = np.stack([np.linalg.lstsq(a[j,keep],v[keep,j],rcond=None)[0] for j in range(u.shape[1])])
    def forward(z):
        return np.einsum('dp,dp->d',_a_basis(z),p)
    def jac(z):
        j = np.arange(len(z)); w = np.roll(z,-1)
        out = np.zeros((len(z),len(z)))
        out[j,j] = p[:,1]+p[:,3]*w+3*p[:,4]*z*z
        out[j,(j+1)%len(z)] = p[:,2]+p[:,3]*z
        return out
    return np.array([least_squares(lambda z:forward(z)-row,row,jac=jac,gtol=1e-12).x for row in x]).reshape(x.shape)


def _fit(panel, key, fun, bounds):
    u,v = (np.asarray(panel[k],dtype=float) for k in ('u',key))
    def basis(x,s): return np.column_stack((np.ones(len(x)),x,fun(x,s)))
    def loss(s):
        a=basis(u,s); inv=np.linalg.pinv(a); r=v-a@(inv@v)
        return np.mean(np.abs(r/np.maximum(1-np.sum(a*inv.T,axis=1),1e-8)[:,None]))
    s=minimize_scalar(loss,bounds=bounds,method='bounded').x
    p=np.linalg.lstsq(basis(u,s),v,rcond=None)[0]
    return lambda x:basis(x,s)@p


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; x=np.asarray(study['train']['x'],dtype=float)
    z=_sensor_inverse(study['panel_a'],np.concatenate((x,study['test']['x'])))
    b=_fit(study['panel_b'],'v',lambda q,s:q**3/(1+s*q*q),(.01,2.))
    d=_fit(study['panel_c'],'v2',lambda q,s:np.expm1(-s*q*q),(.02,1.))
    c=d(z)-b(d(z))
    return {'c':c[:len(x)],'query':c[len(x):]}
