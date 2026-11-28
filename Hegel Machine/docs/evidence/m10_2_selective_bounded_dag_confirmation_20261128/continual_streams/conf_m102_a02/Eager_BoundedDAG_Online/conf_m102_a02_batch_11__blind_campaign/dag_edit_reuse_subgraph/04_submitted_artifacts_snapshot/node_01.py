"""Calibrate the raw design and fit separate endpoint and trajectory readouts."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import minimize_scalar


def _fit(a,b):
    return np.linalg.lstsq(a,b,rcond=1e-9)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)),x))


def _latent(panel,x):
    u,v=map(np.asarray,(panel['u'],panel['v']))
    keep=np.max(np.abs(v-np.median(v,axis=0)),axis=1)<3.5*np.median(np.max(np.abs(v-np.median(v,axis=0)),axis=1))
    u,v=u[keep],v[keep]
    def design(u,w):
        return np.column_stack((_aff(u),np.sin(w*u)))
    def loss(w):
        a=design(u,w)
        return np.mean((a@_fit(a,v)-v)**2)
    w=minimize_scalar(loss,bounds=(1.6,3.3),method='bounded').x
    b=_fit(design(u,w),v)
    linear=_fit(_aff(u),v)
    z=(x-linear[0])@np.linalg.pinv(linear[1:])
    d=x.shape[1]
    for _ in range(16):
        j=b[1:d+1][None,:,:]+(w*np.cos(w*z))[:,:,None]*b[d+1:][None,:,:]
        r=design(z,w)@b-x
        dz=(np.linalg.pinv(j.transpose(0,2,1))@r[:,:,None])[:,:,0]
        z-=np.clip(dz,-.25,.25)
    return z


def _basis(z,k):
    return np.column_stack([np.prod(z[:,ix],axis=1) for degree in range(k+1)
                            for ix in combinations_with_replacement(range(z.shape[1]),degree)])


def _features(study,z):
    c=study['panel_c']; u=c['u']
    def odd(u):return np.column_stack((_aff(u),np.arcsinh(3.39*u)))
    def even(u):return np.column_stack((_aff(u),np.sqrt(1+.927*u*u)-1))
    g=odd(z)@_fit(odd(u),c['v1'])
    h=even(z)@_fit(even(u),c['v2'])
    base=np.column_stack((_basis(z,2),g,h))
    w1,w2=np.asarray(study['telemetry_vec'],float)[:2]
    cosine=np.cos(w1*z)
    modulated=np.column_stack((base,*(cosine*np.roll(z,k,axis=1) for k in (1,2,-1))))
    opposite=np.roll(z,2,axis=1)
    paired=np.column_stack((base,np.arcsinh(w2*(z+opposite)),np.arcsinh(w2*(z-opposite))))
    return modulated,paired,_aff(z)


def _regularized(a,y,alpha,delta):
    scale=np.maximum(a.std(axis=0),1e-7)
    scale[0]=1.
    b=a/scale
    penalty=np.diag(np.r_[0.,np.full(b.shape[1]-1,alpha)])
    weight=np.ones(len(a))
    for _ in range(12):
        c=np.linalg.solve((b.T*weight)@b+penalty,(b.T*weight)@y)
        if delta is None:
            break
        weight=np.minimum(1.,delta/np.maximum(np.abs(y-b@c),1e-9))
    return c/scale


def run_node(inputs: dict) -> dict:
    study,raw=inputs['__input__'],inputs['node_00']
    a=np.vstack((raw['a'],raw['a_test'])); n=len(raw['a'])
    z=_latent(study['panel_a'],a[:,1:])
    ep,paired,tr=_features(study,z)
    y,trajectory=study['train']['y'],study['train']['traj']
    i,j=ep.shape[1],ep.shape[1]+paired.shape[1]
    coef=np.zeros((j+tr.shape[1],1+trajectory.shape[1]))
    coef[:i,0]=.65*_regularized(ep[:n],y,.05,.04)
    coef[i:j,0]=.35*_regularized(paired[:n],y,.25,None)
    coef[j:,1:]=_fit(tr[:n],trajectory)
    return {'test_features':np.column_stack((ep[n:],paired[n:],tr[n:])), 'coefficients':coef}
