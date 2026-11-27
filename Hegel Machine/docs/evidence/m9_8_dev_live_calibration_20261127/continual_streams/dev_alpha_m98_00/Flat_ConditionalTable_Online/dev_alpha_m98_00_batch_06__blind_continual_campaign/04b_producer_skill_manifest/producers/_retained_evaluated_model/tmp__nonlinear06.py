from itertools import combinations_with_replacement
import numpy as np

MONOMIALS = [ix for n in range(1,4) for ix in combinations_with_replacement(range(4),n)]
POWERS = np.array([[ix.count(i) for i in range(4)] for ix in MONOMIALS])

def basis(x,powers):
    return (x[:,None,:] ** powers[None,:,:]).prod(axis=-1)

def bridge_fit(x,raw):
    # The calibration includes a few rows with a gross simultaneous offset.
    affine=np.column_stack((np.ones(len(x)),x))
    resid=np.linalg.norm(affine @ np.linalg.lstsq(affine,raw,rcond=None)[0]-raw,axis=1)
    keep=resid < 5*np.median(resid)
    a=np.column_stack((np.ones(len(x)),basis(x,POWERS)))
    coef=np.linalg.lstsq(a[keep],raw[keep],rcond=None)[0]
    return coef,keep

def invert(raw,coef,powers=POWERS):
    shape=raw.shape
    y=raw.reshape(-1,4)-coef[0]
    loads=y @ np.linalg.inv(coef[1:5])
    jac_pow=np.maximum(powers[None,:,:]-np.eye(4,dtype=int)[:,None,:],0)
    for _ in range(20):
        e=basis(loads,powers) @ coef[1:]-y
        if np.max(np.abs(e)) < 1e-10:
            break
        jac=np.stack([(basis(loads,p)*powers[:,j]) @ coef[1:] for j,p in enumerate(jac_pow)],axis=-1)
        loads-=np.linalg.solve(jac,e[:,:,None])[:,:,0]
    return loads.reshape(shape)
