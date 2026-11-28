import numpy as np
from scipy.optimize import least_squares

def design(z): return np.c_[np.ones(len(z)),z]

def calibrate(p):
    u,v=p['u'],p['v']
    w=np.roll(u,-1,axis=1)
    basis=[np.c_[np.ones(len(u)),u[:,j],w[:,j],u[:,j]*w[:,j],u[:,j]**3] for j in range(4)]
    ids=np.arange(len(u))
    for _ in range(8):
        c=np.column_stack([np.linalg.lstsq(f[ids],v[ids,j],rcond=None)[0] for j,f in enumerate(basis)])
        resid=(np.column_stack([f@c[:,j] for j,f in enumerate(basis)])-v)
        ids=np.argsort(np.linalg.norm(resid,axis=1))[:len(u)-3]
    print('panel A inlier max',np.abs(resid[ids]).max(),'coeff',c)
    return c

def inverse(x,c):
    out=[]
    for row in x:
        def fun(z):
            w=np.roll(z,-1)
            return c[0]+c[1]*z+c[2]*w+c[3]*z*w+c[4]*z**3-row
        out.append(least_squares(fun,(row-c[0])/c[1],gtol=1e-11).x)
    return np.array(out)

def time_modes(t,q,n):
    def basis(r): return -np.expm1(-t[:,None]*r)
    def err(r):
        h=basis(r)
        return (h@np.linalg.lstsq(h,q[:n].T,rcond=None)[0]-q[:n].T).ravel()
    h=basis(np.sort(least_squares(err,[.5,2.],bounds=(.01,10.)).x))
    return h,q@np.linalg.pinv(h.T)
