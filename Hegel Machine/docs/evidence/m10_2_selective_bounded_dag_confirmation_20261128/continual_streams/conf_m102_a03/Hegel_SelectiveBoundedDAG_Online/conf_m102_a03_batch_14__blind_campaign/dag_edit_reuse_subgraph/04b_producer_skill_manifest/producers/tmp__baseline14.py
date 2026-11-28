"""Recover calibrated coordinates and build the frozen heads' feature blocks."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibrate(panel, x):
    u, v = (np.asarray(panel[k], float) for k in ('u', 'v'))
    opp = (np.arange(4) + 2) % 4
    def design(w):
        return np.stack((np.ones_like(u), u, u**3, np.sin(w*u[:,opp])), axis=2)
    w = np.ones(len(u))
    a = design(2.4)
    for _ in range(20):
        c = np.array([_fit(a[:,j]*w[:,None],v[:,j]*w) for j in range(4)])
        r = np.linalg.norm(np.einsum('njk,jk->nj', a,c)-v,axis=1)
        w = np.sqrt(np.minimum(1.,1.5*max(np.median(r),1e-12)/np.maximum(r,1e-12)))
    good = r <= 5*max(np.median(r),1e-12)
    def prof(freq):
        g=design(freq)[good]
        c=np.array([_fit(g[:,j],v[good,j]) for j in range(4)])
        return c,np.einsum('njk,jk->nj',g,c)-v[good]
    f=least_squares(lambda p:prof(p[0])[1].ravel(),[2.4],bounds=(1.5,3.3)).x[0]
    c=prof(f)[0]
    idx=np.arange(4)
    def res(q,row):
        return c[:,0]+c[:,1]*q+c[:,2]*q**3+c[:,3]*np.sin(f*q[opp])-row
    def jac(q, row):
        a=np.diag(c[:,1]+3*c[:,2]*q*q)
        a[idx,opp]=f*c[:,3]*np.cos(f*q[opp]);return a
    return np.array([least_squares(res,np.clip((row-c[:,0])/c[:,1],-1.79,1.79),jac=jac,args=(row,),bounds=(-1.8,1.8),gtol=1e-11).x for row in x])


def _affine(x):
    return np.column_stack((np.ones(len(x)),x))


def _poly(x):
    i,j=np.triu_indices(4)
    return np.column_stack((_affine(x), x[:,i]*x[:,j]))


def _panel(panel,k,x):
    return _poly(x)@_fit(_poly(panel['u']),panel[k])


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']
    train,test=(np.asarray(study[k]['x'],float) for k in ('train','test'))
    x=_calibrate(study['panel_a'],np.vstack((train,test)))
    b=_panel(study['panel_b'],'v',x)
    c=_panel(study['panel_c'],'v1',x)
    d=_panel(study['panel_c'],'v2',x)
    i,j=np.triu_indices(4)
    r=np.column_stack((b,b[:,i]*b[:,j],np.ones(len(x)),x,_poly(x)[:,5:],c,d))
    z=np.column_stack((c,np.ones(len(x)),x,_poly(x)[:,5:],d))
    n=len(train)
    return {'readout_train':r[:n], 'readout_test':r[n:], 'z':z[:n], 'zt':z[n:]}
