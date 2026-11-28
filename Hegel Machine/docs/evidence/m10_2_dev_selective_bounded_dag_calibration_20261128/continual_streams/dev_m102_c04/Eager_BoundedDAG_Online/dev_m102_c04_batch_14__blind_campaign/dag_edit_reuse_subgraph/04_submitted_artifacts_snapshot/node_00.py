import numpy as np
from scipy.optimize import least_squares


def _ls(a,b):
    return np.linalg.lstsq(a,b,rcond=1e-10)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)),x))


def _calibrate(panel,x):
    u,v=panel['u'],panel['v']
    dist=np.linalg.norm(v-u,axis=1)
    keep=dist<max(2.,4*np.median(dist))
    u,v=u[keep],v[keep]
    def estimate(k):
        a=[np.column_stack((np.ones(len(u)),u[:,j],np.sin(k*u[:,j]),np.sin(k*u[:,(j+2)%4]))) for j in range(4)]
        w=np.array([_ls(a[j],v[:,j]) for j in range(4)]).T
        return w,np.column_stack([a[j]@w[:,j]-v[:,j] for j in range(4)])
    k=least_squares(lambda z:estimate(z[0])[1].ravel(),[2.4],bounds=(1.5,3.5),gtol=1e-10).x[0]
    b,d,c,e=estimate(k)[0]
    q=(x-b)/d
    for _ in range(24):
        q=(x-b-c*np.sin(k*q)-e*np.roll(np.sin(k*q),2,axis=1))/d
    for j in range(len(q)):
        if np.linalg.norm(b+d*q[j]+c*np.sin(k*q[j])+e*np.roll(np.sin(k*q[j]),2)-x[j])>1e-5:
            q[j]=least_squares(lambda z:b+d*z+c*np.sin(k*z)+e*np.roll(np.sin(k*z),2)-x[j],q[j],gtol=1e-11).x
    return q


def _design(x,k,mode):
    pairs=[x[:,i]*x[:,j] for i in range(4) for j in range(i+1,4)]
    h=np.tanh(k*x) if mode=='tanh' else x/(1+k*np.abs(x))
    return np.column_stack((_aff(x),h,*pairs))


def _operator(u,v,x,mode):
    def residual(z):
        a=_design(u,z[0],mode)
        return (a@_ls(a,v)-v).ravel()
    k=least_squares(residual,[1.2 if mode=='tanh' else .1],bounds=(.005,4.),gtol=1e-10).x[0]
    return _design(x,k,mode)@_ls(_design(u,k,mode),v)


def _spatial(study,x):
    q=_calibrate(study['panel_a'],x)
    c=study['panel_c']
    p=_operator(c['u'],c['v1'],q,'tanh')
    r=_operator(c['u'],c['v2'],q,'tanh')
    common=np.column_stack((p,r,np.sum(p*r,axis=1)))
    offdiag=np.column_stack([p[:,i]*r[:,j] for i in range(4) for j in range(4) if i!=j])
    return common, offdiag


def run_node(inputs: dict) -> dict:
    s=inputs['__input__']
    train=s['train']; n=len(train['x'])
    common,offdiag=_spatial(s,np.concatenate((train['x'],s['test']['x'])))
    a=_aff(common); z=a[:n]; t=train['y']
    resid=t-z@_ls(z,t)
    orth=offdiag[:n]-z@_ls(z,offdiag[:n])
    gains=(orth.T@resid)**2/np.maximum(np.sum(orth**2,axis=0),1e-20)
    j=np.argmax(gains)
    if gains[j]>.15*np.sum(resid**2):
        a=np.column_stack((a,offdiag[:,j]))
    return {'common':common,'offdiag':offdiag,'y_hat':a[n:]@_ls(a[:n],t)}
