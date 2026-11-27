import numpy as np
from scipy.optimize import least_squares

def fit(stress, dt, t, x, y, rate):
    past, hold, memory = stress
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    total = dt.sum()
    cm, rm = np.isfinite(y), np.isfinite(rate)
    obs = np.r_[y[cm], rate[rm]]

    def arrays(logtau):
        tau = np.exp(logtau)
        decay = np.exp(-dt[:, None]/tau)
        later = np.exp(-tail[:, None]/tau)
        long = np.exp(-total/tau)
        k = later*(1-decay)
        dk = (tail[:, None]*k - dt[:, None]*later*decay)/tau
        delta = hold[:, None] - past@k - memory*long
        dd = -past@dk - memory*long*total/tau
        future = np.exp(-t[:, None]/tau)
        rise = 1-future
        a = np.concatenate((delta[:, None, :]*rise, x[:, None, :]*t[None,:,None]), axis=2)
        da = dd[:,None,:]*rise - delta[:,None,:]*future*t[:,None]/tau
        b = np.column_stack((delta/tau,x))
        db = (dd-delta)/tau
        return np.vstack((a[cm],b[rm])), np.vstack((da[cm],db[rm]))

    def mv(a,b):
        return np.einsum('ij,j->i',a,b)
    def mad(e,axis=None):
        return 1.4826*np.nanmedian(np.abs(e-np.nanmedian(e,axis=axis,keepdims=True)),axis=axis)
    w=np.ones(len(obs))/.012
    def fun(p):
        a,_=arrays(p[:2]); return (mv(a,p[2:])-obs)*w
    def jac(p):
        a,da=arrays(p[:2]); return np.column_stack((da*p[2:4],a))*w[:,None]
    def opt(p,f):
        return least_squares(fun,p,jac=jac,loss='cauchy',f_scale=f,x_scale='jac',bounds=(np.r_[np.log([.051,.051]),np.full(x.shape[1]+2,-np.inf)],np.r_[np.log([24.99,24.99]),np.full(x.shape[1]+2,np.inf)]),max_nfev=180)
    starts=[]
    grid=(.2,.35,.55,.9,1.5,2.5,4.,6.5,10.)
    for i,a in enumerate(grid):
        for b in grid[i+1:]:
            q=np.log([a,b]); A,_=arrays(q)
            theta=np.linalg.lstsq(A,obs,rcond=None)[0]
            for _ in range(4):
                err=mv(A,theta)-obs
                u=1/np.sqrt(1+(err/max(.012,3*np.median(abs(err))))**2)
                theta=np.linalg.lstsq(A*u[:,None],obs*u,rcond=None)[0]
            e=mv(A,theta)-obs
            starts.append((np.mean(np.log1p((e/.025)**2)),np.r_[q,theta]))
    candidates=[opt(p,2.) for _,p in sorted(starts,key=lambda p:p[0])[:3]]
    p=min((v for v in candidates if v.x[0]<v.x[1]),key=lambda v:v.cost).x
    for _ in range(2):
        A,_=arrays(p[:2]); err=mv(A,p[2:])-obs
        ec=np.full(cm.shape,np.nan); ec[cm]=err[:cm.sum()]
        sig=np.maximum(.002,mad(ec,0)); rs=max(.002,mad(err[cm.sum():])) if rm.any() else .01
        w=np.r_[np.broadcast_to(1/sig,cm.shape)[cm],np.full(rm.sum(),1/rs)]
        p=opt(p,1.5).x
    return np.exp(p[:2]),p[2:]
