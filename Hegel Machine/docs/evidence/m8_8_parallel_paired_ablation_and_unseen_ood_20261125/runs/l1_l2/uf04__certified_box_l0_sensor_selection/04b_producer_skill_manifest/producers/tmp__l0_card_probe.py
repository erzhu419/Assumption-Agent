import glob, pickle, sys, time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
sys.path.insert(0, '/holdout_task')
from certified_l0_solver import _Problem

def cardinality_probe(a):
    X,y,r,b,pen = (a[k] for k in ('X','y','ridge','box','l0_penalty'))
    q = _Problem(X,y,r,b,pen)
    p = q.p
    state = np.full(p,-1,dtype=np.int8)
    lower = []
    start = np.r_[np.zeros(p),0.]
    corr = np.argsort(-abs(q.c))
    for k in range(p+1):
        initial = np.zeros(p,dtype=bool); initial[corr[:k]] = True
        z = q.fit_fixed(initial)
        q.improve(z)
        if k in (0,p):
            lower.append(q.bound(z,initial.astype(np.int8))[0]); continue
        def modes(x):
            g = q.K@x-q.c
            v = np.clip(-g/q.d,-q.t,q.t)
            w = np.clip(-g/q.d,-b,b)
            off = v*(g+.5*q.d*v)
            on = pen+w*(g+.5*q.d*w)
            return v,w,off,on
        def fg(point):
            x, multiplier = point[:-1],point[-1]
            v,w,off,on = modes(x)
            diff = (off-on-multiplier)/temp
            zbar = v + expit(diff)*(w-v)
            cost = off-temp*np.logaddexp(0,diff)
            return .5*x@(q.K@x)-cost.sum()+k*multiplier, np.r_[q.K@(x-zbar),k-expit(diff).sum()]
        for temp in (2e-3*pen,2e-5*pen,1e-7*pen):
            start = minimize(fg,start,jac=True,method='L-BFGS-B',bounds=[(-b,b)]*p+[(None,None)],options={'maxiter':250,'ftol':1e-15,'gtol':1e-9}).x
        x = start[:-1]
        _, _, off, on = modes(x)
        mask = np.zeros(p,dtype=bool); mask[np.argsort(on-off)[:k]] = True
        q.improve(x,mask)
        base, off_extra, on_extra, _ = q.bound(x,state)
        # Compare k choices algebraically; these are not k-support enumerations.
        ix = np.argsort(on_extra.astype(np.longdouble)-off_extra.astype(np.longdouble))[:k]
        delta = off_extra.astype(np.longdouble); delta[ix] = on_extra[ix]
        lower.append(float(np.nextafter(float(np.longdouble(base)+delta.sum()),-np.inf)))
    return q.best, q.best-min(lower), int(np.argmin(lower)), np.round(np.sort(lower)[:3],8)

if __name__ == '__main__':
    from l0_coupling_stress import run
    for seed in (703,704,705):
        print('v05',run(('common',seed)),flush=True)
    for f in sorted(glob.glob('/tmp/l0_extra_common_*.pkl')):
        a=pickle.load(open(f,'rb')); t=time.perf_counter(); print('cardinality',f,cardinality_probe(a),'seconds',time.perf_counter()-t,flush=True)
