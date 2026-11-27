from transport_probe import *
from scipy.optimize import minimize
B,E,hx,hy=init(1,'ar','face','velocity'); d0=np.repeat([rx/hx,ry/hy],2)
F=s['source_modes'].reshape(M,-1,order='F').T
Z=s['Z_nuisance']; raw=s['y_obs_raw']; A=np.column_stack((F, F@np.column_stack([p['w_pulse'] for p in pulses]))); ns=len(ids)

def design(d):
    return np.concatenate([solve(B+np.diag(E@d+mu),A,check_finite=False)[ids] for mu in shifts])

dref=least_squares(lambda d:(design(d)[:,M:]-y).ravel(),d0,gtol=1e-12).x
print('tracer fitted change:',dref-d0)
# Eliminate the budget before profiling sources and measurement nuisance.
from scipy.linalg import null_space
Q=null_space(np.ones((1,M))); w0=np.full(M,s['total_budget']/M)
def profile(d,train):
    S=design(d)[:,:M]; X=np.column_stack((S@Q,Z)); z=raw-S@w0
    beta=np.linalg.lstsq(X[train],z[train],rcond=None)[0]
    return X@beta-z, w0+Q@beta[:M-1]
for label,W in [('nominal',np.empty((4,0))),('axis',np.repeat(np.eye(2),2,axis=0)),('walls',np.eye(4)),('tracer',np.empty((4,0)))]:
    for fold in [-1,0,1,2,3]:
        train=np.ones(len(raw),bool) if fold<0 else np.arange(len(raw))//4%4!=fold
        d=(dref if label=='tracer' else d0).copy()
        if W.shape[1]:
            fit=least_squares(lambda a:profile(d0+W@a,train)[0][train],np.zeros(W.shape[1]),bounds=(-1.5,1.5),gtol=1e-12)
            d=d0+W@fit.x
        e,w=profile(d,train)
        print(label,fold,'d',np.round(d-d0,4),'RMSE train/test',np.sqrt(np.mean(e[train]**2)),np.sqrt(np.mean(e[~train]**2)) if fold>=0 else None,'w range',w.min(),w.max())
