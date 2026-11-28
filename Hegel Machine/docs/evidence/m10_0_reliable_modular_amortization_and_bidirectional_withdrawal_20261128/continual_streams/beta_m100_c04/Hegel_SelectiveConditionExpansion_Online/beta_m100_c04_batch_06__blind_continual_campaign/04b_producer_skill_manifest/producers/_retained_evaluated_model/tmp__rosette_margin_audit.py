import sys, pickle
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit, _inverse, _design, _PAIR
with open('/holdout_task/public_batch.pkl', 'rb') as stream:
    s = pickle.load(stream)[0]['study']
c = s['sensor_calibration_sweeps']; X, Y = c['ref_rosette'], c['raw_rosette']
axis = np.count_nonzero(X, axis=1) == 1
clean = np.max(abs(Y), axis=1) < 4  # Scoring eligibility only; all training rows enter each fit.
fold = np.random.default_rng(193).permutation(len(X)) % 5

def robust_profile(x, y, family):
    z = x[:, _PAIR]
    one = np.ones_like(x)
    if family == 'rational':
        p = [2.5, .19, .024]; bounds = ([1.8, 0, 0], [3.3, 4, 4])
        def basis(v):
            k, a, b = v; t = z*z; d = 1 + a*t + b*t*t
            return np.stack((one, x, np.sin(k*x), z, t/d, t*t/d), axis=-1)
    elif family == 'pair-mix':
        p = [2.5, 2., 0., 0.]; bounds = ([1.8, 1.2, -.5, -.5], [3.3, 3., .5, .5])
        def basis(v):
            k, h, a, b = v
            return np.stack((one, x, np.sin(k*(x+a*z)), z, 1-np.cos(h*(z+b*x))), axis=-1)
    else:
        p = [2.5, 2., .25, .21, -.06]; bounds = ([1.8, 1.2, .05, -.2, -.2], [3.3, 3., .4, .5, .1])
        def basis(v):
            k, h, a, b, d = v
            return np.stack((one, x+a*np.sin(k*x)+b*z+d*(1-np.cos(h*z))), axis=-1)
    w = np.ones(len(x))
    for _ in range(12):
        r = np.sqrt(w)
        def prof(v):
            A = basis(v).transpose(1, 0, 2)
            C = np.stack([np.linalg.lstsq(a*r[:,None], b*r, rcond=None)[0] for a,b in zip(A,y.T)])
            return np.einsum('cnp,cp->nc', A, C)-y, C
        p = least_squares(lambda v: (prof(v)[0]*r[:,None]).ravel(), p, bounds=bounds, max_nfev=120, ftol=1e-10, gtol=1e-10).x
        E, C = prof(p); d = np.linalg.norm(E,axis=1)
        w = 1/(1+(d/max(3*np.median(d),1e-10))**2)
    return p, C

def predict(x,p,c,family):
    z=x[:,_PAIR]
    if family=='rational':
        k,a,b=p; t=z*z; d=1+a*t+b*t*t
        A=np.stack((np.ones_like(x),x,np.sin(k*x),z,t/d,t*t/d),axis=-1)
    elif family=='pair-mix':
        k,h,a,b=p
        A=np.stack((np.ones_like(x),x,np.sin(k*(x+a*z)),z,1-np.cos(h*(z+b*x))),axis=-1)
    else:
        return np.einsum('cnp,cp->nc',_design(x,p),c)
    return np.einsum('ncp,cp->nc',A,c)

obs=np.asarray(s['y_obs_raw']); full_f,full_c=_fit(X,Y); base=_inverse(obs,full_f,full_c)
for family in ('current','rational','pair-mix','tied'):
    E=np.full(len(X),np.nan); I=E.copy(); jitter=[]
    for f in range(5):
        tr=fold!=f; te=(~tr)&clean
        if family=='current': p,q=_fit(X[tr],Y[tr])
        else: p,q=robust_profile(X[tr],Y[tr],family)
        if family=='tied':
            k,h,a,b,d=p; offset,gain=q.T
            p=np.array([k,h]); q=np.stack((offset,gain,a*gain,b*gain,d*gain),axis=1)
        E[te]=np.linalg.norm(predict(X[te],p,q,family)-Y[te],axis=1)
        if family in ('current','tied'):
            I[te]=np.linalg.norm(_inverse(Y[te],p,q)-X[te],axis=1)
            jitter.append(np.max(abs(_inverse(obs,p,q)-base)))
    print(family,'forward p90 / axis max / mixed max',np.quantile(E[clean],.9),max(E[clean&axis]),max(E[clean&~axis]))
    if jitter: print('  inverse axis max / mixed max / max observation shift',max(I[clean&axis]),max(I[clean&~axis]),max(jitter))
print('Study context', list(s), 'row bank counts', np.unique(s['row_bank_id'],return_counts=True))
