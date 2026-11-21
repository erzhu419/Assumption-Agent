import math
import numpy as np
from scipy.optimize import least_squares

def _features(r):
    u=max(0.0,float(r['u_now']))
    s=float(r['s_grad'])
    return u, 1.0 - 0.7*s/(1.0+s), math.sqrt(u)

def fit(train_rows):
    # Bridge signatures and increasing-but-saturating response point to the
    # inland-hub/thermal-fin siphon law; fit its 3 identifiable coefficients.
    X=[]; y=[]
    for r in train_rows:
        u,q,root=_features(r)
        X.append([q*u/(1.2+u), root, 1.0])
        y.append(float(r['y']))
    X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float)
    def residual(p):
        return X @ p-y
    starts=[np.array([2.,0.,0.]),np.array([1.,.2,.1]),np.array([3.,-.2,0.])]
    best=None
    for p0 in starts:
        res=least_squares(residual,p0,max_nfev=10000)
        if best is None or np.sum(res.fun**2)<np.sum(best.fun**2): best=res
    return best.x

def predict(row):
    # Accept either an already fitted parameter vector or a row-local fit is
    # not possible; fit() stores fitted parameters in module state.
    u,q,root=_features(row)
    p=_PARAMS
    return float(p[0]*q*u/(1.2+u)+p[1]*root+p[2])

_PARAMS=np.array([0.0,0.0,0.0])
_fit_impl=fit
def fit(train_rows):
    global _PARAMS
    _PARAMS=_fit_impl(train_rows)
    return _PARAMS
