import numpy as np
from scipy.optimize import least_squares

def _features(row):
    x=float(row['x_now']); u=float(row['u_now']); p=float(row['u_past']); h=float(row['H'])
    return np.array([1.0,x,u,p,h,x*x,u*u,x*u,x*h,u*h,p*u],dtype=float)

def fit(train_rows):
    X=np.vstack([_features(r) for r in train_rows])
    y=np.array([float(r['y']) for r in train_rows])
    # Numerical least-squares fit; mild ridge stabilization for correlated predictors.
    def residual(c):
        return np.r_[X@c-y, 0.025*c[1:]]
    c0=np.linalg.lstsq(X,y,rcond=None)[0]
    return least_squares(residual,c0,max_nfev=3000).x

def predict(row):
    c=fit([row]) if False else None
    # Coefficients are fitted on the supplied training set and cached by fit.
    return float(np.dot(_COEF,_features(row)))

_COEF=np.zeros(11)
_fit_original=fit
def fit(train_rows):
    global _COEF
    _COEF=_fit_original(train_rows)
    return _COEF
