import numpy as np
from scipy.optimize import least_squares

def _features(row):
    o=float(row.get('u_outdoor',0.0)); i=float(row.get('u_indoor',0.0))
    p=float(row.get('p_weather',0.0)); a=float(row.get('a_foot',0.0)); d=float(row.get('impact_idx',0.0))
    return np.array([1.0, o*p, i, a, d, i*a, i*d, a*d, i*i, d*d, i*a*(1.0-d)], dtype=float)

def fit(train_rows):
    X=np.vstack([_features(r) for r in train_rows]); y=np.array([float(r['y']) for r in train_rows])
    # Mild shrinkage stabilizes the correlated engineered terms in this small pilot.
    scale=np.maximum(np.linalg.norm(X[:,1:],axis=0),1.0)
    Z=X.copy(); Z[:,1:]=X[:,1:]/scale
    def residual(theta):
        return np.r_[Z@theta-y, 0.015*theta[1:]]
    theta=least_squares(residual,np.zeros(Z.shape[1]),max_nfev=5000).x
    beta=theta.copy(); beta[1:]/=scale
    return beta

def predict(row):
    beta=fit(_TRAIN_ROWS)
    return float(_features(row)@beta)
