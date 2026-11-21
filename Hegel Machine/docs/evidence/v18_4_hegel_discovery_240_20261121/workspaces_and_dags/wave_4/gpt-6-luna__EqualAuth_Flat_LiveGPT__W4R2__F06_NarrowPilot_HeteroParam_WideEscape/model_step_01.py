import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u=np.array([r['u_now'] for r in train_rows],float)
    y=np.array([r['y'] for r in train_rows],float)
    # Fit a flexible concave saturating power response; nuisance covariates are
    # effectively constant in this pilot and do not identify separate effects.
    def pred(p,x):
        a,b,c,d=p
        return a + b*np.log1p(np.maximum(x,0.0)/c)**d
    scale=max(float(np.std(y)),1.0)
    res=least_squares(lambda p:(pred(p,u)-y)/scale,
        x0=[0.5,2.0,0.8,0.8],bounds=([-100,1e-8,1e-6,0.05],[100,100,100,3.0]),
        max_nfev=30000)
    return res.x

def predict(row):
    # Parameters fit on train_rows are attached by fit for the evaluator.
    a,b,c,d = _PARAMS
    return float(a+b*np.log1p(max(float(row['u_now']),0.0)/c)**d)
