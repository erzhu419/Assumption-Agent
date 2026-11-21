import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # y = x_now + a*(u_now-u_past)*exp(-H/tau) + b*(u_now-u_past)*H*exp(-H/tau)
    X=[]; yy=[]
    for r in train_rows:
        d=float(r['u_now'])-float(r['u_past']); h=float(r['H'])
        X.append([d*math.exp(-h/5.0), d*h*math.exp(-h/5.0)])
        yy.append(float(r['y'])-float(r['x_now']))
    X=np.asarray(X,float); yy=np.asarray(yy,float)
    res=least_squares(lambda p: X@p-yy, np.zeros(2))
    return tuple(float(v) for v in res.x)

def predict(row):
    # Fit parameters should be attached by caller via fit output; use available train_rows is not passed.
    # Standard fit/predict protocol can retain the fitted values on the function.
    a,b = predict.params
    d=float(row['u_now'])-float(row['u_past']); h=float(row['H'])
    return float(row['x_now']) + d*math.exp(-h/5.0)*(a+b*h)

# Store parameters from latest fit for predict.
_old_fit=fit
def fit(train_rows):
    params=_old_fit(train_rows)
    predict.params=params
    return params
predict.params=(0.0,0.0)