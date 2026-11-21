import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    # y = V * (1-rho*s/(1+s)) * u/(km+u) + branch constant
    def pred(p, r):
        V, rho, km, b = p
        un = max(float(r['u_now']), 0.0)
        sg = float(r['s_grad'])
        return V * (1.0-rho*sg/(1.0+sg)) * un/(km+un) + b
    def residual(p):
        return np.array([pred(p, r)-r['y'] for r in rows])
    res = least_squares(residual, x0=[4.0, 0.70, 1.2, 0.0], bounds=([0.01,0.0,0.01,-5.0],[20.0,0.99,20.0,5.0]))
    return res.x.tolist()

def predict(row):
    p = fit([row])
    V, rho, km, b = p
    un = max(float(row['u_now']), 0.0)
    sg = float(row['s_grad'])
    return float(V * (1.0-rho*sg/(1.0+sg)) * un/(km+un) + b)
