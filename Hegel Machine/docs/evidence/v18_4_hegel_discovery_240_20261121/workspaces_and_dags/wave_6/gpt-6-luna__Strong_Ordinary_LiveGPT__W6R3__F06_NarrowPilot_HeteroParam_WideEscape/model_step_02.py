import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p):
        j0, beta, rho, c_sqrt = p
        return j0*u*np.exp(-beta/(0.5+u))*(1.0-rho*s/(1.0+s)) + c_sqrt*np.sqrt(np.maximum(u, 0.0))
    starts = ([2.0, 1.0, 0.5, 0.5], [4.0, 2.0, 0.7, 0.0], [1.0, 0.1, 0.2, 1.0])
    best = None
    for p0 in starts:
        res = least_squares(lambda p: pred(p)-y, p0, max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    params = best.x.tolist()
    return params

def predict(row):
    j0, beta, rho, c_sqrt = fit([dict(row, y=row['y'])])
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(j0*u*np.exp(-beta/(0.5+u))*(1.0-rho*s/(1.0+s)) + c_sqrt*np.sqrt(max(u, 0.0)))