import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def predict_with(p, row):
        vmax, rho, tau = p
        u = max(0.0, float(row['u_now']))
        s = max(0.0, float(row['s_grad']))
        un = u / max(1e-8, tau)
        return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    def residual(p):
        return np.array([predict_with(p, r) - float(r['y']) for r in rows])
    result = least_squares(residual, x0=np.array([8.0, 0.7, 4.0]), bounds=([0.0, 0.0, 0.05], [100.0, 0.999, 100.0]))
    fit.params = result.x
    return fit

def predict(row):
    p = fit.params
    vmax, rho, tau = p
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    un = u / max(1e-8, tau)
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)))