import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def pred(p, uu, ss):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * ss / (1.0 + ss)) * (uu / (1.2 + uu)) + c_loc * np.sqrt(uu)
    result = least_squares(lambda p: pred(p, u, s) - y, x0=np.array([6.0, 0.5, 0.1]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    fit.params = result.x
    return fit

def predict(row):
    vmax, rho, c_loc = fit.params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(u))