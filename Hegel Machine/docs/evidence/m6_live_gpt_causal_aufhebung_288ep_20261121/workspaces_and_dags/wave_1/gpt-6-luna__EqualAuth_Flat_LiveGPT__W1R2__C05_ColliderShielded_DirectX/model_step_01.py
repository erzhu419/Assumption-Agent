import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.array([float(r['x_ind']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        vmax, c = p
        return vmax*x/(1.1+x) + c*u - y
    p0 = np.array([5.0, 0.5])
    res = least_squares(residual, p0)
    return {'vmax': float(res.x[0]), 'c_ctrl': float(res.x[1])}

def predict(row):
    p = fit([row]) if False else None
    # Parameters are attached by fit through returned model; see prediction closure below.
    raise RuntimeError('predict requires fitted parameters; use fit_predict-compatible model')