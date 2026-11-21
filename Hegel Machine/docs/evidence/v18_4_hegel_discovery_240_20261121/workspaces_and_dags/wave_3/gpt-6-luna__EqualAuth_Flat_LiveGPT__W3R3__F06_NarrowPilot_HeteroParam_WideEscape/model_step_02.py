import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([max(1e-8, float(r['u_now'])) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    # Saturating base field plus a concave local root contribution and
    # a gradient-dependent sink; fit all target parameters to observations.
    def pred(p):
        vmax, km, c_loc, k_sink = p
        return vmax*u/(km+u) + c_loc*np.sqrt(u) - k_sink*(s-1.0)*u
    res = least_squares(lambda p: pred(p)-y,
                        x0=np.array([5.0, 0.8, 0.5, 0.2]),
                        bounds=([0.0, 1e-6, 0.0, -10.0], [100.0, 100.0, 20.0, 10.0]))
    return res.x

def predict(row):
    # Parameters are stored by fit on the fitted function object below.
    p = predict.params
    u = max(1e-8, float(row['u_now']))
    return float(p[0]*u/(p[1]+u) + p[2]*np.sqrt(u) - p[3]*(float(row['s_grad'])-1.0)*u)

# Cache fitted parameters for the required predict(row) interface.
_fit_impl = fit
def fit(train_rows):
    p = _fit_impl(train_rows)
    predict.params = p
    return p