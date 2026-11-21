import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    base = (u / (1.2 + u))
    g = s / (1.0 + s)
    root = np.sqrt(np.maximum(u, 0.0))
    best = None
    for sign in (-1.0, 1.0):
        def pred(p):
            vmax, rho, c_loc = p
            return vmax * (1.0 - rho * g) * base + sign * c_loc * root
        res = least_squares(lambda p: pred(p) - y, x0=[5.0, 0.7, 0.1], bounds=([0.0, 0.0, 0.0], [20.0, 1.0, 10.0]))
        err = float(np.mean((pred(res.x) - y) ** 2))
        if best is None or err < best[0]:
            best = (err, sign, res.x)
    _, sign, params = best
    return {'sign': sign, 'params': params}

def predict(row):
    sign = fit_state['sign']
    vmax, rho, c_loc = fit_state['params']
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u_n = u
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) + sign * c_loc * np.sqrt(max(u_n, 0.0)))

fit_state = {'sign': -1.0, 'params': np.array([5.0, 0.7, 0.1])}
_original_fit = fit
def fit(train_rows):
    global fit_state
    fit_state = _original_fit(train_rows)
    return fit_state