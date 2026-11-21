import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # Saturating response in current input with an additive gradient-dependent
    # bridge term; all four coefficients are fitted numerically.
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    g = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p, uu, gg):
        a, b, c, d = p
        return a * uu / (b + uu) + c * gg / (d + gg)
    p0 = np.array([3.0, 1.0, 1.0, 1.0])
    res = least_squares(lambda p: pred(p, u, g) - y, p0, max_nfev=20000)
    return res.x

def predict(row):
    p = fit([row]) if False else None
    # Parameters are populated by fit and cached on the function.
    a, b, c, d = predict.params
    u, g = float(row['u_now']), float(row['s_grad'])
    return float(a * u / (b + u) + c * g / (d + g))

_original_fit = fit
def fit(train_rows):
    params = _original_fit(train_rows)
    predict.params = params
    return params
