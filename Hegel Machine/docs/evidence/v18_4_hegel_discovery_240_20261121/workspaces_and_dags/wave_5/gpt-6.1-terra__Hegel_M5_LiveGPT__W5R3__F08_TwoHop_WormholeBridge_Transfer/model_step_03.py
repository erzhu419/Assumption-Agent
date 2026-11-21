import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _values(p, u, s, kind):
    if kind == 0:
        a, b, c, d, e = p
        return a + b*u/(c+u) + d*u/((c+u)*(e+s))
    if kind == 1:
        a, b, c, d, e = p
        return a + b*u/(c+u) + d/(e+s)
    a, b, c, d, e, f = p
    return a + b*u/(c+u) + d/(e+s) + f*u/((c+u)*(e+s))

def fit(train_rows):
    global _model, _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for kind in range(3):
        n = 5 if kind < 2 else 6
        lower = np.full(n, -100.0)
        upper = np.full(n, 100.0)
        lower[2] = 0.001
        lower[4] = 0.001
        for c in (0.3, 1.5, 5.0):
            for e in (0.3, 1.5, 5.0):
                p0 = np.array([0.0, 6.0, c, 2.0, e] + ([1.0] if n == 6 else []))
                res = least_squares(lambda p: _values(p, u, s, kind)-y, p0, bounds=(lower, upper), max_nfev=3000)
                mse = np.mean(res.fun**2)
                score = mse + (n-5)*1e-9
                if best is None or score < best[0]:
                    best = (score, kind, res.x)
    _, _model, _params = best

def predict(row):
    return float(_values(_params, float(row['u_now']), float(row['s_grad']), _model))
