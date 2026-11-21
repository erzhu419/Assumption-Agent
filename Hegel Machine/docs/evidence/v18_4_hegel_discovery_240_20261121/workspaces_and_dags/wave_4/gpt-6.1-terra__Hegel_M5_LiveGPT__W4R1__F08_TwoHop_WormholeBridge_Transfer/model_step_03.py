import numpy as np
from scipy.optimize import least_squares

_params = None
_kind = None

def _eval(p, u, s, kind):
    if kind == 0:
        a, b, c, d, e = p
        return (a*u/(1+b*u) + c*u/(1+d*u))/(1+e*s)
    if kind == 1:
        a, b, c, d, e, f = p
        return a*u/(1+b*u+c*s) + d*u/(1+e*u+f*s)
    a, b, c, d, e = p
    return u/(1+b*u) * (a/(1+c*s) + d/(1+e*s))

def fit(train_rows):
    global _params, _kind
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    starts = {
        0: [[3, 0.5, 1, 0.1, 0.5], [5, 1, 1, 0.5, 1]],
        1: [[3, 0.5, 0.5, 1, 0.1, 0.5], [1, 0.1, 1, 4, 1, 0.1], [4, 1, 1, 2, 0.2, 0.2]],
        2: [[3, 0.5, 0.5, 1, 1], [1, 0.5, 0.1, 4, 1]]
    }
    best = None
    for kind, guesses in starts.items():
        for guess in guesses:
            result = least_squares(lambda p: _eval(p, u, s, kind)-y, guess, bounds=(0, np.inf), max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            loss = float(np.mean(result.fun**2))
            if best is None or loss < best[0]:
                best = (loss, kind, result.x)
    _, _kind, _params = best

def predict(row):
    return float(_eval(_params, float(row['u_now']), float(row['s_grad']), _kind))