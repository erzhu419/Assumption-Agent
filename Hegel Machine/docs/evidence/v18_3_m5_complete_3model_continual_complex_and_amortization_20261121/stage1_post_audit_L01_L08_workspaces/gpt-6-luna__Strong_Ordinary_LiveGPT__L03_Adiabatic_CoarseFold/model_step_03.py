import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([max(float(r['u_now']), 1e-12) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    sat = u / (1.2 + u)
    grad = s / (1.0 + s)
    best = None
    # Compare the source-note's open sqrt branch signs against its linear-leak alternative.
    for kind in ('plus_sqrt', 'minus_sqrt', 'leak'):
        def pred(p):
            vmax, rho, c = p
            base = vmax * (1.0 - rho * grad) * sat
            if kind == 'plus_sqrt':
                return base + c * np.sqrt(u)
            if kind == 'minus_sqrt':
                return base - c * np.sqrt(u)
            return base - c * u
        res = least_squares(lambda p: pred(p) - y, x0=[10.0, 0.7, 0.1], bounds=([0.0, 0.0, 0.0], [100.0, 1.0, 100.0]))
        err = float(np.sum((pred(res.x) - y) ** 2))
        if best is None or err < best[0]:
            best = (err, kind, res.x)
    _, kind, p = best
    return {'kind': kind, 'params': p.tolist()}

def predict(row):
    # fit() stores parameters; prediction uses the fitted state established by fit().
    u = max(float(row['u_now']), 1e-12)
    s = float(row['s_grad'])
    vmax, rho, c = _FIT['params']
    out = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    if _FIT['kind'] == 'plus_sqrt':
        return float(out + c * np.sqrt(u))
    if _FIT['kind'] == 'minus_sqrt':
        return float(out - c * np.sqrt(u))
    return float(out - c * u)
