import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_raw']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    root_u = np.sqrt(np.maximum(u, 0.0))
    best = None
    for key in ('m_1', 'm_2', 'm_3'):
        m = np.array([float(r[key]) for r in rows])
        def residual(p):
            vmax, km, c = p
            return vmax * m / (km + m) + c * root_u - y
        result = least_squares(residual, x0=[8.0, 1.0, 0.5], bounds=([0.0, 1e-8, -10.0], [100.0, 100.0, 10.0]))
        rss = float(np.sum(result.fun ** 2))
        if best is None or rss < best[0]:
            best = (rss, key, result.x)
    _, key, params = best
    return {'mediator': key, 'params': params.tolist()}

def predict(row):
    key = _MODEL['mediator']
    vmax, km, c = _MODEL['params']
    m = float(row[key])
    return float(vmax * m / (km + m) + c * math.sqrt(max(float(row['u_raw']), 0.0)))

_MODEL = {'mediator': 'm_2', 'params': [8.0, 1.0, 0.5]}
_original_fit = fit
def fit(train_rows):
    global _MODEL
    _MODEL = _original_fit(train_rows)
    return _MODEL