import math
import numpy as np
from scipy.optimize import least_squares

_MODEL = None

def _basis(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    sat = u / (1.2 + u)
    sink = s / (1.0 + s)
    return u, sat, sink

def fit(train_rows):
    global _MODEL
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    # Compare the source-note alternatives for the additive branch term.
    branch_forms = {
        'plus_sqrt': lambda u, c: c * math.sqrt(max(u, 0.0)),
        'minus_sqrt': lambda u, c: -c * math.sqrt(max(u, 0.0)),
        'linear_leak': lambda u, c: -c * u,
    }
    best = None
    for name, branch in branch_forms.items():
        def residual(p):
            vmax, rho, c = p
            pred = []
            for r in rows:
                u, sat, sink = _basis(r)
                pred.append(vmax * (1.0 - rho * sink) * sat + branch(u, c))
            return np.asarray(pred) - y
        res = least_squares(residual, x0=[3.0, 0.7, 0.2], bounds=([0.0, 0.0, 0.0], [20.0, 2.0, 20.0]))
        score = float(np.mean(res.fun ** 2))
        if best is None or score < best[0]:
            best = (score, name, res.x)
    _MODEL = {'branch': best[1], 'params': np.asarray(best[2], dtype=float)}
    return _MODEL

def predict(row):
    if _MODEL is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, rho, c = _MODEL['params']
    u, sat, sink = _basis(row)
    base = vmax * (1.0 - rho * sink) * sat
    if _MODEL['branch'] == 'plus_sqrt':
        return float(base + c * math.sqrt(max(u, 0.0)))
    if _MODEL['branch'] == 'minus_sqrt':
        return float(base - c * math.sqrt(max(u, 0.0)))
    return float(base - c * u)
