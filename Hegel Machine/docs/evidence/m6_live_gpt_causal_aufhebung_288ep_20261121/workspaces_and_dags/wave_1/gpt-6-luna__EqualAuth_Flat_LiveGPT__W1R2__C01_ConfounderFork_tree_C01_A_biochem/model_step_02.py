import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    global _model
    rows = list(train_rows)
    names = ['z_1', 'z_2', 'z_3', 'z_4']
    best = None
    # Fit candidate single-input tree splits; threshold search uses observed values.
    for name in names:
        vals = sorted(set(float(r.get(name, 0.0)) for r in rows))
        thresholds = [(a+b)/2 for a,b in zip(vals[:-1], vals[1:])]
        if not thresholds:
            thresholds = [vals[0]]
        for threshold in thresholds:
            gates = np.array([float(float(r.get(name, 0.0)) > threshold) for r in rows])
            x = np.array([float(r.get('x_ind', 0.0)) for r in rows])
            u = np.array([float(r.get('u_ctrl', 0.0)) for r in rows])
            y = np.array([float(r['y']) for r in rows])
            def residual(p):
                vmax, k, c, d = p
                q = (x / 1.5) ** 2
                pred = vmax*q/(k+q) - c*u + d*gates
                return pred-y
            try:
                res = least_squares(residual, [8., 1., .3, 1.], bounds=([0., .01, -5., -20.], [30., 30., 5., 20.]))
                score = float(np.mean(res.fun**2))
                if best is None or score < best[0]:
                    best = (score, name, threshold, res.x)
            except Exception:
                pass
    if best is None:
        _model = ('z_1', 0., np.array([8., 1., .3, 0.]))
    else:
        _model = (best[1], best[2], best[3])
    return _model

def predict(row):
    name, threshold, p = _model
    vmax, k, c, d = p
    x = float(row.get('x_ind', 0.0))
    u = float(row.get('u_ctrl', 0.0))
    q = (x / 1.5) ** 2
    gate = float(float(row.get(name, 0.0)) > threshold)
    return float(vmax*q/(k+q) - c*u + d*gate)