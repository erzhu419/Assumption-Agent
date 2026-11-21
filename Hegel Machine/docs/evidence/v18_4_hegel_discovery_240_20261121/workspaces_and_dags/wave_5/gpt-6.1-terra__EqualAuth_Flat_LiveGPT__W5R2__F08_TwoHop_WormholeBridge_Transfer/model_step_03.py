import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _value(p, r, kind):
    u = float(r['u_now'])
    s = float(r['s_grad'])
    if kind == 0:
        return (p[0] + p[1]*u) / (1.0 + p[2]*u + p[3]*s + p[4]*u*s)
    if kind == 1:
        return (p[0] + p[1]*np.log1p(p[2]*u)) / (1.0 + p[3]*s)
    return (p[0] + p[1]*u + p[2]*u*u) / (1.0 + p[3]*u + p[4]*u*u + p[5]*s + p[6]*u*s)

def fit(train_rows):
    global _model, _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    candidates = []
    for kind, initial, lower, upper in [
        (0, [0.0, 4.0, 0.3, 1.0, 0.0], [-10, 0, 0, 0, 0], [10, 100, 100, 100, 100]),
        (1, [0.0, 3.0, 1.0, 0.3], [-10, 0, 0, 0], [10, 100, 100, 100]),
        (2, [0.0, 4.0, 0.1, 0.3, 0.01, 1.0, 0.1], [-10, 0, 0, 0, 0, 0, 0], [10, 100, 100, 100, 100, 100, 100])
    ]:
        def residual(p, indices):
            return np.array([_value(p, rows[i], kind) - y[i] for i in indices])
        try:
            p = least_squares(residual, initial, args=(range(len(rows)),), bounds=(lower, upper), max_nfev=5000).x
            errors = []
            for i in range(len(rows)):
                ids = [j for j in range(len(rows)) if j != i]
                q = least_squares(residual, p, args=(ids,), bounds=(lower, upper), max_nfev=2000).x
                errors.append(_value(q, rows[i], kind) - y[i])
            candidates.append((float(np.mean(np.square(errors))), kind, p))
        except (ValueError, FloatingPointError):
            continue
    _, _model, _params = min(candidates, key=lambda z: z[0])

def predict(row):
    return float(_value(_params, row, _model))