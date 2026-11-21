import numpy as np
from scipy.optimize import least_squares

_model = None

def _design(rows, logtaus, family):
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    h = np.array([float(r['H']) for r in rows])
    j = u / (1.2 + u)
    q = j * s / (1.0 + s)
    t1, t2 = np.exp(logtaus)
    f = -np.expm1(-h / t1)
    g = -np.expm1(-h / t2)
    if family == 0:
        cols = [j, q, u*f, u*g]
    elif family == 1:
        cols = [j, q, j*f, j*g]
    elif family == 2:
        cols = [j, q, j*f, u*g]
    elif family == 3:
        cols = [j, q, u*f, j*g]
    elif family == 4:
        cols = [j, q, j*f, q*f, u*g]
    else:
        cols = [j, q, u*f, j*g, q*g]
    return np.column_stack(cols)

def fit(train_rows):
    global _model
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    best = None
    for family in range(6):
        def residual(p):
            X = _design(rows, p, family)
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            return X @ coef - y
        for t1 in (0.1, 0.8, 3.0):
            for t2 in (5.0, 15.0, 40.0):
                res = least_squares(residual, np.log([t1, t2]), bounds=(np.log([0.02, 0.1]), np.log([30.0, 200.0])), max_nfev=1500)
                X = _design(rows, res.x, family)
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                mse = np.mean((X @ coef - y)**2)
                # Prefer a simpler coupling when its fit is essentially as good.
                score = mse + (0.00002 if family >= 4 else 0.0)
                if best is None or score < best[0]:
                    best = (score, family, res.x.copy(), coef.copy())
    _model = best

def predict(row):
    if _model is None:
        raise RuntimeError('fit must be called before predict')
    _, family, logtaus, coef = _model
    return float(_design([row], logtaus, family)[0] @ coef)
