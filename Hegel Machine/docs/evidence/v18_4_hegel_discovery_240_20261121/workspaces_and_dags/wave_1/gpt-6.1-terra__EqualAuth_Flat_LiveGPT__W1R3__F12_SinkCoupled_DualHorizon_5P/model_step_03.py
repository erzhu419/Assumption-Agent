import numpy as np
from scipy.optimize import least_squares

_model = None

def _evaluate(p, row, variant):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    x = float(row['x_now'])
    base, fast, slow, tf, ts, sink = p
    response = base + fast * (-np.expm1(-h / tf)) - slow * (-np.expm1(-h / ts))
    if variant == 0:
        return x + u * (response - sink * s)
    if variant == 1:
        return x + u * response - sink * s
    if variant == 2:
        return x + u * response / (1.0 + sink * s)
    return x + u * (response - sink * s * (-np.expm1(-h / tf)))

def fit(train_rows):
    global _model
    y = np.array([float(r['y']) for r in train_rows])
    best = None
    for variant in range(4):
        def residual(p):
            return np.array([_evaluate(p, r, variant) for r in train_rows]) - y
        for tf in (0.3, 1.0, 3.0):
            for ts in (5.0, 15.0, 40.0):
                start = [1.0, 1.5, 3.0, tf, ts, 0.3]
                result = least_squares(residual, start, bounds=([-5, -10, -10, 0.03, 1.0, 0], [5, 10, 10, 8.0, 100.0, 5]), max_nfev=3000)
                score = np.mean(residual(result.x) ** 2)
                if best is None or score < best[0]:
                    best = (score, result.x, variant)
    _model = best[1:]

def predict(row):
    return float(_evaluate(_model[0], row, _model[1]))