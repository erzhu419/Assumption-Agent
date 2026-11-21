import numpy as np
from scipy.optimize import least_squares

_model = None

def _estimate(p, X, envelope, loss):
    outdoor, indoor, weather, area, impact = X.T
    gain, half, space, penalty = p
    if envelope == 0:
        factor = 1.0 / (1.0 + space * area)
    elif envelope == 1:
        factor = 1.0 / (1.0 + space * area ** 2)
    elif envelope == 2:
        factor = np.exp(-space * area)
    else:
        factor = np.maximum(0.0, 1.0 - space * area)
    dose = outdoor * weather + indoor * factor
    if loss == 0:
        damage = penalty * impact
    elif loss == 1:
        damage = penalty * impact * area
    elif loss == 2:
        damage = penalty * impact * area ** 2
    else:
        damage = penalty * impact ** 2 * area
    return gain * dose / (half + dose + 1e-12) - damage

def fit(train_rows):
    global _model
    keys = ('u_outdoor', 'u_indoor', 'p_weather', 'a_foot', 'impact_idx')
    X = np.array([[float(r[k]) for k in keys] for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    best = None
    for envelope in range(4):
        for loss in range(4):
            for space0 in (0.1, 0.5, 1.0):
                result = least_squares(
                    lambda p: _estimate(p, X, envelope, loss) - y,
                    x0=[7.0, 0.8, space0, 2.0],
                    bounds=([0.0, 1e-6, 0.0, 0.0], [100.0, 100.0, 20.0, 100.0]),
                    max_nfev=3000
                )
                error = float(np.sum(result.fun ** 2))
                if best is None or error < best[0]:
                    best = (error, result.x, envelope, loss)
    _model = best[1:]
    return _model

def predict(row):
    keys = ('u_outdoor', 'u_indoor', 'p_weather', 'a_foot', 'impact_idx')
    X = np.array([[float(row[k]) for k in keys]])
    p, envelope, loss = _model
    return float(_estimate(p, X, envelope, loss)[0])