import numpy as np
from scipy.optimize import least_squares

_model = None

def _features(rows):
    return np.array([[float(r['u_outdoor']), float(r['u_indoor']), float(r['p_weather']), float(r['a_foot']), float(r['impact_idx'])] for r in rows], dtype=float)

def _envelope(a, c, kind):
    if kind == 0:
        return 1.0 / (1.0 + c * a)
    if kind == 1:
        return 1.0 / (1.0 + c * a * a)
    if kind == 2:
        return np.exp(-c * a)
    if kind == 3:
        return np.maximum(0.0, 1.0 - c * a)
    if kind == 4:
        return 1.0 / (1.0 + c * np.maximum(a - 1.0, 0.0))
    return np.exp(-c * np.maximum(a - 1.0, 0.0) ** 2)

def _loss(a, impact, kind):
    if kind == 0:
        return impact
    if kind == 1:
        return impact * a
    if kind == 2:
        return impact * a * a
    if kind == 3:
        return impact * impact
    return impact * impact * a

def _evaluate(x, params, envelope_kind, loss_kind):
    g, k, indoor_scale, c, loss_scale = params
    outdoor, indoor, weather, area, impact = x.T
    dose = outdoor * weather + indoor_scale * indoor * _envelope(area, c, envelope_kind)
    return g * dose / (k + dose) - loss_scale * _loss(area, impact, loss_kind)

def fit(train_rows):
    global _model
    rows = list(train_rows)
    x = _features(rows)
    y = np.array([float(r['y']) for r in rows])
    best = None
    for e in range(6):
        for l in range(5):
            for c0 in (0.2, 1.0, 3.0):
                result = least_squares(lambda p: _evaluate(x, p, e, l) - y, [8.0, 1.0, 1.0, c0, 2.0], bounds=([0.0, 1e-6, 1e-6, 0.0, 0.0], [100.0, 100.0, 100.0, 20.0, 100.0]), max_nfev=3000)
                score = float(np.mean(result.fun ** 2))
                if best is None or score < best[0]:
                    best = (score, result.x, e, l)
    _model = best[1:]

def predict(row):
    if _model is None:
        raise RuntimeError('fit must be called before predict')
    params, envelope_kind, loss_kind = _model
    return float(_evaluate(_features([row]), params, envelope_kind, loss_kind)[0])