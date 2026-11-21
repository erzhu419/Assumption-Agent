import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _features(rows):
    return np.array([[r['u_past'], r['u_now'], r['x_now'], r['mem_conv'], r['s_grad'], r['H']] for r in rows], dtype=float)

def _evaluate(p, X, model):
    a, b, c, d, rate = p
    past, u, x, mem, s, h = X.T
    usat, ssat, sink_x, mem_target = model
    source = u / (1.0 + u) if usat else u
    sink = s / (1.0 + s) if ssat else s
    target = past if mem_target else u
    decay = b + c * sink if sink_x else np.full_like(u, b)
    constant = a * source + d * target - (0.0 if sink_x else c * sink)
    z = decay * h
    integ = np.where(np.abs(decay) < 1e-7, h * (1.0 - z / 2.0), -np.expm1(-z) / np.where(np.abs(decay) < 1e-7, 1.0, decay))
    diff = decay - rate
    zd = diff * h
    transient_int = np.where(np.abs(diff) < 1e-7, h * (1.0 + zd / 2.0), np.expm1(zd) / np.where(np.abs(diff) < 1e-7, 1.0, diff))
    return x * np.exp(-z) + constant * integ + d * (mem - target) * np.exp(-z) * transient_int

def fit(train_rows):
    global _model, _params
    X = _features(train_rows)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for usat in (False, True):
        for ssat in (False, True):
            for sink_x in (False, True):
                for mem_target in (False, True):
                    model = (usat, ssat, sink_x, mem_target)
                    for start_rate in (0.4, 2.0):
                        result = least_squares(lambda p: _evaluate(p, X, model) - y,
                            [1.0, 0.5, 0.3, 0.3, start_rate],
                            bounds=([0.0, 0.0, 0.0, -5.0, 0.01], [10.0, 10.0, 10.0, 5.0, 10.0]),
                            max_nfev=1200)
                        loss = float(np.mean(result.fun ** 2))
                        if best is None or loss < best[0]:
                            best = (loss, model, result.x)
    _, _model, _params = best

def predict(row):
    if _params is None:
        raise RuntimeError('fit must be called before predict')
    return float(_evaluate(_params, _features([row]), _model)[0])