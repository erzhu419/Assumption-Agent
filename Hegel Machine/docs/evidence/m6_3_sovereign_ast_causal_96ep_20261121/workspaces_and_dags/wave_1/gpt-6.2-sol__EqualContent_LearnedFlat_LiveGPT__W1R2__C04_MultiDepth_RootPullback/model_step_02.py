import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_outdoor', 'u_indoor', 'p_weather', 'a_foot', 'impact_idx')

def _eval(x, spec, par, ref, base):
    outdoor, u, weather, area, impact = x.T
    response, gate, up, qp, placement = spec
    b, c, threshold, sharp, penalty = par[:5]
    h = np.maximum(area - threshold, 0.0)
    if gate == 'hinge':
        indoor = c * u * np.maximum(1.0 - sharp * h, 0.0)
    elif gate == 'exp':
        indoor = c * u * np.exp(-sharp * h)
    elif gate == 'sigmoid':
        indoor = c * u / (1.0 + np.exp(np.clip(sharp * (area - threshold), -60.0, 60.0)))
    else:
        indoor = np.maximum(c * u - sharp * h, 0.0)
    cost = penalty * np.power(u, up) * np.power(impact, qp)
    if placement == 'area_loss':
        cost *= 1.0 + par[5] * h
    dose = outdoor * weather + indoor
    if placement == 'dose_loss':
        dose = outdoor * weather + np.maximum(indoor - cost, 0.0)
    def gain(z):
        if response == 'exp':
            return -np.expm1(-b * z)
        if response == 'log':
            return np.log1p(b * z)
        if response == 'rational':
            return z / (1.0 + b * z)
        return np.tanh(b * z)
    value = base * gain(dose) / gain(ref)
    return value if placement == 'dose_loss' else value - cost

def fit(train_rows):
    x = np.array([[float(row[key]) for key in _KEYS] for row in train_rows])
    y = np.array([float(row['y']) for row in train_rows])
    anchor = (x[:, 1] == 0) & (x[:, 4] == 0)
    ref = float(np.mean(x[anchor, 0] * x[anchor, 2]))
    base = float(np.mean(y[anchor]))
    low = np.array([0.03, 0.05, 0.70, 0.05, 0.0])
    high = np.array([4.0, 4.0, 2.15, 22.0, 20.0])
    candidates = []
    def solve(spec, start, extra=False):
        lo = np.r_[low, 0.0] if extra else low
        hi = np.r_[high, 12.0] if extra else high
        def resid(par):
            return _eval(x, spec, par, ref, base) - y
        res = least_squares(resid, np.clip(start, lo + 1e-8, hi - 1e-8), bounds=(lo, hi), max_nfev=160)
        rmse = math.sqrt(float(np.mean(res.fun ** 2)))
        complexity = 1.0 * extra + 0.35 * (spec[3] == 3.0) + 0.20 * (spec[4] == 'dose_loss')
        score = math.log(max(rmse, 1.5e-7)) + complexity
        candidates.append((score, rmse, spec, res.x))
    for response in ('exp', 'log', 'rational', 'tanh'):
        for gate in ('hinge', 'exp', 'sigmoid', 'dose_cap'):
            for up, qp in ((0., 1.), (0., 2.), (1., 1.), (1., 2.), (1., 3.), (2., 2.)):
                start = [0.8, 1.0, 1.25 if gate != 'sigmoid' else 1.65, 1.4 if gate != 'sigmoid' else 6., 1.5]
                solve((response, gate, up, qp, 'output_loss'), start)
    ranked = sorted(candidates, key=lambda item: item[0])
    if ranked[0][1] > 1e-6:
        for _, _, spec, par in ranked[:8]:
            response, gate, up, qp, _ = spec
            solve((response, gate, up, qp, 'area_loss'), np.r_[par, 0.2], True)
            solve((response, gate, up, qp, 'dose_loss'), np.r_[par[:4], min(1.5, par[4])])
    _, _, spec, par = min(candidates, key=lambda item: item[0])
    def predict(row) -> float:
        values = np.array([[float(row[key]) for key in _KEYS]])
        return float(_eval(values, spec, par, ref, base)[0])
    return predict
