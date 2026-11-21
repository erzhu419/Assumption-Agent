import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    dose = float(row['u_indoor']) + float(row['p_weather']) * float(row['u_outdoor'])
    impact = float(row['impact_idx'])
    area = float(row['a_foot'])
    return max(dose, 0.0), impact, area

def fit(train_rows):
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    xs = [_features(r) for r in train_rows]
    # Saturating effective dose, attenuated by impact with a foot-area interaction.
    def residual(p):
        amp, half, b, c = p
        pred = np.asarray([amp*d/(half+d+1e-12)*math.exp(-b*i-c*i*a) for d,i,a in xs])
        return pred - ys
    opt = least_squares(residual, x0=[8.0, 1.0, 0.5, 0.5], bounds=([0.0, 1e-5, 0.0, 0.0], [30.0, 30.0, 10.0, 10.0]))
    return opt.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    d, i, a = _features(row)
    amp, half, b, c = p
    return float(amp*d/(half+d+1e-12)*math.exp(-b*i-c*i*a))