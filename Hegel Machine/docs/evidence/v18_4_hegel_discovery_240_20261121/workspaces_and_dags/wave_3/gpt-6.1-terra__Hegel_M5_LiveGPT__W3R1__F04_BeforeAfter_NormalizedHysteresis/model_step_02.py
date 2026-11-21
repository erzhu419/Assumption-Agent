import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def residual(p):
        out = []
        for r in train_rows:
            h = float(r['H'])
            gain = p[0] + p[1]*h + p[2]*h*h + p[3]*h*h*h
            pred = float(r['x_now']) + (float(r['u_past']) - float(r['u_now']))*gain
            out.append(pred - float(r['y']))
        return np.asarray(out)
    _params = least_squares(residual, [80.0, -60.0, 20.0, -5.0]).x

def predict(row):
    h = float(row['H'])
    gain = _params[0] + _params[1]*h + _params[2]*h*h + _params[3]*h*h*h
    return float(row['x_now']) + (float(row['u_past']) - float(row['u_now']))*gain
