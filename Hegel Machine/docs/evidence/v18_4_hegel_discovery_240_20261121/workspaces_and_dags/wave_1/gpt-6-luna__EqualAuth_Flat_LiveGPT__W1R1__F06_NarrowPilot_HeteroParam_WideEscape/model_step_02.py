import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([max(1e-4, float(r['u_now'])) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        vmax, km, k = p
        return vmax * u / (km + u) * np.exp(-k * s)
    res = least_squares(lambda p: pred(p) - y, x0=[8.0, 1.2, 0.5], bounds=([0.0, 1e-5, -5.0], [100.0, 100.0, 5.0]))
    return res.x.tolist()

def predict(row):
    vmax, km, k = fit(_TRAIN_ROWS)
    u = max(1e-4, float(row['u_now']))
    s = float(row['s_grad'])
    return float(vmax * u / (km + u) * np.exp(-k * s))