import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_raw']) for r in train_rows], dtype=float)
    m = np.array([float(r.get('m_2', r['s_surf'])) for r in train_rows], dtype=float)
    t = np.array([float(r.get('t_proc', 0.0)) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # The pilot is entirely cold; consequently the thermal coefficient is not
    # identifiable and is fixed at zero rather than estimated from no variation.
    def pred(p):
        vmax, km, c = p
        return vmax*m/(km+m) + c*np.sqrt(np.maximum(u, 0.0))
    res = least_squares(lambda p: pred(p)-y, x0=[5.0, 1.0, 1.0], bounds=([0.0, 1e-8, -10.0], [30.0, 100.0, 10.0]))
    _params = tuple(float(v) for v in res.x)
    return predict

def predict(row):
    vmax, km, c = _params
    m = float(row.get('m_2', row['s_surf']))
    u = max(0.0, float(row['u_raw']))
    return float(vmax*m/(km+m) + c*math.sqrt(u))