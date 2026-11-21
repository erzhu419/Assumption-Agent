import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_raw']) for r in rows])
    s = np.array([float(r['s_surf']) for r in rows])
    t = np.array([float(r['t_proc']) for r in rows])
    m1 = np.array([float(r['m_1']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p):
        vmax, km, c_raw, a_thermal, b_m1 = p
        return vmax*s/(km+s) + c_raw*np.sqrt(np.maximum(u, 0.0)) + t*(a_thermal + b_m1*m1)
    p0 = np.array([8.0, 1.0, 0.5, 0.0, -0.5])
    res = least_squares(lambda p: pred(p)-y, p0, bounds=([-100, 1e-6, -100, -100, -100], [100, 100, 100, 100, 100]), max_nfev=20000)
    return res.x

def predict(row):
    vmax, km, c_raw, a_thermal, b_m1 = _PARAMS
    u = float(row['u_raw']); s = float(row['s_surf']); t = float(row['t_proc']); m1 = float(row['m_1'])
    return float(vmax*s/(km+s) + c_raw*math.sqrt(max(u, 0.0)) + t*(a_thermal + b_m1*m1))