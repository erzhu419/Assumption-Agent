import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = [r for r in train_rows if isinstance(r, dict) and all(k in r for k in ('u_raw','t_proc','m_2','y'))]
    if not rows:
        raise ValueError('No usable training rows')
    def pred(p, r):
        vmax, km, c_raw, c_thermal = p
        m = max(0.0, float(r['m_2']))
        u = max(0.0, float(r['u_raw']))
        t = float(r['t_proc'])
        return vmax*m/(km+m) + c_raw*math.sqrt(u) + c_thermal*t
    y = np.array([float(r['y']) for r in rows])
    p0 = np.array([6.0, 1.5, 0.2, 0.0])
    res = least_squares(lambda p: np.array([pred(p, r) for r in rows]) - y,
                        p0, bounds=([0.001,0.001,-10.0,-10.0],[100.0,100.0,10.0,10.0]))
    return tuple(float(v) for v in res.x)

def predict(row):
    vmax, km, c_raw, c_thermal = fit([row])
    return vmax*max(0.0,float(row['m_2']))/(km+max(0.0,float(row['m_2']))) + c_raw*math.sqrt(max(0.0,float(row['u_raw']))) + c_thermal*float(row['t_proc'])