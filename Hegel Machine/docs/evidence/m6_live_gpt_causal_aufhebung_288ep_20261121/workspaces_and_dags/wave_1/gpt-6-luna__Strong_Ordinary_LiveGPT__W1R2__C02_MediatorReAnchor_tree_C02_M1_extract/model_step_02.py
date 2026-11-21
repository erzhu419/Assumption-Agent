import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_raw']) for r in rows])
    s = np.array([float(r['s_surf']) for r in rows])
    t = np.array([float(r['t_proc']) for r in rows])
    m = np.array([[float(r['m_1']), float(r['m_2']), float(r['m_3'])] for r in rows])
    y = np.array([float(r['y']) for r in rows])
    # Cold-process response is a saturating surface term plus raw-input contribution;
    # thermal processing permits separate mediator effects, regularized toward zero.
    def pred(p):
        vmax, km, c_raw, b1, b2, b3, bt = p
        base = vmax * s / (km + s) + c_raw * np.sqrt(np.maximum(u, 0.0))
        return base + t * (b1*m[:,0] + b2*m[:,1] + b3*m[:,2] + bt)
    def residual(p):
        return np.r_[pred(p)-y, 0.03*p[3:]]
    result = least_squares(residual, x0=np.array([8.0, 1.0, 0.5, 0., 0., 0., 0.]), max_nfev=5000)
    return result.x

def predict(row):
    p = fit(_train_rows)
    u = float(row['u_raw']); s = float(row['s_surf']); t = float(row['t_proc'])
    return float(p[0]*s/(p[1]+s) + p[2]*math.sqrt(max(0.0, u)) + t*(p[3]*float(row['m_1']) + p[4]*float(row['m_2']) + p[5]*float(row['m_3']) + p[6]))