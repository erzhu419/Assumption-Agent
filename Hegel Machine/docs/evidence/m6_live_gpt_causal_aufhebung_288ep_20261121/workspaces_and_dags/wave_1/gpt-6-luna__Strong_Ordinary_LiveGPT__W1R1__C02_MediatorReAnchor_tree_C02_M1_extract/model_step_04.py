import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = [r for r in train_rows if all(k in r for k in ('u_raw','t_proc','m_1','m_2','m_3','y'))]
    if not rows:
        raise ValueError('No usable training rows')
    # Saturating response anchored on the mediator singled out by the clamp data,
    # with raw-input and processing corrections.
    def pred(p, r):
        vmax, km, c_raw, c_t, c_m1, c_m2, c_m3 = p
        x = max(0.0, float(r['m_2']))
        sat = vmax * x / (km + x)
        return sat + c_raw * math.sqrt(max(0.0, float(r['u_raw']))) + c_t * float(r['t_proc']) + c_m1 * float(r['m_1']) + c_m2 * float(r['m_2']) + c_m3 * float(r['m_3'])
    p0 = np.array([6.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    lo = [0.001, 0.001, -20, -20, -20, -20, -20]
    hi = [100, 100, 20, 20, 20, 20, 20]
    res = least_squares(lambda p: np.array([pred(p,r)-float(r['y']) for r in rows]), p0, bounds=(lo,hi), max_nfev=20000)
    return res.x

def predict(row):
    p = getattr(predict, '_params', None)
    if p is None:
        raise ValueError('Call fit(train_rows) before predict')
    vmax, km, c_raw, c_t, c_m1, c_m2, c_m3 = p
    x = max(0.0, float(row.get('m_2', 0.0)))
    return float(vmax*x/(km+x) + c_raw*math.sqrt(max(0.0,float(row.get('u_raw',0.0)))) + c_t*float(row.get('t_proc',0.0)) + c_m1*float(row.get('m_1',0.0)) + c_m2*float(row.get('m_2',0.0)) + c_m3*float(row.get('m_3',0.0)))

# Store fitted coefficients for the required predict(row) interface.
_old_fit = fit
def fit(train_rows):
    p = _old_fit(train_rows)
    predict._params = p
    return p
