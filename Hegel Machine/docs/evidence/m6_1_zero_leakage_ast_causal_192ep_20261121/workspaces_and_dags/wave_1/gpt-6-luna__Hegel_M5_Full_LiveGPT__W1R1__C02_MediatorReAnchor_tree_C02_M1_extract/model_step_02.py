import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    X = np.array([[float(r['u_raw']), float(r['t_proc']), float(r['s_surf']),
                   float(r['m_1']), float(r['m_2']), float(r['m_3'])] for r in rows])
    y = np.array([float(r['y']) for r in rows])
    # Parameters: vmax, km, c_raw, and additive effects of t_proc, m_1, m_2, m_3.
    def pred(p, x):
        u, t, s, m1, m2, m3 = x.T
        return p[0]*s/(p[1]+s) + p[2]*np.sqrt(np.maximum(u, 0.0)) + p[3]*t + p[4]*m1 + p[5]*m2 + p[6]*m3
    p0 = np.array([8.0, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    result = least_squares(lambda p: pred(p, X)-y, p0,
                           bounds=([0.0, 1e-6, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
                                   [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
                           max_nfev=20000)
    return result.x

def predict(row):
    # fit() stores parameters on the function for callers that use the usual fit/predict interface.
    p = predict._params
    u, t, s = float(row['u_raw']), float(row['t_proc']), float(row['s_surf'])
    return float(p[0]*s/(p[1]+s) + p[2]*np.sqrt(max(u, 0.0)) + p[3]*t
                 + p[4]*float(row['m_1']) + p[5]*float(row['m_2']) + p[6]*float(row['m_3']))
