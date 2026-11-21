import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # Thermal rows identify a linear contribution from independently clamped mediators;
    # cold rows identify the surface-saturation plus sqrt(raw-input) baseline.
    cold = [r for r in train_rows if abs(float(r.get('t_proc', 0.0))) < 1e-12]
    warm = [r for r in train_rows if abs(float(r.get('t_proc', 0.0))) >= 1e-12]
    def base(p, r):
        vmax, km, craw = p
        s = float(r['s_surf']); u = float(r['u_raw'])
        return vmax*s/(km+s) + craw*math.sqrt(max(0.0,u))
    p0 = [7.0, 1.0, 0.5]
    if cold:
        res = least_squares(lambda p: np.array([base(p,r)-float(r['y']) for r in cold]), p0, bounds=([0.001,0.001,-10],[100,100,10]))
        p0 = list(res.x)
    # Fit mediator effects as deviations from cold-process law; allow process offset.
    if warm:
        X=[]; y=[]
        for r in warm:
            X.append([float(r.get('t_proc',0)), float(r.get('m_1',0)), float(r.get('m_2',0)), float(r.get('m_3',0))])
            y.append(float(r['y'])-base(p0,r))
        beta=np.linalg.lstsq(np.asarray(X),np.asarray(y),rcond=None)[0]
    else: beta=np.zeros(4)
    return (p0, beta)

def predict(row):
    p,b=fit([row])
    return float(p[0]*float(row['s_surf'])/(p[1]+float(row['s_surf']))+p[2]*math.sqrt(max(0.0,float(row['u_raw']))) + b[0]*float(row.get('t_proc',0)) + b[1]*float(row.get('m_1',0))+b[2]*float(row.get('m_2',0))+b[3]*float(row.get('m_3',0)))