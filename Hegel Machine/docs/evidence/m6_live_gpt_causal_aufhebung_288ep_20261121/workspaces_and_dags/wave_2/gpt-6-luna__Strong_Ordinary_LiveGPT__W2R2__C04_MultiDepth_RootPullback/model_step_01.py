import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # The supplied L2 mechanisms fix the functional form. Fit only root gain g_root
    # and impact coefficient c_imp; indoor envelope parameters are specified constants.
    def features(r):
        uo=float(r['u_outdoor']); ui=float(r['u_indoor']); p=float(r['p_weather'])
        af=float(r['a_foot']); impact=float(r['impact_idx'])
        eind=ui/(1.0+math.exp(6.0*(af-1.5)))
        eeff=uo*p+eind
        loss=(ui*impact)**2
        return eeff,loss
    X=[features(r) for r in train_rows]
    y=np.array([float(r['y']) for r in train_rows])
    def residual(q):
        g,k,c=q
        return np.array([g*(e/(k+e))-c*l for e,l in X])-y
    # k_e is not reported as known; estimate it along with root gain and impact coefficient.
    result=least_squares(residual,x0=[6.0,1.0,0.5],bounds=([0.0,1e-8,0.0],[100.0,100.0,100.0]))
    return result.x

def predict(row):
    # fit() parameters are retained by the caller; use a lightweight attribute convention.
    g,k,c=_params
    ui=float(row['u_indoor']); af=float(row['a_foot'])
    eind=ui/(1.0+math.exp(6.0*(af-1.5)))
    eeff=float(row['u_outdoor'])*float(row['p_weather'])+eind
    return float(g*(eeff/(k+eeff))-c*((ui*float(row['impact_idx']))**2))
