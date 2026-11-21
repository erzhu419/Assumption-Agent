import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u0=np.array([r['u_now']-r['u_past'] for r in train_rows],float)
    H=np.array([r['H'] for r in train_rows],float)
    y=np.array([r['y'] for r in train_rows],float)
    def res(p):
        E,tau=p
        return E*u0*np.exp(-H/tau)-y
    p0=[1.0,0.5]
    z=least_squares(res,p0,bounds=([-np.inf,1e-8],[np.inf,np.inf]))
    return z.x

def predict(row):
    E,tau=_params
    return float(E*(row['u_now']-row['u_past'])*np.exp(-row['H']/tau))