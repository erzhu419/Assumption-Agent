import math
import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    # Two serial first-order bridge states, initialized by the prior pull.
    amp, tau1, tau2, cap, grad = p
    h = max(float(row['H']), 0.0)
    n = max(20, int(math.ceil(h / 0.01)))
    dt = h / n if n else 0.0
    z1 = float(row['u_past'])
    z2 = float(row['u_past'])
    un = float(row['u_now'])
    t1 = max(tau1, 1e-5)
    t2 = max(tau2, 1e-5)
    for _ in range(n):
        def f(a, b):
            return (un-a)/t1, (a-b)/t2
        k1 = f(z1,z2)
        k2 = f(z1+dt*k1[0]/2,z2+dt*k1[1]/2)
        k3 = f(z1+dt*k2[0]/2,z2+dt*k2[1]/2)
        k4 = f(z1+dt*k3[0],z2+dt*k3[1])
        z1 += dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
        z2 += dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
    q = max(0.0, amp*z2 - grad*float(row['s_grad']))
    return cap * (1.0 - math.exp(-q/max(cap,1e-8))) + float(row['x_now']) + float(row['mem_conv'])

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_simulate(r,p) for r in rows]) - y
    p0 = np.array([2.5, 0.8, 1.0, 3.0, 0.3])
    lo = np.array([0.01, 0.03, 0.03, 0.1, 0.0])
    hi = np.array([10.0, 10.0, 10.0, 20.0, 5.0])
    result = least_squares(residual, p0, bounds=(lo,hi), max_nfev=3000)
    return result.x.tolist()

def predict(row):
    # fit(train_rows) must be called first; its fitted parameters are attached below.
    return _simulate(row, predict.params)
