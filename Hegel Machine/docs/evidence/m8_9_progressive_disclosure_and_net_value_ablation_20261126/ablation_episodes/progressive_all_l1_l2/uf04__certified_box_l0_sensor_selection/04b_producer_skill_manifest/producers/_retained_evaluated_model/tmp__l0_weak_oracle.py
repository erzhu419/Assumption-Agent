import sys
sys.path.insert(0, '/holdout_task')
import numpy as np
from certified_l0_solver import solve_certified_box_l0
rng = np.random.default_rng(820)
Q = np.linalg.qr(rng.normal(size=(32, 15)))[0]
failures = []
for ridge, fee in ((.12,.55),(.35,.85)):
    X = Q[:,np.arange(15)//2] + 1e-4*rng.normal(size=Q.shape)
    y = Q[:,:8] @ np.full(8,np.sqrt(2*fee*(1+ridge))+.12)
    args = dict(X=X,y=y,ridge=ridge,box=2.,l0_penalty=fee)
    out = None
    try:
        out = solve_certified_box_l0(args)
        w = np.array(out['weights'])
        print('solver',ridge,out['objective'],out['certified_lower_bound'],out['objective']-out['certified_lower_bound'],out['nodes_explored'],'recomputed',.5*np.sum((X@w-y)**2)+.5*ridge*(w@w)+fee*np.count_nonzero(abs(w)>1e-9),flush=True)
    except RuntimeError as e:
        print('solver',ridge,e,flush=True)
    # Independent test oracle only: solve all box modes in parallel via coordinate descent.
    H = X.T @ X + ridge*np.eye(15)
    h = X.T @ y
    modes = (np.arange(1<<15)[:,None] >> np.arange(15)) & 1
    lim = np.where(modes,2.,1e-9)
    w = np.zeros_like(lim)
    for sweep in range(240):
        change = 0.
        for j in range(15):
            v = np.clip(w[:,j]+(h[j]-w@H[:,j])/H[j,j],-lim[:,j],lim[:,j])
            change = max(change,np.max(abs(v-w[:,j])))
            w[:,j] = v
        if change < 1e-14:
            break
    g = w@H-h
    residual = w@X.T-y
    val = .5*np.sum(residual**2,axis=1)+.5*ridge*np.sum(w*w,axis=1)+fee*modes.sum(axis=1)
    lb = val-np.sum(g*w+np.abs(g)*lim,axis=1)
    print('oracle',ridge,np.min(lb),np.min(val),'sweeps',sweep+1,flush=True)
    if out is None or not (out['certified_lower_bound'] <= np.min(lb) + 1e-11 and np.min(val) - 1e-11 <= out['objective'] and 0 <= out['objective'] - out['certified_lower_bound'] <= 1e-5 and 1 <= out['nodes_explored'] <= 600):
        failures.append(ridge)
assert not failures, failures
