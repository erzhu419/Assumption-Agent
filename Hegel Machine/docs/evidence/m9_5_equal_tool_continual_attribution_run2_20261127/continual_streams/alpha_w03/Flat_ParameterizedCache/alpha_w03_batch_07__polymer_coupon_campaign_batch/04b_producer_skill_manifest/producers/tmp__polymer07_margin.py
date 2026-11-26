import runpy
import numpy as np

s = runpy.run_path('/tmp/polymer07_cv.py')
m, d, x, h, p, dt, t, q, z, rate, lab, complete, fold, err = (
    s[k] for k in ('m','d','x','h','p','dt','t','q','z','rate','lab','complete','fold','err'))
# These candidate omitted drift terms have the same t-dependence as the fitted drift.
# Fit each challenger on outer-training coupons only; report within-lab results.
def extras(f):
    a,b,c,e = f[:,1:].T
    return [np.column_stack((a*b,a*c,a*e,b*c,b*e,c*e)),
            np.column_stack(((a-b)/(1+c*c),(a+b)/(1+e*e),
                             (c-e)/(1+a*a),(c+e)/(1+b*b)))]

out = np.repeat(err[1:2], 3, axis=0)
for k in range(4):
    tr, te = (fold!=k)&complete, (fold==k)&complete
    tau, w = m._fit(x[tr],h[tr],p[tr],dt,t,q,z[tr],rate[tr])
    _, Rtr = m._design(tau,x[tr],h[tr],p[tr],dt,t,q)
    for i,(atr,ate) in enumerate(zip(extras(x[tr]),extras(x[te])),1):
        # Remove the existing linear span using the training fold alone.
        linear = np.linalg.lstsq(Rtr[:,:7],atr,rcond=None)[0]
        _, Rte = m._design(tau,x[te],h[te],p[te],dt,t,q)
        atr = atr-Rtr[:,:7]@linear
        ate = ate-Rte[:,:7]@linear
        coef = np.linalg.lstsq(atr,rate[tr]-Rtr@w,rcond=None)[0]
        out[i,te] += (ate@coef)[:,None]*t

for j in range(4):
    for observed in (True,False):
        mask = (lab==j)&(complete==observed)
        if not mask.any():
            continue
        for name, grid in [('early',t<=np.median(t)),('late',t>np.median(t))]:
            delta = np.abs(out[:,mask][:,:,grid]); v = delta.reshape(3,-1)
            print(f'lab={j} complete={observed} {name} n={mask.sum()} baseline/pairwise/rational median={np.round(np.median(v,axis=1),6)} RMSE={np.round(np.sqrt(np.mean(v*v,axis=1)),6)}')

lines = m._calibrate(d['calibration_load_cells'])
checks = d['calibration_load_cells']
for j, (gain,bias) in enumerate(lines):
    cmd, true = m._checks(checks,j)
    residual = np.abs(true-cmd*gain-bias)
    tj = d['test']['lab_id']==j
    _, hall, pall = m._prep(d)
    n = len(lab)
    stress = max(np.max(np.abs(hall[n:][tj])),np.max(np.abs(pall[n:][tj])))
    print(f'lab={j} sweep residuals={np.round(np.sort(residual),6).tolist()} test yield clearance={q-stress:.6g}')
