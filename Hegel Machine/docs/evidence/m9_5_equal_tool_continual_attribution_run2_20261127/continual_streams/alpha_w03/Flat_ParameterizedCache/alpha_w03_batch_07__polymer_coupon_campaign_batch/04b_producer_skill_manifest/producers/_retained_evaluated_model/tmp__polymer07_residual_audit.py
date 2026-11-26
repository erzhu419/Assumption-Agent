import contextlib
import io
import runpy
import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    s = runpy.run_path('/tmp/polymer07_cv.py')
m, x, h, p, dt, t, q, z, rate, lab, complete, fold, err = (
    s[k] for k in ('m','x','h','p','dt','t','q','z','rate','lab','complete','fold','err'))
e = err[1]
large = np.abs(e) > 0.2
print('large residuals by time', large.sum(axis=0).tolist())
print('coupons by number of large residuals', np.bincount(large.sum(axis=1), minlength=len(t)+1).tolist())
print('fraction of squared curve error at large residuals', round(np.sum(e[large]**2)/np.sum(e**2), 6))

# Estimate a laboratory-specific drift offset using only the outer-training onset observations.
# Compare with the existing wing anchoring rule, without fitting to held-out curves.
raw_onset, adjusted_onset = np.empty(len(lab)), np.empty(len(lab))
adjusted_curve = e.copy()
for k in range(4):
    tr, te = fold != k, fold == k
    fit = tr & complete
    tau, w = m._fit(x[fit],h[fit],p[fit],dt,t,q,z[fit],rate[fit])
    _, rf = m._design(tau,x[fit],h[fit],p[fit],dt,t,q)
    _, rv = m._design(tau,x[te],h[te],p[te],dt,t,q)
    u = rate[fit] - rf @ w
    offset = np.array([np.median(u[lab[fit] == j]) for j in range(4)])
    blend = m._wing_blend(x[tr],h[tr],p[tr],dt,t,q,z[tr],rate[tr],lab[tr],complete[tr])
    remain = np.where(complete[te], 1-blend[lab[te]//2], 0.)
    inc = remain*offset[lab[te]]
    raw_onset[te] = remain*(rv @ w-rate[te])
    adjusted_onset[te] = raw_onset[te]+inc
    adjusted_curve[te] += inc[:,None]*t
for j in range(4):
    sel = lab == j
    print('lab',j,'curve MAE baseline/centered',np.round([np.abs(v[sel]).mean() for v in (e,adjusted_curve)],6),'curve median abs',np.round([np.median(np.abs(v[sel])) for v in (e,adjusted_curve)],6),'onset RMSE',np.round([np.sqrt(np.mean(v[sel]**2)) for v in (raw_onset,adjusted_onset)],6))
