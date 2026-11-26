import contextlib, io, runpy
import numpy as np
from scipy.special import expit
with contextlib.redirect_stdout(io.StringIO()):
    e = runpy.run_path('/tmp/visco_eval.py')
m, data, auc = (e[k] for k in ('m', 'data', 'auc'))
# Integrate missing predictor uncertainty around the mandated mean, without changing the imputed moduli.
nodes, ghw = np.polynomial.hermite.hermgauss(16)
for i, case in enumerate(data):
    co = case['cohort']; tr = co['train']
    x = m._aligned(tr, np.asarray(co['taus']))
    c,s,y = [np.asarray(tr[k]) for k in ('center_id','stratum','y')]
    for cc in np.unique(c):
        w = c != cc; xv=x[~w]; yv=y[~w]
        mu, weight = m._cell_stats(x[w],c[w],s[w]); xt = x[w]
        at = np.column_stack((np.where(np.isnan(xt),mu,xt),np.eye(3)[s[w]]))
        av = np.column_stack((np.where(np.isnan(xv),mu,xv),np.eye(3)[s[~w]]))
        missing = np.isnan(xv); complete = np.isfinite(xt).all(1)
        bv,cw = m._cell_stats(xt[complete],c[w][complete],s[w][complete]); del bv
        b0 = m._logistic(at,y[w],weight); b1=m._logistic(at[complete],y[w][complete],cw)
        var = np.nanvar(xt, axis=0)
        scenarios = [('baseline', expit(av@b0))]
        for name,b in [('complete+marginal',b1),('all+marginal',b0)]:
            sd = np.sqrt(missing@(var*b[:3]**2))
            scenarios.append((name,expit((av@b)[:,None]+np.sqrt(2)*sd[:,None]*nodes)@ghw/np.sqrt(np.pi)))
        print(i, cc, 'missing rows',missing.any(1).sum(), [(name,round(auc(yv,p),4),round(np.mean((p-yv)**2),4)) for name,p in scenarios])
