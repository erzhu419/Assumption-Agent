import contextlib, io, runpy
import numpy as np
from scipy.special import expit
with contextlib.redirect_stdout(io.StringIO()):
    env = runpy.run_path('/tmp/visco_eval.py')
m, data, auc = (env[k] for k in ('m','data','auc'))

def metrics(y,p):
    return np.array([auc(y,p), np.mean((p-y)**2), np.mean(p[y==1]>=.5), np.mean(p[y==0]<.5)])

for i, case in enumerate(data):
    co = case['cohort']; tr = co['train']; taus = np.asarray(co['taus'])
    x = m._aligned(tr, taus)
    c,s,y = [np.asarray(tr[k]) for k in ('center_id','stratum','y')]
    raw = x.copy()
    for cc in np.unique(c):
        w = c == cc; panel = tr['controls'][w]
        raw[w] = raw[w]*np.nanmedian(np.nanstd(panel,axis=0)) + np.nanmedian(np.nanmean(panel,axis=0))
    pred = np.zeros((3, len(y)))
    for cc in np.unique(c):
        w = c != cc
        for j,xx in enumerate((raw,x)):
            mu, wt = m._cell_stats(xx[w],c[w],s[w]); a = np.column_stack((np.where(np.isnan(xx),mu,xx),np.eye(3)[s]))
            b = m._logistic(a[w],y[w],wt)
            pred[j,~w] = expit(a[~w]@b)
            if j == 1:
                # Fit center nuisances on training rows, then refit only stratum intercepts with slopes held fixed.
                d = np.eye(4)[c[w]][:,np.unique(c[w])[1:]]
                slopes = m._logistic(np.column_stack((a[w],d)),y[w],wt)[:3]
                offset = a[w,:3]@slopes
                intercept = []
                from scipy.optimize import brentq
                for ss in range(3):
                    k = s[w]==ss
                    intercept.append(brentq(lambda b: wt[k]@(expit(offset[k]+b)-y[w][k]),-50,50))
                pred[2,~w] = expit(a[~w,:3]@slopes+np.asarray(intercept)[s[~w]])
    print('cohort',i,'columns: AUROC/Brier/failure recall/nonfailure recall')
    for j, name in enumerate(('raw','panel aligned','panel aligned + nuisance slopes')):
        print(name,'pooled AUROC',round(auc(y,pred[j]),4))
        for cc in np.unique(c):
            w=c==cc
            print(' center',cc,metrics(y[w],pred[j,w]).round(4))
