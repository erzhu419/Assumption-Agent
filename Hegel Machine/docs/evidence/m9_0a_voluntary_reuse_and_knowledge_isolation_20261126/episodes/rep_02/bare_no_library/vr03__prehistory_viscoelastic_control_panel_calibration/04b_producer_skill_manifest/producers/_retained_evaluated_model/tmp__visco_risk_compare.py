import contextlib, io, runpy
with contextlib.redirect_stdout(io.StringIO()):
    env = runpy.run_path('/tmp/visco_eval.py')
import numpy as np
from scipy.special import expit
m, data, auc = (env[k] for k in ('m','data','auc'))
for i, case in enumerate(data):
    co,g = case['cohort'],case['gt']; tr,q = co['train'],co['query']
    x,xq = [m._aligned(d,np.asarray(co['taus'])) for d in (tr,q)]
    c,s,y = [np.asarray(tr[k]) for k in ('center_id','stratum','y')]
    cq,sq = [np.asarray(q[k]) for k in ('center_id','stratum')]
    print('cohort',i,'missing train/query',np.isnan(x).mean(),np.isnan(xq).mean(),'impute means',m._cell_stats(x,c,s)[0])
    for test in list(np.unique(c))+['query']:
        iq = test == 'query'; w = np.ones(len(x),bool) if iq else c!=test
        xv,sv,yv = (xq,sq,g['y_query']) if iq else (x[~w],s[~w],y[~w])
        cv = cq if iq else c[~w]
        mu,weight = m._cell_stats(x[w],c[w],s[w]); xt=x[w]
        at=np.column_stack((np.where(np.isnan(xt),mu,xt),np.eye(3)[s[w]])); av=np.column_stack((np.where(np.isnan(xv),mu,xv),np.eye(3)[sv])); complete=~np.isnan(xt).any(1)
        print(' fold',test,'complete-vs-missing errors')
        for label,ww,keep in [('cell',weight,np.ones(len(xt),bool)),('pooled',np.ones(len(xt)),np.ones(len(xt),bool)),('complete-cell',weight,complete),('complete-pooled',np.ones(len(xt)),complete)]:
            b=m._logistic(at[keep],y[w][keep],ww[keep]); p=expit(av@b)
            scores=[(round(auc(yv[z],p[z]),4),round(np.mean((p[z]-yv[z])**2),4)) for z in [cv==cc for cc in np.unique(cv)]]
            missing=np.isnan(xv).any(1); mb=[round(np.mean((p[z]-yv[z])**2),4) for z in (~missing,missing) if z.any()]
            print(' ',label,scores,'Brier by missing status',mb,'beta',b[:3].round(3))
        if iq:
            p=g['prob_query']; print(' oracle brier by missing', [round(np.mean((p[z]-yv[z])**2),4) for z in (~missing,missing)])
