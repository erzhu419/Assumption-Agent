import sys, pickle
from itertools import combinations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/holdout_task')
import viscoelastic_panel_calibration as m

def predict(x, k, s, c, y, *, balanced, C, center_terms=False):
    w = m.balanced_cell_weights(c[k], s[k]) if balanced else np.ones(k.sum())
    fill = m.weighted_column_fill(x[k], w)
    a, b = m._features(np.where(np.isnan(x), fill, x), s)[k], m._features(np.where(np.isnan(x), fill, x), s)[~k]
    if center_terms:
        d = 1.0*(c[k,None] == np.unique(c[k]))
        a = np.column_stack([a, d - d.mean(axis=0)])
        b = np.column_stack([b, np.zeros((len(b),d.shape[1]))])
    fit = LogisticRegression(C=C, max_iter=500, tol=1e-9).fit(a,y[k],sample_weight=w)
    return fit.predict_proba(b)[:,1]

all_metrics={}
for i, case in enumerate(pickle.load(open('/holdout_task/public_viscoelastic_cohort.pkl','rb'))):
    cohort=case['cohort']; tr=cohort['train']; s,c,y=[tr[key] for key in ['stratum','center_id','y']]
    x=m._aligned(tr,cohort['taus'])
    control=tr['controls']; raw=x.copy()
    for g in np.unique(c):
        q=control[c==g]; raw[c==g]=x[c==g]*np.median(np.nanstd(q,axis=0))+np.median(np.nanmean(q,axis=0))
    print('case',i,'control observations per column by center',[(int(g),np.isfinite(control[c==g]).sum(axis=0).tolist()) for g in np.unique(c)])
    settings=[('raw',raw,True,5,False),('baseline',x,True,5,False),('aligned_unweighted',x,False,5,False),('aligned_center_intercepts',x,True,5,True),('aligned_C1',x,True,1,False),('aligned_C30',x,True,30,False)]
    for name,xx,w,C,fe in settings:
        metrics=[]
        for g in np.unique(c):
            k=c!=g; p=predict(xx,k,s,c,y,balanced=w,C=C,center_terms=fe)
            metrics.append((roc_auc_score(y[~k],p),np.mean((y[~k]-p)**2)))
        all_metrics.setdefault(name,[]).extend(metrics)
        print(name,np.round(metrics,4).tolist())
    # Worst-case query metrics remain separate from LOCO model comparisons.
    out=m.calibrate_viscoelastic_cohort(cohort); z=out['aligned_query_moduli']; p=out['query_probabilities']; q=cohort['query']; gt=case['gt']
    assert z.shape==(len(p),3) and p.shape==q['stratum'].shape and out['modulus_weights'].shape==(3,)
    assert np.isfinite(z).all() and np.isfinite(out['modulus_weights']).all() and ((0<p)&(p<1)).all()
    for g in np.unique(q['center_id']):
        k=q['center_id']==g
        missing=q['missing_channel_mask'][k]; err=np.abs(z[k]-gt['Z_query'][k]); yy=gt['y_query'][k]
        print('query',int(g),'MAE/all observed missing by-channel',err.mean(),err[~missing].mean(),err[missing].mean(),err.mean(axis=0),'AUC',roc_auc_score(yy,p[k]),'Brier',np.mean((yy-p[k])**2))
        assert err.mean() <= .22 and roc_auc_score(yy,p[k]) >= .88 and np.mean((yy-p[k])**2) <= .160
for name, a in all_metrics.items():
    a=np.array(a); print('LOCO summary',name,'min AUC',a[:,0].min(),'max Brier',a[:,1].max(),'mean',a.mean(axis=0))
