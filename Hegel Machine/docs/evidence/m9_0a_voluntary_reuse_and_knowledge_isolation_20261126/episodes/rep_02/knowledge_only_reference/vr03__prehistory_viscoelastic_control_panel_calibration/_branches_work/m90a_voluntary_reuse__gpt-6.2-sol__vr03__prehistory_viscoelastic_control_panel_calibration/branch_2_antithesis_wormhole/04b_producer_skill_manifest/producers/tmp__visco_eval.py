import sys, pickle, importlib.util, time
import numpy as np
from sklearn.metrics import roc_auc_score
spec=importlib.util.spec_from_file_location('cal','/holdout_task/viscoelastic_panel_calibration.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with open('/holdout_task/public_viscoelastic_cohort.pkl','rb') as f: data=pickle.load(f)
for idx, entry in enumerate(data):
    c,g=entry['cohort'],entry['gt']; start=time.monotonic(); o=m.calibrate_viscoelastic_cohort(c); print('cohort',idx,'seconds',round(time.monotonic()-start,3),'beta',o['modulus_weights'])
    z=o['aligned_query_moduli']; p=o['query_probabilities']; y=g['y_query']; pc=g['prob_query']; err=np.abs(z-g['Z_query']); mask=c['query']['missing_channel_mask']; cs=c['query']['center_id']; print('pooled',err.mean(),np.mean((p-y)**2),'prob-mse',np.mean((p-pc)**2),'observed MAE',err[~mask].mean())
    assert z.shape == g['Z_query'].shape and p.shape == y.shape and o['modulus_weights'].shape == (3,)
    assert np.isfinite(z).all() and np.isfinite(o['modulus_weights']).all() and ((p > 0) & (p < 1)).all()
    for k in np.unique(cs):
        a=cs==k; print('query center',k,'mae',err[a].mean(),'auc',roc_auc_score(y[a],p[a]),'oracle auc',roc_auc_score(y[a],pc[a]),'brier',np.mean((p[a]-y[a])**2),'means predicted observed oracle',p[a].mean(),y[a].mean(),pc[a].mean())
        assert err[a].mean() <= .22 and roc_auc_score(y[a],p[a]) >= .88 and np.mean((p[a]-y[a])**2) <= .160
    b=c['train']; cc=b['center_id']; ss=b['stratum']; yy=b['y']; zz=m._align(m._extract(b,np.asarray(c['taus'])),b['controls'],cc)
    for k in np.unique(cc):
        a=cc==k; w=m._cell_weights(cc[~a],ss[~a]); xt,xv=m._resolve_missing(zz[~a],zz[a],w); co=m._logistic(m._features(xt,ss[~a]),yy[~a].astype(float),w); pp=m._risk(m._features(xv,ss[a]),co,~np.isfinite(zz[a]),zz[~a])
        print('train LOCO', k, 'auc',roc_auc_score(yy[a],pp),'brier',np.mean((yy[a]-pp)**2),'prevalence',yy[a].mean(),'pred mean',pp.mean(),'n_s',np.bincount(ss[a]))
