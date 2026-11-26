import pickle, sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/holdout_task')
import viscoelastic_panel_calibration as m
for i, case in enumerate(pickle.load(open('/holdout_task/public_viscoelastic_cohort.pkl','rb'))):
    c,g = case['cohort'],case['gt']
    out = m.calibrate_viscoelastic_cohort(c)
    z,p,y = out['aligned_query_moduli'],out['query_probabilities'],g['y_query']
    print('CASE',i,'taus',c['taus'],'weights',out['modulus_weights'],'GT',g['beta'])
    tr = c['train']; print('training counts/prevalence',[(int(j),int(s),int(k.sum()),round(float(tr['y'][k].mean()),3)) for j in np.unique(tr['center_id']) for s in np.unique(tr['stratum']) if (k:=(tr['center_id']==j)&(tr['stratum']==s)).any()])
    for j in [None,*np.unique(c['query']['center_id'])]:
        k = np.ones(len(y),bool) if j is None else c['query']['center_id']==j
        present = ~c['query']['missing_channel_mask'][k]
        d = np.abs(z[k]-g['Z_query'][k]); oracle = g['prob_query'][k]
        print('center',j,'MAE',d.mean(),'observed channels',d[present].mean(),'AUC',roc_auc_score(y[k],p[k]),'oracle-AUC',roc_auc_score(y[k],oracle),'Brier',np.mean((p[k]-y[k])**2),'oracle-Brier',np.mean((oracle-y[k])**2),'prob RMSE',np.sqrt(np.mean((p[k]-oracle)**2)),'mean p/oracle/y',p[k].mean(),oracle.mean(),y[k].mean())
