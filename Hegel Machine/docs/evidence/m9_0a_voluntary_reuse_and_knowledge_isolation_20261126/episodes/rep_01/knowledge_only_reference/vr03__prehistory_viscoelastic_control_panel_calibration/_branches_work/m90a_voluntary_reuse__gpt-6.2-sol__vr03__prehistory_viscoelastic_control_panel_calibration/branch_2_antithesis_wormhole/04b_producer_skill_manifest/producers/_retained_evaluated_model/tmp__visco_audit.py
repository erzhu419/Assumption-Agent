import importlib.util
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
spec = importlib.util.spec_from_file_location('visco','/holdout_task/viscoelastic_panel_calibration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_viscoelastic_cohort.pkl','rb') as f:
    data = pickle.load(f)
for i,d in enumerate(data):
    co,gt=d['cohort'],d['gt']
    t,q=co['train'],co['query']
    X,Z=[m._aligned_moduli(s,np.asarray(co['taus'])) for s in (t,q)]
    c,s,y=(t[k] for k in ('center_id','stratum','y'))
    cq,sq=q['center_id'],q['stratum']; yq=gt['y_query']
    print('cohort', i,'taus',co['taus'],'train prevalence',[(j,round(y[c==j].mean(),3)) for j in np.unique(c)], 'query prevalence',[(j,round(yq[cq==j].mean(),3)) for j in np.unique(cq)])
    for effect in (True,False):
        zs,pr,b=m._risk(X,y,c,s,Z,sq,effect)
        err=np.abs(zs-gt['Z_query']); mask=q['missing_channel_mask']
        print('center effects',effect,'weights',np.round(b,3),'MAE total/observed/missing',*[round(np.mean(e),4) for e in (err,err[~mask],err[mask])], 'per center AUC/Brier/MAE',[(round(roc_auc_score(yq[cq==j],pr[cq==j]),4),round(np.mean((yq[cq==j]-pr[cq==j])**2),4),round(err[cq==j].mean(),4)) for j in np.unique(cq)], 'oracle Brier',round(np.mean((yq-gt['prob_query'])**2),4))
        print('LOCO',[(j, round(roc_auc_score(y[h],pred),4), round(np.mean((y[h]-pred)**2),4)) for j in np.unique(c) for h in [c==j] for _,pred,_ in [m._risk(X[~h],y[~h],c[~h],s[~h],X[h],s[h],effect)]])
