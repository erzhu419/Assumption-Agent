import importlib.util
import pickle
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, recall_score, roc_auc_score

spec = importlib.util.spec_from_file_location('assay', '/holdout_task/multicenter_calibration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_multicenter_assay.pkl','rb') as f:
    cohorts = pickle.load(f)['cohorts']

def metrics(y,p):
    if np.unique(y).size < 2:
        return None
    return tuple(np.round([roc_auc_score(y,p), brier_score_loss(y,p), balanced_accuracy_score(y,p>=.5), *recall_score(y,p>=.5,average=None,labels=[0,1],zero_division=0)],3))

def predict(x,s,c,y,tr,te,mode):
    st = s[:,None] == np.unique(s[tr])
    if mode == 'raw':
        fit = np.c_[x[tr], st[tr]]
        test = np.c_[x[te], st[te]]
    else:
        ct = (c[tr,None] == np.unique(c[tr])).astype(float)
        ct -= ct.mean(0)
        fit = np.c_[x[tr],st[tr],3*ct]
        test = np.c_[x[te],st[te],np.zeros((te.sum(),ct.shape[1]))]
        if mode != 'linear':
            i,j = np.triu_indices(x.shape[1]); products = x[:,i]*x[:,j]
            products -= products[tr].mean(0)
            products /= np.maximum(products[tr].std(0),1e-6)
            fit = np.c_[fit,mode*products[tr]]
            test = np.c_[test,mode*products[te]]
    return LogisticRegression(C=.3,max_iter=1000).fit(fit,y[tr]).predict_proba(test)[:,1]

for k,co in enumerate(cohorts):
    d,g=co['dataset'],co['gt']; t,q=d['train'],d['query']
    x=m._align(t,*m._reference_transforms(d['reference_controls']))
    raw=np.nan_to_num(t['X']); y,s,c=(t[a] for a in ('y','stratum','center_id'))
    groups=np.unique(c); modes=('linear',.15,.5)
    print('cohort',k,'metric order: AUC, Brier, balanced accuracy, recall0, recall1')
    for ctr in groups:
        te=c==ctr; tr=~te
        inner=groups[groups!=ctr]
        def loss(mode):
            return np.mean([log_loss(y[c==j],predict(x,s,c,y,tr&(c!=j),c==j,mode),labels=[0,1]) for j in inner])
        mode=min(modes,key=loss)
        print(' LOCO',ctr,'raw',metrics(y[te],predict(raw,s,c,y,tr,te,'raw')),'linear',metrics(y[te],predict(x,s,c,y,tr,te,'linear')),'nested',mode,metrics(y[te],predict(x,s,c,y,tr,te,mode)))
    out=m.calibrate_multicenter_assay(d); pq=out['query_probabilities']; xq=out['aligned_query_features']; w=out['feature_weights']; yq=g['y_query']
    alignment_delta=(xq-g['Z_query'])@w
    for ctr in np.unique(q['center_id']):
        sel=q['center_id']==ctr
        print(' query',ctr,'fit',metrics(yq[sel],pq[sel]),'oracle',metrics(yq[sel],g['prob_query'][sel]),'fit given true Z',metrics(yq[sel],expit(logit(pq[sel])-alignment_delta[sel])),'by stratum',[(int(j),metrics(yq[z],pq[z])) for j in np.unique(q['stratum'][sel]) if (z:=sel&(q['stratum']==j)).sum()])
