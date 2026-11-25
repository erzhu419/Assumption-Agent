import importlib.util, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

spec = importlib.util.spec_from_file_location('assay', '/holdout_task/multicenter_calibration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_multicenter_assay.pkl','rb') as f:
    cohorts = pickle.load(f)['cohorts']

# The full indicator basis remains present. The optional contrast only changes
# the ridge penalty on an ordered trend, not the expressivity of the model.
variants = [(0.3, 1., 0.), (0.3, 2., 0.), (0.3, 3., 0.), (0.3, 1., 1.), (0.3, 1., 2.), (0.3, 2., 1.)]

def predict(x,s,c,y,tr,xx,ss,setting):
    C, penalty_scale, trend_scale = setting
    levels = np.unique(s[tr]); levels_c = np.unique(c[tr]); ct = (c[tr,None] == levels_c).astype(float)
    ct -= ct.mean(0)
    def clinical(strata):
        indicator = (strata[:,None] == levels).astype(float)
        trend = levels.astype(float) - levels.mean()
        return np.c_[penalty_scale * indicator, trend_scale * (indicator @ trend)]
    fit = np.c_[x[tr],clinical(s[tr]),3*ct]
    test = np.c_[xx,clinical(ss),np.zeros((len(xx),len(levels_c)))]
    model = LogisticRegression(C=C,max_iter=1000).fit(fit,y[tr])
    return model.predict_proba(test)[:,1], model.predict_proba(np.c_[np.zeros((len(levels),x.shape[1])),clinical(levels),np.zeros((len(levels),len(levels_c)))])[:,1]

def scores(y,p):
    return roc_auc_score(y,p), brier_score_loss(y,p), log_loss(y,p,labels=[0,1])

all_oof = {v:[] for v in variants}; nested_oof=[]
for k,co in enumerate(cohorts):
    d=co['dataset']; t,q=d['train'],d['query']; x=m._align(t,*m._reference_transforms(d['reference_controls'])); xq=m._align(q,*m._reference_transforms(d['reference_controls']))
    y,s,c=(t[j] for j in ('y','stratum','center_id')); centers=np.unique(c)
    def fold(tr,te,v):
        return scores(y[te],predict(x,s,c,y,tr,x[te],s[te],v)[0])
    for ctr in centers:
        tr=c!=ctr; te=~tr; remaining=centers[centers!=ctr]
        best=min(variants,key=lambda v: np.mean([fold(tr&(c!=j),c==j,v)[2] for j in remaining]))
        oof=fold(tr,te,best); nested_oof.append(oof)
        print('cohort',k,'LOCO',ctr,'selected',best,'AUROC/Brier',np.round(oof[:2],4))
        for v in variants: all_oof[v].append(fold(tr,te,v))
    best=min(variants,key=lambda v: np.mean([fold(c!=j,c==j,v)[2] for j in centers]))
    for v in dict.fromkeys((variants[0],best)):
        prob, zero = predict(x,s,c,y,np.ones(len(y),bool),xq,q['stratum'],v)
        bycenter=[]
        for ctr in np.unique(q['center_id']):
            te=q['center_id']==ctr
            bycenter.append(scores(co['gt']['y_query'][te],prob[te])[:2])
        print(' query',v,'center scores',np.round(bycenter,4),'neutral-site P at Z=0 by stratum',np.round(zero,3))
for v,rr in list(all_oof.items())+[('nested',nested_oof)]:
    a=np.asarray(rr); print(v,'mean',np.round(a.mean(0),4),'worst AUC/Brier',np.round([a[:,0].min(),a[:,1].max()],4))
