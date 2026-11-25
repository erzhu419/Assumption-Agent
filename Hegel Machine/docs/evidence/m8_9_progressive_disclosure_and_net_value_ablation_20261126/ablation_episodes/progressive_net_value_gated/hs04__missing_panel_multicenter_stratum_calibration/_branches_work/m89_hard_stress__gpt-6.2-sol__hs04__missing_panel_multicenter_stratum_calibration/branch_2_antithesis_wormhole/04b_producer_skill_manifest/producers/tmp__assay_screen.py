import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

cohorts = pickle.load(open('/holdout_task/public_multicenter_assay.pkl','rb'))['cohorts']

def aligned(part, ref, shrink, robust=False):
    out = np.zeros_like(part['X'], dtype=float)
    for c in np.unique(part['center_id']):
        block = ref['X'][ref['center_id']==c]
        mu, sig, n = [], [], []
        for v in block.T:
            v = v[np.isfinite(v)]
            n.append(len(v))
            m = np.median(v) if robust and len(v) else np.mean(v) if len(v) else 0
            sd = 1.4826*np.median(abs(v-m)) if robust and len(v)>1 else np.std(v,ddof=1) if len(v)>1 else 0
            mu.append(m); sig.append(sd)
        mu,sig,n = map(np.asarray,(mu,sig,n)); ok = (n>=2)&(sig>0)
        m0,s0 = np.median(mu[ok]),np.median(sig[ok]); w = np.where(ok,n/(n+shrink),0)
        group = part['center_id']==c
        out[group] = (part['X'][group]-(w*mu+(1-w)*m0))/(w*sig+(1-w)*s0)
    return np.nan_to_num(out)

def score(y,p,c):
    return np.array([(roc_auc_score(y[c==i],p[c==i]),brier_score_loss(y[c==i],p[c==i])) for i in np.unique(c)])

def pred(xt,t,xq,q,cscale,C):
    st=np.eye(3)[t['stratum']]; sq=np.eye(3)[q['stratum']]
    centers=np.unique(t['center_id']); ct=(t['center_id'][:,None]==centers).astype(float)
    design=np.c_[xt,st,cscale*(ct-ct.mean(0))]
    model=LogisticRegression(C=C,max_iter=1000).fit(design,t['y'])
    return model.predict_proba(np.c_[xq,sq,np.zeros((len(xq),len(centers)))])[:,1], model.coef_[0,:8]

np.set_printoptions(precision=3,suppress=True)
for i,co in enumerate(cohorts):
    d,g=co['dataset'],co['gt']; r,t,q=[d[k] for k in ('reference_controls','train','query')]
    print('cohort',i,'oracle scores',score(g['y_query'],g['prob_query'],q['center_id']))
    for c in np.unique(t['center_id']):
        m=t['center_id']==c
        x=aligned(t,r,120)[m]; y=t['y'][m]
        print('train center',c,'prev',y.mean(),'aligned means',x.mean(0),'SD',x.std(0))
    for shrink,robust in [(0,False),(30,False),(120,False),(500,False),(1e9,False),(120,True),(1e9,True)]:
        xt,xq=aligned(t,r,shrink,robust),aligned(q,r,shrink,robust)
        err=np.abs(xq-g['Z_query']); cq=q['center_id']; yq=g['y_query']
        print('alignment',shrink,robust,'MAE',err.mean(),[err[cq==c].mean() for c in np.unique(cq)],'max column mean',err.mean(0).max())
        if shrink in [0,120,1e9] and not robust:
            for mode,cs,C in [('raw',0,1),('aligned',0,1),('fixed-weak',1,1),('fixed',5,1),('fixed-reg',5,.1),('fixed-free',5,10)]:
                p,b=pred(np.nan_to_num(t['X']) if mode=='raw' else xt,t,np.nan_to_num(q['X']) if mode=='raw' else xq,q,cs,C)
                print(mode, score(yq,p,cq).ravel(), 'pooled',round(roc_auc_score(yq,p),3),round(brier_score_loss(yq,p),3),'beta',b)
