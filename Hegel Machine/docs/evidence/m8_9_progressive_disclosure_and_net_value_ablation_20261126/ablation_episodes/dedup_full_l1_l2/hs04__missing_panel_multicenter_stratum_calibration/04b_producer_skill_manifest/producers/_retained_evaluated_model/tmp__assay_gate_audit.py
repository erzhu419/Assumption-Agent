import pickle, runpy
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

m = runpy.run_path('/holdout_task/multicenter_calibration.py')
with open('/holdout_task/public_multicenter_assay.pkl','rb') as f:
    cohorts = pickle.load(f)['cohorts']

def sub(d,k):
    return {a:b[k] for a,b in d.items()}

def fit(x,t,z,q,kind):
    levels=np.unique(t['stratum'])
    a=np.c_[x,3.*(t['stratum'][:,None]==levels)]
    b=np.c_[z,3.*(q['stratum'][:,None]==levels)]
    g, idx=np.unique(t['center_id'],return_inverse=True)
    y=t['y']
    if kind=='quadratic':
        j,k=np.triu_indices(x.shape[1]); a=np.c_[a,.2*x[:,j]*x[:,k]]; b=np.c_[b,.2*z[:,j]*z[:,k]]
    if kind=='center_cell':
        # Case/control reweighting is used only to estimate slopes. Refit stratum
        # offsets on the original observations before returning probabilities.
        cell=2*idx+y
        counts=np.bincount(cell)
        w=len(y)/(np.count_nonzero(counts)*counts[cell])
        nuisance=6.*((idx[:,None]==np.arange(len(g)))-1./len(g))
        model=LogisticRegression(C=.25,max_iter=500,tol=1e-8).fit(np.c_[a,nuisance],y,sample_weight=w)
        slope=model.coef_[0,:x.shape[1]]
        d=np.c_[np.ones(len(x)),a[:,x.shape[1]:]]
        score=x@slope
        def loss(h):
            eta=score+d@h
            return (np.sum(np.logaddexp(0.,eta)-y*eta)+2*np.dot(h[1:],h[1:]),d.T@(expit(eta)-y)+np.r_[0.,4*h[1:]])
        h=minimize(loss,np.zeros(d.shape[1]),jac=True,method='BFGS').x
        return expit(z@slope+np.c_[np.ones(len(z)),b[:,z.shape[1]:]]@h)
    model=LogisticRegression(C=.25,max_iter=500,tol=1e-8).fit(a,y)
    return model.predict_proba(b)[:,1]

def metrics(y,p):
    pred=p>=.5
    return [balanced_accuracy_score(y,pred),min(pred[y==1].mean(),(~pred[y==0]).mean()),roc_auc_score(y,p),np.mean((y-p)**2)]

for i,e in enumerate(cohorts):
    d,g=e['dataset'],e['gt']; t,q=d['train'],d['query']; a,b=m['_reference_affines'](d['reference_controls'])
    x,z=m['_align'](t,a,b),m['_align'](q,a,b)
    for kind in ('raw','aligned','center_cell','quadratic'):
        u,v=(np.nan_to_num(t['X']),np.nan_to_num(q['X'])) if kind=='raw' else (x,z)
        folds=[]; oof=np.empty(len(u))
        for c in np.unique(t['center_id']):
            h=t['center_id']==c
            oof[h]=fit(u[~h],sub(t,~h),u[h],sub(t,h),kind)
            folds.append(metrics(t['y'][h],oof[h]))
        p=fit(u,t,v,q,kind)
        qs=[metrics(g['y_query'][k],p[k]) for c in np.unique(q['center_id']) for k in [q['center_id']==c]]
        s=np.array(folds)
        print(i,kind,'LOCO pooled BA',round(balanced_accuracy_score(t['y'],oof>=.5),4),'LOCO worst BA/recall/AUC/Brier',np.round([*s[:,:3].min(0),s[:,3].max()],4),'query BA/recall/AUC/Brier',np.round(qs,4).tolist())
