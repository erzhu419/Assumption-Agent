from assay_stress_helpers import m, simulate
import pickle
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score, brier_score_loss

# Compare out-of-center transfer without using query labels in any fit.
def predict(d):
    r,t,q=(d[k] for k in ('reference_controls','train','query'))
    panels,pool=m._reference_transforms(r)
    z,zq=m._align(t,panels,pool),m._align(q,panels,pool)
    y=t['y']; c=t['center_id']; u=np.unique(c)
    A=m._design(z,t['stratum']); Aq=m._design(zq,q['stratum']); k=A.shape[1]
    X=np.column_stack([A,c[:,None]==u]); P=np.r_[0.,np.full(z.shape[1],3.),.25,.25,np.full(len(u),.25)]
    def fit(P):
        def fg(b):
            e=X@b
            return np.sum(np.logaddexp(0.,e)-y*e)+.5*np.dot(P*b,b),X.T@(expit(e)-y)+P*b
        return minimize(fg,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',options={'ftol':1e-12}).x
    b=fit(P); f=expit(X@b)
    H=X.T@((f*(1-f))[:,None]*X)+np.diag(P)
    V=np.linalg.inv(H)[k:,k:]
    contrast=np.eye(len(u))-1/len(u)
    observed=b[k:]@contrast@b[k:]
    noise=np.trace(contrast@V)
    tau2=np.clip((observed-2*noise)/max(len(u)-1,1),.04,10.)
    pred={}
    for name,pen in [('fixed',25.),('adaptive',1/tau2)]:
        P[k:]=pen; b=fit(P); eta=Aq@b[:k]
        pred[name]=expit(eta)
        missing=np.isnan(q['X'])
        missing_variance=missing@(b[1:1+z.shape[1]]**2)
        pred[name+'-integrated']=expit(eta/np.sqrt(1+np.pi/8*missing_variance))
    return pred,tau2

def audit(name, data):
    scores={}; tau=[]
    for dataset,g in data:
        ps,v=predict(dataset); tau.append(v)
        for key,p in ps.items():
            for c in np.unique(dataset['query']['center_id']):
                i=dataset['query']['center_id']==c; y=g['y_query'][i]; p0=g.get('prob_query')
                scores.setdefault(key,[]).append((roc_auc_score(y,p[i]),brier_score_loss(y,p[i]),np.mean((p[i]-p0[i])**2) if p0 is not None else 0))
    print(name,'tau2 range',np.round([min(tau),max(tau)],4))
    for key,rows in scores.items():
        a=np.asarray(rows); print(key,'mean',a.mean(0).round(4),'worst',np.array([a[:,0].min(),a[:,1].max(),a[:,2].max()]).round(4))

with open('/holdout_task/public_multicenter_assay.pkl','rb') as f: entries=pickle.load(f)['cohorts']
for j,e in enumerate(entries):
    d,g=e['dataset'],e['gt']; mis=np.isnan(d['query']['X']); print('query missing latent |Z| mean',j,np.mean(np.abs(g['Z_query'][mis])))
    audit(('public query',j),[(d,g)])
    t=d['train']; folds=[]
    for c in np.unique(t['center_id']):
        i=t['center_id']!=c
        folds.append(({'reference_controls':d['reference_controls'],'train':{k:v[i] for k,v in t.items()},'query':{k:v[~i] for k,v in t.items() if k!='y'}},{'y_query':t['y'][~i]}))
    audit(('LOCO',j),folds)
for het in (.05,.23):
    for shift in (0.,2.8):
        audit(('independent synthetic seeds',het,shift),[simulate(seed,het,shift) for seed in range(12,28)])
