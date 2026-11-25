import pickle
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

with open('/holdout_task/public_multicenter_assay.pkl','rb') as f:
    cohorts = pickle.load(f)['cohorts']

def moments(r):
    def single(x):
        n = np.isfinite(x).sum(0)
        mu = np.divide(np.nansum(x,0), n, out=np.zeros(x.shape[1]), where=n>0)
        v = np.divide(np.nansum((x-mu)**2,0), n-1, out=np.zeros(x.shape[1]), where=n>1)
        return n, mu, v
    return {c:single(r['X'][r['center_id']==c]) for c in np.unique(r['center_id'])}

def align(a, stats, mass):
    x = np.array(a['X'],float,copy=True)
    for c in np.unique(a['center_id']):
        n,mu,v = stats[c]
        ok = (n>=2)&(v>0)
        center = np.median(mu[ok]); var = np.median(v[ok])
        w = np.divide(n-1,n-1+mass,out=np.zeros_like(mu),where=ok)
        m = center+w*(mu-center)
        s = np.sqrt(var+w*(v-var))
        rows = a['center_id']==c
        x[rows] = (x[rows]-m)/s
    return np.where(np.isfinite(x),x,0.)

def fit(x,t,z,q,C=1.,nuisance=0.,raw=False):
    s=np.union1d(t['stratum'],q['stratum'])
    enc=lambda a: (a['stratum'][:,None]==s).astype(float)*3.
    x=np.column_stack([x,enc(t)]); z=np.column_stack([z,enc(q)])
    if nuisance:
        c=np.unique(t['center_id']); g=(t['center_id'][:,None]==c).astype(float)-1./len(c)
        x=np.column_stack([x,nuisance*g]); z=np.column_stack([z,np.zeros((len(z),len(c)))])
    m=LogisticRegression(C=C,max_iter=1000,tol=1e-9).fit(x,t['y'])
    return m.predict_proba(z)[:,1],m.coef_[0,:8]

def subset(a,m): return {k:v[m] for k,v in a.items()}

def score(a,y,p):
    slices=[]
    for c in np.unique(a['center_id']):
        m=a['center_id']==c
        slices.append((roc_auc_score(y[m],p[m]), np.mean((p[m]-y[m])**2)))
    return np.array(slices)

if __name__=='__main__':
 for i,e in enumerate(cohorts):
    d,g=e['dataset'],e['gt']; t,q=d['train'],d['query']; r=moments(d['reference_controls']); y=g['y_query']
    print('cohort',i,'oracle',np.round(score(q,y,g['prob_query']),4).tolist())
    err=lambda z: [round(np.mean(abs(z[q['center_id']==c]-g['Z_query'][q['center_id']==c])),4) for c in np.unique(q['center_id'])]
    for mass in (0.,30.,100.,300.,1500.,1e6):
        x,z=align(t,r,mass),align(q,r,mass)
        print('mass',mass,'alignment mae',err(z))
    for mass in (0.,100.,1500.):
     for C in (.1,.5,3.,100.):
      for nuisance in (0.,6.):
        x,z=align(t,r,mass),align(q,r,mass); p,b=fit(x,t,z,q,C,nuisance)
        oof=np.zeros(len(x))
        for c in np.unique(t['center_id']):
            m=t['center_id']==c
            oof[m],_=fit(x[~m],subset(t,~m),x[m],subset(t,m),C,nuisance)
        sc=score(t,t['y'],oof)
        print('model',mass,C,nuisance,'loco',np.round([sc[:,0].min(),sc[:,1].max(),sc[:,1].mean()],4),'query',np.round(score(q,y,p),4).tolist(),'Brier',round(np.mean((p-y)**2),4),'probRMSE',round(np.mean((p-g['prob_query'])**2)**.5,4),'weights',round(np.linalg.norm(b),3))
    x,z=np.nan_to_num(t['X']),np.nan_to_num(q['X']); p,_=fit(x,t,z,q,3.)
    print('raw query',score(q,y,p))
