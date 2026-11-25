import importlib.util, pickle
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, brier_score_loss
spec=importlib.util.spec_from_file_location('assay','/holdout_task/multicenter_calibration.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def predictors(d):
    r,t,q=[d[k] for k in ('reference_controls','train','query')]; y=t['y']; c=t['center_id']; u=np.unique(c)
    panel,pool=m._reference_transforms(r); z=m._align(t,panel,pool); zq=m._align(q,panel,pool)
    X=np.column_stack((m._design(z,t['stratum']),c[:,None]==u)); Xq=m._design(zq,q['stratum']); p=Xq.shape[1]
    def fit(lam):
        ridge=np.r_[0.,np.full(z.shape[1],3.),.25,.25,np.repeat(lam,len(u))]
        def fg(b):
            eta=X@b
            return (np.logaddexp(0,eta).sum()-y@eta+.5*np.dot(ridge*b,b),X.T@(expit(eta)-y)+ridge*b)
        return minimize(fg,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',options={'ftol':1e-12}).x
    b=fit(.05); f=expit(X@b); W=f*(1-f)
    # Unpenalized group dispersion after subtracting its sampling variance.
    precision=X.T@(W[:,None]*X)+np.diag(np.r_[0.,np.full(z.shape[1],3.),.25,.25,np.full(len(u),.05)])
    cov=np.linalg.inv(precision)[p:,p:]; proj=np.eye(len(u))-1/len(u)
    var=max(.04,(np.sum(b[p:]**2)-np.trace(proj@cov))/(len(u)-1)) if len(u)>1 else .04
    return zq,{name:expit(Xq@fit(lam)[:p]) for name,lam in [('weak',.5),('fixed',25.),('adaptive',1/var)]},var

def scores(c,z,p,g):
    return [[np.abs(z[c==j]-g['Z_query'][c==j]).mean(),roc_auc_score(g['y_query'][c==j],p[c==j]),brier_score_loss(g['y_query'][c==j],p[c==j])] for j in np.unique(c)]

def simulate(seed,heter,confound):
    rng=np.random.default_rng(seed); D=8
    shift=np.array([3.5,3.5,-3.5,-3.5,-2.7,2.8])[:,None]+rng.normal(0,heter,(6,D)); scale=np.array([1.3,.9,1.1,1.05,1.45,.75])[:,None]*np.exp(rng.normal(0,heter/2,(6,D)))
    def raw(z,c,miss):
        a=shift[c]+scale[c]*z; a[rng.random(a.shape)<miss]=np.nan; return a
    rc=np.repeat(np.arange(6),[80,75,75,75,24,18]); rx=raw(rng.normal(size=(rc.size,D)),rc,.15)
    ix=np.flatnonzero(rc==4); rx[ix[1:],0]=np.nan; rx[ix,1]=7.; ix=np.flatnonzero(rc==5); rx[ix,5]=np.nan; rx[ix[1:],7]=np.nan
    beta=rng.normal(size=D); beta*=3.3/np.linalg.norm(beta)
    def patient(c,n,train):
        c=np.repeat(c,n); s=rng.choice(3,c.size,p=[.60,.25,.15]); z=rng.normal(size=(c.size,D)); base=z@beta+np.array([-.65,.55,1.4])[s]
        if train: base+=np.array([1,1,-1,-1,0,0])[c]*confound
        p=expit(base); y=rng.binomial(1,p); return {'center_id':c,'X':raw(z,c,.045),'stratum':s},z,y,p
    t,zt,yt,pt=patient(np.arange(4),100,True); t['y']=yt
    q,z,y,p=patient(np.arange(4,6),180,False)
    return {'reference_controls':{'X':rx,'center_id':rc},'train':t,'query':q},{'Z_query':z,'y_query':y,'prob_query':p}

def compare(name, entries, report_each=False):
    stats={k:[] for k in ('weak','fixed','adaptive','oracle')}; variances=[]
    for d,g in entries:
        z,ps,v=predictors(d); variances.append(v); ps['oracle']=g['prob_query']; c=d['query']['center_id']
        for k,p in ps.items(): stats[k].extend(scores(c,z,p,g))
    print(name,'adaptive center variance range',np.round([min(variances),max(variances)],3))
    for k,rows in stats.items():
        a=np.array(rows); print(k,'mean',a.mean(0).round(4),'worst',np.array([a[:,0].max(),a[:,1].min(),a[:,2].max()]).round(4),a.round(4).tolist() if report_each else '')
