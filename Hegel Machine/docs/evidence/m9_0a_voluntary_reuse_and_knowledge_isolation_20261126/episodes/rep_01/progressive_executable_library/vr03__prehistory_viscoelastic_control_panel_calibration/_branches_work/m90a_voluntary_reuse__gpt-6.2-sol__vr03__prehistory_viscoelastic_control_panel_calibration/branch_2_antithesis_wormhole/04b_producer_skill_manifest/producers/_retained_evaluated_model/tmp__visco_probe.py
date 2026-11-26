import pickle
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def design(tr, taus):
    p,u = np.asarray(tr['past_timestamps']), np.asarray(tr['past_strain_history'])
    t,x = np.asarray(tr['timestamps']), np.asarray(tr['strain_u'])
    m = np.full(len(taus),u[0],dtype=float)
    out = np.empty((len(t),len(taus)))
    dec = -np.expm1(-np.diff(np.r_[p,t])[:,None]/taus)
    for d,v in zip(dec[:len(p)-1],u[1:]): m += d*(v-m)
    for k,(d,v) in enumerate(zip(dec[len(p)-1:],x)):
        m += d*(v-m)
        out[k] = m
    return np.column_stack((np.ones(len(t)),x,out))

def fit(A,y):
    b = np.linalg.lstsq(A,y,rcond=None)[0]
    for _ in range(14):
        r = y-A@b
        s = max(.015, 1.4826*np.median(abs(r-np.median(r))))
        w = 1/np.hypot(1,r/(2*s))
        b = np.linalg.lstsq(w[:,None]*A,w*y,rcond=None)[0]
    return b

def extract(part,taus):
    raw, ols, details = [],[],[]
    for tr in part['traces']:
        A = design(tr,taus); y=np.asarray(tr['stress_obs']); b=fit(A,y)
        raw.append(b[1:]); ols.append(np.linalg.lstsq(A,y,rcond=None)[0][1:]); r=y-A@b
        details.append([np.linalg.cond(A),np.median(abs(r)),np.sum(abs(r)>1)])
    return np.array(raw),np.array(ols),np.array(details)

def aligned(raw,part):
    z=raw.copy()
    z[part['missing_channel_mask']]=np.nan
    for c in np.unique(part['center_id']):
        i=part['center_id']==c; ctl=part['controls'][i]
        z[i]=(z[i]-np.nanmedian(np.nanmean(ctl,axis=0)))/np.nanmedian(np.nanstd(ctl,axis=0))
    return z

def features(x,s,c,nuisance):
    S=np.eye(3)[s,1:]
    return np.column_stack((x,S,np.eye(4)[c,1:] if nuisance and c is not None else np.zeros((len(x),3)) if nuisance else np.empty((len(x),0))))

def predict(z,s,c,y,q,sq,nuisance=False):
    _,inv,cnt=np.unique(np.c_[c,s],axis=0,return_inverse=True,return_counts=True)
    w=1/cnt[inv];w/=w.mean()
    f=np.nansum(z*w[:,None],axis=0)/np.sum(np.isfinite(z)*w[:,None],axis=0)
    x=np.where(np.isfinite(z),z,f); q=np.where(np.isfinite(q),q,f)
    # Symmetric penalty on center offsets via all four one-hot columns.
    a=features(x,s,None,False); b=features(q,sq,None,False)
    if nuisance:
        G=np.eye(4)[c];g=np.unique(c);a=np.c_[a,G];b=np.c_[b,np.tile(np.isin(np.arange(4),g)/len(g),(len(q),1))]
    m=LogisticRegression(C=10,max_iter=500).fit(a,y,sample_weight=w)
    return m.predict_proba(b)[:,1],q,m.coef_[0],m.intercept_[0]

def scores(y,p): return [round(roc_auc_score(y,p),4),round(np.mean((y-p)**2),4)]
if __name__=='__main__':
    for k,d in enumerate(pickle.load(open('/holdout_task/public_viscoelastic_cohort.pkl','rb'))):
        co,gt=d['cohort'],d['gt']; tr,qu=co['train'],co['query']; t=np.asarray(co['taus'])
        rt,ot,dt=extract(tr,t); rq,oq,dq=extract(qu,t)
        z,q=aligned(rt,tr),aligned(rq,qu); ol=aligned(oq,qu)
        c,s,y=tr['center_id'],tr['stratum'],tr['y']; cq,sq=qu['center_id'],qu['stratum']; truth=gt['Z_query']; yq=gt['y_query']
        print('cohort',k,'trace diagnostic cond, median abs residual, glitch count',np.percentile(np.r_[dt,dq],[50,95,100],axis=0).round(4).tolist())
        print('per center stats means, prevalence, strata fraction')
        for zi,si,ci,yi in [(z,s,c,y),(q,sq,cq,yq)]:
            for j in np.unique(ci):
                i=ci==j; print(j,np.nanmean(zi[i],axis=0).round(3),round(yi[i].mean(),3),np.bincount(si[i])/i.sum())
        logit=np.log(gt['prob_query'])-np.log1p(-gt['prob_query']); print('query true nonmodulus logit by center/stratum',[[round(np.mean((logit-truth@gt['beta'])[(cq==j)&(sq==h)]),3) for h in range(3)] for j in np.unique(cq)])
        print('nonmissing aligned robust / OLS MAE',np.nanmean(abs(q-truth)),np.nanmean(abs(ol-truth)))
        for nuis in [False,True]:
            p,qf,b,a=predict(z,s,c,y,q,sq,nuis)
            print('center nuisance',nuis,'coef/intercept',b.round(3),round(a,3),'MAE',np.mean(abs(qf-truth)),'query pooled',scores(yq,p),'query centers',[scores(yq[cq==j],p[cq==j]) for j in np.unique(cq)],'loco',[scores(y[c==j],predict(z[c!=j],s[c!=j],c[c!=j],y[c!=j],z[c==j],s[c==j],nuis)[0]) for j in np.unique(c)])
