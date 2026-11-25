import pickle
import numpy as np
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
np.set_printoptions(precision=3,suppress=True)
cohorts=pickle.load(open('/holdout_task/public_multicenter_assay.pkl','rb'))['cohorts']
def moments(r,c):
    with np.errstate(invalid='ignore',divide='ignore'):
        x=r['X'][r['center_id']==c]; n=np.isfinite(x).sum(0); m=np.nanmean(x,axis=0); sd=np.nanstd(x,axis=0,ddof=1)
    return n,m,sd
def align(r,p,a):
    x=p['X']; out=np.zeros_like(x)
    for c in np.unique(p['center_id']):
        n,m,s=moments(r,c); good=(n>=2)&(s>0); mm=np.median(m[good]); ss=np.median(s[good]); w=np.where(good,a,0)
        m=mm+w*(np.where(good,m,mm)-mm); s=ss+w*(np.where(good,s,ss)-ss); ix=p['center_id']==c
        out[ix]=np.nan_to_num((x[ix]-m)/s)
    return out
def design(z,s,c=None,centers=None):
    st=np.asarray(s)[:,None]==[1,2]; parts=[z, st.astype(float)]
    if c is not None: parts.append(5*((np.asarray(c)[:,None]==centers)-1/len(centers)))
    return np.column_stack(parts)
if __name__=='__main__':
 for entry in cohorts:
    d,g=entry['dataset'],entry['gt']; r,t,q=[d[k] for k in ('reference_controls','train','query')]; tc,qc=t['center_id'],q['center_id']; y=t['y']; u=np.unique(tc)
    truez=g['Z_query']; print('\nbeta',g['beta']); print('class counts',[int(y[tc==c].sum()) for c in u]); print('query target')
    for c in np.unique(qc):
        ix=qc==c; b=np.linalg.lstsq(design(truez[ix],q['stratum'][ix])[:,8:].tolist() if False else np.column_stack([np.ones(ix.sum()),q['stratum'][ix,None]==[1,2]]),logit(g['prob_query'][ix])-truez[ix]@g['beta'],rcond=None)[0]
        raw=q['X'][ix]; true=truez[ix]; aff=np.array([np.polyfit(true[np.isfinite(raw[:,j]),j],raw[np.isfinite(raw[:,j]),j],1) for j in range(8)]).T
        print(c,'rates',g['y_query'][ix].mean(),g['prob_query'][ix].mean(),'offsets',b,'oracle Brier/AUC',brier_score_loss(g['y_query'][ix],g['prob_query'][ix]),roc_auc_score(g['y_query'][ix],g['prob_query'][ix]),'true affine',aff,'patient latent mean',true.mean(0))
    for a in [0,.25,.5,.75,1]:
        zt=align(r,t,a); zq=align(r,q,a); print('shrink fraction',a,'MAE', [round(np.abs(zq[qc==c]-truez[qc==c]).mean(),4) for c in np.unique(qc)])
        for name,ztrain,zquery in [('plain',design(zt,t['stratum']), design(zq,q['stratum'])),('fixed',design(zt,t['stratum'],tc,u),np.column_stack([design(zq,q['stratum']),np.zeros((len(zq),len(u)))]))]:
            mdl=LogisticRegression(C=3,max_iter=500).fit(ztrain,y); p=mdl.predict_proba(zquery)[:,1]
            print(name,'int',mdl.intercept_, 'coefs',mdl.coef_.ravel()[:10], 'center bias', mdl.coef_.ravel()[10:]*5,'AUC/Brier/mean',[(round(roc_auc_score(g['y_query'][ix],p[ix]),4),round(brier_score_loss(g['y_query'][ix],p[ix]),4), round(p[ix].mean(),3)) for ix in [qc==c for c in np.unique(qc)]])
