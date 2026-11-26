import pickle
import numpy as np
from scipy.special import logit
P='/holdout_task/public_viscoelastic_cohort.pkl'
def load():
    with open(P,'rb') as f: return pickle.load(f)
def design(trace,taus):
    tp=np.asarray(trace['past_timestamps']); u=np.asarray(trace['strain_u']); t=np.asarray(trace['timestamps']); hp=np.asarray(trace['past_strain_history']); taus=np.asarray(taus)
    x=np.r_[hp,u]; time=np.r_[tp,t]; m=np.empty((len(x),2)); m[0]=x[0]
    for k,a in enumerate(np.exp(-np.diff(time)[:,None]/taus),1): m[k]=a*m[k-1]+(1-a)*x[k]
    return np.column_stack((np.ones(len(t)), u, m[len(tp):]))
if __name__=='__main__':
    R=np.random.default_rng(518)
    subs=np.array([R.choice(36,4,replace=False) for _ in range(140)])
    for j,d in enumerate(load()):
        D,G=d['cohort'],d['gt']; print('case',j,'taus',D['taus'],'beta',G['beta'])
        for key,block in [(x,D[x]) for x in ('train','query')]:
            raw=[]; res=[]; cond=[]
            for tr in block['traces']:
                X=design(tr,D['taus']); y=np.array(tr['stress_obs']); cond.append(np.linalg.cond(X))
                cand=np.einsum('ijk,ik->ij',np.linalg.pinv(X[subs]),y[subs]); e=np.abs(y[:,None]-X@cand.T)
                fit=cand[np.argmin(np.median(e,axis=0))]; err=np.abs(y-X@fit); ok=err<=max(0.1,5*np.median(err)); fit=np.linalg.lstsq(X[ok],y[ok],rcond=None)[0]; raw.append(fit[1:]); res.append(np.sort(np.abs(y-X@fit)))
            raw=np.array(raw); res=np.array(res); keep=~block['missing_channel_mask']; controls=block['controls']; s=block['stratum']; cid=block['center_id']; y=block.get('y',G['y_query']); print(key,'condition',np.round(np.quantile(cond,[0,.5,.95,1]),2),'order residual median',np.round(np.median(res,axis=0),4),'mask',np.mean(~keep,axis=0))
            for c in np.unique(cid):
                i=cid==c; panel=controls[i]; off=np.median(np.nanmean(panel,axis=0)); gain=np.median(np.nanstd(panel,axis=0)); Z=(raw[i]-off)/gain
                print('center',c,'panel',np.round([off,gain],3),'strata n/p',[(int(sum(i&(s==t))),round(y[i&(s==t)].mean(),3)) for t in range(3)],'Z cell means',np.round([Z[s[i]==t].mean(0) for t in range(3)],3))
                if key=='query':
                    gt=G['Z_query'][i]; print('  observed MAE',np.abs(Z-gt)[keep[i]].mean(),'miss oracle panel MAE',np.abs(Z-gt)[~keep[i]].mean(),'latent affine residual',np.round(np.quantile((logit(G['prob_query'][i])-gt@G['beta']),[0,.1,.5,.9,1]),3),'strata logits', [np.round(np.unique(np.round((logit(G['prob_query'][i])-gt@G['beta'])[s[i]==t],6)),3).tolist()[:8] for t in range(3)])
