import importlib.util, pickle
import numpy as np
from scipy.stats import rankdata
from scipy.special import logit
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('vp',p+'viscoelastic_panel_calibration.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def auc(y,v):
    n=sum(y); k=len(y)-n
    return (rankdata(v)[y==1].sum()-n*(n+1)/2)/(n*k) if n*k else np.nan
with open(p+'public_viscoelastic_cohort.pkl','rb') as f: data=pickle.load(f)
for i,case in enumerate(data):
    c,g=case['cohort'],case['gt']; tr,q=c['train'],c['query']; r=m.calibrate_viscoelastic_cohort(c)
    z,y,prob=g['Z_query'],g['y_query'],g['prob_query']; x,v,b=r.values()
    print('public',i,'taus',c['taus'],'beta fit/true',b,g['beta'],'MAE/Brier/AUC', np.mean(abs(x-z)),np.mean((v-y)**2),auc(y,v),'oracle AUC/Brier',auc(y,prob),np.mean((prob-y)**2))
    implied=logit(prob)-z@g['beta']; print('query true logit offsets by center/stratum:')
    for cc in np.unique(q['center_id']):
        w=q['center_id']==cc; mask=q['missing_channel_mask'][w]; e=(x-z)[w]; print(cc,w.sum(),'mae / unmasked / auc / Brier',np.abs(e).mean(),np.abs(e[~mask]).mean(),auc(y[w],v[w]),np.mean((v[w]-y[w])**2),'oracle auc',auc(y[w],prob[w]),'offset',[(np.mean(implied[w&(q['stratum']==s)]),np.std(implied[w&(q['stratum']==s)])) for s in range(3)],'loc error',e.mean(0))
    xt=m._aligned(tr,np.asarray(c['taus']))
    print('train center counts/rates/latent means',[(k,[(int(w.sum()),round(tr['y'][w].mean(),3)) for w in [(tr['center_id']==k)&(tr['stratum']==s) for s in range(3)]],np.round(np.nanmean(xt[tr['center_id']==k],axis=0),3)) for k in np.unique(tr['center_id'])])
    for k in np.unique(tr['center_id']):
        w=tr['center_id']==k
        sub=lambda w:{a:np.asarray(tr[a])[w] for a in ('center_id','stratum','y')}
        _, pv, bv=m._predict(xt[~w],xt[w],sub(~w),sub(w)); print('LOCO',k,'AUC/Brier',auc(tr['y'][w],pv),np.mean((pv-tr['y'][w])**2),'beta',bv)
