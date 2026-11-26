import importlib.util
import pickle
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

spec=importlib.util.spec_from_file_location('calibration','/holdout_task/viscoelastic_panel_calibration.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_viscoelastic_cohort.pkl','rb') as f: data=pickle.load(f)

def score(y,p):
    if len(np.unique(y))<2:
        return np.array([np.nan, np.mean((y-p)**2), np.nan])
    return np.array([roc_auc_score(y,p), np.mean((y-p)**2), balanced_accuracy_score(y,p>=.5)])

for case,d in enumerate(data):
    cohort, gt=d['cohort'],d['gt']
    t,q=cohort['train'],cohort['query']
    out=m.calibrate_viscoelastic_cohort(cohort)
    p=out['query_probabilities']; z=out['aligned_query_moduli']; y=gt['y_query']; ctr=q['center_id']; mask=q['missing_channel_mask']; err=np.abs(z-gt['Z_query'])
    print('public',case,'MAE',err.mean(),'masked',err[mask].mean(),'unmasked',err[~mask].mean(),'by channel',err.mean(0),'weights',out['modulus_weights'],'pooled',score(y,p))
    for c in np.unique(ctr):
        rows=ctr==c
        print(' query center',c,'MAE',err[rows].mean(),'metrics',score(y[rows],p[rows]),'stratum metrics',[(s,np.round(score(y[rows&(q['stratum']==s)],p[rows&(q['stratum']==s)]),4)) for s in range(3)])
    rng=np.random.default_rng(518); subsets=np.array([rng.choice(36,4,replace=False) for _ in range(96)])
    raw=m._extract(t,np.asarray(cohort['taus']),subsets); aligned=m._align(raw,t); c=t['center_id']; s=t['stratum']; y=t['y']; scores={}
    for held in np.unique(c):
        test=c==held; train=~test; idx,counts=m._cell_index(c[train],s[train]); means=m._imputation(aligned[train],idx,len(counts)); arr=np.where(np.isnan(aligned),means,aligned); x=m._risk_design(arr,s)
        for power in (0,.5,1):
            w=counts[idx].astype(float)**-power; w*=sum(train)/w.sum()
            for ridge in (.1,1,3):
                b=m._fit_risk(x[train],y[train],w,ridge)
                scores.setdefault((power,ridge),[]).append(score(y[test],expit(x[test]@b)))
        rawmean=m._imputation(raw[train],idx,len(counts)); xr=m._risk_design(np.where(np.isnan(raw),rawmean,raw),s); b=m._fit_risk(xr[train],y[train],np.ones(sum(train)))
        scores.setdefault('unaligned',[]).append(score(y[test],expit(xr[test]@b)))
    for variant, sc in scores.items():
        a=np.array(sc); print(' LOCO',variant,'mean',np.round(a.mean(0),4),'worst',np.round([a[:,0].min(),a[:,1].max(),a[:,2].min()],4))
