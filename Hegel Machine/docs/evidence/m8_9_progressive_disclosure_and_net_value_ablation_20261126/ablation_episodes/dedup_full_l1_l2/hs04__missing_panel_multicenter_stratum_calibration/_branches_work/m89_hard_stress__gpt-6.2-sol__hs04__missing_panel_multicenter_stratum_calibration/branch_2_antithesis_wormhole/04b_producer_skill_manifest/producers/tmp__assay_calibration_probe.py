from assay_probe import *
from scipy.special import logit

for i,e in enumerate(cohorts):
    d,g=e['dataset'],e['gt']; t,q=d['train'],d['query']; r=moments(d['reference_controls']); y=g['y_query']
    x,z=align(t,r,100.),align(q,r,100.)
    p,b=fit(x,t,z,q,.5)
    raw,_=fit(np.nan_to_num(t['X']),t,np.nan_to_num(q['X']),q,3.)
    print('cohort',i,'pooled query Brier oracle/raw/aligned',np.round([np.mean((v-y)**2) for v in (g['prob_query'],raw,p)],4),'center oracle/raw/aligned',*[np.round(score(q,y,v),4).tolist() for v in (g['prob_query'],raw,p)])
    print('fitted weights',np.round(b,2))
    for name,a,h,f in [('train',t,x,t['y']),('query',q,z,y)]:
        for c in np.unique(a['center_id']):
            m=a['center_id']==c; v=np.where(np.isfinite(a['X'][m]),h[m],np.nan)
            corr=np.corrcoef(h[m].T); j=np.triu_indices_from(corr,1)
            print(name,c,'latent mean',np.round(np.nanmean(v,0),2),'latent sd',np.round(np.nanstd(v,0,ddof=1),2),'mean abs corr',round(abs(corr[j]).mean(),2),'missing rows',np.sum(~np.isfinite(v).all(1)))
    for center in np.unique(q['center_id']):
        m=q['center_id']==center
        ptrue=g['prob_query'][m]; mu=ptrue[:,None]*(1-ptrue[None,:]); expected_auc=(mu*(ptrue[:,None]>ptrue[None,:])).sum()/(mu.sum()-np.trace(mu))
        print('query',center,'oracle expected pairwise AUC',round(expected_auc,4),'irreducible expected Brier',round(np.mean(ptrue*(1-ptrue)),4),'estimated probability excess Brier',round(np.mean((p[m]-ptrue)**2),4))
