import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_m97_c02_s00_solver as s
with open('/holdout_task/public_alpha_m97_c02_s00.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
tr,te=d['train'],d['test']; n=len(tr['lab_id']); labs=tr['lab_id']; t=d['eval_times']; dt=d['past_dt']; cal=s._calibration(d); p,h=s._physical(tr,cal)
x,ref,g,b=[np.concatenate((tr[k],te[k])) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')]
missing=~np.isfinite(x[:n]).all(1)
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; v=tr['instant_rate_obs']
for l in range(4):
 m=g==l; shift=np.nanmedian(np.nanmean(ref[m],0)); scale=np.nanmedian(np.nanstd(ref[m],0)); print('panel',l,'shift',round(shift,5),'scale',round(scale,5)); x[m]=(x[m]-shift)/scale
cells=np.unique(np.c_[g,b],axis=0); means=np.array([np.nanmean(x[(g==l)&(b==j)],0) for l,j in cells]); default=np.nanmean(means,0)
for mode in ('global','local','stratum'):
 z=x.copy()
 for i,(l,j) in enumerate(cells):
  m=(g==l)&(b==j); pick={'global':default,'local':means[i],'stratum':np.nanmean(means[cells[:,1]==j],0)}[mode]; z[m]=np.where(np.isfinite(z[m]),z[m],pick)
 X=np.c_[np.ones(n),z[:n]]
 if mode=='global': np.testing.assert_allclose(z,s._align(d))
 misses=[]; scores=[]; drift=[]
 for l in range(4):
  m=labs==l; u,w=s._fit(p[~m],h[~m],dt,t,X[~m],y[~m],v[~m]); A,B=s._design(u,p[m],h[m],dt,t,X[m]); r=A@w-y[m]; q=B@w-v[m]; scores.append([np.max(np.median(abs(r),0)),np.median(abs(q))]); misses.append(np.c_[np.median(abs(r[missing[m]]),1),abs(q[missing[m]])]); drift.append(np.median((r/t)[~missing[m]],axis=0)[-1])
 a=np.concatenate(misses); print(mode,'worst lab time / rate',np.max(scores,0),'missing medians',np.median(a,0),'missing p90',np.quantile(a,.9,axis=0),'signed complete late drift',np.round(drift,6))
