import pickle, numpy as np, runpy
from scipy.optimize import least_squares
s=runpy.run_path('/tmp/hegel_continual_archive/alpha_m100_c04_batch_03_solver.py'); it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=it['dataset']; d=it['campaign_diagnostics']; tr,te=ds['train'],ds['test']; dt,t=ds['past_dt'],ds['eval_times']; x=s['_features'](tr,te); xd=s['_features'](tr,d)[len(tr['lab_id']):]; c=ds['calibration_load_cells']; r,l=c['bridge_raw'],c['true_channel_loads']; print('cal max sorted',np.sort(np.max(abs(r),axis=1))[-15:]); print('sweeps',[(k,np.shape(v)) for k,v in c['single_channel_sweeps'].items()]); b=s['_calibrate'](c); print('cal coeff',b.round(5)); print('cal residual quantiles',np.quantile(np.max(abs(np.einsum('nck,ck->nc',s['_bridge_design'](l),b)-r),axis=1),[.25,.5,.75,.9,.95,1])); inp=s['_inputs'](tr,x[:len(tr['lab_id'])],b); tau,th=s['_fit'](inp,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,th); p,h,z=s['_inputs'](d,xd,b); dr=z@th[2:]; y=d['strain_curves_obs']-d['strain_0'][:,None]; rate=d['instant_rate_obs']; a,g=s['_design'](tau,(p,h,z),dt,t); ar,gr=s['_chamber_response'](tau,th,h,z,t); print('baseline/reset rmse',[(np.sqrt(np.mean((v-y)**2)),np.sqrt(np.mean((w-rate)**2))) for v,w in [(a@th,g@th),(ar,gr)]]); tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; state=p@(-np.expm1(-dt[:,None]/tau)*np.exp(-tail[:,None]/tau)); R=-np.expm1(-t[:,None]/tau); T=np.vstack((np.c_[R,t],np.r_[1/tau,1])); Y=np.c_[y,rate]; print('diagnostic low-rank residual',np.quantile(abs(Y-np.linalg.lstsq(T,Y.T,rcond=None)[0].T@T.T),[.5,.9,.99,1])); print('hold and past',h[:5],p[:5]);
print('shapes',[(len(a['lab_id']),a['ref_coupons'].shape) for a in (tr,te,d)]); print('diag keys',list(d)); print('dataset keys',list(ds))
C0=np.c_[h[:,None]*th[:2],np.zeros(len(h))]; E=Y-C0@T.T
_,sing,V=np.linalg.svd(E,full_matrices=False); v=V[0]*np.sign(V[0,-1]); L=E@v; print('singular values',sing.round(4),'shape',np.linalg.lstsq(T,v,rcond=None)[0]); print('rank-one max',abs(E-L[:,None]*v).max())
from itertools import combinations
parts=[tr,te,d]
for ns in range(1,4):
 for ids in combinations(range(3),ns):
  pool=[parts[i] for i in ids]; labs=np.concatenate([a['lab_id'] for a in pool]); ref=np.concatenate([a['ref_coupons'] for a in pool]); xx=d['coupon_features_raw'].copy()
  for j in np.unique(d['lab_id']):
   m=d['lab_id']==j; g=ref[labs==j]; xx[m]=(xx[m]-np.nanmedian(np.nanmean(g,axis=0)))/np.nanmedian(np.nanstd(g,axis=0))
  for fill in ['known','cofit','joint']:
   phi=xx.copy(); missing=~np.isfinite(phi); phi[missing]=0
   w=np.c_[np.ones(len(h)),phi]; ww=w.copy()
   if fill=='known': ww=ww@th[2:]; valid=~missing.any(1); ww=ww[:,None]
   elif fill=='cofit': ww=np.c_[ww,missing]; valid=np.ones(len(h),bool)
   else:
    k={a:np.concatenate([z[a] for z in pool]) for a in ('lab_id','stratum','ref_coupons','coupon_features_raw')}; w2=s['_features'](k,k)[:len(labs)]; wn=np.concatenate([a['coupon_features_raw'] for a in pool]); _,im,counts=np.unique(np.c_[labs,k['stratum']],axis=0,return_inverse=True,return_counts=True); obs=np.isfinite(wn); imp=(w2*obs/counts[im,None]).sum(0)/(obs/counts[im,None]).sum(0); ww=(np.c_[np.ones(len(h)),np.where(missing,imp,xx)]@th[2:])[:,None]; valid=np.ones(len(h),bool)
   qq=np.linalg.lstsq(ww[valid],L[valid],rcond=None)[0]; res=L-ww@qq; print(ids,fill,'err max/med',np.max(abs(res[valid])).round(5),np.median(abs(res[valid])).round(5),'worst rows',np.argsort(abs(res[valid]))[-3:])
print('latent',np.c_[d['lab_id'],np.isnan(xx).any(1),dr,L].round(5))
