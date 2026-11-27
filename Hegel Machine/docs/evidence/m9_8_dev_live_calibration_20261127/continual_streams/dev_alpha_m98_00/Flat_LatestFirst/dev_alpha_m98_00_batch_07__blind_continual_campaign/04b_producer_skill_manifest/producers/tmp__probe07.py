import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('m','/tmp/hegel_continual_archive/dev_alpha_m98_00_batch_06_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=a['dataset']; tr,te,g=d['train'],d['test'],a['campaign_diagnostics']; t,dt=d['eval_times'],d['past_dt']; tx,ex=m._prepare(d); tau,b=m._fit(dt,t,tx,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); P,H,_=m._prepare(dict(d,test=g))[1]; Y=g['strain_curves_obs']-g['strain_0'][:,None]; R=g['instant_rate_obs']; labs=g['lab_id']; Z=np.zeros((len(H),2))
for ds,p in zip(dt,P.T): Z+=-np.expm1(-ds/tau)*(p[:,None]-Z)
G=-np.expm1(-t[:,None]/tau); C=G-t[:,None]/tau; RM=lambda e:np.sqrt(np.mean(e**2)); print('tau,b',tau,b); print('diag keys:',list(g)); print('reference shape:',*[h['ref_coupons'].shape for h in [tr,te,g]])
for k in np.unique(labs):
 print('panel',k,[(round(np.median(np.nanmean((p:=h['ref_coupons'][h['lab_id']==k]),0)),4),round(np.median(np.nanstd(p,0)),4)) for h in [tr,te,g]])
keys=['lab_id','stratum','coupon_features_raw','ref_coupons']; merge=lambda *hs:{k:np.concatenate([h[k] for h in hs]) for k in keys}; ng=len(H)
Xall={'tr+diag':m._aligned_features(tr,g)[-ng:], 'all':m._aligned_features(merge(tr,te),g)[-ng:], 'diag':m._aligned_features(g,g)[-ng:]}
raw=np.array(g['coupon_features_raw']); lab=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]); fr=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]);
for k in np.unique(labs):
 p=refs[lab==k]; ctr=np.median(np.nanmean(p,0)); sd=np.median(np.nanstd(p,0)); raw[labs==k]=(raw[labs==k]-ctr)/sd; fr[lab==k]=(fr[lab==k]-ctr)/sd
_,ids,n=np.unique(np.c_[lab,np.r_[tr['stratum'],te['stratum']]],return_inverse=True,return_counts=True,axis=0); w=1/n[ids,None]; fill=np.nansum(fr*w,0)/np.sum(w*~np.isnan(fr),0); Xall['deploy panel']=np.c_[np.ones(ng),np.where(np.isnan(raw),fill,raw)]
for name,X in Xall.items():
 drift=X@b[2:]; amp=(H[:,None]-Z)*b[:2]; C0=amp@G.T+drift[:,None]*t; R0=(amp/tau).sum(1)+drift; YY=Y-C0; RR=R-R0; print('\n',name,'base rate',RM(RR),'no missing',RM(RR[~np.isnan(raw).any(1)])); print('rate errors',np.round(RR,3)); print('corr rate',np.round(np.corrcoef(np.c_[RR,drift,H,Z,X[:,1:]].T)[0,1:],2));
 for model,M in [('force',drift[:,None,None]*C[None,:,:]),('force+drift scale',np.concatenate([drift[:,None,None]*C[None,:,:],drift[:,None,None]*t[None,:,None]],2)),('force+z scale',np.concatenate([drift[:,None,None]*C[None,:,:],Z[:,None,:]*G[None,:,:]],2))]:
  M=M.reshape(ng,len(t),-1); Es=np.zeros_like(YY); pars=[]
  for k in np.unique(labs):
   fit=labs!=k; V=M[fit].reshape(-1,M.shape[-1]); v=YY[fit].ravel(); p=least_squares(lambda q: (V@q-v)/.014,np.zeros(M.shape[-1]),loss='soft_l1').x; pars.append(p); Es[~fit]=YY[~fit]-M[~fit]@p
  print(model,'pars',np.round(pars,4),'worst curve',max(RM(Es[labs==k]) for k in np.unique(labs)),'med',np.median(abs(Es)))
