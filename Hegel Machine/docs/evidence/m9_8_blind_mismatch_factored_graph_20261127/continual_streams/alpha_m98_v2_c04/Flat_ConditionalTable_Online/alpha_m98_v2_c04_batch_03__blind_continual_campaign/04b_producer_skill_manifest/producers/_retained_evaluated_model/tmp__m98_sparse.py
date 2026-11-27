import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('base','/tmp/hegel_continual_archive/alpha_m98_v2_c04_batch_02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr,te=d['train'],d['test']; cd=i['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; b=m._bridge(d['calibration_load_cells']); xt=m._features(tr,te); n=len(tr['lab_id']); tau,th=m._fit(xt[:n],*m._loads(tr,b),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,th)
p,h=m._loads(cd,b); y=cd['strain_curves_obs']-cd['strain_0'][:,None]; r=cd['instant_rate_obs']; Y=np.c_[y,r]; print('chamber first row',Y[0].round(5)); print('chamber ranges',np.ptp(Y,axis=0).round(5),'svd',np.linalg.svd(Y,compute_uv=False).round(5)); print('diagnostic labs',np.unique(cd['lab_id'],return_counts=True),'missing',np.isnan(cd['coupon_features_raw']).sum(0))
merged={k:np.concatenate((te[k],cd[k])) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}; xd={'train+diag':m._features(tr,cd)[n:],'all':m._features(tr,merged)[len(xt):]}
labs=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]); x=cd['coupon_features_raw'].copy()
for lab in np.unique(cd['lab_id']):
 q=refs[labs==lab]; mask=cd['lab_id']==lab; x[mask]=(x[mask]-np.nanmedian(np.nanmean(q,0)))/np.nanmedian(np.nanstd(q,0))
_,ix,cnt=np.unique(np.c_[labs,np.r_[tr['stratum'],te['stratum']]],axis=0,return_inverse=True,return_counts=True); raw=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); w=1/cnt[ix,None]; fill=(xt*w*(~np.isnan(raw))).sum(0)/(w*(~np.isnan(raw))).sum(0); xd['train+test norm']=np.where(np.isnan(x),fill,x)
later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]
def memory(u):
 return p@(-np.expm1(-dt[:,None]/np.atleast_1d(u))*np.exp(-later[:,None]/np.atleast_1d(u)))
def design(q,X,mode):
 u=np.sort(np.exp(q[:2])); v=np.exp(q[2:]); dr=th[2]+X@th[3:]; one=np.ones(len(h)); gaps=h[:,None]-memory(tau)
 T=np.vstack((np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1]))
 if mode=='intrinsic_gap': blocks=[np.c_[gaps[:,0],dr],np.c_[gaps[:,1],dr],dr[:,None]]
 if mode=='chamber_gap':
  g=h[:,None]-memory(u); blocks=[np.c_[g[:,0],dr],np.c_[g[:,1],dr],dr[:,None]]
 if mode=='mix_gap': blocks=[np.c_[gaps,dr]]*2+[dr[:,None]]
 if mode=='free_gap':
  g=h[:,None]-memory(v); blocks=[np.c_[g[:,0],dr],np.c_[g[:,1],dr],dr[:,None]]
 if mode=='free_memory':
  z=memory(v); blocks=[np.c_[h,z[:,0],dr],np.c_[h,z[:,1],dr],np.c_[h,dr]]
 if mode=='full_past': blocks=[np.c_[h,p,dr]]*2+[np.c_[h,p,dr]]
 if mode=='free_features':
  z=memory(v); blocks=[np.c_[h,z[:,0],one,X],np.c_[h,z[:,1],one,X],np.c_[h,one,X]]
 if mode=='intrinsic_plus_fast': blocks=[np.c_[gaps[:,0],dr],np.c_[gaps[:,1],dr],np.c_[h,dr]]
 if mode=='kernel': blocks=[np.c_[gaps[:,0],h,dr],np.c_[gaps[:,1],h,dr],np.c_[h,dr]]
 if mode=='mixedkernel': blocks=[np.c_[gaps,h,dr]]*2+[np.c_[h,dr]]
 if mode=='near_archive': blocks=[np.c_[gaps[:,0],dr],np.c_[gaps[:,1],h,dr],np.c_[h,dr]]
 if mode=='near_archive_center': blocks=[np.c_[gaps[:,0],dr,one],np.c_[gaps[:,1],h,dr,one],np.c_[h,dr,one]]
 if mode=='paired_memory':
  g=h[:,None]-memory(v); blocks=[np.c_[g[:,0],h,dr],np.c_[g[:,1],h,dr],np.c_[h,dr]]
 return np.concatenate([B[:,None,:]*T[None,:,j,None] for j,B in enumerate(blocks)],axis=2)
def fit(X,mode,mask):
 def func(q):
  A=design(q,X,mode)[mask].reshape(mask.sum()*Y.shape[1],-1); b=Y[mask].ravel(); w=np.ones(len(b)); c=None
  for _ in range(3):
   c=np.linalg.lstsq(A*w[:,None],b*w,rcond=1e-9)[0]; w=(1+((A@c-b)/.007)**2)**(-.25)
  return A@c-b,c
 guesses=[[.465,1.55],[.45,4.5]] if mode not in ('free_gap','free_memory','free_features','paired_memory') else [[.465,1.55,.5,4.5],[.45,3.,.3,1.5],[.45,1.5,1.5,.5]]
 out=min((least_squares(lambda q:func(q)[0],np.log(g),bounds=(np.log(.05),np.log(24.9)),loss='soft_l1',f_scale=.006,max_nfev=100) for g in guesses),key=lambda a:a.cost)
 return out.x,func(out.x)[1]
def stats(e):return np.round([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))],5)
for name,X in xd.items():
 for mode in ('kernel','mixedkernel','near_archive','near_archive_center','paired_memory'):
  q,c=fit(X,mode,np.ones(len(Y),bool)); pred=design(q,X,mode)@c
  worst=np.zeros(2); oof=np.zeros_like(Y)
  for lab in np.unique(cd['lab_id']):
   mask=cd['lab_id']!=lab; qi,ci=fit(X,mode,mask); oof[~mask]=design(qi,X,mode)[~mask]@ci-Y[~mask]; worst=np.maximum(worst,stats(oof[~mask]))
  print(name,mode,np.exp(q).round(4),'fit',stats(pred-Y),'oof',stats(oof),'worst',worst,'coef',c.round(4),flush=True)
