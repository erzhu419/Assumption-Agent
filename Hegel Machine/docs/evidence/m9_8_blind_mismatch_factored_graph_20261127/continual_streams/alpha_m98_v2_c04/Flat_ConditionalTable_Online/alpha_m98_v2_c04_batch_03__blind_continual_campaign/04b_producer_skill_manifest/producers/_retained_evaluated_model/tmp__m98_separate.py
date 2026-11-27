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
 return np.concatenate([B[:,None,:]*T[None,:,j,None] for j,B in enumerate(blocks)],axis=2)
def fit(X,mode,mask):
 def func(q):
  A=design(q,X,mode)[mask].reshape(mask.sum()*Y.shape[1],-1); b=Y[mask].ravel(); w=np.ones(len(b)); c=None
  for _ in range(3):
   c=np.linalg.lstsq(A*w[:,None],b*w,rcond=1e-9)[0]; w=(1+((A@c-b)/.007)**2)**(-.25)
  return A@c-b,c
 guesses=[[.465,1.55],[.45,4.5]] if mode not in ('free_gap','free_memory','free_features') else [[.465,1.55,.5,4.5],[.45,3.,.3,1.5],[.45,1.5,1.5,.5]]
 out=min((least_squares(lambda q:func(q)[0],np.log(g),bounds=(np.log(.05),np.log(24.9)),loss='soft_l1',f_scale=.006,max_nfev=100) for g in guesses),key=lambda a:a.cost)
 return out.x,func(out.x)[1]
def stats(e):return np.round([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))],5)
for k in ['ref_coupons','past_channel_raw','hold_channel_raw','coupon_features_raw']:
 a=cd[k].reshape(len(h),-1); b=np.concatenate((tr[k],te[k])).reshape(len(xt),-1); print('exact matches',k,np.sum(np.any(np.all(a[:,None]==b[None],axis=-1),axis=1)))
print('diagnostic load quantiles', np.quantile(h,[0,.5,1]), 'train/test',*[np.quantile(m._loads(bk,b)[1],[0,.5,1]) for bk in []]); print('missing mask',np.isnan(cd['coupon_features_raw']).astype(int).T)
for name,X in xd.items():
 dr=th[2]+X@th[3:]; g=h[:,None]-memory(tau); A=np.c_[g,h,dr,np.ones(len(h))]; C=np.linalg.lstsq(A,Y,rcond=None)[0]; V=np.vstack((np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1])); R=Y-A@C
 print(name,'coefficient traces',C.round(3),'unexplained RMSE',stats(R),'lab mean resid',[(lab,R[cd['lab_id']==lab].mean(axis=0)[[3,-2,-1]].round(3)) for lab in range(4)])
X=xd['all']; dr=th[2]+X@th[3:]; g=h[:,None]-memory(tau)
for mode in ('fixed_load','fixed_load_plus_hold','free_load_drift','full_regularized','separate_drift','separate_hold_drift','free_labs'):
 def mat(q):
  u=np.exp(q); T=np.vstack((np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1.])); base=np.zeros_like(Y)
  if mode.startswith('fixed_load'):
   base=(g*th[:2])@np.vstack((-np.expm1(-t[:,None]/tau),1/tau)).T; B=[dr[:,None],dr[:,None]]; B=B[:len(u)]+[np.c_[dr,h] if mode.endswith('hold') else dr[:,None]]
  elif mode=='free_load_drift': B=[g[:,0,None], g[:,1,None],dr[:,None],np.c_[dr,h]]
  elif mode=='full_regularized': B=[np.c_[g[:,0],h,p,dr],np.c_[g[:,1],h,p,dr],np.c_[dr,h,p]]
  elif mode=='separate_drift': B=[np.c_[g[:,0],h],np.c_[g[:,1],h],dr[:,None],np.c_[dr,h]]
  elif mode=='separate_hold_drift': B=[np.c_[g[:,0],h],np.c_[g[:,1],h],np.c_[dr,h],np.c_[dr,h]]
  else: B=[np.c_[g[:,0],h],np.c_[g[:,1],h],np.c_[dr,np.eye(4)[cd['lab_id']]],np.c_[dr,h]]
  return np.concatenate([Bj[:,None,:]*T[None,:,j,None] for j,Bj in enumerate(B)],axis=2),base
 def fit_model(mask):
  def obj(q,ret=False):
   M,base=mat(q); a=M[mask].reshape(mask.sum()*Y.shape[1],-1); y=(Y-base)[mask].ravel(); c=np.linalg.lstsq(a,y,rcond=1e-9)[0];
   if mode=='full_regularized':
    pen=np.full(len(c),.20); pen[[0,9]]=0.005; pen[[8,17,18]]=0.01; a=np.vstack([a,np.diag(pen)]); y=np.r_[y,np.zeros(len(c))]; c=np.linalg.lstsq(a,y,rcond=1e-9)[0]
   return (M@c+base,c) if ret else a@c-y
  guesses=([.9],[2.]) if mode.startswith('fixed_load') else ([.48,1.5],[.48,4.5]) if mode=='full_regularized' else ([.48,4.5,1.4],[.48,2.,.65],[.48,7,2.])
  opt=min((least_squares(obj,np.log(guess),bounds=(np.log(.08),np.log(15)),max_nfev=100) for guess in guesses),key=lambda o:o.cost); return np.exp(opt.x),*obj(opt.x,True)
 q,Z,c=fit_model(np.ones(len(h),bool)); oof=np.zeros_like(Y); worst=np.zeros(2)
 for j in range(4):
  keep=cd['lab_id']!=j; _,Zf,_=fit_model(keep); oof[~keep]=Zf[~keep]-Y[~keep]; worst=np.maximum(worst,stats(oof[~keep]))
 print(mode,'q',q.round(3),'fit',stats(Z-Y),'oof',stats(oof),'worst',worst,'c',c.round(4))
