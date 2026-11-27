import pickle,runpy,numpy as np
from scipy.optimize import least_squares
m=runpy.run_path('/tmp/hegel_continual_archive/alpha_m99_c02_batch_04_solver.py')
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=i['dataset']; tr,te,d=ds['train'],ds['test'],i['campaign_diagnostics']; dt,t=ds['past_dt'],ds['eval_times']
b=m['_bridge'](ds['calibration_load_cells']); xt,xe=m['_features'](tr,te); tt,w=m['_material'](tr,b,xt,dt,t)
ref={k:np.concatenate([tr[k],te[k]]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')}
_,x=m['_features'](ref,d); p,h=m['_loads'](d,b); g=w[2]+x@w[3:]; ac,ar=m['_basis'](tt,p,h,x,dt,t); z=h[:,None]-ar[:,:2]*tt
F=np.row_stack([np.column_stack([-np.expm1(-t[:,None]/tt),t]),np.r_[1/tt,1]]); Y=np.column_stack([d['strain_curves_obs']-d['strain_0'][:,None],d['instant_rate_obs']]); Fcurve=F[:-1]; A0=np.column_stack([h[:,None]*w[:2]+g[:,None]*m['_CHAMBER_ROUTE'][:2],g*m['_CHAMBER_ROUTE'][2]]); E=Y-A0@F.T
lab=d['lab_id']; good=np.isfinite(d['coupon_features_raw']).all(1); ones=np.ones(len(h));
_, xd=m['_features'](d,d)
gd=w[2]+xd@w[3:]
base=h[:,None]*(w[:2]@F[:,:2].T)
obs=Y-base
# The initial derivative of the three redistributed components must sum to 1.
P=F[:,:2]-F[:,2,None]/tt
for label,f in [('conserved',P),('free',F)]:
 for mask_label,valid in [('all',np.ones(len(h),bool)),('complete',good)]:
  Q=gd[:,None,None]*f[None,:,:]
  T=obs-gd[:,None]*F[:,2] if label=='conserved' else obs
  def fit(mask):
   a=Q[mask].reshape(-1,f.shape[1]); y=T[mask].ravel()
   return np.linalg.lstsq(a,y,rcond=None)[0]
  v=fit(valid); out=np.zeros_like(T)
  for j in np.unique(lab):
   out[lab==j]=T[lab==j]-Q[lab==j]@fit(valid&(lab!=j))
  print(label,mask_label,'route',np.r_[v,1-v@(1/tt)] if label=='conserved' else v)
  print('held-lab curve/rate RMSE',[(int(j),*(np.sqrt(np.mean(out[lab==j,k]**2)) for k in (slice(None,-1),-1))) for j in np.unique(lab)],'missing',np.sqrt(np.mean(out[~good]**2)),'worst time',np.sqrt(np.mean(out**2,axis=0)).max())
for name,split,xx in [('train joint',tr,xt),('test joint',te,xe),('diag self',d,xd)]:
 print(name,'feature mean/std',np.mean(xx,0).round(4),np.std(xx,0).round(4),'missing',np.isnan(split['coupon_features_raw']).any(1).sum())
print('times',t,dt)
