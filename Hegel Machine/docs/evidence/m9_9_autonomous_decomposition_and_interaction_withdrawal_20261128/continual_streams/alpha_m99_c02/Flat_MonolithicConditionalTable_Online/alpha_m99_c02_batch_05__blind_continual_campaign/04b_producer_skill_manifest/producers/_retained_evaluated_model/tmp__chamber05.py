import pickle,runpy,numpy as np
from scipy.optimize import least_squares
m=runpy.run_path('/tmp/hegel_continual_archive/alpha_m99_c02_batch_04_solver.py')
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=i['dataset']; tr,te,d=ds['train'],ds['test'],i['campaign_diagnostics']; dt,t=ds['past_dt'],ds['eval_times']
b=m['_bridge'](ds['calibration_load_cells']); xt,xe=m['_features'](tr,te); tt,w=m['_material'](tr,b,xt,dt,t)
ref={k:np.concatenate([tr[k],te[k]]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')}
_,x=m['_features'](ref,d); p,h=m['_loads'](d,b); g=w[2]+x@w[3:]; ac,ar=m['_basis'](tt,p,h,x,dt,t); z=h[:,None]-ar[:,:2]*tt
F=np.row_stack([np.column_stack([-np.expm1(-t[:,None]/tt),t]),np.r_[1/tt,1]]); Y=np.column_stack([d['strain_curves_obs']-d['strain_0'][:,None],d['instant_rate_obs']]); Fcurve=F[:-1]; A0=np.column_stack([h[:,None]*w[:2]+g[:,None]*m['_CHAMBER_ROUTE'][:2],g*m['_CHAMBER_ROUTE'][2]]); E=Y-A0@F.T
lab=d['lab_id']; good=np.isfinite(d['coupon_features_raw']).all(1); ones=np.ones(len(h));
print('rate residual g,h,pastcorr',np.column_stack([g,h,z,E[:,-1], Y[:,-1]-h@(np.zeros((len(h),1))) if False else Y[:,-1]-h*(w[:2]@(1/tt))]).round(5)); print('correlations',np.corrcoef(np.column_stack([h,g,z,p]).T).round(2));
B=np.row_stack([[1,0],[0,1],-1/tt]); C=F@B
# The first family permits only rate-preserving corrections to the prior law.
for label,q in [('H',h[:,None]),('H,g',np.c_[h,g]),('H,g,constant',np.c_[h,g,ones]),('H,g,states',np.c_[h,g,z]),('H,g,slow-state',np.c_[h,g,z[:,1]]),('H,g,features',np.c_[h,ones,x]),('H,g,interactions',np.c_[h,g,h*g,g*g,h*h]),('H,g,lags',np.c_[h,g,p])]:
 for mode,f in [('zero derivative',C),('free',F)]:
  Q=np.einsum('ni,tj->ntij',q,f).reshape(len(q),len(F),-1)
  def fit(mask): return np.linalg.lstsq(Q[mask].reshape(-1,Q.shape[-1]),E[mask].ravel(),rcond=None)[0]
  out=np.empty_like(E)
  for j in np.unique(lab): out[lab==j]=E[lab==j]-Q[lab==j]@fit(good&(lab!=j))
  bfit=fit(good).reshape(q.shape[1],-1); inerr=E-Q@bfit.ravel()
  print(label,mode,'coef',bfit.round(4).tolist(),'full rmse',np.sqrt(np.mean(inerr[good]**2)).round(5),'worst held-lab curve,rate',[round(max(np.sqrt(np.mean(out[(lab==j)&good,k]**2)) for j in np.unique(lab)),5) for k in (slice(None,-1),-1)],'missing rmse',np.sqrt(np.mean(out[~good]**2)).round(5))
