import pickle,runpy,numpy as np
from scipy.optimize import least_squares
m=runpy.run_path('/tmp/hegel_continual_archive/alpha_m99_c02_batch_04_solver.py')
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=i['dataset']; tr,te,d=ds['train'],ds['test'],i['campaign_diagnostics']; dt,t=ds['past_dt'],ds['eval_times']
b=m['_bridge'](ds['calibration_load_cells']); xt,xe=m['_features'](tr,te); tt,w=m['_material'](tr,b,xt,dt,t)
ref={k:np.concatenate([tr[k],te[k]]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')}
_,x=m['_features'](ref,d); p,h=m['_loads'](d,b); g=w[2]+x@w[3:]; ac,ar=m['_basis'](tt,p,h,x,dt,t); z=h[:,None]-ar[:,:2]*tt
F=np.row_stack([np.column_stack([-np.expm1(-t[:,None]/tt),t]),np.r_[1/tt,1]]); Y=np.column_stack([d['strain_curves_obs']-d['strain_0'][:,None],d['instant_rate_obs']]); Fcurve=F[:-1]; A0=np.column_stack([h[:,None]*w[:2]+g[:,None]*m['_CHAMBER_ROUTE'][:2],g*m['_CHAMBER_ROUTE'][2]]); E=Y-A0@F.T
lab=d['lab_id']; good=np.isfinite(d['coupon_features_raw']).all(1); ones=np.ones(len(h));
from itertools import combinations
print('feature implementation:'); import inspect; print(inspect.getsource(m['_features']))
_, xd=m['_features'](tr,d); _, xs=m['_features'](d,d); gs=[g,w[2]+xd@w[3:],w[2]+xs@w[3:]]
init=Y[:,-1]-h*(w[:2]@(1/tt)); base=h[:,None]*(w[:2]@F[:,:2].T); diff=Y-base; amps=np.linalg.lstsq(F,diff.T,rcond=None)[0].T
for name,gd in zip(['train,test,diagnostics','train,diagnostics','diagnostics'],gs):
 for label,q in [('g',gd[:,None]),('g+features',np.c_[ones,x if gd is gs[0] else xd if gd is gs[1] else xs]),('g+hold',np.c_[gd,h])]:
  fit=lambda mask: least_squares(lambda v:(q@v-init)[mask],np.linalg.lstsq(q[mask],init[mask],rcond=None)[0],loss='soft_l1',f_scale=.004).x
  err=init-q@fit(good); out=np.zeros(len(h));
  for j in np.unique(lab): out[lab==j]=init[lab==j]-q[lab==j]@fit(good&(lab!=j))
  print(name,label,'coef',fit(good).round(5),'in med/max',np.percentile(abs(err[good]),[50,100]).round(5),'oof med/max',np.percentile(abs(out[good]),[50,100]).round(5))
print('row,lab,good, h, g, inferred g, free delta amplitudes, state'); print(np.c_[np.arange(len(h)),lab,good,h,g,init,amps,z].round(5))
for label,q in [('init',init[:,None]),('init,h',np.c_[init,h]),('init,h,z',np.c_[init,h,z]),('init,h,z,interaction',np.c_[init,h,z,init*z[:,0],init*z[:,1],init*h]),('init,h,past',np.c_[init,h,p]),('init,h,constant',np.c_[init,h,ones])]:
 Q=np.einsum('ni,tj->ntij',q,C).reshape(len(h),len(F),-1); target=diff-init[:,None]*F[:,2];
 def fit(mask):
  a=Q[mask].reshape(-1,Q.shape[-1]); y=target[mask].ravel(); return least_squares(lambda v:a@v-y,np.linalg.lstsq(a,y,rcond=None)[0],loss='soft_l1',f_scale=.004).x
 out=np.zeros_like(target);
 for j in np.unique(lab): out[lab==j]=target[lab==j]-Q[lab==j]@fit(lab!=j)
 print(label,'coeff',fit(np.ones(len(h),bool)).reshape(-1,2).round(4),'worst lab',max(np.mean(abs(out[lab==j,:-1])) for j in np.unique(lab)),'row RMSE',np.sqrt(np.mean(out[:,:-1]**2,1)).round(4))
