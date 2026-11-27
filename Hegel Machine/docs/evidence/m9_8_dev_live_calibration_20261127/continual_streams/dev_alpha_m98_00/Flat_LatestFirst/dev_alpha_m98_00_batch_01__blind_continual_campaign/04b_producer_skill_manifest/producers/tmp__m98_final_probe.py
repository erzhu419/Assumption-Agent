import importlib.util, pickle, numpy as np
path='/holdout_task/batch_solver.py'
s=importlib.util.spec_from_file_location('solver',path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; g=i['campaign_diagnostics']; out=m.solve_batch(d); ts=out['estimated_taus']; w=out['hereditary_weights']; t=d['eval_times']; cal=m._stress_calibration(d['calibration_load_cells']); h=g['hold_channel_raw']@cal[:-1]+cal[-1]
keys=('lab_id','stratum','coupon_features_raw','ref_coupons'); complete=np.isfinite(g['coupon_features_raw']).all(1); y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; q=np.r_[1/ts,1.]; labs=g['lab_id']; strata=g['stratum']; n=len(labs)
def score(e): return np.sqrt(np.mean(e[:,:-1]**2)), np.sqrt(np.mean(e[:,-1]**2))
def data():
 x=m._aligned_features({k:g[k][:0] for k in keys},g); C,R=m._chamber_design(ts,t,h,x@w[2:]); D=np.concatenate((C,R[:,None,:]),axis=1); return D[:,:,[1,3,4]],D[:,:,[0,2]]@w[:2]
A,base=data(); y0=y-base
splits=[labs==v for v in np.unique(labs)]+[strata==v for v in np.unique(strata)]+[~complete]
for label in ('all','complete','unit_rate'):
 def fit(mask):
  if label=='complete': mask=mask & complete
  F=A[mask].reshape(-1,3); z=y0[mask].ravel()
  if label=='unit_rate':
   c=np.linalg.lstsq(F[:,:2]-F[:,2,None]*q[:2],z-F[:,2],rcond=None)[0]; return np.r_[c,1-c@q[:2]]
  return np.linalg.lstsq(F,z,rcond=None)[0]
 eo=[score((A@fit(~mask)-y0)[mask]) for mask in splits]; c=fit(np.ones(n,bool)); print(label,'coeff',c,'initial drift gain',c@q,'pooled',score(A@c-y0),'worst blocked',np.max(eo,axis=0))
c=np.array([.3452,1.6072,.0532]); e=A@c-y0; print('final artifact public chamber RMSE',score(e),'worst group',np.max([score(e[mask]) for mask in splits],axis=0)); print('final artifact horizon max RMSE',np.sqrt(np.mean(e[:,:-1]**2,axis=0)).max()); print('finite deploy outputs:',all(np.isfinite(v).all() for v in out.values())); assert all(np.isfinite(v).all() for v in out.values()) and .05<ts[0]<ts[1]<25
