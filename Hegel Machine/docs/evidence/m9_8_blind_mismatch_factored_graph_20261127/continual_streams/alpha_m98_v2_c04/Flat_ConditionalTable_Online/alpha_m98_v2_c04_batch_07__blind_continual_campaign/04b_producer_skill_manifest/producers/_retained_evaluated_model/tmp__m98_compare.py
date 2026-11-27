import importlib.util, pathlib, pickle, numpy as np
from scipy.optimize import least_squares
p=pathlib.Path('/tmp/hegel_continual_archive'); mods={}
for f in p.glob('*solver.py'):
 s=importlib.util.spec_from_file_location(f.stem,f); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); mods[f.stem.rsplit('_solver',1)[0].split('c04_')[1]]=m
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=it['dataset']; tr,te,q=d['train'],d['test'],it['campaign_diagnostics']; t,dt=d['eval_times'],d['past_dt']; n=len(tr['lab_id']); nq=len(q['lab_id']); yc=q['strain_curves_obs']-q['strain_0'][:,None]; yr=q['instant_rate_obs']; labs=q['lab_id']
def score(c,r):
 a=c-yc; b=r-yr; return np.round([np.median(abs(a)),np.median(abs(b)),max(np.median(abs(a[labs==l])) for l in np.unique(labs)),max(np.median(abs(b[labs==l])) for l in np.unique(labs))],6)
for name,m in sorted(mods.items()):
 if hasattr(m,'_CHAMBER_TIMES'): print(name,'archived chamber',m._CHAMBER_TIMES,m._CHAMBER_COEFFICIENTS)
 b=m._bridge(d['calibration_load_cells']); x=m._features(tr,te); tau,th=m._fit(x[:n],*m._loads(tr,b),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); xq=m._features({k:np.concatenate((tr[k],te[k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')},q)[-nq:]; past,hold=m._loads(q,b); c,r=m._design(xq,past,hold,dt,t,tau); print(name,'intrinsic',tau.round(5),th.round(5),'baseline',score((c@th).reshape(nq,-1),r@th))
 if hasattr(m,'_CHAMBER_TIMES'):
  T=m._CHAMBER_TIMES; F=np.column_stack((hold,th[2]+xq@th[3:])); A=F@m._CHAMBER_COEFFICIENTS.T; print(name,'archived chamber residual',score(A@np.vstack((-np.expm1(-t[None,:]/T[:,None]),t)),A@np.r_[1/T,1]))
 if name=='batch_06':
  np.savez('/tmp/m98_arrays.npz',t=t,dt=dt,tau=tau,th=th,x=xq,past=past,hold=hold,yc=yc,yr=yr,labs=labs)
