import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
p='/holdout_task/'; s=importlib.util.spec_from_file_location('m',p+'alpha_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open(p+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d,g=i['dataset'],i['gt']; tr,te=d['train'],d['test']; x,xt=m._align(tr,te); cal=m._calibrate(d['calibration_load_cells']); dt,t=d['past_dt'],d['eval_times']; args=cal,dt,t,d['crit_stress'],d['shear_banding_index']; build=m._design(tr,x,*args); test=m._design(te,xt,*args); yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; lab=tr['lab_id']; good=np.isfinite(tr['coupon_features_raw']).all(1)
rm=lambda v: np.sqrt(np.mean(v*v))
for kind in ('baseline','cauchy'):
 def fit(use):
  b=lambda q: tuple(v[use] for v in build(q)); tau,beta=m._fit(b,yc[use],yr[use],dt,t,good[use]); q=np.r_[np.log(tau),beta]
  if kind=='baseline': return q
  sub=use & good
  def residual(q):
   c,r=build(q[:2]); return (c[sub]@q[2:]-yc[sub]).ravel(), r[sub]@q[2:]-yr[sub]
  scale=[max(1.4826*np.median(abs(e-np.median(e))),1e-6) for e in residual(q)]
  new=least_squares(lambda v: np.concatenate([e/a for e,a in zip(residual(v),scale)]),q,loss='cauchy',x_scale='jac').x
  return new if new[0]<new[1] else q
 print(kind)
 for j in range(4):
  q=fit(lab!=j); c,r=build(q[:2]); u=(lab==j)&good; ec=c[u]@q[2:]-yc[u]; er=r[u]@q[2:]-yr[u]; print('held lab',j,'curve median abs / rate RMSE:',np.median(abs(ec)),rm(er),'tau',np.exp(q[:2]))
 q=fit(np.ones(len(lab),bool)); c,r=test(q[:2]); ec=c@q[2:]+te['strain_0'][:,None]-g['te_clean']; er=r@q[2:]-g['te_true_rate0']; missing=~np.isfinite(te['coupon_features_raw']).all(1); print('tau',np.exp(q[:2]),'max coefficient error',np.max(abs(q[2:]-np.r_[g['theta'],g['vp_theta']])),'complete/incomplete strain and rate RMSE',[(rm(ec[u]),rm(er[u])) for u in (~missing,missing)])
