import importlib.util,pickle,numpy as np
p='/tmp/hegel_continual_archive/dev_alpha_m98_00_batch_01_solver.py'
s=importlib.util.spec_from_file_location('s',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; q=i['campaign_diagnostics']; print('diagnostics',[(k,np.shape(v)) for k,v in q.items()]); dt,t=d['past_dt'],d['eval_times']; tr=d['train']; a,b=m._prepare(d); tau,th=m._fit(dt,t,a,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,th)
coef=m._stress_calibration(d['calibration_load_cells']); stress=lambda v:v@coef[:4]+coef[4]
print('cal residual',np.sqrt(np.mean((stress(d['calibration_load_cells']['bridge_raw'])-d['calibration_load_cells']['true_channel_loads'].sum(-1))**2)))
x=m._aligned_features(tr,q)[len(tr['lab_id']):]; past=stress(q['past_channel_raw']); hold=stress(q['hold_channel_raw']); z=np.zeros((len(hold),2))
for alpha,st in zip(-np.expm1(-dt[:,None]/tau),past.T): z+=alpha*(st[:,None]-z)
y=q['strain_curves_obs']-q['strain_0'][:,None]; r=q['instant_rate_obs']; drift=x@th[2:]; A,R=m._design(tau,dt,t,(past,hold,x)); C,S=m._chamber_response(tau,t,hold,x,th)
for name,pc,pr in [('bench',A@th,R@th),('reset',C,S)]: print(name,'curve/rate rmse',np.mean((pc-y)**2)**.5,np.mean((pr-r)**2)**.5)
g=np.c_[-np.expm1(-t[:,None]/tau),t]; rates=np.r_[1/tau,1.]; amps=np.c_[y,r]@np.linalg.pinv(np.vstack([g,rates]).T); print('fixed basis rmse',np.mean((amps@g.T-y)**2)**.5,np.mean((amps@rates-r)**2)**.5)
from scipy.optimize import least_squares
n=len(hold); xx=m._aligned_features({k:v[:0] for k,v in q.items()},q); dd=xx@th[2:]; gap=(hold[:,None]-z)*th[:2]; obs=np.c_[y,r]; G=np.vstack([g,rates]); residual=obs-gap@G[:,:2].T
print('lab counts',np.unique(q['lab_id'],return_counts=True),'per-sample range',dd.min(),dd.max())
models={'independent':np.eye(3),'tied ratios':np.array([[1.],[5.],[-.5]])}
for name, map_ in models.items():
 F=dd[:,None,None]*(G@map_)[None,:,:]
 for part,labels in [('coupon',np.arange(n)),('lab',q['lab_id']),('stratum',q['stratum'])]:
  pred=np.empty_like(obs)
  for j in np.unique(labels):
   take=labels!=j; beta=np.linalg.lstsq(F[take].reshape(-1,F.shape[-1]),residual[take].ravel(),rcond=None)[0]; pred[~take]=F[~take]@beta
  e=pred-residual; print(name,part,'curve/rate/worst',np.round([np.mean(e[:,:-1]**2)**.5,np.mean(e[:,-1]**2)**.5,np.max(np.mean(e**2,axis=1)**.5)],6))
 beta=np.linalg.lstsq(F.reshape(-1,F.shape[-1]),residual.ravel(),rcond=None)[0]; print('coefficients',map_@beta)
 for wt in (1,3,8):
  W=np.r_[np.ones(len(t)),wt]; bh=least_squares(lambda b:((F@b-residual)*W).ravel(),beta,loss='soft_l1',f_scale=.01).x; print('robust rate weight',wt,'coefs',map_@bh)
print('train/test lab counts',np.unique(tr['lab_id'],return_counts=True),np.unique(d['test']['lab_id'],return_counts=True))
