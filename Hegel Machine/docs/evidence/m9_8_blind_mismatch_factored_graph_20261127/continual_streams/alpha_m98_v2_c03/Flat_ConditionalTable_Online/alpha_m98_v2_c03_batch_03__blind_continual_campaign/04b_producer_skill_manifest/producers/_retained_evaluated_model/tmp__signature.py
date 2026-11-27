import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('base','/tmp/hegel_continual_archive/alpha_m98_v2_c03_batch_02_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; c=i['campaign_diagnostics']; tr=d['train']; dt,t=d['past_dt'],d['eval_times']; train,diag=m._prepare(dict(d,test=c)); y=np.column_stack([tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]); tau,theta=m._fit(train,dt,t,y)
yc=np.column_stack([c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs']]); pred=m._design(diag,dt,t,tau)@theta; r=yc-pred; past,h,f=diag; drift=f@theta[2:]; g=m._design(diag,dt,t,tau)[:,-1,:2]*tau; z=h[:,None]-g
rm=lambda a:np.sqrt(np.mean(a*a,axis=0))
from scipy.optimize import least_squares
import json
np.set_printoptions(precision=4,suppress=True,linewidth=140)
archive=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for a in archive:
 print('apparatus',[(k,str(v)[:400]) for k,v in a.items() if 'trace' not in k]);
E=np.vstack([-np.expm1(-t[:,None]/tau),1/tau]); T=np.r_[t,1.]; base=pred-drift[:,None]*T
C=yc-base
F=f[:,1:]; lab=c['lab_id']; good=~np.isnan(c['coupon_features_raw']).any(axis=1); print('time',T); print('residual rank',np.linalg.svd(C,compute_uv=False)); print('rows of C',np.c_[lab,good,drift,C[:,[0,2,5,8,11,12]]]);
for subset in [np.ones(len(h),bool),good]:
 u,S,v=np.linalg.svd(C[subset],full_matrices=False); print('subset size',subset.sum(),'singular values',S,'leading signatures'); print(v[:3]/v[:3,-1:]);
 for label,A in [('drift + constant',np.c_[np.ones(len(h)),drift]),('features',f),('lab-specific drift and intercept',np.c_[f,lab[:,None]==np.unique(lab)[1:],drift[:,None]*(lab[:,None]==np.unique(lab)[1:])]),('lab-specific intercept',np.c_[f,lab[:,None]==np.unique(lab)[1:]])]:
  p=np.linalg.lstsq(A[subset],C[subset],rcond=None)[0]; print(label,'fit rmse',np.sqrt(np.mean((C[subset]-A[subset]@p)**2)),'coeffs at start/middle/end/rate',p[:,[0,5,11,12]])
  if label=='features': print('residual RMS of individual rows',np.sqrt(np.mean((C-A@p)**2,axis=1)))
