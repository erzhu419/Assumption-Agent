import runpy,pickle,numpy as np
from scipy.optimize import least_squares
m=runpy.run_path('/tmp/hegel_continual_archive/alpha_m100_c01_batch_04_solver.py'); globals().update({k:v for k,v in m.items() if k.startswith('_') and not k.startswith('__')})
b=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=b['dataset']; tr,te,g=d['train'],d['test'],b['campaign_diagnostics']; c=d['calibration_load_cells']; B,keep=_fit_bridge(c['bridge_raw'],c['true_channel_loads']); dt,t=d['past_dt'],d['eval_times']; xtr,xte=_features(tr,te); tau,th=_fit(*_calibrated_inputs(tr,xtr,B),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); p,H,F=_calibrated_inputs(g,_features(tr,g)[1],B); D=F@th[2:]; R=np.r_[-np.expm1(-t[:,None]/tau),[1/tau]]; Y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; base=(H[:,None]*th[:2])@R.T; W=np.r_[np.ones(len(t)),3.]; labs=g['lab_id']; allrows=np.ones(len(H),bool)
ramps={'exp':lambda u:-u*np.expm1(-t/u),'log':lambda u:u*np.log1p(t/u),'rational':lambda u:t/(1+t/u),'sqrt':lambda u:2*t/(1+np.sqrt(1+t/u)),'quadratic':lambda u:t-t*t/u,'projection':lambda u:R[:-1]@np.array([u,tau[1]*(1-u/tau[0])])}
for name, ramp in ramps.items():
 for cov in ['fixed','scale','features','hold']:
  def pred(v):
   g1=np.r_[ramp(np.exp(v[0]) if name!='projection' else v[0]),1.]; drift=D if cov=='fixed' else D*v[1] if cov in ('scale','hold') else F@v[1:]; return base+np.outer(drift,g1)+(np.outer(H,R[:,1])*v[2] if cov=='hold' else 0.)
  init=np.r_[np.log(4.) if name!='projection' else 0., [] if cov=='fixed' else [1.] if cov=='scale' else th[2:] if cov=='features' else [1.,0.]]
  def fit(s): return least_squares(lambda v: ((pred(v)-Y)[s]*W).ravel(),init,loss='soft_l1',f_scale=.016,max_nfev=100).x
  e=np.zeros_like(Y); v=fit(allrows); pairs=[]
  for lab in np.unique(labs):
   s=labs==lab; e[s]=(pred(fit(~s))-Y)[s]; pairs.append(np.median(abs(e[s]),axis=0))
  print(name,cov,'p',v.round(5),'OOF med/worst lab',np.round([np.median(abs(e[:,:-1])),np.median(abs(e[:,-1])),max(np.median(abs(e[labs==l,:-1])) for l in np.unique(labs)),max(np.median(abs(e[labs==l,-1])) for l in np.unique(labs))],4),'time bias',np.round(np.median(e,axis=0)[[0,3,7,11,12]],4))
print('missing diag',np.isnan(g['coupon_features_raw']).sum(axis=0)); print('rate reset residual vs drift',np.c_[labs,D,Y[:,-1]-base[:,-1]].round(3))
