import sys,pickle,numpy as np,importlib.util
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
p=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m99_c04_batch_02_solver.py'); prior=importlib.util.module_from_spec(p);p.loader.exec_module(prior)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=calibrate_upstream({**i['dataset'],'chamber_verification_trials':i['campaign_diagnostics']}); tr,te,g=[d[k] for k in ('train','test','chamber_verification_trials')]; t,dt=d['eval_times'],d['past_dt']; n=len(g['lab_id'])
x,z=prior._features(tr,te); _,xg=prior._features(tr,g); taus,theta=prior._fit(tr,x,dt,t); age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0.]; P=g['past_channel_raw'].sum(axis=-1); H=g['hold_channel_raw'].sum(axis=-1); Z=P@(-np.expm1(-dt[:,None]/taus)*np.exp(-age[:,None]/taus)); drift=theta[2]+xg@theta[3:]; Y=g['strain_curves_obs']-g['strain_0'][:,None]; R=g['instant_rate_obs'];
if __name__=='__main__':
 print('taus/theta',taus,theta)
 c,r=prior._design(tr,x,dt,t,taus); print('train absolute residual quantiles',np.quantile(abs(c@theta-(tr['strain_curves_obs']-tr['strain_0'][:,None])),[.25,.5,.8,.95,.99]),np.quantile(abs(r@theta-tr['instant_rate_obs']),[.25,.5,.8,.95,.99]))
 c,r=prior._design(g,xg,dt,t,taus); q,s=prior._chamber_response(g,xg,dt,t,taus,theta)
 for name,C,S in [('bench',c@theta,r@theta),('batch02',q,s)]: print(name,'curve/rate MAE',np.mean(abs(C-Y)),np.mean(abs(S-R)),'rate residual',np.round(R-S,3))
 print('singular values',np.linalg.svd(Y,compute_uv=False)); print('diagnostic examples [hold,hist0,hist1,drift,rate,curve ...]',np.round(np.c_[H,Z,drift,R,Y[:,[0,3,7,11]]],3));
 for linear in [False,True]:
  def temporal(p):
   poles=np.exp(p); C=-np.expm1(-t[:,None]/poles)*poles; S=np.ones((1,len(p))); return np.r_[np.c_[C,t] if linear else C,np.c_[S,[1.]] if linear else S]
  for use_rate in [False,True]:
   obs=np.c_[Y,R] if use_rate else Y
   def profile(p):
    A=temporal(p) if use_rate else temporal(p)[:-1]; B=np.linalg.lstsq(A,obs.T,rcond=None)[0]; return obs-B.T@A.T
   f=least_squares(lambda p:profile(p).ravel(),np.log(taus),bounds=(np.log(.04),np.log(35.))); E=profile(f.x);print('free amplitudes linear/rate',linear,use_rate, 'poles',np.exp(f.x),'MAE',abs(E).mean(),'median',np.median(abs(E)))
