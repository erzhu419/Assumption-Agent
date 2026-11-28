import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_alpha_m100_02_batch_06_solver as m
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=item['dataset']; tr,te=ds['train'],ds['test']; dg=item['campaign_diagnostics']; t,dt=ds['eval_times'],ds['past_dt']
xtr,xte=m._features(tr,te)
features=('coupon_features_raw','ref_coupons','lab_id','stratum')
pool={k:np.concatenate([tr[k],te[k]]) for k in features}; _,xd=m._features(pool,dg)
cal=ds['calibration_load_cells']; c=m._bridge(cal)
a=cal['true_channel_loads']; y=cal['bridge_raw']; keep=np.max(np.abs(y),axis=1)<5
print('cal',c,'calibration load absolute residual quantiles',np.quantile(np.abs(a[keep]-m._loads(y[keep],c)),[.5,.9,1]))
def obs(p): return p['strain_curves_obs']-p['strain_0'][:,None],p['instant_rate_obs']
s=m._prepare(tr,xtr,c); tau,theta=m._fit(s,dt,t,*obs(tr)); print('tau,theta:',tau,theta)
for k,p,x in [('train',tr,xtr),('diag',dg,xd)]:
 s=m._prepare(p,x,c); u,v=m._design(np.log(tau),s,dt,t); Y,R=obs(p); hold=s[1]; drift=s[2]@theta[2:]; Z=hold[:,None]-v[:,:2]*tau
 def err(C,S):
  return np.round(np.quantile(np.abs(C-Y),[.5,.9,1]),4),np.round(np.quantile(np.abs(S-R),[.5,.9,1]),4)
 print(k,'base',err(u@theta,v@theta),'reset',err(*m._chamber(hold,drift,tau,theta,t)))
 growth=-np.expm1(-t[:,None]/tau); B=np.column_stack((growth,t)); A=np.linalg.lstsq(B,Y.T,rcond=None)[0].T; S=A@np.r_[1/tau,1.]; print('free curve',err(A@B.T,S))
 D=np.column_stack([hold,Z,drift,np.ones(len(hold))]); coef=np.linalg.lstsq(D,A,rcond=None)[0]; print('amp vs [hold,z0,z1,drift,1]',coef,'max amp err',np.max(abs(D@coef-A),axis=0)); print('rate fit',np.linalg.lstsq(D,R,rcond=None)[0]); print('ranges H/Z/drift',D.min(0),D.max(0)); print('sample [A, obsR, freeR]',np.column_stack([A,R,S])[:8])
 if k=='diag':
  def r(logtau):
   tt=np.exp(logtau); bb=np.row_stack((np.column_stack((-np.expm1(-t[:,None]/tt),t)),np.r_[1/tt,1.])); yy=np.column_stack((Y,R)); return (np.linalg.lstsq(bb,yy.T,rcond=None)[0].T@bb.T-yy).ravel()
  for seed in [tau,tau*[.5,.5],tau*[2,2],tau*[1,3]]:
   fit=least_squares(r,np.log(seed),bounds=(np.log(.02),np.log(100))); print('free tau',np.sort(np.exp(fit.x)),np.quantile(abs(fit.fun),[.5,.9,1]))
