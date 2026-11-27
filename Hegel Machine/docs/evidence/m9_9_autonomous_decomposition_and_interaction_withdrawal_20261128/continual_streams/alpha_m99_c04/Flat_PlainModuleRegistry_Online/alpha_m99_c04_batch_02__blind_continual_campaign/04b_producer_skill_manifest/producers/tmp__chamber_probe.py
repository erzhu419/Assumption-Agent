import pickle, sys
import numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m99_c04_s00_solver as m
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d=item['dataset']; q=item['campaign_diagnostics']; tr=d['train']; te=d['test']; t=d['eval_times']; dt=d['past_dt']; s=m._stress_map(d['calibration_load_cells'])
x,_=m._features(tr,te); tau,th=m._fit(tr,x,s,dt,t)
_,xq=m._features(tr,q); xlocal,_=m._features(q,q)
p=s[0]+q['past_channel_raw']@s[1:]; h=s[0]+q['hold_channel_raw']@s[1:]; age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0.]; z=p@(-np.expm1(-dt[:,None]/tau)*np.exp(-age[:,None]/tau)); gap=h[:,None]-z
F={'hold':h,'z0':z[:,0],'z1':z[:,1],'g0':gap[:,0],'g1':gap[:,1],'g0*g1':gap.prod(1),'g0-g1':gap[:,0]-gap[:,1],'lab':q['lab_id'],'stratum':q['stratum']}
for i in range(p.shape[1]): F[f'p{i}']=p[:,i]; F[f'h-p{i}']=h-p[:,i]
for i in range(4): F[f'x{i}']=xq[:,i]; F[f'rawx{i}']=np.nan_to_num(q['coupon_features_raw'][:,i])
for name,xx in [('train+diag',xq),('diag',xlocal)]:
 c,r=m._design(q,xx,s,dt,t,tau); E=q['strain_curves_obs']-q['strain_0'][:,None]-c@th; er=q['instant_rate_obs']-r@th
 def profile(tt,rate=False):
  A=np.column_stack([-np.expm1(-t[:,None]/np.atleast_1d(tt)),t]); Y=E
  if rate: A=np.vstack([A, np.r_[1/np.atleast_1d(tt),1.]*1.3]); Y=np.column_stack([Y,er*1.3]);
  B=np.linalg.lstsq(A,Y.T,rcond=None)[0].T; delta=Y-B@A.T; return np.sqrt(np.mean(delta**2)),B
 print(name,'baseline curve/rate rmse',np.sqrt(np.mean(E**2)),np.sqrt(np.mean(er**2)))
 for rate in (False,True):
  opt=minimize_scalar(lambda v:profile(np.exp(v),rate)[0],bounds=(np.log(.06),np.log(20)),method='bounded'); best=np.exp(opt.x); print('rate included',rate,'best tau',round(best,5),'rmse at best/fast/slow/both',[round(profile(a,rate)[0],6) for a in (best,tau[0],tau[1],tau)])
 if name!='train+diag': continue
 B=profile(tau[0],True)[1]; print('rows: exp amplitude, drift, hold, g0, g1, x0,x1,x2,x3');print(np.column_stack([B,h,gap,xq]).round(4));
 for k,b in enumerate(B.T):
  print('component',k,'correlations',sorted([(f,round(np.corrcoef(v,b)[0,1],3)) for f,v in F.items()],key=lambda a:-abs(a[1]))[:14]); M=np.column_stack([np.ones(len(b)),h,gap,p,xx]); coef=np.linalg.lstsq(M,b,rcond=None)[0]; print('linear rmse',np.sqrt(np.mean((M@coef-b)**2)))
