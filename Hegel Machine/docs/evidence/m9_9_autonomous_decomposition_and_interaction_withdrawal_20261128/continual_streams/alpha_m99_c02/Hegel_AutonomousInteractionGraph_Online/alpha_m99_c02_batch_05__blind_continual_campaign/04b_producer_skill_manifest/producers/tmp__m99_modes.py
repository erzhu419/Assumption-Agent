import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import *
item = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
ds = calibrate_upstream(dict(item['dataset'],chamber_verification_trials=item['campaign_diagnostics']))
tr,te,d = (ds[k] for k in ('train','test','chamber_verification_trials'))
dt,t=ds['past_dt'],ds['eval_times']
x,xt,xd=features(tr,te,d)
taus,theta=material(tr,x,dt,t)
print('dt,tau,theta',dt,taus,theta)
from scipy.optimize import least_squares
Y=np.c_[d['instant_rate_obs'],d['strain_curves_obs']-d['strain_0'][:,None]]
P,H=loads(d); D=theta[2]+xd@theta[3:]
def modes(q, linear):
 A=np.vstack((1/q,-np.expm1(-t[:,None]/q)))
 return np.c_[A,np.r_[1,t]] if linear else A
for n in (2,3,4):
 for linear in (False,True):
  def res(p):
   A=modes(np.exp(p),linear); return (Y-(Y@np.linalg.pinv(A.T))@A.T).ravel()
  f=least_squares(res,np.log(np.geomspace(.5,5,n)),bounds=(np.log(.05),np.log(25)),max_nfev=400)
  q=np.sort(np.exp(f.x)); A=modes(q,linear); amp=Y@np.linalg.pinv(A.T); E=Y-amp@A.T
  print('free modes',linear,np.round(q,5),'rms rate/curve',np.sqrt(np.mean(E[:,0]**2)),np.sqrt(np.mean(E[:,1:]**2)))
  if n in (2,3):
   G=gap_for(q,P,H,dt)
   print('corr amp vs gaps, hold, drift, past',np.round(np.corrcoef(np.c_[amp,G,H,D,P].T)[:amp.shape[1],amp.shape[1]:],3))
for driftmode in ('linear','filtered','mixed'):
 def fun(p):
  q=np.exp(p[:2]); g=gap_for(q,P,H,dt); A=modes(q,False)
  pred=(g*(theta[:2]*p[2:4]))@A.T
  v=np.r_[1,t]; vf=np.r_[1,-np.expm1(-t/np.exp(p[5]))*np.exp(p[5])]
  k={'linear':v,'filtered':vf,'mixed':p[6]*v+(1-p[6])*vf}[driftmode]
  return pred+D[:,None]*p[4]*k
 z=least_squares(lambda p:(fun(p)-Y).ravel(), np.r_[np.log(taus),1,1,1,0,.3],max_nfev=250).x
 print(driftmode,np.r_[np.exp(z[:2]),z[2:5],np.exp(z[5]),z[6]],'rmse',np.sqrt(np.mean((fun(z)-Y)**2,axis=0)))
