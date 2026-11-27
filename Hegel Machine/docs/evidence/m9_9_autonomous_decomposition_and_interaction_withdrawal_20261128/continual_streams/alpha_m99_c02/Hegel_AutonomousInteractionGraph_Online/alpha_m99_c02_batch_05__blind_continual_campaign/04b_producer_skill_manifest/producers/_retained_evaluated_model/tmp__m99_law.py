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
Y=np.c_[d['instant_rate_obs'],d['strain_curves_obs']-d['strain_0'][:,None]]
P,H=loads(d); D=theta[2]+xd@theta[3:]; G=gap_for(taus,P,H,dt)
A=np.c_[np.r_[1/taus[0],-np.expm1(-t/taus[0])],np.r_[1/taus[1],-np.expm1(-t/taus[1])],np.r_[1,t]]
amp=Y@np.linalg.pinv(A.T)
print('amp[:,0:2] / theta, H, Hist, D, drift-mode'); print(np.round(np.c_[amp[:,:2]/theta[:2],H,H[:,None]-G,D,amp[:,2]],4))
for j in range(3):
 for name,Z in [('own',np.c_[G[:,min(j,1)],D]),('hold-history',np.c_[H,H[:,None]-G,D]),('stress_steps',np.c_[H,P,D]),('drift-features',np.c_[np.ones(len(H)),xd])]:
  b=np.linalg.lstsq(Z,amp[:,j],rcond=None)[0]; C=Z@b-amp[:,j]
  print(j,name,'coef',np.round(b,4),'rmse',np.sqrt(np.mean(C*C)))
# Estimate each input's contribution to the entire trace jointly, not from ill-conditioned per-coupon modal amplitudes.
for name,Zs in [('hold+history',[np.c_[H,H[:,None]-G]]*2+[np.c_[np.ones(len(H)),xd]]),('steps',[np.c_[H,P]]*2+[np.c_[np.ones(len(H)),xd]]),('simple',[np.c_[H,H-G[:,0]],np.c_[H,H-G[:,1]],D[:,None]])]:
 B=np.concatenate([(Z[:,None,:]*A[None,:,i,None]).reshape(Y.size,-1) for i,Z in enumerate(Zs)],axis=1)
 E=[]
 for lab in np.unique(d['lab_id']):
  m=np.repeat(d['lab_id']!=lab,A.shape[0]); b=np.linalg.lstsq(B[m],Y.ravel()[m],rcond=None)[0]; e=(B[~m]@b-Y.ravel()[~m]).reshape(-1,A.shape[0]); E.append([np.sqrt(np.mean(e[:,0]**2)),np.sqrt(np.mean(e[:,1:]**2))])
 b=np.linalg.lstsq(B,Y.ravel(),rcond=None)[0]; print('joint',name,np.round(b,5),'LOLO rate,curve',np.round(E,5))
