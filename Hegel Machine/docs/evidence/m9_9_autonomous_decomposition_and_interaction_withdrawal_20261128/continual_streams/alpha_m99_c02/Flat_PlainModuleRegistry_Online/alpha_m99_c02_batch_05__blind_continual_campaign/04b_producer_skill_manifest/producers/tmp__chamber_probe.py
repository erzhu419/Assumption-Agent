import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import _verified_upstream as up
import downstream_solver as dn
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d=up.calibrate_upstream(dict(it['dataset'],chamber_verification_trials=it['campaign_diagnostics']))
a,b,q=d['train'],d['test'],d['chamber_verification_trials']; t,dt=d['eval_times'],d['past_dt']; bridge=np.r_[0.,np.ones(4)]
x,_=dn._features(a,b); tau,theta=dn._material(a,bridge,x,dt,t)
print('intrinsic',tau,theta)
p,h=dn._loads(q,bridge); y=q['strain_curves_obs']-q['strain_0'][:,None]; r=q['instant_rate_obs']; relax=-np.expm1(-t[:,None]/tau)
print('groups',[(key,np.unique(q[key],return_counts=True)) for key in ('lab_id','stratum')],'feature missing',np.isnan(q['coupon_features_raw']).sum(0))
for name,xx in [('joint',dn._features(a,q)[1]),('self',dn._features(q,q)[0]),('deployment',dn._features(dict((k,np.concatenate((a[k],b[k]))) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')),q)[1])]:
 drift=theta[2]+xx@theta[3:]; H=h[:,None]*theta[:2]; curve,rate=dn._basis(tau,p,h,xx,dt,t); F=rate[:,:2]*tau*theta[:2]; print('\n',name,'drift',drift.round(3))
 def score(c,v):
  e=abs(c-y); u=abs(v-r); return np.round([np.median(e),np.quantile(e,.9),np.median(u),np.quantile(u,.9)],5)
 for label,w in [('full',F),('zero',H)]:
  for k,g in [('linear',t),('unit',-np.expm1(-t)),('fast',tau[0]*relax[:,0])]: print(label,k,score(w@relax.T+drift[:,None]*g,w@(1/tau)+drift))
  z=(y-w@relax.T); est=np.array([np.median(row[abs(drift)>.06]/drift[abs(drift)>.06]) for row in z.T]); print(label,'empirical kernel',est.round(3))
 def pred(v):
  mix,fast,slow,df=v; w=H+mix*(F-H); g=fast*(-np.expm1(-t/tau[0]))+(df-fast/tau[0])*slow*(-np.expm1(-t/slow)); return w@relax.T+drift[:,None]*g,w@(1/tau)+df*drift
 for init in ([0,.2,1,1],[1,0,2,1]):
  fit=least_squares(lambda v: np.r_[(pred(v)[0]-y).ravel(),pred(v)[1]-r],init,bounds=([-3,-4,.05,-5],[3,4,50,5]),loss='soft_l1',f_scale=.014); print('fit mix,fast amp,slow tau,drift slope',fit.x.round(5),score(*pred(fit.x)))
