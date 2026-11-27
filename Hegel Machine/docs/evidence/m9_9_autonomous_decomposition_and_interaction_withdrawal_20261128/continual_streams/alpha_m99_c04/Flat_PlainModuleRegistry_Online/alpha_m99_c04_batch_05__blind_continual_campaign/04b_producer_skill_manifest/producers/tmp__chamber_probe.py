import numpy as np,pickle,sys
sys.path.insert(0,'/holdout_task')
sys.path.insert(0,'/tmp/hegel_continual_archive')
from _verified_upstream import calibrate_upstream
from alpha_m99_c04_batch_02_solver import _features,_fit,_design,_mad
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d=calibrate_upstream(dict(item['dataset'],chamber_verification_trials=item['campaign_diagnostics']))
a,b,c=(d[k] for k in ('train','test','chamber_verification_trials'))
dt,t=d['past_dt'],d['eval_times']; stress=np.r_[0.,np.ones(4)]
x,z=_features(a,b)
tau,theta=_fit(a,x,stress,dt,t)
ca,ra=_design(a,x,stress,dt,t,tau)
ea=ca@theta+a['strain_0'][:,None]-a['strain_curves_obs']; er=ra@theta-a['instant_rate_obs']
print('tau/theta',tau,theta,'train MAD',_mad(ea),_mad(er),'max',abs(ea).max(),abs(er).max())
for name,xc in [('train+diag',_features(a,c)[1]),('test+diag',_features(b,c)[1]),('diag only',_features(c,c)[0])]:
 cc,rc=_design(c,xc,stress,dt,t,tau)
 y=c['strain_curves_obs']-c['strain_0'][:,None]; r=c['instant_rate_obs']; m=cc[:,:,:2]@theta[:2]; mr=rc[:,:2]@theta[:2]; drift=rc[:,2:]@theta[2:]
 print(name,'drift',np.round(drift,4),'rate err',np.round(r-mr-drift,4)); print('curve err percentile',np.round(np.percentile(y-m-drift[:,None]*t,[10,50,90],axis=0),4)); print('corr and scaling inferred drift at times',np.round(np.array([np.linalg.lstsq(np.c_[drift,np.ones(len(drift))],v,rcond=None)[0] for v in (y-m).T]),4).T)
 if name=='train+diag':
  np.savez('/tmp/chamber_probe.npz',t=t,dt=dt,tau=tau,theta=theta,x=xc,y=y,r=r,m=m,mr=mr,drift=drift,past=c['past_channel_raw'].sum(-1),hold=c['hold_channel_raw'].sum(-1),labs=c['lab_id'])
