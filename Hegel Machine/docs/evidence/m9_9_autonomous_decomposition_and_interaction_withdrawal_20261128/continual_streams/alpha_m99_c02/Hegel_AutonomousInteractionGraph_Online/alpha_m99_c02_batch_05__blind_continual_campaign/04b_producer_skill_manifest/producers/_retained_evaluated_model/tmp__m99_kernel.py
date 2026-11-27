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
P,H=loads(d); Y=np.c_[d['instant_rate_obs'],d['strain_curves_obs']-d['strain_0'][:,None]]; L=d['lab_id']; Q=np.vstack((1/taus,-np.expm1(-t[:,None]/taus))); V=np.r_[1,t]; E=Y-H[:,None]*(Q@theta[:2]); K=.25*V+Q@(taus*np.array([.25,.5])); implied=E@K/(K@K); missing=np.isnan(d['coupon_features_raw']).any(axis=1)
print('fixed reset/drift kernel with free coupon amplitudes, rate and curve rms',np.sqrt(np.mean((E-implied[:,None]*K)**2,axis=0)).round(5)); u,s,v=np.linalg.svd(E,full_matrices=False); print('sv',s[:5]); print('rank1 normalized', (v[0]/v[0,0]).round(5)); print('fixed kernel',K.round(5)); print('rate projected vs actual',np.c_[implied,E[:,0]].round(5).T)
comb=lambda *ss:{k:np.concatenate([z[k] for z in ss]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')}
opts={'train/test':xd,'diag':features(d,d)[0],'train/diag':features(tr,d)[1],'test/diag':features(te,d)[1],'all':features(comb(tr,te),d)[1],'test':features(te,te,d)[2],'train':features(tr,tr,d)[2]}
for n,xq in opts.items():
 D=theta[2]+xq@theta[3:]; er=implied-D; print(n,'rmse',np.sqrt(np.mean(er**2)).round(5),'by lab',[(i, np.round(er[(L==i)&~missing],4).tolist()) for i in np.unique(L)],'missing',er[missing].round(4))
print('cal keys',list(ds)); print('test lab sizes',np.unique(te['lab_id'],return_counts=True)); print('past range diag/test',[(np.min(loads(z)[0]),np.max(loads(z)[0])) for z in (d,te)])
