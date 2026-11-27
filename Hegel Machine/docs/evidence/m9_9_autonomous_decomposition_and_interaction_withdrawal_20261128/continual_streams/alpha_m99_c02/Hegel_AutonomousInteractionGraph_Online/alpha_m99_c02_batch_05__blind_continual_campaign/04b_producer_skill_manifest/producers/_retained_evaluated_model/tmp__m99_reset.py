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
from itertools import product
X=features(d,d)[0]; D=theta[2]+X@theta[3:]; good=~missing
B=np.c_[V,Q*taus]; Y0=E
for name,J in [('nominal',K),('free_mix',None)]:
 if J is None:
  a=np.linalg.lstsq((D[good,None,None]*B).reshape(-1,3),Y0[good].ravel(),rcond=None)[0]; J=B@a; print('estimated kernel fractions [linear,fast,slow]',a,'sum',a.sum())
 print(name,'complete-case RMSE rate,strain',np.sqrt(np.mean((Y0[good]-D[good,None]*J)**2,axis=0)).round(5))
 implied=Y0@J/(J@J)
 for i in np.flatnonzero(missing):
  j=np.flatnonzero(np.isnan(d['coupon_features_raw'][i])); print('missing row',i,'lab/stratum',L[i],d['stratum'][i],'columns',j,'inferred delta',np.round((implied-D)[i]/theta[3+j],4),'used',X[i,j].round(4))
print('reference transforms: split/lab/count/center/scale')
for name,s in [('train',tr),('test',te),('diag',d)]:
 for l in np.unique(L):
  p=s['ref_coupons'][s['lab_id']==l];print(name,l,len(p),round(np.median(p.mean(0)),4),round(np.median(p.std(0)),4))
print('missing per split:',[np.isnan(z['coupon_features_raw']).sum(axis=0).tolist() for z in (tr,te,d)])
