import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m98_v2_c04_batch_04_solver as m
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr=d['train']; te=d['test']; q=p['campaign_diagnostics']; t=d['eval_times']; dt=d['past_dt']; n=len(tr['lab_id']); b=m._fit_bridge(d['calibration_load_cells']); x=m._features(tr,te); tau,th=m._fit(x[:n],*m._loads(tr,b),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); past,hold=m._loads(q,b); y=q['strain_curves_obs']-q['strain_0'][:,None]; r=q['instant_rate_obs']; X=m._features(tr,q)[n:]; xself=m._features(q,q)[:len(r)]; xraw=q['coupon_features_raw'];
def temporal(u):
 return np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1.]
def extract(u,rw=1):
 a,f=temporal(u); aa=np.vstack([a,rw*f]); z=np.c_[y,rw*r]; w=np.linalg.lstsq(aa,z.T,rcond=None)[0].T
 for _ in range(5):
  e=w@aa.T-z; wt=1/np.sqrt(1+(e/.025)**2); w=np.linalg.solve(np.einsum('ni,ij,ik->njk',wt,aa,aa),np.einsum('ni,ij,ni->nj',wt,aa,z)[...,None])[...,0]
 return w,w@aa.T-z
print('base tau theta',tau,th,'hold range',hold.min(),hold.max(),'raw features nan',np.isnan(xraw).sum())
for rw in (0,0.25,1,2):
 fit=least_squares(lambda u:extract(np.exp(u),rw)[1].ravel(),np.log(tau),bounds=(np.log(.051),np.log(24.99)),loss='soft_l1',f_scale=.025,max_nfev=100); u=np.sort(np.exp(fit.x)); w,e=extract(u,rw); print('free rates rw',rw,'tau',u,'curve/rate MAE',np.median(abs(e[:,:-1])),np.median(abs(w@temporal(u)[1]-r)))
for name,u in [('base',tau),('free',u)]:
 w,e=extract(u,.5); des=np.c_[np.ones(len(r)),hold,past,X];
 print(name,'amp stats',np.round(np.percentile(w,[0,50,100],axis=0),4));
 for label,Z in [('past hold',des[:,:8]),('features',np.c_[des[:,0],X]),('all',des)]:
  coeff=np.linalg.lstsq(Z,w,rcond=None)[0]; pred=Z@coeff; print(label,'rmse',np.round(np.sqrt(np.mean((pred-w)**2,axis=0)),4),'coeff',np.round(coeff.T,3).tolist())
print('coupons columns: hold, past loads, baseline drift, Y at five times, rate')
print(np.round(np.c_[hold,past,th[2]+X@th[3:],y[:,[0,2,5,8,11]],r],3))
