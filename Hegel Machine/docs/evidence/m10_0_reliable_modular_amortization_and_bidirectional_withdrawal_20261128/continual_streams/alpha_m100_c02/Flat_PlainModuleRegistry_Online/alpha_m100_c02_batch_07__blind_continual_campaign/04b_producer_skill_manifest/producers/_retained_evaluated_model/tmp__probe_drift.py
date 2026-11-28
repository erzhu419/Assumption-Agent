import sys
sys.path.insert(0,'/holdout_task')
import downstream_solver as m
from _verified_upstream import calibrate_upstream
import pickle,numpy as np,json
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=item['dataset']; diag=item['campaign_diagnostics']; pay=json.load(open("/holdout_task/_upstream_param_payload.json")); d=calibrate_upstream(raw,pay); g=calibrate_upstream(dict(raw,test=diag),pay)['test']; tr,te=d['train'],d['test']; t=d['eval_times']; dt=d['past_dt']; x,xt=m._pooled_features(tr,te); tau,th=m._fit_intrinsic(tr,x,dt,t); print('material',tau,th)
ll=np.r_[tr['lab_id'],te['lab_id']]; rr=np.vstack([tr['ref_coupons'],te['ref_coupons']]); xx=m._standardize(np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]),ll,rr); mu=np.nanmean(xx,axis=0); mu=np.array([xt[np.isnan(te['coupon_features_raw'][:,j]),j+1][0] if np.isnan(te['coupon_features_raw'][:,j]).any() else (x[np.isnan(tr['coupon_features_raw'][:,j]),j+1][0] if np.isnan(tr['coupon_features_raw'][:,j]).any() else mu[j]) for j in range(4)])
gx=np.array(g['coupon_features_raw'],copy=True)
for lab in np.unique(ll):
 p=rr[ll==lab]; gx[g['lab_id']==lab]=(gx[g['lab_id']==lab]-np.median(p.mean(0)))/np.median(p.std(0))
gx=m._with_drift_intercept(gx,mu); print('diag missing',np.argwhere(np.isnan(g['coupon_features_raw'])))
past,hold=m._loads(g); drift=gx@th[2:]; y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; w=np.r_[np.ones(len(t)),.5]; old=m._predict_chamber(g,gx,tau,th,t); print('old median/max',np.median(abs(old-y),axis=0),np.max(abs(old-y),axis=0));
def report(name,a,b):
 e=a-b; print(name,'strain med/rms/max',np.round([np.median(abs(e[:,:-1])),np.mean(e[:,:-1]**2)**.5,np.max(abs(e[:,:-1]))],5),'rate',np.round([np.median(abs(e[:,-1])),np.mean(e[:,-1]**2)**.5,np.max(abs(e[:,-1]))],5))
def robust(a,b):
 return least_squares(lambda p:(a@p-b)/.012,np.linalg.lstsq(a,b,rcond=None)[0],jac=lambda p:a/.012,loss='soft_l1',max_nfev=120).x
bas=np.c_[m._basis(tau,t),np.r_[t,1.]]
ind=np.array([robust(bas*w[:,None],row*w) for row in y]); report('individual',ind@bas.T,y); print('amp sample',np.round(np.c_[hold,drift,ind][:12],5))
loads=np.c_[hold,past]; decay=np.exp(-dt[:,None]/tau); z= (1-decay)*np.vstack([np.prod(decay[i+1:],axis=0) for i in range(len(dt))]); mem=past@z
from itertools import product
base=((hold[:,None]-mem)*th[:2])@bas[:,:2].T
# Test whether the extra chamber response has a different time profile, or depends on a volume/load or coupon covariate.
G=lambda times: np.c_[m._basis(times,t),np.r_[t,1.]]
features={'r':drift[:,None], 'r+1':np.c_[drift,np.ones(len(y))], 'r+nonlinear':np.c_[drift,drift**2,np.ones(len(y))], 'features':gx, 'r+hold':np.c_[drift,hold,np.ones(len(y))], 'r*load':np.c_[drift,drift[:,None]*np.c_[hold,mem]], 'r+load':np.c_[drift,hold,mem]}
rawgx=g['coupon_features_raw']; gd=m._standardize(rawgx,g['lab_id'],g['ref_coupons']); features['r_diagpanel']=(m._with_drift_intercept(gd,mu)@th[2:])[:,None]
for key,sel in [('full_rows', np.ones(len(y),bool)),('complete',np.isfinite(rawgx).all(axis=1))]:
 print('slice',key); e=y-base-drift[:,None]*(bas@np.array([.3258,2.4696,-.2545])); print('mean residual by time',np.round(e[sel].mean(axis=0),4)); print('signed rate',np.round(e[sel,-1],4)); print('signed final',np.round(e[sel,-2],4))
def fit(mask,name,weight=.5):
 w=np.r_[np.ones(len(t)),weight]; f=features.get(name,features['r']); yy=((y-base)[mask]*w).ravel(); n=3*f.shape[1]
 def given(q):
  B=G(np.exp(q)) if name=='free_drift_tau' else bas
  D=(B[None,:,:,None]*f[:,None,None,:]).reshape(len(y),len(w),n); A=(D[mask]*w[None,:,None]).reshape(-1,n); p=robust(A,yy); return base+D@p,p
 if name=='free_drift_tau':
  fn=lambda q: ((given(q)[0][mask]-y[mask])*w/.012).ravel()
  q=min((least_squares(fn,np.log(s),loss='soft_l1',bounds=(np.log(.06),np.log(24)),max_nfev=65) for s in (tau,[.25,2.],[.65,6.])),key=lambda r:r.cost).x
 else: q=np.log(tau)
 return (*given(q),np.exp(q))
for name in [*features,'free_drift_tau']:
 p,c,q=fit(np.ones(len(y),bool),name); cv=np.zeros_like(y); worst=[]
 for lab in np.unique(g['lab_id']):
  s=g['lab_id']==lab; cv[s]=fit(~s,name)[0][s]; e=(cv-y)[s]; worst.append([np.mean(e[:,:-1]**2)**.5,np.mean(e[:,-1]**2)**.5])
 print(name,'worst',np.round(np.max(worst,axis=0),5),'basis taus',np.round(q,4),'coeff',np.round(c,4)); report('LOLO',cv,y)
