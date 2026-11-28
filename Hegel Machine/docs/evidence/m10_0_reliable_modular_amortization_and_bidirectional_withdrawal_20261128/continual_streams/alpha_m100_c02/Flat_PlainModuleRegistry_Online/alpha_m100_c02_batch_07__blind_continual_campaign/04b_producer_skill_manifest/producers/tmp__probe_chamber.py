import sys
sys.path.insert(0,'/holdout_task')
import downstream_solver as m
from _verified_upstream import calibrate_upstream
import pickle,numpy as np
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=item['dataset']; diag=item['campaign_diagnostics']; d=calibrate_upstream(raw); g=calibrate_upstream(dict(raw,test=diag))['test']; tr,te=d['train'],d['test']; t=d['eval_times']; dt=d['past_dt']; x,xt=m._pooled_features(tr,te); tau,th=m._fit_intrinsic(tr,x,dt,t); print('material',tau,th)
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
configs={'drift only addition':(np.c_[drift],)*3,'hold/drift':(np.c_[hold,drift],)*3,'hold/drift/known memory':tuple(np.c_[hold,drift,mem[:,i]] for i in range(2))+(np.c_[hold,drift],),'free history':(np.c_[loads,drift],)*3,'feature/history':(np.c_[loads,gx],)*3,'quadratic drift':(np.c_[hold,drift,drift**2,np.ones(len(y))],)*3}
for name,fs in configs.items():
 base=hold[:,None]*(bas[:,:2]@th[:2]) if name=='drift only addition' else np.zeros_like(y); A=np.concatenate([f[:,None,:]*bas[None,:,k,None] for k,f in enumerate(fs)],axis=2); Aw=A*w[None,:,None]; b=(y-base)*w; n=A.shape[-1]; p=robust(Aw.reshape(-1,n),b.ravel()); pred=base+A@p; cv=np.zeros_like(y)
 for lab in np.unique(g['lab_id']):
  sel=g['lab_id']==lab; q=robust(Aw[~sel].reshape(-1,n),b[~sel].ravel()); cv[sel]=base[sel]+A[sel]@q
 print(name,'coeff',np.round(p,4));report(' fit',pred,y);report('LOLO',cv,y)
