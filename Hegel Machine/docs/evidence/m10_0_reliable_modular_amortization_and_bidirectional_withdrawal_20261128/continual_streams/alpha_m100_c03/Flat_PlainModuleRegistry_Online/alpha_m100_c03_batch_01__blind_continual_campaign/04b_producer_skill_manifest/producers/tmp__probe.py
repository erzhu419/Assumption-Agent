import pickle,numpy as np
from bench import _features,_bridge,_loads,_fit,_design
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True,linewidth=135)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=i['dataset'];tr,te,di=d['train'],d['test'],i['campaign_diagnostics'];dt,t=d['past_dt'],d['eval_times']; b=_bridge(d['calibration_load_cells']);X=_features(tr,te);n=len(tr['lab_id']);tau,theta=_fit(X[:n],*_loads(tr,b),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']);print('material',tau,theta,'bridge',b,'dt',dt)
def diagnostic_features(di):
 x=di['coupon_features_raw'].copy();rr=np.vstack((tr['ref_coupons'],te['ref_coupons']));ll=np.r_[tr['lab_id'],te['lab_id']];xx=np.vstack((tr['coupon_features_raw'],te['coupon_features_raw']));fill=np.array([X[np.flatnonzero(np.isnan(xx[:,j]))[0],j] for j in range(4)]);
 for lab in np.unique(ll):
  p=rr[ll==lab];m=di['lab_id']==lab;x[m]=(x[m]-np.median(p.mean(0)))/np.median(p.std(0))
 return np.where(np.isnan(x),fill,x)
Xd=diagnostic_features(di)
for s,x in [(tr,X[:n]),(di,Xd)]:
 p,h=_loads(s,b);a,r=_design(tau,x,p,h,dt,t);ec=s['strain_curves_obs']-s['strain_0'][:,None]-a@theta;er=s['instant_rate_obs']-r@theta;print('baseline med/mad/rmse by time',np.median(ec,0),np.median(abs(ec),0),np.sqrt(np.mean(ec**2,0)));print('rate med mad rmse',np.median(er),np.median(abs(er)),np.sqrt(np.mean(er**2)))
p,h=_loads(di,b);a,r=_design(tau,Xd,p,h,dt,t);y=np.column_stack((di['strain_curves_obs']-di['strain_0'][:,None],di['instant_rate_obs']));dr=theta[2]+Xd@theta[3:];amp0=a[:,0,:2]*theta[:2]/(-np.expm1(-t[0]/tau));
def basis(ts):return np.column_stack((np.r_[-np.expm1(-t[:,None]/ts),[1/ts]],np.r_[t,1]))
def fit(ts):
 f=basis(ts);return np.array([least_squares(lambda w:f@w-row,np.linalg.lstsq(f,row,rcond=None)[0],loss='soft_l1',f_scale=.005).x for row in y])
for ts in [tau,tau*[.8,1.2],tau*[1.2,.8]]:
 f=basis(ts);q=fit(ts);e=y-q@f.T;print('tau',ts,'med abs',np.median(abs(e),0),'bycoupon',np.round(np.median(abs(e),1),4));print('amp correction first 12',np.c_[q[:12]-np.c_[amp0,dr][:12],di['lab_id'][:12],di['stratum'][:12]])
q=fit(tau);z=q-np.c_[amp0,dr];u=np.c_[h,p,h-p[:,-1],amp0,dr,Xd,di['strain_0'],di['lab_id'],di['stratum']];print('correlation h,p0..5,h-p5,amp0..1,dr,X0..3,strain0,lab,stratum',np.corrcoef(z.T,u.T)[:3,3:]); print('diag ranges',np.min(y,0),np.max(y,0));
