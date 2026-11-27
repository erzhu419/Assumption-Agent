import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m98_v2_c01_batch_02_solver as s
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=item['dataset']; g=item['campaign_diagnostics']; tr,te=ds['train'],ds['test']; dt,t=ds['past_dt'],ds['eval_times']; cal=ds['calibration_load_cells']; bridge=s._axial_load_map(cal)
xtr,xte=s._features(tr,te); tau,theta=s._fit(s._specimen(tr,xtr,bridge),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs'])
keys=('coupon_features_raw','ref_coupons','lab_id','stratum'); raw,ref,lab,strata=(np.concatenate((tr[k],te[k])) for k in keys); xx=np.vstack((xtr,xte)); x=g['coupon_features_raw'].copy()
for l in np.unique(lab):
 r=ref[lab==l]; x[g['lab_id']==l]=(x[g['lab_id']==l]-np.median(r.mean(0)))/np.median(r.std(0))
_,cell,n=np.unique(np.c_[lab,strata],axis=0,return_inverse=True,return_counts=True); x=np.where(np.isfinite(x),x,np.average(xx,axis=0,weights=1/n[cell]))
p,h,f=spec=s._specimen(g,x,bridge); c,r=s._branches(spec,dt,t,tau); drift=f@theta[2:]; A=r*tau*theta[:2]; E=-np.expm1(-t[:,None]/tau); y=g['strain_curves_obs']-g['strain_0'][:,None]; yr=g['instant_rate_obs']; dy=y-A@E.T; dr=yr-A@(1/tau)
print('cal axial RMSE',np.sqrt(np.mean((((cal['bridge_raw']-bridge[0])@bridge[1])-cal['true_channel_loads'].sum(1))**2)))
print('diagnostic y residual after subtracting train memory / drift (rows sorted by drift)'); print(np.round(np.c_[drift,dr,dy[:,[0,2,5,8,11]]][np.argsort(drift)],4))
print('fit across coupons each time: (drift slope, intercept, median abs residual)')
for k in range(len(t)):
 b=np.linalg.lstsq(np.c_[drift,drift*0+1],dy[:,k],rcond=None)[0];print(round(t[k],3),np.round(b,4),round(np.median(abs(dy[:,k]-np.c_[drift,drift*0+1]@b)),4))
print('frequency profiles and residual scales for constrained law candidates')
for nm,fn,p0 in [('linear',lambda q:(t,1.),[]),('exp_unit_slope',lambda q:(-np.expm1(-q[0]*t)/q[0],1.),[1.]),('exp_free_slope',lambda q:(-q[1]*np.expm1(-q[0]*t)/q[0],q[1]),[1.,1.]),('tanh',lambda q:(np.tanh(q[0]*t)/q[0],1.),[1.]),('rational',lambda q:(t/(1+q[0]*t),1.),[1.]),('damped',lambda q:(t*np.exp(-q[0]*t),1.),[1.]),('log',lambda q:(np.log1p(q[0]*t)/q[0],1.),[1.])]:
 def errors(q):
  curve,rate=fn(q);return dy-drift[:,None]*curve,dr-drift*rate
 def fun(q):
  a,b=errors(q);return np.r_[a.ravel(),b]
 q=least_squares(fun,p0,bounds=(1e-5,np.inf),loss='soft_l1',f_scale=.02).x if p0 else p0
 a,b=errors(q);print(nm,np.round(q,5),'median',np.median(abs(a)),np.median(abs(b)),'max coupon',np.max(np.sqrt(np.mean(a*a,1))),np.max(abs(b)))
