import pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m98_v2_c04_s00_solver as s
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=item['dataset']; tr=d['train']; te=d['test']; dg=item['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; br=d['calibration_load_cells']; raw=br['bridge_raw']; true=br['true_channel_loads']; b=s._bridge(br)
xb=s._features(tr,te); x=xb[:len(tr['lab_id'])]; p,h=s._loads(tr,b); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; tau,theta=s._fit(x,p,h,dt,t,y,r)
print('tau',tau,'theta',theta); print('linear bridge rms, max by channel',np.sqrt(np.mean((np.c_[np.ones(len(raw)),raw]@b-true)**2,0)),np.max(np.abs(np.c_[np.ones(len(raw)),raw]@b-true),0))
for k in br['single_channel_sweeps']: print('sweep',k,np.shape(br['single_channel_sweeps'][k]),np.asarray(br['single_channel_sweeps'][k]).ravel()[:10])
for name in ['ref_coupons','lab_id','stratum','coupon_features_raw']:
 pass
joint={k:np.concatenate([te[k],dg[k]]) for k in ['lab_id','stratum','coupon_features_raw','ref_coupons']}; xd=s._features(tr,joint)[-len(dg['lab_id']):]
from scipy.optimize import minimize_scalar
p,h=s._loads(dg,b); Y=dg['strain_curves_obs']-dg['strain_0'][:,None]; R=dg['instant_rate_obs']; A=np.c_[-np.expm1(-t[:,None]/tau),t]; B=np.c_[np.r_[A[:,0],1/tau[0]],np.r_[A[:,1],1/tau[1]],np.r_[t,1.]]; obs=np.c_[Y,R];
lab=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]); xnew=dg['coupon_features_raw'].copy(); st=np.r_[tr['stratum'],te['stratum']]; counts=np.unique(np.c_[lab,st],axis=0,return_inverse=True,return_counts=True); w=1./counts[2][counts[1],None]; rawx=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); fill=np.sum(w*xb*(~np.isnan(rawx)),0)/np.sum(w*(~np.isnan(rawx)),0)
for g in np.unique(lab):
 panel=refs[lab==g]; xnew[dg['lab_id']==g]=(xnew[dg['lab_id']==g]-np.median(panel.mean(0)))/np.median(panel.std(0))
xnew=np.where(np.isnan(xnew),fill,xnew)
print('max diag feature coordinate difference',np.max(abs(xnew-xd),0))
fns={'exp':lambda k:-np.expm1(-t/k)*k,'tanh':lambda k:k*np.tanh(t/k),'rational1':lambda k:t/(1+t/k),'rational2':lambda k:t/(1+(t/k)**2),'arctan':lambda k:k*np.arctan(t/k),'log':lambda k:k*np.log1p(t/k),'gaussian_integral':lambda k:k*np.sqrt(np.pi)/2*__import__('scipy').special.erf(t/k),'quadratic':lambda k:t-k*t*t,'exp_decay':lambda k:t*np.exp(-t/k)}
for use_rate in [False,True]:
 for name,fn in fns.items():
  def fit(k, ret=False):
   C=np.c_[B,np.r_[fn(k),1.]]; C,ob=(C,obs) if use_rate else (C[:-1],Y); co=np.linalg.lstsq(C,ob.T,rcond=None)[0].T; e=ob-co@C.T
   return (co,e) if ret else np.sqrt(np.mean(e**2))
  opt=minimize_scalar(lambda u:fit(np.exp(u)),bounds=(-4,5),method='bounded'); k=np.exp(opt.x); co,e=fit(k,True); print('shape',use_rate,name,round(k,5),round(opt.fun,6),'linear drift std',round(co[:,2].std(),4),'nonlinear drift corr',round(np.corrcoef(co[:,3],theta[2]+xnew@theta[3:])[0,1],4))
# Fit a generic common time correction with the train-estimated drift amplitude fixed.
for xx in [xd,xnew]:
 drift=theta[2]+xx@theta[3:]; later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; z=p@(-np.expm1(-dt[:,None]/tau)*np.exp(-later[:,None]/tau)); gap=h[:,None]-z; pred=(gap*theta[:2])@B[:,:2].T; e=obs-pred
 print('implied drift time',np.linalg.lstsq(np.c_[np.ones(len(drift)),drift],e,rcond=None)[0].round(4)); print('row regressors drift corr',np.corrcoef(np.c_[drift,e].T)[0,1:].round(4)); print('early and late residual regression onto drift and gaps',np.linalg.lstsq(np.c_[np.ones(len(drift)),drift,gap],e,rcond=None)[0][:,[0,2,6,11,12]].round(4))
 for name,fn in fns.items():
  def fit(k,ret=False):
   E=e-drift[:,None]*np.r_[fn(k),1.]; delta=np.linalg.lstsq(B[:,:2],E.T,rcond=None)[0].T; rem=E-delta@B[:,:2].T; return (delta,rem) if ret else np.sqrt(np.mean(rem**2))
  opt=minimize_scalar(lambda u:fit(np.exp(u)),bounds=(-4,5),method='bounded'); co,err=fit(np.exp(opt.x),True); print('fixed-drift',name,round(np.exp(opt.x),5),round(opt.fun,6),'delta rms',np.sqrt(np.mean(co**2,0)).round(4))
