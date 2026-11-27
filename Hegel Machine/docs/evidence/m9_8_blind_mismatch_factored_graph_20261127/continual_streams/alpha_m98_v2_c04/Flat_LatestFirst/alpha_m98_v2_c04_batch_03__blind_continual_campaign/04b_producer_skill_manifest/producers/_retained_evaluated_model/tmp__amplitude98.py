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
p,h=s._loads(dg,b); n=len(h); later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; m=-np.expm1(-dt[:,None]/tau)*np.exp(-later[:,None]/tau); z=p@m; gap=h[:,None]-z; print('dt, t',dt,t); print('diag keys',list(dg)); print('raw and calibrated hold ranges',h.min(),h.max())
D=B[:,:2]; target=obs-(gap*theta[:2])@D.T
Z=lambda k:np.r_[-k*np.expm1(-t/k),1.]
from scipy.optimize import least_squares
raw=dg['coupon_features_raw']; complete=np.isfinite(raw).all(axis=1); gr=dg['lab_id']; q1=theta[2]+xnew@theta[3:]; q2=theta[2]+xd@theta[3:];
for k in [0.95,1.0,1.05,1.1,1.15]:
 v=Z(k); ideal=target@v/(v@v); rem=target-ideal[:,None]*v; free=np.linalg.lstsq(np.c_[D,v],target.T,rcond=None)[0].T; ef=target-free@np.c_[D,v].T
 print('time',k,'baseline memory RMS',np.sqrt(np.mean(rem**2)), 'first/last/rate RMS',np.sqrt(np.mean(rem[:,[0,-2,-1]]**2,axis=0)), 'free memory RMS',np.sqrt(np.mean(ef**2)), 'complete drift difference RMS', [np.sqrt(np.mean((ideal[complete]-q[complete])**2)) for q in [q1,q2]])
 if k==1.0: print('lab complete drift_traintest drift_all fitted_amp amp_with_free_memory corrected_rate_residual', np.round(np.c_[gr,complete,q1,q2,ideal,free[:,-1],rem[:,-1]],4))
# Estimate observable chamber changes without assigning a free correction to each coupon.
for keep in [np.ones(n,dtype=bool),complete]:
 for label,F in [('intercept',np.ones((n,1))),('hold+memory',np.c_[np.ones(n),h,z]),('full history',np.c_[np.ones(n),h,p])]:
  G=F[keep]; aa=np.c_[G,q1[keep]]; coef=np.linalg.lstsq(aa,target[keep],rcond=None)[0]; print('projected chamber',keep.sum(),label,'drift response',np.round(coef[-1,[0,1,3,6,9,11,12]],4),'remaining RMS',round(np.sqrt(np.mean((target[keep]-aa@coef)**2)),5))
