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
for label,F in [('free',np.eye(n)),('fixed drift',None),('affine drift',np.c_[np.ones(n),theta[2]+xd@theta[3:]]),('joint affine drift',np.c_[np.ones(n),theta[2]+xnew@theta[3:]]),('joint features',np.c_[np.ones(n),xnew]),('raw features',np.c_[np.ones(n),np.nan_to_num(dg['coupon_features_raw'])])]:
 def fit(k,ret=False):
  q=Z(k); v=q-D@np.linalg.lstsq(D,q,rcond=None)[0]; amp=theta[2]+xd@theta[3:] if F is None else F@np.linalg.lstsq(F,target@v/(v@v),rcond=None)[0]; delta=np.linalg.lstsq(D,(target-amp[:,None]*q).T,rcond=None)[0].T; e=target-amp[:,None]*q-delta@D.T; return (amp,delta,e) if ret else np.sqrt(np.mean(e*e))
 opt=minimize_scalar(lambda k:fit(k),bounds=(0.5,2.5),method='bounded'); amp,delta,e=fit(opt.x,True); print(label,'time',opt.x,'rms',opt.fun,'delta branch rms',np.sqrt((delta**2).mean(0)))
 if label=='free':
  drift=theta[2]+xnew@theta[3:]; print('hold past-last drift amplitude delta branch:'); print(np.round(np.c_[dg['lab_id'],h,p[:,-1],drift,amp,delta],5))
  X=np.c_[np.ones(n),h,p,z,drift,xnew]; names=['1','h']+[f'p{i}' for i in range(p.shape[1])]+['z0','z1','drift','x0','x1','x2','x3']; Y0=np.c_[amp-drift,delta]; print('correlation to amp minus drift and memory deltas',dict(zip(names[1:],np.round(np.corrcoef(np.c_[X[:,1:],Y0].T)[:-3,-3:],3).tolist()))); print('delta fits:');
  for cols in [[0,1,2,3,4,5,6,7],[0,1,8,9],[0,1,8,9,10],[0,1,8,9,11,12,13,14]]:
   C=np.linalg.lstsq(X[:,cols],Y0,rcond=None)[0]; print([names[i] for i in cols],np.round(C,4).tolist(),'remaining',np.sqrt(np.mean((X[:,cols]@C-Y0)**2,axis=0)))
