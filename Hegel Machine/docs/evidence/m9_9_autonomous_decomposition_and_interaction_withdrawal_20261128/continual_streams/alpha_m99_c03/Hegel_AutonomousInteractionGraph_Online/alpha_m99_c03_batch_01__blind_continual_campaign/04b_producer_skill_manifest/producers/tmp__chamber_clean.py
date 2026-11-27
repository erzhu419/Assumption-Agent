import numpy as np,pickle,sys
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m99_c03_s00_solver as m
z=np.load('/tmp/chamber_probe.npz'); p,h,dt,t,tau,th,Y,v=[z[k] for k in ('past','hold','dt','t','taus','theta','curve','rate')]; it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; q=it['campaign_diagnostics']; d=it['dataset']; tr,te=d['train'],d['test']; x=m._features(q,{k:q[k][:0] for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')}); good=np.isfinite(q['coupon_features_raw']).all(1); r=th[2]+x@th[3:]; F=np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1.]]); y=np.c_[Y,v]; base=h[:,None]*(F[:,:2]@th[:2]); mem=p@(-np.expm1(-dt[:,None]/tau)*np.exp(-(np.cumsum(dt[::-1])[::-1]-dt)[:,None]/tau));
for name,co in [('half',[.5,.5,0]),('half-plus',[.5,.45,.05]),('equal-plus',[.475,.475,.05])]:
 pred=base+r[:,None]*(F@(np.array(co)*np.r_[tau,1.])); err=y-pred; print(name,'clean',np.round(np.sqrt(np.mean(err[good]**2,0)),4),'missing',np.round(np.max(abs(err[~good]),1),4));
for name,A in [('r',r[:,None]),('r+1',np.c_[r,np.ones(len(r))]),('r+features',np.c_[r,np.ones(len(r)),x]),('r+hold+mem',np.c_[r,h,mem])]:
 def fit(mask):
  B=np.linalg.lstsq(A[mask],y[mask]-base[mask],rcond=None)[0]; return B,base+A@B
 B,pred=fit(good); errs=np.stack([y[i]-fit(good & (np.arange(len(r))!=i))[1][i] for i in range(len(r))]); print(name,'clean fitted latent',np.round(B@np.linalg.pinv(F.T),5),'OOF RMSE/max',np.round([np.sqrt(np.mean(errs[:,:-1]**2)),abs(errs[:,:-1]).max(),np.sqrt(np.mean(errs[:,-1]**2)),abs(errs[:,-1]).max()],5));
print('diag errors by lab and missing (half-plus)'); err=y-base-r[:,None]*(F@(np.r_[tau,1.]*[.5,.45,.05])); print(np.round(np.c_[q['lab_id'],~good,r,err[:,0],err[:,-2:]],5));
for block in [tr,te,q]:
 print('telemetry',[(k,np.shape(val),np.round(np.mean(val,axis=0),3)) for k,val in block.items() if 'telemetry' in k]); print('refs',[(i,np.round(block['ref_coupons'][block['lab_id']==i].mean(0),3).tolist()) for i in np.unique(block['lab_id'])]);
