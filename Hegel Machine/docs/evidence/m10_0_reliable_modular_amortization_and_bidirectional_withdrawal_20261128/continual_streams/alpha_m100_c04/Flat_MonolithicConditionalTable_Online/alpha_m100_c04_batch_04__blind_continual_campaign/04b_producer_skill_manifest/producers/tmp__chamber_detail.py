import pickle, numpy as np, runpy
from scipy.optimize import least_squares
s=runpy.run_path('/tmp/hegel_continual_archive/alpha_m100_c04_batch_03_solver.py'); it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=it['dataset']; d=it['campaign_diagnostics']; tr,te=ds['train'],ds['test']; dt,t=ds['past_dt'],ds['eval_times']; x=s['_features'](tr,te); xd=s['_features'](tr,d)[len(tr['lab_id']):]; c=ds['calibration_load_cells']; r,l=c['bridge_raw'],c['true_channel_loads']; print('cal max sorted',np.sort(np.max(abs(r),axis=1))[-15:]); print('sweeps',[(k,np.shape(v)) for k,v in c['single_channel_sweeps'].items()]); b=s['_calibrate'](c); print('cal coeff',b.round(5)); print('cal residual quantiles',np.quantile(np.max(abs(np.einsum('nck,ck->nc',s['_bridge_design'](l),b)-r),axis=1),[.25,.5,.75,.9,.95,1])); inp=s['_inputs'](tr,x[:len(tr['lab_id'])],b); tau,th=s['_fit'](inp,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,th); p,h,z=s['_inputs'](d,xd,b); dr=z@th[2:]; y=d['strain_curves_obs']-d['strain_0'][:,None]; rate=d['instant_rate_obs']; a,g=s['_design'](tau,(p,h,z),dt,t); ar,gr=s['_chamber_response'](tau,th,h,z,t); print('baseline/reset rmse',[(np.sqrt(np.mean((v-y)**2)),np.sqrt(np.mean((w-rate)**2))) for v,w in [(a@th,g@th),(ar,gr)]]); tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; state=p@(-np.expm1(-dt[:,None]/tau)*np.exp(-tail[:,None]/tau)); R=-np.expm1(-t[:,None]/tau); T=np.vstack((np.c_[R,t],np.r_[1/tau,1])); Y=np.c_[y,rate]; print('diagnostic low-rank residual',np.quantile(abs(Y-np.linalg.lstsq(T,Y.T,rcond=None)[0].T@T.T),[.5,.9,.99,1])); print('hold and past',h[:5],p[:5]);
C=np.linalg.lstsq(T,Y.T,rcond=None)[0].T
base=h[:,None]*th[:2]; diff=C[:,:2]-base
print('lab, missing, h, drift, state0, state1, dA0, dA1, linear, total initial excess')
print(np.c_[d['lab_id'],np.isnan(d['coupon_features_raw']).sum(1),h,dr,state,diff,C[:,2],rate-base@(1/tau)].round(4))
print('feature panels drift changes')
for blocks in [(tr,d),(tr,te,d),(d,)]:
 u=s['_features'](blocks[0],blocks[1]) if len(blocks)==2 else None
 if u is None:
  common=set.intersection(*(set(k) for k in blocks)); j={k:np.concatenate([a[k] for a in blocks]) for k in common}; u=s['_features'](j,d) if len(blocks)==1 else s['_features']({k:np.concatenate([a[k] for a in blocks[:-1]]) for k in common},d)
 print((np.c_[np.ones(len(d['lab_id'])),u[-len(d['lab_id']):]]@th[2:]).round(4))
print('h, p ranges tr/test/diag')
for k in [tr,te,d]:
 yy=s['_stress'](k['past_channel_raw'],b); hh=s['_stress'](k['hold_channel_raw'],b); print(np.min(yy,axis=0),np.max(yy,axis=0),hh.min(),hh.max())
