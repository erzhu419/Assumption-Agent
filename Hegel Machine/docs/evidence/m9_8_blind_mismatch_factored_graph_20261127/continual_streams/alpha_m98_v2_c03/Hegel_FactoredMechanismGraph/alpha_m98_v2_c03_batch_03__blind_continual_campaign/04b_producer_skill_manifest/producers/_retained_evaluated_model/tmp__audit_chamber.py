import importlib.util, json, pickle
import numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; c=i['campaign_diagnostics']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; out=m.solve_batch(d); tau,theta=out['estimated_taus'],out['hereditary_weights']; o,b=m._calibration(d['calibration_load_cells']); past=(c['past_channel_raw']-o)@b; hold=(c['hold_channel_raw']-o)@b; x=m._features(c,c)[:len(hold)]; feat=np.c_[np.ones(len(x)),x]; coupon=(past,hold,feat); drift=feat@theta[2:]; obs=np.c_[c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs']]; base=m._deployment_design(coupon,dt,t,tau)@theta; bench=m._design(coupon,dt,t,tau)@theta; kernel=np.r_[-np.expm1(-t),1.]; e=obs-base
# Keep the deployed law fixed. Use leave-one-lab-out to test whether stress or feature contrasts, interactions, or rational saturation improve transfer.
z=hold[:,None]-m._design(coupon,dt,np.array([1.]),tau)[:,-1,:2]*tau
j=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)); pairs=[x[:,a]*x[:,b] for a in range(x.shape[1]) for b in range(a+1,x.shape[1])]; contrast=x[:,0]-x[:,1]; rat=np.c_[drift/(1+drift**2)-drift,drift/(1+contrast**2)-drift]; families={'linear amplitude correction':feat[:,None,:]*kernel[None,:,None],'pair interactions':np.array(pairs).T[:,None,:]*kernel[None,:,None],'rational drift correction':rat[:,None,:]*kernel[None,:,None],'memory correction':z[:,None,:]*j[None,:,:]}
def rms(r): return np.array([np.sqrt(np.mean(r[:,:-1]**2)),np.sqrt(np.mean(r[:,-1]**2))])
lab=c['lab_id']; labs=np.unique(lab); print('schema description:',json.load(open('/holdout_task/public_asset_index.json'))['schema']); print('deployed vs benchtop chamber RMSE',rms(e),rms(obs-bench))
for name,a in families.items():
 r=e.copy()
 for l in labs:
  mask=lab!=l; design=a[mask].reshape(-1,a.shape[-1]); target=e[mask].ravel(); p=np.linalg.lstsq(design,target,rcond=None)[0]; r[~mask]-=a[~mask]@p
 print(name,'OOF RMSE',rms(r),'worst lab RMSE',np.max([rms(r[lab==l]) for l in labs],axis=0))
for l in labs: print('lab',l,'fixed RMSE',rms(e[lab==l]),'mean error',e[lab==l].mean(0)[[0,len(t)-1,len(t)]])
missing=np.isnan(c['coupon_features_raw']).any(1); print('missing slice RMSE',rms(e[missing])); print('worst horizon',np.sqrt(np.mean(e[:,:-1]**2,0)).max())
# Leave calibration rows out to distinguish constitutive error from a bridge fit that does not transfer.
cal=d['calibration_load_cells']; raw=cal['bridge_raw']; y=cal['true_channel_loads']; ids=np.arange(len(raw)); err=np.empty(len(raw))
for f in range(5):
 val=ids%5==f; x0,x1=m._calibration({'bridge_raw':raw[~val],'true_channel_loads':y[~val]}); err[val]=(raw[val]-x0)@x1-y[val].sum(1)
print('bridge worst fold stress RMSE',max(np.sqrt(np.mean(err[ids%5==f]**2)) for f in range(5)))
print('train and test local-vs-joint feature change'); pool=m._features(tr,te); n=len(tr['lab_id']); local=m._features(te,te)[:len(te['lab_id'])]; print(np.sqrt(np.mean(((pool[n:]-local)@theta[3:])**2)))
