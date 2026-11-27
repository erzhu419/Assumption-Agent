import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; g=i['campaign_diagnostics']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; b=m._bridge(d['calibration_load_cells']); xtr,xte=m._features(tr,te); tau,theta=m._material(tr,b,xtr,dt,t)
assert 0.05 < tau[0] < tau[1] < 25
p,h=m._loads(te,b); zz,_=m._chamber(tau,theta,p,h,xte,dt,np.array([0.])); assert np.array_equal(zz,np.zeros_like(zz))
eps=1e-6; zz,rr=m._chamber(tau,theta,p,h,xte,dt,np.array([eps])); de=np.max(abs(zz[:,0]/eps-rr)); assert de<2e-5; print('initial derivative max error',de)
cal=d['calibration_load_cells']; u0=cal['bridge_raw']; q0=cal['true_channel_loads'].sum(1); idx=np.arange(len(u0)); err=[]
for j in range(5):
 k=idx%5==j; a=np.c_[np.ones(len(u0)),u0]; w=np.linalg.lstsq(a[~k],q0[~k],rcond=None)[0]; err.append(np.sqrt(np.mean((a[k]@w-q0[k])**2)))
print('bridge worst held-out total-load RMSE',max(err))
for l in np.unique(tr['lab_id']):
 k=tr['lab_id']!=l; train={name:v[k] for name,v in tr.items() if isinstance(v,np.ndarray) and v.shape and len(v)==len(k)}; u,w=m._material(train,b,xtr[k],dt,t)
 p,h=m._loads(tr,b); c,r=m._basis(u,p[~k],h[~k],xtr[~k],dt,t)
 ec=c@w+tr['strain_0'][~k,None]-tr['strain_curves_obs'][~k]; er=r@w-tr['instant_rate_obs'][~k]
 print('intrinsic held-out lab',l,'tau',u.round(5),'curve/rate MAE and max time MAE',np.mean(abs(ec)),np.mean(abs(er)),np.max(np.mean(abs(ec),axis=0)))
# Diagnostic panel normalization is separate from the deployment train/test panel.
x,_=m._features(g,{key:v[:0] for key,v in g.items()}); p,h=m._loads(g,b); c,r=m._basis(tau,p,h,x,dt,t); base=c@theta; y=g['strain_curves_obs']-g['strain_0'][:,None]; u=t[:,None]/tau; a=(theta[2]+x@theta[3:])[:,None,None]*(-np.expm1(-u)-u); lab=g['lab_id']; complete=np.isfinite(g['coupon_features_raw']).all(1); worst=np.zeros(3)
for l in np.unique(lab):
 k=lab==l; fit=~k&complete; aa=a[fit].reshape(-1,2); yy=(y-base)[fit].ravel(); q=least_squares(lambda q:aa@q-yy, np.linalg.lstsq(aa,yy,rcond=None)[0],loss='soft_l1',f_scale=.003).x
 cc,rr=m._chamber(tau,theta,p[k],h[k],x[k],dt,t,route=q); ec=abs(cc-y[k]); er=abs(rr-g['instant_rate_obs'][k]); scores=np.array([ec.mean(),er.mean(),ec.mean(0).max()]); worst=np.maximum(worst,scores); print('chamber held-out lab',l,'curve/rate MAE and max time MAE',scores)
print('chamber worst',worst)
cc,rr=m._chamber(tau,theta,p,h,x,dt,t); np.testing.assert_allclose(cc,base+a@m._ROUTE,atol=1e-12,rtol=0); print('actual candidate full diagnostic MAE',abs(cc-y).mean(),abs(rr-g['instant_rate_obs']).mean())
