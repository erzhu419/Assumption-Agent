import importlib.util, json, pickle
import numpy as np
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/dev_alpha_m99_00_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=a['dataset']; tr,te,d=ds['train'],ds['test'],a['campaign_diagnostics']; dt,t=ds['past_dt'],ds['eval_times']; n=len(tr['lab_id']); b=m._bridge(ds['calibration_load_cells']); xf=m._features(tr,te); tau,theta=m._fit(m._coupon(tr,xf[:n],b),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs'])[:2]
lab=d['lab_id']; refs=np.concatenate([tr['ref_coupons'],te['ref_coupons']]); labs=np.r_[tr['lab_id'],te['lab_id']]; x=d['coupon_features_raw'].copy(); strata=np.r_[tr['stratum'],te['stratum']]; _,i,c=np.unique(np.c_[labs,strata],axis=0,return_inverse=True,return_counts=True); w=1/c[i]; imp=(w@xf)/w.sum()
for l in np.unique(lab):
 p=refs[labs==l]; x[lab==l]=(x[lab==l]-np.median(p.mean(0)))/np.median(p.std(0))
x=np.where(np.isfinite(x),x,imp); past,h,dr=m._coupon(d,x,b); g=dr@theta[2:]; C,R=m._design(tau,(past,h,dr),dt,t); base=np.c_[R@theta,C@theta]; dc,rr=m._deployment(h,x,tau,theta,t); prev=np.c_[rr,dc]; y=np.c_[d['instant_rate_obs'], d['strain_curves_obs']-d['strain_0'][:,None]]; E=y-base
print('intrinsic',tau,theta,'ranges h,g,past',[(v.min(),v.max()) for v in (h,g,past)]);print('SVD of benchtop residual',np.linalg.svd(E,compute_uv=False).round(4)); print('SVD of prior chamber residual',np.linalg.svd(y-prev,compute_uv=False).round(4));
def report(name,P):
 e=E-P
 print(name, 'rmse(rate, curves,last)', np.round(np.sqrt([np.mean(e[:,0]**2),np.mean(e[:,1:]**2),np.mean(e[:,-1]**2)]),4),'worst lab median abs(rate,curves)',np.round(np.max([[np.median(abs(e[lab==l,0])),np.median(abs(e[lab==l,1:]))] for l in np.unique(lab)],axis=0),4))
report('benchtop',0);report('prior chamber',prev-base)
z=h[:,None]-R[:,:2]*tau
features={'1':np.ones(len(h)),'h':h,'g':g,'z0':z[:,0],'z1':z[:,1],'h*g':h*g,'g2':g*g,'h2':h*h,'z0*g':z[:,0]*g,'z1*g':z[:,1]*g,'gap0*g':(h-z[:,0])*g,'gap1*g':(h-z[:,1])*g,'pastmean':past@dt/dt.sum(),'plast':past[:,-1]}
print('correlations with rate,last and first two residual PCs')
U,S,V=np.linalg.svd(E,full_matrices=False)
for k,v in list(features.items())[1:]+[(f'x{i}',x[:,i]) for i in range(4)]+[(f'p{i}',past[:,i]) for i in range(6)]:print(k,np.corrcoef(np.c_[v,E[:,0],E[:,-1],U[:,:2]].T)[0,1:].round(3))
for keys in ('1 g','1 h g','1 z0 z1 g','1 z0 z1 h g','1 g h*g','1 g g2','1 g gap0*g','1 g gap1*g','1 g z0*g z1*g','1 g h*g z0*g z1*g','1 h g z0 z1 h*g z0*g z1*g'):
 X=np.column_stack([features[k] for k in keys.split()]); P=np.empty_like(E)
 for l in np.unique(lab):
  s=lab!=l; P[~s]=X[~s]@np.linalg.lstsq(X[s],E[s],rcond=None)[0]
 report(keys,P)
print('columns: row, h, g, z0, z1, observed rate, residual rate, residual late')
print(np.c_[np.arange(len(h)),h,g,z,y[:,0],E[:,0],E[:,-1]].round(4))
idx=json.load(open('/holdout_task/public_asset_index.json')); print('asset metadata top-level',list(idx['schema']) if isinstance(idx['schema'],dict) else type(idx['schema']).__name__)
np.savez('/tmp/chamber_probe.npz',tau=tau,theta=theta,t=t,dt=dt,past=past,h=h,x=x,g=g,z=z,y=y,E=E,lab=lab)
