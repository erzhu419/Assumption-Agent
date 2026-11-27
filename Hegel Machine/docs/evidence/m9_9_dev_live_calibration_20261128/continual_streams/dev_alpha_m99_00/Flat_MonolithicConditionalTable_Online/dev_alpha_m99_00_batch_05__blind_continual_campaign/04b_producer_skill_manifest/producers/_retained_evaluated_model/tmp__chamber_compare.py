import importlib.util, pickle, numpy as np
from pathlib import Path
p=Path('/holdout_task/public_batch.pkl')
with p.open('rb') as f: item=pickle.load(f)[0]
d=item['dataset']; obs=item['campaign_diagnostics']; tr=d['train']; te=d['test']; t=d['eval_times']; dt=d['past_dt']
root=Path('/tmp/hegel_continual_archive')
def load(path):
 s=importlib.util.spec_from_file_location(path.stem,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m=load(root/'dev_alpha_m99_00_batch_04_solver.py'); n=len(tr['lab_id']); bridge=m._bridge(d['calibration_load_cells']); X=m._features(tr,te); tau,th=m._fit(m._coupon(tr,X[:n],bridge),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,th)
# Use the train/test lab reference coordinates on verification coupons, and exclude missing features from this mechanism comparison.
lab=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]); x=obs['coupon_features_raw'].copy(); ol=obs['lab_id']; good=np.isfinite(x).all(axis=1)
for l in np.unique(ol):
 panel=refs[lab==l]; x[ol==l]=(x[ol==l]-np.median(panel.mean(0)))/np.median(panel.std(0))
x=x[good]; o={k:v[good] for k,v in obs.items()}; y=o['strain_curves_obs']-o['strain_0'][:,None]; r=o['instant_rate_obs']; coupon=m._coupon(o,x,bridge); past,hold,_=coupon
A,B=m._design(tau,coupon,dt,t)
def report(name,curve,rate):
 e=curve-y; q=rate-r; print(name,'curve/rate median abs',np.median(abs(e)),np.median(abs(q)),'worst lab medians',max(np.median(abs(e[o['lab_id']==l])) for l in np.unique(o['lab_id'])),max(np.median(abs(q[o['lab_id']==l])) for l in np.unique(o['lab_id'])))
report('benchtop',A@th,B@th)
k=-np.expm1(-t[:,None]/tau); score=th[2]+x@th[3:]; report('reset only',hold[:,None]*(k@th[:2])+score[:,None]*t,hold*np.sum(th[:2]/tau)+score)
report('batch04',*m._deployment(hold,x,tau,th,t))
for idx,args in [(1,(hold,x,tau,th,t)),(2,(tau,th,coupon,dt,t))]:
 try: report('batch0'+str(idx),*load(root/f'dev_alpha_m99_00_batch_0{idx}_solver.py')._deployment(*args))
 except Exception as e: print(type(e).__name__,str(e))
c=d['calibration_load_cells']; raw=c['bridge_raw']; loads=c['true_channel_loads']; print('calibration max total error clean',max(abs(bridge(raw[np.linalg.norm(raw,axis=1)<10])-loads[np.linalg.norm(raw,axis=1)<10].sum(axis=1)))); print('verification complete rows per lab',np.unique(o['lab_id'],return_counts=True)); print('verification increment range',np.percentile(y,[0,25,50,75,100]),'rate range',np.percentile(r,[0,25,50,75,100]))
