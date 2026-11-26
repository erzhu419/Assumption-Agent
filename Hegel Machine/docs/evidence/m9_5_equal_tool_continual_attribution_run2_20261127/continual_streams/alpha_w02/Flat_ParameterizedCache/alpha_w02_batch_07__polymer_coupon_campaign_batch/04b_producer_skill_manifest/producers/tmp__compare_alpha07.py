import importlib.util,pickle,sys,numpy as np
paths=['/holdout_task/alpha_w02_batch_07_solver.py',*sys.argv[1:]]
mods=[]
for p in paths:
 s=importlib.util.spec_from_file_location('m',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); mods.append(m)
d=pickle.load(open('/holdout_task/public_alpha_w02_batch_07.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; dt,t,sy=(d[k] for k in ('past_dt','eval_times','yield_stress')); lab=tr['lab_id']; n=len(lab); f=mods[0]._features(tr,te)[:n]; inp=mods[0]._inputs(tr,f,mods[0]._calibrate(d['calibration_load_cells'],sy),dt,sy); y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; full=np.isfinite(tr['coupon_features_raw']).all(axis=1)
rmse=lambda x: np.sqrt(np.mean(x*x))
for j in range(4):
 fit=full&(lab!=j); val=lab==j; ps=[]
 for m in mods:
  tau,c=m._fit(tuple(v[fit] for v in inp),dt,t,y[fit]); ps.append((tau,m._design(tuple(v[val] for v in inp),dt,t,tau)@c))
 r=y[val]-ps[0][1]; drift=np.median(r[:,:-1]/t,axis=1); q=r[:,:-1]-drift[:,None]*t; clean=abs(q)<10*mods[0]._mad(q); print('lab',j,'spikes excluded',int((~clean).sum()))
 for path,(tau,p) in zip(paths,ps):
  e=y[val]-p; drift=np.median(e[:,:-1]/t,axis=1); q=e[:,:-1]-drift[:,None]*t; complete=full[val]; print(path.rsplit('/',1)[-1],np.round(tau,5),'clean complete curve/shape/rate RMSE',*[round(x,6) for x in (rmse(e[complete,:-1][clean[complete]]),rmse(q[complete][clean[complete]]),rmse(e[complete,-1]))])
