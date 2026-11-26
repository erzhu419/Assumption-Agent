import importlib.util,pickle,numpy as np
from pathlib import Path
root=Path('/holdout_task')
spec=importlib.util.spec_from_file_location('solver',root/'alpha_w03_batch_06_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.loads((root/'public_alpha_w03_batch_06.pkl').read_bytes())[0]['dataset']; tr,te=d['train'],d['test']; n=len(tr['lab_id'])
x,h,p=m._prep(d); dt,t,sy=[d[k] for k in ('past_dt','eval_times','yield_stress')]; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']
import inspect
lab=tr['lab_id']; comp=np.isfinite(tr['coupon_features_raw']).all(1); x,h,p=x[:n],h[:n],p[:n]
source=inspect.getsource(m._fit)
# Pilot and spike rejection are unchanged. Compare the final weighting only.
variants={'baseline':m._fit}
for name,curve_factor,rate_factor in [('curve_dominant',1,0.45),('rate_dominant',1,2),('equal_horizons',0,1)]:
 s=source.replace('optimum, model = profile(keep / scale, optimum.x)',f'final_scale = np.r_[np.tile(sc if {curve_factor} else np.full(k,np.median(sc)),n),np.full(n,sr/{rate_factor})]\n    optimum, model = profile(keep / final_scale, optimum.x)')
 ns={}; exec(s,m.__dict__,ns); variants[name]=ns['_fit']
for name,fit in variants.items():
 metrics=[]; spectra=[]
 for held in ([],[0],[1],[2],[3],[0,1],[2,3]):
  val=np.isin(lab,held); i=np.flatnonzero(~val&comp); j=np.flatnonzero(val&comp)
  tau,co=fit(x[i],h[i],p[i],dt,t,sy,y[i],r[i]); spectra.append(tau)
  if not len(j): continue
  c,rr=m._design(tau,x[j],h[j],p[j],dt,t,sy); e=y[j]-c@co; er=r[j]-rr@co
  clean=abs(e)<.12
  for active in (abs(h[j])>sy, (abs(h[j])<=sy)&((np.maximum(abs(p[j])-sy,0)**2@dt)>0), (abs(h[j])<=sy)&((np.maximum(abs(p[j])-sy,0)**2@dt)==0)):
   if active.any():
    metrics.append((np.mean(abs(e[active][clean[active]])),np.mean(abs(e[active,-3:][clean[active,-3:]])),np.mean(abs(er[active])),np.median(abs(e[active]/t-np.median(e[active]/t,axis=1)[:,None]))))
 print(name,'full tau',np.round(spectra[0],6),'held tau ranges',np.round(np.ptp(spectra[1:],axis=0),6),'worst clean curve/late/rate/shape',np.round(np.max(metrics,axis=0),6))
