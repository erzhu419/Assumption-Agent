import importlib.util,pickle,numpy as np
from pathlib import Path
root=Path('/holdout_task')
spec=importlib.util.spec_from_file_location('solver',root/'alpha_w03_batch_06_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.loads((root/'public_alpha_w03_batch_06.pkl').read_bytes())[0]['dataset']; tr,te=d['train'],d['test']; n=len(tr['lab_id'])
x,h,p=m._prep(d); dt,t,sy=[d[k] for k in ('past_dt','eval_times','yield_stress')]; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']
print('transfer',m._calibrate(d['calibration_load_cells'],np.arange(4)))
comp=np.isfinite(tr['coupon_features_raw']).all(1)
o=m.solve_alpha_w03_batch_06(d); tau,co=o['estimated_taus'],np.r_[o['hereditary_weights'],o['viscoplastic_weights']]; print('parameters',tau,co)
c,rr=m._design(tau,x,h,p,dt,t,sy); e=y-c[:n]@co; er=r-rr[:n]@co
hist=np.maximum(np.abs(p)-sy,0)**2@dt; hold=np.abs(h)>sy; state=te['residual_plastic_state']; print('plastic state sample',state[:12], 'range',np.min(state),np.max(state)); print('plastic correlations with history, signed history, hold and observed rate residual',np.corrcoef(np.vstack([state,hist[n:],hist[n:]*h[n:],h[n:],te['instant_rate_obs']-rr[n:]@co]))[0]); print('plastic state inactive / active quantiles',[(int(q.sum()),np.quantile(state[q],[0,.5,1]).tolist()) for q in (hist[n:]==0,hist[n:]>0) if q.any()])
def rms(a): return float(np.sqrt(np.mean(a*a)))
print('curve abs quantiles',np.quantile(abs(e[comp]),[.5,.75,.9,.95,.98,.99,1]),'rate abs quantiles',np.quantile(abs(er[comp]),[.5,.75,.9,.95,.98,.99,1])); print('large curve cells per row histogram',np.bincount((abs(e[comp])>.1).sum(1)))
for held in ([0],[1],[2],[3],[0,1],[2,3]):
 val=np.isin(tr['lab_id'],held); keep=~val; i=np.flatnonzero(keep&comp); j=np.flatnonzero(val)
 tau,co=m._fit(x[i],h[i],p[i],dt,t,sy,y[i],r[i]); c0,r0=m._design(tau,x[j],h[j],p[j],dt,t,sy)
 dc=y[j]-c0@co; dr=r[j]-r0@co; print('held',held,'tau',np.round(tau,6),'co',np.round(co,5))
 for name,q in [('elastic',(hist[j]==0)&~hold[j]),('history',(hist[j]>0)&~hold[j]),('hold',hold[j]),('missing',~comp[j])]:
  if name!='missing': q &= comp[j]
  if q.any():
   print(name,q.sum(),'rate MAE',round(np.mean(abs(dr[q])),5),'curve medAE onset shrinkage',np.round([np.median(abs(dc[q]-s*dr[q,None]*t)) for s in [0,.25,.5,1]],5),'clean curve MAE',np.round([np.mean(abs((dc[q]-s*dr[q,None]*t)[abs(dc[q])<.1])) if (abs(dc[q])<.1).any() else np.nan for s in [0,.25,.5,1]],5))
