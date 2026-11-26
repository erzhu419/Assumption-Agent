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
for a in np.unique(tr['lab_id']):
 for s in ('elastic','history','hold'):
  mask={'elastic':(hist[:n]==0)&~hold[:n],'history':(hist[:n]>0)&~hold[:n],'hold':hold[:n]}[s]; mask &= (tr['lab_id']==a)&comp
  if mask.any(): print('in-fit',a,s,int(mask.sum()),'curve/rate RMSE',rms(e[mask]),rms(er[mask]),'median by horizon',np.round(np.median(e[mask],0),5))
for a in np.unique(tr['lab_id']):
 keep=tr['lab_id']!=a; val=~keep
 inner=dict(d,train={k:v[keep] for k,v in tr.items()},test={k:np.concatenate((tr[k][val],te[k])) for k in te if k in tr}); q=m.solve_alpha_w03_batch_06(inner); z=int(val.sum()); dp=q['predicted_strain'][:z]-tr['strain_curves_obs'][val]; dr=q['initial_rate_pred'][:z]-r[val]; aux=q['predicted_strain'][:z]-tr['strain_curves_obs'][val]-dr[:,None]*t
 print('leave lab',a,'taus',q['estimated_taus'],'curve/rate/onset-anchored curve RMSE',rms(dp),rms(dr),rms(aux),'late',rms(dp[:,-3:]));
