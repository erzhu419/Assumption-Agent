import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d,di=p['dataset'],p['campaign_diagnostics']; tr,te=d['train'],d['test']; t,dt=d['eval_times'],d['past_dt']
cal=d['calibration_load_cells']; raw=cal['bridge_raw']; y=cal['true_channel_loads'].sum(1)
A=np.c_[np.ones(len(raw)),raw]; B=np.c_[A,*[raw[:,i]*raw[:,j] for i in range(4) for j in range(i,4)]]
for name,a in [('affine',A),('quadratic',B)]:
 pred=np.empty_like(y)
 for q in np.array_split(np.argsort(y),5):
  k=np.ones(len(y),bool); k[q]=False; pred[q]=a[q]@np.linalg.lstsq(a[k],y[k],rcond=None)[0]
 errors=abs(pred-y); print('bridge',name,'MAE',errors.mean(),'worst stress band',max(errors[q].mean() for q in np.array_split(np.argsort(y),5)))
x,_=m._standardize(tr,te); _,xd=m._standardize({k:v[:0] for k,v in di.items()},di); stress=m._stress_calibration(cal)
z=di['strain_curves_obs']-di['strain_0'][:,None]; rate=di['instant_rate_obs']; f=-np.expm1(-t); labs=di['lab_id']; train_y=tr['strain_curves_obs']-tr['strain_0'][:,None]
terms=[np.ones(len(xd))]; names=['constant']
for i in range(4):
 for j in range(i+1,4):
  for sign in (-1,1):
   for c in (.5,1.,2.):
    v=xd[:,i]+sign*c*xd[:,j]; terms.extend([v,v*v]); names.extend([f'pair{i}{sign:+}{c}x{j}',f'pair_sq{i}{sign:+}{c}x{j}'])
for i in range(4):
 for j in range(4):
  for a in (.5,1.,2.):
   terms.append(xd[:,i]/(1+a*xd[:,j]**2)); names.append(f'ratio{i}{a}x{j}')
terms=np.array(terms).T
rows=[]
for g in np.unique(labs):
 q=labs==g; qtr=tr['lab_id']!=g; source={k:v[qtr] for k,v in tr.items()}
 tau,b=m._fit(m._design(source,x[qtr],stress,dt,t),train_y[qtr],tr['instant_rate_obs'][qtr]); s,r=m._design(di,xd,stress,dt,t,chamber=True)(tau)
 ec=z-s@b; er=rate-r@b; amp=np.median(ec/f,axis=1)
 scores=[]
 for j,v in enumerate(terms.T):
  folds=[]
  for h in np.unique(labs[~q]):
   test=labs==h; fit=~(q|test); coef=np.linalg.lstsq(v[fit,None],amp[fit],rcond=None)[0][0]; folds.append(np.mean(abs(amp[test]-coef*v[test])))
  scores.append((np.mean(folds),j))
 _,j=min(scores); v=terms[:,j]; coef=np.linalg.lstsq(v[~q,None],amp[~q],rcond=None)[0][0]; corr=coef*v[q]
 def metrics(curve,rr): return [np.mean(abs(curve)),np.mean(abs(rr))]
 rows.append([metrics(ec[q],er[q]),metrics(ec[q]-corr[:,None]*f,er[q]-corr)]); print('lab',g,'baseline/challenger',np.round(rows[-1],6),names[j])
print('worst lab baseline/challenger',np.max(rows,axis=0))
# Worst-band bridge bias or a repeatable held-out drift gain would justify a candidate edit.
out=m.solve_batch(d)
assert set(out)=={'estimated_taus','hereditary_weights','predicted_strain','initial_rate_pred'}
assert all(np.isfinite(v).all() for v in out.values())
assert .05<out['estimated_taus'][0]<out['estimated_taus'][1]<25
print('isolated contract OK')
