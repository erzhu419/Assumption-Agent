import importlib.util,pickle,numpy as np
from pathlib import Path
root=Path('/holdout_task')
spec=importlib.util.spec_from_file_location('solver',root/'alpha_w03_batch_06_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.loads((root/'public_alpha_w03_batch_06.pkl').read_bytes())[0]['dataset']; tr,te=d['train'],d['test']; n=len(tr['lab_id'])
x,h,p=m._prep(d); dt,t,sy=[d[k] for k in ('past_dt','eval_times','yield_stress')]; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']
assert np.all(np.diff(t)>0)
for name, z in [('yield-gated', d), ('fully subyield', dict(d, yield_stress=float(max(abs(h).max(),abs(p).max())+1.)))]:
 out=m.solve_alpha_w03_batch_06(z)
 tau=out['estimated_taus']; coef=np.r_[out['hereditary_weights'],out['viscoplastic_weights']]
 assert np.all(tau > 0) and tau[0] < tau[1]
 assert all(np.isfinite(v).all() for v in out.values())
 if name=='fully subyield':
  assert np.array_equal(out['viscoplastic_weights'], [0.,0.]), coef
 tau=m.solve_alpha_w03_batch_06(d)['estimated_taus'] if name=='fully subyield' else tau
 eh=1e-7
 a,rate=m._design(tau,x,h,p,dt,np.array([0.,eh]),z['yield_stress'])
 err=np.max(abs((a[:,1]-a[:,0])/eh-rate))
 assert np.max(abs(a[:,0]))==0 and err<1e-5, err
 print(name,'derivative-basis max error',err,'plastic coefficients',out['viscoplastic_weights'])
