import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; n=len(x); idx=np.arange(n); rng=np.random.default_rng(43)
cal=m._calibration(s['panel_a']); z=m._invert(x,cal); print('Calibration:',cal)
bp=s['panel_b']; w=m._affine(z) @ m._fit(m._affine(bp['u']),bp['v']);
for q in (z,w):
 print('Scalar coefficients',np.round(m._fit(m._quadratic(q[:52]),t['y'][:52]),5)); print('Trajectory affine coefficient singular values',np.linalg.svd(m._fit(m._affine(q[:52]),t['traj'][:52])[1:],compute_uv=False))
splits=[('52/20',idx[52:])]+[('random'+str(i),rng.permutation(n)[:20]) for i in range(16)]+[('feature'+str(j)+str(sgn),np.argsort(sgn*z[:,j])[-20:]) for j in range(4) for sgn in [-1,1]]+[('radius',np.argsort(np.linalg.norm(z,axis=1))[-20:])]
results=[]
for name,te in splits:
 tr=np.setdiff1d(idx,te); st=dict(s,train={k:v[tr] if k!='summary' else v for k,v in t.items()},test={'x':x[te]}); o=m.solve_batch(st)
 e=np.abs(o['y_hat']-t['y'][te]); f=np.abs(o['state_pred']-t['traj'][te]); results.append([e.mean(),f.mean(),f.mean(0).max(),e.max(),f.max()]); print(name,np.round(results[-1],6))
assert np.max(results,0)[:3].max()<.038
print('worst',np.max(results,0))
