import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('bs','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; rng=np.random.default_rng(114)
def score(a,b):
 p=m.solve_batch(dict(s,train={k:v[a] for k,v in tr.items() if k!='summary'},test={'x':x[b]})); y=np.abs(tr['y'][b]-p['y_hat']); t=np.abs(tr['traj'][b]-p['state_pred']); return y.mean(),t.mean(),y.max(),*t.mean(0)
cal=m._calibrate(s['panel_a'],x)
for desc,idx in [('required',np.arange(72)), *[(f'random {i}',rng.permutation(72)) for i in range(6)], ('radial tail',np.argsort(np.linalg.norm(cal,axis=1))), *[(f'axis {i} tail',np.argsort(abs(cal[:,i]))) for i in range(4)]]:
 print(desc,np.round(score(idx[:52],idx[52:]),6))
