import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y,T=tr['y'],tr['traj']; off,scale=m._calibrate(s['panel_a']); op=m._operator(s['panel_b']); b=op((tr['x']-off)/scale); H=m._affine(b)
print('scalar coefficients', m._fit(m._quadratic(b[:52]),y[:52]),m._fit(m._quadratic(b),y))
print('trajectory affine noise',np.std(H@m._fit(H,T)-T))
rng=np.random.default_rng(7); tests={'public':np.arange(52,72)}
for k in range(15): tests[f'random{k}']=rng.choice(72,20,replace=False)
for k in range(4):
 for end,ix in [('low',slice(0,20)),('high',slice(-20,None))]:
  tests[f'x{k}_{end}']=np.argsort(tr['x'][:,k])[ix]; tests[f'b{k}_{end}']=np.argsort(b[:,k])[ix]
for k in range(4): tests[f'radial{k}']=np.argsort(np.linalg.norm(b-b.mean(0),axis=1))[-(12+k*2):]
metrics={}
for key, q in tests.items():
 a=np.setdiff1d(np.arange(len(y)),q); d=dict(s,train={k:tr[k][a] for k in ['x','y','traj']},test={'x':tr['x'][q]}); o=m.solve_batch(d)
 dy=np.abs(o['y_hat']-y[q]); dt=np.abs(o['state_pred']-T[q]); plain=np.abs(H[q]@m._fit(H[a],T[a])-T[q]); metrics[key]=[dy.mean(),dt.mean(),dt.mean(0).max(),dt[:,-1].mean(),plain.mean(),dy.max()]
for key in ['public',*sorted(metrics,key=lambda k:max(metrics[k][:3]),reverse=True)[:7]]: print(key,np.round(metrics[key],7))
print('worst',np.max(list(metrics.values()),axis=0))
