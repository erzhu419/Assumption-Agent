import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
a = m._calibration(s['panel_a']); p=s['panel_b']; b=m._fit(m._affine(p['u']),p['v'])
t=s['train']; z=m._affine(m._decode(t['x'],a))@b; zt=m._affine(m._decode(s['test']['x'],a))@b
f=m._scalar_design(z); y=t['y']; r=t['traj']; w=m._fit(f,y); print('scalar coefs',w,'resid RMS',np.mean((y-f@w)**2)**.5)
print('train max decoded norm',np.max(np.linalg.norm(z,axis=1)),'test',np.max(np.linalg.norm(zt,axis=1)))
splits=[('52/20',np.arange(52),np.arange(52,72)),('56/16',np.arange(56),np.arange(56,72))]
ix=np.random.default_rng(4).permutation(72)
for i,j in enumerate(np.array_split(ix,6)):
 splits.append((f'fold{i}',np.setdiff1d(ix,j),j))
j=np.argsort(np.linalg.norm(z,axis=1))[-16:]; splits.append(('outer radius',np.setdiff1d(ix,j),j))
for name,i,j in splits:
 sub=dict(s,train={k:v[i] for k,v in t.items() if k!='summary'},test={'x':t['x'][j]})
 o=m.solve_batch(sub)
 e=np.abs(o['y_hat']-y[j]); d=np.abs(o['state_pred']-r[j]); print(name,'y/traj/terminal/worst time',*[round(float(q),7) for q in [e.mean(),d.mean(),d[:,-1].mean(),d.mean(0).max()]])
