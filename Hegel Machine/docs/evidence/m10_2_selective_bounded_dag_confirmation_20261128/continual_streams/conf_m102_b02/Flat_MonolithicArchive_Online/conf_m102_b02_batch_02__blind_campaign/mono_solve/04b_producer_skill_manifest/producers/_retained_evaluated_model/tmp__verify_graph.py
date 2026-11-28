import copy, importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; ids=np.arange(len(tr['x'])); rng=np.random.default_rng(492)
dy, ds=m._features(s,tr['x']); y,T=tr['y'],tr['traj']; modes=np.linalg.svd(T,full_matrices=False)[2][:2]
print('rank',*[np.linalg.matrix_rank(d) for d in (dy,ds)])
print('coeff response',m._ls(dy,y)); print('coef T cubic mode 1 and 2',m._ls(ds,T@modes.T)[-16:].reshape(4,4,2)); print('T coefficient temporal singular values',np.linalg.svd(m._ls(ds,T),compute_uv=False));print('telemetry',s['telemetry_vec'])
def check(a,b):
 d=dict(s, test={'x':tr['x'][b]}, train=dict(tr,**{k:tr[k][a] for k in ('x','y','traj')})); p=m.solve_batch(d)
 e=np.abs(p['y_hat']-y[b]); et=np.abs(p['state_pred']-T[b]);
 return np.array([e.mean(),et.mean(),et.mean(0).max(),e.max(),et.max()])
fixed=check(ids[:52],ids[52:]); print('fixed',fixed); assert max(fixed[:3])<.038
E=[]
for _ in range(200):
 k=rng.permutation(ids); E.append(check(k[:52],k[52:]));
for f in (np.median,np.max):print('random',f.__name__,f(E,axis=0))
for name,x in [('radius',np.linalg.norm(tr['x']-tr['x'].mean(0),axis=1)),('leverage',np.sum(np.linalg.qr(ds)[0]**2,axis=1)), *[(f'coordinate {i} {sign:+}',sign*tr['x'][:,i]) for i in range(tr['x'].shape[1]) for sign in (-1,1)]]:
 k=np.argsort(x); print('extrapolation',name,check(k[:52],k[52:]));
print('test maximum leverage',max(np.sum((m._features(s,s['test']['x'])[1]@np.linalg.pinv(ds))**2,axis=1)))
