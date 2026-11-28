import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['y']); rng=np.random.default_rng(709)
splits=[np.arange(n)]+[rng.permutation(n) for _ in range(32)]
x=t['x']; z=m._calibrate(s['panel_a'],x)
splits += [np.argsort(np.linalg.norm(z,axis=1))]+[np.argsort(sign*z[:,j]) for j in range(4) for sign in (-1,1)]
a=[]
for k,p in enumerate(splits):
 i,j=p[:52],p[52:]; q=dict(s,train={l: t[l][i] for l in ('x','y','traj')} | {'summary':t['summary']},test={'x':x[j]}); d=m.solve_batch(q)
 dy=np.abs(d['y_hat']-t['y'][j]); ds=np.abs(d['state_pred']-t['traj'][j]); a.append([dy.mean(),ds.mean(),ds.mean(0).max(),ds[:,-1].mean()]);
 if k==0 or k>=33: print(k,np.round(a[-1],7))
print('worst random',np.max(a[:33],axis=0),'worst stress',np.max(a,axis=0))
