import importlib.util, pickle, numpy as np
sp=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; N=len(t['x']); rng=np.random.default_rng(228); ids=np.arange(N)
splits=[('public',ids[:52],ids[52:])]+[(f'random{i}',p[:52],p[52:]) for i,p in enumerate(rng.permutation(N) for _ in range(15))]; d=np.linalg.norm((t['x']-t['x'].mean(0))/t['x'].std(0),axis=1); p=np.argsort(d); splits+=[('radial',p[:56],p[56:])];
for j in range(4):
 p=np.argsort(t['x'][:,j]); splits.extend([(f'axis{j}lo',p[12:],p[:12]),(f'axis{j}hi',p[:-12],p[-12:])])
E=[]
for name,a,b in splits:
 u=dict(s,train={k:t[k][a] for k in ['x','y','traj']},test={'x':t['x'][b]}); ys,ts=m._components(u); e=np.r_[abs(ys-t['y'][b]).mean(1),abs(np.array([.30,.45,.25]) @ ys-t['y'][b]).mean(),abs(ts-t['traj'][b]).mean((1,2)),abs(ts.mean(0)-t['traj'][b]).mean()]; E.append(e); print(name,np.round(e,5))
E=np.array(E); print('mean/worst random',E[:16].mean(0),E[:16].max(0)); print('worst outer',E[16:].max(0))
