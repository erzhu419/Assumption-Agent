import pickle,numpy as np,importlib.util
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); rng=np.random.default_rng(811)
bs=m._coordinates(s,tr['x']); splits=[np.arange(n)]+[rng.permutation(n) for _ in range(20)]+[np.argsort(np.linalg.norm(bs,axis=1))]
errors=[]
for p in splits:
 i,j=p[:52],p[52:]; q={**s,'train':{**tr,**{k:tr[k][i] for k in ('x','y','traj')}},'test':{'x':tr['x'][j]}}; out=m.solve_batch(q)
 e=np.abs(out['state_pred']-tr['traj'][j]); errors.append([np.abs(out['y_hat']-tr['y'][j]).mean(),e.mean(),e[:,-1].mean(),max(e.mean(axis=0))])
e=np.array(errors); print('fixed, random mean, random worst, radial extrapolation', np.vstack((e[0],e[1:-1].mean(0),e[1:-1].max(0),e[-1]))); print('fixed horizon errors checked through worst horizon above')
