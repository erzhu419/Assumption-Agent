import importlib.util,pickle,sys,numpy as np
p=sys.argv[1] if len(sys.argv)>1 else '/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; r=s['train']; x=r['x']; y=r['y']; t=r['traj']; rng=np.random.default_rng(801)
for j,p in enumerate([np.arange(len(x))]+[rng.permutation(len(x)) for _ in range(8)]+[np.argsort(np.linalg.norm(x,axis=1)),np.argsort(y)]):
 tr,va=p[:52],p[52:]; out=m.solve_batch({**s,'train':{**r,**{k:r[k][tr] for k in ('x','y','traj')}},'test':{'x':x[va]}}); a,b=out['y_hat'],out['state_pred']; e=abs(a-y[va]); d=abs(b-t[va]); print(j,*(round(q,6) for q in [e.mean(),d.mean(),d[:,-1].mean(),e.max()]),flush=True)
