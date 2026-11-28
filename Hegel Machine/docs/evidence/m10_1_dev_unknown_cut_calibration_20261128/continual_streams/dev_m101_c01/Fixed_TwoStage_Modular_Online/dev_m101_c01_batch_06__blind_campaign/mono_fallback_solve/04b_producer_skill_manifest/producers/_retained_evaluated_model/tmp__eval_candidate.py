import importlib.util,pickle,numpy as np
P='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('candidate',P); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,z=(tr[k] for k in ('x','y','traj'))
ss=[('fixed',np.arange(72)),*[(f'random-{seed}',np.random.default_rng(seed).permutation(72)) for seed in (4,91,133,287)],('high-radius',np.argsort((x*x).sum(1))),('far-neighbor',np.argsort(np.sort(((x[:,None,:]-x[None,:,:])**2).sum(2),axis=1)[:,3]))]
a=[]
for label,ix in ss:
 i,j=ix[:56],ix[56:]; v=m.solve_batch({**s,'train':{**tr,**{k:tr[k][i] for k in ('x','y','traj')}},'test':{'x':x[j]}}); yp,zp=v['y_hat'],v['state_pred']; end=zp[:,-1]; assert np.allclose(v['diagnostics_vec'],[yp.mean(),yp.std(),end.mean(),end.std()],rtol=0,atol=1e-13); e=np.abs(zp-z[j]); f=[np.mean(abs(yp-y[j])),e.mean(),e[:,-1].mean(),e.mean(0).max()]; a.append(f); print(label,np.round(f,6))
print('worst (scalar, trajectory, endpoint, worst horizon)',np.round(np.max(a,axis=0),6))
v=m.solve_batch(s); assert all(np.isfinite(u).all() for u in v.values()); print('actual shapes',{k:w.shape for k,w in v.items()})
