import numpy as np, pickle, runpy
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; m=runpy.run_path('/holdout_task/batch_solver.py'); t=s['train']; x=t['x']; y=t['y']; z=t['traj']; w=m['_calibration'](s['panel_a']); print('calibration',w)
u=m['_inverse'](x,w); orders=[np.arange(72)]+[np.random.default_rng(k).permutation(72) for k in range(12)]+[np.argsort(g) for g in [np.linalg.norm(u,axis=1),*u.T,*(-u).T]]
for k,o in enumerate(orders):
 i,j=o[:52],o[52:]; a=m['solve_batch']({**s,'train':{**t,'x':x[i],'y':y[i],'traj':z[i]},'test':{'x':x[j]}}); e=abs(a['y_hat']-y[j]); r=abs(a['state_pred']-z[j]); print(k, np.round([e.mean(),r.mean(),r.mean(axis=0).max(),e.max(),r.max()],6))
