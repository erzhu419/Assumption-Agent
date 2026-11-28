import importlib.util, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; d=s['train']; x,y,z=(d[k] for k in ('x','y','traj')); n=len(x); u=m._coordinates(s,x); t=s['eval_times']
for name,folds in [('random',[a for seed in (3,91,201,17,49) for a in np.array_split(np.random.default_rng(seed).permutation(n),4)]),('tail',[np.arange(56,n)]),('extremes', [np.argsort(q)[-12:] for q in [*x.T,*(-x).T,*u.T,*(-u).T]])]:
 rows=[]
 for va in folds:
  tr=np.setdiff1d(np.arange(n),va); st={**s,'train':{k:d[k][tr] for k in ('x','y','traj')},'test':{'x':x[va]}}; o=m.solve_batch(st)
  yz=abs(o['y_hat']-y[va]); zz=abs(o['state_pred']-z[va]);
  base=m._affine(x[va])@m._ls(m._affine(x[tr]),z[tr]); g=u[:,0]*u[:,2]+u[:,1]*u[:,3]; y18=u[va]@m._ls(u[tr],y[tr]-.18*g[tr])+.18*g[va]
  rows.append([yz.mean(),abs(y18-y[va]).mean(),zz.mean(),zz.mean(0).max(),zz[:,-1].mean(),abs(base-z[va]).mean(),abs(base-z[va]).mean(0).max()]);
 print(name,'y, y_fixed, traj, max_horizon, final, affine_traj, affine_max_horizon; mean',np.mean(rows,axis=0),'worst',np.max(rows,axis=0))
# Confirm that no single panel observation drives the actual test predictions.
o=m.solve_batch(s)
for panel in ('panel_a','panel_b'):
 errs=[]
 for i in range(len(s[panel]['u'])):
  st={**s,panel:{k:np.delete(a,i,axis=0) for k,a in s[panel].items()}}; q=m.solve_batch(st); errs.append([np.max(abs(o[k]-q[k])) for k in ('y_hat','state_pred','diagnostics_vec')]);
 print(panel,'jackknife max change',np.max(errs,axis=0))
