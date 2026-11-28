import pickle,importlib.util,numpy as np,sys
spec=importlib.util.spec_from_file_location('candidate',sys.argv[1] if len(sys.argv)>1 else '/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train']; xx=t['x']; n=len(xx)
slices=[np.arange(k,k+20) for k in (52,0,20,40)]+[np.random.default_rng(k).permutation(n)[:20] for k in range(8)]
for te in slices:
 tr=np.setdiff1d(np.arange(n),te);p=m.solve_batch(dict(s,train={k:t[k][tr] for k in ('x','y','traj')},test={'x':xx[te]})); ey=abs(p['y_hat']-t['y'][te]); et=abs(p['state_pred']-t['traj'][te]);print('split',te[0],np.round([ey.mean(),et.mean(),et[:,-1].mean(),et.mean(0).max()],6))
