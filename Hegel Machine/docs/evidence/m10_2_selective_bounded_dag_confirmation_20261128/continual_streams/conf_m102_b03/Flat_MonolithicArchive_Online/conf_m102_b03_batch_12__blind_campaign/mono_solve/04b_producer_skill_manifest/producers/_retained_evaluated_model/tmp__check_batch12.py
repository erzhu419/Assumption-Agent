import importlib.util,numpy as np,pickle
q=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(q);q.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; np.set_printoptions(precision=4,suppress=True)
F=m._features(s,tr['x']); H=np.load('/tmp/features.npz')['H']; a=m.ls(H,tr['traj'].T).T; C=m.ls(F[:52],np.c_[tr['y'],a][:52]); print('intercept,p,q',C[:9]); print('cross',C[9:].T.reshape(3,4,4)); print('telemetry',s['telemetry_vec']);
for label,te in [('fixed',np.arange(52,72))]+[(f'partition{i}',p[:20]) for i,p in enumerate([np.random.default_rng(i).permutation(72) for i in range(10)])]+[('edge'+str(j),np.argsort(abs(tr['x'][:,j]-np.median(tr['x'][:,j])))[-20:]) for j in range(4)]:
 ix=np.setdiff1d(np.arange(72),te); pred=m.solve_batch(dict(s,train={k:(v if k=='summary' else v[ix]) for k,v in tr.items()},test={'x':tr['x'][te]})); yr=abs(pred['y_hat']-tr['y'][te]); sr=abs(pred['state_pred']-tr['traj'][te]);print(label,'y',yr.mean(),yr.max(),'state',sr.mean(),sr.mean(axis=0).max(),'terminal',sr[:,-1].mean());
 if label=='fixed': assert yr.mean()<=.038
