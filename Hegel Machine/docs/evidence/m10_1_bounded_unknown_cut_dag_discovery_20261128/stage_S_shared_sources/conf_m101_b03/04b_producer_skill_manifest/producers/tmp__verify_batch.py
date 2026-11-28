import importlib.util,pickle,numpy as np,sys
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; tr=s['train']; ids=np.arange(len(tr['y'])); rng=np.random.default_rng(14)
for val in [ids[52:],*np.array_split(rng.permutation(ids),4)]:
 train=np.setdiff1d(ids,val); t={k:v[train] if k!='summary' else v for k,v in tr.items()}; out=m.solve_batch(dict(s,train=t,test={'x':tr['x'][val]})); e=abs(out['y_hat']-tr['y'][val]); g=abs(out['state_pred']-tr['traj'][val]);print('split',len(train),len(val),'scalar',e.mean(),'state',g.mean(),'time',g.mean(0),'scalar max',max(e),flush=True)
