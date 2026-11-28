import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,T=(tr[k] for k in ('x','y','traj'))
def evaluate(a,c):
 sub={**s,'train':{'x':x[a],'y':y[a],'traj':T[a]},'test':{'x':x[c]}};o=m.solve_batch(sub); return abs(o['y_hat']-y[c]),abs(o['state_pred']-T[c]);
def scores(e): return np.array([e[0].mean(),e[1].mean(axis=0).max()])
print('56/16',scores(evaluate(np.arange(56),np.arange(56,72))))
for seed in (932, 83):
 all_idx=np.arange(len(x)); err=[]
 for c in np.array_split(np.random.RandomState(seed).permutation(len(x)),6):
  err.append(evaluate(np.setdiff1d(all_idx,c),c))
 print('seed',seed,'OOF',scores(tuple(np.concatenate(v) for v in zip(*err))),'worst',np.max([scores(e) for e in err],axis=0))
h,j=m._panel_features(s,x)
for k,v in enumerate((np.linalg.norm(m._calibrate(s['panel_a'],x),axis=1),(h*j).sum(1))):
 order=np.argsort(v); print('edge',k,'low/high',*[scores(evaluate(np.setdiff1d(np.arange(len(x)),c),c)) for c in (order[:12],order[-12:])])
