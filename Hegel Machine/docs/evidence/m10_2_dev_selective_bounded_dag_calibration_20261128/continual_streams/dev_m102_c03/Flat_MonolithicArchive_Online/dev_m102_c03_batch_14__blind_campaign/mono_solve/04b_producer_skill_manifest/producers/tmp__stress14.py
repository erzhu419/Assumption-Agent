import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); ix=np.arange(n)
splits=[('required',ix[52:])]+[(f'random{i}',np.random.default_rng(i).permutation(n)[52:]) for i in range(20)]
for j in range(4):
 order=np.argsort(tr['x'][:,j]); splits += [(f'x{j}_tails',np.r_[order[:10],order[-10:]]),(f'x{j}_low',order[:20]),(f'x{j}_high',order[-20:])]
results=[]
for name,test in splits:
 train=np.setdiff1d(ix,test); out=solve_batch(dict(s,train={k:v if k=='summary' else v[train] for k,v in tr.items()},test={'x':tr['x'][test]}))
 y=np.abs(out['y_hat']-tr['y'][test]); t=np.abs(out['state_pred']-tr['traj'][test]); results.append((name,y.mean(),t.mean(),t.mean(0).max(),y.max()))
print('required',results[0]); print('worst y',max(results,key=lambda v:v[1])); print('worst trajectory',max(results,key=lambda v:v[2])); print('worst single horizon',max(results,key=lambda v:v[3])); print('worst random',max(max(v[1:4]) for v in results[1:21])); print('test output shape',[(k,v.shape) for k,v in solve_batch(s).items()])
