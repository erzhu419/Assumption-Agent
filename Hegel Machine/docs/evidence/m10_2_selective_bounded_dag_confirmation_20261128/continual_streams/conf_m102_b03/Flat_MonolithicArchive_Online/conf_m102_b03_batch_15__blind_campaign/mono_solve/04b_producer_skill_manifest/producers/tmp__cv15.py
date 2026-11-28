import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; T=tr['traj']; y=tr['y']
rng=np.random.default_rng(8)
slices=[np.arange(72), *[rng.permutation(72) for _ in range(12)], np.argsort(np.linalg.norm(tr['x'],axis=1)), np.argsort(np.linalg.norm(tr['x'],axis=1))[::-1]]
for i, idx in enumerate(slices):
 a,b=idx[:52],idx[52:]; d={**s,'train':{'x':tr['x'][a],'y':y[a],'traj':T[a],'summary':np.array([y[a].mean(),y[a].std(),T[a,-1].mean(),T[a,-1].std()])},'test':{'x':tr['x'][b]}}
 o=solve_batch(d); e=np.abs(o['y_hat']-y[b]); E=np.abs(o['state_pred']-T[b]);print(i,e.mean(),E.mean(),E.mean(0).max(),E[:,-1].mean())
