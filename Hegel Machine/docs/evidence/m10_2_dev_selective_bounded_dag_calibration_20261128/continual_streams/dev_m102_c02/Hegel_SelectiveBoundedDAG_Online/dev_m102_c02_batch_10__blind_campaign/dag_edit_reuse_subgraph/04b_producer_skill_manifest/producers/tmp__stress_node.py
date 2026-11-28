import pickle,runpy,numpy as np
from time import perf_counter
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; f=runpy.run_path('/holdout_task/node_00.py')['run_node']; g=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']; tr=s['train']; n=len(tr['x']); rng=np.random.default_rng(420)
def score(st, i,j):
 p=dict(st,train={k:tr[k][i] for k in ('x','y','traj')},test={'x':tr['x'][j]}); r=g({'__input__':p,'node_00':f({'__input__':p})}); e=np.abs(r['state_pred']-tr['traj'][j]); return np.mean(abs(r['y_hat']-tr['y'][j])),e.mean(),e.mean(0).max()
start=perf_counter(); folds=[]
for k in range(8):
 for j in np.array_split(rng.permutation(n),4): folds.append(score(s,np.setdiff1d(np.arange(n),j),j))
print('32 CV splits: worst (scalar, trajectory, horizon)',np.max(folds,0),'average',np.mean(folds,0))
a=range(n-20); b=range(n-20,n); folds=[]
for name in ['panel_a','panel_b']:
 folds=[]
 for j in range(len(s[name]['u'])):
  st=dict(s); st[name]={key:np.delete(value,j,axis=0) for key,value in s[name].items()}; folds.append(score(st,a,b))
 print(name,'calibration jackknife worst',np.max(folds,0),'worst row by metric',np.argmax(folds,0))
print('stress elapsed',perf_counter()-start)
