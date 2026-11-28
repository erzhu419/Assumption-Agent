import sys,pickle,runpy,numpy as np
from metric import _map
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; lat=runpy.run_path('/tmp/raw_node.py')['_latent']; k=runpy.run_path('/holdout_task/_frozen_nodes/node_02.py')['run_node']; p=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']; n=len(s['train']['x']); folds=[np.arange(52,72),*np.array_split(np.random.default_rng(42).permutation(n),4)]
for j in folds:
 i=np.setdiff1d(np.arange(n),j); st=dict(s,train={a:s['train'][a][i] for a in ('x','y','traj')},test={'x':s['train']['x'][j]}); x=lat(st)
 for mode in range(3):
  cand=_map(x,st,mode); out=[]
  for h,z in cand[:4]:
   o=k({'node_00':{'z':z[:len(i)],'zt':z[len(i):]},'node_01':p({'__input__':st})}); out.append([h,np.mean(abs(o['y_hat']-s['train']['y'][j])),np.mean(abs(o['state_pred']-s['train']['traj'][j]))])
  print(j[0],mode,np.round(out,5).tolist(),flush=True)
