import numpy as np,pickle,runpy
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=len(s['train']['x']); k=runpy.run_path('/holdout_task/_frozen_nodes/node_02.py')['run_node']; p=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']; folds=[np.arange(52,72),*np.array_split(np.random.default_rng(42).permutation(n),4)]
for f in ['/tmp/baseline11.py','/tmp/challenger11.py']:
 r=runpy.run_path(f)['run_node']; result=[]
 for j in folds:
  i=np.setdiff1d(np.arange(n),j); st=dict(s,train={a:s['train'][a][i] for a in ('x','y','traj')},test={'x':s['train']['x'][j]}); o=k({'node_00':r({'__input__':st}),'node_01':p({'__input__':st})}); result.append([np.mean(abs(o['y_hat']-s['train']['y'][j])),np.mean(abs(o['state_pred']-s['train']['traj'][j]))]);
 print(f,np.round(result,6).tolist(),flush=True)
