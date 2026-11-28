import importlib.util,pickle,numpy as np, sys
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00,node_02
spec=importlib.util.spec_from_file_location('edited','/holdout_task/node_01.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['y']); ix=np.arange(n)
for seed in [102,701,208]:
 rng=np.random.default_rng(seed); splits=np.array_split(rng.permutation(n),4)
 if seed==102: splits=[np.arange(52,n)]+splits
 vals=[]
 for j in splits:
  i=np.setdiff1d(ix,j); p=node_00.run_node({'__input__':dict(s,train={k:t[k][i] for k in ('x','y','traj')},test={'x':t['x'][j]})}); r=node_02.run_node({'node_00':p}); y=m.run_node({'node_00':p,'node_02':r})['y_hat']; vals.append((np.mean(abs(y-t['y'][j])), np.mean(abs(r['state_pred']-t['traj'][j]))))
 print(seed,np.round(vals,6),'worst',np.max(vals,axis=0))
