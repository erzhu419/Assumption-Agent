import numpy as np,pickle,runpy
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
root='/holdout_task/'
s=pickle.load(open(root+'public_batch.pkl','rb'))[0]['study']; A=runpy.run_path(root+'_frozen_nodes/node_00.py')['run_node']; B=runpy.run_path(root+'_frozen_nodes/node_01.py')['run_node']; C=runpy.run_path(root+'node_02.py'); predict=C['_predict']; run=C['run_node']; y=s['train']['y']; state=s['train']['traj']; raw=s['train']['x']; x=A({'__input__':s})['x'][:len(y)]; n=len(y); rng=np.random.default_rng(18); i=np.arange(n); folds=[i[52:]]+list(np.array_split(rng.permutation(n),4))+[np.argsort(np.linalg.norm(raw,axis=1))[-14:]]
for te in folds:
 tr=np.setdiff1d(i,te); st=dict(s,train={k:s['train'][k][tr] for k in ('x','y','traj')},test={'x':raw[te]}); a=A({'__input__':st}); b=B({'node_00':a}); p=run({'__input__':st,'node_00':a,'node_01':b}); e=np.abs(p['state_pred']-state[te]); print('candidate',round(np.abs(p['y_hat']-y[te]).mean(),6),round(e.mean(),6),round(e.mean(0).max(),6))
