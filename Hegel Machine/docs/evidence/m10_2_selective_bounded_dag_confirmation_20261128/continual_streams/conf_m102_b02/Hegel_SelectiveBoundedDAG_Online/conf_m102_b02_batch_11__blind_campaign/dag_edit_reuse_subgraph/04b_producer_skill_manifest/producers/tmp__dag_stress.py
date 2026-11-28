import copy, pickle, runpy, numpy as np
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; a=s['train']
nodes=[runpy.run_path(p+('' if i==0 else '_frozen_nodes/')+f'node_{i:02}.py')['run_node'] for i in range(4)]
def cv(tr,te):
 d=copy.deepcopy(s); d['train'].update({k:a[k][tr] for k in ['x','y','traj']}); d['test']['x']=a['x'][te]
 i={'__input__':d}; o=nodes[3]({'node_01':nodes[1]({'node_00':nodes[0](i)}),'node_02':nodes[2](i)})
 return [np.abs(o['y_hat']-a['y'][te]).mean(), np.abs(o['state_pred']-a['traj'][te]).mean()]
n=len(a['x']); idx=np.arange(n)
for key, tests in [('random',[te for seed in range(8) for te in np.array_split(np.random.default_rng(seed).permutation(n),4)]),('large radius',[np.argsort(np.linalg.norm(a['x'],axis=1))[-14:]]),('coordinate tails',[np.argsort(sign*a['x'][:,j])[-12:] for j in range(4) for sign in (1,-1)])]:
 v=np.array([cv(np.setdiff1d(idx,te),te) for te in tests]); print(key,len(v),'mean',v.mean(0),'worst',v.max(0))
