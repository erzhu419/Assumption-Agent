import copy,pickle,runpy,numpy as np
root='/holdout_task/'
fns=[runpy.run_path(root+('_frozen_nodes/' if i in (0,3) else '')+f'node_{i:02d}.py')['run_node'] for i in range(4)]
s=pickle.load(open(root+'public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; n=len(x); N=np.arange(n)
rng=np.random.default_rng(427); splits=[(f'random-{j}',v) for j,v in enumerate(np.array_split(rng.permutation(n),4))]; splits += [('outer-radius',np.argsort(np.linalg.norm(x,axis=1))[-12:])]
for label,v in splits:
 t=np.setdiff1d(N,v); study=copy.deepcopy(s)
 for key in ('x','y','traj'): study['train'][key]=s['train'][key][t]
 study['train']['summary']=np.array([study['train']['y'].mean(),study['train']['y'].std(),study['train']['traj'][:,-1].mean(),study['train']['traj'][:,-1].std()]); study['test']['x']=x[v]
 shared=fns[0]({'__input__':study}); a=fns[1]({'__input__':study,'node_00':shared}); b=fns[2]({'node_00':shared}); out=fns[3]({'node_01':a,'node_02':b})
 err=abs(out['state_pred']-s['train']['traj'][v]); print(label,round(abs(out['y_hat']-s['train']['y'][v]).mean(),6),round(err.mean(),6),round(err.mean(0).max(),6))
