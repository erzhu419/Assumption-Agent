import pickle, runpy, sys
import numpy as np
root='/holdout_task/'
s=pickle.load(open(root+'public_batch.pkl','rb'))[0]['study']
node=runpy.run_path(sys.argv[1] if len(sys.argv)>1 else root+'node_00.py')['run_node']
sink=runpy.run_path(root+'_frozen_nodes/node_01.py')['run_node']
F=node({'__input__':s}); t=s['train']; n=len(t['x']); z=F['z'][:n]
rng=np.random.default_rng(35); inds=np.arange(n)
slices={'official':inds[-20:], 'large_latent_norm':np.argsort(np.linalg.norm(z,axis=1))[-16:], 'large_latent_coordinate':np.argsort(np.max(np.abs(z),axis=1))[-16:]}
for j, idx in enumerate(np.array_split(rng.permutation(n),4)): slices[f'fold{j}']=idx
for j in range(z.shape[1]):
 o=np.argsort(z[:,j]); slices[f'axis{j}_low']=o[:14]; slices[f'axis{j}_high']=o[-14:]
worst=np.zeros(2)
for k,te in slices.items():
 tr=np.setdiff1d(inds,te); st={**s,'train':{a:t[a][tr] for a in ('x','y','traj')},'test':{'x':t['x'][te]}}; ft=node({'__input__':st}); assert set(ft)=={'z','L','Q','R','product','cosine'} and all(v.shape==(n,4) and np.all(np.isfinite(v)) for v in ft.values())
 p=sink({'__input__':st,'node_00':ft}); y=np.mean(abs(p['y_hat']-t['y'][te])); e=np.mean(abs(p['state_pred']-t['traj'][te]),axis=0); worst=np.maximum(worst,[y,e.mean()]); print(k,round(y,6),round(e.mean(),6),round(max(e),6))
print('worst',worst)
