import copy, importlib.util, pickle, numpy as np

def load(path):
    spec=importlib.util.spec_from_file_location('candidate',path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
root='/holdout_task/'
a,m,b=[load(root+p) for p in ['_frozen_nodes/node_00.py','node_01.py','_frozen_nodes/node_02.py']]
s=pickle.load(open(root+'public_batch.pkl','rb'))[0]['study']; Y=s['train']['y']; T=s['train']['traj']; n=len(Y); ix=np.arange(n)
folds=[('public',ix>=n-20)]+[(f'block{i}',ix//18==i) for i in range(4)]+[(f'interleaved{i}',ix%4==i) for i in range(4)]
Z=s['train']['x']; lead=np.linalg.norm((Z-np.median(Z,0))/np.maximum(np.std(Z,0),1e-9),axis=1); folds += [('high-leverage',lead>=np.sort(lead)[-14])]
all_errors=[]
for name,hold in folds:
    d=copy.deepcopy(s); d['train']={k:s['train'][k][~hold] for k in ['x','y','traj']}; d['test']={'x':Z[hold]}
    raw=a.run_node({'__input__':d}); fitted=m.run_node({'__input__':d,'node_00':raw}); out=b.run_node({'node_00':raw,'node_01':fitted})
    ye=np.abs(out['y_hat']-Y[hold]); te=np.abs(out['state_pred']-T[hold]); assert np.isfinite(ye).all() and np.isfinite(te).all()
    last=out['state_pred'][:,-1]; np.testing.assert_allclose(out['diagnostics_vec'],[out['y_hat'].mean(),out['y_hat'].std(),last.mean(),last.std()]); row=[ye.mean(),te.mean(),te.mean(0).max()]; all_errors.append(row); print(name,np.round(row,6))
print('worst slice endpoint/trajectory/horizon',np.round(np.max(all_errors,axis=0),6))
