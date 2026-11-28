import numpy as np, pickle, runpy
s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
node = runpy.run_path('/holdout_task/node_01.py')
upfn = runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']
x,y,z = (s['train'][k] for k in ('x','y','traj'))
k = np.arange(len(x)); q = node['_paired_coordinates'](x,s['panel_a'],s['telemetry_vec'][0]); tests={}
for j, ids in enumerate(np.array_split(np.random.default_rng(12).permutation(len(x)),5)): tests[f'random{j}'] = ids
for j in range(q.shape[1]):
    order = np.argsort(q[:, j]); tests[f'low{j}']=order[:14]; tests[f'high{j}']=order[-14:]
last=[]
for label, ids in tests.items():
    tr = np.setdiff1d(k, ids)
    batch = dict(s, train=dict(s['train'], x=x[tr], y=y[tr], traj=z[tr]), test={'x':x[ids]})
    out = node['run_node']({'__input__': batch, 'node_00': upfn({'__input__':batch})})
    ey=np.mean(abs(y[ids]-out['y_hat'])); ez=np.mean(abs(z[ids]-out['state_pred']),axis=0); last.append((ey,ez.mean(),ez.max())); print(label,np.round(last[-1],6))
    assert out['y_hat'].shape==y[ids].shape and out['state_pred'].shape==z[ids].shape and out['diagnostics_vec'].shape==(4,) and all(np.isfinite(a).all() for a in out.values())
print('max random',np.max(last[:5],axis=0),'max tail',np.max(last[5:],axis=0))
