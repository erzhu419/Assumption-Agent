import importlib.util, pickle
import numpy as np

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_node

frozen=load('frozen','/holdout_task/_frozen_nodes/node_00.py')
models=[load('baseline','/tmp/node01_dense_baseline.py'),load('candidate','/holdout_task/node_01.py')]
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
t=s['train']; rows=np.arange(len(t['y']))
folds=[]
for seed in (11,23,41,73,111):
    folds.extend(np.array_split(np.random.default_rng(seed).permutation(rows),4))
folds.extend(np.array_split(rows[np.argsort(t['x'][:,d])],6)[j] for d in range(t['x'].shape[1]) for j in (0,-1))
res=[]
for j in folds:
    i=np.setdiff1d(rows,j)
    z=t['traj'][i]; y=t['y'][i]
    study=dict(s,train=dict(x=t['x'][i],y=y,traj=z,summary=np.array([y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()])),test={'x':t['x'][j]})
    features=frozen({'__input__':study})
    scores=[]
    for run in models:
        out=run({'__input__':study,'node_00':features}); yp,zp=out['y_hat'],out['state_pred']
        assert yp.shape==t['y'][j].shape and zp.shape==t['traj'][j].shape
        assert np.isfinite(yp).all() and np.isfinite(zp).all()
        np.testing.assert_allclose(out['diagnostics_vec'],[yp.mean(),yp.std(),zp[:,-1].mean(),zp[:,-1].std()])
        scores.append([np.abs(yp-t['y'][j]).mean(),np.abs(zp-t['traj'][j]).mean()])
    res.append(scores)
a=np.array(res)
for name, b in [('original random',a[:4]),('additional random',a[4:20]),('coordinate tails',a[20:])]:
    print(name, 'mean [baseline,candidate]', b.mean(0).round(6).tolist(), 'worst [baseline,candidate]',b.max(0).round(6).tolist())
