from pathlib import Path
from copy import deepcopy
import pickle
import numpy as np
root=Path('/holdout_task')
ns={'__file__':str(root/'test_dag_candidate.py')}
exec(compile((root/'test_dag_candidate.py').read_text().split('st = pickle.load')[0], ns['__file__'], 'exec'),ns)
s=pickle.loads((root/'public_batch.pkl').read_bytes())[0]['study']; tr=s['train']; n=len(tr['x']); print('train size',n)
# Use response-blind folds and a covariate-space boundary slice.
x=tr['x']; far=np.argsort(np.linalg.norm((x-np.median(x,axis=0))/np.std(x,axis=0),axis=1))[-min(15,n//4):]
folds=[(f'random-{j}', idx) for j,idx in enumerate(np.array_split(np.random.default_rng(214).permutation(n),5))]+[('boundary',far)]
for name,ii in folds:
    jj=np.setdiff1d(np.arange(n),ii); st=deepcopy(s)
    st['train']={k:tr[k][jj] for k in ('x','y','traj')}; y=tr['y'][jj]; t=tr['traj'][jj,-1]; st['train']['summary']=np.array([y.mean(),y.std(),t.mean(),t.std()]); st['test']={'x':x[ii]}
    p,ec=ns['run_dag'](st); assert all(v==1 for v in ec.values())
    e=np.abs(p['state_pred']-tr['traj'][ii]); yh=p['y_hat']; terminal=p['state_pred'][:,-1]
    np.testing.assert_allclose(p['diagnostics_vec'], [yh.mean(),yh.std(),terminal.mean(),terminal.std()])
    print(name, 'y_mae',np.mean(abs(yh-tr['y'][ii])), 'traj_mae',e.mean(), 'worst_time_mae',np.max(e.mean(axis=0)))
