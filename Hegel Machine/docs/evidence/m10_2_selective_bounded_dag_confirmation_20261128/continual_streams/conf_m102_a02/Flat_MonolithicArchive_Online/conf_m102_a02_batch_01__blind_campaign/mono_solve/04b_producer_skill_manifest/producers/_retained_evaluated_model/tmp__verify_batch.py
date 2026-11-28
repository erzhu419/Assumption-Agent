import sys, pickle, importlib.util
import numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch, _calibration
spec=importlib.util.spec_from_file_location('baseline','/tmp/baseline_solver.py')
baseline=importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,z=[tr[k] for k in ('x','y','traj')]; b,d=_calibration(s['panel_a']); X=(x-b)/d
folds=[np.arange(len(x))]+[np.random.default_rng(i).permutation(len(x)) for i in range(1,41)]
for f in [baseline.solve_batch,solve_batch]:
    v=[]
    for q in folds:
        a,c=q[:52],q[52:]; r=f(dict(s,train=dict(x=x[a],y=y[a],traj=z[a]),test={'x':x[c]})); ey=abs(r['y_hat']-y[c]); ez=abs(r['state_pred']-z[c]); m=abs(X[c]).max(1)>=1
        v.append([ey.mean(),ez.mean(),ez.mean(0).max(),ey[m].mean(),ez[m].mean()])
    v=np.array(v)
    print(f.__module__,'columns y, traj, worst timestamp, edge y, edge traj');print('52/20',v[0]); print('resplit 1:12 max',v[:12].max(0));print('new resplits max',v[12:].max(0)); print('new resplits mean',v[12:].mean(0))
    if f is solve_batch:
        assert np.max(v[0,:3]) <= .038
        o=f(s); yy,zz=o['y_hat'],o['state_pred']; assert yy.shape==(36,) and zz.shape==(36,8)
        assert all(np.isfinite(k).all() for k in o.values()); np.testing.assert_array_equal(o['diagnostics_vec'],[yy.mean(),yy.std(),zz[:,-1].mean(),zz[:,-1].std()])
