import importlib.util
import pickle
import numpy as np

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

m=load('solver','/holdout_task/batch_solver.py')
old=load('reference','/tmp/batch_reference.py')
with open('/holdout_task/public_batch.pkl','rb') as f:
    s=pickle.load(f)[0]['study']
t=s['train']; x=t['x']
def check(f,r):
    y,z,d=(r[k] for k in ('y_hat','state_pred','diagnostics_vec'))
    n=len(f['test']['x'])
    assert y.shape==(n,) and z.shape==(n,len(f['eval_times'])) and d.shape==(4,)
    assert all(isinstance(v,np.ndarray) and np.isfinite(v).all() for v in (y,z,d))
    np.testing.assert_array_equal(d,[y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()])
    return y,z
check(s,m.solve_batch(s))
a=s['panel_a']; shift,gain,*_=m._operators(s)
r=np.linalg.norm(a['v']-shift-a['u']*gain,axis=1)
print('Panel A final diagonal fit:', (r<.00034).sum(),'inliers below .00034;', (r>12).sum(),'outliers above 12')
assert (r<.00034).sum()==45 and (r>12).sum()==3
folds=[np.arange(56,72),np.arange(16),np.arange(28,44),np.argsort(np.linalg.norm(x,axis=1))[-16:],*[np.random.default_rng(seed).choice(72,16,replace=False) for seed in (76,174,419)]]
for solver in [old,m]:
    es=[]
    for j in folds:
        i=np.setdiff1d(np.arange(len(x)),j)
        f={**s,'train':{**t,**{k:t[k][i] for k in ('x','y','traj')}},'test':{'x':x[j]}}
        y,z=check(f,solver.solve_batch(f)); e=np.abs(z-t['traj'][j])
        es.append([np.abs(y-t['y'][j]).mean(),e.mean(),e[:,-1].mean()])
    print(solver.__name__,np.round(es,5),'mean',np.mean(es,0),'worst',np.max(es,0))
