import importlib.util, pickle, numpy as np

def load(path):
    spec = importlib.util.spec_from_file_location('candidate', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

old = load('/tmp/batch10_dense.py')
new = load('/holdout_task/batch_solver.py')
s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
tr = s['train']; x = tr['x']; y = np.c_[tr['y'],tr['traj']]; w = new._panels(s,new._coordinates(s,x))
rng = np.random.default_rng(302)
splits = [('primary',np.arange(len(x)))]+[('random',rng.permutation(len(x))) for _ in range(24)]
splits += [('tail',np.argsort(np.linalg.norm(v,axis=1))) for v in (x,w)]
splits += [('coordinate',np.argsort(sign*x[:,j])) for j in range(x.shape[1]) for sign in (1,-1)]
for module in (old,new):
    groups={}
    for label, ix in splits:
        i,j = ix[:52],ix[52:]
        o = module.solve_batch({**s,'train':{**tr,**{k:tr[k][i] for k in ('x','y','traj')}},'test':{'x':x[j]}})
        d = np.abs(np.c_[o['y_hat'],o['state_pred']]-y[j]); groups.setdefault(label,[]).append([d[:,0].mean(),d[:,1:].mean(),d[:,1:].mean(0).max()])
    print(module.__file__, {k:np.max(l,axis=0).tolist() for k,l in groups.items()})
    if module is new:
        assert np.max(np.concatenate(list(groups.values()))) <= 0.038
        o = new.solve_batch(s); a, b = o['y_hat'], o['state_pred'][:,-1]
        assert np.allclose(o['diagnostics_vec'],[a.mean(),a.std(),b.mean(),b.std()],rtol=0,atol=1e-12)
        print('public-test ranges:',(a.min(),a.max()),(b.min(),b.max()))
