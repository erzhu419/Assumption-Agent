import copy, importlib.util, pickle, sys
import numpy as np

def load(path):
    spec = importlib.util.spec_from_file_location('solver', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.solve_batch

s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
x = s['train']['x']; ii = np.arange(len(x))
folds = [('prescribed',ii[52:]), ('high-norm',np.argsort(np.linalg.norm(x,axis=1))[-20:]), ('coordinate-edge',np.argsort(abs(x).max(1))[-20:]), ('first-20',ii[:20])]
folds += [(f'random{i}',np.random.default_rng(i).choice(len(x),20,False)) for i in range(8)]
folds += [(f'axis{j}{sign:+}', np.argsort(sign*x[:,j])[-20:]) for j in range(4) for sign in [1,-1]]
for path in sys.argv[1:]:
    solve = load(path); m=[]; print(path)
    for label,te in folds:
        tr=np.setdiff1d(ii,te); st=copy.deepcopy(s)
        st['train']={k:s['train'][k][tr] for k in ('x','y','traj')}; st['test']={'x':x[te]}
        out=solve(st); y=out['y_hat']; z=out['state_pred']; e=abs(z-s['train']['traj'][te]); a=[abs(y-s['train']['y'][te]).mean(),e.mean(),e.mean(0).max()]; m.append(a)
        assert np.allclose(out['diagnostics_vec'],[y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()]); print(label, np.round(a,5))
    print('max',np.max(m,0),'mean',np.mean(m,0))
    assert m[0][0] <= .038
