import importlib.util
import pickle
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

root = '/holdout_task/'
a = load('frozen', root + '_frozen_nodes/node_00.py')
b = load('candidate', root + 'node_01.py')
for batch in pickle.load(open(root + 'public_batch.pkl', 'rb')):
    s = batch['study']
    tr = s['train']; y = tr['y'].ravel(); T = tr['traj']; n = len(y)
    pred = a.run_node({'__input__': s}); q = pred['q_tr']
    p = s['panel_c']; c = b._readout(p['u'], p['v2'], q)
    loo = []
    for j in range(len(p['u'])):
        keep = np.arange(len(p['u'])) != j
        cp = b._readout(p['u'][keep], p['v2'][keep], p['u'][~keep]); loo.append(np.mean(abs(cp-p['v2'][~keep])))
    print('C calibration LOO mean/worst', np.mean(loo), np.max(loo))
    nq = 20; idx = np.arange(n); splits = [('first', idx[:-nq], idx[-nq:])]
    for seed in range(24):
        r = np.random.default_rng(seed).permutation(n)
        splits.append((str(seed),r[:-nq],r[-nq:]))
    for k in range(q.shape[1]):
        r = np.argsort(q[:,k]); splits.extend([(f'q{k} low',r[nq:],r[:nq]),(f'q{k} high',r[:-nq],r[-nq:])])
    d0 = ((q[:,None]-q[None,:])**2).sum(2); dc = ((c[:,None]-c[None,:])**2).sum(2)
    pairs = b._quadratic(q); Kpoly = pairs @ pairs.T
    family = []
    for ell in (.65,.85,1.,1.2,1.45,1.75,2.1):
        for eps in (1e-5,3e-4,3e-3):
            for trend in (0,.1,1.):
                K = np.exp(-.5*(d0+.04*dc)/ell**2) + trend * Kpoly + 10
                family.append(((ell,eps,trend),K))
    errs = []; trajerrs = []; shapeerrs = []
    for name,i,j in splits:
        sub = dict(s, train=dict(tr,x=tr['x'][i],y=tr['y'][i],traj=T[i]),test=dict(s['test'],x=tr['x'][j]))
        result = b.run_node({'__input__': sub, 'node_00': a.run_node({'__input__':sub})})
        assert result['y_hat'].shape == tr['y'][j].shape and result['state_pred'].shape == T[j].shape
        assert result['diagnostics_vec'].shape == (4,) and all(np.isfinite(v).all() for v in result.values())
        base = np.abs(result['y_hat'].ravel()-y[j]); traj = np.abs(result['state_pred']-T[j]); trajerrs.append(traj.mean(0)); shapeerrs.append(base.max())
        options=[]
        for config,K in family:
            eps=config[1]; inv=np.linalg.inv(K[np.ix_(i,i)]+eps*np.eye(len(i))); w=inv @ y[i]
            cv=np.mean(np.abs(w / inv.diagonal())); options.append((cv,config,K[np.ix_(j,i)]@w))
        cv,config,new=min(options,key=lambda x:x[0]); errs.append((name,base.mean(),np.mean(abs(new-y[j])),traj.mean(),config))
    for title, rows in [('first',errs[:1]),('random',errs[1:25]),('tails',errs[25:])]:
        print(title,'mean base/nested/traj',np.mean([r[1:4] for r in rows],axis=0), 'worst',np.max([r[1:4] for r in rows],axis=0)); print('worst cases',max(rows,key=lambda r:r[1]),max(rows,key=lambda r:r[2]))
    print('trajectory worst per horizon', np.max(trajerrs,axis=0)); print('worst single response error',max(shapeerrs))
