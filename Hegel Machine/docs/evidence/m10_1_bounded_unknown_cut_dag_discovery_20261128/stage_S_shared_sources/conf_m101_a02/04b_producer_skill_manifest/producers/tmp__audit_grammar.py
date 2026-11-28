import importlib.util
import pickle
from itertools import combinations_with_replacement
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
x, y, traj = (s['train'][k] for k in ('x', 'y', 'traj'))
offset, gain = m._calibration(s['panel_a'])
pb = s['panel_b']
z = m._affine((x-offset)/gain) @ m._fit(m._affine(pb['u']), pb['v'])
a = m._affine(z)
pairs = ((0,2,1,3), (0,1,2,3), (0,3,1,2))
base = m._label_basis(z)
models = {
    'affine': a,
    'tied': base,
    'tied_no_offset': base[:,1:],
    'full_quadratic': np.column_stack([a, *[z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(4),2)]]),
    'tied_plus_triples': np.column_stack([base, *[np.prod(np.delete(z,i,axis=1),axis=1) for i in range(4)]]),
}
for i,j,k,l in pairs:
    p, q = z[:,i]*z[:,j], z[:,k]*z[:,l]
    models[f'pair{i}{j}_sum'] = np.column_stack((a,p+q))
    models[f'pair{i}{j}_difference'] = np.column_stack((a,p-q))
    models[f'pair{i}{j}_untied'] = np.column_stack((a,p,q))
for i,j,k,l in pairs:
    for left,right in ((i,j),(k,l)):
        for sign in (-1,1):
            q2 = (z[:,left]+sign*z[:,right])**2
            for strength in (0.003,0.015,0.06):
                models[f'rational_{left}_{right}_{sign}_{strength}'] = base / (1 + strength*q2[:,None])

def err(design, tr, va):
    return np.abs(y[va] - design[va] @ m._fit(design[tr], y[tr]))

rows = np.arange(len(x))
splits = [('fixed', rows[52:]), *[(f'fold{i}', v) for i,v in enumerate(np.array_split(np.random.default_rng(190).permutation(rows),4))], ('outer-input', np.argsort(np.linalg.norm(x,axis=1))[-18:])]
for name, va in splits:
    tr = np.setdiff1d(rows,va)
    inner = [(np.setdiff1d(tr,v),v) for v in np.array_split(np.random.default_rng(2026).permutation(tr),4)]
    def select(keys):
        return min(keys,key=lambda k: np.concatenate([err(models[k],it,iv) for it,iv in inner]).mean())
    alternatives = [k for k in models if not k.startswith('rational') and k not in ('affine','tied','pair02_sum')]
    structur = select(alternatives)
    rational = select([k for k in models if k.startswith('rational')])
    result = m.solve_batch(dict(s,train={'x':x[tr],'y':y[tr],'traj':traj[tr]},test={'x':x[va]}))
    pred_err = np.abs(y[va]-result['y_hat'])
    np.testing.assert_allclose(pred_err,err(base,tr,va),atol=1e-12)
    state_err = np.abs(traj[va]-result['state_pred']).mean(axis=0)
    print(f'{name}: affine={err(a,tr,va).mean():.6f}, solver={pred_err.mean():.6f}, quadratic={err(models["full_quadratic"],tr,va).mean():.6f}, no_offset={err(models["tied_no_offset"],tr,va).mean():.6f}, selected_structure={structur}:{err(models[structur],tr,va).mean():.6f}, selected_rational={rational}:{err(models[rational],tr,va).mean():.6f}, worst_horizon={state_err.max():.6f}')
    assert max(pred_err.mean(),state_err.max()) <= 0.038
print('Study fields:', sorted(s), '; training fields:',sorted(s['train']))
