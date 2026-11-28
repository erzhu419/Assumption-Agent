import importlib.util
import pickle
from itertools import combinations
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m = load('candidate', '/holdout_task/batch_solver.py')
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
tr = s['train']; x, y, T = (tr[k] for k in ('x', 'y', 'traj'))
z = m._decode(x, m._calibration(s['panel_a']))
p, b = m._operator(s['panel_b'])
w = m._affine(m._warp(z, p)) @ b
D = m._readout(w)
fit = lambda X, Y: np.linalg.lstsq(X, Y, rcond=None)[0]
mae = lambda a, b: np.mean(np.abs(a-b))
ids = np.arange(len(x))
print('study fields:', sorted(s), 'train fields:', sorted(tr), 'telemetry shape:', s['telemetry_vec'].shape)

# Explicitly compare individual stages to their composition with the same readout families.
i, j = ids[:52], ids[52:]
sub = dict(s, train={k: v[i] if k != 'summary' else v for k,v in tr.items()}, test={'x':x[j]})
out = m.solve_batch(sub)
for name, X in [('raw', x), ('calibration_only', z), ('threshold_without_calibration', m._affine(m._warp(x,p)) @ b), ('composed', w)]:
    A = m._readout(X)
    print('stage_ablation',name,'scalar',mae(A[j] @ fit(A[i],y[i]), y[j]), 'linear_state',mae(X[j] @ fit(X[i],T[i]), T[j]))
print('artifact_direct', mae(out['y_hat'],y[j]), mae(out['state_pred'],T[j]))

# Equip the challenger with all independent pairwise products and squares of sums/differences.
P = np.column_stack([w[:,k]*w[:,l] for k in range(4) for l in range(k,4)])
Q = [w[:,k] for k in range(4)] + [w[:,k]+sign*c*w[:,l] for k,l in combinations(range(4),2) for sign in (-1,1) for c in (.5,1,2)]
R = [(f'q{k}:a{a}', D/(1+a*q[:,None]**2)) for k,q in enumerate(Q) for a in (.01,.05,.2,1.)]
families = {'sparse':[('reference',D)], 'all_quadratic':[('affine-quadratic',np.column_stack([np.ones(len(x)),w,P]))], 'rational':R}
rng = np.random.default_rng(2407)
slices = {'prescribed':j,'high_operator_norm':np.argsort(np.linalg.norm(w,axis=1))[-20:], 'activation_boundary':np.argsort(np.min(np.abs(np.abs(z)-p[0]),axis=1))[:20]}
slices.update({f'random{k}':rng.permutation(ids)[-20:] for k in range(8)})
errors = {key: [] for key in families}; signed = []
for name, va in slices.items():
    ti = np.setdiff1d(ids,va)
    inner = np.array_split(rng.permutation(ti), 4)
    splits = [(np.setdiff1d(ti,k),k) for k in inner]
    for family, pool in families.items():
        label, A = min(pool, key=lambda pp: np.mean([mae(pp[1][v] @ fit(pp[1][t], y[t]),y[v]) for t,v in splits]))
        pred = A[va] @ fit(A[ti],y[ti]); err = mae(pred,y[va]); errors[family].append(err)
        if family == 'sparse': signed.append((name,float(np.mean(pred-y[va]))))
    print(name, *(f'{key}={errors[key][-1]:.6f}' for key in families))
for key, seq in errors.items(): print('family',key,'mean',np.mean(seq),'worst',max(seq))
print('largest_absolute_slice_bias',max(signed,key=lambda pair:abs(pair[1])))
assert mae(out['y_hat'],y[j]) < .038 and mae(out['state_pred'],T[j]) < .038
assert max(errors['sparse']) < .038
