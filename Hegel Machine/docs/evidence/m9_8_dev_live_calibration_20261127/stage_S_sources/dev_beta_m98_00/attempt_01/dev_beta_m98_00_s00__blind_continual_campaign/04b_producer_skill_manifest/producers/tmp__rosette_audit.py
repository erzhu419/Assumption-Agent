import importlib.util
import itertools
import pickle
import numpy as np

p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', p + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open(p + 'public_batch.pkl', 'rb'))[0]
s = item['study']
r = s['sensor_calibration_sweeps']['ref_rosette']
y = s['sensor_calibration_sweeps']['raw_rosette']
single = np.count_nonzero(r, axis=1) == 1
rng = np.random.default_rng(98)
folds = [np.r_[i,j] for i,j in zip(np.array_split(rng.permutation(np.flatnonzero(single)), 4), np.array_split(rng.permutation(np.flatnonzero(~single)), 4))]
aff = np.c_[np.ones(len(r)), r]
pairs = list(itertools.combinations(range(4), 2))
contrasts = [r[:,i] + sign*r[:,j] for i,j in pairs for sign in (-1,1)]
models = {'affine': aff, 'pair_products_and_curvature': np.column_stack([aff, r*r] + [r[:,i]*r[:,j] for i,j in pairs])}
for a in (.001,.01,.1):
    for j,q in enumerate([*r.T, *contrasts]):
        models[f'quotient_{a}_{j}'] = np.column_stack([aff, r*q[:,None]**2/(1+a*q[:,None]**2)])

def predict(x, fit, test):
    return x[test] @ np.linalg.lstsq(x[fit], y[fit], rcond=None)[0]

def nested(names):
    errors = np.empty_like(y)
    selected = []
    for k,test in enumerate(folds):
        train_folds = [fold for j,fold in enumerate(folds) if j != k]
        fit = np.concatenate(train_folds)
        def score(name):
            x = models[name]
            return sum(np.sum((predict(x, np.concatenate([b for j,b in enumerate(train_folds) if i!=j]), a)-y[a])**2) for i,a in enumerate(train_folds))
        name = min(names, key=score)
        selected.append(name)
        errors[test] = predict(models[name],fit,test)-y[test]
    slices = [np.sqrt(np.mean(errors[mask]**2,axis=0)) for mask in (single,~single) if mask.any()]
    return float(np.sqrt(np.mean(errors**2))), float(np.max(slices)), selected

print('Calibration pooled and worst sweep-type/channel outer RMSE:')
for label,names in [('affine',['affine']), ('paired curvature',['pair_products_and_curvature']), ('rational',[n for n in models if n.startswith('quotient')])]:
    print(label, nested(names))
X,_ = m._design(s)
yc = m._calibrate(s)
z = s['Z_nuisance']
P = np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
out = m.solve_batch(s)
res = X@out['weights']-yc
raw = lambda t: [float(np.sqrt(np.mean(a*a))) for a in np.split(t,len(s['mu_shifts']))]
print('Study keys:', sorted(s))
print('Measured residual RMSE by shift, before and after profiling:', raw(res), raw(res-z@(P@res)))
print('Public certificate gap:', out['kkt_certificate']['primal_objective']-out['kkt_certificate']['dual_lower_bound'])
