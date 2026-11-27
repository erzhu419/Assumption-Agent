import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
c = s['sensor_calibration_sweeps']
x, y = (np.asarray(c[k]) for k in ('ref_rosette', 'raw_rosette'))
rows = np.arange(len(x))
base = np.c_[np.ones(len(x)), x]

# Compare the affine calibration with a nested test for saturating pairwise terms.
def design(pair, alpha, train):
    if pair is None:
        return base
    i, j, sign = pair
    q = x[:, i] if j is None else x[:, i] + sign*x[:, j]
    q = q / np.sqrt(np.mean(q[train]**2))
    return np.c_[base, q*q/(1 + alpha*q*q), q**3/(1 + alpha*q*q)]

def error(train, test, pair=None, alpha=0.):
    A = design(pair, alpha, train)
    return y[test] - A[test] @ np.linalg.lstsq(A[train], y[train], rcond=None)[0]

pairs = [(i,None,1) for i in range(4)] + [(i,j,sign) for i in range(4) for j in range(i+1,4) for sign in (-1,1)]
options = [(None,0.)] + [(p,a) for p in pairs for a in (0., 0.125, 0.5, 2.)]
E0 = np.empty_like(y)
E1 = np.empty_like(y)
selected = []
for fold in range(5):
    test = rows % 5 == fold
    train = ~test
    groups = np.cumsum(train) % 4
    def score(opt):
        return sum(np.sum(error(train & (groups != k), train & (groups == k), *opt)**2) for k in range(4))
    best = min(options, key=score)
    selected.append(best)
    E0[test] = error(train, test)
    E1[test] = error(train, test, *best)
r0, r1 = (np.sqrt(np.mean(e**2, axis=0)) for e in (E0, E1))
print('Outer-CV RMSE per bank, affine:',r0,'nested challenger:',r1)
print('Worst outer-CV bank RMSE, affine/challenger:',max(r0),max(r1),'selected:',selected)
