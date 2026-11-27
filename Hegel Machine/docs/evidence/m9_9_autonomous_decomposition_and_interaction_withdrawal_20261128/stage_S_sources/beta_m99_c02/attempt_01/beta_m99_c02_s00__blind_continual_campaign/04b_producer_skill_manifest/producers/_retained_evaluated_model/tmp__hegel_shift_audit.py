import importlib.util
import pickle
import numpy as np

p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', p+'batch_solver.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
s = pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
y, z = b._calibrate(s), s['Z_nuisance']
cap, budget = s['u_box'], s['total_budget']
ns = len(s['sensor_ij'])
groups = np.arange(len(y)) // ns
banks = np.asarray(s['row_bank_id']).ravel()
assert len(banks) == len(y)

def candidate(rx, ry):
    axes = []
    for a, scale in zip('xy', (rx, ry)):
        k, v, h, r = s['kappa_'+a], s['vel_'+a], s['L'+a]/(s['n'+a]-1), s['robin_'+a]
        A = b._axis(k, v, h, r*scale)
        expected = np.zeros(len(k)); expected[[0,-1]] = r*scale/h
        np.testing.assert_allclose(np.asarray(A.sum(axis=1)).ravel(), expected, rtol=0, atol=1e-12)
        axes.append(A)
    return b._responses(s, axes)[0]

# A per-bank intercept is an audit challenger, not an assumed part of the published nuisance model.
bank_dummies = (banks[:,None] == np.unique(banks)[None,1:]).astype(float)
for name, S, Z in [('Robin', candidate(1,1), z), ('Robin plus bank intercepts', candidate(1,1), np.c_[z, bank_dummies]), ('Reflect x', candidate(0,1), z), ('Reflect y', candidate(1,0), z), ('Reflect both', candidate(0,0), z)]:
    err = np.empty_like(y)
    for j in np.unique(groups):
        tr = groups != j
        cof = np.linalg.lstsq(Z[tr], np.c_[S[tr],y[tr]], rcond=None)[0]
        X = S[tr]-Z[tr]@cof[:,:-1]; Y = y[tr]-Z[tr]@cof[:,-1]
        w, _ = b._box_simplex(X.T@X+1e-9*np.eye(X.shape[1]), -X.T@Y, cap, budget)
        err[~tr] = S[~tr]@w+Z[~tr]@(cof[:,-1]-cof[:,:-1]@w)-y[~tr]
    rms = lambda a: np.sqrt(np.mean(a*a))
    print(name, 'shift RMS', [round(rms(err[groups==j]), 7) for j in np.unique(groups)], 'worst bank-shift RMS', round(max(rms(err[(groups==j)&(banks==k)]) for j in np.unique(groups) for k in np.unique(banks)), 7))
