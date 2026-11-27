import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
c = s['sensor_calibration_sweeps']; x, y = c['ref_rosette'], c['raw_rosette']
b = m._rosette_basis(x)
# Independent extra terms test omitted pairwise contrasts and the leading
# perturbation from dividing a response by 1 + a * contrast**2.
contrast = np.stack([x[:, j] - x[:, k] for j in range(4) for k in range(j+1,4)], axis=1)
pair = np.stack([x[:, j] * x[:, k] for j in range(4) for k in range(j+1,4)], axis=1)
stratum = np.count_nonzero(x, axis=1)
fold = np.zeros(len(x), dtype=int)
for a in np.unique(stratum):
    ii = np.flatnonzero(stratum == a); fold[ii] = np.arange(len(ii)) % 5
errors = {name:np.empty_like(y) for name in ('saved','shared','pairwise','rational')}
for f in range(5):
    t = fold != f; v = ~t
    w = m._fit_rosette(x[t], y[t]); p = np.einsum('njk,jk->nj',b,w)
    errors['saved'][v] = p[v]-y[v]
    r = p-y
    keep = t[:,None] & (abs(r) < 1e-4)
    def coeff(q):
        return np.column_stack((q[:4], q[4:8], q[4:8,None]*q[8:]))
    q0 = np.r_[w[:,0],w[:,1],np.mean(w[:,2:]/w[:,1,None],axis=0)]
    q = least_squares(lambda q:(np.einsum('njk,jk->nj',b,coeff(q))-y)[keep],q0).x
    errors['shared'][v] = (np.einsum('njk,jk->nj',b,coeff(q))-y)[v]
    for j in range(4):
        for name, extra in [('pairwise',pair), ('rational', -p[:,j,None]*contrast**2)]:
            design = np.column_stack([b[:,j],extra]); k = keep[:,j]
            coef = np.linalg.lstsq(design[k],y[k,j],rcond=None)[0]
            errors[name][v,j] = design[v]@coef-y[v,j]
for a in np.unique(stratum):
    group = stratum == a
    print('active channels',a,'n',group.sum())
    for name,e in errors.items():
        # Common cap keeps bad sweep readings visible without letting them dominate.
        rmse = np.sqrt(np.mean(np.minimum(abs(e[group]),1e-4)**2,axis=0))
        print(name,'worst bank capped OOF RMSE',float(rmse.max()),'median abs',float(np.median(abs(e[group]))))
