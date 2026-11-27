import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    batch = pickle.load(f)

def extra(x, j):
    a,b,c,d = (x[:,(j+k)%4] for k in range(4))
    return np.column_stack((a*b,a*d,b*d,a*a*b,a*a*d,a*b*d, a/(1+.1*(b-d)**2)-a, a/(1+(b-d)**2)-a))

for e in batch:
    s = e['study']; sw = s['sensor_calibration_sweeps']; x,y = sw['ref_rosette'],sw['raw_rosette']
    F = m._rosette_terms(x); err = np.empty_like(y); ext = np.empty_like(y)
    for test in np.array_split(np.random.default_rng(98).permutation(len(x)), 5):
        tr = np.ones(len(x), dtype=bool); tr[test] = False
        fit = m._fit_rosette(x[tr],y[tr]); residual = y-F@fit
        err[test] = residual[test]
        r = np.max(abs(residual[tr]),axis=1); med = np.median(r)
        clean = tr & (np.max(abs(residual),axis=1) <= med + 10*max(med,1e-12))
        for j in range(4):
            cols = [0,1+j,1+(j-1)%4,1+(j+1)%4,5+j,9+j]
            G = np.column_stack((F[:,cols],extra(x,j)))
            coef = np.linalg.lstsq(G[clean],y[clean,j],rcond=None)[0]
            ext[test,j] = y[test,j]-G[test]@coef
    # Report contamination separately; compare the same held-out rows for both models.
    inlier = np.max(abs(err),axis=1) < .01
    print('sweep rows / gross outliers:', len(x), np.count_nonzero(~inlier))
    for name,E in [('structured',err),('pair/triple/rational extension',ext)]:
        print(name,'per-channel held-out inlier RMSE:',np.sqrt(np.mean(E[inlier]**2,axis=0)), 'worst inlier absolute:',np.max(abs(E[inlier])))
    S,U = m._design(s); fit = m._fit_rosette(x,y)
    for i,p in enumerate(e['campaign_diagnostics'] or []):
        q = (S@p['w_pulse']).reshape(-1,4)
        pred = (m._rosette_terms(q)@fit).ravel()
        print('raw forward pulse',i,'bank RMSE:',[float(np.sqrt(np.mean((pred-p['y_pulse_raw'])[s['row_bank_id']==k]**2))) for k in np.unique(s['row_bank_id'])])
    r=m.solve_batch(s); w=np.asarray(r['weights']); c=r['kkt_certificate']; Z=s['Z_nuisance']
    d=S@w-m._calibrate(s); Pz=np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T); d2=d-Z@(Pz@d)
    print('observation residual RMSE before/after projection, by bank:',[(float(np.sqrt(np.mean(d[ix]**2))),float(np.sqrt(np.mean(d2[ix]**2)))) for k in np.unique(s['row_bank_id']) for ix in [s['row_bank_id']==k]])
    print('isolated public dual gap:', c['primal_objective']-c['dual_lower_bound'])
