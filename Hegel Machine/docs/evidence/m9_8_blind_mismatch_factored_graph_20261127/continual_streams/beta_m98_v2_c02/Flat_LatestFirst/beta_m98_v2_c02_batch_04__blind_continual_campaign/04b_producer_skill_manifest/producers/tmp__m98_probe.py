import glob, importlib.util, pickle
import numpy as np
from scipy.optimize import root

item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
c = s['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']

def basis(x, j):
    p, q = (j-1) % 4, (j+1) % 4
    a = x[..., j]
    return np.stack((np.ones_like(a), a, x[..., p], x[..., q], a*x[..., q], a**3), axis=-1)

clean = np.max(abs(y), axis=1) < 9
ids = np.flatnonzero(clean)
for j in range(4):
    errs = []
    for fold in range(5):
        a, b = ids[np.arange(len(ids))%5 != fold], ids[np.arange(len(ids))%5 == fold]
        v = np.linalg.lstsq(basis(x[a], j), y[a,j], rcond=None)[0]
        errs.append(np.mean((basis(x[b], j)@v-y[b,j])**2)**.5)
    print('bank', j, 'worst sparse cubic CV', max(errs))
coef = np.stack([np.linalg.lstsq(basis(x[clean], j), y[clean,j], rcond=None)[0] for j in range(4)])
print('sparse coefficients', coef)
single = c['single_bank_sweeps']; bank = single['bank_id']; a = single['ref_signal']; predicted = coef[bank,0] + coef[bank,1]*a + coef[bank,-1]*a**3
print('single-bank abs residuals', np.round(abs(predicted-single['raw_readout']).reshape(4,-1),5))

def forward(z):
    return np.array([basis(z,j)@coef[j] for j in range(4)])

raw = s['y_obs_raw'].reshape(-1,4)
sol = [root(lambda z:forward(z)-row, np.zeros(4), tol=1e-11) for row in raw]
yc = np.stack([r.x for r in sol])
print('inverse roundtrip max', max(abs(forward(z)-row).max() for z,row in zip(yc,raw)))
Z = s['Z_nuisance']; P = np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
def proj(a): return a-Z@(P@a)
r = proj(yc.ravel()); cap, total, ridge = (s[k] for k in ('u_box','total_budget','source_ridge'))
print('mesh/shifts/sensors',s['nx'],s['ny'],s['mu_shifts'],len(s['sensor_ij']))
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
    spec = importlib.util.spec_from_file_location('past', f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    S, U = m._forward(s); A = proj(S)
    w, nu = m._box_budget_qp(A.T@A+ridge*np.eye(A.shape[1]), A.T@r, cap, total)
    err = A@w-r
    design = np.column_stack((S,Z)); group = np.repeat(np.arange(len(raw)),4)
    rms = []
    for held in range(len(raw)):
        use = group != held; Ss, Zs, ys = S[use], Z[use], yc.ravel()[use]
        Psub = np.linalg.solve(Zs.T@Zs+s['nuisance_ridge']*np.eye(Zs.shape[1]), Zs.T)
        As, rs = Ss-Zs@(Psub@Ss), ys-Zs@(Psub@ys)
        ws, _ = m._box_budget_qp(As.T@As+ridge*np.eye(As.shape[1]), As.T@rs, cap, total)
        gamma = Psub@(ys-Ss@ws)
        rms.append(np.mean((S[~use]@ws+Z[~use]@gamma-yc.ravel()[~use])**2)**.5)
    print(f.split('/')[-1], 'J', .5*(err@err+ridge*(w@w)), 'RMSE per shift', np.mean(err.reshape(len(s['mu_shifts']),-1)**2,axis=1)**.5, 'held-rosette rms/median/max', np.sqrt(np.mean(np.square(rms))), np.median(rms), max(rms), 'weights', w.round(5), 'terminal range', np.min(w@U.reshape(len(w),-1)),np.max(w@U.reshape(len(w),-1)))
