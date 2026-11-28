import importlib.util
import pickle
import numpy as np

p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', p+'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
i = pickle.load(open(p+'public_batch.pkl', 'rb'))[0]
s = i['study']; c = s['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']
S, _ = m._forward(s, ('periodic','reflect'))
w = np.stack([d['w_pulse'] for d in i['campaign_diagnostics']])
yp = np.stack([d['y_pulse_raw'] for d in i['campaign_diagnostics']])
xp = (w@S.T).reshape(-1,4)

# Compare plausible algebraic extensions only when they improve held-out sweeps and tracer predictions.
def basis(t, j, model):
    A = m._design(t,j)
    a,b,d,o = (t[:,(j+k)%4] for k in (0,1,-1,2))
    if model == 'contrasts':
        return np.column_stack((A, a-b, a-d, o, (a-b)**2, (a-d)**2, (a-b)**3, (a-d)**3, a*b*d))
    if model == 'rational':
        q = np.column_stack(tuple(v**2 for v in (a,b,a-b,a+b)))
        return np.column_stack((A,q[:,:,None].__mul__(t[:,j,None,None]).reshape(len(t),-1)/(1+q)))
    return A

rng = np.random.default_rng(8)
held = np.array_split(rng.permutation(len(x)), 5)
for model in ('polynomial','contrasts','rational'):
    e = np.zeros_like(y); pulse=[]
    for test in held:
        tr = np.setdiff1d(np.arange(len(x)),test)
        anchor = m._calibration(dict(c, ref_rosette=x[tr], raw_rosette=y[tr]))
        pred = np.empty_like(xp)
        for j in range(4):
            good = tr[abs(m._design(x[tr],j)@anchor[j]-y[tr,j])<2e-4]
            A = basis(x[good],j,model)
            coef = np.linalg.lstsq(A,y[good,j],rcond=None)[0]
            e[test,j] = basis(x[test],j,model)@coef-y[test,j]
            pred[:,j] = basis(xp,j,model)@coef
        pulse.append(np.sqrt(np.mean((pred.reshape(yp.shape)-yp).reshape(len(w),len(s['mu_shifts']),-1)**2,axis=2)).max())
    gross = np.max(abs(e),axis=1)>0.1
    print(model,'gross rows',np.flatnonzero(gross).tolist(),'worst held-out inlier bank RMSE',np.sqrt(np.mean(e[~gross]**2,axis=0)).max(),'worst pulse-shift RMSE over fits',max(pulse))

# Quantify fitted nuisance impact within the actual study, rather than treating archived campaigns as labeled cohorts.
ycal = m._calibrate(s); sol=m.solve_batch(s); ww=np.asarray(sol['weights']); Z=s['Z_nuisance']
r=ycal-S@ww
n=Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@r)
print('per-shift unprojected residual RMS',np.sqrt(np.mean(r.reshape(len(s['mu_shifts']),-1)**2,axis=1)))
print('per-shift projected residual RMS',np.sqrt(np.mean((r-n).reshape(len(s['mu_shifts']),-1)**2,axis=1)))
print('certificate gap',sol['kkt_certificate']['primal_objective']-sol['kkt_certificate']['dual_lower_bound'])
