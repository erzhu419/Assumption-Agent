import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
cal = s['sensor_calibration_sweeps']
x, raw = cal['ref_rosette'], cal['raw_rosette']
D = m._rosette_design(x)
pairs = [(i,j) for i in range(4) for j in range(i+1,4)]
Q = np.column_stack([x*x] + [(x[:,i]-x[:,j])**2 for i,j in pairs] + [(x[:,i]+x[:,j])**2 for i,j in pairs])
T = np.column_stack([D, x*x] + [x[:,i]**2*x[:,j] for i in range(4) for j in range(4) if i!=j] + [x[:,i]*x[:,j]*x[:,k] for i in range(4) for j in range(i+1,4) for k in range(j+1,4)])
errors = {k: [] for k in ('baseline','expanded_poly','rational')}
for held in np.array_split(np.random.default_rng(24).permutation(len(x)), 5):
    tr = np.setdiff1d(np.arange(len(x)),held)
    c = m._fit_rosette(x[tr],raw[tr])
    e = raw[tr]-D[tr]@c
    med = np.median(e,axis=0)
    scale = np.maximum(1.4826*np.median(abs(e-med),axis=0),1e-7)
    fit = tr[np.all(abs(e-med)<8*scale,axis=1)]
    pred = D[held]@c
    poly = T[held]@np.linalg.lstsq(T[fit],raw[fit],rcond=None)[0]
    rat = np.empty_like(pred)
    for j in range(4):
        # Fit P/(1+Q a) by linearization; choose denominator on training sweeps only.
        best = (np.inf, pred[:,j])
        for q in Q.T:
            z = np.column_stack((D[fit],-raw[fit,j]*q[fit]))
            b = np.linalg.lstsq(z,raw[fit,j],rcond=None)[0]
            den = 1+q*b[-1]
            if np.min(den[fit])<0.5 or np.min(den[held])<0.5:
                continue
            loss = np.mean((D[fit]@b[:-1]/den[fit]-raw[fit,j])**2)
            if loss < best[0]:
                best = loss, D[held]@b[:-1]/den[held]
        rat[:,j] = best[1]
    # Calibration gross outliers stay in the all-row score; robust median also measures nominal fit.
    for name, pred_ in [('baseline',pred),('expanded_poly',poly),('rational',rat)]:
        e = raw[held]-pred_
        errors[name].append((np.median(abs(e),axis=0),np.sqrt(np.mean(e**2,axis=0))))
for name, values in errors.items():
    print(name, 'worst fold/channel median abs:', np.max(np.array(values)[:,0]), 'worst fold/channel RMS:', np.max(np.array(values)[:,1]))

out = m.solve_batch(s)
S,U = m._forward(s)
y = m._calibrate(s)
Z = s['Z_nuisance']
P = np.linalg.solve(Z.T@Z + s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
A, b = S-Z@(P@S), y-Z@(P@y)
w = np.asarray(out['weights']); c = out['kkt_certificate']; nu = c['nu_sum']
lo,hi = np.asarray(c['lam_lower']),np.asarray(c['lam_upper'])
cap,budget,ridge = (s[k] for k in ('u_box','total_budget','source_ridge'))
e = A@w-b; g = A.T@e+ridge*w
J = (e@e+ridge*(w@w))/2
lb = J-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
metrics = {'equality':abs(w.sum()-budget),'box':max(0,-w.min(),w.max()-cap),'stationarity':max(abs(g-nu-lo+hi)),'complementarity':max(max(abs(lo*w)),max(abs(hi*(cap-w)))),'duality_gap':J-lb,'certificate_mismatch':max(abs(c['primal_objective']-J),abs(c['dual_lower_bound']-lb),max(abs(np.asarray(c['weights'])-w))),'field_contraction':np.max(abs(out['terminal_field']-np.tensordot(w,U,axes=1)))}
print('KKT and field audit:',metrics)
assert min(lo.min(),hi.min())>=0 and max(abs(v) for v in metrics.values())<1e-8
