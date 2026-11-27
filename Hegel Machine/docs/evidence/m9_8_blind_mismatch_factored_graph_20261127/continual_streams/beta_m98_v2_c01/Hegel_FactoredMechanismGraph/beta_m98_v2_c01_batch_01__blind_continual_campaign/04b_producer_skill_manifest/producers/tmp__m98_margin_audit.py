import pickle
import runpy
import numpy as np
from scipy.optimize import minimize

m = runpy.run_path('/holdout_task/batch_solver.py')
i = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s = i['study']
cal = s['sensor_calibration_sweeps']
x, y = cal['ref_rosette'], cal['raw_rosette']
base = np.c_[np.ones(len(x)), x]
pairs = np.column_stack([f for j in range(4) for k in range(j+1,4) for f in (x[:,j]*x[:,k], (x[:,j]-x[:,k])**2)])
for label, design in [('affine',base), ('pairwise',np.c_[base,pairs]), ('rational',np.c_[base,np.column_stack([x/(1+a*x*x) for a in (.1,1.)])])]:
    errs = []
    for fold in range(5):
        tr = np.arange(len(x)) % 5 != fold
        coef = np.linalg.lstsq(design[tr], y[tr], rcond=None)[0]
        errs.append(np.mean((design[~tr]@coef-y[~tr])**2,axis=0)**.5)
    print(label, 'worst held-out bank/fold RMSE',np.max(errs), 'pooled RMSE',np.mean(np.asarray(errs)**2)**.5)

for d in 'xy':
    k = s['kappa_'+d]; h = s['L'+d]/(s['n'+d]-1)
    A = m['_axis'](k,s['vel_'+d],h)
    assert np.linalg.norm(A@np.ones(len(k)),np.inf) < 1e-12
    assert np.allclose(A.diagonal()[[0,-1]], k[[0,-1]]/h**2,rtol=1e-14,atol=0)
print('Reflecting row balance and endpoint-kappa wall coefficients verified')

S,U = m['_forward'](s)
Z = s['Z_nuisance']; P = np.eye(len(Z))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
D = P@S; ridge = s['source_ridge']; n = D.shape[1]; cap = s['u_box']
cases = [('production', s['y_obs_raw'])] + [(f'tracer_{j}',p['y_pulse_raw']) for j,p in enumerate(i['campaign_diagnostics'] or [])]
max_gap=max_err=max_field=0.; patterns=set()
for label,obs in cases:
    for budget in (0., .01*cap, cap, s['total_budget'], (n-1)*cap, (n-.01)*cap, n*cap):
        t = dict(s,y_obs_raw=obs,total_budget=budget)
        b = P@m['_calibrate'](t)
        def J(w):
            r = D@w-b
            return .5*(r@r+ridge*(w@w))
        def grad(w): return D.T@(D@w-b)+ridge*w
        out = m['solve_batch'](t); w=out['weights']; c=out['kkt_certificate']; nu=c['nu_sum']; lo=np.array(c['lam_lower']); hi=np.array(c['lam_upper']); g=grad(w)
        assert abs(w.sum()-budget) < 1e-11 and w.min() >= -1e-12 and w.max() <= cap+1e-12
        assert np.max(np.abs(g-nu-lo+hi)) < 1e-11
        assert max(np.max(np.abs(lo*w)),np.max(np.abs(hi*(cap-w)))) < 1e-11
        dual=J(w)-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
        assert max(abs(c['primal_objective']-J(w)),abs(c['dual_lower_bound']-dual)) < 1e-11
        gap=abs(J(w)-dual); assert gap < 1e-11
        max_gap=max(max_gap,gap)
        patterns.add((int(sum(w<1e-10)),int(sum(w>cap-1e-10))))
        field=np.einsum('i,ijk->jk',w,U); max_field=max(max_field,np.max(np.abs(field-out['terminal_field'])))
        if 0<budget<n*cap:
            ref=minimize(J,np.full(n,budget/n),jac=grad,bounds=[(0,cap)]*n,constraints=[{'type':'eq','fun':lambda w:w.sum()-budget,'jac':lambda w:np.ones(n)}],method='SLSQP',options={'ftol':1e-14,'maxiter':200})
            err=np.max(np.abs(ref.x-w)); assert ref.success and err < 1e-6,(label,budget,ref.message,err)
            max_err=max(max_err,err)
assert max_field < 1e-12
print(f'{7*len(cases)} real-observation budget cases, {len(patterns)} active-bound patterns; worst weight error {max_err:.3g}, dual gap {max_gap:.3g}, field assembly error {max_field:.3g}')
