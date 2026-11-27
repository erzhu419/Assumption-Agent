import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
archive = Path('/tmp/hegel_continual_archive')
def load(p):
    spec = importlib.util.spec_from_file_location('solver', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m = load(root / 'batch_solver.py')
s = pickle.load((root / 'public_batch.pkl').open('rb'))[0]['study']
a = s['sensor_calibration_sweeps']
x, raw = a['ref_rosette'], a['raw_rosette']
coef = m._fit_rosette(x, raw)
def pred(x, c):
    return np.einsum('njk,jk->nj', m._rosette_basis(x), c)
errors = []
for test in np.array_split(np.arange(len(x)), 4) + [np.flatnonzero(np.count_nonzero(x, axis=1) == 1)]:
    tr = ~np.isin(np.arange(len(x)), test)
    r = abs(raw[test] - pred(x[test], m._fit_rosette(x[tr], raw[tr])))
    gross = np.max(r, axis=1) > 1
    assert not np.any((np.max(r, axis=1) > 1e-4) & ~gross)
    errors.append(np.max(r[~gross]))
b = a['single_bank_sweeps']; j, u = b['bank_id'], b['ref_signal']
r = abs(b['raw_readout'] - (coef[j, 0] + coef[j, 1] * u + coef[j, 5] * u**3))
assert np.count_nonzero(r > 1e-4) == 1 and r.max() > 10
obs = s['y_obs_raw'].reshape(-1, 4)
y = m._calibrate(s)
roundtrip = np.max(abs(pred(y.reshape(-1, 4), coef) - obs))
assert max(errors) < 1e-4 and roundtrip < 1e-11
print('blocked/isolated calibration worst clean abs error', max(errors), 'independent bank clean max', max(r[r<1]), 'inverse abs error', roundtrip)

Z = s['Z_nuisance']; ridge, sr = s['nuisance_ridge'], s['source_ridge']; cap, budget = s['u_box'], s['total_budget']
ns = len(s['sensor_ij']); nk = len(s['mu_shifts']); rows = np.arange(len(y)).reshape(nk, ns)
outputs = {}
for name, mm in [('Robin', m), ('sealed', load(archive/'beta_m98_v2_c04_batch_01_solver.py')), ('batch03', load(archive/'beta_m98_v2_c04_batch_03_solver.py'))]:
    S, _ = mm._forward(s)
    for split, folds in [('rosette', [rows[:,i:i+4].ravel() for i in range(0, ns, 4)]), ('sensor', list(rows.T)), ('shift', list(rows))]:
        rms = []
        for test in folds:
            tr = ~np.isin(np.arange(len(y)), test)
            z, G, t = Z[tr], S[tr], y[tr]
            p = np.linalg.solve(z.T@z + ridge*np.eye(z.shape[1]), z.T)
            D, q = G-z@(p@G), t-z@(p@t)
            w, _ = m._box_budget(D.T@D + sr*np.eye(S.shape[1]), -D.T@q, cap, budget)
            g = p@(t-G@w)
            rms.append(np.sqrt(np.mean((S[test]@w+Z[test]@g-y[test])**2)))
        outputs[name, split] = max(rms)
        print(name, split, 'mean / worst held-out RMSE', np.mean(rms), max(rms))
assert all(outputs['Robin',k] < min(outputs['sealed',k],outputs['batch03',k]) for k in ('rosette','sensor','shift'))

# Rebuild the stencil directly and test the delivered field against its independent residual.
def axis(k, v, L, rob):
    n = len(k); h = L/(n-1); A = np.zeros((n,n))
    for i in range(1,n-1):
        a = (k[i]+k[i-1])/(2*h*h); b = (k[i]+k[i+1])/(2*h*h)
        A[i,i-1] = -a-max(v[i],0)/h
        A[i,i+1] = -b-max(-v[i],0)/h
        A[i,i] = a+b+abs(v[i])/h
    for i,j in [(0,1),(n-1,n-2)]:
        A[i,i], A[i,j] = k[i]/h**2 + rob/h, -k[i]/h**2
    return A
nx,ny=s['nx'],s['ny']; modes=s['source_modes']
A,B=[axis(s['kappa_'+d],s['vel_'+d],s['L'+d],s['robin_'+d]) for d in 'xy']
L=np.kron(np.eye(ny),A)+np.kron(B,np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
idx=np.array(s['sensor_ij']) @ [1,nx]
rhs=np.stack([u.ravel(order='F') for u in modes],axis=1)
S=np.vstack([np.linalg.solve(L+shift*np.eye(nx*ny),rhs)[idx] for shift in s['mu_shifts']])
out=m.solve_batch(s); w=out['weights']; k=out['kkt_certificate']; nu=k['nu_sum']
p=np.linalg.solve(Z.T@Z+ridge*np.eye(Z.shape[1]),Z.T)
D=S-Z@(p@S); q=y-Z@(p@y); e=D@w-q
J=0.5*(e@e+sr*(w@w)); g=D.T@e+sr*w
lower,upper=np.asarray(k['lam_lower']),np.asarray(k['lam_upper'])
gap=J-k['dual_lower_bound']; stat=np.max(abs(g-nu-lower+upper))
res=np.max(abs((L+s['mu_shifts'][-1]*np.eye(nx*ny))@out['terminal_field'].ravel(order='F')-rhs@w))
assert np.allclose(k['weights'],w,atol=1e-13) and abs(w.sum()-budget)<1e-12
assert min(w)>=-1e-12 and max(w)<=cap+1e-12 and min(lower)>=0 and min(upper)>=0
assert max(np.max(abs(lower*w)),np.max(abs(upper*(cap-w))),stat,abs(gap),abs(J-k['primal_objective']),res)<1e-10
print('independent objective',J,'dual gap',gap,'KKT stationarity',stat,'terminal PDE residual',res)
