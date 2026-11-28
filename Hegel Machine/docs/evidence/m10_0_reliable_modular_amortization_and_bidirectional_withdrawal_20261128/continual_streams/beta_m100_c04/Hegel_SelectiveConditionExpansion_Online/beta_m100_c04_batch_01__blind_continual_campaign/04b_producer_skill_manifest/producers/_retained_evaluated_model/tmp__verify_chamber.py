import importlib.util
import itertools
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
nx, ny = s['nx'], s['ny']

def axis(d, reflecting):
    k, v = s['kappa_'+d], s['vel_'+d]
    h = s['L'+d] / (len(k)-1)
    A = np.zeros((len(k), len(k)))
    for i in range(len(k)):
        if i in (0,len(k)-1):
            A[i,i] = k[i]/h**2 + (0 if reflecting else s['robin_'+d]/h)
            A[i,1 if i == 0 else i-1] = -k[i]/h**2
        else:
            kp = (k[i]+k[i+1])/2
            km = (k[i]+k[i-1])/2
            A[i,i-1] = -km/h**2-max(v[i],0)/h
            A[i,i+1] = -kp/h**2-max(-v[i],0)/h
            A[i,i] = (kp+km)/h**2+abs(v[i])/h
    return A

def forward(reflecting):
    L = np.kron(np.eye(ny), axis('x', reflecting)) + np.kron(axis('y', reflecting), np.eye(nx))
    L += np.diag(s['V_reaction_2d'].flatten('F'))
    rhs = np.array([mode.flatten('F') for mode in s['source_modes']]).T
    U = [np.linalg.solve(L+mu*np.eye(nx*ny), rhs) for mu in s['mu_shifts']]
    ij = s['sensor_ij']
    return np.vstack([u[ij[:,0]+nx*ij[:,1]] for u in U]), U[-1]

S, U = forward(True)
S_open, _ = forward(False)
Sp, Up = m._forward(s)
np.testing.assert_allclose(Sp, S, atol=2e-12, rtol=0)
np.testing.assert_allclose(np.array([u.flatten('F') for u in Up]).T, U, atol=2e-12, rtol=0)
P = item['campaign_diagnostics']
W = np.array([p['w_pulse'] for p in P]).T
cal = s['sensor_calibration_sweeps']
ref, raw = cal['ref_rosette'], cal['raw_rosette']
X = np.c_[np.ones(len(ref)),ref]
# Contiguous sweep blocks, stratified so every fold trains on the mixed rows.
errs, sweep = [], []
for group in np.array_split(np.arange(len(ref)), 5):
    keep = np.ones(len(ref), bool)
    keep[group] = False
    coeff = np.linalg.lstsq(X[keep], raw[keep], rcond=None)[0]
    sweep.append(np.sqrt(np.mean((X[group] @ coeff-raw[group])**2)))
    Y = np.column_stack([np.linalg.solve(coeff[1:].T,(p['y_pulse_raw'].reshape(-1,4)-coeff[0]).T).T.ravel() for p in P])
    rms = np.sqrt(np.mean((np.array([S_open @ W,S @ W])-Y)**2, axis=1))
    for j in range(len(P)):
        k = np.argmin(np.mean(rms[:, np.arange(len(P)) != j]**2, axis=1))
        assert k == 1
        errs.append(rms[k,j])
print('worst held-out sweep RMS', max(sweep), 'worst held-out pulse RMS', max(errs))
assert max(errs) < 5e-6

coef = np.linalg.lstsq(X, raw, rcond=None)[0]
y = np.linalg.solve(coef[1:].T, (s['y_obs_raw'].reshape(-1,4)-coef[0]).T).T.ravel()
Z = s['Z_nuisance']
M = np.eye(len(Z))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
A, b = M@S, M@y
r, cap, budget = (s[k] for k in ('source_ridge','u_box','total_budget'))
H, f = A.T@A+r*np.eye(A.shape[1]), -A.T@b

def exact_faces(H,f,cap,budget):
    best, wbest = np.inf, None
    for state in itertools.product((-1,0,1), repeat=len(f)):
        st = np.array(state)
        F, B = np.flatnonzero(st==0), np.flatnonzero(st!=0)
        w = cap*(st==1).astype(float)
        remaining = budget-w.sum()
        if remaining < -1e-12 or remaining > len(F)*cap+1e-12:
            continue
        if len(F):
            one = np.ones((len(F),1))
            mat = np.block([[H[np.ix_(F,F)], -one], [one.T, np.zeros((1,1))]])
            w[F] = np.linalg.solve(mat,np.r_[-f[F]-H[np.ix_(F,B)]@w[B],remaining])[:-1]
        if w.min() < -1e-12 or w.max() > cap+1e-12:
            continue
        q = .5*w@H@w+f@w
        if q < best:
            best, wbest = q, w
    return wbest

out = m.solve_batch(s)
w = out['weights']
c = out['kkt_certificate']
np.testing.assert_allclose(w,exact_faces(H,f,cap,budget),rtol=0,atol=2e-10)
np.testing.assert_allclose(out['terminal_field'],(U@w).reshape((nx,ny),order='F'),rtol=0,atol=2e-12)
np.testing.assert_array_equal(c['weights'],w)
g = A.T@(A@w-b)+r*w
lo, hi = np.asarray(c['lam_lower']),np.asarray(c['lam_upper'])
nu = c['nu_sum']
J = .5*(np.linalg.norm(A@w-b)**2+r*(w@w))
D = J-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
np.testing.assert_allclose([c['primal_objective'],c['dual_lower_bound']],[J,D],atol=2e-12,rtol=0)
np.testing.assert_allclose(g-nu-lo+hi,0,atol=2e-12,rtol=0)
assert lo.min() >= 0 and hi.min() >= 0 and abs(J-D) < 2e-12

# Stress the certifier on empty/full feasible sets, exact box vertices,
# and nearby budgets where a newly free coordinate carries very little mass.
worst_gap = 0.
for B in [0.,1e-10,cap-1e-10,cap,cap+1e-10,3*cap,7*cap,8*cap-1e-10,8*cap]:
    for ff in (f,np.linspace(-1.,1.,len(f))):
        ww, nn = m._box_simplex(H,ff,cap,B)
        gg = H@ww+ff
        gap = gg@ww-nn*B-cap*np.minimum(0,gg-nn).sum()
        worst_gap = max(worst_gap,abs(gap))
        assert ww.min() >= 0 and ww.max() <= cap and abs(ww.sum()-B) < 2e-12 and abs(gap) < 2e-12
print('public KKT gap',J-D,'worst stress KKT gap',worst_gap)
