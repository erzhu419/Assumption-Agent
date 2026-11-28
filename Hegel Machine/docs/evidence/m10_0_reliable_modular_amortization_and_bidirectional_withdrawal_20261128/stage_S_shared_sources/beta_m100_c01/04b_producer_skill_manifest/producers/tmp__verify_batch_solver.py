import importlib.util
import itertools
import pickle
import numpy as np

p = '/holdout_task/batch_solver.py'
spec = importlib.util.spec_from_file_location('candidate', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
study = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']

def axis(k, v, h, r):
    n = len(k)
    D = np.zeros((n, n))
    for i, j in ((0, 1), (n-1, n-2)):
        D[i, j] = -k[i] / h**2
        D[i, i] = k[i] / h**2 + r / h
    for i in range(1, n-1):
        D[i, i-1] = -(k[i]+k[i-1])/(2*h*h) - max(v[i], 0)/h
        D[i, i+1] = -(k[i]+k[i+1])/(2*h*h) - max(-v[i], 0)/h
        D[i, i] = -D[i, i-1] - D[i, i+1]
    return D

nx, ny = study['nx'], study['ny']
ax, ay = [axis(study['kappa_'+d], study['vel_'+d], study['L'+d]/(n-1), study['robin_'+d]) for d,n in zip('xy',(nx,ny))]
L = np.kron(np.eye(ny),ax) + np.kron(ay,np.eye(nx)) + np.diag(study['V_reaction_2d'].flatten('F'))
B = np.stack([u.flatten('F') for u in study['source_modes']],axis=1)
fields = [np.linalg.solve(L+shift*np.eye(nx*ny), B) for shift in study['mu_shifts']]
ij = study['sensor_ij']
S = np.concatenate([F[ij[:,0]+nx*ij[:,1]] for F in fields])
print('dense transport max error', np.max(abs(S-m._transport(study)[0])))
np.testing.assert_allclose(S, m._transport(study)[0], atol=1e-12, rtol=0)

# Independent optimality oracle: enumerate feasible faces of the public and
# small stress problems, solving their equality-constrained quadratic systems.
def oracle(H,c,cap,b):
    n = len(c)
    best = np.inf
    for t in itertools.product((-1,0,1), repeat=n):
        t = np.asarray(t)
        w = np.where(t>0,cap,0.)
        i = np.flatnonzero(t==0)
        rem = b-w.sum()
        if rem < -1e-10 or rem > len(i)*cap+1e-10:
            continue
        if len(i):
            M = np.ones((len(i)+1,len(i)+1))
            M[:-1,:-1] = H[np.ix_(i,i)]
            M[-1,-1] = 0.
            w[i] = np.linalg.solve(M,np.r_[-c[i]-H[i]@w,rem])[:-1]
        if min(w) >= -1e-10 and max(w) <= cap+1e-10:
            best = min(best, 0.5*w@H@w+c@w)
    return best

cap,b = study['u_box'],study['total_budget']
Z = study['Z_nuisance']; R = np.eye(len(S))-Z@np.linalg.solve(Z.T@Z+study['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
A = R@S; ridge = study['source_ridge']
H = A.T@A + ridge*np.eye(A.shape[1])

def verify(study, y):
    o = m.solve_batch(study)
    w = o['weights']; k = o['kkt_certificate']; nu = k['nu_sum']
    r = R@y; e = A@w-r; g = A.T@e+ridge*w
    c = -A.T@r; primal = .5*(e@e+ridge*(w@w))
    gap = primal-k['dual_lower_bound']
    stationarity = np.max(abs(g-nu-k['lam_lower']+k['lam_upper']))
    complementarity = max(np.max(abs(w*k['lam_lower'])),np.max(abs((cap-w)*k['lam_upper'])))
    field_error = np.max(abs(o['terminal_field']-(fields[-1]@w).reshape(nx,ny,order='F')))
    obj_error = abs(.5*w@H@w+c@w-oracle(H,c,cap,b))
    assert abs(w.sum()-b)<1e-11 and w.min()>=0 and w.max()<=cap
    assert np.array_equal(w,k['weights']) and min(k['lam_lower'])>=0 and min(k['lam_upper'])>=0
    assert abs(primal-k['primal_objective'])<1e-11 and abs(gap)<1e-11
    assert max(stationarity,complementarity,field_error,obj_error)<1e-11
    print('certificate gap, stationarity, complementarity, terminal field and oracle errors:',gap,stationarity,complementarity,field_error,obj_error)

verify(study, m._calibrate(study))
rng = np.random.default_rng(62)
ref = rng.normal(size=(60,4)); M = np.eye(4)+rng.normal(scale=.15,size=(4,4)); d = rng.normal(scale=.2,size=4)
y = S @ np.array([.42,0,.42,0,.42,0,.14,0]) + Z @ np.array([.1,-.2,.3])
alt = dict(study, sensor_calibration_sweeps={'ref_rosette':ref,'raw_rosette':ref@M+d}, y_obs_raw=(y.reshape(-1,4)@M+d).ravel())
np.testing.assert_allclose(m._calibrate(alt),y,atol=1e-12,rtol=0)
verify(alt,y)
worst = 0.
for j in range(120):
    n = 5; F = rng.normal(size=(n,n)); Hs = F.T@F+.01*np.eye(n); c = rng.normal(size=n)*10
    budget = (0.,.1,1.,2.,4.9,5.)[j % 6]
    w, nu = m._box_simplex(Hs,c,1.,budget)
    g = Hs@w+c
    err = max(abs(w.sum()-budget), -w.min(), w.max()-1, abs(g@w-nu*budget-np.minimum(g-nu,0).sum()), abs(.5*w@Hs@w+c@w-oracle(Hs,c,1.,budget)))
    assert err < 1e-10, (j,err)
    worst = max(worst,err)
print('120 stress QPs worst feasibility / dual gap / oracle error',worst)
