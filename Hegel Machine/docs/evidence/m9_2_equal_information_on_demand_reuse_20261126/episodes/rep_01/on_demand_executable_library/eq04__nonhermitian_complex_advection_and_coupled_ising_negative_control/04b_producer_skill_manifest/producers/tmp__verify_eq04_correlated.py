import importlib.util
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve

path = '/holdout_task/nonhermitian_advection_coupled_select.py'
spec = importlib.util.spec_from_file_location('eq04', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
n = 32
zero = np.zeros((n,n))
basis = np.zeros((11,n,n))
basis[0,1,1] = 1
basis[1,1,1] = 1
basis[1,30,30] = .3
nu, lam2 = 1e-4, 1e-5
D = diags([-np.ones(31),2*np.ones(32),-np.ones(31)],[-1,0,1])
A = (nu*33**2*(kron(D,eye(n))+kron(eye(n),D)) + (2+1j)*eye(n*n)).tocsc()
F = spsolve(A,basis.reshape(11,-1).T.astype(complex))
sensors = np.array([[1,1],[30,30]])
G = F[sensors[:,0]*n+sensors[:,1],:2]
# Manufacture the optimum by prescribing its residual to satisfy the disk
# optimality conditions. Opposing coefficients make the field more sensitive
# than the scalar objective to an inaccurate disk solve.
worst_obj = worst_field = 0.
for separation in (.03, .01, .003):
    b = basis.copy() * 1e8
    b[1,30,30] = separation * 1e8
    f = spsolve(A, b.reshape(11,-1).T.astype(complex))
    h = f[sensors[:,0]*n+sensors[:,1],:2]
    wref = 1e-8*np.exp(.7j)*np.array([1., -.95+.05j])
    expected = abs(f[:,:2]@wref).reshape(n,n)
    normal = np.array([30.*np.exp(.7j),0.])
    y = h@wref + np.linalg.solve(h.conj().T, lam2*wref+normal)
    assert .5*np.linalg.norm(y)**2 < .01
    out = mod.solve_nonhermitian_coupled_inversion(dict(nu=nu,vx=zero,vy=zero,sigma=zero+2,omega=1.,basis=b,sensor_ij=sensors,lam0=.01,lam2=lam2,pair_penalties=[],y_obs_re=y.real,y_obs_im=y.imag))
    expected_obj = .5*np.linalg.norm(h@wref-y)**2 + .5*lam2*np.linalg.norm(wref)**2
    err_obj = abs(out['objective']-expected_obj)
    err_field = np.linalg.norm(out['predicted_field_abs']-expected)/np.linalg.norm(expected)
    worst_obj, worst_field = max(worst_obj,err_obj), max(worst_field,err_field)
    print('basis separation:', separation, 'absolute objective error:', err_obj, 'relative field error:', err_field, flush=True)
assert worst_obj <= 1e-6 and worst_field <= 1e-5, (worst_obj,worst_field)
