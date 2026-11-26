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
wref = 1e-9*np.array([1+1j,2-3j])
y = G@wref + np.linalg.solve(G.conj().T, lam2*wref)
out = mod.solve_nonhermitian_coupled_inversion(dict(nu=nu,vx=zero,vy=zero,sigma=zero+2,omega=1.,basis=basis,sensor_ij=sensors,lam0=.01,lam2=lam2,pair_penalties=[],y_obs_re=y.real,y_obs_im=y.imag))
expected = abs(F[:,:2]@wref).reshape(n,n)
relative = np.linalg.norm(out['predicted_field_abs']-expected)/np.linalg.norm(expected)
print('Weak correlated interior optimum: relative field error', relative, flush=True)
assert relative <= 1e-5, relative
