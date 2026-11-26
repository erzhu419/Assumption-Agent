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
scale = 1e8
F = F * scale
G = G * scale
for name, m, wref, normal in [
    ('scalar boundary', 1, 1e-8*np.array([np.exp(.7j)]), np.array([3e6*np.exp(.7j)])),
    ('correlated boundary/interior', 2, 1e-8*np.array([np.exp(.7j), .48*np.exp(-1.2j)]), np.array([3e6*np.exp(.7j), 0.])),
]:
    h = G[:m,:m]
    y = h@wref + np.linalg.solve(h.conj().T, lam2*wref + normal)
    b = np.zeros_like(basis)
    b[:m] = basis[:m]*scale
    lam0 = 1.
    assert .5*np.linalg.norm(y)**2 < lam0
    out = mod.solve_nonhermitian_coupled_inversion(dict(nu=nu,vx=zero,vy=zero,sigma=zero+2,omega=1.,basis=b,sensor_ij=sensors[:m],lam0=lam0,lam2=lam2,pair_penalties=[],y_obs_re=y.real,y_obs_im=y.imag))
    w = out['weights_real']+1j*out['weights_imag']
    assert np.max(np.abs(w)) <= 1e-8
    j = .5*np.linalg.norm(h@wref-y)**2 + .5*lam2*np.linalg.norm(wref)**2
    f = np.abs(F[:,:m]@wref).reshape(n,n)
    obj = abs(out['objective']-j)
    rel = np.linalg.norm(out['predicted_field_abs']-f)/np.linalg.norm(f)
    print(name, 'objective error', obj, 'relative field error', rel)
    assert obj <= 1e-6 and rel <= 1e-5
