import importlib.util
import numpy as np

path = '/holdout_task/nonhermitian_advection_coupled_select.py'
spec = importlib.util.spec_from_file_location('candidate', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
n = 32
x = np.sin(np.pi*np.arange(1,n+1)/(n+1))
mode = np.outer(x,x)
nu, omega = .01, -.7
basis = np.zeros((11,n,n)); basis[0] = mode
f = mode/(8*nu*33**2*np.sin(np.pi/66)**2+1j*omega)
sensors = np.array([[0,0],[11,16],[15,15],[31,31]])
g = f[tuple(sensors.T)]
y = 1.1e-8*g
lam2 = 1e-4
p = dict(Lx=1.,Ly=1.,nu=nu,omega=omega,vx=np.zeros_like(mode),vy=np.zeros_like(mode),sigma=np.zeros_like(mode),basis=basis,sensor_ij=sensors,y_obs_re=y.real,y_obs_im=y.imag,lam2=lam2,lam0=1e-6,pair_penalties=[])
out = mod.solve_nonhermitian_coupled_inversion(p)
w = 1e-8
expected_field = abs(w*f)
expected_j = .5*(np.linalg.norm(w*g-y)**2+lam2*w*w)
print('cutoff weight', out['weights_real'][0], 'expected', w)
print('cutoff objective gap', out['objective']-expected_j, 'relative field error', np.linalg.norm(out['predicted_field_abs']-expected_field)/np.linalg.norm(expected_field))
assert np.linalg.norm(out['predicted_field_abs']-expected_field) <= 1e-5*np.linalg.norm(expected_field)
