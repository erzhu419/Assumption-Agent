import importlib.util
import numpy as np
from scipy.linalg import lstsq

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/nonhermitian_advection_coupled_select.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
rng = np.random.default_rng(2504)
n, m = 32, 11
lx, ly, nu = 1.2, .73, .002
hx, hy = lx/33, ly/33
vx, vy, omega = .11, -.13, -.003
dx, dy, ux, uy = nu/hx**2, nu/hy**2, vx/hx, -vy/hy
j = np.arange(1, n+1)
# A separable upwind operator is diagonally similar to two symmetric
# tridiagonals. Build the reference from their analytic sine modes.
phi = np.sqrt(2/33)*np.sin(np.pi*j[:,None]*j[None,:]/33)
sx = (np.sqrt((dx+ux)/dx))**(j-1)
sy = (np.sqrt(dy/(dy+uy)))**(j-1)
ex = 2*dx+ux-2*np.sqrt(dx*(dx+ux))*np.cos(np.pi*j/33)
ey = 2*dy+uy-2*np.sqrt(dy*(dy+uy))*np.cos(np.pi*j/33)
sigma = -ex[0]-ey[0]-.0005
scale = sx[:,None]*sy[None,:]
basis = rng.normal(size=(m,n,n))
fields = np.stack([scale*(phi@((phi.T@(b/scale)@phi)/(ex[:,None]+ey[None,:]+sigma+1j*omega))@phi.T) for b in basis], axis=-1)
worst = np.zeros(3)
for k in (5, 57):
    ij = np.column_stack(divmod(rng.choice(n*n,k,replace=False),n))
    g = fields[ij[:,0],ij[:,1]]
    truth = np.zeros(m, complex)
    truth[[1,4,8]] = [1.2+.9j,-.8+.5j,1.1-.6j]
    y = g@truth + .002*(rng.normal(size=k)+1j*rng.normal(size=k))
    lam0, lam2 = .0002, .00005
    pairs = [[1,4,.015],[4,8,.009],[0,6,.006]]
    p = dict(Lx=lx,Ly=ly,nu=nu,vx=np.full((n,n),vx),vy=np.full((n,n),vy),sigma=np.full((n,n),sigma),omega=omega,basis=basis,sensor_ij=ij,y_obs_re=y.real,y_obs_im=y.imag,lam0=lam0,lam2=lam2,pair_penalties=pairs)
    def score(w):
        a = np.abs(w)>1e-8
        return .5*np.linalg.norm(g@w-y)**2 + .5*lam2*np.linalg.norm(w)**2 + lam0*a.sum() + sum(c for u,v,c in pairs if a[u] and a[v])
    best, wref = score(np.zeros(m)), np.zeros(m,complex)
    for mask in range(1,1<<m):
        s = [i for i in range(m) if mask>>i&1]
        w = np.zeros(m,complex)
        w[s] = lstsq(np.vstack((g[:,s],np.sqrt(lam2)*np.eye(len(s)))),np.r_[y,np.zeros(len(s))],lapack_driver='gelsy')[0]
        val = score(w)
        if val < best:
            best, wref = val, w
    out = mod.solve_nonhermitian_coupled_inversion(p)
    w = out['weights_real']+1j*out['weights_imag']
    ref_abs = np.abs(fields@wref)
    errs = np.array([np.linalg.norm(mod._responses(p)-fields)/np.linalg.norm(fields),max(abs(out['objective']-best),abs(out['objective']-score(w))),np.linalg.norm(out['predicted_field_abs']-ref_abs)/np.linalg.norm(ref_abs)])
    print(f'Near-resonant operator, {k} sensors: forward relative L2={errs[0]:.3e}, objective absolute error={errs[1]:.3e}, field relative L2={errs[2]:.3e}')
    assert np.all(errs <= [1e-8,1e-6,1e-5]), errs
    worst = np.maximum(worst, errs)
print('Audit worst:', worst)
