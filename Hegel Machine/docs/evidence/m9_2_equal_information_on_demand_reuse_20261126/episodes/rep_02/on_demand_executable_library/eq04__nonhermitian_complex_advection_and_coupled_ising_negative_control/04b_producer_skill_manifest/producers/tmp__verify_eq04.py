import importlib.util
import itertools
import time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

path = '/holdout_task/nonhermitian_advection_coupled_select.py'
spec = importlib.util.spec_from_file_location('candidate', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def reference_field(p):
    m, nx, ny = p['basis'].shape
    hx, hy = p['Lx']/(nx+1), p['Ly']/(ny+1)
    ax, ay = p['nu']/hx**2, p['nu']/hy**2
    matrix = lil_matrix((nx*ny, nx*ny), dtype=np.complex128)
    for i, j in itertools.product(range(nx), range(ny)):
        r = i*ny+j
        matrix[r, r] = 2*ax+2*ay+p['sigma'][i,j]+1j*p['omega']
        for di, dj, c in ((-1,0,ax),(1,0,ax),(0,-1,ay),(0,1,ay)):
            if 0 <= i+di < nx and 0 <= j+dj < ny:
                matrix[r, (i+di)*ny+j+dj] = -c
        for axis, h, vel in ((0,hx,p['vx'][i,j]),(1,hy,p['vy'][i,j])):
            c = abs(vel)/h
            matrix[r,r] += c
            step = -1 if vel >= 0 else 1
            ii, jj = (i+step,j) if axis == 0 else (i,j+step)
            if 0 <= ii < nx and 0 <= jj < ny:
                matrix[r,ii*ny+jj] -= c
    matrix = matrix.tocsc()
    rhs = p['basis'].reshape(m,-1).T.astype(complex)
    return matrix, rhs, spsolve(matrix, rhs)

def objective(g,y,w,p):
    a = abs(w)>1e-8
    return (0.5*np.linalg.norm(g@w-y)**2 + 0.5*p['lam2']*np.linalg.norm(w)**2
            + p['lam0']*a.sum() + sum(q for u,v,q in p['pair_penalties'] if a[u] and a[v]))

def reference_select(g,y,p):
    m = g.shape[1]
    w = np.zeros(m,dtype=complex)
    best = w.copy()
    val = objective(g,y,w,p)
    for mask in range(1,1<<m):
        s = [j for j in range(m) if (mask>>j)&1]
        a = np.vstack((g[:,s],np.sqrt(p['lam2'])*np.eye(len(s))))
        b = np.concatenate((y,np.zeros(len(s))))
        w.fill(0)
        w[s] = np.linalg.lstsq(a,b,rcond=None)[0]
        candidate = objective(g,y,w,p)
        if candidate < val:
            best,val = w.copy(),candidate
    return best,val

rng = np.random.default_rng(274)
worst = np.zeros(4)
for name,nu,lengths,omega,speed,ns,lam2 in [
    ('mixed_sign_anisotropic',.04,(1.6,.7),-7.,3.,55,1e-5),
    ('advection_dominated',1e-5,(.8,2.1),2.5,11.,32,1e-6),
    ('rapid_oscillation',.013,(1.,1.),160.,0.,80,1e-8),
    ('redundant_basis_few_sensors',.01,(1.1,.9),-12.,1.4,6,1e-10),
    ('diffusion_and_empty_support',.2,(.7,1.3),.07,.05,70,1e-5),
]:
    n=32
    x,ygrid=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n),indexing='ij')
    basis=np.stack([np.sin(np.pi*(j%4+1)*x)*np.cos(np.pi*(j//4+1)*ygrid) for j in range(11)])
    basis += rng.normal(0,.12,basis.shape)
    if 'redundant' in name:
        basis[1]=basis[0]+1e-7*basis[1]
    sh=(n,n)
    flat=np.r_[0,n-1,n*(n-1),n*n-1,rng.choice(np.arange(1,n*n-1),ns-4,replace=False)]
    p=dict(Lx=lengths[0],Ly=lengths[1],nu=nu,omega=omega,
           vx=speed*rng.uniform(-1,1,sh),vy=speed*rng.uniform(-1,1,sh),
           sigma=rng.uniform(.1,1,sh),basis=basis,
           sensor_ij=np.column_stack(np.unravel_index(flat,sh)),lam2=lam2)
    a,b,field=reference_field(p)
    wanted=np.zeros(11,dtype=complex)
    wanted[[0,2,7]]=[1+1.4j,-2+.6j,.7-1.6j]
    g=field[flat]
    y=g@wanted
    energy=np.vdot(y,y).real
    p.update(y_obs_re=y.real,y_obs_im=y.imag,
             lam0=(2 if 'empty' in name else .025)*energy,
             pair_penalties=[(0,2,.35*energy),(2,7,.4*energy),(1,8,.1*energy)])
    t=time.perf_counter()
    r=mod.solve_nonhermitian_coupled_inversion(p)
    runtime=time.perf_counter()-t
    w=r['weights_real']+1j*r['weights_imag']
    wt,jt=reference_select(g,y,p)
    ft=abs(field@wt).reshape(sh)
    responses=mod._responses(p,basis)
    metrics=np.array([np.linalg.norm(a@responses-b)/np.linalg.norm(b),abs(r['objective']-jt),
           np.linalg.norm(r['predicted_field_abs']-ft)/(np.linalg.norm(ft) or 1),
           abs(objective(g,y,w,p)-r['objective'])])
    worst=np.maximum(worst,metrics)
    assert set(r)=={'weights_real','weights_imag','objective','predicted_field_abs'}
    assert isinstance(r['objective'],float)
    for key, shape in [('weights_real',(11,)),('weights_imag',(11,)),('predicted_field_abs',sh)]:
        assert r[key].shape==shape and r[key].dtype==np.float64 and np.isfinite(r[key]).all()
    assert np.all(metrics < [1e-9,1e-6,1e-5,1e-6]), (name,metrics)
    if 'empty' in name:
        assert not np.any(w) and not np.any(r['predicted_field_abs'])
    print(name, 'errors', metrics.tolist(), 'seconds',round(runtime,4), 'support',np.flatnonzero(abs(w)>1e-8).tolist())

g=np.eye(11,dtype=complex)
y=np.array([2+1j,1+1j]+[0j]*9)
w0,_=mod._select(g,y,.1,.01,[])
w1,_=mod._select(g,y,.1,.01,[(0,1,2.)])
assert np.array_equal(np.flatnonzero(abs(w0)>1e-8),[0,1])
assert np.array_equal(np.flatnonzero(abs(w1)>1e-8),[0])
print('Pair penalty changes support from [0, 1] to [0].')
print('Worst stencil residual, objective error, relative field error, objective self-consistency:',worst.tolist())
