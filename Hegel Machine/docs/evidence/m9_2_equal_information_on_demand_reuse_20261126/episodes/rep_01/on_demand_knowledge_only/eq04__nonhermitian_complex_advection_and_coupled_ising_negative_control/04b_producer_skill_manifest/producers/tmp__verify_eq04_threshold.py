import importlib.util
import itertools
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/nonhermitian_advection_coupled_select.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Lx, Ly, nu = .81, 1.73, .015
hx, hy = Lx/33, Ly/33
x = np.arange(1, 33)
ks = [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(3,2),(2,3),(3,3),(4,1),(1,4)]
phi = np.stack([2/33 * np.outer(np.sin(np.pi*a*x/33), np.sin(np.pi*b*x/33)) for a,b in ks])
eig = np.array([4*nu*(np.sin(np.pi*a/66)**2/hx**2 + np.sin(np.pi*b/66)**2/hy**2) for a,b in ks])
sigma = -eig[0] + 1e-5
edges = [(0,1,.65),(0,2,.42),(1,3,.35),(2,3,.40),(3,4,.38),(4,5,.24),(0,5,.40),(7,10,9.)]

def fixture(omega, scales, beta, lam0, lam2):
    denom = eig + sigma + 1j*omega
    gains = scales*abs(denom)/denom
    y = np.einsum('m,mij->ij', beta, phi)
    p = dict(Lx=Lx,Ly=Ly,nu=nu,omega=omega,sigma=np.full((32,32),sigma),vx=np.zeros((32,32)),vy=np.zeros((32,32)),basis=(scales*abs(denom))[:,None,None]*phi,sensor_ij=np.indices((32,32)).reshape(2,-1).T,y_obs_re=y.real.ravel(),y_obs_im=y.imag.ravel(),lam0=lam0,lam2=lam2,pair_penalties=edges)
    return p, gains

for omega in (-1e-3,1e-3):
    beta = np.array([1.05,.93,.84,.81,.74,.71,.23,0,.19,.11,0])*np.exp(1j*np.arange(11)*.91)
    beta[10] = 5e-9*(1+.2j)
    p, gains = fixture(omega,np.ones(11),beta,.075,.13)
    w_ridge = gains.conj()*beta/(abs(gains)**2+p['lam2'])
    best = (np.inf,None)
    for flags in itertools.product((False,True),repeat=11):
        w = w_ridge * flags
        active = abs(w)>1e-8
        obj = .5*np.sum(abs(gains*w-beta)**2 + p['lam2']*abs(w)**2)+p['lam0']*sum(active)+sum(c for u,v,c in edges if active[u] and active[v])
        if obj < best[0]: best = obj,w
    got = module.solve_nonhermitian_coupled_inversion(p)
    field = abs(np.einsum('m,mij->ij',gains*best[1],phi))
    jerr = abs(got['objective']-best[0]); ferr = np.linalg.norm(field-got['predicted_field_abs'])/np.linalg.norm(field)
    wf = got['weights_real']+1j*got['weights_imag']
    print(f'near-resonant omega={omega:+g}: objective_abs={jerr:.3g}, field_rel={ferr:.3g}, subthreshold_weight={abs(wf[10]):.3g}')
    assert jerr<=1e-6 and ferr<=1e-5 and abs(wf[10])<=1e-8

# An orthogonal mode with a large response makes a subthreshold weight
# physically significant. The inactive disk then has a nonzero boundary optimum.
scales = np.ones(11); scales[0] = 1e8
beta = np.zeros(11,dtype=complex); beta[0] = 2
p, gains = fixture(.7,scales,beta,3.,.13)
got = module.solve_nonhermitian_coupled_inversion(p)
radius = 1e-8
inactive = .5*((2-abs(gains[0])*radius)**2 + p['lam2']*radius**2)
active = 3 + .5*4*p['lam2']/(abs(gains[0])**2+p['lam2'])
assert inactive < active
field = abs(phi[0])*abs(gains[0])*radius
jerr = abs(got['objective']-inactive); ferr = np.linalg.norm(got['predicted_field_abs']-field)/np.linalg.norm(field)
print(f'inactive-disk slice: expected_objective={inactive:.12g}, returned={got["objective"]:.12g}, objective_abs={jerr:.3g}, field_rel={ferr:.3g}',flush=True)
assert jerr <= 1e-6 and ferr <= 1e-5
