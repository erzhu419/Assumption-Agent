import importlib.util, pickle
import numpy as np
from scipy.sparse import csc_matrix, eye, kron
from scipy.sparse.linalg import splu
path = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', path+'multibudget_kronecker_kkt_transfer_solver.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
with open(path+'public_suc02_T_studies.pkl','rb') as f:
    item = pickle.load(f)[0]
validator = {}
with open(path+'validate_public_schema.py') as f:
    exec(f.read().split('\n_d =')[0], validator)
verify = validator['verify_box_simplex_kkt_certificate']

def axis(s, a):
    k,v,r = s['kappa_'+a],s['vel_'+a],s['robin_'+a]
    n = len(k); h = s['L'+a]/n; A = np.zeros((n,n))
    for i in range(n):
        A[i,i] = abs(v[i])/h
        for j in (i-1,i+1):
            if 0 <= j < n:
                diffusion = 2/(1/k[i]+1/k[j])/h**2
                A[i,i] += diffusion
                A[i,j] = -diffusion - max(0.,v[i]*np.sign(i-j))/h
            else:
                A[i,i] += r*k[i]/(h*(k[i]+r*h/2))
    return A

def reference(s):
    F = s['source_modes']; m,nx,ny = F.shape
    Ax,By = axis(s,'x'),axis(s,'y')
    K = kron(csc_matrix(Ax),eye(ny)) + kron(eye(nx),csc_matrix(By))
    idx = np.ravel_multi_index(s['sensor_ij'].T,(nx,ny)); blocks=[]
    for mu in s['mu_shifts']:
        U = splu((K+mu*eye(nx*ny)).tocsc()).solve(F.reshape(m,-1).T)
        np.testing.assert_allclose((K@U+mu*U), F.reshape(m,-1).T, atol=2e-11, rtol=2e-10)
        blocks.append(U[idx])
    return np.concatenate(blocks), U.T.reshape(F.shape)

def check(label,s,gt=None):
    S, U = reference(s)
    S1,U1 = mod._transport(s)
    np.testing.assert_allclose(S1,S,atol=2e-11,rtol=2e-9)
    np.testing.assert_allclose(U1,U,atol=2e-11,rtol=2e-9)
    Z=s['Z_nuisance']; T=np.column_stack([S,s['y_obs']]); r=s['nuisance_ridge']
    Qa,ss,_=np.linalg.svd(Z,full_matrices=False)
    Q=Qa[:,ss>np.finfo(float).eps*max(Z.shape)*ss.max(initial=0)]
    if r: Q=Qa*np.sqrt(ss**2/(ss**2+r))
    T=T-Q@(Q.T@T)
    if gt:
        np.testing.assert_allclose(T[:,:-1],gt['S_tilde'],atol=2e-11,rtol=2e-9)
        np.testing.assert_allclose(T[:,-1],gt['r_y'],atol=2e-11,rtol=2e-9)
    out=mod.solve_multibudget_kronecker_transfer(s); w=out['weights_by_budget']
    gaps=[]
    for bi,(b,c) in enumerate(zip(s['budget_grid'],out['kkt_certificates'],strict=True)):
        np.testing.assert_array_equal(w[bi],c['weights'])
        result=verify(T[:,:-1],T[:,-1],s['source_ridge'],s['u_box'],b,c)
        assert result['valid'], (label,bi,result)
        gaps.append(result['gap'])
    np.testing.assert_allclose(out['terminal_field_mid_budget'],np.einsum('m,mij->ij',w[1],U),atol=2e-11,rtol=2e-9)
    if gt:
        np.testing.assert_allclose(w,gt['weights_by_budget'],atol=5e-7,rtol=2e-6)
        np.testing.assert_allclose(out['terminal_field_mid_budget'],gt['terminal_field_mid_budget'],atol=2e-8,rtol=2e-6)
    print(label, 'sensor max error',np.max(abs(S-S1)), 'terminal-mode max error',np.max(abs(U-U1)), 'worst gap',max(gaps), 'active lower/upper',np.count_nonzero(w==0,axis=1),np.count_nonzero(w==s['u_box'],axis=1))

s=item['study']; check('public',s,item['gt'])
t=dict(s,Lx=3.6,Ly=1.1,kappa_x=s['kappa_x']*.28,kappa_y=s['kappa_y']*2.3,vel_x=-2*s['vel_x'],vel_y=-1.8*s['vel_y'],robin_x=.17,robin_y=3.5,mu_shifts=np.geomspace(.015,12,12),budget_grid=np.array([.275,2.2,4.125]),u_box=.275)
S,_=reference(t); t['y_obs']=15*(S[:,2]-S[:,7])+s['Z_nuisance'][:,0]; check('reversed geometry and active bounds',t)
u=dict(t,budget_grid=np.array([0.,2.2,16*t['u_box']]),nuisance_ridge=0.,source_ridge=0.)
u['Z_nuisance']=np.column_stack((t['Z_nuisance'][:,:2],t['Z_nuisance'][:,1]))
check('endpoint budgets and rank-deficient nuisance',u)
