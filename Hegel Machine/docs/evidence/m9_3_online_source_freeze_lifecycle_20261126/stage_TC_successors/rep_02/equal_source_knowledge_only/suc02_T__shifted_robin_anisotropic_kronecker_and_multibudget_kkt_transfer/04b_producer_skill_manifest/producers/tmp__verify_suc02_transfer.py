import importlib.util
import pickle
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root / 'multibudget_kronecker_kkt_transfer_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
source = (root / 'validate_public_schema.py').read_text()
validator = {}
exec(source.split('\n_d = ')[0], validator)
verify = validator['verify_box_simplex_kkt_certificate']
item = pickle.loads((root / 'public_suc02_T_studies.pkl').read_bytes())[0]
public, gt = item['study'], item['gt']

def axis(length, k, v, r):
    n = len(k)
    h = length / n
    mat = np.zeros((n, n))
    for i in range(n):
        for j in (i-1, i+1):
            if 0 <= j < n:
                coeff = 2 / (1/k[i] + 1/k[j]) / h**2
                mat[i,j] -= coeff
                mat[i,i] += coeff
        mat[i,i] += abs(v[i])/h
        j = i-1 if v[i] >= 0 else i+1
        if 0 <= j < n:
            mat[i,j] -= abs(v[i])/h
    for i in (0, n-1):
        mat[i,i] += r / (1 + r*h/(2*k[i])) / h
    return mat

def check(name, st, refs=None):
    m, nx, ny = st['source_modes'].shape
    ax, by = refs if refs is not None else [axis(st['L'+a], st['kappa_'+a], st['vel_'+a], st['robin_'+a]) for a in 'xy']
    ident = sparse.eye(nx*ny, format='csc')
    kron = sparse.kron(ax, sparse.eye(ny), format='csc') + sparse.kron(sparse.eye(nx), by, format='csc')
    si, sj = st['sensor_ij'].T
    rhs = st['source_modes'].reshape(m,-1).T
    samples = []
    for shift in st['mu_shifts']:
        terminal = splu(kron + shift*ident).solve(rhs).T.reshape(m,nx,ny)
        samples.append(terminal[:,si,sj].T)
    raw = np.vstack(samples)
    candidate_raw, candidate_terminal = mod._sensor_design(st)
    raw_error = np.max(abs(candidate_raw-raw))
    mode_error = np.max(abs(candidate_terminal-terminal))
    assert np.allclose(candidate_raw, raw, rtol=2e-8, atol=2e-12), (name,raw_error)
    assert np.allclose(candidate_terminal, terminal, rtol=2e-8, atol=2e-12), (name,mode_error)
    z = st['Z_nuisance']
    aug_z = np.vstack([z, np.sqrt(st['nuisance_ridge'])*np.eye(z.shape[1])])
    values = np.column_stack([raw, st['y_obs']])
    coefficients = np.linalg.lstsq(aug_z,np.vstack([values, np.zeros((z.shape[1], values.shape[1]))]),rcond=None)[0]
    values -= z @ coefficients
    design, target = values[:,:-1], values[:,-1]
    out = mod.solve_multibudget_kronecker_transfer(st)
    w = out['weights_by_budget']
    gaps = []
    for row, b, cert in zip(w,st['budget_grid'],out['kkt_certificates']):
        assert np.array_equal(row, cert['weights']), name
        result = verify(design,target,st['source_ridge'],st['u_box'],b,cert)
        assert result['valid'], (name,b,result)
        gaps.append(result['gap'])
    field = np.einsum('m,mij->ij',w[1],terminal)
    assert np.allclose(out['terminal_field_mid_budget'], field, rtol=2e-8, atol=2e-12), name
    print(name, 'stack_max_error', raw_error, 'terminal_modes_max_error',mode_error,'worst_KKT_gap',max(gaps),'active_lower/upper',[((r==0).sum(),(r==st['u_box']).sum()) for r in w])
    return out

out = check('public',public,(gt['Ax'],gt['By']))
for key in ('weights_by_budget','terminal_field_mid_budget'):
    print('public reference',key,'max_error',np.max(abs(out[key]-gt[key])))
assert np.max(abs(out['terminal_field_mid_budget']-gt['terminal_field_mid_budget'])) < 1e-6
st = dict(public, Lx=1.15, Ly=2.8, vel_x=-public['vel_x'], vel_y=-public['vel_y'], robin_x=2.35, robin_y=0.12, u_box=.23, budget_grid=np.array([0.,1.15,16*.23]))
check('reversed anisotropic / budget endpoints', st)
st = dict(st, vel_x=st['vel_x']*np.cos(np.linspace(0, 2*np.pi,48)), vel_y=st['vel_y']*np.cos(np.linspace(0, 3*np.pi,36)), source_ridge=.0001, mu_shifts=np.geomspace(.003,12.,12),budget_grid=np.array([.03,1.15,3.62]))
check('mixed signs / wider shifts / active caps', st)
