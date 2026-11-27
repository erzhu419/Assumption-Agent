import importlib.util
import pickle
from pathlib import Path
import numpy as np
from scipy.linalg import solve

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

p = Path('/holdout_task')
m = load('candidate', p/'beta_m97_c03_batch_01_solver.py')
a = load('prior', '/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py')
item = pickle.loads((p/'public_beta_m97_c03_batch_01.pkl').read_bytes())[0]
s, gt = item['study'], item['gt']
nx, ny = s['nx'], s['ny']
source = np.asarray(s['source_modes']).reshape(-1, nx*ny).T
ij = np.asarray(s['sensor_ij']); rows = ij[:,0]*ny + ij[:,1]
worst = dict(design=0., terminal=0., neumann=0., kkt=0.)
for mode, trace_name in [(0,'measured'), (1,'measured'), (0,'threshold')]:
    q = dict(s, recirculation_loop_mode=mode)
    if trace_name == 'threshold':
        q['past_vortex_circulation_trace'] = np.minimum(s['past_vortex_circulation_trace'], s['critical_vortex_circulation'])
    excess = [float(v)-s['critical_vortex_circulation'] for v in q['past_vortex_circulation_trace'] if v>s['critical_vortex_circulation']]
    memory = sum(t*t for t in excess)
    sealed = mode == 0 and bool(excess)
    axes = []
    for tag,n in [('x',nx),('y',ny)]:
        k = np.asarray(q['kappa_'+tag]); length = q['L'+tag]
        op = a._axis(k, np.asarray(q['vel_'+tag]), q['robin_'+tag], length).toarray()
        if sealed:
            for end,neighbor in [(0,1),(n-1,n-2)]:
                d = k[end]*((n-1)/length)**2
                op[end] = 0.; op[end,end] = d; op[end,neighbor] = -d
            worst['neumann'] = max(worst['neumann'], float(np.max(np.abs(op[[0,-1]].sum(axis=1)))))
    
        axes.append(op)
    x, y = np.meshgrid(np.linspace(0, q['Lx'], nx), np.linspace(0, q['Ly'], ny), indexing='ij')
    v = np.array(q['V_reaction_2d'], copy=True)
    if sealed:
        v += (0.95+0.55*np.tanh(memory))*(np.exp(-16*(x/q['Lx'])**2)+np.exp(-16*(1-x/q['Lx'])**2))*(0.6+0.4*np.sin(np.pi*y/q['Ly']))
    operator = np.kron(axes[0],np.eye(ny)) + np.kron(np.eye(nx),axes[1]) + np.diag(v.ravel())
    blocks = []
    for mu in q['mu_shifts']:
        dense = solve(operator+mu*np.eye(nx*ny),source)
        blocks.append(dense[rows])
    stack = np.vstack(blocks)
    design, _ = m._forward(q)
    out = m.solve_beta_m97_c03_batch_01(q)
    w = out['weights']; cap = q['u_box']; c = out['kkt_certificate']
    d_err = float(np.max(np.abs(stack-design)))
    f_err = float(np.max(np.abs((dense@w).reshape(nx,ny)-out['terminal_field'])))
    z = q['Z_nuisance']; obs = np.column_stack((m._calibrate(q),stack))
    r = obs-z@solve(z.T@z+q['nuisance_ridge']*np.eye(z.shape[1]),z.T@obs,assume_a='pos')
    d = r[:,1:].T@(r[:,1:]@w-r[:,0])+q['source_ridge']*w-c['nu_sum']
    k_err = max(abs(float(w.sum()-q['total_budget'])),float(np.max(np.abs(w-np.clip(w-d,0,cap)))))
    print('mode',mode,trace_name,'sealed',sealed,'memory',memory,'design/terminal/KKT',d_err,f_err,k_err)
    for key,val in [('design',d_err),('terminal',f_err),('kkt',k_err)]:
        worst[key] = max(worst[key],val)
    assert max(d_err,f_err,k_err)<1e-9, worst
    if sealed:
        nuisance_error = float(np.max(np.abs(r[:,0]-gt['r_y'])))
        design_error = float(np.max(np.abs(r[:,1:]-gt['S_tilde'])))
        print('measured profiled reference target/design errors',nuisance_error,design_error)
        assert max(nuisance_error,design_error)<1e-6
print('worst across mode and trace slices',worst)
assert worst['neumann']<1e-12
