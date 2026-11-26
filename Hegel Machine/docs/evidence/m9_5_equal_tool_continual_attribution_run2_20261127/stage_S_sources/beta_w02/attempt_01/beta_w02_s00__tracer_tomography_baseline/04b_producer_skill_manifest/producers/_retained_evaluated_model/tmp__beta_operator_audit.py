import importlib.util
import pickle
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

p = '/holdout_task/'
def load(name):
    spec = importlib.util.spec_from_file_location(name, p + name + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

solver = load('beta_w02_s00_solver')
# The validator runs its CLI at import time. Load only the independent check.
import ast
import math
from types import SimpleNamespace
path = p + 'validate_public_schema.py'
tree = ast.parse(open(path).read(), filename=path)
check = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == 'verify_box_simplex_kkt_certificate')
scope = {'np': np, 'math': math}
exec(compile(ast.Module(body=[check], type_ignores=[]), path, 'exec'), scope)
validator = SimpleNamespace(verify_box_simplex_kkt_certificate=scope[check.name])
for case in pickle.load(open(p + 'public_beta_w02_s00.pkl', 'rb')):
    s, gt = case['study'], case['gt']
    nx, ny = s['nx'], s['ny']
    axes = []
    for axis, n in zip('xy', (nx, ny)):
        h = s['L' + axis] / n
        k, v, r = [s[name + '_' + axis] for name in ('kappa', 'vel', 'robin')]
        A = np.diag(s['reaction_' + axis].copy())
        for i in range(n - 1):
            f = 2 / (1 / k[i] + 1 / k[i+1]) / (h*h)
            A[i, i] += f
            A[i+1, i+1] += f
            A[i, i+1] -= f
            A[i+1, i] -= f
        for i in (0, n-1):
            A[i, i] += r*k[i] / (h*(k[i] + r*h/2))
        for i, speed in enumerate(v / h):
            A[i, i] += abs(speed)
            j = i - 1 if speed >= 0 else i + 1
            if 0 <= j < n:
                A[i, j] -= abs(speed)
        axes.append(A)
    V = s['V_reaction_2d']
    L = kron(axes[0], eye(ny)) + kron(eye(nx), axes[1]) + diags(V.ravel())
    rhs = s['source_modes'].reshape(-1, nx*ny).T
    i, j = s['sensor_ij'].T
    direct = []
    for shift in s['mu_shifts']:
        u = splu((L + shift*eye(nx*ny)).tocsc()).solve(rhs)
        direct.append(u[i*ny+j])
    D, terminal = solver._forward(s)
    exact = np.concatenate(direct)
    n_sensor = len(i)
    shift_errors = [np.linalg.norm(a-b)/np.linalg.norm(b) for a,b in zip(D.reshape(len(direct), n_sensor, -1), direct)]
    end_error = np.linalg.norm(terminal.reshape(rhs.shape[1], -1).T-u)/np.linalg.norm(u)
    assert max(max(shift_errors), end_error) < 1e-10
    print('Independent sparse solve: worst shift relative error', max(shift_errors), 'terminal basis relative error', end_error)
    cap = s['u_box']
    m = rhs.shape[1]
    for b in (0, cap, 2.5*cap, (m-1)*cap, m*cap):
        out = solver.solve_beta_w02_s00({**s, 'total_budget': b})
        cert = validator.verify_box_simplex_kkt_certificate(gt['S_tilde'], gt['r_y'], s['source_ridge'], cap, b, out['kkt_certificate'])
        print('Public observations, budget', b, 'bound counts', np.sum(out['weights']==0), np.sum(out['weights']==cap), 'independent certificate', cert)
        assert cert['valid'], cert
