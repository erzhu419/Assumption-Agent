import importlib.util
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/beta_w04_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_beta_w04_batch_04.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']

# Assemble the stencil cell by cell, independently of the candidate's Kronecker factors.
def matrix(multiplier, reaction):
    nx, ny = s['nx'], s['ny']
    rows, cols, vals = [], [], []
    for i in range(nx):
        for j in range(ny):
            r = i * ny + j
            d = reaction[i, j]
            for a, idx, stride in [('x', i, ny), ('y', j, 1)]:
                k = multiplier * s['kappa_' + a]
                h = s['L' + a] / len(k)
                v, robin = s['vel_' + a][idx], s['robin_' + a]
                d += s['reaction_' + a][idx] + abs(v) / h
                for step in (-1, 1):
                    other = idx + step
                    if 0 <= other < len(k):
                        q = 2 * k[idx] * k[other] / ((k[idx] + k[other]) * h*h)
                        d += q
                        rows.append(r); cols.append(r + step*stride); vals.append(-q - max(-step*v, 0)/h)
                    else:
                        d += robin*k[idx] / (h*(robin*h/2+k[idx]))
            rows.append(r); cols.append(r); vals.append(d)
    return coo_matrix((vals, (rows, cols)), shape=(nx*ny, nx*ny)).tocsc()

from scipy.sparse import eye
v = s['V_reaction_2d']
additive = v.mean(0) + v.mean(1)[:, None] - v.mean()
idx = np.asarray(s['sensor_ij'])
flat = idx[:, 0]*s['ny'] + idx[:, 1]
rhs = s['source_modes'].reshape(len(s['source_modes']), -1).T
identity = eye(v.size, format='csc')
actual, terminal = m._forward(s)
actual = actual.reshape(len(s['mu_shifts']), len(flat), -1)
def forward(op):
    fields = [splu(op + mu*identity).solve(rhs) for mu in s['mu_shifts']]
    return np.array([u[flat] for u in fields]), fields[-1]
def rel(a, b):
    return np.linalg.norm(a-b)/np.linalg.norm(b)
expected, last = forward(matrix(1.45, v))
err = max(rel(a, b) for a,b in zip(actual, expected))
terminal_err = rel(terminal.reshape(last.shape), last)
print('Worst shift design error / terminal mode error:', err, terminal_err)
assert max(err, terminal_err) < 1e-10
for label, mult, reaction in [('no eddy enhancement', 1., v), ('additive reaction', 1.45, additive)]:
    alt, _ = forward(matrix(mult, reaction))
    print(label, 'relative design change:', rel(alt, expected))

c = s['sensor_calibration_sweeps']
base = m._calibrate(s)
for bank in range(4):
    rows = s['row_bank_id'] == bank
    ids = np.flatnonzero(c['bank_id'] == bank)
    worst_ref = worst_row = 0.
    for i in ids:
        train = ids[ids != i]
        g, o = m._fit_bank(c['ref_signal'][train], c['raw_readout'][train])
        worst_ref = max(worst_ref, abs(g*c['ref_signal'][i] + o - c['raw_readout'][i]))
        sweeps = {k: np.delete(val, i) for k,val in c.items()}
        worst_row = max(worst_row, np.max(np.abs(m._calibrate(dict(s, sensor_calibration_sweeps=sweeps))[rows] - base[rows])))
    print('Bank', bank, 'max held-out reference error / sensor correction change:', worst_ref, worst_row)
    assert max(worst_ref, worst_row) < 1e-6
hist = json.loads(Path('/tmp/hegel_continual_archive/archive_calibration_history.json').read_text())
for bank in range(4):
    here = c['bank_id'] == bank
    x, y = c['ref_signal'][here], c['raw_readout'][here]
    g, o = m._fit_bank(x, y)
    past = []
    for episode in hist:
        b, ref, raw = (np.asarray(episode[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
        ref, raw = ref[b == bank], raw[b == bank]
        if len(ref) >= 2 and np.ptp(ref) > 1e-12:
            past.append(m._fit_bank(ref, raw))
    if past:
        gp, op = past[-1]
        # One current observation identifies an offset conditional on historical gain, not a new slope.
        miss = abs(gp-g)*np.ptp(x)
        print('Bank', bank, 'latest historical vs current gain / one-spot end-to-end drift:', gp, g, miss)
print('Nuisance rank and column count:', np.linalg.matrix_rank(s['Z_nuisance']), s['Z_nuisance'].shape[1])
