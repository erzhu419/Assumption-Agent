import importlib.util
import pickle
import numpy as np
from scipy.sparse import eye, diags, kron
from scipy.sparse.linalg import splu


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load('candidate', '/holdout_task/batch_solver.py')
archive = load('archive', '/tmp/hegel_continual_archive/beta_m98_v2_c03_batch_02_solver.py')
rmse = lambda e: np.sqrt(np.mean(np.square(e)))
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    s = item['study']
    r = s['sensor_calibration_sweeps']['ref_rosette']
    raw = s['sensor_calibration_sweeps']['raw_rosette']
    X = np.c_[np.ones(len(r)), r]
    for kind in ('diagonal', 'full', 'pairwise', 'quadratic'):
        errs = []
        for j in range(4):
            D = X[:, [0,j+1]] if kind == 'diagonal' else X
            if kind in ('pairwise', 'quadratic'):
                D = np.c_[D, *[r[:,p]*r[:,q] for p in range(4) for q in range(p+1,4)]]
            if kind == 'quadratic':
                D = np.c_[D, r*r]
            pinv = np.linalg.pinv(D)
            err = (raw[:,j]-D@(pinv@raw[:,j]))/(1-np.sum(D*pinv.T,axis=1))
            errs.append(rmse(err))
        print('calibration LOO', kind, 'worst channel RMSE', max(errs))
    nx, ny = s['nx'], s['ny']
    pulses = item['campaign_diagnostics']
    W = np.column_stack([p['w_pulse'] for p in pulses])
    Y = np.column_stack([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses])
    F = np.column_stack([a.ravel(order='F') for a in np.tensordot(W.T,s['source_modes'],axes=1)])
    ix, iy = s['sensor_ij'].T
    sr = ix+nx*iy
    I = eye(nx*ny,format='csc')
    V = diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    choices = []
    for d,n in [('x',nx),('y',ny)]:
        k,v = s['kappa_'+d],s['vel_'+d]
        h = s['L'+d]/(n-1)
        periodic = m._axis(k,v,h)
        assert np.max(np.abs(periodic @ np.ones(n))) < 1e-11
        reflect = archive._axis(k,v,h,0)
        wrap = reflect.tolil()
        q = (k[0]+k[-1])/(2*h*h)
        wrap[0,-1] = wrap[-1,0] = -q
        wrap[0,0] += q
        wrap[-1,-1] += q
        choices.append({'periodic':periodic,'reflecting':reflect,'diffusive-wrap':wrap.tocsc(),'open':archive._axis(k,v,h,s['robin_'+d])})
    E = {}
    for x,ax in choices[0].items():
        for y,ay in choices[1].items():
            L = kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+V
            E[x,y] = np.vstack([splu(L+t*I).solve(F)[sr] for t in s['mu_shifts']])-Y
    bank = np.asarray(s['row_bank_id']).ravel()
    if len(bank) == len(sr):
        bank = np.tile(bank, len(s['mu_shifts']))
    assert len(bank) == len(Y)
    shift = np.repeat(np.arange(len(s['mu_shifts'])),len(sr))
    for train in range(len(pulses)):
        scores = sorted((rmse(e[:,train]), k) for k,e in E.items())
        best = scores[0][1]
        assert best == ('periodic','periodic')
        held = E[best][:,1-train]
        worst = max(rmse(held[(bank==b)&(shift==t)]) for b in np.unique(bank) for t in np.unique(shift) if np.any((bank==b)&(shift==t)))
        print(f'train pulse {train}: {best}, train RMSE {scores[0][0]:.6g}, runner-up {scores[1][0]:.6g}; held pulse RMSE {rmse(held):.6g}; worst held bank-by-shift RMSE {worst:.6g}')
        assert worst < 1e-5
