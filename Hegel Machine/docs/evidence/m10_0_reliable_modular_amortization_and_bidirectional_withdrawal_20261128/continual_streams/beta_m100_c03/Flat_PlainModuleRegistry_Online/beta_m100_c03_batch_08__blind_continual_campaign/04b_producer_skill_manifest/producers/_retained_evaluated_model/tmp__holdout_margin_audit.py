import itertools
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/holdout_task')
from _verified_upstream import _fit, _invert
from downstream_solver import _responses
p = {}
exec(Path('/tmp/probe_walls.py').read_text().split('sx,sy=')[0], p)
s, y, w = p['s'], p['y'], p['weights']
raw_study = p['item']['study']
ref = np.asarray(raw_study['sensor_calibration_sweeps']['ref_rosette'])
raw = np.asarray(raw_study['sensor_calibration_sweeps']['raw_rosette'])
cal_err = []
for fold in range(5):
    hold = np.arange(len(raw)) % 5 == fold
    cal_err.append(_invert(raw[hold], _fit(ref[~hold], raw[~hold])) - ref[hold])
cal_err = np.concatenate(cal_err)
print('out-of-fold calibration inverse RMSE per channel:', np.sqrt(np.mean(cal_err**2, axis=0)))
sx, sy = [p['states'](d) for d in 'xy']
rows = [[state[j] for state in axis] for axis in (sx, sy) for j in range(2)]
keys = list(itertools.product(range(3), repeat=4))
coefs = [np.array([rows[i][k] for i, k in enumerate(key)]) for key in keys]
base = coefs[keys.index((2,2,1,1))]
for wall in range(4):
    d, edge = 'xy'[wall//2], wall%2
    k, v = s['kappa_'+d], s['vel_'+d]
    n = len(k); h = s['L'+d]/(n-1); i, j = (0,1) if edge == 0 else (n-1,n-2)
    face = (k[i]+k[j])/(2*h*h)
    rob = s['robin_'+d]/h
    inward = max(0., (-1 if edge == 0 else 1)*v[i])/h
    outward = max(0., (1 if edge == 0 else -1)*v[i])/h
    seam = (k[0]+k[-1])/(2*h*h)
    base_in, base_seam, base_leak = base[wall]
    for label, row in [('face', (face,base_seam,base_leak)),('double', (2*base_in,base_seam,base_leak)),('leaky', (base_in,base_seam,rob)),('inward_upwind',(face+inward,base_seam,base_leak)),('reversed_upwind',(face+outward,seam+inward if d=='x' else 0,base_leak)),('diffusion_seam',(base_in,seam,base_leak))]:
        a = base.copy(); a[wall] = row
        if any(np.allclose(a, b, rtol=0, atol=1e-12) for b in coefs):
            continue
        keys.append((wall,label)); coefs.append(a)
errors = np.stack([p['pred'](a.ravel())-y for a in coefs])
shift_count = len(s['mu_shifts'])
Z = s['Z_nuisance']
S, _ = _responses(s)
assert np.max(np.abs((S@w-y)-errors[keys.index((2,2,1,1))])) < 1e-12
for profile in (False, True):
    E = errors.reshape(len(keys), shift_count, -1, y.shape[1]).copy()
    if profile:
        for j, z in enumerate(np.array_split(Z, shift_count)):
            q = np.linalg.qr(z, mode='reduced')[0]
            E[:,j] -= q @ (q.T @ E[:,j])
    floors, margins = [], []
    for sh, pul in itertools.product(range(shift_count),range(y.shape[1])):
        shifts = np.arange(shift_count) != sh
        pulses = np.arange(y.shape[1]) != pul
        score = np.mean(E[:,shifts][:,:,:,pulses]**2, axis=(1,2,3))
        i,j = np.argsort(score)[:2]
        floors.append(float(np.sqrt(np.mean(E[i,sh,:,pul]**2))))
        margins.append(float(np.sqrt(score[j])-np.sqrt(score[i])))
        assert keys[i] == (2,2,1,1), (profile,sh,pul,keys[i])
    assert max(floors) < 5e-6
    print('within-shift nuisance profiling:', profile, 'wall candidates:', len(keys), 'worst joint held-out RMSE:', max(floors), 'minimum training RMSE margin:', min(margins))
