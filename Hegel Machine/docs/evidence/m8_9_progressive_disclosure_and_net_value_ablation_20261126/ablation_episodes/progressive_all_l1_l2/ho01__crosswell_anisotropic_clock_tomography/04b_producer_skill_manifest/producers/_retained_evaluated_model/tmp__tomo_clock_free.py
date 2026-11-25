import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from tomography_solver import _RayModel

with open('/holdout_task/public_surveys.pkl', 'rb') as f:
    entries = pickle.load(f)
for n, entry in enumerate(entries, 1):
    s = entry['survey']
    full = _RayModel(s)
    cols = [0, *range(2, full.design.shape[1])]
    residual = np.empty((2, full.ns - 1, full.nr))
    for i in range(full.ns - 1):
        rows = np.ones(full.ns, dtype=bool)
        rows[i:i+2] = False
        train = dict(s, z_src_m=s['z_src_m'][rows], T_obs_s=s['T_obs_s'][rows])
        a = slice(i*full.nr, (i+1)*full.nr)
        b = slice((i+1)*full.nr, (i+2)*full.nr)
        da = full.design[b] - full.design[a]
        assert np.all(da[:, 2:] == 0)
        for j in range(2):
            model = _RayModel(train)
            if j:
                model.design = model.design[:, cols]
            p = model.solve()
            k = full.kernel(*p[:2])
            residual[j, i] = full.y[b] - full.y[a] - da[:, :2-j] @ p[3:5-j] - p[2]*(k[b]-k[a])
    # Use identical pairs in both models; the gate separates differential skips
    # (about 5.5 ms) from ordinary inlier differences (about 0.05 ms).
    core = np.abs(residual[0]) < 2.5
    rms = 1000*np.sqrt(np.mean(residual[:, core]**2, axis=1))
    receiver_rms = np.array([1000*np.sqrt(np.mean(residual[:, core[:,j], j]**2, axis=1)) for j in range(full.nr)])
    print('survey', n, 'pairs kept', core.sum(), 'held-out differential RMS [anisotropic, isotropic] us', np.round(rms,2), 'worst receiver RMS us', np.round(receiver_rms.max(axis=0),2))
    assert rms[0] < rms[1]
