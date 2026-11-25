import importlib.util
import pickle
import time
import numpy as np

spec = importlib.util.spec_from_file_location('tomography_solver', '/holdout_task/tomography_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_surveys.pkl', 'rb') as f:
    data = pickle.load(f)

def check(s, gt):
    start = time.perf_counter()
    got = m.invert_plume_and_clocks(s)
    clock = np.asarray(got['rcv_clock_bias_s'])
    assert abs(clock.sum()) < 1e-12 and np.max(abs(clock)) <= .004 + 1e-14
    return (np.linalg.norm(np.asarray(got['plume_xz_m']) - gt['plume_xz_m']),
            1000 * np.sqrt(np.mean((clock - gt['rcv_clock_bias_s'])**2)),
            time.perf_counter() - start)

for k, row in enumerate(data):
    s, gt = row['survey'], row['gt']
    print(k, 'full (m, ms, s)', check(s, gt), flush=True)
    slices = []
    for i in range(len(s['z_src_m'])):
        leave = dict(s, T_obs_s=s['T_obs_s'].copy())
        leave['T_obs_s'][i] = np.nan
        slices.append(check(leave, gt))
    v = np.asarray(slices)
    print(k, 'leave-one-source-out max (m, ms, s)', v.max(axis=0), 'worst rows', v.argmax(axis=0), flush=True)
