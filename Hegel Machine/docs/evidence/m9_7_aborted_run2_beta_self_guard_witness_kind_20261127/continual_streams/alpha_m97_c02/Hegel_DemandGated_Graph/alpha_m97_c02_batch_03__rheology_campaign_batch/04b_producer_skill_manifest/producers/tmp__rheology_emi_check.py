import importlib.util
import itertools
import pickle
from unittest.mock import patch
import numpy as np
path = '/holdout_task/'
s = importlib.util.spec_from_file_location('solver', path+'alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open(path+'public_alpha_m97_c02_batch_03.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']; base = m._calibration(d)
for n in (2, 3, 4):
    count = 0
    for lab in range(4):
        ids = np.flatnonzero(c['lab_id']==lab)
        for rows in itertools.combinations(ids,n):
            keep = c['lab_id']!=lab; keep[list(rows)] = True
            q = {k:v[keep] for k,v in c.items()}
            with patch.object(m,'_HISTORY','/archive_not_needed_for_clean_calibration'):
                got = m._calibration(dict(d, calibration_load_cells=q))[lab]
            np.testing.assert_allclose(got, m._line(c['cmd_stress'][list(rows)],c['true_stress'][list(rows)]),rtol=0,atol=1e-12)
            count += 1
    print('Clean',n,'row sweeps:',count,'direct fits without archive access.')
for lab in range(4):
    ids = np.flatnonzero(c['lab_id']==lab)
    for n in (0, 1):
        keep = c['lab_id']!=lab; keep[ids[:n]] = True
        q = {k:v[keep] for k,v in c.items()}
        got = m._calibration(dict(d,calibration_load_cells=q))[lab]
        np.testing.assert_allclose(got, base[lab], rtol=0, atol=0.001)
failed = False
for n, amplitude in [(3,3.01),(3,3.05),(3,6.),(4,3.01),(4,3.05),(4,6.),(5,3.05),(10,3.05)]:
    worst = 0.; count = failures = 0; witness = None
    for lab in range(4):
        ids = np.flatnonzero(c['lab_id']==lab)
        for rows in itertools.combinations(ids,n):
            keep = c['lab_id']!=lab; keep[list(rows)] = True
            q = {k:v[keep] for k,v in c.items()}
            for j, i in enumerate(np.flatnonzero(q['lab_id']==lab)):
                for sign in (-1,1):
                    y = q['true_stress'].copy(); y[i] += sign*amplitude
                    got = m._calibration(dict(d, calibration_load_cells=dict(q,true_stress=y)))[lab]
                    stress_error = np.max(np.abs(np.polyval(got-base[lab],c['cmd_stress'][ids])))
                    count += 1; failures += int(stress_error > .01)
                    if stress_error > worst:
                        worst = stress_error; witness = (lab,rows,j,sign)
    print('rows',n,'EMI',amplitude,'failures',failures,'/',count,'worst stress error',worst,'witness',witness)
    failed |= bool(failures)
reference = importlib.util.spec_from_file_location('reference','/tmp/.hegel_candidate_archive/v02_71c2e126/alpha_m97_c02_batch_03_solver.py')
old = importlib.util.module_from_spec(reference); reference.loader.exec_module(old)
actual = m.solve_alpha_m97_c02_batch_03(d)
for k,v in old.solve_alpha_m97_c02_batch_03(d).items():
    np.testing.assert_array_equal(actual[k],v)
print('All five public outputs unchanged from the validated baseline.')
assert not failed
