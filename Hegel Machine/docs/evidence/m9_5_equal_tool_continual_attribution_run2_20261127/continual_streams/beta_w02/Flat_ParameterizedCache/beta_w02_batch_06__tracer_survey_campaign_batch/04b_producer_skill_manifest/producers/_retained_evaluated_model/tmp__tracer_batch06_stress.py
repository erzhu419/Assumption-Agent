import importlib.util, json, pickle
from pathlib import Path
import numpy as np
p = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', p/'beta_w02_batch_06_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
v = {}
exec((p/'validate_public_schema.py').read_text().split('\n_d = ')[0], v)
with (p/'public_beta_w02_batch_06.pkl').open('rb') as f:
    case = pickle.load(f)[0]
s, gt = case['study'], case['gt']
archive = json.loads(Path('/tmp/hegel_continual_archive/archive_calibration_history.json').read_text())
sweeps = [('public', s['sensor_calibration_sweeps'])]
for e in archive:
    tag = e['episode_short_name'][-2:]
    if tag not in ('03', '04', '05'):
        continue
    b, x, y = (np.asarray(e[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    def add(label, mask, yy):
        sweeps.append((tag + '_' + label, dict(bank_id=b[mask], ref_signal=x[mask], raw_readout=yy[mask])))
    add('full', b >= 0, y)
    for j in np.unique(b):
        add(f'omit{j}', b != j, y)
        yy = y.copy()
        ids = np.flatnonzero(b == j)
        yy[ids[len(ids)//2]] += 2.0
        add(f'emi{j}', b >= 0, yy)
worst = np.zeros(3)
for label, sweep in sweeps:
    out = m.solve_beta_w02_batch_06({**s, 'sensor_calibration_sweeps': sweep})
    cert = v['verify_box_simplex_kkt_certificate'](gt['S_tilde'], gt['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'])
    err = (cert['gap'], np.max(np.abs(out['weights']-gt['w_opt'])), np.linalg.norm(out['terminal_field']-gt['terminal_field_A'])/np.linalg.norm(gt['terminal_field_A']))
    print(label, cert['reason'], *(f'{x:.3g}' for x in err))
    assert cert['valid'] and err[1] < 1e-5 and err[2] < 1e-5
    worst = np.maximum(worst, err)
print('worst gap, weight max error, field relative error:', worst, 'slices:', len(sweeps))
# A coherent change in both affine coefficients must still be taken from the sweep.
e = next(e for e in archive if e['episode_short_name'].endswith('_04'))
b, x = (np.asarray(e[k]) for k in ('bank_id','ref_signal'))
rb = s['row_bank_id']
truth = m._calibrate(s)
coef = m._ARCHIVED_CALIBRATION + np.array([[.01,-.02],[-.02,.01],[.03,.02],[-.01,-.03]])
sweep = dict(bank_id=b,ref_signal=x,raw_readout=coef[b,0]*x+coef[b,1])
actual = m._calibrate({**s,'sensor_calibration_sweeps':sweep,'y_obs_raw':truth*coef[rb,0]+coef[rb,1]})
print('coherent drift inversion max error:', np.max(np.abs(actual-truth)))
assert np.max(np.abs(actual-truth)) < 1e-12
