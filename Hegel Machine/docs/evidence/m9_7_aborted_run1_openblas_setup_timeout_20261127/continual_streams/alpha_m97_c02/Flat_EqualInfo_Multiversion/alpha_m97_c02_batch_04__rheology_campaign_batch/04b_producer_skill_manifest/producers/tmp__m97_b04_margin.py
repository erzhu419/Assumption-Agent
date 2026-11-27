exec(open('/tmp/m97_b04_cv.py').read().split('v, w =')[0])
import json
from itertools import product
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f:
    history = json.load(f)[-3:]
calibrate = m._calibrate
base_cal = calibrate(d['calibration_load_cells'])
base = m.solve_alpha_m97_c02_batch_04(d)
base_p = m._prepare(d)[0]
base_act = m._activation(d, base_p, dt) > 0
channels = np.flatnonzero(np.bincount(d['calibration_load_cells']['lab_id'], minlength=4) < 2)
choices = []
for channel in channels:
    fits = []
    for sweep in history:
        q = np.asarray(sweep['lab_id']) == channel
        fits.append(np.polyfit(np.asarray(sweep['cmd_stress'])[q], np.asarray(sweep['true_stress'])[q], 1))
    choices.append(fits)
    print('archive-dependent channel', channel, 'gain/bias range', np.ptp(fits, axis=0))
max_delta = np.zeros(2)
flips = 0
try:
    for pick in product(*choices):
        candidate_cal = base_cal.copy()
        candidate_cal[channels] = pick
        m._calibrate = lambda table: candidate_cal
        act = m._activation(d, m._prepare(d)[0], dt) > 0
        flips += np.count_nonzero(act != base_act)
        out = m.solve_alpha_m97_c02_batch_04(d)
        for i,key in enumerate(('predicted_strain', 'initial_rate_pred')):
            max_delta[i] = max(max_delta[i], np.max(np.abs(out[key]-base[key])))
finally:
    m._calibrate = calibrate
print('activation flips across measured archive combinations', flips, 'max absolute strain/rate change', max_delta)
for lab in np.unique(tr['lab_id']):
    q = tr['lab_id']==lab; f = ~q
    lt,b = m._fit(p[f],h[f],x[f],dt,t,a[f],y[f],r[f]); c,g = m._design(lt,p[q],h[q],x[q],dt,t,a[q])
    resolved = ~np.isnan(tr['coupon_features_raw'][q]).any(axis=1)
    for state in (False,True):
        sub = resolved & ((a[q]>0)==state)
        if sub.any():
            print('held lab',lab,'active',state,'complete rows',sub.sum(),'signed median strain/rate residuals',np.round([np.median((c@b-y[q])[sub]),np.median((g@b-r[q])[sub])],6))
