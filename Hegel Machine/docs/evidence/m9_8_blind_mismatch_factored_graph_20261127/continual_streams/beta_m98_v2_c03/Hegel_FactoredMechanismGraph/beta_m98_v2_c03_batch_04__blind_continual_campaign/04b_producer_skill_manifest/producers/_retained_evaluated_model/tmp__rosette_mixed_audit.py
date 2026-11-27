import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as m

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
st = item['study']
s = st['sensor_calibration_sweeps']
x, y = s['ref_rosette'], s['raw_rosette']
norms = np.linalg.norm(y, axis=1)
a = np.sort(norms)
j = np.argmax(np.diff(a))
clean = norms < (a[j] + a[j+1]) / 2
mixed = np.count_nonzero(x, axis=1) > 1
pairs = list(zip(*m._PAIRS))
worst = 0.
for i in range(4):
    k = (i + 1) % 4
    p = x[:, i] * x[:, k]
    for sign in (-1, 1):
        hold = mixed & (sign*p >= 0.5)
        if not np.any(hold & clean):
            continue
        train, test = ~hold, hold & clean
        c = m._fit(x[train], y[train])
        e = float(np.max(abs(m._invert(y[test], c) - x[test])))
        worst = max(worst, e)
        print('interaction slice', (i,k,sign), 'n', int(test.sum()), 'inverse max', e)

# On the actual observations, isolate any calibration bias caused by retained EMI.
c_all = m._fit(x, y)
c_clean = np.linalg.lstsq(m._design(x[clean]), y[clean], rcond=None)[0]
raw = st['y_obs_raw']
a = m._invert(raw, c_all)
b = m._invert(raw, c_clean)
shift = float(np.max(abs(a-b)))
print('worst signed-interaction inverse error', worst, 'observation shift vs clean-only fit', shift)
print('calibration scope: public studies=1; study label/batch fields:', [k for k in st if 'label' in k or 'batch' in k]); print('campaign diagnostics present:', item.get('campaign_diagnostics') is not None)
assert 0 < worst < 1e-4
assert shift < 1e-5
