import itertools
import pickle
import sys

import numpy as np

sys.path.insert(0, '/holdout_task')
import upstream_transducer as m

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)

pairs = list(itertools.combinations(range(4), 2))

def designs(x):
    d = m._design(x)
    contrasts = np.column_stack([x[:, i] + s*x[:, j] for i, j in pairs for s in (-1, 1)])
    # Linear and quadratic pair contrasts already lie in d. Test the
    # non-redundant cubic contributions and smooth rational corrections.
    return {
        'deployed': d,
        'pair-cubic': np.column_stack((d, contrasts**3)),
        'rational': np.column_stack((d, x**5/(1 + x**2), contrasts**3/(1 + contrasts**2))),
    }

for item in items:
    st = item['study']
    sweeps = st['sensor_calibration_sweeps']
    x = sweeps['ref_rosette']
    y = sweeps['raw_rosette']
    norms = np.linalg.norm(y, axis=1)
    sorted_norms = np.sort(norms)
    split = np.argmax(np.diff(sorted_norms))
    cutoff = (sorted_norms[split] + sorted_norms[split + 1])/2
    clean = norms < cutoff
    print('EMI row indices:', np.flatnonzero(~clean), 'norm separation:', sorted_norms[split:split+2])
    amp = np.max(abs(x), axis=1)
    axis = np.count_nonzero(x, axis=1) == 1
    groups = {'all-axis': axis, 'upper-amplitude': amp >= 1.6, 'low-amplitude': amp <= 0.8}
    d = designs(x)
    worst = 0.
    for name, group in groups.items():
        tr, te = ~group, group & clean
        errs = {}
        for label, a in d.items():
            coef = m._fit(x[tr], y[tr]) if label == 'deployed' else np.linalg.lstsq(a[tr & clean], y[tr & clean], rcond=None)[0]
            errs[label] = float(np.max(abs(a[te] @ coef - y[te])))
            if label == 'deployed':
                inverse_error = float(np.max(abs(m._invert(y[te], coef) - x[te])))
                worst = max(worst, inverse_error)
        print(name, 'n', int(te.sum()), 'worst forward errors', errs, 'deployed inverse error', inverse_error)
    print('worst grouped inverse error', worst)
    print('public items', len(items), 'item keys', sorted(item), 'relevant study keys', [k for k in st if any(t in k for t in ('cohort', 'label', 'technical_batch', 'relay', 'meter', 'tracer'))])
    assert worst < 1e-4, 'Grouped holdout exceeds the public calibration verification target'
