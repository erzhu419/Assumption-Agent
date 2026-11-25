import importlib.util
import pickle
from collections import defaultdict
import numpy as np

spec = importlib.util.spec_from_file_location('assay_ic50', '/holdout_task/assay_ic50.py')
assay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assay)
with open('/holdout_task/public_screen.pkl', 'rb') as f:
    data = pickle.load(f)
keys = sorted(data['gt'])
fields = ('log10_ic50_uM', 'hill_slope')
truth = np.array([[data['gt'][k][f] for f in fields] for k in keys])
limits = np.array([0.09, 0.18])

def position(w):
    r, c = w['row'], w['col']
    return min(r, 15-r, c, 23-c), r, c

worst = np.zeros(2)
for side in (0, -1):
    plates = []
    separated = 0
    for p in data['plates']:
        retained = []
        replicates = defaultdict(list)
        for w in p['wells']:
            if w['well_type'] == 'COMPOUND':
                replicates[w['compound_id'], w['log10_conc_uM']].append(w)
            else:
                retained.append(w)
        for wells in replicates.values():
            ordered = sorted(wells, key=position)
            separated += position(ordered[0])[0] != position(ordered[-1])[0]
            retained.append(ordered[side])
        plates.append(dict(p, wells=retained))
    assert separated > 0, 'Replicates do not vary in perimeter distance'
    fit = assay.estimate_compound_hill_parameters(plates)
    assert set(fit) == set(keys)
    estimate = np.array([[fit[k][f] for f in fields] for k in keys])
    err = np.abs(estimate-truth)
    worst = np.maximum(worst, err.max(axis=0))
    print('nearer' if side == 0 else 'farther', 'groups at different depths', separated, 'max errors', err.max(axis=0).round(5), 'compounds', [keys[i] for i in err.argmax(axis=0)])
assert np.all(worst <= limits), worst
