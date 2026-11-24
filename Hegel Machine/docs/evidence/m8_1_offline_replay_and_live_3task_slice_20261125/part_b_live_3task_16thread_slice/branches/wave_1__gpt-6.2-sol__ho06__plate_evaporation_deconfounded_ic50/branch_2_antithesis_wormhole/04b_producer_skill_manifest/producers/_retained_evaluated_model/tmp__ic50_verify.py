import pickle
from itertools import combinations
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from assay_ic50 import estimate_compound_hill_parameters

with open('/holdout_task/public_screen.pkl', 'rb') as f:
    public = pickle.load(f)
truth = public['gt']
fields = ('log10_ic50_uM', 'hill_slope')

def errors(plates, gt):
    out = estimate_compound_hill_parameters(plates)
    assert out.keys() == gt.keys()
    assert all(v.keys() == set(fields) and all(type(x) is float and np.isfinite(x) for x in v.values()) for v in out.values())
    e = np.array([[abs(out[k][m]-g[m]) for m in fields] for k,g in gt.items()])
    return e.max(axis=0), np.array(list(gt))[e.argmax(axis=0)]

for size in (4, 3, 1):
    checks = [(tuple(p['plate_id'] for p in ps), *errors(ps, truth)) for ps in combinations(public['plates'], size)]
    print('plates per fit', size, [(checks[i][0], checks[i][2][j], round(checks[i][1][j], 5)) for j, i in enumerate(np.array([x[1] for x in checks]).argmax(axis=0))])
