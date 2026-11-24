import pickle
import sys
from itertools import product
import numpy as np
from scipy.special import expit
sys.path.insert(0, '/holdout_task')
from assay_ic50 import estimate_compound_hill_parameters

with open('/holdout_task/public_screen.pkl', 'rb') as f:
    layout = pickle.load(f)['plates']
ids = sorted({w['compound_id'] for p in layout for w in p['wells'] if w['well_type'] == 'COMPOUND'})
index = {name: i for i, name in enumerate(ids)}
fields = ('log10_ic50_uM', 'hill_slope')

for interaction, broad in product((False, True), repeat=2):
    errors, details = [], []
    for seed in range(32, 64):
        rng = np.random.default_rng(seed)
        ic = np.linspace(-2.25, 1.15, len(ids)) + rng.uniform(-0.14, 0.14, len(ids))
        slopes = rng.uniform(0.77, 1.92, len(ids))
        simulated = []
        for p in layout:
            scale, background = rng.uniform(1700, 3000), rng.uniform(100, 280)
            decay = rng.uniform(0.95, 3.5) if broad else rng.uniform(1.7, 2.2)
            ar, ac = rng.uniform(-0.09, 0.09, 2)
            amplitude = rng.uniform(0.28, 0.42)
            wells = []
            layout_rng = np.random.default_rng(100000 + 10 * seed + len(simulated))
            compounds = [w for w in p['wells'] if w['well_type'] == 'COMPOUND']
            used = {(w['row'], w['col']) for w in compounds}
            free = [(r,c) for r in range(16) for c in range(24) if (r,c) not in used]
            edge = [x for x in free if min(x[0],15-x[0],x[1],23-x[1]) == 0]
            inner = [x for x in free if x not in edge]
            controls = []
            for points, count in ((edge, 7), (inner, 11)):
                selected = layout_rng.choice(len(points), size=2*count, replace=False)
                for j,k in enumerate(selected):
                    r,c = points[k]
                    controls.append(dict(row=r,col=c,well_type='DMSO' if j < count else 'POS_KILL',compound_id=None,log10_conc_uM=None))
            for w in compounds + controls:
                r, c, kind = w['row'], w['col'], w['well_type']
                distance = min(r, 15-r, c, 23-c)
                perimeter = 1 + amplitude * np.exp(-distance/decay)
                tilt = ar*(r/15-.5) + ac*(c/23-.5)
                gain = perimeter*(1+tilt) if interaction else perimeter+tilt
                if kind == 'COMPOUND':
                    j = index[w['compound_id']]
                    v = expit(-np.log(10)*slopes[j]*(w['log10_conc_uM']-ic[j]))
                else:
                    v = float(kind == 'DMSO')
                y = background + scale*(gain*v+rng.normal(0, .006))
                if rng.random() < .03:
                    y = background + rng.normal(0, 15) if rng.random() < .65 else y + scale*rng.uniform(-.65, .65)
                wells.append(dict(w, fluorescence=y))
            simulated.append(dict(p, wells=wells))
        est = estimate_compound_hill_parameters(simulated)
        truth = np.column_stack((ic, slopes))
        e = abs(np.array([[est[n][m] for m in fields] for n in ids])-truth)
        errors.append(e.max(axis=0))
        details.append([(ids[j], round(ic[j], 3), round(slopes[j], 3)) for j in e.argmax(axis=0)])
    errors = np.array(errors)
    print('interaction', interaction, 'broad decay', broad, 'p95', np.round(np.percentile(errors, 95, axis=0), 5), 'worst', np.round(errors.max(axis=0), 5), 'failures', (errors > [.09, .18]).sum(axis=0), 'worst seed/compound', [(int(i), details[i][j]) for j,i in enumerate(errors.argmax(axis=0))], flush=True)
