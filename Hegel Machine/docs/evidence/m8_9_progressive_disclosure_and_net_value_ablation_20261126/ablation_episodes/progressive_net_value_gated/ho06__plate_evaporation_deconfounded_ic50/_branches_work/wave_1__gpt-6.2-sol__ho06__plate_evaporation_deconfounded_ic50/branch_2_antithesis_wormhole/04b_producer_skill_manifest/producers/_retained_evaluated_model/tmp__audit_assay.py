from collections import defaultdict
from copy import deepcopy
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from verify_assay import data, module, score

plates = data['plates']
controls = []
for p in plates:
    dm = [w for w in p['wells'] if w['well_type'] == 'DMSO']
    kill = np.array([w['fluorescence'] for w in p['wells'] if w['well_type'] == 'POS_KILL'])
    bg = np.median(kill)
    y = np.array([w['fluorescence'] for w in dm]) - bg
    xy = [(w['row'], w['col']) for w in dm]
    controls.append((np.asarray(xy), module._positions(dm), y, 1.4826*np.median(np.abs(kill-bg))))

def basis(p, scale, form):
    if form == 'constant': return p[:, :1]
    d = p[:, 3]/scale
    e = 1/(1+d*d) if form == 'rational' else np.exp(-d)
    x = np.column_stack((p[:, :3], e))
    return np.column_stack((x, e[:, None]*p[:, 1:3])) if form == 'interaction' else x

for form in ('constant', 'exponential', 'rational', 'interaction'):
    err = defaultdict(list)
    for site in sorted({tuple(rc) for xy, *_ in controls for rc in xy}):
        splits = [(np.any(xy != site, axis=1), p, y, n) for xy,p,y,n in controls]
        def fit(t, report=False):
            cost = 0.
            for keep, pos, y, n in splits:
                if np.all(keep): continue
                # Only training controls determine the scale of each fold.
                scale = np.median(y[keep]); s = max(.005, n/scale)
                x = basis(pos, np.exp(t), form)
                a = x[keep]; b = y[keep]/scale
                coef = least_squares(lambda k: a@k-b, np.linalg.lstsq(a,b,rcond=None)[0],
                                     loss='soft_l1', f_scale=2*s)
                cost += coef.cost/s**2
                if report:
                    for distance, residual in zip(pos[~keep,3], x[~keep]@coef.x-y[~keep]/scale):
                        err[int(distance)].append(float(residual))
            return cost
        t = 0. if form == 'constant' else minimize_scalar(fit, bounds=(np.log(.4),np.log(7)), method='bounded').x
        fit(t, report=True)
    a = np.concatenate(list(err.values()))
    print(form, 'cross-fitted DMSO relative RMS/max', np.round([np.sqrt(np.mean(a*a)), np.max(abs(a))],5),
          'edge-distance signed biases', {k: round(float(np.mean(v)),5) for k,v in sorted(err.items())})

for j in (0,-1):
    split = deepcopy(plates)
    for p in split:
        samples = defaultdict(list)
        retain = []
        for w in p['wells']:
            if w['well_type'] == 'COMPOUND':
                samples[(w['compound_id'], w['log10_conc_uM'])].append(w)
            else:
                retain.append(w)
        def distance(w):
            r,c = w['row'], w['col']
            return min(r,15-r,c,23-c),r,c
        retain.extend(sorted(ws, key=distance)[j] for ws in samples.values())
        p['wells'] = retain
    assert np.all(score(split, 'nearer replicates' if j==0 else 'farther replicates') <= [.09,.18])
