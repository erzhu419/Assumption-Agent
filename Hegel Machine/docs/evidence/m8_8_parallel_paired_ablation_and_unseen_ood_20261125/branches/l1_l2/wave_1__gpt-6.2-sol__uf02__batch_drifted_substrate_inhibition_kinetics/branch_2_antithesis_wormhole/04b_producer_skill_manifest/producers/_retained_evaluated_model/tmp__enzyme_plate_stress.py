import json
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, '/holdout_task')
from enzyme_plate_solver import _fit, _surface

public = json.load(open('/holdout_task/public_enzyme_plates.json'))
keys = ('V_max', 'K_m', 'K_i')
names = tuple(public['ground_truth'])
base = np.array([[public['ground_truth'][v][k] for k in keys] for v in names])

def simulate(seed, n, regime):
    rng = np.random.default_rng(seed)
    truth = base * np.exp(rng.uniform(-0.30, 0.30, size=base.shape))
    if regime == 'weak_curvature':
        truth[:, 1] *= 0.50
        truth[:, 2] *= 2.0
    plates = []
    for i in range(n):
        wells = [dict(w) for w in public['plates'][i % 3]['wells']]
        coords = rng.permutation(96)
        for w, j in zip(wells, coords):
            w.update(row=int(j // 12), col=int(j % 12))
        f = _surface(wells)
        gain = f @ np.r_[rng.uniform(65, 130), rng.uniform(-16, 16, 2), rng.uniform(-10, 10, 3)]
        b = rng.uniform(8, 35)
        rate = np.zeros(len(wells))
        for j, w in enumerate(wells):
            if w['well_type'] == 'REF_WT':
                rate[j] = 1
            elif w['well_type'] == 'VARIANT':
                s = w['substrate_mM']
                v, km, ki = truth[names.index(w['variant_id'])]
                rate[j] = v*s/(km+s+s*s/ki)
        noise = rng.uniform(0.7, 1.5)
        y = b + gain*rate + rng.normal(0, noise, len(wells))
        weights = np.ones(len(wells))
        if regime == 'tail_bubbles':
            weights *= [3 if w['well_type'] == 'VARIANT' and w['substrate_mM'] >= 11 else 1 for w in wells]
        out = rng.choice(len(wells), 4, replace=False, p=weights/weights.sum())
        y[out] += rng.choice([-1, 1], 4)*rng.uniform(0.22, 0.55, 4)*gain[out]
        for w, yy in zip(wells, y):
            w['raw_rate'] = float(yy)
        plates.append({'plate_id': str(i), 'wells': wells})
    return plates, truth

def score(result, truth):
    pred = np.array([[result[v][k] for k in keys] for v in names])
    return np.abs(pred/truth - 1)

if __name__ == '__main__':
    for regime in ('nominal', 'tail_bubbles', 'weak_curvature', 'large_batch'):
        n, seeds = (16, range(5)) if regime == 'large_batch' else (2, range(100))
        values = defaultdict(list)
        for seed in seeds:
            plates, truth = simulate(seed, n, regime)
            for joint in (False, True):
                t0 = time.perf_counter()
                e = score(_fit(plates, joint), truth)
                values[joint].append((e.max(), seed, np.unravel_index(e.argmax(), e.shape), time.perf_counter()-t0))
        for joint, vals in values.items():
            e = np.array([v[0] for v in vals])
            print(regime, 'joint' if joint else 'staged', 'p50/p95/max %', np.round(100*np.quantile(e,[.5,.95,1]),2), 'failures', int(np.sum(e > .12)), 'worst', max(vals)[:3], 'max s', round(max(v[3] for v in vals),3), flush=True)
