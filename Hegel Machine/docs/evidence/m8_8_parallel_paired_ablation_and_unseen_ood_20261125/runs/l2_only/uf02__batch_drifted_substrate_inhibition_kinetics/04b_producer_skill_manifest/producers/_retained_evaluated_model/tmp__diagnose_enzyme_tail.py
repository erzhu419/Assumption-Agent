import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '/holdout_task')
import enzyme_plate_solver as base

j = json.load(open('/holdout_task/public_enzyme_plates.json'))
plates, truth = j['plates'], j['ground_truth']
keys = ('V_max', 'K_m', 'K_i')
methods = [base.estimate_variant_haldane_parameters]

def errors(fit, expected):
    return np.array([[abs(fit[v][k]/expected[v][k]-1) for k in keys] for v in expected])

for label, subset in [('full', plates)] + [(f'without {p["plate_id"]}', [q for q in plates if q is not p]) for p in plates]:
    print(label, [(np.round(e.max(axis=0),4).tolist(), round(float(e.max()),4)) for e in (errors(m(subset), truth) for m in methods)])

rng = np.random.default_rng(int(sys.argv[1]))
start = time.perf_counter()
for scenario, count, bubble_rate, noise, hetero in [('nominal',3,.04,1.1,0.), ('two plates',2,.04,1.1,0.), ('more bubbles',3,.08,1.1,0.), ('signal dependent',3,.04,1.1,.008), ('noisier, signal dependent',3,.04,2.,.012)]:
    err = []
    for rep in range(96):
        selected = rng.choice(len(plates), count, replace=False)
        data = [copy.deepcopy(plates[i]) for i in selected]
        expected = {v:dict(zip(keys, np.array([truth[v][k] for k in keys])*np.exp(rng.uniform(-.5,.5,3)))) for v in truth}
        clean = []
        for p in data:
            clean_plate = {**p, "wells": []}
            clean.append(clean_plate)
            b = rng.uniform(8,35)
            gain = np.r_[rng.uniform(75,130),rng.uniform(-14,14,5)]
            coords = rng.permutation([(w['row'],w['col']) for w in p['wells']])
            for w,(r,c) in zip(p['wells'],coords):
                w['row'], w['col'] = int(r),int(c)
                typ = w['well_type']
                rate = 0. if typ == 'BLANK' else 1. if typ == 'REF_WT' else base._rate(w['substrate_mM'], np.log(list(expected[w['variant_id']].values())))
                signal = np.dot(base._basis(r,c),gain)*rate
                raw = b + signal + rng.normal(scale=np.hypot(noise, hetero*signal))
                clean_plate['wells'].append({**w, 'raw_rate': raw})
                if rng.random() < bubble_rate:
                    raw += rng.choice([-1,1])*rng.uniform(12,65)
                w['raw_rate'] = raw
        e = errors(methods[0](data),expected)
        err.append([e])
        if e.max() > .12:
            oracle = errors(methods[0](clean),expected)
            index = np.unravel_index(e.argmax(),e.shape)
            print('failure', scenario, rep, list(expected)[index[0]],keys[index[1]],round(e.max(),4),'without bubbles',round(oracle[index],4),'clean worst',round(oracle.max(),4))
    a = np.array(err)
    for name,e in zip(('final',), a.transpose(1,0,2,3)):
        w = e.max(axis=(1,2))
        print(scenario,name,'p95/max',np.round([np.quantile(w,.95),w.max()],4),'failures',sum(w>.12),'max by parameter',np.round(e.max(axis=(0,1)),4))
print('stress runtime',round(time.perf_counter()-start,2))
