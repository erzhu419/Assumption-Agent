import importlib.util
import pickle

import numpy as np
from scipy.special import expit

spec = importlib.util.spec_from_file_location('assay_ic50', '/holdout_task/assay_ic50.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_screen.pkl', 'rb') as f:
    screen = pickle.load(f)

keys = ('log10_ic50_uM', 'hill_slope')
limits = np.array([0.09, 0.18])
worst = [(-1, None), (-1, None)]
failures = 0
for seed in range(40):
    rng = np.random.default_rng(seed)
    truth = {cid: (rng.uniform(-2.2, 1.75), rng.uniform(0.8, 1.55)) for cid in screen['gt']}
    plates = []
    for p in screen['plates']:
        background = rng.uniform(90, 270)
        amplitude = rng.uniform(1450, 3200)
        length = rng.uniform(1.2, 3.2)
        edge = rng.uniform(0.18, 0.42)
        row_tilt, col_tilt = rng.uniform(-0.07, 0.07, 2)
        wells = []
        for w in p['wells']:
            r, c = w['row'], w['col']
            d = min(r, 15-r, c, 23-c)
            kind = w['well_type']
            if kind == 'COMPOUND':
                midpoint, slope = truth[w['compound_id']]
                value = expit(np.log(10) * slope * (midpoint - w['log10_conc_uM']))
            else:
                value = float(kind == 'DMSO')
            gain = 1 + edge*np.exp(-d/length) + row_tilt*(r-7.5)/7.5 + col_tilt*(c-11.5)/11.5
            readout = background + amplitude*gain*value + rng.normal(0, 0.008*amplitude)
            if rng.random() < 0.03:
                readout -= amplitude*rng.uniform(0.3, 0.8)
            wells.append(dict(w, fluorescence=float(readout)))
        plates.append(dict(p, wells=wells))
    pred = m.estimate_compound_hill_parameters(plates)
    case = np.array([[abs(pred[cid][k] - true) for k, true in zip(keys, values)] for cid, values in truth.items()])
    failures += bool(np.any(case > limits))
    for j in range(2):
        i = case[:,j].argmax()
        if case[i,j] > worst[j][0]:
            worst[j] = (float(case[i,j]), (seed, list(truth)[i]))
print('Synthetic worst midpoint and slope (error, (seed, compound)):', worst)
print('Four-plate replicates exceeding either tolerance:', failures, '/ 40')
