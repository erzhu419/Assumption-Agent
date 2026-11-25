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
for seed in (6, 7):
    rng = np.random.default_rng(seed)
    truth = {cid: (rng.uniform(-2.2, 1.75), rng.uniform(0.8, 1.55)) for cid in screen['gt']}
    plates = []
    clean_controls = []
    for p in screen['plates']:
        background = rng.uniform(90, 270)
        amplitude = rng.uniform(1450, 3200)
        length = rng.uniform(1.2, 3.2)
        edge = rng.uniform(0.18, 0.42)
        row_tilt, col_tilt = rng.uniform(-0.07, 0.07, 2)
        wells = []
        clean_wells = []
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
            expected = background + amplitude*gain*value
            if rng.random() < 0.03:
                readout -= amplitude*rng.uniform(0.3, 0.8)
            wells.append(dict(w, fluorescence=float(readout)))
            clean_wells.append(dict(w, fluorescence=float(expected if kind != "COMPOUND" else readout)))
        plates.append(dict(p, wells=wells))
        clean_controls.append(dict(p, wells=clean_wells))
    original = m._fit_gain
    def refit(r, c, y):
        t, b = original(r, c, y)
        residual = y - m._spatial_basis(r, c, t) @ b
        keep = residual > -0.07 * np.median(y)
        return original(r[keep], c[keep], y[keep]) if not keep.all() else (t, b)
    for label, data, fit in [('baseline', plates, original), ('clean controls', clean_controls, original), ('refit', plates, refit)]:
        m._fit_gain = fit
        pred = m.estimate_compound_hill_parameters(data)
        err = np.array([[abs(pred[cid][k] - true) for k, true in zip(keys, values)] for cid, values in truth.items()])
        cid = 'CMP_04' if seed == 6 else 'CMP_08'
        print(seed, label, 'worst', err.max(axis=0).round(5), cid, 'truth', np.round(truth[cid], 5), 'pred', pred[cid])
    m._fit_gain = original
