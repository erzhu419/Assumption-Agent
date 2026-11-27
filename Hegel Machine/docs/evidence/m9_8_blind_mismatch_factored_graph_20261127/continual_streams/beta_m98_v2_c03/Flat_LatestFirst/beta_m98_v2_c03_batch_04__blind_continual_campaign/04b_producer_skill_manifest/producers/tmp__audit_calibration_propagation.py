import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
fit_full = m._fit_calibration

def fit_compact(ref, raw):
    design = m._features(ref)
    seed = fit_full(ref, raw)
    inlier = np.linalg.norm(raw - design @ seed, axis=1) < .002
    out = np.zeros_like(seed)
    for j in range(raw.shape[1]):
        active = abs(seed[:, j]) > .001
        out[active, j] = np.linalg.lstsq(design[inlier][:, active], raw[inlier, j], rcond=None)[0]
    return out

def relative(a, b):
    return np.linalg.norm(a-b) / np.linalg.norm(b)

for item in pickle.load(open('/holdout_task/public_batch.pkl', 'rb')):
    s = item['study']; sw = s['sensor_calibration_sweeps']; x = sw['ref_rosette']
    cv = np.array_split(np.random.default_rng(13).permutation(len(x)), 10)
    results = {}
    for name, fit in [('full', fit_full), ('compact', fit_compact)]:
        m._fit_calibration = fit
        base = m.solve_batch(s)
        diffs = []
        for te in cv:
            keep = np.ones(len(x), dtype=bool); keep[te] = False
            sub = dict(s, sensor_calibration_sweeps=dict(sw, ref_rosette=x[keep], raw_rosette=sw['raw_rosette'][keep]))
            out = m.solve_batch(sub)
            diffs.append([relative(out[k], base[k]) for k in ('weights', 'terminal_field')])
        results[name] = base
        print(name, 'worst calibration-fold relative variation (weights, field):', np.max(diffs, axis=0))
    print('full-to-compact relative change (weights, field):', [relative(results['compact'][k], results['full'][k]) for k in ('weights', 'terminal_field')])
    m._fit_calibration = fit_full
