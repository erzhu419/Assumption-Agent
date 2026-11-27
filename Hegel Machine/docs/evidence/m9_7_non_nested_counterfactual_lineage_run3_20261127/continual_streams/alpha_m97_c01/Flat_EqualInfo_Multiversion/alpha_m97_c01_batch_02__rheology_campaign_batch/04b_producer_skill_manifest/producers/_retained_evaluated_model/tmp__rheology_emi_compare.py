import functools
import importlib.util
import itertools
import pickle
import numpy as np

path = '/holdout_task/alpha_m97_c01_batch_02_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_02.pkl', 'rb'))[0]
d, gt = item['dataset'], item['gt']
cal, te = d['calibration_load_cells'], d['test']
m._prior_line = functools.lru_cache(None)(m._prior_line)
source = open(path).read()
needle = 'if keep.sum() < 4 and span < 0.2:'
variants = {'current': m._calibrate}
if needle in source:
    scope = {}
    exec(compile(source.replace(needle, 'if keep.sum() < 4:' ).replace('        span = np.ptp(clean_x)\n', ''), path, 'exec'), scope)
    scope['_prior_line'] = m._prior_line
    variants['compatible-prior'] = scope['_calibrate']
for c in range(4):
    q = cal['lab_id'] == c
    print('channel', c, 'archived minus full in-batch gain/bias', np.array(m._prior_line(c)) - m._line(cal['cmd_stress'][q], cal['true_stress'][q]))
for k in (3, 4, 5):
    for name, fit in variants.items():
        count = bad = 0
        worst = (0., None)
        for c in range(4):
            q = cal['lab_id'] == c
            x, y = cal['cmd_stress'][q], cal['true_stress'][q]
            sel = te['lab_id'] == c
            probe = np.r_[te['stress_hold_cmd'][sel], te['past_stress_cmd'][sel].ravel()]
            ref = np.r_[gt['te_hold_true'][sel], gt['te_hist_true'][sel].ravel()]
            for ix in itertools.combinations(range(len(x)), k):
                z, w = x[list(ix)], y[list(ix)]
                for j in range(k):
                    for spike in (-4., 4.):
                        v = w.copy(); v[j] += spike
                        a, b = fit(z, v, c)
                        error = np.max(abs(a*probe + b - ref))
                        count += 1; bad += error > .01
                        if error > worst[0]:
                            keep = np.arange(k) != j
                            worst = (error, (c, ix, j, spike, np.ptp(z[keep]), np.ptp(w[keep] - m._prior_line(c)[0]*z[keep])))
        print(k, name, 'above .01', bad, '/', count, 'worst (error; channel, rows, spike row, size, clean span, prior contrast range)', worst)
