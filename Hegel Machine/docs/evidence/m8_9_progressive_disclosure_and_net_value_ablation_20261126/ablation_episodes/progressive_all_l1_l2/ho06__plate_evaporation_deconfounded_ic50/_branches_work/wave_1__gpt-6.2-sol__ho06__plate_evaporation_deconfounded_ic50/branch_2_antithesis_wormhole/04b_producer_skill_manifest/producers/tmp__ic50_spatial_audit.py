import pickle
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

with open('/holdout_task/public_screen.pkl', 'rb') as f:
    data = pickle.load(f)
plates, gt = data['plates'], data['gt']
keys = sorted(gt)
fields = ('log10_ic50_uM', 'hill_slope')
truth = np.array([[gt[k][f] for f in fields] for k in keys])
source = Path('/holdout_task/assay_ic50.py').read_text()
rational = source.replace('np.exp(-edge / decay)', '1 / (1 + (edge / decay)**2)').replace('amplitude * attenuation * edge / decay**2', '2 * amplitude * attenuation**2 * edge**2 / decay**3')
assert rational != source and 'attenuation**2 * edge**2' in rational

def depth(w):
    r, c = w['row'], w['col']
    return min(r, 15-r, c, 23-c)

def errors(result):
    return np.abs(np.array([[result[k][f] for f in fields] for k in keys]) - truth).max(axis=0)

for label, text in [('exponential', source), ('rational', rational)]:
    ns = {}
    exec(compile(text, '/holdout_task/assay_ic50.py', 'exec'), ns)
    state = {}
    def capture(*args, **kwargs):
        result = least_squares(*args, **kwargs)
        state['fit'] = result
        return result
    ns['least_squares'] = capture
    estimate = ns['estimate_compound_hill_parameters']
    full = errors(estimate(plates))
    lopo = np.max([errors(estimate(plates[:i]+plates[i+1:])) for i in range(len(plates))], axis=0)
    held_errors, held_bias = [], []
    for i, plate in enumerate(plates):
        controls = [w for w in plate['wells'] if w['well_type'] == 'DMSO']
        # Hold out one spatial block of DMSO at a time, retaining its opposite anchor.
        for border in (True, False):
            keep = [w for w in controls if (depth(w) == 0) != border]
            omit = [w for w in controls if (depth(w) == 0) == border]
            if len(keep) < 3 or not omit:
                continue
            subset = plates.copy()
            subset[i] = dict(plate, wells=[w for w in plate['wells'] if w['well_type'] != 'DMSO'] + keep)
            held_errors.append(errors(estimate(subset)))
            t = state['fit'].x
            n = len(plates)
            amplitude, decay, tx, tz = t[2*n+2*len(keys)+4*i:2*n+2*len(keys)+4*i+4]
            bg = np.median([w['fluorescence'] for w in plate['wells'] if w['well_type'] == 'POS_KILL'])
            scale = np.median([w['fluorescence'] for w in keep]) - bg
            residual = []
            for w in omit:
                q = depth(w) / decay
                a = np.exp(-q) if label == 'exponential' else 1/(1+q*q)
                g = 1 + amplitude*a + tx*(w['col']-11.5)/11.5 + tz*(w['row']-7.5)/7.5
                residual.append(t[i]+t[n+i]*g-(w['fluorescence']-bg)/scale)
            held_bias.append((plate['plate_id'], 'border' if border else 'interior', float(np.median(residual))))
    worst = max(held_bias, key=lambda a: abs(a[2]))
    print(label, 'full', full.round(5), 'LOPO', lopo.round(5), 'DMSO-block holdout parameter errors', np.max(held_errors, axis=0).round(5), 'worst held-out control block median residual', worst)
