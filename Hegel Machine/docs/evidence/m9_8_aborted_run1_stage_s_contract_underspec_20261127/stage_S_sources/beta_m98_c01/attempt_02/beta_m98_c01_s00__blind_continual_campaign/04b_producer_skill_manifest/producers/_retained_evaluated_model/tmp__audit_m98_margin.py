import importlib.util
import pickle
import numpy as np

p = '/holdout_task/batch_solver.py'
spec = importlib.util.spec_from_file_location('candidate', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']
x, raw = c['ref_rosette'], c['raw_rosette']
y = raw - x
rms = lambda a: np.sqrt(np.mean(a*a))
p2, p3 = m._monomials(2), m._monomials(3)
phi2, phi3 = m._features(x, p2), m._features(x, p3)
models = {'affine': phi2[:, :5], 'quadratic': phi2, 'cubic': phi3}
for j in range(4):
    for k in range(j+1, 4):
        for sign in (-1, 1):
            q = (x[:, j] + sign*x[:, k])/np.sqrt(2)
            for a in (.25, 1.):
                models[f'rational_{j}_{k}_{sign}_{a}'] = phi2/(1+a*q[:, None]**2)

def error(a, tr, te):
    b = np.linalg.lstsq(a[tr], y[tr], rcond=None)[0]
    return a[te] @ b - y[te]

ix = np.arange(len(x))
for family in ('affine', 'polynomial', 'rational'):
    options = [name for name in models if (family == 'rational' and name.startswith('rational')) or (family == 'polynomial' and not name.startswith('rational')) or name == family]
    chosen, scores, banks = [], [], []
    for outer in range(4):
        test = ix%4 == outer
        train = ~test
        inner = np.cumsum(train)%3
        def score(name):
            a = models[name]
            folds = [(train & (inner!=k), train & (inner==k)) for k in range(3)]
            if any(tr.sum() <= a.shape[1]+4 for tr, _ in folds):
                return np.inf
            return np.mean([rms(error(a, tr, te))**2 for tr, te in folds])
        name = min(options, key=score)
        chosen.append(name)
        e = error(models[name], train, test)
        scores.append(rms(e)); banks.append(max(rms(z) for z in e.T))
    print('Nested calibration', family, 'worst rosette/bank RMSE', max(scores), max(banks), 'chosen', chosen)

forward, nominal, sources = m._chamber(s)
pulses = s['reference_tracer_pulses']
invert = m._calibration(c)
for ax in ('x','y'):
    a = m._axis(s['kappa_'+ax], s['vel_'+ax], s['L'+ax]/(s['n'+ax]-1))
    defect = np.max(np.abs(np.asarray(a.sum(axis=1))))
    print('Interior no-flux constant-mode defect', ax, defect)
    assert defect < 1e-12
banks = np.asarray(s['row_bank_id'])
worst = {'nominal': 0., 'fitted': 0.}
for i, train in enumerate(pulses):
    test = pulses[1-i]
    robin = m._pulse_fit([train], invert, forward, nominal, sources)
    for name, r in [('nominal', nominal), ('fitted', robin)]:
        pred = forward(r, sources @ test['w_pulse'])[0]
        e = pred - invert(test['y_pulse_raw'])
        e, ids = e.reshape(len(s['mu_shifts']), -1), banks.reshape(len(s['mu_shifts']), -1)
        v = max(rms(row[bid == k]) for row, bid in zip(e, ids) for k in np.unique(bid))
        worst[name] = max(worst[name], v)
print('Worst held-out pulse x shift x channel bank RMSE', worst)
assert worst['fitted'] < 5e-5 and worst['fitted'] < worst['nominal']
