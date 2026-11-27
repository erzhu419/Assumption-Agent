import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
d, c = item['dataset'], item['campaign_diagnostics']
t, dt = d['eval_times'], d['past_dt']
material = m.solve_batch(d)
th = material['hereditary_weights']
cal = d['calibration_load_cells']
P, H = m._loads(c, m._bridge(cal))
r = th[2] + m._chamber_features(c) @ th[3:]
lab = c['lab_id']
s = cal['single_channel_sweeps']
gain = np.array([np.polyfit(s['true_stress'][s['lab_id'] == l], s['cmd_stress'][s['lab_id'] == l], 1)[0] for l in lab])
later = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
tau = material['estimated_taus']
Z = P @ (-np.expm1(-dt[:, None] / tau) * np.exp(-later[:, None] / tau))
V = np.c_[c['strain_curves_obs'] - c['strain_0'][:, None], 2 * c['instant_rate_obs']]
names = ['reset', 'pairwise_gap', 'memory', 'quadratic', 'rational', 'sweep_nuisance']

def predict(name, mask):
    mu = r[mask].mean()
    hg = H[mask].mean()
    gg = gain[mask].mean()
    def prof(q):
        F = np.c_[H, r]
        if name == 'pairwise_gap':
            F = np.c_[H - Z[:, 0], r]
        elif name == 'memory':
            F = np.c_[F, Z[:, 0] - Z[:, 1], Z[:, 1]]
        elif name == 'quadratic':
            F = np.c_[F, (r-mu)**2, (H-hg)*(r-mu)]
        elif name == 'rational':
            F = np.c_[H, r / (1 + q[2]*(r-mu)**2)]
        elif name == 'sweep_nuisance':
            F = np.c_[F, gain-gg, (gain-gg)*(r-mu)]
        p = np.exp(q[:2]); T = np.vstack([np.c_[-np.expm1(-t[:, None]/p), t], 2*np.r_[1/p, 1.]])
        Q = np.linalg.lstsq(F[mask], V[mask], rcond=None)[0]
        return F @ np.linalg.lstsq(T, Q.T, rcond=None)[0].T @ T.T
    q = np.log([.47, 2.45]); lower = np.log([.2, 1.]); upper = np.log([.75, 10.])
    if name == 'rational':
        q = np.r_[q, .01]; lower = np.r_[lower, 0.]; upper = np.r_[upper, 10.]
    opt = least_squares(lambda q: (prof(q)[mask] - V[mask]).ravel(), q, bounds=(lower, upper))
    return prof(opt.x)

def score(e):
    return np.array([np.mean(e[:, :-1]**2)**.5, np.mean(e[:, -1]**2)**.5/2])

res = {}; groups = np.unique(lab)
for name in names:
    oo = np.empty_like(V)
    for l in groups:
        mask = lab != l
        oo[~mask] = predict(name, mask)[~mask] - V[~mask]
    res[name] = oo
    print(name, 'OOF', score(oo).round(6), 'worst lab', np.max([score(oo[lab==l]) for l in groups], 0).round(6))
for l in groups:
    e = res['reset'][lab==l]
    print('reset mean bias lab', l, np.r_[e[:, :-1].mean(), e[:, -1].mean()/2].round(6))
print('Continuous strain/rate regression; no classification labels or relay modes in data.')
