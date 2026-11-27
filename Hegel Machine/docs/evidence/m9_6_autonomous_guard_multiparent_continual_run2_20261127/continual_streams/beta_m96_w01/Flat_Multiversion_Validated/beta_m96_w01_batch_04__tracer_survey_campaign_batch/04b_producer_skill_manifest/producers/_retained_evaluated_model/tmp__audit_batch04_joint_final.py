import importlib.util
import itertools
import pickle
from pathlib import Path
import numpy as np

path = Path('/holdout_task/beta_m96_w01_batch_04_solver.py')
spec = importlib.util.spec_from_file_location('solver', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_beta_m96_w01_batch_04.pkl', 'rb') as f:
    case = pickle.load(f)[0]
s, gt = case['study'], case['gt']
q = s['sensor_calibration_sweeps']; ids = np.asarray(q['bank_id'])
choices = []
for bank in range(4):
    idx = np.flatnonzero(ids == bank)
    x, raw = m._samples(q, bank)
    gain, offset = m._current_line(x, raw)
    residual = raw - (gain*x + offset)
    pairs = list(itertools.combinations(range(len(idx)), 2))
    scores = np.array([residual[list(p)].mean() for p in pairs])
    variants = [(idx, None), (idx[:0], None), (idx[:1], None), (idx[list(pairs[scores.argmin()])], None), (idx[list(pairs[scores.argmax()])], None), (idx[:2], 0), (idx[:2], 1)]
    sweeps = []
    for keep, spike in variants:
        qq = {k: np.asarray(v)[keep].copy() for k,v in q.items()}
        if spike is not None:
            qq['raw_readout'][spike] += 0.22
        sweeps.append(qq)
    choices.append(sweeps)

# Only calibration changes across slices, so reuse the exact transport result.
forward = m._forward(s)
S, y = gt['S_tilde'], gt['r_y']
ridge, cap, budget = (s[k] for k in ('source_ridge','u_box','total_budget'))
m._forward = lambda study: forward
module = vars(m)
worst = np.zeros(4); witness = None
for labels in itertools.product(range(len(choices[0])), repeat=4):
    sweep = {k: np.concatenate([choices[b][i][k] for b,i in enumerate(labels)]) for k in q}
    out = module['solve_beta_m96_w01_batch_04'](dict(s, sensor_calibration_sweeps=sweep))
    w, cert = out['weights'], out['kkt_certificate']
    r = S @ w - y
    j = .5 * (r @ r + ridge * (w @ w))
    g = S.T @ r + ridge * w
    nu = cert['nu_sum']
    lb = j - g @ w + nu * budget + cap * np.minimum(g-nu, 0).sum()
    metrics = np.array([np.linalg.norm(out['terminal_field']-gt['terminal_field_A']) / np.linalg.norm(gt['terminal_field_A']), j-min(lb, cert['dual_lower_bound']), np.max(np.abs(g-nu-cert['lam_lower']+cert['lam_upper'])), abs(j-cert['primal_objective'])])
    if metrics[1] > worst[1]: witness = labels
    worst = np.maximum(worst, metrics)
print('saved solver: 2401 joint slices, max [relative field, gap, stationarity, objective error]', worst, 'gap witness', witness, flush=True)
assert np.all(worst < [1e-5, 1e-6, 1e-4, 1e-6])
