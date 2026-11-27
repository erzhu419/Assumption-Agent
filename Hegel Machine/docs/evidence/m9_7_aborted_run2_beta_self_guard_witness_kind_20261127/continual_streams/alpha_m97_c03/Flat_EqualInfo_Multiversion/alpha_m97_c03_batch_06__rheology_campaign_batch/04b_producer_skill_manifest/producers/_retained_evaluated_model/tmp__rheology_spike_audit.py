import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
path = root / 'alpha_m97_c03_batch_06_solver.py'
source = path.read_text()
needle = 'w = base / (scale * np.sqrt(1 + (r / (2.5 * scale))**2))**0.5'
assert source.count(needle) == 1
variants = {'baseline': source, 'strain_spike_gate': source.replace(needle, needle + '\n        w[(groups < len(times)) & (np.abs(r) > 8 * scale)] = 0.0').replace('project(p, w)[0] / w', 'project(p, np.ones_like(base), w)[0]').replace('def project(p, w):','def project(p, w, fit_w=None):').replace('a = matrix * w[:, None]\n        b = y * w', 'fit_w = w if fit_w is None else fit_w\n        a = matrix * fit_w[:, None]\n        b = y * fit_w').replace("return np.einsum('ij,j->i', a, weights) - b, weights", "return (np.einsum('ij,j->i', matrix, weights) - y) * w, weights")}
with (root / 'public_alpha_m97_c03_batch_06.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
for label, src in variants.items():
    ns = {}
    exec(compile(src, str(path), 'exec'), ns)
    solve = ns['solve_alpha_m97_c03_batch_06']
    metrics, taus = [], []
    for lab in np.unique(tr['lab_id']):
        m = tr['lab_id'] == lab
        n = m.sum()
        fold = dict(d, train={k:v[~m] for k,v in tr.items()}, test={k:np.concatenate((v,tr[k][m])) for k,v in te.items()})
        o = solve(fold)
        taus.append(o['estimated_taus'])
        ae = np.abs(o['predicted_strain'][-n:] - tr['strain_curves_obs'][m])
        r = o['initial_rate_pred'][-n:] - tr['instant_rate_obs'][m]
        complete = np.isfinite(tr['coupon_features_raw'][m]).all(axis=1)
        row = [np.nanmax(np.nanmedian(ae[complete], axis=0)), np.nanmean(np.minimum(ae[complete], .08)), np.sqrt(np.nanmean(r[complete]**2)), np.sqrt(np.nanmean(r**2)), np.nanmax(np.sqrt(np.nanmean(ae**2, axis=0))) ]
        metrics.append(row)
        print(label, lab, np.round(row, 6))
    print(label, 'worst', np.round(np.max(metrics, axis=0), 6), 'tau ranges', np.round(np.ptp(taus, axis=0), 6), 'full taus', np.round(solve(d)['estimated_taus'], 6))
