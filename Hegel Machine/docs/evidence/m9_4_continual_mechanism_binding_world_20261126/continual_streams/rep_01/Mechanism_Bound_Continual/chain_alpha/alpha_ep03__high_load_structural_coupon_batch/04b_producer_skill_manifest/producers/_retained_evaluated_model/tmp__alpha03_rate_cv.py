import importlib.util, pickle, numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_ep03_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_alpha_ep03.pkl', 'rb') as f:
    ds = pickle.load(f)[0]['dataset']
a, _, y, complete = m._prepare(ds)
complete = complete[:len(y)]
lab = ds['train']['lab_id']
err = {k: np.empty_like(y[:, :-1]) for k in ('raw', 'current', 'robust', 'robust_more', 'full_missing', 'clip_missing')}
for c in np.unique(lab):
    val = lab == c
    logs, theta = m._fit(a, y, complete & ~val)
    p = a.matrix(logs) @ theta
    d = y[:, -1] - p[:, -1]
    scale = m._scale((y-p)[complete & ~val])[-1]
    print('fold', c, 'onset rate MAD', round(scale, 6))
    for mode, adj in {
        'raw': np.zeros(len(d)),
        'current': np.where(complete, .25, 1.) * d,
        'robust': np.where(complete, .25 * np.clip(d, -3*scale, 3*scale), d),
        'robust_more': np.where(complete, .1 * np.clip(d, -3*scale, 3*scale), d),
        'full_missing': np.where(complete, 0., d),
        'clip_missing': np.where(complete, .25*np.clip(d, -3*scale, 3*scale), np.clip(d, -.45, .45)),
    }.items():
        err[mode][val] = (p[:, :-1] - y[:, :-1] + adj[:, None] * a.t)[val]
for mode, e in err.items():
    print(mode)
    for name, mask in [('all', lab >= 0), ('complete', complete), ('missing', ~complete), ('subyield', (a.work == 0) & (a.ov == 0)), ('positive', a.ov > 0), ('negative', a.ov < 0)] + [(f'lab{c}', lab == c) for c in np.unique(lab)]:
        r = e[mask]
        print(name, np.round([np.sqrt(np.mean(r*r)), np.median(abs(r)), np.quantile(abs(r), .90), np.sqrt(np.mean(r[:, -1]**2))], 6))
