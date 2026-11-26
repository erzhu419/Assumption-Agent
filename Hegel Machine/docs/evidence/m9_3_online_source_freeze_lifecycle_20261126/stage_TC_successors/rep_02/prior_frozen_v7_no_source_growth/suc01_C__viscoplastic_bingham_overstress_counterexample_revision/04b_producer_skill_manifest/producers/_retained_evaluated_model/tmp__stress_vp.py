import importlib.util
import pickle
import numpy as np

path = '/holdout_task/viscoplastic_overstress_hereditary_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl', 'rb'))[0]

def check(d, clean, tau, vp, label):
    out = m.solve_viscoplastic_hereditary_revision(d)
    e = np.abs(out['predicted_strain'] - clean)
    q, y = d['test'], d['yield_stress']
    hold = np.abs(q['stress_hold_0']) > y
    hist = np.any(np.abs(q['past_stress']) > y, axis=1)
    terr = np.max(np.abs(out['estimated_taus'] / tau - 1))
    verr = np.linalg.norm(out['viscoplastic_weights'] - vp) / np.linalg.norm(vp)
    slices = {(a,b): round(float(e[k].mean()), 6) for a in (0,1) for b in (0,1) if (k := ((hold == a) & (hist == b))).any()}
    print(label, 'tau error', round(terr,6), 'weight error', round(verr,6), 'MAE', round(e.mean(),6), 'hold/history MAE', slices, 'late-time MAE', round(e[:,-1].mean(),6), flush=True)
    assert np.isfinite(e).all() and .20 <= out['estimated_taus'][0] < out['estimated_taus'][1] <= 8.
    assert terr <= .10 and verr <= .14 and e.mean() <= .070
    assert e[~(hold|hist)].mean() <= .065 and all(v <= .080 for k,v in slices.items() if k != (0,0))

check(p['dataset'], p['gt']['te_clean'], p['gt']['taus'], p['gt']['vp_theta'], 'public')
for seed, tau, vp in [(19, [.27, 7.2], [.18,.21]), (37, [1.05, 2.6], [.34,.11])]:
    rng = np.random.default_rng(seed)
    src = p['dataset']
    d = dict(src, train=dict(src['train']), test=dict(src['test']))
    x = m._aligned_coupons(d['train'], d['test'])
    dt, t, y = d['past_dt'], d['eval_times'], d['yield_stress']
    for mode, coupon in zip(('train', 'test'), x):
        q = d[mode]
        n = len(coupon)
        c = rng.permutation(np.arange(n) % 4)
        h = q['past_stress'].copy()
        h = np.clip(h, -.96*y, .96*y)
        h[c >= 2, :6] = rng.choice([-1.,1.], ((c>=2).sum(),6)) * rng.uniform(1.05*y, 2*y, ((c>=2).sum(),6))
        s = rng.choice([-1.,1.], n) * rng.uniform(.55*y, .97*y, n)
        s[c%2==1] *= rng.uniform(1.9, 2.05, (c%2==1).sum())
        q['past_stress'], q['stress_hold_0'] = h, s
        remaining = np.cumsum(dt[::-1])[::-1] - dt
        w = -np.expm1(-dt[:,None]/tau) * np.exp(-remaining[:,None]/tau)
        amp = (s[:,None]-h@w) * [0.68,0.54]
        amp[:,0] += vp[1] * (np.maximum(np.abs(h)-y,0)**2@dt)*s
        slope = .08 + coupon @ [.15,-.12,.11,-.09] + vp[0]*np.sign(s)*np.maximum(np.abs(s)-y,0)**2
        clean = q['strain_0'][:,None] + amp@(-np.expm1(-t[:,None]/tau)).T + slope[:,None]*t
        q['instant_rate_obs'] = (amp/np.array(tau)).sum(axis=1) + slope + rng.normal(0,.010,n)
        if mode == 'train':
            mask = rng.random(clean.shape) < .08
            q['strain_curves_obs'] = clean + rng.normal(0,.023,clean.shape) + mask*np.clip(.35*rng.standard_cauchy(clean.shape),-.95,.95)
    check(d, clean, np.array(tau), np.array(vp), f'stress {seed}')
