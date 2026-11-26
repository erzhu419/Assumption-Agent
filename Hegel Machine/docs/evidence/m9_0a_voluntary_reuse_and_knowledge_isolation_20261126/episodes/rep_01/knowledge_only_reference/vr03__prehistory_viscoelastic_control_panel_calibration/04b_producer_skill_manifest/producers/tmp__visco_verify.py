exec(open('/tmp/visco_audit.py').read().split('for i,d in enumerate(data):')[0])
from copy import deepcopy
for i, d in enumerate(data):
    co, gt = d['cohort'], d['gt']
    result = m.calibrate_viscoelastic_cohort(co)
    assert set(result) == {'aligned_query_moduli', 'query_probabilities', 'modulus_weights'}
    z, p, beta = (result[k] for k in ('aligned_query_moduli','query_probabilities','modulus_weights'))
    n = len(co['query']['traces'])
    assert z.shape == (n, 3) and p.shape == (n,) and beta.shape == (3,)
    assert all(a.dtype.kind == 'f' and np.isfinite(a).all() for a in (z,p,beta))
    assert np.all((p > 0) & (p < 1)) and np.all(np.sign(beta) == np.sign(gt['beta']))
    y, c = gt['y_query'], co['query']['center_id']
    for j in np.unique(c):
        h = c == j
        mae = np.abs(z[h]-gt['Z_query'][h]).mean()
        auc = roc_auc_score(y[h],p[h]); brier = np.mean((y[h]-p[h])**2)
        print(f'public {i} center {j}: MAE {mae:.4f}, AUROC {auc:.4f}, Brier {brier:.4f}')
        assert mae <= .22 and auc >= .88 and brier <= .16
    # Relabeling center IDs or shifting their corresponding clocks changes no prediction.
    if i == 0:
        changed = deepcopy(co)
        for split in ('train','query'):
            block = changed[split]
            block['center_id'] = 11 - block['center_id']
            for tr in block['traces']:
                tr['timestamps'] = np.asarray(tr['timestamps']) + 37.25
                tr['past_timestamps'] = np.asarray(tr['past_timestamps']) + 37.25
        invariant = m.calibrate_viscoelastic_cohort(changed)
        for key in result:
            np.testing.assert_allclose(invariant[key],result[key],rtol=1e-7,atol=1e-8)
rng = np.random.default_rng(1901)
subsets = [rng.choice(36, 8, replace=False) for _ in range(36)]
errors=[]
for i in range(120):
    n = 3 + i % 12
    t = np.cumsum(rng.uniform(.035, .41, n+36))
    u = rng.normal(size=t.size) + 2 * np.sin(t*.95)
    taus = np.array([rng.uniform(.27,.8), rng.uniform(1.65,3.7)])
    mem = np.empty((len(t), 2)); mem[0] = u[0]
    for k in range(1,len(t)):
        a = np.exp(-(t[k]-t[k-1])/taus)
        mem[k] = a*mem[k-1] + (1-a)*u[k]
    design = np.column_stack((np.ones(36),u[n:],mem[n:]))
    truth = rng.normal(size=4)*2 + rng.choice([-4,4])
    y = design @ truth + rng.normal(0,.012,36)
    idx = rng.choice(36,3+i%4,replace=False)
    y[idx] += rng.uniform(4.5,9,len(idx))*rng.choice([-1,1],len(idx))
    tr = {'past_timestamps': t[:n], 'past_strain_history': u[:n], 'timestamps': t[n:], 'strain_u': u[n:]}
    inferred = m._stress_design(tr,taus)
    np.testing.assert_allclose(inferred,design,rtol=1e-13,atol=1e-13)
    errors.append(np.max(np.abs(m._robust_moduli(inferred,y,subsets)-truth[1:])))
print(f'120 variable-history glitch stresses: worst coefficient absolute error {max(errors):.4f}')
assert max(errors) < .12
