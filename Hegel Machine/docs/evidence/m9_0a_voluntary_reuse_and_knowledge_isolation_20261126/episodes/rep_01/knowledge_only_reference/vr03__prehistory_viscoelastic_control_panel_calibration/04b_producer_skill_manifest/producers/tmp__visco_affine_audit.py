exec(open('/tmp/visco_audit.py').read().split('for i,d in enumerate(data):')[0])
from copy import deepcopy

for i, entry in enumerate(data):
    co, gt = entry['cohort'], entry['gt']
    base = m.calibrate_viscoelastic_cohort(co)
    changed = deepcopy(co)
    taus = np.asarray(co['taus'])
    for name in ('train', 'query'):
        block = changed[name]
        for c in np.unique(block['center_id']):
            gain = (0.50, 1.8, 0.75, 2.2, 0.65, 1.6)[c]
            offset = (-5.0, 4.0, 2.0, -3.0, 5.0, -4.0)[c]
            rows = block['center_id'] == c
            block['controls'][rows] = gain * block['controls'][rows] + offset
            for j in np.flatnonzero(rows):
                trace = block['traces'][j]
                design = m._stress_design(trace, taus)
                trace['stress_obs'] = gain * np.asarray(trace['stress_obs']) + offset * design[:, 1:].sum(axis=1)
    out = m.calibrate_viscoelastic_cohort(changed)
    for key in base:
        delta = np.max(np.abs(out[key] - base[key]))
        print(f'cohort {i}, {key}: max affine-transformation change {delta:.3g}')
        np.testing.assert_allclose(out[key], base[key], atol=1e-7, rtol=1e-7)
