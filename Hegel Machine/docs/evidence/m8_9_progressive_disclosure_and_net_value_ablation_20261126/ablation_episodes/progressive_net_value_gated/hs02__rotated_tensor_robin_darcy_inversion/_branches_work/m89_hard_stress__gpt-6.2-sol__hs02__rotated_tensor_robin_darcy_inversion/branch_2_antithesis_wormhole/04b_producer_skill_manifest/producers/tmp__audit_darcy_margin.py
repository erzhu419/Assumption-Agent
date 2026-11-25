from pathlib import Path
import numpy as np

# Reuse the independent flux balance and isolated import from the first stress test.
exec(Path('/tmp/check_tensor_darcy.py').read_text().split('worst_balance = 0.', 1)[0])
base = cases[0]['problem']
for number, case in enumerate(cases):
    p, g = case['problem'], case['gt']
    ij = p['sensor_ij']
    clean = abs(p['y_obs'] - g['u_true'][tuple(ij.T)] - g['sensor_offset']) < .75
    flat = np.ravel_multi_index(ij.T, g['u_true'].shape)
    a = mod._forward(p)[flat, 1:]
    centered = a[clean] - a[clean].mean(axis=0)
    leverage = np.sum(np.linalg.qr(centered)[0]**2, axis=1)
    clean_ids = np.flatnonzero(clean)
    # Move the observed spikes, including their signs, onto the leverage extremes;
    # retain all measured inlier residuals and the true field.
    spikes = (p['y_obs'] - g['u_true'][tuple(ij.T)] - g['sensor_offset'])[~clean]
    ranked = clean_ids[np.argsort(leverage)[-len(spikes):]]
    y0 = p['y_obs'].copy()
    y0[~clean] -= spikes
    worst = np.zeros(3)
    for k in range(len(spikes)):
        y = y0.copy()
        y[ranked] += np.roll(spikes, k)
        worst = np.maximum(worst, errors(dict(p, y_obs=y), g['source_weights'], g['sensor_offset'], g['u_true']))
    print('relocated measured spikes; public', number, 'worst source/offset/field', worst)

# Use a genuinely non-square array, not merely unequal physical side lengths.
p = dict(base, Lx=1.85, Ly=.65, gamma_robin=.16,
         **{key:base[key][:17, :22] for key in ('k_xx','k_xy','k_yy','sigma')},
         basis=base['basis'][:, :17, :22])
p['sensor_ij'] = base['sensor_ij'][base['sensor_ij'][:, 0] < 17]
nx, ny = p['k_xx'].shape
u = rng.normal(size=(nx, ny))
A,b,_ = mod._assemble(p)
mismatch = np.max(abs((A @ u.ravel()-b).reshape(u.shape)-direct_balance(p,u)))
assert mismatch < 1e-12
print('17x22 weak-Robin pointwise stencil mismatch', mismatch)

# Construct the source perturbation that maximizes the field-mean change at
# fixed centered sensor RMS. The offset cancels its mean exactly at sensors.
for label, p in [('public', base), ('weak rectangular', p)]:
    r = mod._forward(p)
    shape = p['k_xx'].shape
    ij = tuple(p['sensor_ij'].T)
    x = r[np.ravel_multi_index(ij, shape), 1:]
    n = len(x)
    d = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(d, full_matrices=False)
    c = r[:, 1:].mean(axis=0)
    dt = vt.T @ ((vt @ c) / s**2)
    dt *= .001 * np.sqrt(n) / np.linalg.norm(d @ dt)
    offset_change = -float(x.mean(axis=0) @ dt)
    field_change = r[:, 1:] @ dt
    sensor_change = x @ dt + offset_change
    print(label, 'indistinguishability: sensor RMS', np.linalg.norm(sensor_change)/np.sqrt(n),
          'field mean change', field_change.mean(), 'offset change', offset_change,
          'source perturbation norm', np.linalg.norm(dt))
