exec(open('/tmp/stress_thermal.py').read().split('for c in cases:')[0])
worst = np.zeros(4)
for i, c in enumerate(cases):
    m, gt = c['measurement'], c['gt']
    assert set(m) == {'N', 'kappa', 'gamma_robin', 'sigma_m', 'sensor_ij', 'T_obs'}
    k, ij, obs = m['kappa'], m['sensor_ij'], m['T_obs']
    xy, amp = np.array(gt['source_xy']), gt['amplitude']
    q, temp = forward(k, m['gamma_robin'], m['sigma_m'], xy, amp)
    green = s._sensor_green(k, m['gamma_robin'], ij)
    pred = green @ q.ravel()
    np.testing.assert_allclose(pred, temp[tuple(ij.T)], atol=1e-12, rtol=1e-10)
    # Pairwise contrasts cancel a common additive junction offset exactly.
    np.testing.assert_allclose(pred[:, None] - pred, (green[:, None] - green) @ q.ravel(), atol=1e-12)
    axis = (np.arange(len(k)) + .5) / len(k)
    x, y = np.meshgrid(axis, axis, indexing='ij')
    j = np.column_stack((green @ (q * (x-xy[0]) / m['sigma_m']**2).ravel(), green @ (q * (y-xy[1]) / m['sigma_m']**2).ravel(), pred, np.ones(len(ij))))
    r = obs - pred - gt['junction_offset']
    dev = abs(r - np.median(r))
    clean = np.flatnonzero(dev <= 5 * 1.4826 * np.median(dev))
    sensitivity = np.linalg.pinv(j[clean])
    centered = j[clean, :3] - j[clean, :3].mean(axis=0)
    print('public case', i, 'clean readings', clean.size, 'offset-free Jacobian singular values', np.linalg.svd(centered, compute_uv=False))
    # Delete the two clean sensors with greatest local influence on each parameter.
    pairs = {tuple(sorted(clean[np.argsort(abs(row))[-2:]])) for row in sensitivity}
    for pair in sorted(pairs):
        keep_mask = ~np.isin(np.arange(len(ij)), pair)
        row = check({**m, 'sensor_ij': ij[keep_mask], 'T_obs': obs[keep_mask]}, gt)
        worst = np.maximum(worst, row)
        print('dropped', pair, 'errors and seconds', row)
    original = s.localize_thermal_source(m)
    reflected = s.localize_thermal_source({**m, 'kappa': k[::-1], 'sensor_ij': np.column_stack((len(k)-1-ij[:, 0], ij[:, 1])), 'T_obs': obs + .125})
    np.testing.assert_allclose([*reflected['source_xy'], reflected['amplitude'], reflected['junction_offset']], [1-original['source_xy'][0], original['source_xy'][1], original['amplitude'], original['junction_offset']+.125], rtol=1e-6, atol=1e-7)
print('worst targeted public two-sensor deletion', worst)
assert np.all(worst <= np.r_[limits, 15])
