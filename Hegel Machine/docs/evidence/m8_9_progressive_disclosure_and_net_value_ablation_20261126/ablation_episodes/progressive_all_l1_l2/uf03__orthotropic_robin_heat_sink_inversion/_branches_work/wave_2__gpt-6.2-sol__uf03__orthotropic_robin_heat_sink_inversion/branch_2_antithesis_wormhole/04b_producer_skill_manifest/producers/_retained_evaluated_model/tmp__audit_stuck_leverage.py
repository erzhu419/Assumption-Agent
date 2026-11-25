exec(open('/tmp/stress_thermal.py').read().split('for c in cases:')[0])
worst = np.zeros(4)
for i, c in enumerate(cases):
    m, gt = c['measurement'], c['gt']
    ij, obs = m['sensor_ij'], m['T_obs']
    sigma = m['sigma_m']
    q, temp = forward(m['kappa'], m['gamma_robin'], sigma, gt['source_xy'], gt['amplitude'])
    exact = temp[tuple(ij.T)] + gt['junction_offset']
    r = obs - exact
    dev = np.abs(r - np.median(r))
    good = np.flatnonzero(dev <= 5 * 1.4826 * np.median(dev))
    bad = np.setdiff1d(np.arange(obs.size), good)
    green = s._sensor_green(m['kappa'], m['gamma_robin'], ij)
    x, y = (np.indices(q.shape) + .5) / m['N']
    sx, sy = gt['source_xy']
    j = np.column_stack((green @ (q*(x-sx)/sigma**2).ravel(), green @ (q*(y-sy)/sigma**2).ravel(), green @ q.ravel(), np.ones(obs.size)))
    influence = np.linalg.pinv(j[good])
    targets = {tuple(sorted(good[np.argsort(np.abs(row))[-bad.size:]])) for row in influence}
    # Preserve actual clean readings; move the observed stuck levels to high-leverage sites.
    restored = obs.copy()
    restored[bad] = exact[bad] + np.median(r[good])
    for site in sorted(targets):
        for level in np.unique(obs[bad]):
            shifted = restored.copy()
            shifted[list(site)] = level
            row = check({**m, 'T_obs': shifted}, gt)
            worst = np.maximum(worst, row)
            if np.any(row > np.r_[limits, 15]):
                print('miss', i, site, level, row)
    print('public case', i, 'high-leverage site sets', len(targets), 'empirical stuck levels', len(np.unique(obs[bad])))
print('worst relocated stuck-reading slice', worst)
assert np.all(worst <= np.r_[limits, 15])
