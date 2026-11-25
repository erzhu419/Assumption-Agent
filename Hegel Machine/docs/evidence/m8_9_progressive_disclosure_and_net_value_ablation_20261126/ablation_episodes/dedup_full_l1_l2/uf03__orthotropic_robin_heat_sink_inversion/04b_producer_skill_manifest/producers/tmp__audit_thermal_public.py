import runpy
import numpy as np

ns = runpy.run_path('/tmp/check_thermal.py')
solve = ns['mod'].localize_thermal_source
for i, case in enumerate(ns['cases']):
    m = case['measurement']
    ans = solve(m)
    n = m['N']
    cell = (np.arange(n) + .5)/n
    s2 = m['sigma_m']**2
    green = ns['mod']._sensor_green(m['kappa'], m['gamma_robin'], m['sensor_ij'])
    def terms(xy, green=green):
        dx, dy = cell - np.asarray(xy)[:, None]
        q = np.exp(-.5*(dx[:, None]**2 + dy[None, :]**2)/s2)
        return np.einsum('mij,kij->mk', green, np.stack((q, q*dx[:, None]/s2, q*dy[None, :]/s2)))
    v, vx, vy = terms(ans['source_xy']).T
    gain, offset = ans['amplitude'], ans['junction_offset']
    y = m['T_obs']
    residual = y - (gain*v + offset)
    mad = 1.4826*np.median(abs(residual-np.median(residual)))
    good = abs(residual-np.median(residual)) <= 4.5*mad
    # Centering spans the same contrast space as all pairwise differences.
    V = np.column_stack((gain*vx, gain*vy, gain*v))[good]
    V -= V.mean(axis=0)
    y0, v0 = y[good]-y[good].mean(), v[good]-v[good].mean()
    j,k = np.triu_indices(len(y0),1)
    dv,dy = v0[j]-v0[k], y0[j]-y0[k]
    contrast_gain = np.dot(dv,dy)/np.dot(dv,dv)
    assert np.isclose(contrast_gain,gain,rtol=1e-5), (i,contrast_gain,gain)
    sv = np.linalg.svd(V,compute_uv=False)
    assert sv[-1] > 0
    shift = .1
    shifted = solve({**m,'T_obs':y+shift})
    np.testing.assert_allclose(shifted['source_xy'],ans['source_xy'],rtol=0,atol=1e-7)
    np.testing.assert_allclose(shifted['amplitude'],gain,rtol=1e-7)
    np.testing.assert_allclose(shifted['junction_offset'],offset+shift,rtol=0,atol=1e-8)
    # Assess each held-out reading against a fit that never saw it.
    deleted = []
    for j in range(len(y)):
        keep = np.arange(len(y)) != j
        fit = solve({**m,'T_obs':y[keep],'sensor_ij':m['sensor_ij'][keep]})
        pred = fit['amplitude']*terms(fit['source_xy'],green[j:j+1])[0,0] + fit['junction_offset']
        deleted.append(float(y[j]-pred))
    deleted = np.abs(deleted)
    clean_max, gross_min = max(deleted[good]), min(deleted[~good])
    print(f'public {i}: centered Jacobian sigma_min={sv[-1]:.6g}, pairwise amplitude relative difference={abs(contrast_gain/gain-1):.3g}, deleted clean max={clean_max:.6g}, deleted gross min={gross_min:.6g}')
    assert clean_max <= 5*mad and gross_min >= 10*clean_max
