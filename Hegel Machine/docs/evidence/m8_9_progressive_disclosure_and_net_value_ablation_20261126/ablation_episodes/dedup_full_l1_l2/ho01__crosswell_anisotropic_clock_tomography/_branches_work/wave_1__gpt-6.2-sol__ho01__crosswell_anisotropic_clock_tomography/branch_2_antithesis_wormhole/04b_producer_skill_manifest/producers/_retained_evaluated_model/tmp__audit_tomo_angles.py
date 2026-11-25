import contextlib
import io
import runpy
import numpy as np
from scipy.special import erf

with contextlib.redirect_stdout(io.StringIO()):
    audit = runpy.run_path('/tmp/audit_tomo_folds.py')
invert = audit['load_variant']('exact')
for k, row in enumerate(audit['data']):
    s, gt = row['survey'], row['gt']
    w, h, r = s['W_m'], s['H_m'], s['R_plume_m']
    zs = s['z_src_m'][:, None]
    dz = s['z_rcv_m'][None, :] - zs
    length = np.hypot(w, dz)
    y = 1000 * s['T_obs_s']
    folds = np.array_split(np.argsort(np.abs(dz).ravel(), kind='stable'), 5)
    scores = []
    for held in folds:
        mask = np.zeros(dz.size, dtype=bool)
        mask[held] = True
        mask = mask.reshape(dz.shape)
        assert np.all(np.sum(~mask, axis=0) > 0)
        beta, amp, out = invert(dict(s, T_obs_s=np.where(mask, np.nan, s['T_obs_s'])))
        x, z = out['plume_xz_m']
        c = np.array(out['rcv_clock_bias_s'])
        tangent = dz / w
        sin2 = tangent**2 / (1 + tangent**2)
        bg = length * (beta[0] - beta[1] * sin2)
        np.testing.assert_allclose(bg, length * beta[0] - beta[1] * dz**2 / length, atol=1e-12)
        along = (w*x + dz*(z-zs)) / length
        normal = (-dz*x + w*(z-zs)) / length
        ray = np.sqrt(np.pi/2)*r*np.exp(-.5*(normal/r)**2)*(erf(along/(np.sqrt(2)*r))+erf((length-along)/(np.sqrt(2)*r)))
        e = (y - bg - amp*ray - 1000*c)[mask]
        ordinary = abs(e) < 2.5
        scores.append([np.linalg.norm(np.array([x,z])-gt['plume_xz_m']), 1000*np.sqrt(np.mean((c-gt['rcv_clock_bias_s'])**2)), np.sqrt(np.mean(e[ordinary]**2)), int(np.sum(~ordinary))])
    solver = audit['load_variant']('exact')
    _, _, nominal = solver(s)
    _, _, mirror = solver(dict(s, z_src_m=h-s['z_src_m'], z_rcv_m=h-s['z_rcv_m']))
    x, z = nominal['plume_xz_m']
    np.testing.assert_allclose(mirror['plume_xz_m'], [x, h-z], atol=1e-5)
    np.testing.assert_allclose(mirror['rcv_clock_bias_s'], nominal['rcv_clock_bias_s'], atol=1e-9)
    print('survey', k, 'angle bands shallow to steep [center m, clock RMSE ms, ordinary RMS ms, skips]:', np.round(scores, 5).tolist(), flush=True)
    assert np.all(np.asarray(scores)[:, :2] <= [1.8, .25])
