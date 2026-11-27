import pickle, runpy
from pathlib import Path
import numpy as np
p = Path('/holdout_task')
m = runpy.run_path(str(p/'beta_m97_c01_batch_01_solver.py'))
for case in pickle.loads((p/'public_beta_m97_c01_batch_01.pkl').read_bytes()):
    st, gt = case['study'], case['gt']
    q = st['sensor_calibration_sweeps']
    banks, x, y = (np.asarray(q[k]) for k in ('bank_id','ref_signal','raw_readout'))
    err = []
    for b in range(4):
        idx = np.flatnonzero(banks == b)
        err.append(max(abs(y[i] - np.dot([x[i], 1], m['_fit_line'](x[idx[idx!=i]], y[idx[idx!=i]]))) for i in idx))
    print('Leave-one-point-out raw calibration max errors by bank:', err)
    assert max(err) < 1e-5
    z = st['Z_nuisance']; lam = st['nuisance_ridge']
    def project(a):
        c = np.linalg.solve(z.T@z + lam*np.eye(z.shape[1]), z.T@a)
        r = a-z@c
        assert np.max(abs(z.T@r-lam*c)) < 1e-10
        return r
    S,F = m['_forward'](st)
    S_open,_ = m['_forward'](dict(st, recirculation_loop_mode=1))
    e_sealed = np.max(abs(project(S)-gt['S_tilde']))
    e_open = np.max(abs(project(S_open)-gt['S_tilde']))
    e_target = np.max(abs(project(m['_calibrated_target'](st))-gt['r_y']))
    print(f'Projected design max errors: sealed={e_sealed:.3g}, open control={e_open:.3g}; projected target={e_target:.3g}')
    assert e_sealed < 1e-12 and e_open > 1e-4 and e_target < 1e-6
    for axis in ('x','y'):
        k = st['kappa_'+axis]; h = st['L'+axis]/(len(k)-1)
        for robin in (0.0, st['robin_'+axis]):
            a = m['_axis_operator'](k, st['vel_'+axis], st['L'+axis], robin).toarray()
            end = [0,len(k)-1]; inward = [1,len(k)-2]
            np.testing.assert_allclose(a[end,end], k[end]/h**2 + robin/h, atol=1e-12, rtol=0)
            np.testing.assert_allclose(a[end,inward], -k[end]/h**2, atol=1e-12, rtol=0)
            np.testing.assert_allclose(a[end].sum(axis=1), robin/h, atol=1e-12, rtol=0)
    solve = m['solve_beta_m97_c01_batch_01']
    out = solve(st)
    control = solve(dict(st,aux_manifold_telemetry=-10*st['aux_manifold_telemetry']))
    for k in ('weights','terminal_field'):
        np.testing.assert_array_equal(out[k],control[k])
    ew = np.max(abs(out['weights']-gt['w_opt'])); ef = np.max(abs(out['terminal_field']-gt['terminal_field_A']))
    print(f'Public max absolute errors: weights={ew:.3g}, field={ef:.3g}; boundary identities and telemetry invariance passed')
    assert ew < 1e-5 and ef < 1e-6
