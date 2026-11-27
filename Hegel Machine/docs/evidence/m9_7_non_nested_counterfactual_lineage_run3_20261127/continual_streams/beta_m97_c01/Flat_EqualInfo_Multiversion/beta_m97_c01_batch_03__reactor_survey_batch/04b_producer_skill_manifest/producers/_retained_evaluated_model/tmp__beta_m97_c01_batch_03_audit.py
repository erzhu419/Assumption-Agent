from pathlib import Path
import importlib.util
import pickle
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'beta_m97_c01_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ns = {}
exec((root / 'validate_public_schema.py').read_text().split('\n_d = ')[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
worst_stencil = worst_sum = worst_gap = worst_w = worst_field = 0.0
for case in pickle.loads((root / 'public_beta_m97_c01_batch_03.pkl').read_bytes()):
    s, gt = case['study'], case['gt']
    # Exercise both sides of the loop threshold on the measured grid.
    old_spec = importlib.util.spec_from_file_location('previous', '/tmp/hegel_continual_archive/beta_m97_c01_batch_02_solver.py')
    old = importlib.util.module_from_spec(old_spec)
    old_spec.loader.exec_module(old)
    for loop in (0.0, 0.5):
        trial = dict(s, recirculation_loop_mode=loop)
        for actual, expected in zip(m._forward(trial), old._forward(trial)):
            np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)
    for axis in 'xy':
        k = np.asarray(s['kappa_' + axis]); v = np.asarray(s['vel_' + axis]); length = s['L' + axis]
        n = len(k); h = length / (n - 1)
        A = m._axis_operator(length, k, v, s['robin_' + axis], periodic=True)
        j = np.arange(n)
        f = np.cos(2*np.pi*j/n) + 0.2*np.sin(4*np.pi*j/n)
        minus = f - np.roll(f, 1); plus = f - np.roll(f, -1)
        expected = ((k + np.roll(k, 1))/(2*h*h) + np.maximum(v, 0)/h)*minus + ((k + np.roll(k, -1))/(2*h*h) + np.maximum(-v, 0)/h)*plus
        worst_stencil = max(worst_stencil, float(np.max(abs(A @ f - expected))))
        worst_sum = max(worst_sum, float(np.max(abs(A @ np.ones(n)))))
    # Keep the measured observations fixed and spike one real sweep point in every bank at once.
    sweep = s['sensor_calibration_sweeps']
    bank_ids = np.asarray(sweep['bank_id']); bank_rows = [np.flatnonzero(bank_ids == b) for b in range(4)]
    original_archive = m._archived_affine
    def unexpected_archive(bank):
        raise AssertionError(f'Bank {bank} has enough clean in-batch calibration points')
    m._archived_affine = unexpected_archive
    try:
        for j in range(min(map(len, bank_rows))):
            swept = {key: np.array(value, copy=True) for key, value in sweep.items()}
            for b, indices in enumerate(bank_rows):
                swept['raw_readout'][indices[j]] += 8.0 * (-1)**(b+j)
            out = m.solve_beta_m97_c01_batch_03(dict(s, sensor_calibration_sweeps=swept))
            result = verify(gt['S_tilde'], gt['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'])
            assert result['valid'], (j, result)
            worst_gap = max(worst_gap, result['gap'])
            worst_w = max(worst_w, float(np.max(abs(out['weights'] - gt['w_opt']))))
            worst_field = max(worst_field, float(np.max(abs(out['terminal_field'] - gt['terminal_field_A']))))
    finally:
        m._archived_affine = original_archive
assert max(worst_stencil, worst_sum) < 1e-11
assert worst_w < 1e-6 and worst_field < 1e-7
print('Nonperiodic forward branches match the archive at loop modes 0 and 0.5.')
print('Periodic action and constant-null residuals:', worst_stencil, worst_sum)
print('Simultaneous four-bank EMI worst gap, weights, and field errors:', worst_gap, worst_w, worst_field)
