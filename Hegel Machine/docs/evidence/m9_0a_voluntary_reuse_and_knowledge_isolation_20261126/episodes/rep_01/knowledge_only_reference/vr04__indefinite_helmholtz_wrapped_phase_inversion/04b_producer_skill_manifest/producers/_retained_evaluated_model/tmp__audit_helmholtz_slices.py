import runpy
import numpy as np
v = runpy.run_path('/tmp/verify_helmholtz.py')
solver, metrics, polar = (v[key] for key in ('solver', 'metrics', 'polar'))
worst = np.zeros(3)
for case_id, case in enumerate(v['CASES']):
    p, gt = case['problem'], case['gt']
    s, q = p['survey'], p['pilot_calibration']
    out = solver.invert_helmholtz_sources_and_phase(p)
    phase = out['global_phase_rad']
    known = q['known_field_re'] + 1j*q['known_field_im']
    # Each calibrated pilot pair must obey the same rotation, including differences.
    ii, jj = np.triu_indices(len(known), 1)
    d_known = known[ii] - known[jj]
    measured = polar(q)
    d_measured = measured[ii] - measured[jj]
    pair_phase = np.angle(np.vdot(d_known, d_measured))
    pair_error = abs(np.angle(np.exp(1j*(pair_phase-gt['global_phase_rad']))))
    assert pair_error <= .020
    fields = solver._unit_source_fields(p)
    ij = s['sensor_ij']
    x = fields[tuple(ij.T)]
    y = polar(s)*np.exp(-1j*phase)
    nx, ny = fields.shape[:2]
    wall_dist = np.min(np.c_[ij, np.array([nx-1, ny-1])-ij], axis=1)
    masks = {'wall band': wall_dist <= 3, 'interior': wall_dist > 3}
    for left, bottom in np.ndindex(2, 2):
        masks[f'quadrant {left}{bottom}'] = ((ij[:,0] < nx/2) == left) & ((ij[:,1] < ny/2) == bottom)
    for name, held in masks.items():
        a = solver._fit_real_sources(x[~held], y[~held], s['tikhonov_alpha'])
        error = metrics(dict(out, source_amplitudes=a, predicted_field_abs=np.abs(fields @ a)), gt)
        r = y[held] - x[held] @ a
        center = np.median(r.real) + 1j*np.median(r.imag)
        worst = np.maximum(worst, error)
        print(f'public {case_id}, withheld {name} ({held.sum()}): errors {error}; held-out median |residual| {np.median(abs(r)):.4g}, |complex componentwise median| {abs(center):.4g}')
    print(f'public {case_id}: pilot pair-difference phase error {pair_error:.6g} rad')
print('worst grouped omission (source, phase, field):', worst)
assert np.all(worst <= v['limits'])
