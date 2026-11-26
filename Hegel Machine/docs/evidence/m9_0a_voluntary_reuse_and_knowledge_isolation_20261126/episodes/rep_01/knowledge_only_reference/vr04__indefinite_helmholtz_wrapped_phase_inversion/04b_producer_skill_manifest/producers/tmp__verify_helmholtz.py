import importlib.util
import time
import numpy as np
from pathlib import Path

# Import the exact deliverable without relying on the working directory.
path = Path('/holdout_task/helmholtz_phase_inversion.py')
spec = importlib.util.spec_from_file_location('helmholtz_phase_inversion', path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
exec(Path('/tmp/helmholtz_public_probe.py').read_text().split("if __name__ == '__main__':")[0])


def metrics(out, gt):
    a, t, f = (out[key] for key in ('source_amplitudes', 'global_phase_rad', 'predicted_field_abs'))
    assert set(out) == set(gt)
    assert a.shape == gt['source_amplitudes'].shape and a.dtype.kind == 'f' and np.isfinite(a).all()
    assert isinstance(t, float) and -np.pi < t <= np.pi
    assert f.shape == gt['predicted_field_abs'].shape and f.dtype.kind == 'f' and np.isfinite(f).all() and (f >= 0).all()
    return np.array([np.linalg.norm(a - gt['source_amplitudes']) / np.linalg.norm(gt['source_amplitudes']),
                     abs(np.angle(np.exp(1j * (t - gt['global_phase_rad'])))),
                     np.linalg.norm(f - gt['predicted_field_abs']) / np.linalg.norm(gt['predicted_field_abs'])])


limits = np.array([.030, .020, .030])
for number, case in enumerate(CASES):
    p, gt = case['problem'], case['gt']
    start = time.perf_counter()
    out = solver.invert_helmholtz_sources_and_phase(p)
    elapsed = time.perf_counter() - start
    error = metrics(out, gt)
    transfer = solver._unit_source_fields(p)
    ref = reference_transfer(p).reshape(transfer.shape)
    transfer_error = np.linalg.norm(transfer - ref) / np.linalg.norm(ref)
    survey, pilot = p['survey'], p['pilot_calibration']
    phase = out['global_phase_rad']
    fields = transfer[tuple(survey['sensor_ij'].T)]
    y = polar(survey) * np.exp(-1j * phase)
    all_fields = []
    held_out = []
    for j in range(len(y)):
        take = np.arange(len(y)) != j
        a = solver._fit_real_sources(fields[take], y[take], survey['tikhonov_alpha'])
        all_fields.append(metrics(dict(out, source_amplitudes=a, predicted_field_abs=np.abs(transfer @ a)), gt))
        held_out.append(abs(fields[j] @ a - y[j]))
    pilot_errors = []
    for j in range(len(pilot['known_field_re'])):
        take = np.arange(len(pilot['known_field_re'])) != j
        reduced = {k: val[take] for k, val in pilot.items()}
        pilot_errors.append(metrics(solver.invert_helmholtz_sources_and_phase(dict(p, pilot_calibration=reduced)), gt))
    survey_worst = np.max(all_fields, axis=0)
    pilot_worst = np.max(pilot_errors, axis=0)
    print(f'public {number}: source/phase/field {error}; seconds {elapsed:.4f}; transfer relative error {transfer_error:.3g}')
    print(f'  survey leave-one-out worst {survey_worst}; held-out residual median/p90/max {np.quantile(held_out,[.5,.9,1])}')
    print(f'  pilot leave-one-out worst {pilot_worst}')
    assert transfer_error < 1e-12
    assert np.all(np.maximum.reduce([error, survey_worst, pilot_worst]) <= limits)
