from pathlib import Path
import json
import numpy as np
root = Path('/tmp/hegel_continual_archive')
source = (root/'alpha_w03_batch_06_solver.py').read_text()
records = json.loads((root/'archive_calibration_history.json').read_text())
# Keep the historical full sweeps, not the sparse subsequent spot checks.
sweeps = []
for record in records:
    lab, x, y = (np.asarray(record[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    if any(np.sum(lab == i) < 8 for i in range(4)):
        continue
    sweeps.append([np.polyfit(x[lab == i], y[lab == i], 1).tolist() for i in range(4)])
start = source.index('# Two recurring')
end = source.index('def _checks', start)
source = source[:start] + '# Affine fits from the archived full load-cell sweeps.\n_ARCHIVE_LINES = np.array(' + repr(sweeps) + ')\n\n\n' + source[end:]
start = source.index('def _archive_calibration')
end = source.index('def _prep', start)
source = source[:start] + '''def _calibrate(checks):
    lines = np.empty((4, 2))
    for lab in range(4):
        x, y = _checks(checks, lab)
        local = _supported_line(x, y)
        if local is not None:
            lines[lab] = local
            continue
        prior = _ARCHIVE_LINES[:, lab]
        hit = (np.abs(y[:, None] - x[:, None]*prior[:, 0] - prior[:, 1]) <= .002).sum(axis=0)
        if hit.max() > 0:
            # Average repeated sweeps of the configuration supported by the spot checks.
            plausible = prior[hit == hit.max()]
            lines[lab] = np.median(plausible, axis=0)
        else:
            lines[lab] = prior[-1]
    return lines


''' + source[end:]
source = source.replace('solve_alpha_w03_batch_06', 'solve_alpha_w03_batch_07')
Path('/holdout_task/alpha_w03_batch_07_solver.py').write_text(source)
