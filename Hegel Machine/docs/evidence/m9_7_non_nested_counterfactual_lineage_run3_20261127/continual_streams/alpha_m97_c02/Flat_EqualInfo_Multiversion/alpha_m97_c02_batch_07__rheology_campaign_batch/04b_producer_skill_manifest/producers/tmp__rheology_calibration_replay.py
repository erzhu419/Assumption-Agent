import importlib.util
import json
import numpy as np

path = '/holdout_task/alpha_m97_c02_batch_07_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
history = json.loads(m._HISTORY.read_text())
for i, sweep in enumerate(history):
    for lab in range(4):
        x, y = m._channel(sweep, lab)
        if len(x) < 2:
            continue
        A = np.column_stack((x, np.ones_like(x)))
        local = m._line(x, y)
        strict = m._line(x, y, archived=True) if len(x) >= 3 else None
        if strict is not None:
            clean = np.abs(y - A @ strict) <= .02
            if clean.sum() >= 4:
                shift = np.inf if local is None else np.max(np.abs(A @ (local - strict)))
                if shift > .02:
                    print('independently identified clean line differs from direct fit', i, lab, 'points', len(x), 'clean', clean.sum(), 'stress shift', shift)
        if strict is None:
            previous = next((p for h in reversed(history[:i]) if len(m._channel(h, lab)[0]) >= 4 and (p := m._line(*m._channel(h, lab), archived=True)) is not None), None)
            if previous is not None:
                residual = y - A @ previous
                if np.max(abs(residual)) > 3:
                    print('sparse EMI replay', i, lab, 'residuals vs previous clean sweep', np.round(residual, 4), 'direct fit accepted', local is not None)
