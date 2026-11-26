from pathlib import Path
p = Path('alpha_w01_s00_solver.py')
s = p.read_text().replace('import numpy as np', 'import json\nimport pickle\nfrom pathlib import Path\n\nimport numpy as np', 1)
start, end = s.index('def _load_cells('), s.index('def _alignment(')
s = s[:start] + '''def _lab_checks(checks, lab):
    labs = np.asarray(checks['lab_id'])
    x = np.asarray(checks['cmd_stress'], dtype=float)
    y = np.asarray(checks['true_stress'], dtype=float)
    good = (labs == lab) & np.isfinite(x) & np.isfinite(y)
    return x[good], y[good]


def _affine(x, y, prior):
    """Return the best local affine fit and whether the checks identify it."""
    if len(x) < 2 or np.ptp(x) < .05:
        return np.array([prior[0], np.median(y - prior[0]*x) if len(x) else prior[1]]), False
    a, b = np.triu_indices(len(x), k=1)
    sep = np.abs(x[a] - x[b]) >= max(.05, .05 * np.ptp(x))
    a, b = a[sep], b[sep]
    gain = (y[b] - y[a]) / (x[b] - x[a])
    bias = y[a] - gain * x[a]
    residual = np.abs(y - gain[:, None] * x - bias[:, None])
    # Trim gross EMI. If only two points agree, prefer the line nearest the prior.
    h = max(2, len(x) // 2 + 1)
    score = (np.sort(residual, axis=1)[:, :h]**2).sum(axis=1)
    support = (residual <= .001).sum(axis=1)
    if support.max() < h or h == 2:
        candidates = np.flatnonzero(support == support.max())
        delta = (gain[candidates] - prior[0]) * max(1., np.ptp(x))
        best = candidates[np.argmin(delta**2 + (bias[candidates] - prior[1])**2)]
    else:
        best = np.argmin(score)
    err = residual[best]
    clean = err <= max(.001, min(.005, 5 * np.median(err)))
    A = np.column_stack((x[clean], np.ones(clean.sum())))
    fit = np.linalg.lstsq(A, y[clean], rcond=None)[0]
    identified = clean.sum() >= max(h, 3) and np.ptp(x[clean]) >= .1 and fit[0] > 0
    return fit, identified


def _check_panels(value):
    if isinstance(value, dict):
        if {'lab_id', 'cmd_stress', 'true_stress'} <= value.keys():
            yield value
        for key in ('dataset', 'calibration_load_cells'):
            if key in value:
                yield from _check_panels(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _check_panels(item)


def _archived_maps(labs, root):
    """Find the latest identifiable historical calibration per requested lab."""
    files = [p for p in Path(root).rglob('*')
             if p.suffix.lower() in ('.json', '.pkl', '.pickle', '.npz') and p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    found = {}
    for p in files:
        if len(found) == len(labs):
            break
        try:
            ext = p.suffix.lower()
            if ext == '.json':
                value = json.loads(p.read_text())
            elif ext == '.npz':
                with np.load(p, allow_pickle=False) as arrays:
                    if not {'lab_id', 'cmd_stress', 'true_stress'} <= set(arrays.files):
                        continue
                    value = {k: arrays[k] for k in ('lab_id', 'cmd_stress', 'true_stress')}
            else:
                with p.open('rb') as handle:
                    value = pickle.load(handle)
            for panel in reversed(list(_check_panels(value))):
                for lab in labs - found.keys():
                    fit, reliable = _affine(*_lab_checks(panel, lab), (1., 0.))
                    if reliable:
                        found[lab] = fit
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            continue
    return found


def _load_cells(checks, archive_root='/tmp/hegel_continual_archive'):
    nominal = np.array([1., 0.])
    data = [_lab_checks(checks, lab) for lab in range(4)]
    local = [_affine(x, y, nominal) for x, y in data]
    sparse = {lab for lab, (_, identified) in enumerate(local) if not identified}
    archive = _archived_maps(sparse, archive_root) if sparse else {}
    for lab, prior in archive.items():
        x, y = data[lab]
        # A spot check can update the offset but cannot identify a gain.
        offset = y - (prior[0] * x + prior[1])
        close = offset[np.abs(offset) <= .02]
        gain, bias = prior
        if len(close):
            bias += np.median(close)
        local[lab] = (np.array([gain, bias]), True)
    return np.array([fit for fit, _ in local])


''' + s[end:]
p.write_text(s)
