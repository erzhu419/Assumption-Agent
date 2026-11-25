import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('assay', '/holdout_task/multicenter_calibration.py')
assay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assay)
with open('/holdout_task/public_multicenter_assay.pkl', 'rb') as handle:
    cohorts = pickle.load(handle)['cohorts']

# Can other centers predict a repeatable biomarker-specific departure from
# the local panel anchor when that biomarker has no usable local controls?
for k, cohort in enumerate(cohorts):
    d = cohort['dataset']; ref = d['reference_controls']; ids = ref['center_id']; x = ref['X']
    rng = np.random.default_rng(1318)
    centers = np.unique(ids)
    errors = {c: [] for c in centers}
    for _ in range(20):
        cal = rng.random(len(x)) < .7
        stats = {c: assay._panel_statistics(x[cal & (ids == c)]) for c in centers}
        for c in centers:
            for j in range(x.shape[1]):
                vals = x[~cal & (ids == c), j]
                vals = vals[np.isfinite(vals)]
                if vals.size < 6:
                    continue
                residuals = []
                for other in centers[centers != c]:
                    n, loc, log_s, ok = stats[other]
                    anchor = ok & (n >= 4); anchor[j] = False
                    if n[j] >= 4 and ok[j] and anchor.any():
                        a, b = np.median(loc[anchor]), np.median(log_s[anchor])
                        residuals.append(((loc[j]-a)/np.exp(b), log_s[j]-b))
                n, loc, log_s, ok = stats[c]
                anchor = ok & (n >= 4); anchor[j] = False
                if not residuals or not anchor.any():
                    continue
                a, b = np.median(loc[anchor]), np.median(log_s[anchor])
                da, db = np.median(residuals, axis=0)
                row = []
                for weight in (0., .5, 1.):
                    mu = a + weight * da * np.exp(b)
                    log_sd = b + weight * db
                    row.append(log_sd + .5 * np.mean(((vals-mu)/np.exp(log_sd))**2))
                errors[c].append(row)
    print('cohort', k, 'held-out control Gaussian NLL; columns: local only / half / full cross-center correction')
    for c, rows in errors.items():
        print(' center', c, np.round(np.mean(rows, axis=0), 4))
