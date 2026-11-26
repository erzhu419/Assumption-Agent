import importlib.util
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location('calibration', '/holdout_task/viscoelastic_panel_calibration.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
run = solver.calibrate_viscoelastic_cohort
with open('/holdout_task/public_viscoelastic_cohort.pkl', 'rb') as handle:
    data = pickle.load(handle)

for k, item in enumerate(data):
    co, gt = item['cohort'], item['gt']
    runs = []
    for omitted in (-1, 0, 1, 2, 3, 4):
        stress = dict(co)
        if omitted >= 0:
            for split in ('train', 'query'):
                part = co[split]
                stress[split] = dict(part, controls=np.delete(part['controls'], omitted, axis=1))
        out = run(stress)
        p, z = out['query_probabilities'], out['aligned_query_moduli']
        for c in np.unique(co['query']['center_id']):
            rows = co['query']['center_id'] == c
            y = gt['y_query'][rows]
            sens = np.mean(p[rows][y == 1] >= .5)
            specf = np.mean(p[rows][y == 0] < .5)
            runs.append((omitted, c, roc_auc_score(y, p[rows]), np.mean((p[rows] - y)**2), np.mean(abs(z[rows] - gt['Z_query'][rows])), (sens + specf)/2, sens, specf))
    print('cohort', k, 'rows: omitted control (-1 = full), center, AUROC, Brier, MAE, BA, sensitivity, specificity')
    for row in runs:
        print(tuple(round(float(v), 5) for v in row))
