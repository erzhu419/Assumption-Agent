from pathlib import Path
import numpy as np

artifact = Path('/holdout_task/multicenter_calibration.py').read_text()
stress_path = '/tmp/assay_stress.py'
stress = Path(stress_path).read_text()
load = 'spec.loader.exec_module(assay)'
assert artifact.count('spread = max(shared, local)') == 1 and artifact.count('center = np.median(x)') == 1
for name, spread, center in (
    ('baseline', 'max(shared, local)', 'np.median(x)'),
    ('shared', 'shared', 'np.median(x)'),
    ('tempered', 'shared + 0.25 * max(0.0, local - shared)', 'np.median(x)'),
    ('tempered_precision', 'shared + 0.25 * max(0.0, local - shared)', 'np.average(x, weights=1.0 / (v + spread))'),
):
    candidate = artifact.replace('spread = max(shared, local)', f'spread = {spread}').replace('center = np.median(x)', f'center = {center}')
    code = stress.replace(load, 'exec(compile(candidate, "candidate", "exec"), assay.__dict__)')
    code = code.replace('range(12)', 'range(48)').replace('for hetero in (0.08, 0.30):', 'for hetero in (0.08, 0.30):')
    anchor = "'oracle AUROC/Brier mean',np.round(np.mean(oracle,0),4)"
    assert anchor in code
    code = code.replace(anchor, "'MAE p90', round(np.quantile(a[:,0], .90),4), 'MAE fail count', int((a[:,0] > .24).sum()), " + anchor)
    print(name, flush=True)
    exec(compile(code,stress_path,'exec'), {'candidate':candidate})
