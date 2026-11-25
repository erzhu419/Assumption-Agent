import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('assay_ic50','/holdout_task/assay_ic50.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with open('/holdout_task/public_screen.pkl','rb') as f:
    data = pickle.load(f)

fields = ('log10_ic50_uM', 'hill_slope')
def score(plates, name, gt=data['gt']):
    predictions = module.estimate_compound_hill_parameters(plates)
    assert predictions.keys() == gt.keys()
    assert all(set(p) == set(fields) and all(type(p[k]) is float and np.isfinite(p[k]) for k in fields) for p in predictions.values())
    err = [{c: abs(p[k]-gt[c][k]) for c,p in predictions.items()} for k in fields]
    print(name, [(max(e,key=e.get),round(max(e.values()),5)) for e in err], flush=True)
    return np.array([max(e.values()) for e in err])

if __name__ == '__main__':
    plots = data['plates']
    limit = np.array([0.09, 0.18])
    assert np.all(score(plots, 'all plates') <= limit)
    for i,p in enumerate(plots):
        assert np.all(score(plots[:i]+plots[i+1:], 'without '+p['plate_id']) <= limit)
        assert np.all(score([p], 'only '+p['plate_id']) <= limit)
