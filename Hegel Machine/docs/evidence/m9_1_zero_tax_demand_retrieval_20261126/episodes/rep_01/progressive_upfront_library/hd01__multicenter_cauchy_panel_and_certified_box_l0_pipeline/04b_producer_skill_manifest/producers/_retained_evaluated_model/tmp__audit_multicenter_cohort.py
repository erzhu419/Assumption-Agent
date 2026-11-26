import importlib.util
import pickle
import numpy as np

path = '/holdout_task/multicenter_certified_l0.py'
spec = importlib.util.spec_from_file_location('solver', path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

def fit_predict(c, fold, mode):
    groups, strata, y = (c[k] for k in ('center_id', 'stratum', 'y_potency'))
    train = groups != fold
    x = c['X_raw'].copy()
    if mode != 'raw':
        for g in np.unique(groups):
            rows = groups == g
            panel = c['controls'][rows]
            panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
            x[rows] = (x[rows] - np.median(np.nanmean(panel, axis=0))) / np.median(np.nanstd(panel, axis=0))
    _, cells, sizes = np.unique(np.column_stack((groups[train], strata[train])), axis=0, return_inverse=True, return_counts=True)
    base = (len(cells) / len(sizes)) / sizes[cells] if mode != 'aligned_unweighted' else np.ones(len(cells))
    observed = np.isfinite(x[train]); fill = (base @ np.where(observed, x[train], 0)) / (base @ observed)
    x = np.where(np.isfinite(x), x, fill)
    design = np.column_stack((np.ones(len(y)), strata == 1, strata == 2, x))
    a, b = design[train], y[train]
    def ls(w):
        w = np.sqrt(w)
        return np.linalg.lstsq(a*w[:,None],b*w,rcond=None)[0]
    coef = ls(base)
    r = b-a@coef
    scale = 1.4826*np.median(np.abs(r-np.median(r)))
    for _ in range(100):
        new = ls(base/(1+((b-a@coef)/scale)**2))
        if np.max(abs(new-coef))<1e-10:
            coef=new
            break
        coef=new
    return coef, np.abs(y[~train]-design[~train]@coef), strata[~train]

for i, case in enumerate(pickle.load(open('/holdout_task/public_multicenter_l0_studies.pkl','rb'))):
    study, gt = case['study'], case['gt']; c = study['cohort']
    out = solver.solve_multicenter_certified_l0(study)
    for mode in ('raw', 'aligned_weighted', 'aligned_unweighted'):
        center, cell, slopes = [], [], []
        for fold in np.unique(c['center_id']):
            coef, err, s = fit_predict(c, fold, mode)
            center.append(np.median(err))
            cell.extend(np.median(err[s==k]) for k in np.unique(s))
            slopes.append(np.linalg.norm(coef[3:]-gt['beta_potency'])/np.linalg.norm(gt['beta_potency']))
        print(i, mode, 'worst LOCO center MedAE', round(max(center),5), 'worst held-out cell MedAE',round(max(cell),5),'worst held-out slope error',round(max(slopes),5))
    align = np.abs(out['aligned_biomarkers']-gt['Z_true'])
    miss = ~np.isfinite(c['X_raw'])
    print(i,'alignment total/observed/imputed MAE',*[round(float(q),6) for q in (align.mean(),align[~miss].mean(),align[miss].mean())], 'worst center MAE', round(max(align[c['center_id']==g].mean() for g in np.unique(c['center_id'])),6))
