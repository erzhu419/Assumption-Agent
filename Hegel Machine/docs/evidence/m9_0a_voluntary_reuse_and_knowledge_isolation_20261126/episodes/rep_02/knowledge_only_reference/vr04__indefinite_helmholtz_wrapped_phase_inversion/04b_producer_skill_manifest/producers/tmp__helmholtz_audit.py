import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('inversion', '/holdout_task/helmholtz_phase_inversion.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

for case_id, case in enumerate(cases):
    p, truth = case['problem'], case['gt']
    s = p['survey']; pilot = p['pilot_calibration']
    a0 = truth['source_amplitudes']; phase0 = truth['global_phase_rad']; f0 = truth['predicted_field_abs']
    field = m._responses(p)
    i,j = s['sensor_ij'].T
    x = field[i,j]
    out = m.invert_helmholtz_sources_and_phase(p)
    y = m._phasors(s) * np.exp(-1j*out['global_phase_rad'])
    alpha = s['tikhonov_alpha']
    pred = x @ a0
    def score(a):
        return np.array([np.linalg.norm(a-a0)/np.linalg.norm(a0),np.linalg.norm(np.abs(field@a)-f0)/np.linalg.norm(f0)])
    stacked = np.r_[x.real,x.imag]
    ridge = np.linalg.solve(stacked.T@stacked+alpha*np.eye(x.shape[1]), stacked.T@np.r_[y.real,y.imag])
    print('case',case_id, 'source / field plain ridge:',score(ridge),'grouped Cauchy:',score(out['source_amplitudes']))
    # Spatial block holdouts test whether neighboring channels hide a local failure.
    block = 2*(i >= field.shape[0]//2).astype(int) + (j >= field.shape[1]//2)
    folds=[]; predicted=[]; ridge_predicted=[]
    for held in np.unique(block):
        keep = block != held
        fit = m._robust_sources(x[keep], y[keep], alpha)
        a_train = np.r_[x[keep].real,x[keep].imag]; y_train = np.r_[y[keep].real,y[keep].imag]
        fit_l2 = np.linalg.solve(a_train.T@a_train+alpha*np.eye(x.shape[1]),a_train.T@y_train)
        folds.append(score(fit))
        predicted.append(np.linalg.norm(x[~keep]@fit-pred[~keep])/np.linalg.norm(pred[~keep]))
        ridge_predicted.append(np.linalg.norm(x[~keep]@fit_l2-pred[~keep])/np.linalg.norm(pred[~keep]))
    known = pilot['known_field_re']+1j*pilot['known_field_im']
    pilot_z = m._phasors(pilot)
    pilot_errors=[]; pilot_source_field=[]
    for held in range(len(known)):
        keep=np.arange(len(known))!=held
        ph=np.angle(np.vdot(known[keep],pilot_z[keep]))
        pilot_errors.append(abs(np.angle(np.exp(1j*(ph-phase0)))))
        pilot_source_field.append(score(m._robust_sources(x,m._phasors(s)*np.exp(-1j*ph),alpha)))
    print('spatial blocks counts:',np.bincount(block),'worst source / field:',np.max(folds,axis=0),'worst held-out complex prediction Cauchy / ridge:',max(predicted),max(ridge_predicted))
    print('pilot jackknife worst phase / source / field:',max(pilot_errors),*np.max(pilot_source_field,axis=0))
    assert np.all(np.max(folds,axis=0)<=.03) and np.all(np.max(pilot_source_field,axis=0)<=.03) and max(pilot_errors)<=.02
