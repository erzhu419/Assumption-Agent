import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('helmholtz', '/holdout_task/helmholtz_phase_inversion.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
for case in cases:
    p, gt = case['problem'], case['gt']
    out = m.invert_helmholtz_sources_and_phase(p)
    print('geometry', p['Lx'], p['Ly'], p['k0'], p['beta_imp'])
    for key, expected in gt.items():
        error = np.angle(np.exp(1j*(out[key]-expected))) if key == 'global_phase_rad' else np.linalg.norm(out[key]-expected)/np.linalg.norm(expected)
        print(key, abs(error))
    fields = m._source_fields(p)
    survey = p['survey']
    i,j = survey['sensor_ij'].T
    design = fields[i,j]
    true = design @ gt['source_amplitudes']
    obs = m._complex_observations(survey) * np.exp(-1j*gt['global_phase_rad'])
    a_ls = np.linalg.lstsq(np.r_[design.real, design.imag], np.r_[obs.real, obs.imag], rcond=None)[0]
    print('unweighted source error', np.linalg.norm(a_ls-gt['source_amplitudes'])/np.linalg.norm(gt['source_amplitudes']))
    print('signal and true residual quantiles', np.quantile(abs(true), [.1,.5,.9]), np.quantile(abs(obs-true), [.1,.5,.9,.93,.95,1]))
    print('condition', np.linalg.cond(np.r_[design.real, design.imag]), 'true-field forward error', np.linalg.norm(abs(fields @ gt['source_amplitudes']) - gt['predicted_field_abs'])/np.linalg.norm(gt['predicted_field_abs']))
