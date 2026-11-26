import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/helmholtz_phase_inversion.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
def rel(x,y): return np.linalg.norm(x-y)/np.linalg.norm(y)
for i,c in enumerate(cases):
    p,g = c['problem'],c['gt']
    out = s.invert_helmholtz_sources_and_phase(p)
    a,u = g['source_amplitudes'],g['predicted_field_abs']
    survey = p['survey']; ij = survey['sensor_ij']; response = s._response(p)
    D = response[ij[:,0],ij[:,1],:]; z = s._polar(survey)*np.exp(-1j*out['global_phase_rad']); alpha = survey['tikhonov_alpha']
    err = [rel(out['source_amplitudes'],a), abs(np.angle(np.exp(1j*(out['global_phase_rad']-g['global_phase_rad'])))), rel(out['predicted_field_abs'],u)]
    truth_resid = np.abs(z-D@a)
    jack = []
    for j in range(len(z)):
        keep = np.arange(len(z)) != j
        est = s._robust_fit(D[keep],z[keep],alpha)
        jack.append((rel(est,a),rel(np.abs(response@est),u),abs(D[j]@(est-a))))
    regless = s._robust_fit(D,z,0.)
    ordinary = np.linalg.lstsq(np.vstack([D.real,D.imag]),np.r_[z.real,z.imag],rcond=None)[0]
    print(i, 'source/phase/field',err,'LOSO max source/field/clean sensor abs',np.max(jack,0),'LS source',rel(ordinary,a),'unregularized robust source',rel(regless,a))
    print('  clean forward error',rel(np.abs(response@a),u),'signal median',np.median(np.abs(D@a)),'observed residual quantiles',np.quantile(truth_resid,[0,.5,.9,.93,.95,1]),'singular values',np.linalg.svd(np.vstack([D.real,D.imag]),compute_uv=False))
