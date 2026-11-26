import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('helmholtz_phase_inversion', '/holdout_task/helmholtz_phase_inversion.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)


def rel(x, y):
    return np.linalg.norm(x-y) / np.linalg.norm(y)


def cauchy(A, z, alpha, coef):
    X = np.vstack((A.real, A.imag))
    y = np.r_[z.real, z.imag]
    penalty = np.sqrt(alpha) * np.eye(A.shape[1])
    for _ in range(35):
        r = np.abs(A @ coef-z)
        scale = max(np.median(r)/np.sqrt(2*np.log(2)), np.finfo(float).eps)
        root = np.sqrt(np.tile(1/(1+(r/(2.385*scale))**2), 2))
        new = np.linalg.lstsq(np.vstack((X*root[:, None], penalty)), np.r_[y*root, np.zeros(A.shape[1])], rcond=None)[0]
        if np.linalg.norm(new-coef) <= 1e-9*(1+np.linalg.norm(new)):
            return new
        coef = new
    return coef


worst = []
for i, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    survey, pilot = p['survey'], p['pilot_calibration']
    shape = p['n_sq'].shape
    R = m._responses(p)
    baseline = m.invert_helmholtz_sources_and_phase(p)
    phi = baseline['global_phase_rad']
    phase_error = abs(np.angle(np.exp(1j*(phi-gt['global_phase_rad']))))
    assert phase_error <= .02 and -np.pi < phi <= np.pi
    forward_truth = rel(np.abs(R @ gt['source_amplitudes']).reshape(shape), gt['predicted_field_abs'])
    assert forward_truth < 1e-12, forward_truth
    k = pilot['known_field_re'] + 1j*pilot['known_field_im']
    w = m._complex_obs(pilot)
    j, l = np.triu_indices(len(k), 1)
    raw_resid = np.abs(w-np.exp(1j*phi)*k)
    pair_resid = np.abs((w[j]-w[l])-np.exp(1j*phi)*(k[j]-k[l]))
    print(f'case {i}: phase error={phase_error:.7g} rad, truth forward error={forward_truth:.3g}, pilot direct RMS={np.sqrt(np.mean(raw_resid**2)):.4g}, pair-difference RMS={np.sqrt(np.mean(pair_resid**2)):.4g}, opposite-sign pilot RMS={np.sqrt(np.mean(np.abs(w+np.exp(1j*phi)*k)**2)):.4g}')
    ij = survey['sensor_ij']
    A = R[np.ravel_multi_index(ij.T, shape)]
    z = m._complex_obs(survey)*np.exp(-1j*phi)
    quadrant = 2*(ij[:,0] >= shape[0]/2) + (ij[:,1] >= shape[1]/2)
    high = np.argsort(np.linalg.norm(A, axis=1))[-len(A)//4:]
    high_mask = np.isin(np.arange(len(A)), high)
    splits = [(f'quadrant {q}', quadrant==q) for q in range(4)] + [('highest leverage quartile', high_mask)]
    public = [rel(baseline['source_amplitudes'], gt['source_amplitudes']), rel(baseline['predicted_field_abs'], gt['predicted_field_abs'])]
    worst.append(public)
    all_scores = []
    for name, test in splits:
        train = ~test
        reduced = dict(survey, **{c: survey[c][train] for c in ('sensor_ij','amplitude_obs','wrapped_phase_obs')})
        out = m.invert_helmholtz_sources_and_phase(dict(p, survey=reduced))
        a = out['source_amplitudes']
        b = cauchy(A[train], z[train], survey['tikhonov_alpha'], a)
        scores = []
        for coef in (a, b):
            scores.append([rel(coef, gt['source_amplitudes']), rel(np.abs(R@coef).reshape(shape), gt['predicted_field_abs']), np.median(np.abs(A[test]@coef-z[test]))])
        all_scores.append(scores)
        worst.append(scores[0][:2])
        print(f'  hold out {name}: n={test.sum()}, current vs rational [source rel, field rel, held-out median complex residual] = {np.array(scores).round(6).tolist()}')
    print('  worst slab [current, rational]:', np.max(all_scores, axis=0))
print('worst current public and blocked holdouts [source, field]:', np.max(worst, axis=0))
assert np.all(np.max(worst, axis=0) <= .03)
