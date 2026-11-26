import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('robin', '/holdout_task/robin3d_amg_tomography.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin3d_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
for case in cases:
    p = case['problem']
    out = mod.solve_3d_robin_tomography(p)
    U, k = out['basis_fields'], p['kappa']
    h = np.array([p['Lx'], p['Ly'], p['Lz']]) / k.shape
    vol, gamma = np.prod(h), p['gamma_robin']
    influx = vol * p['candidate_sources'].sum(axis=(1, 2, 3))
    losses = vol * (p['sigma'] * U).sum(axis=(1, 2, 3))
    for axis in range(3):
        for face in (0, -1):
            kb = k.take(face, axis=axis)
            ub = U.take(face, axis=axis + 1)
            losses += (ub * ((vol / h[axis]) / (1 / gamma + 0.5 * h[axis] / kb))).sum(axis=(1, 2))
    conservation = np.max(np.abs(losses - influx) / np.abs(influx))
    print('public global balance, worst RHS:', conservation)
    assert conservation <= 1e-7
    ijk = p['sensor_ijk']
    boundary = ((ijk == 0) | (ijk == np.array(k.shape) - 1)).any(axis=1)
    Z, Y = p['nuisance_design'], p['Y_obs']
    S = U[(slice(None), *ijk.T)].T
    S_ref = case['gt']['basis_fields'][(slice(None), *ijk.T)].T
    lz, ls = p['nuisance_ridge'], p['source_ridge']
    for label, keep in [('interior only', ~boundary), ('boundary only', boundary)]:
        z, y, s, sref = Z[keep], Y[keep], S[keep], S_ref[keep]
        if len(z) <= Z.shape[1]:
            continue
        single, joint = mod._profile(s, z, y, lz, ls)
        def direct(x, penalty):
            a, b = z.shape[1], x.shape[1]
            design = np.block([[z, x], [np.sqrt(lz)*np.eye(a), np.zeros((a,b))], [np.zeros((b,a)), np.sqrt(penalty)*np.eye(b)]])
            return np.linalg.lstsq(design, np.vstack((y, np.zeros((a+b,y.shape[1])))), rcond=None)[0][a:]
        a_ref = np.vstack([direct(sref[:, j:j+1], 0) for j in range(sref.shape[1])])
        w_ref = direct(sref, ls)
        print(label, len(z), 'sensors; max single/joint error:', np.max(np.abs(single-a_ref)), np.max(np.abs(joint-w_ref)))
        np.testing.assert_allclose(single, a_ref, atol=1e-9, rtol=1e-8)
        np.testing.assert_allclose(joint, w_ref, atol=1e-9, rtol=1e-8)
