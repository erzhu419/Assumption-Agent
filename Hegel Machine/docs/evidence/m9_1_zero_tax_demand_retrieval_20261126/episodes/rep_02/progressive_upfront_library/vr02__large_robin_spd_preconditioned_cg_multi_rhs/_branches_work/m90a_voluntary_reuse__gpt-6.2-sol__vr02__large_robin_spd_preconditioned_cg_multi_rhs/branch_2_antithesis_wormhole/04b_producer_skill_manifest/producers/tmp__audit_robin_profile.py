import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('robin_spd_profiler', '/holdout_task/robin_spd_profiler.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin_spd_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

for i, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    base = mod.solve_robin_spd_and_profile_sources(p)
    fields = base['basis_fields']
    k, g = p['kappa'], p['gamma_robin']
    dx, dy = p['Lx']/k.shape[0], p['Ly']/k.shape[1]
    mass = (dx*dy*p['candidate_sources']).sum(axis=(1,2))
    loss = (dx*dy*p['sigma']*fields).sum(axis=(1,2))
    for edge in (0, -1):
        loss += (dy * fields[:,edge,:] / (1/g + dx/(2*k[edge,:]))).sum(axis=1)
        loss += (dx * fields[:,:,edge] / (1/g + dy/(2*k[:,edge]))).sum(axis=1)
    balance = np.max(np.abs(loss-mass)/np.abs(mass))
    assert balance < 1e-8, (i, 'global flux balance', balance)

    ij = p['sensor_ij']; Z, Y = p['nuisance_design'], p['Y_obs']
    S = fields[:,ij[:,0],ij[:,1]].T
    mz, ms = Z.shape[1], S.shape[1]
    c = np.linspace(-.6,.6,mz)[:,None] * np.linspace(.5,1.5,Y.shape[1])
    dy_obs = Z @ c
    changed = mod.solve_robin_spd_and_profile_sources(dict(p, Y_obs=Y+dy_obs))
    errs = []
    for key, ridge, groups in (
        ('single_source_profiled_amplitudes', 0., [S[:,j:j+1] for j in range(ms)]),
        ('joint_source_weights', p['source_ridge'], [S]),
    ):
        expected = []
        for G in groups:
            d = G.shape[1]
            X = np.column_stack((Z, G))
            penalty = np.diag(np.sqrt(np.r_[np.full(mz,p['nuisance_ridge']),np.full(d,ridge)]))
            coef = np.linalg.lstsq(np.vstack((X, penalty)), np.pad(dy_obs, ((0,mz+d),(0,0))), rcond=None)[0]
            expected.append(coef[mz:])
        expected = np.vstack(expected)
        delta = changed[key] - base[key]
        np.testing.assert_allclose(delta, expected, rtol=1e-8, atol=1e-10)
        errs.append((key, float(np.max(np.abs(delta-expected))), float(np.max(np.abs(expected)))))

    q = p['candidate_sources'].copy(); q[0] -= .7*q[1]
    mixed = mod.solve_robin_spd_and_profile_sources(dict(p, candidate_sources=q))['basis_fields'][0]
    ref = gt['basis_fields'][0] - .7*gt['basis_fields'][1]
    superposition = np.linalg.norm(mixed-ref)/np.linalg.norm(ref)
    assert superposition < 1e-8, (i, 'signed superposition', superposition)
    print('public', i, 'global balance rel', balance, 'signed superposition rel', superposition, 'nuisance response (max error, max expected magnitude)', errs)
