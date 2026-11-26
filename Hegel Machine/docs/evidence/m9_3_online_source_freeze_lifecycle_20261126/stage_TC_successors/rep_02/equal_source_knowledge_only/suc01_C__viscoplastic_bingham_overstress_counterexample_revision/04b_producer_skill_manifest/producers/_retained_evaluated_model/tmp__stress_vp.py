import importlib.util
import pickle
import numpy as np
p = '/holdout_task/viscoplastic_overstress_hereditary_solver.py'
spec = importlib.util.spec_from_file_location('vp', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_suc01_C_datasets.pkl', 'rb') as f:
    cases = pickle.load(f)
for i, case in enumerate(cases):
    d, g = case['dataset'], case['gt']
    tr, te = d['train'], d['test']
    steps, times, sy = d['past_dt'], d['eval_times'], d['yield_stress']
    out = m.solve_viscoplastic_hereditary_revision(d)
    tau, vp, pred = (out[k] for k in ('estimated_taus', 'viscoplastic_weights', 'predicted_strain'))
    h, u = te['past_stress'], te['stress_hold_0']
    hist, hold = (abs(h)>sy).any(axis=1), abs(u)>sy
    hot = hist | hold
    err = np.abs(pred-g['te_clean'])
    tau_err = np.max(abs(tau/g['taus']-1))
    vp_err = np.linalg.norm(vp-g['vp_theta'])/np.linalg.norm(g['vp_theta'])
    scores = [err.mean(),err[hot].mean(),err[~hot].mean()]
    print(i, 'relative tau/vp error:', tau_err, vp_err, 'MAE overall/hot/cold:', scores)
    assert 0.20 - 1e-14 <= tau[0] < tau[1] <= 8 + 1e-14
    assert tau_err <= .10 and vp_err <= .14 and np.all(np.array(scores) <= [.070,.080,.065])
    for name, mask in [('neither', ~hot), ('history only', hist & ~hold), ('hold only', hold & ~hist), ('both', hist & hold), ('positive yielding hold', u>sy), ('negative yielding hold', u< -sy)]:
        if mask.any():
            print(name, int(mask.sum()), 'MAE', round(err[mask].mean(),5), 'worst time MAE', round(err[mask].mean(axis=0).max(),5))
    cells = np.column_stack((te['lab_id'], te['stratum']))
    print('worst lab-stratum MAE', max((err[(cells==cell).all(axis=1)].mean(),tuple(cell)) for cell in np.unique(cells,axis=0)))
    xtr, xte, wt = m._align(tr, te)
    a, rate = m._Response(te,xte,steps,times,sy).design(tau)
    z = np.zeros((len(u),2))
    for stress, dt in zip(h.T,steps):
        z += (stress[:,None]-z)*(-np.expm1(-dt/tau))
    gap = u[:,None]-z
    ov = np.maximum(abs(u)-sy,0)**2*np.sign(u)
    surge = u*((np.maximum(abs(h)-sy,0)**2)@steps)
    rise = -np.expm1(-times[:,None]/tau)
    np.testing.assert_allclose(a[:,:,:2], gap[:,None,:]*rise,atol=1e-13)
    np.testing.assert_allclose(a[:,:,-2],ov[:,None]*times,atol=1e-13)
    np.testing.assert_allclose(a[:,:,-1],surge[:,None]*rise[:,0],atol=1e-13)
    np.testing.assert_allclose(rate[:,[0,1,-2,-1]],np.column_stack((gap/tau,ov,surge/tau[0])),atol=1e-13)
    assert not a[~hot,:,-2:].any() and not rate[~hot,-2:].any()
    t0,b0 = m._profile(m._Response(tr,xtr,steps,times,sy),tr,wt,slice(0,7))
    a0,_ = m._Response(te,xte,steps,times,sy).design(t0)
    e0 = abs(te['strain_0'][:,None]+a0[:,:,:7]@b0-g['te_clean'])
    print('refitted Prony-only MAE hot/cold:',e0[hot].mean(),e0[~hot].mean())
