from pathlib import Path
exec(Path('/tmp/rheology03_cv.py').read_text().split('for weight in')[0])

ec, er = np.empty_like(y), np.empty_like(r)
fold_taus = []
for lab in range(4):
    val = labs == lab
    tau, w = s._fit_robust(model, y, r, dt, t, row_mask=~val)
    c, rr = model(tau)
    ec[val], er[val] = (c @ w-y)[val], (rr @ w-r)[val]
    fold_taus.append(tau)
print('Fold taus:', np.round(fold_taus, 5).tolist())
for name, err in [('curve',ec),('rate',er)]:
    print(name, 'worst slice RMSE:', max((rmse(err[m]), k) for k,m in slices.items() if m.any()), 'worst slice MedAE:', max((np.median(np.abs(err[m])), k) for k,m in slices.items() if m.any()))
print('Worst horizon MedAE:', np.max(np.median(np.abs(ec), axis=0)))
assert max(rmse(ec[m]) for m in slices.values() if m.any()) < 3.045
assert max(rmse(er[m]) for m in slices.values() if m.any()) < 0.176
assert np.max(np.median(np.abs(ec), axis=0)) < 0.016
out = s.solve_alpha_m97_c03_batch_03(d)
tau, w, rw = (out[k] for k in ('estimated_taus','hereditary_weights','regime_mode_weights'))
assert np.all((tau > 0) & (tau >= np.min(fold_taus, axis=0)*0.95) & (tau <= np.max(fold_taus, axis=0)*1.05))
_, x_test = s._align(train, test)
g, b = params[test['lab_id']].T
hold = g*test['stress_hold_cmd'] + b
z = np.zeros((len(hold),2))
for k, step in enumerate(dt):
    decay = np.exp(-step/tau)
    stress = g*test['past_stress_cmd'][:,k]+b
    z = z*decay + stress[:,None]*(1-decay)
a, c = (hold[:,None]-z).T
flow = w[2]+x_test@w[3:]+rw[0]*np.sign(hold)*np.sqrt(np.abs(hold))
prony = 1-np.exp(-t[:,None]/tau)
f1 = w[0]*a+rw[1]*(a-c)
expected = test['strain_0'][:,None] + f1[:,None]*prony[:,0] + (w[1]*c)[:,None]*prony[:,1] + flow[:,None]*t/(1+0.85*t)
np.testing.assert_allclose(out['predicted_strain'], expected, rtol=0, atol=1e-12)
np.testing.assert_allclose(out['initial_rate_pred'], f1/tau[0]+w[1]*c/tau[1]+flow, rtol=0, atol=1e-12)
print('Final taus:', tau, '; independent curve and initial-rate checks passed')
