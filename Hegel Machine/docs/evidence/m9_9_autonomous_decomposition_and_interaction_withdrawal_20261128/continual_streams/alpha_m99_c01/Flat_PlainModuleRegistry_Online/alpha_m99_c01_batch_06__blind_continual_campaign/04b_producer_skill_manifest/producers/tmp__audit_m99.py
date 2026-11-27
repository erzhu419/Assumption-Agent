import sys, pickle
import numpy as np
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from _verified_upstream import calibrate_upstream
import downstream_solver as d
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    raw = pickle.load(f)[0]['dataset']
out = solve_batch(raw)
cal = calibrate_upstream(raw)
tr, te = cal['train'], cal['test']
t, dt = cal['eval_times'], cal['past_dt']
x, xt = d._features(tr, te)
p, h = tr['past_channel_raw'].sum(-1), tr['hold_channel_raw'].sum(-1)
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
np.testing.assert_allclose(out['hereditary_weights'], d.solve_downstream(cal)['hereditary_weights'])
assert all(np.isfinite(v).all() for v in out.values())
results = []
for lab in np.unique(tr['lab_id']):
    v = tr['lab_id'] == lab
    tau, beta = d._fit(x[~v], p[~v], h[~v], dt, t, y[~v], r[~v])
    a, b = d._design(tau, x, p, h, dt, t)
    z = np.zeros((len(h), 2))
    for step, delta in enumerate(dt):
        z = np.exp(-delta/tau)*z - np.expm1(-delta/tau)*p[:, step, None]
    np.testing.assert_allclose(b[:, :2], (h[:, None]-z)/tau, atol=1e-12)
    ey, er = y-a@beta, r-b@beta
    diff = h[:, None]-z
    # Fixed interaction families. Fit corrections only on the outer training labs.
    nl = np.column_stack((diff**2, diff/(1+diff**2), diff[:, 0]*diff[:, 1], x*diff[:, :1], x*diff[:, 1:], x[:, 0]*x[:, 1]*x[:, 2]))
    deviations = []
    for sl in (slice(0,0), slice(0,5), slice(5,None)):
        q = nl[:,sl]
        if q.shape[1]:
            scale = np.maximum(q[~v].std(0), 1e-10)
            q = (q-q[~v].mean(0))/scale
            basis = -np.expm1(-t[:,None]/tau)
            ay = (basis[None,:,:,None]*q[:,None,None,:]).reshape(len(h),len(t),-1)
            ar = (q[:,None,:]/tau[None,:,None]).reshape(len(h),-1)
            mat = np.vstack((ay[~v].reshape(-1,ay.shape[-1]),ar[~v]))
            res = np.r_[ey[~v].ravel(),er[~v]]
            weight = 1/np.hypot(1,res/.03)
            aw = mat*weight[:,None]
            coef = np.linalg.solve(aw.T@aw+2*np.eye(mat.shape[1]), aw.T@(weight*res))
            e, f = ey[v]-ay[v]@coef, er[v]-ar[v]@coef
        else:
            e, f = ey[v], er[v]
        deviations.append([abs(e).mean(),abs(e[:,-3:]).mean(),abs(f).mean()])
    results.append(deviations)
    print('lab',lab,'mean residual strain/rate',round(ey[v].mean(),5),round(er[v].mean(),5),'MAEs baseline | nonlinear stress | coupon interactions',np.round(deviations,5).tolist())
print('Worst held-out MAEs by model (curve, late curve, rate):',np.round(np.max(results,0),6).tolist())
# Both stress-memory boundaries must agree at zero history; nonzero history must affect only the carrying boundary.
h = te['hold_channel_raw'].sum(-1)
tau,beta = out['estimated_taus'],out['hereditary_weights']
tt = np.r_[0., 1e-6, t]
reset, slope = d._predict(tau,beta,xt,h,tt)
carry_a,carry_b = d._design(tau,xt,te['past_channel_raw'].sum(-1),h,dt,tt)
zero_a,zero_b = d._design(tau,xt,np.zeros((len(h),len(dt))),h,dt,tt)
carry = reset+(carry_a-zero_a)@beta
carry_slope = slope+(carry_b-zero_b)@beta
for curve, rate in ((reset,slope),(carry,carry_slope)):
    np.testing.assert_array_equal(curve[:,0],0.)
    np.testing.assert_allclose(curve[:,1]/tt[1],rate,rtol=2e-5,atol=5e-6)
np.testing.assert_allclose(out['predicted_strain'], reset[:,2:]+te['strain_0'][:,None])
np.testing.assert_allclose(out['initial_rate_pred'],slope)
print('Reset versus carry max absolute curve/rate difference:',abs(reset-carry).max(),abs(slope-carry_slope).max())
