import pickle
import runpy
import numpy as np

m = runpy.run_path('/holdout_task/batch_solver.py')
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    batch = pickle.load(f)

worst = dict(pulse=0., wall=0., interior=0., terminal=0., budget=0., gap=0., complementarity=0.)
for item in batch:
    s = item['study']
    nx, ny = s['nx'], s['ny']
    A = np.diag(s['V_reaction_2d'].ravel(order='F'))
    for j in range(ny):
        for i in range(nx):
            row = i + nx*j
            for a, t, n, stride in [('x', i, nx, 1), ('y', j, ny, nx)]:
                h = s['L'+a]/(n-1)
                k, v = s['kappa_'+a], s['vel_'+a]
                if t in (0, n-1):
                    rates = [(1 if t == 0 else -1, k[t]/h**2)]
                else:
                    rates = [(-1, (k[t]+k[t-1])/(2*h**2)+max(v[t],0)/h),
                             (1, (k[t]+k[t+1])/(2*h**2)+max(-v[t],0)/h)]
                for step, rate in rates:
                    A[row, row+stride*step] -= rate
                    A[row, row] += rate
    np.testing.assert_allclose(A.sum(axis=1), s['V_reaction_2d'].ravel(order='F'), atol=3e-14)
    rhs = np.column_stack([u.ravel(order='F') for u in s['source_modes']])
    fields = [np.linalg.solve(A+mu*np.eye(nx*ny), rhs) for mu in s['mu_shifts']]
    i, j = s['sensor_ij'].T
    S = np.concatenate([u[i+nx*j] for u in fields], axis=0)
    ref = s['sensor_calibration_sweeps']['ref_rosette']
    raw = s['sensor_calibration_sweeps']['raw_rosette']
    C = np.linalg.lstsq(np.c_[ref, np.ones(len(ref))], raw, rcond=None)[0]
    def calibrate(raw):
        return ((raw.reshape(-1,4)-C[-1]) @ np.linalg.inv(C[:-1])).ravel()
    for p in item.get('campaign_diagnostics') or []:
        r = (calibrate(p['y_pulse_raw'])-S@p['w_pulse']).reshape(len(fields),-1)
        wall = (i == 0)|(i == nx-1)|(j == 0)|(j == ny-1)
        for k, e in [('pulse',r), ('wall',r[:,wall]), ('interior',r[:,~wall])]:
            worst[k] = max(worst[k], np.linalg.norm(e))
    Z = s['Z_nuisance']
    P = np.eye(len(S))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
    D, y = P@S, P@calibrate(s['y_obs_raw'])
    cap, ridge = s['u_box'], s['source_ridge']
    for budget in (s['total_budget'], 0., cap, 0.5*cap*rhs.shape[1], cap*rhs.shape[1]):
        out = m['solve_batch'](dict(s, total_budget=budget))
        w, cert = out['weights'], out['kkt_certificate']
        np.testing.assert_array_equal(w, cert['weights'])
        np.testing.assert_allclose(w, w.clip(0,cap), atol=1e-12)
        res = D@w-y
        g = D.T@res+ridge*w
        nu = cert['nu_sum']
        lo, hi = [np.asarray(cert[k]) for k in ('lam_lower','lam_upper')]
        np.testing.assert_allclose(g-nu, lo-hi, atol=1e-12)
        assert min(lo.min(),hi.min()) >= 0
        obj = 0.5*(res@res+ridge*(w@w))
        dual = obj-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
        np.testing.assert_allclose([cert['primal_objective'],cert['dual_lower_bound']], [obj,dual], atol=1e-12, rtol=0)
        expected = (fields[-1]@w).reshape((nx,ny),order='F')
        metrics = dict(terminal=np.max(np.abs(expected-out['terminal_field'])), budget=abs(w.sum()-budget), gap=abs(obj-dual), complementarity=max(np.max(w*lo),np.max((cap-w)*hi)))
        for name, error in metrics.items():
            assert error < 1e-9, (name,budget,error)
            worst[name] = max(worst[name], error)
        print('budget',budget,'active lower',np.sum(w<1e-9),'active upper',np.sum(w>cap-1e-9))
assert worst['pulse'] < 3e-5, worst
print('worst:',worst)
