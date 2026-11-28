import importlib.util
import itertools
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
items = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))
worst = dict(transport=0., field=0., feasibility=0., stationarity=0., complementarity=0., gap=0., oracle_weight=0.)
for item in items:
    st = item['study']
    nx, ny = int(st['nx']), int(st['ny'])
    modes = np.asarray(st['source_modes'])
    rhs = np.column_stack([a.flatten(order='F') for a in modes])
    ij = np.asarray(st['sensor_ij'])
    shifts = st['mu_shifts']
    ident = np.eye(nx * ny)
    ds = item.get('campaign_diagnostics')
    sw = st['sensor_calibration_sweeps']
    x, raw = sw['ref_rosette'], sw['raw_rosette']
    q = sw['single_bank_sweeps']
    gain, bias = np.zeros(4), np.zeros(4)
    for channel in range(4):
        mask = q['bank_id'] == channel
        xx = np.concatenate((x[:, channel], q['ref_signal'][mask]))
        yy = np.concatenate((raw[:, channel], q['raw_readout'][mask]))
        xc = xx - xx.mean()
        gain[channel] = (xc @ (yy - yy.mean())) / (xc @ xc)
        bias[channel] = yy.mean() - gain[channel] * xx.mean()
    def calibrate(y):
        return ((np.asarray(y).reshape(-1, 4) - bias) / gain).ravel()
    np.testing.assert_allclose(mod._calibrate(st), calibrate(st['y_obs_raw']), rtol=0, atol=1e-12)
    def reference(closed):
        axes = []
        for axis, n, shut in zip('xy', (nx, ny), closed):
            h = st['L' + axis] / (n - 1)
            k, v = st['kappa_' + axis], st['vel_' + axis]
            a = np.zeros((n, n))
            for i in range(n):
                if 0 < i < n-1:
                    left = (k[i] + k[i-1]) / (2*h*h)
                    right = (k[i] + k[i+1]) / (2*h*h)
                    a[i, i-1] = -left - max(v[i], 0)/h
                    a[i, i+1] = -right - max(-v[i], 0)/h
                    a[i, i] = left + right + abs(v[i])/h
                else:
                    a[i, 1 if i == 0 else n-2] = -k[i]/h**2
                    a[i, i] = k[i]/h**2 + (0 if shut else st['robin_' + axis]/h)
            axes.append(a)
        L = np.kron(np.eye(ny), axes[0]) + np.kron(axes[1], np.eye(nx)) + np.diag(st['V_reaction_2d'].flatten(order='F'))
        fields = [np.linalg.solve(L + mu*ident, rhs) for mu in shifts]
        return np.concatenate([f[ij[:,0] + nx*ij[:,1]] for f in fields]), fields[-1]
    S, F = reference((True, True))
    observed_S, _ = mod._transport(st)
    worst['transport'] = max(worst['transport'], np.max(abs(observed_S - S)))
    if ds:
        Y = np.column_stack([calibrate(d['y_pulse_raw']) for d in ds])
        W = np.column_stack([d['w_pulse'] for d in ds])
        residuals = {}
        for walls in itertools.product((False, True), repeat=2):
            candidate = S if all(walls) else reference(walls)[0]
            residuals[walls] = (candidate @ W - Y).reshape(len(shifts), len(ij), len(ds))
        edge = (ij[:,0] == 0) | (ij[:,0] == nx-1) | (ij[:,1] == 0) | (ij[:,1] == ny-1)
        scores = []
        for p in range(len(ds)):
            chosen = min(residuals, key=lambda walls: np.mean(residuals[walls][:,:,np.arange(len(ds)) != p]**2))
            assert chosen == (True, True), chosen
            r = residuals[chosen][:,:,p]
            scores.extend(np.sqrt(np.mean(r[:,mask]**2, axis=1)).max() for mask in (edge, ~edge) if mask.any())
        print('Worst held-out pulse × shift × wall/interior RMS:', max(scores))
        assert max(scores) < 5e-6
    Z = st['Z_nuisance']
    P = np.eye(len(S)) - Z @ np.linalg.solve(Z.T @ Z + st['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
    A = P @ S
    for budget in (st['total_budget'], 0., st['u_box'], len(modes)*st['u_box'], (len(modes)-0.1)*st['u_box']):
        out = mod.solve_batch(dict(st, total_budget=budget))
        w, cert = out['weights'], out['kkt_certificate']
        cap, ridge = st['u_box'], st['source_ridge']
        r = P @ calibrate(st['y_obs_raw'])
        H, c = A.T @ A + ridge*np.eye(len(w)), -A.T @ r
        J = lambda v: (np.linalg.norm(A @ v - r)**2 + ridge*(v @ v))/2
        g, nu = H @ w + c, cert['nu_sum']
        lower, upper = cert['lam_lower'], cert['lam_upper']
        dual = J(w) - g @ w + nu*budget + cap*np.minimum(0., g-nu).sum()
        assert np.allclose(cert['weights'], w) and min(lower.min(), upper.min()) >= 0
        assert abs(cert['primal_objective'] - J(w)) < 1e-10
        assert abs(cert['dual_lower_bound'] - dual) < 1e-10
        worst['field'] = max(worst['field'], np.max(abs(out['terminal_field'] - (F @ w).reshape((nx, ny), order='F'))))
        worst['feasibility'] = max(worst['feasibility'], abs(w.sum()-budget), -w.min(), w.max()-cap)
        worst['stationarity'] = max(worst['stationarity'], np.max(abs(g-nu-lower+upper)))
        worst['complementarity'] = max(worst['complementarity'], np.max(abs(w*lower)), np.max(abs((cap-w)*upper)))
        worst['gap'] = max(worst['gap'], abs(J(w)-dual))
        if budget == st['total_budget'] and len(w) <= 9:
            best = (np.inf, None)
            for labels in itertools.product((-1, 0, 1), repeat=len(w)):
                labels = np.asarray(labels)
                free = np.flatnonzero(labels == 0)
                v = cap*(labels == 1).astype(float)
                need = budget - v.sum()
                if need < -1e-12 or need > cap*len(free)+1e-12:
                    continue
                if len(free):
                    one = np.ones((len(free), 1))
                    K = np.block([[H[np.ix_(free, free)], one], [one.T, np.zeros((1,1))]])
                    v[free] = np.linalg.solve(K, np.r_[-(H @ v + c)[free], need])[:-1]
                if v.min() >= -1e-12 and v.max() <= cap+1e-12:
                    obj = J(v)
                    if obj < best[0]:
                        best = (obj, v)
            worst['oracle_weight'] = max(worst['oracle_weight'], np.max(abs(w - best[1])))
        print('budget', budget, 'lower/free/upper', np.sum(w < 1e-10), np.sum((w >= 1e-10) & (w <= cap-1e-10)), np.sum(w > cap-1e-10))
print('Worst verification errors:', worst)
assert max(worst.values()) < 1e-9, worst
