import numpy as np
from ert_solver import invert_ert_anomaly
from ert_robust_init import invert_ert_anomaly as bounded

rng = np.random.default_rng(421)
L, D, a = 36., 72., 6.
for label in ('irregular depths', 'strong wall anomaly'):
    errors = [[], []]
    pathological = [[], []]
    for rep in range(56):
        t = np.linspace(6., 66., 14)
        r = np.linspace(5., 67., 15)
        if label == 'irregular depths':
            t = np.sort(t + rng.uniform(-2., 2., t.size))
            r = np.sort(r + rng.uniform(-2., 2., r.size))
            x, z = rng.uniform(.25, .75)*L, rng.uniform(.2, .8)*D
            amp = rng.uniform(6., 9.)
        else:
            x, z = rng.uniform(0., .14)*L, rng.uniform(.12, .88)*D
            if rep % 2:
                x = L - x
            amp = rng.uniform(25., 90.)
        d = r[None, :] - t[:, None]
        R = np.hypot(L, d)
        k = 1. / np.sqrt((x*x+(z-t[:, None])**2+a*a)*((L-x)**2+(z-r[None, :])**2+a*a))
        tx, rx = rng.normal(0., .008, t.size), rng.normal(0., .008, r.size)
        tx -= tx.mean(); rx -= rx.mean()
        v = 1.9/R - .3*d*d/R**3 + tx[:, None] + rx + amp*k
        v += rng.normal(0., .00023, v.shape)
        idx = rng.choice(v.size, int(.05*v.size), replace=False)
        v.flat[idx] += rng.choice([-.06, .06], idx.size)
        s = dict(L_m=L, D_m=D, z_tx_m=t, z_rx_m=r, V_obs=v)
        for j, fn in enumerate((invert_ert_anomaly, bounded)):
            p = fn(s)
            row = (np.linalg.norm(np.array(p['anomaly_xz_m'])-[x,z]), 1e3*np.sqrt(np.mean((np.array(p['rx_contact_offset_v'])-rx)**2)))
            errors[j].append(row)
            if row[0] > 2. or row[1] > 1.2:
                pathological[j].append((rep, round(amp, 2), np.round(row, 3).tolist()))
    for method, rows, bad in zip(('baseline', 'bounded'), errors, pathological):
        e = np.asarray(rows)
        print(label, method, 'p90', np.quantile(e,.9,axis=0), 'max', e.max(axis=0), 'fail', np.sum(e>[2.,1.2],axis=0), 'worst examples', sorted(bad, key=lambda r:max(r[2][0]/2,r[2][1]/1.2), reverse=True)[:4])
