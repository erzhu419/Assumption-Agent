import time
import numpy as np
from tomo_public_check import ray_terms, public_cases
from tomography_solver import invert_plume_and_clocks

base = public_cases()[0]['survey']
centers = [(1,38),(39,41),(17,5),(24,75),(2,7),(37,72),(20,40),(11,57),(31,26)]
metrics = []
for k, xy in enumerate(centers):
    group = []
    for seed in range(6):
        rng = np.random.default_rng(9271+seed+20*k)
        s = dict(base)
        if seed >= 3:
            s['z_src_m'] = np.asarray(base['z_src_m']) + rng.uniform(-1.2,1.2,15)
            s['z_rcv_m'] = np.asarray(base['z_rcv_m']) + rng.uniform(-1.2,1.2,16)
        b = rng.uniform(-1,1,len(s['z_rcv_m']))
        b -= b.mean()
        b *= 3.98 / np.max(np.abs(b))
        s0 = rng.uniform(.37,.44)
        eps = rng.uniform(.11,.21)
        strength = [.032,.052,.085,.108][seed%4]
        terms = ray_terms(s, xy)
        y = terms @ [s0,-s0*eps,strength] + b + rng.normal(0,.04,terms.shape[:2])
        if seed == 5:
            col = int(np.argmax(terms[...,2].sum(axis=0)))
            bad = rng.choice(y.shape[0],5,replace=False)*y.shape[1]+col
            bad = np.r_[bad,rng.choice(np.setdiff1d(np.arange(y.size),bad),5,replace=False)]
        else:
            bad = rng.choice(y.size,round(y.size*.04),replace=False)
        y.flat[bad] += 6*rng.choice([-1,1],len(bad)) + rng.normal(0,.49,len(bad))
        s['T_obs_s'] = y / 1e3
        t = time.perf_counter()
        out = invert_plume_and_clocks(s)
        elapsed = time.perf_counter()-t
        p = np.asarray(out['plume_xz_m'])
        c = np.asarray(out['rcv_clock_bias_s'])
        assert abs(c.sum()) < 1e-12 and max(abs(c)) <= .004 + 1e-12
        m = [np.linalg.norm(p-xy), np.sqrt(np.mean((c*1000-b)**2))*1000, elapsed]
        group.append(m)
        if m[0] > 1.8 or m[1] > 250:
            print('miss',xy,seed,'estimated',p,'center m, clock us, seconds',m,flush=True)
    print(xy, 'max [center m, clock us, seconds]',np.max(group,axis=0),flush=True)
    metrics.extend(group)
print('overall max',np.max(metrics,axis=0))
