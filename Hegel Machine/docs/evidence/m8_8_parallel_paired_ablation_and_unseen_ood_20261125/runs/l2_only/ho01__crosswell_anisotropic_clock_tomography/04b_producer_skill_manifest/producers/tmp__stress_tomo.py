import pickle
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from tomography_solver import _fit, invert_plume_and_clocks

samples = pickle.load(open('/holdout_task/public_surveys.pkl', 'rb'))

def errors(xz, clocks, gt_xz, gt_clocks):
    return np.linalg.norm(np.subtract(xz, gt_xz)), np.sqrt(np.mean(np.subtract(clocks, gt_clocks)**2))

for i, item in enumerate(samples):
    survey, gt = item['survey'], item['gt']
    t, clean_predictions = np.asarray(survey['T_obs_s']), []
    worst = [[], []]
    for j in range(t.shape[0]):
        keep = np.arange(t.shape[0]) != j
        subset = dict(survey, z_src_m=survey['z_src_m'][keep], T_obs_s=t[keep])
        rays, p, _ = _fit(subset)
        clocks = (rays.clocks @ p[5:]) / 1000
        worst[0].append(errors(p[:2], clocks, gt['plume_xz_m'], gt['rcv_clock_bias_s']))
        from tomography_solver import _Rays
        held = _Rays(survey['W_m'], survey['z_src_m'][j:j+1], survey['z_rcv_m'], survey['R_plume_m'])
        pred = held.design @ p[3:] + p[2] * held.plume(*p[:2])
        res = t[j] * 1000 - pred
        clean_predictions.extend(res[abs(res) < 2.])
    for j in range(t.shape[1]):
        keep = np.arange(t.shape[1]) != j
        subset = dict(survey, z_rcv_m=survey['z_rcv_m'][keep], T_obs_s=t[:, keep])
        r = invert_plume_and_clocks(subset)
        clocks = np.asarray(gt['rcv_clock_bias_s'])[keep]
        worst[1].append(errors(r['plume_xz_m'], r['rcv_clock_bias_s'], gt['plume_xz_m'], clocks - clocks.mean()))
    print('public', i, 'worst leave-source-out', np.max(worst[0], axis=0), 'worst leave-receiver-out', np.max(worst[1], axis=0), 'held-source non-skip RMS ms', np.sqrt(np.mean(np.square(clean_predictions))))

rng = np.random.default_rng(728)
nodes, weights = leggauss(64)
u = (nodes + 1) / 2
out = []
for regime in ('public-like', 'wider', 'near-well', 'end-depth', 'weak'):
    for j in range(12):
        w, h = (60., 100.) if regime == 'wider' else (40., 80.)
        n, m = (12, 13) if regime == 'wider' else (15, 16)
        src = np.linspace(.075*h, .925*h, n) + rng.uniform(-.6,.6,n)
        rcv = np.linspace(.0625*h, .9375*h, m) + rng.uniform(-.6,.6,m)
        x, z = rng.uniform(.15,.85)*w, rng.uniform(.17,.83)*h
        if regime == 'near-well': x = 2. if j % 2 else w - 2.
        if regime == 'end-depth': z = 3. if j % 2 else h - 3.
        amp = 0.000022 if regime == 'weak' else rng.uniform(.00005,.00009)
        dz = rcv[None, :] - src[:, None]; dl = np.hypot(w, dz)
        distance_sq = (w*u-x)**2 + (src[:,None,None] + dz[:,:,None]*u-z)**2
        plume = dl * ((np.exp(-distance_sq/(2*5.5**2)) @ weights) / 2)
        s0, eps = rng.uniform(.00034,.00046), rng.uniform(.08,.23)
        clock = rng.uniform(-1,1,m); clock -= clock.mean(); clock *= (0.004 if j % 2 else 0.0032)/max(abs(clock))
        t = dl*s0*(1-eps*(dz/dl)**2) + amp*plume + clock + rng.normal(0,0.000045,dl.shape)
        idx = rng.choice(t.size, round(.04*t.size), replace=False)
        t.flat[idx] += rng.choice([-0.006,0.006], len(idx))
        survey = dict(W_m=w, H_m=h, z_src_m=src, z_rcv_m=rcv, T_obs_s=t, R_plume_m=5.5)
        start = time.monotonic(); result = invert_plume_and_clocks(survey); elapsed = time.monotonic() - start
        a,b = errors(result['plume_xz_m'],result['rcv_clock_bias_s'],[x,z],clock)
        assert abs(sum(result['rcv_clock_bias_s'])) < 1e-12 and max(abs(np.asarray(result['rcv_clock_bias_s']))) <= .004 + 1e-14
        out.append((regime,j,a,b,elapsed,(x,z),result['plume_xz_m']))
for regime in dict.fromkeys(r[0] for r in out):
    rows = [r for r in out if r[0]==regime]
    print(regime,'worst location row',max(rows,key=lambda r:r[2]),'worst clock RMSE s',max(r[3] for r in rows),'max time s',max(r[4] for r in rows),'failed',sum(r[2]>1.8 or r[3]>.00025 for r in rows))
