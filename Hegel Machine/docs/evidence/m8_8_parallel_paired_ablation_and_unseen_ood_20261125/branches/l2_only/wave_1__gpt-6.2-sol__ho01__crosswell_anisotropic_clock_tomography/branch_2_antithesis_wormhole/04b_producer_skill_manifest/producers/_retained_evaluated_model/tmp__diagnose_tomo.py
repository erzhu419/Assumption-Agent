import pickle
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from tomography_solver import _fit, invert_plume_and_clocks

samples = pickle.load(open('/holdout_task/public_surveys.pkl', 'rb'))

def errors(xz, clocks, gt_xz, gt_clocks):
    return np.linalg.norm(np.subtract(xz, gt_xz)), np.sqrt(np.mean(np.subtract(clocks, gt_clocks)**2))

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
        if regime in ('near-well', 'end-depth', 'weak'):
            rays, p, residual = _fit(survey)
            good = abs(residual) < 2
            a = rays.design[good]
            g, gx, gz = [v[good] for v in rays.plume(*p[:2], derivative=True)]
            X = np.column_stack((p[2]*gx,p[2]*gz,g,a))
            sigma = np.linalg.norm(residual[good])/np.sqrt(good.sum()-len(p))
            cov = sigma**2*np.linalg.pinv(X.T @ X, hermitian=True)
            gt = np.array([x,z,1000*amp])
            g0 = rays.plume(x,z)[good]
            Q = np.column_stack((g0,a))
            target = t.ravel()[good]*1000
            er = target - Q @ np.linalg.lstsq(Q,target,rcond=None)[0]
            delta = (er@er - residual[good]@residual[good])/sigma**2
            if np.linalg.norm(p[:2]-gt[:2]) > 1 or sigma > .065:
                print(regime,j,'truth',gt,'fit',p[:3],'SE',np.sqrt(np.diag(cov)[:3]),'profile delta chi^2',round(delta,3),'residual noise ms',round(sigma,4),'max clock ms',round(max(abs(rays.clocks@p[5:])),4),'mismasked',np.sum(np.flatnonzero(~good)[:,None]==idx[None,:])!=len(idx) or (~good).sum()!=len(idx))
        start = time.monotonic(); result = invert_plume_and_clocks(survey); elapsed = time.monotonic() - start
        a,b = errors(result['plume_xz_m'],result['rcv_clock_bias_s'],[x,z],clock)
        assert abs(sum(result['rcv_clock_bias_s'])) < 1e-12 and max(abs(np.asarray(result['rcv_clock_bias_s']))) <= .004 + 1e-14
        out.append((regime,j,a,b,elapsed,(x,z),result['plume_xz_m']))
for regime in dict.fromkeys(r[0] for r in out):
    rows = [r for r in out if r[0]==regime]
    print(regime,'worst location row',max(rows,key=lambda r:r[2]),'worst clock RMSE s',max(r[3] for r in rows),'max time s',max(r[4] for r in rows),'failed',sum(r[2]>1.8 or r[3]>.00025 for r in rows))
