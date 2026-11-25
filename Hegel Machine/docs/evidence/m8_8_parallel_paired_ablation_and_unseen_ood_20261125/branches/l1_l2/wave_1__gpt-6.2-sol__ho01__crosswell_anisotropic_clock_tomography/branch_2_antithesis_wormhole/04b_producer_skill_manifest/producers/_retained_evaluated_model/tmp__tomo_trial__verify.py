import pickle, sys, time
import numpy as np
sys.path.insert(0,'/tmp/tomo_trial')
from tomography_solver import invert_plume_and_clocks

cases = pickle.load(open('/holdout_task/public_surveys.pkl','rb'))

def measure(s, center, clocks):
    t = time.perf_counter()
    result = invert_plume_and_clocks(s)
    elapsed = time.perf_counter() - t
    c = np.array(result['rcv_clock_bias_s'])
    xz = np.array(result['plume_xz_m'])
    assert set(result) == {'plume_xz_m','rcv_clock_bias_s'}
    assert xz.shape == (2,) and c.shape == clocks.shape
    assert np.all(xz >= 0) and np.all(xz <= [s['W_m'],s['H_m']])
    assert abs(c.sum()) < 1e-12 and np.max(np.abs(c)) <= .004 + 1e-12
    return np.array([np.linalg.norm(xz-center),1e3*np.sqrt(np.mean((c-clocks)**2)),elapsed])

if '--timing' in sys.argv:
    for case in cases:
        print('default env [m, ms, s]',measure(case['survey'],case['gt']['plume_xz_m'],case['gt']['rcv_clock_bias_s']))
    sys.exit()

for k, case in enumerate(cases):
    s, gt = case['survey'],case['gt']
    xz, c = gt['plume_xz_m'],np.array(gt['rcv_clock_bias_s'])
    scores = [measure(s,xz,c)]
    for i in range(len(s['z_src_m'])):
        t = s.copy(); t['z_src_m'] = np.delete(s['z_src_m'],i); t['T_obs_s'] = np.delete(s['T_obs_s'],i,axis=0)
        scores.append(measure(t,xz,c))
    for j in range(len(s['z_rcv_m'])):
        t = s.copy(); t['z_rcv_m'] = np.delete(s['z_rcv_m'],j); t['T_obs_s'] = np.delete(s['T_obs_s'],j,axis=1)
        # Re-express the remaining clocks in the reduced system's zero-sum gauge.
        cr = np.delete(c,j)
        t['T_obs_s'] = t['T_obs_s'] - cr.mean()
        scores.append(measure(t,xz,cr-cr.mean()))
    print('public',k,'full',scores[0],'source drop max',np.max(scores[1:16],axis=0),'receiver drop max',np.max(scores[16:],axis=0))

import os
rng = np.random.default_rng(int(os.environ.get('TOMO_SEED', '2409')))
u, weights = np.polynomial.legendre.leggauss(64)
u, weights = (u+1)/2, weights/2
for regime in ('interior','near-well','near-depth-edge','sparser-noisier','weak-contrast','strong-contrast'):
    scores = []
    for k in range(32):
        w,h = rng.uniform(30,55),rng.uniform(65,95)
        n,m = (10,11) if regime == 'sparser-noisier' else (15,16)
        zs = np.linspace(5,h-6,n) + rng.uniform(-1,1,n)
        zr = np.linspace(5,h-5,m) + rng.uniform(-1,1,m)
        x,z = rng.uniform(.2,.8)*w,rng.uniform(.2,.8)*h
        if regime == 'near-well': x = rng.uniform(1,6) if k%2 else w-rng.uniform(1,6)
        if regime in ('near-depth-edge', 'weak-contrast', 'strong-contrast'): z = rng.uniform(2,10) if k%2 else h-rng.uniform(2,10)
        dz = zr[None,:]-zs[:,None]; ll = np.hypot(w,dz)
        d2 = (w*u-x)**2 + (zs[:,None,None]+dz[:,:,None]*u-z)**2
        g = ll * (np.exp(-d2/(2*5.5**2)) @ weights)
        slowness, eps, amp = rng.uniform(.00033,.0005),rng.uniform(.04,.23),rng.uniform(.000042,.000085)
        if regime == 'weak-contrast': amp *= .6
        if regime == 'strong-contrast': amp *= 1.6
        c = rng.uniform(-.004,.004,m); c -= c.mean(); c *= min(1,.004/np.abs(c).max())
        picks = ll*slowness*(1-eps*(dz/ll)**2) + amp*g + c + rng.normal(0,.000075 if regime=='sparser-noisier' else .000035,(n,m))
        skips = rng.choice(n*m,round(.04*n*m),replace=False)
        picks.flat[skips] += rng.choice([-1,1],skips.size)*rng.uniform(.0055,.0065,skips.size)
        s = dict(W_m=w,H_m=h,z_src_m=zs,z_rcv_m=zr,T_obs_s=picks,R_plume_m=5.5)
        score = measure(s,np.array([x,z]),c)
        scores.append(score)
        if score[0]>1.8 or score[1]>.25:
            print('stress failure',regime,k,'true center',np.round([x,z],2),'extent',np.round([w,h],2),'errors',score)
    print('stress',regime,'max [m, ms, s]',np.max(scores,axis=0),'p90 center',np.quantile(np.asarray(scores)[:,0],.9),'center failures',sum(r[0]>1.8 for r in scores))
