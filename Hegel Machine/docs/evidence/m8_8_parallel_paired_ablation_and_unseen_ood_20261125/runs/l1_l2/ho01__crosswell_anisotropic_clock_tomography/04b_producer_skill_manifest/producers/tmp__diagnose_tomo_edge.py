import pickle, sys, time
import numpy as np
sys.path.insert(0,'/holdout_task')
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

rng = np.random.default_rng(2409)
u, weights = np.polynomial.legendre.leggauss(64)
u, weights = (u+1)/2, weights/2
for regime in ('interior','near-well','near-depth-edge','sparser-noisier'):
    scores = []
    for k in range(32):
        w,h = rng.uniform(30,55),rng.uniform(65,95)
        n,m = (10,11) if regime == 'sparser-noisier' else (15,16)
        zs = np.linspace(5,h-6,n) + rng.uniform(-1,1,n)
        zr = np.linspace(5,h-5,m) + rng.uniform(-1,1,m)
        x,z = rng.uniform(.2,.8)*w,rng.uniform(.2,.8)*h
        if regime == 'near-well': x = rng.uniform(1,6) if k%2 else w-rng.uniform(1,6)
        if regime == 'near-depth-edge': z = rng.uniform(2,10) if k%2 else h-rng.uniform(2,10)
        dz = zr[None,:]-zs[:,None]; ll = np.hypot(w,dz)
        d2 = (w*u-x)**2 + (zs[:,None,None]+dz[:,:,None]*u-z)**2
        g = ll * (np.exp(-d2/(2*5.5**2)) @ weights)
        slowness, eps, amp = rng.uniform(.00033,.0005),rng.uniform(.04,.23),rng.uniform(.000042,.000085)
        c = rng.uniform(-.004,.004,m); c -= c.mean(); c *= min(1,.004/np.abs(c).max())
        picks = ll*slowness*(1-eps*(dz/ll)**2) + amp*g + c + rng.normal(0,.000075 if regime=='sparser-noisier' else .000035,(n,m))
        skips = rng.choice(n*m,round(.04*n*m),replace=False)
        picks.flat[skips] += rng.choice([-1,1],skips.size)*rng.uniform(.0055,.0065,skips.size)
        s = dict(W_m=w,H_m=h,z_src_m=zs,z_rcv_m=zr,T_obs_s=picks,R_plume_m=5.5)
        score = measure(s,np.array([x,z]),c)
        scores.append(score)
        if score[0]>1.8 or score[1]>.25:
            print('stress failure',regime,k,'true center',np.round([x,z],2),'extent',np.round([w,h],2),'errors',score)

            from scipy.optimize import least_squares
            from tomography_solver import _invert, _geometry, _kernel
            pfit, _, _, kept = _invert(s)
            _, _, radius, z0, length, sn, cs, contrast, A, y = _geometry(s)
            true = np.r_[x,z,amp*radius*1000,slowness*w*1000,slowness*eps*w*1000,contrast.T@(1000*c)]
            clean = np.ones(y.size, dtype=bool); clean[skips] = False
            def residual(p):
                return p[2]*_kernel(*p[:2],radius,z0,length,sn,cs)+A@p[3:]-y
            def jac(p):
                g,gx,gz = _kernel(*p[:2],radius,z0,length,sn,cs,True)
                return np.column_stack((p[2]*gx,p[2]*gz,g,A))
            lower = np.r_[0.,0.,0.,np.full(A.shape[1],-np.inf)]
            upper = np.r_[w,h,np.full(A.shape[1]+1,np.inf)]
            oracle = least_squares(lambda p:residual(p)[clean],true,jac=lambda p:jac(p)[clean],bounds=(lower,upper),x_scale='jac').x
            sigma = np.linalg.norm(residual(oracle)[clean])/np.sqrt(clean.sum()-len(oracle))
            J = jac(true)[clean]
            cov = sigma**2*np.linalg.pinv(J.T@J)
            print('  fit and oracle x,z,amp',np.round(pfit[:3],4),np.round(oracle[:3],4),'true amp',round(true[2],4),'mask mismatches',np.sum(kept!=clean))
            print('  local sd x,z,amp',np.round(np.sqrt(np.diag(cov)[:3]),4),'depth-amplitude corr',round(cov[1,2]/np.sqrt(cov[1,1]*cov[2,2]),4),'true and fit clean SSE',round(np.sum(residual(true)[clean]**2),5),round(np.sum(residual(oracle)[clean]**2),5))

    print('stress',regime,'max [m, ms, s]',np.max(scores,axis=0))
