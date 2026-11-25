import pickle, sys, time
import numpy as np
sys.path.insert(0,sys.argv[1])
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

import os
from time import perf_counter
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
        sizes = rng.uniform(.0055,.0065,skips.size)
        if os.environ.get('TOMO_CYCLE') == 'fixed': sizes[:] = .006
        picks.flat[skips] += rng.choice([-1,1],skips.size)*sizes
        s = dict(W_m=w,H_m=h,z_src_m=zs,z_rcv_m=zr,T_obs_s=picks,R_plume_m=5.5)
        if (int(os.environ['TOMO_SEED']), regime, k) not in ((9361,'near-depth-edge',30),(18417,'strong-contrast',17),(9361,'weak-contrast',12)):
            continue
        from tomography_solver import _geometry, _invert, _kernel
        from scipy.optimize import least_squares
        _,_,r,z0,ll,sn,cs,C,A,y = _geometry(s)
        truth = np.r_[x,z,amp*r*1000,slowness*w*1000,slowness*eps*w*1000,C.T@(1000*c)]
        started = perf_counter()
        p,_,_,keep = _invert(s)
        elapsed = perf_counter() - started
        print("default-thread inversion seconds", round(elapsed, 3))
        assert elapsed < 15
        if regime != "weak-contrast":
            assert np.linalg.norm(p[:2]-truth[:2]) <= 1.8
        clean = np.ones(y.size,dtype=bool); clean[skips] = False
        def residual(t):
            return t[2]*_kernel(*t[:2],r,z0,ll,sn,cs)+A@t[3:]-y
        def jac(t):
            g,gx,gz=_kernel(*t[:2],r,z0,ll,sn,cs,True)
            return np.column_stack((t[2]*gx,t[2]*gz,g,A))
        def solve(t, mask):
            lower=np.r_[0,0,0,np.full(A.shape[1],-np.inf)]
            upper=np.r_[w,h,np.full(1+A.shape[1],np.inf)]
            return least_squares(lambda q:residual(q)[mask],t,jac=lambda q:jac(q)[mask],bounds=(lower,upper),x_scale='jac').x
        ref=solve(truth,clean); fit=solve(p,clean)
        print(regime,k,'skip mismatches',np.count_nonzero(clean!=keep),'usual pick RMS ms',round(np.std(residual(truth)[clean]),4))
        for label,t in [('truth',truth),('deployed',p),('oracle from truth',ref),('oracle from deployed',fit)]:
            print(label,'xz/amp',np.round(t[:3],4),'SSE',round(np.linalg.norm(residual(t)[clean])**2,6),'error m',round(np.linalg.norm(t[:2]-truth[:2]),4))
