import pickle
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from tomography_solver import _invert, invert_plume_and_clocks


def metrics(xy, clock, truth_xy, truth_clock):
    return np.linalg.norm(np.asarray(xy)-truth_xy), 1e6*np.sqrt(np.mean((np.asarray(clock)-truth_clock)**2))


def run(s):
    start = time.perf_counter()
    out = invert_plume_and_clocks(s)
    dt = time.perf_counter()-start
    assert set(out) == {'plume_xz_m', 'rcv_clock_bias_s'}
    xy, c = out['plume_xz_m'], out['rcv_clock_bias_s']
    assert isinstance(xy, list) and len(xy) == 2 and isinstance(c, list) and len(c) == len(s['z_rcv_m'])
    assert all(type(x) is float and np.isfinite(x) for x in xy+c)
    assert 0 <= xy[0] <= s['W_m'] and 0 <= xy[1] <= s['H_m']
    assert abs(sum(c)) < 1e-12 and max(map(abs,c)) <= .004 + 1e-15 and dt < 15
    return xy,c,dt

nodes, weights = leggauss(64)
t = (nodes[:,None,None]+1.)/2.
rng = np.random.default_rng(625309)
for label, n in [('strong-plume',80),('strong-near-wall',80),('strong-wide-well',80)]:
    records, misses = [], []
    for k in range(n):
        w,h,ns,nr = (rng.uniform(28,65),rng.uniform(55,105),int(rng.integers(10,23)),int(rng.integers(10,23))) if label=='strong-wide-well' else (40.,80.,15,16)
        zs = np.linspace(.075*h,.925*h,ns)
        zr = np.linspace(.0625*h,.9375*h,nr)
        if label=='strong-wide-well':
            zs += rng.uniform(-.8,.8,ns)
            zr += rng.uniform(-.8,.8,nr)
        if label=='strong-near-wall':
            x = rng.uniform(0,4) if k%2 else w-rng.uniform(0,4)
            z = rng.uniform(.18*h,.82*h)
        else:
            x,z = rng.uniform(.09*w,.91*w),rng.uniform(.13*h,.87*h)
        dz = zr[None,:]-zs[:,None]
        length = np.hypot(w,dz)
        dd = (t*w-x)**2 + (zs[None,:,None]+t*dz-z)**2
        plume = length*np.einsum('q,qij->ij',weights/2,np.exp(-.5*dd/5.5**2))
        s0,eps = rng.uniform(.00030,.00055),rng.uniform(.02,.27)
        amp = rng.uniform(.00010,.00060)
        c = rng.uniform(-1,1,nr)
        c -= c.mean()
        c *= .004/np.max(np.abs(c))
        noise = rng.uniform(.000025,.000060)
        y = length*s0*(1-eps*(dz/length)**2)+amp*plume+c+rng.normal(0,noise,length.shape)
        kslip = round(y.size*.04)
        ids = rng.choice(y.size, kslip, replace=False)
        if label=='clustered-slips':
            j = int(rng.integers(nr))
            row = rng.choice(ns,6,replace=False)
            outside = np.flatnonzero(np.arange(y.size)%nr!=j)
            ids = np.r_[row*nr+j,rng.choice(outside,kslip-6,replace=False)]
        signs = rng.choice([-1.,1.],kslip)
        if label=='clustered-slips':
            signs[:6] = signs[0]
        y.flat[ids] += signs*rng.uniform(.0055,.006)
        xy,ch,dt = run(dict(W_m=w,H_m=h,z_src_m=zs,z_rcv_m=zr,T_obs_s=y,R_plume_m=5.5))
        if np.linalg.norm(np.asarray(xy)-[x,z]) > 1.8:
            _, _, pred = _invert(dict(W_m=w,H_m=h,z_src_m=zs,z_rcv_m=zr,T_obs_s=y,R_plume_m=5.5))
            flagged = np.abs(y.ravel()-pred) >= .0027
            actual = np.zeros(y.size,bool)
            actual[ids] = True
            if len(misses) < 5:
                print('miss',label,k,'center error m', round(float(np.linalg.norm(np.asarray(xy)-[x,z])),2), 'peak delay ms',round(1000*amp*plume.max(),2),'inliers wrongly removed',np.sum(flagged & ~actual),'slips retained',np.sum(actual & ~flagged))
            misses.append(k)
        records.append([*metrics(xy,ch,np.array([x,z]),c),dt])
    a = np.asarray(records)
    print(label,'worst center m / clock us / runtime s', np.max(a,axis=0),'number over target',np.sum(a[:,:2]>[1.8,250],axis=0),'worst case indices',np.argmax(a,axis=0))
