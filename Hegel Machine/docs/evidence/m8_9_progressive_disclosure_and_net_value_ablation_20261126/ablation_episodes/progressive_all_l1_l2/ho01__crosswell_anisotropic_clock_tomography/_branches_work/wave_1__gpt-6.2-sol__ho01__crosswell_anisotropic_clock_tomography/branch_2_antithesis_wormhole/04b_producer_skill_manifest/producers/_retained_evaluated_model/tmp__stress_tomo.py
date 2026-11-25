import pickle, sys, time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
sys.path.insert(0, '/holdout_task')
from tomography_solver import invert_plume_and_clocks

rng = np.random.default_rng(501)

def check(s, xy, clocks):
    start = time.perf_counter()
    out = invert_plume_and_clocks(s)
    dt = time.perf_counter() - start
    assert set(out) == {'plume_xz_m', 'rcv_clock_bias_s'}
    assert isinstance(out['plume_xz_m'], list) and len(out['plume_xz_m']) == 2
    assert isinstance(out['rcv_clock_bias_s'], list) and len(out['rcv_clock_bias_s']) == len(clocks)
    assert all(type(v) is float and np.isfinite(v) for v in out['plume_xz_m'] + out['rcv_clock_bias_s'])
    c = np.array(out['rcv_clock_bias_s'])
    assert abs(c.sum()) < 1e-12 and np.max(abs(c)) <= 0.004 + 1e-14
    return (np.linalg.norm(np.array(out['plume_xz_m']) - xy), np.sqrt(np.mean((c-clocks)**2)), dt)

def report(label, scores):
    a = np.asarray(scores)
    print(label, 'n', len(a), 'max [m,s,runtime s]', a.max(axis=0), 'failures [location,clock,runtime]', (a > [1.8, .00025, 15]).sum(axis=0), flush=True)

public=[]; sources=[]; receivers=[]
for entry in pickle.loads(Path('/holdout_task/public_surveys.pkl').read_bytes()):
    s, gt = entry['survey'], entry['gt']
    x = np.asarray(gt['plume_xz_m']); c = gt['rcv_clock_bias_s']
    public.append(check(s,x,c))
    for i in range(len(s['z_src_m'])):
        sources.append(check(dict(s,z_src_m=np.delete(s['z_src_m'],i), T_obs_s=np.delete(s['T_obs_s'],i,axis=0)),x,c))
    for j in range(len(c)):
        sub = np.delete(c,j)
        receivers.append(check(dict(s,z_rcv_m=np.delete(s['z_rcv_m'],j), T_obs_s=np.delete(s['T_obs_s'],j,axis=1)),x,sub-sub.mean()))
report('public full',public)
report('public omit source',sources)
report('public omit receiver; retained clocks recentered',receivers)

for label, count in [('nominal', 20), ('near wells', 24), ('near depth ends', 24), ('sparser weak plume', 20)]:
    results=[]; meta=[]
    for i in range(count):
        w,h=(40.,80.) if label != 'sparser weak plume' else (rng.uniform(30,55), rng.uniform(60,95))
        ns,nr=(15,16) if label != 'sparser weak plume' else (rng.integers(9,13),rng.integers(9,14))
        zs=np.linspace(.075*h,.925*h,ns); zr=np.linspace(.0625*h,.9375*h,nr)
        x,z=rng.uniform(.22*w,.78*w), rng.uniform(.18*h,.82*h)
        if label=='near wells':
            x=[0.0, 1.5, 4., w-4., w-1.5, w][i%6]
        if label=='near depth ends':
            z=[.04*h,.075*h,.12*h,.88*h,.925*h,.96*h][i%6]
        amp=rng.uniform(5.5e-5,1e-4) if label!='sparser weak plume' else rng.uniform(2.5e-5,5.5e-5)
        dz=zr[None,:]-zs[:,None]; length=np.hypot(w,dz)
        times=length*rng.uniform(.00035,.0005)*(1-rng.uniform(.07,.22)*(dz/length)**2)
        # Independently evaluate the exact straight-ray integral by quadrature.
        for j,k in np.ndindex(times.shape):
            times[j,k]+=amp*length[j,k]*quad(lambda t: np.exp(-((w*t-x)**2+(zs[j]+t*dz[j,k]-z)**2)/(2*5.5**2)),0,1)[0]
        c=rng.uniform(-1,1,nr); c-=c.mean(); c*=.004/max(abs(c))
        times+=c+rng.normal(0, 3.5e-5 if label!='sparser weak plume' else 6e-5, times.shape)
        skip=rng.choice(times.size,round(.04*times.size),replace=False)
        times.flat[skip]+=rng.choice([-1,1],len(skip))*rng.uniform(.0055,.0065,len(skip))
        s=dict(W_m=w,H_m=h,z_src_m=zs,z_rcv_m=zr,T_obs_s=times)
        results.append(check(s,(x,z),c)); meta.append((i,w,h,ns,nr,x,z,amp))
    report(label,results)
    for col in (0,1):
        i=np.argmax(np.asarray(results)[:,col]); print('worst case',meta[i],results[i],flush=True)
