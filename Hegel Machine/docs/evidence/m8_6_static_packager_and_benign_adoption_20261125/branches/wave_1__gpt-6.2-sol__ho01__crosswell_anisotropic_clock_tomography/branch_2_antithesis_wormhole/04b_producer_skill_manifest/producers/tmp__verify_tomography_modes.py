import pickle
import sys
from time import perf_counter
from collections import defaultdict
import numpy as np
from scipy.special import roots_legendre
sys.path.insert(0, '/holdout_task')
from clock_mode_candidate import invert_plume_and_clocks

stats = defaultdict(list)
def evaluate(label, name, survey, center, clock):
    start = perf_counter()
    result = invert_plume_and_clocks(survey)
    elapsed = perf_counter() - start
    xyz = np.asarray(result['plume_xz_m'])
    bias = np.asarray(result['rcv_clock_bias_s'])
    assert xyz.shape == (2,) and bias.shape == np.shape(clock)
    assert np.all(np.isfinite(xyz)) and np.all(np.isfinite(bias))
    assert 0 <= xyz[0] <= survey['W_m'] and 0 <= xyz[1] <= survey['H_m']
    assert abs(bias.sum()) < 1e-12 and np.max(np.abs(bias)) <= .004 + 1e-12
    err = np.linalg.norm(xyz - center)
    rmse = np.mean((bias - clock)**2)**.5
    stats[label].append((err, rmse, elapsed, name, tuple(xyz)))
    if err > 1.8 or rmse > .00025 or elapsed >= 15:
        print('MISS', label, name, 'center_m', round(err,4), 'clock_ms', round(1000*rmse,4), 'runtime_s',round(elapsed,4),'estimated',xyz,'true',center, flush=True)

with open('/holdout_task/public_surveys.pkl','rb') as f:
    for i, sample in enumerate(pickle.load(f)):
        s, truth = sample['survey'], sample['gt']
        clock, point = np.asarray(truth['rcv_clock_bias_s']), np.asarray(truth['plume_xz_m'])
        evaluate('public full', str(i), s, point, clock)
        for j in range(len(s['z_src_m'])):
            reduced = dict(s, z_src_m=np.delete(s['z_src_m'],j), T_obs_s=np.delete(s['T_obs_s'],j,axis=0))
            evaluate('public leave-source-out', f'{i}/{j}', reduced, point, clock)

nodes, weights = roots_legendre(100)
u, qw = (nodes+1)/2, weights/2
rng = np.random.default_rng(148522)
for i in range(120):
    w, h = rng.uniform(32,55), rng.uniform(70,95)
    ns,nr = rng.integers(12,20,size=2)
    sz = np.linspace(.07*h,.93*h,ns) + rng.uniform(-1,1,ns)
    rz = np.linspace(.06*h,.94*h,nr) + rng.uniform(-1,1,nr)
    position = i % 6
    if position < 3:
        center = np.array([rng.uniform(.15,.85)*w, rng.uniform(.16,.84)*h])
        region = 'interior'
    elif position < 5:
        center = np.array([rng.choice([1.,w-1.]), rng.uniform(.16,.84)*h])
        region = 'near well'
    else:
        center = np.array([rng.uniform(.13,.87)*w, rng.choice([2.,h-2.])])
        region = 'near depth edge'
    clock = rng.uniform(-1,1,nr)
    clock -= clock.mean()
    clock *= rng.choice([.0025,.004])/np.max(abs(clock))
    dz = rz[None,:] - sz[:,None]
    dl = np.hypot(w,dz)
    quad = dl * np.sum(qw*np.exp(-.5*((w*u-center[0])**2+(sz[:,None,None]+dz[:,:,None]*u-center[1])**2)/5.5**2),axis=-1)
    times = dl*rng.uniform(.00034,.00048)*(1-rng.uniform(.07,.24)*(dz/dl)**2) + rng.uniform(.000045,.000095)*quad + clock
    times += rng.normal(0,rng.choice([.00003,.000055]),times.shape)
    bad = rng.choice(times.size, int(.04*times.size+.5), replace=False)
    times.flat[bad] += rng.choice([-1,1],len(bad))*rng.uniform(.0054,.0062,len(bad))
    survey = dict(W_m=w,H_m=h,z_src_m=sz,z_rcv_m=rz,T_obs_s=times,R_plume_m=5.5)
    evaluate(region,str(i),survey,center,clock)

for label, rows in stats.items():
    worst_center = max(rows,key=lambda r:r[0])
    worst_clock = max(rows,key=lambda r:r[1])
    print(f'{label}: n={len(rows)} max center = {worst_center[0]:.5f} m ({worst_center[3]}), max clock = {1000*worst_clock[1]:.5f} ms ({worst_clock[3]}), max runtime = {max(r[2] for r in rows):.5f} s', flush=True)
