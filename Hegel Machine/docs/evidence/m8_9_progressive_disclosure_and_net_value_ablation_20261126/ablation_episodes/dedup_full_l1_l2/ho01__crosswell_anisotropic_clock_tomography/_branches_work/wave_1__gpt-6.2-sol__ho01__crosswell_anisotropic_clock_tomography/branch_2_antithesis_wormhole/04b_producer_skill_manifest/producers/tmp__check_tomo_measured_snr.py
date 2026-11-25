import runpy
import numpy as np
from numpy.polynomial.legendre import leggauss

import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    base = runpy.run_path('/tmp/check_tomo_public.py', run_name='stress')
check = base['check']

nodes, weights = leggauss(48)
u = (nodes + 1)/2
out = []
labels = []
for seed in range(60, 120):
    rng = np.random.default_rng(seed)
    w, h = (32., 68.) if seed % 2 else (50., 95.)
    ns, nr = [(15, 16), (17, 14), (19, 21)][seed % 3]
    src = np.linspace(5, h-5, ns) + rng.uniform(-.75, .75, ns)
    rcv = np.linspace(5, h-5, nr) + rng.uniform(-.75, .75, nr)
    fx, fz = [(0.07, 0.15), (.93, .85), (.1, .82), (.89, .18), (.5, .5)][seed % 5]
    x, z = w*fx, h*fz
    a = src[:, None]
    dz = rcv[None, :] - a
    length = np.hypot(w, dz)
    integrand = np.exp(-((w*u - x)**2 + (a[..., None] + dz[..., None]*u - z)**2)/(2*5.5**2))
    path = length * (integrand @ (weights/2))
    clock = rng.uniform(-1, 1, nr)
    clock -= clock.mean()
    clock *= .004/np.max(np.abs(clock))
    s0, eps, amp = rng.uniform(.00033, .0005), rng.uniform(.04, .24), [.000055, .00007, .00010][seed % 3]
    times = s0 * (length - eps*dz**2/length) + amp*path + clock + rng.normal(0, .000035, (ns, nr))
    idx = rng.choice(times.size, round(.04 * times.size), replace=False)
    times.flat[idx] += .006 * rng.choice([-1, 1], len(idx))
    if seed % 4 == 0:
        times.flat[rng.choice(times.size, round(.10*times.size), replace=False)] = np.nan
    s = dict(W_m=w, H_m=h, z_src_m=src, z_rcv_m=rcv, T_obs_s=times)
    out.append(check(s, dict(plume_xz_m=[x, z], rcv_clock_bias_s=clock)))
    labels.append((seed, fx, fz, ns, nr, amp))
a = np.array(out)
print('synthetic measured-SNR 60-case maxima (m, ms, s):', a.max(axis=0), 'worst cases:', [labels[k] for k in a.argmax(axis=0)])
failures = [(labels[k], a[k].tolist()) for k in np.flatnonzero(np.any(a > [1.8, .25, 15], axis=1))]
print('failures:', failures)
assert not failures
