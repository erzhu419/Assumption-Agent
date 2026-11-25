import runpy
import numpy as np
from numpy.polynomial.legendre import leggauss

from tomo_weak_diagnose import invert_plume_and_clocks

def check(s, gt):
    got = invert_plume_and_clocks(s, gt['plume_xz_m'])
    return (np.linalg.norm(np.asarray(got['plume_xz_m'])-gt['plume_xz_m']), 1000*np.sqrt(np.mean((np.array(got['rcv_clock_bias_s'])-gt['rcv_clock_bias_s'])**2)), 0.)

nodes, weights = leggauss(48)
u = (nodes + 1)/2
out = []
labels = []
for seed in range(40):
    rng = np.random.default_rng(seed)
    w, h = (32., 68.) if seed % 2 else (50., 95.)
    ns, nr = [(12, 11), (15, 16), (19, 21)][seed % 3]
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
    s0, eps, amp = rng.uniform(.00033, .0005), rng.uniform(.04, .24), [.000035, .00007, .00012][seed % 3]
    times = s0 * (length - eps*dz**2/length) + amp*path + clock + rng.normal(0, .00007, (ns, nr))
    idx = rng.choice(times.size, round(.04 * times.size), replace=False)
    times.flat[idx] += .006 * rng.choice([-1, 1], len(idx))
    if seed % 4 == 0:
        times.flat[rng.choice(times.size, round(.10*times.size), replace=False)] = np.nan
    s = dict(W_m=w, H_m=h, z_src_m=src, z_rcv_m=rcv, T_obs_s=times)
    out.append(check(s, dict(plume_xz_m=[x, z], rcv_clock_bias_s=clock)))
    labels.append((seed, fx, fz, ns, nr, amp))
a = np.array(out)
print('synthetic 40-case maxima (m, ms, s):', a.max(axis=0), 'worst cases:', [labels[k] for k in a.argmax(axis=0)])
print('failures:', [(labels[k], a[k].tolist()) for k in np.flatnonzero(np.any(a > [1.8, .25, 15], axis=1))])
