import pickle
from time import perf_counter
import numpy as np
from peclet_plume import forward_peclet_field
from peclet_marginal import invert_point_source

with open('/holdout_task/public_peclet.pkl', 'rb') as file:
    data = pickle.load(file)
base = data['problem']
truth = np.asarray(data['gt_source_xy'])
strength = data['gt_Q']
h = base['h']
x = (np.arange(base['N']) + .5)*h

def field(p, loc, Q):
    source = Q*np.exp(-.5*((x[:,None]-loc[0])**2+(x[None,:]-loc[1])**2)/p['sigma_s']**2)
    return forward_peclet_field(p['ux_face'], p['uy_face'], p['D'], p['kappa'], source, h)

def run(name, p, loc, Q):
    t = perf_counter()
    r = invert_point_source(p)
    dt = perf_counter()-t
    xy = np.asarray(r['source_xy'])
    epos = np.linalg.norm(xy-loc)/(p['N']*h)
    eq = abs(r['strength_Q']/Q-1)
    print(f'{name:25s} {epos:.5f} {eq:.5f} {dt:.3f}s fit={xy.round(5)} Q={r["strength_Q"]:.4f}', flush=True)
    return epos, eq, dt

u = field(base, truth, strength)
clean = u[tuple(base['sensor_ij'].T)]
print('forward relative L2:', np.linalg.norm(u-data['gt_field'])/np.linalg.norm(data['gt_field']))
print('public readings minus clean:', np.round(base['c_obs']-clean, 5))
print('slice                     pos/L  rel_Q   time')
public = run('public',base,truth,strength)
errors = []
for case, (loc, Q, D, sigma, fault) in enumerate([
    ((.28,.36),4.2,.0025,.045,'off'),
    ((.28,.36),4.2,.0025,.045,'hits'),
    ((.24,.31),5.4,.00025,.045,'off'),
    ((.22,.24),3.8,.00030,.055,'hits'),
    ((.30,.45),4.6,.00025,.065,'off'),
    ((.45,.24),5.0,.00030,.045,'off'),
    ((.53,.35),5.0,.00025,.050,'off'),
    ((.18,.20),4.0,.00250,.035,'positive'),
    ((.27,.52),4.8,.00100,.055,'positive'),
]):
    p = dict(base,D=D,sigma_s=sigma)
    signal = field(p,loc,Q)[tuple(p['sensor_ij'].T)]
    rng = np.random.default_rng(420+case)
    order = np.argsort(signal)
    bad = order[-2:] if fault=='hits' else rng.choice(order[:-5],2,replace=False)
    noise = .02*signal.max()
    obs = signal+rng.normal(0,noise,len(signal))
    obs[bad] += signal.max()*np.array([3., 2.6 if fault=='positive' else -2.6])
    p['c_obs'] = obs
    print('  clean channels above 5 noise sigma:', np.count_nonzero(np.delete(signal,bad)>5*noise))
    errors.append(run(f'{case} {fault} Pe={p["ux_face"].max()*h/D:.0f}',p,np.asarray(loc),Q))
print('worst synthetic pos/L, rel_Q, s:',np.max(errors,axis=0))
