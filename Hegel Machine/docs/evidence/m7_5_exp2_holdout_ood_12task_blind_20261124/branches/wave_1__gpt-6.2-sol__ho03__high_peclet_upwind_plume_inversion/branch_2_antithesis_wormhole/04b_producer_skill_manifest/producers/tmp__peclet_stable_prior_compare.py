import contextlib, io, os, runpy
from time import perf_counter
import numpy as np
from peclet_stable import invert_point_source as baseline
from peclet_stable_prior import invert_point_source as weak_amp
with contextlib.redirect_stdout(io.StringIO()):
    v=runpy.run_path('/tmp/peclet_validate.py')
base, truth, strength, field=(v[k] for k in ('base','truth','strength','field'))
sets={'public':[(base,truth,strength)],'loo':[],'previous':[],'fresh':[],'new':[]}
for j in range(18):
    mask=np.arange(18)!=j
    sets['loo'].append((dict(base,sensor_ij=base['sensor_ij'][mask],c_obs=base['c_obs'][mask]),truth,strength))
rows=[(1,(.28,.36),4.2,.0025,.045,True),(2,(.24,.31),5.4,.00025,.045,False),(4,(.30,.45),4.6,.00025,.065,False),(6,(.53,.35),5.,.00025,.05,False)]
def append(group,p,loc,Q,seed,hits):
    signal=field(p,loc,Q)[tuple(p['sensor_ij'].T)]
    order=np.argsort(signal); rng=np.random.default_rng(seed)
    bad=order[-2:] if hits else rng.choice(order[:-5],2,replace=False)
    y=signal+rng.normal(0,.02*signal.max(),len(signal)); y[bad]+=signal.max()*np.array([3.,-2.6])
    sets[group].append((dict(p,c_obs=y),np.asarray(loc),Q))
for j,loc,Q,D,sigma,hits in rows:
    for seed in (420+j,821+j,1470+j,3409+j):
        append('previous' if seed==420+j else 'fresh',dict(base,D=D,sigma_s=sigma),loc,Q,seed,hits)
rng=np.random.default_rng(7354)
for j in range(24):
    loc=rng.uniform((.16,.17),(.59,.59)); Q=float(rng.uniform(2.,8.)); D=[.00023,.00036,.0011,.0025][j%4]; sigma=rng.uniform(.031,.067)
    p=dict(base,D=D,sigma_s=sigma)
    if j%3==0:
        # Reflected geometry and flow retain the same physical transport problem.
        p.update(ux_face=-p['ux_face'][::-1,:],uy_face=p['uy_face'][::-1,:],sensor_ij=p['sensor_ij']*[-1,1]+[p['N']-1,0]); loc[0]=1-loc[0]
    append('new',p,loc,Q,7354+j,j%2==0)
for k in (0,1,4,12):
    os.environ['PECLET_PRIOR_TEST']=str(k)
    f=weak_amp if k else baseline
    for name, rows in sets.items():
        e=[]
        for p,xy,Q in rows:
            t=perf_counter(); r=f(p); dt=perf_counter()-t
            e.append((np.linalg.norm(r['source_xy']-xy),abs(r['strength_Q']/Q-1),dt))
        a=np.asarray(e); fail=np.flatnonzero((a[:,0]>.025)|(a[:,1]>.08)); print(k,name,'max',a.max(0).round(5),'fails',fail.tolist(),'p90',np.quantile(a,.9,axis=0).round(5),flush=True)
        if name in ('fresh','new'): print('  failed errors',a[fail,:2].round(4).tolist(),flush=True)
