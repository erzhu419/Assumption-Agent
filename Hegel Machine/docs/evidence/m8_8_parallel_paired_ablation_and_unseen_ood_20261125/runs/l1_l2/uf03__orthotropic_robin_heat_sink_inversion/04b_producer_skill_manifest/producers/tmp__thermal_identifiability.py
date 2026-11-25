from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr
from scipy.sparse.linalg import splu
from thermal_sink_solver import _thermal_operator
src = Path('/tmp/thermal_validate.py').read_text()
exec(src.split('cases = pickle.load',1)[0])
exec(src[src.index('def truth_grid'):src.index('\nfor name,n,m,gamma,sigma,spread')])

def simulate(n,m,gamma,sigma,spread,xy,seed,noise,corruption):
    rng = np.random.default_rng(seed)
    x=(np.arange(n)+.5)/n
    k=np.exp(spread*(np.sin(2*np.pi*x[:,None]+1.4*seed)*np.cos(2*np.pi*x[None,:])+0.4*np.cos(4*np.pi*x[:,None]+3*x[None,:])))
    amp=rng.uniform(15.,29.); offset=rng.uniform(-.07,.07)
    i,j=np.unravel_index(rng.choice(n*n,m,replace=False),(n,n)); ij=np.column_stack((i,j))
    clean=truth_grid(n,k,gamma,sigma,xy,amp)[i,j]+offset
    obs=clean+rng.normal(0,noise,m)
    order=np.argsort(np.linalg.norm((ij+.5)/n-xy,axis=1)); bad=order[:round(.06*m)]
    if corruption=='moderate':
        obs[bad]+=rng.choice([-1.,1.],len(bad))*rng.uniform(5,20,len(bad))*noise
    else:
        obs[bad]+=rng.choice([-1.,1.],len(bad))*rng.uniform(.16,.26,len(bad))
    d=dict(N=n,kappa=k,gamma_robin=gamma,sigma_m=sigma,sensor_ij=ij,T_obs=obs)
    g=dict(source_xy=xy,amplitude=amp,junction_offset=offset)
    return d,g,bad,clean

for label,center in [('interior',True),('near_edge',False)]:
    for corruption in ('gross','moderate'):
        results=[]
        for seed in range(48):
            rng=np.random.default_rng(19020+seed)
            xy=rng.uniform(.18,.82,2) if center else np.array([rng.uniform(.02,.16),rng.uniform(.16,.84)])
            d,g,bad,_=simulate(22,52,2.2,.085,.6,xy,29020+seed,.0013,corruption)
            results.append((seed,run(d,g)))
        report(label+' '+corruption,results)

# Compare two valid parameter sets using the known clean sensor set and equal noise.
n,m,gamma,sigma=24,34,2.2,.058
xy=np.array([.035,.045]); noise=.0013
for seed in (4200,4212,4221):
    d,g,bad,clean=simulate(n,m,gamma,sigma,.8,xy,seed,noise,'gross')
    i,j=d['sensor_ij'].T; indices=i*n+j
    keep=np.ones(m,bool); keep[bad]=False
    unit=np.zeros((n*n,m),order='F'); unit[indices,np.arange(m)]=1
    G=splu(_thermal_operator(d['kappa'],gamma)).solve(unit).T[keep]
    x=(np.arange(n)+.5)/n
    def residual(p,amp):
        q=np.exp(-((x[:,None]-p[0])**2+(x[None,:]-p[1])**2)/(2*sigma**2))/n**2
        return (amp*(G@q.ravel())+p[2]-clean[keep])/noise
    for ratio in (1.18,.84):
        o=least_squares(residual,np.r_[xy,g['junction_offset']],args=(ratio*g['amplitude'],),bounds=([0,0,-np.inf],[1,1,np.inf]),gtol=1e-9)
        distance=np.linalg.norm(o.fun)
        print('ambiguity',seed,'amplitude ratio',ratio,'position separation',round(np.linalg.norm(o.x[:2]-xy),4),'whitened mean distance',round(distance,3),'equal-prior discrimination error',round(ndtr(-distance/2),3),flush=True)
