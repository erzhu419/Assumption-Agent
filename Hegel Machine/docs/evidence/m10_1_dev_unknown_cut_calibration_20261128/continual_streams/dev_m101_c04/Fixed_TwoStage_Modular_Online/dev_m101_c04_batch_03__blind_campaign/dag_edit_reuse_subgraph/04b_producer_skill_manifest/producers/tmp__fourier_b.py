import numpy as np
from scipy.optimize import least_squares
from probe_routes import s, ax, scores
u,v=(s['panel_b'][k] for k in ('u','v'))
def design(x,f,kind):
    cols=[np.ones(len(x))]
    if kind.startswith('lin'): cols.append(x)
    cols.append(np.sin(x*f[:4]+(f[4:] if kind.endswith('phase') else 0)))
    if kind.endswith('cos'): cols.append(np.cos(x*f[:4]))
    return np.column_stack(cols)
def fit(a,b,kind,starts):
    def solve(f):
        d=design(a,f,kind); w=np.linalg.lstsq(d,b,rcond=None)[0]; return (d@w-b).ravel(),w
    best=None
    for f in starts:
        if kind.endswith('phase'):f=np.r_[f,np.zeros(4)]
        low=np.r_[np.full(4,.1),np.full(len(f)-4,-np.pi)]
        high=np.r_[np.full(4,8.),np.full(len(f)-4,np.pi)]
        r=least_squares(lambda p:solve(p)[0],f,bounds=(low,high),max_nfev=150,gtol=1e-11)
        err,w=solve(r.x); rmse=np.mean(err**2)**.5
        if best is None or rmse<best[0]:best=rmse,r.x,w
    return best
starts=[np.full(4,f) for f in (1.,2.,3.,4.,5.)]+[np.random.default_rng(19).uniform(1,5,(12,4))[i] for i in range(12)]
for kind in ('sin','linsin','sinphase','linsinphase','sincos','linsincos'):
    rms,f,w=fit(u,v,kind,starts)
    loo=[]
    for i in range(len(u)):
        keep=np.arange(len(u))!=i
        _,ff,ww=fit(u[keep],v[keep],kind,[f[:4],starts[2]])
        loo.append(np.max(abs(design(u[i:i+1],ff,kind)@ww-v[i])))
    print('FOURIER',kind,'residual',rms,'worst anchor LOO',max(loo),'freq',f.round(6),'downstream',scores(design(ax,f,kind)@w),flush=True)
    if kind in ('sin','linsin'): print('weights',w.round(6),flush=True)
