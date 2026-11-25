import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import prony_rheology as base
from prony_prune_probe import fits
from prony_benchmark import p,t,x,y,tt,xx,design

def compare(t,x,y,tt,xx,truth,g):
    options,profile=fits(t,x,y)
    fit=base.fit_prony_and_predict_stress(dict(train_t=t, train_strain=x, train_stress_raw=y, test_t=tt, test_strain=xx))
    prior=np.r_[fit['G_inf'],fit['test_stress']]
    tau,c=options[1.]
    reduced=np.r_[c[0],base._design(tt,xx,tau)@c]
    gap=profile[1][1] if len(profile)>1 else np.inf
    vals={}
    for name,w in [('base',0.),('equal-if-tied',.5*max(0.,1-gap)),('prefer-parsimonious',.8*max(0.,1-gap)),('half-pruned',.5 if gap < 1 else 0.)]:
        pred=(1-w)*prior+w*reduced
        delta=pred[1:]-truth
        vals[name]=(abs(pred[0]/g-1),np.linalg.norm(delta)/np.linalg.norm(truth),max(np.linalg.norm(delta[j])/np.linalg.norm(truth[j]) for j in np.array_split(np.arange(len(tt)),4)))
    return gap,vals

print('public',compare(t,x,y,tt,xx,p['gt_test_stress'],p['gt_G_inf']),flush=True)
families=('separated','irregular','slow','short-hold','changed-cadence','steps')
for seeds in (range(6),range(6,12)):
    out={}
    for i,f in enumerate(families):
        for seed in seeds:
            rng=np.random.default_rng(717+seed+100*i); m=3+seed%3
            end=23 if f=='short-hold' else 30
            t0=np.arange(round(end/.05)+1)*.05; h=8 if f=='short-hold' else 14
            x0=np.where(t0<2,.025*t0,np.where(t0<=h,.05,.05-.035*np.sin((t0-h)*(.53+.11*seed))))
            if f=='steps': x0=x0+.02*(t0>=h+4)-.03*(t0>=h+10)
            if f=='changed-cadence':
                t0[1:-1]+=rng.uniform(-.012,.012,len(t0)-2); x0=np.interp(t0,t,x)
            lo=rng.uniform(.029,.09); hi=rng.uniform(13,19.5) if f=='slow' else rng.uniform(6,12)
            z=np.linspace(np.log(lo),np.log(hi),m); z[1:-1]+=rng.normal(0,.48 if f=='irregular' else .11,m-2)
            ts=np.exp(np.sort(z)); c=np.r_[rng.uniform(8,21),rng.uniform(7,30,m)]
            tt0=tt if f!='changed-cadence' else np.linspace(0,24,344)
            xx0=np.interp(tt0,tt,xx)+(f=='steps')*(.025*(tt0>=7)-.04*(tt0>=14))
            y0=design(t0,x0,ts)@c; truth=design(tt0,xx0,ts)@c
            y0+=rng.uniform(-3,3)+rng.uniform(-.24,.24)*t0+rng.normal(0,.012+.004*(seed%3),len(t0))
            ix=rng.choice(len(t0),round(.02*len(t0)),replace=False); y0[ix]+=rng.choice([-1,1],len(ix))*rng.uniform(.8,6,len(ix))
            gap,vals=compare(t0,x0,y0,tt0,xx0,truth,c[0]); out.setdefault(f,[]).append(vals)
            if max(v[0] for v in vals.values())>.35:
                print('large error',f,seed,'cost gap',gap,'modulus %',[round(100*v[0],2) for v in vals.values()],flush=True)
    for k in vals:
        a=np.array([r[k] for runs in out.values() for r in runs]); print('seed start',seeds.start,k,'max',np.round(100*a.max(axis=0),3),'median',np.round(100*np.median(a,axis=0),3),'pass',np.sum((a[:,0]<=.06)&(a[:,1]<=.022)))
        print('family worst modulus %',{f:round(100*max(r[k][0] for r in runs),2) for f,runs in out.items()})
