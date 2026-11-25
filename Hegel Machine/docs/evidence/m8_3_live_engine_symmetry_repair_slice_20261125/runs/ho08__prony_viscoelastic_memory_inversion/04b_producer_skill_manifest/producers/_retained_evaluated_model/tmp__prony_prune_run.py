import numpy as np
import pickle
from scipy.linalg import qr
from scipy.optimize import nnls, least_squares
from prony_probe import memory

p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); e=p['experiment']
t,x,y,tt,xx=[e[k] for k in ('train_t','train_strain','train_stress_raw','test_t','test_strain')]
grid=np.geomspace(.02,20,96)

def design(t,x,ts): return np.c_[x,memory(t,x,ts)]
from prony_prune_probe import scores
if __name__=='__main__':
    print('public',scores(t,x,y,tt,xx,p['gt_test_stress'],p['gt_G_inf']),flush=True)
    res={}
    for family in ('separated','irregular','slow','short-hold','changed-cadence','steps'):
        for seed in range(6):
            rng=np.random.default_rng(717+seed+100*('separated','irregular','slow','short-hold','changed-cadence','steps').index(family)); m=3+seed%3
            end=30 if family!='short-hold' else 23; t0=np.arange(round(end/.05)+1)*.05
            h=14 if family!='short-hold' else 8
            x0=np.where(t0<2,.025*t0,np.where(t0<=h,.05,.05-.035*np.sin((t0-h)*(.53+.11*seed))))
            if family=='steps': x0=x0+.02*(t0>=h+4)-.03*(t0>=h+10)
            if family=='changed-cadence':
                t0[1:-1]+=rng.uniform(-.012,.012,len(t0)-2)
                x0=np.interp(t0,t,x)
            lo=rng.uniform(.029,.09); hi=rng.uniform(6,12) if family!='slow' else rng.uniform(13,19.5)
            z=np.linspace(np.log(lo),np.log(hi),m); z[1:-1]+=rng.normal(0,.48 if family=='irregular' else .11,m-2); ts=np.exp(np.sort(z))
            coef=np.r_[rng.uniform(8,21), rng.uniform(7,30,m)]
            tt0=tt if family!='changed-cadence' else np.linspace(0,24,344)
            xx0=np.interp(tt0,tt,xx)+(family=='steps')*(.025*(tt0>=7)-.04*(tt0>=14))
            y0=design(t0,x0,ts)@coef; truth=design(tt0,xx0,ts)@coef
            y0+=rng.uniform(-3,3)+rng.uniform(-.24,.24)*t0+rng.normal(0,.012+.004*(seed%3),len(t0)); ix=rng.choice(len(t0),round(.02*len(t0)),replace=False); y0[ix]+=rng.choice([-1,1],len(ix))*rng.uniform(.8,6,len(ix))
            k,r=scores(t0,x0,y0,tt0,xx0,truth,coef[0]); print(family,seed,'true/fitted modes',m,k,'modulus %',[round(100*r[i][0],1) for i in r],'worst slices %',[round(100*r[i][2],2) for i in r],flush=True)
            for pen,a in r.items(): res.setdefault((family,pen),[]).append(a[:3])
    for pen in ('base',1.,2.,4.,7.):
        all_rows=np.concatenate([v for (f,k),v in res.items() if k==pen]); print('overall',pen,'worst/median percent',np.round(100*all_rows.max(0),2),np.round(100*np.median(all_rows,axis=0),2),'both pass',sum((all_rows[:,0]<=.06)&(all_rows[:,1]<=.022)))
    for (f,pen),a in res.items():
        b=np.asarray(a); print(f,pen,'max %',np.round(100*b.max(0),2),'median %',np.round(100*np.median(b,axis=0),2),'both pass',sum((b[:,0]<=.06)&(b[:,1]<=.022)))
