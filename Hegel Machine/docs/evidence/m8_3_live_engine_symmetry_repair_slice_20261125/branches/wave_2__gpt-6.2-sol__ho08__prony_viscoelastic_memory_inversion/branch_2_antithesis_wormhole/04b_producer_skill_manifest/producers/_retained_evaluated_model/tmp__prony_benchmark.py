import numpy as np
import pickle
from scipy.linalg import qr
from scipy.optimize import nnls, least_squares
from prony_probe import memory

p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); e=p['experiment']
t,x,y,tt,xx=[e[k] for k in ('train_t','train_strain','train_stress_raw','test_t','test_strain')]
grid=np.geomspace(.02,20,96)

def design(t,x,ts): return np.c_[x,memory(t,x,ts)]
def problem(t,x,y):
    D=np.c_[np.ones(len(t)),(t-t.mean())/np.ptp(t)]; A=design(t,x,grid)
    def solve(A,D,y,w=None):
        if w is not None: A=A*w[:,None]; D=D*w[:,None]; y=y*w
        Q=qr(D,mode='economic')[0]; B=A-Q@(Q.T@A); yy=y-Q@(Q.T@y)
        norm=np.maximum(np.linalg.norm(B,axis=0),1e-30)
        c=nnls(B/norm,yy,maxiter=2500)[0]/norm
        return c,yy-B@c
    mad=lambda r:max(1.4826*np.median(abs(r-np.median(r))),1e-12)
    w=np.ones(len(t))
    for _ in range(12):
        c,_=solve(A,D,y,w); b=np.linalg.lstsq(D*w[:,None],(y-A@c)*w,rcond=None)[0]
        r=y-A@c-D@b; s=mad(r); w=1/np.hypot(1,r/(2.4*s))
    k=abs(r-np.median(r))<=6*s; d=D[k]; obs=y[k]; c,r=solve(A[k],d,obs)
    ix=np.flatnonzero(c[1:]>1e-7*max(c)); groups=np.split(ix,np.where(np.diff(ix)>2)[0]+1) if len(ix) else []
    z0=np.array([np.average(np.log(grid[g]),weights=c[1:][g]) for g in groups]); Q=qr(d,mode='economic')[0]; obs=obs-Q@(Q.T@obs)
    def fit(pen):
        if pen<0 or not len(z0): return c,grid,np.linalg.norm(r)**2/s**2
        def calc(z):
            A=design(t,x,np.exp(z))[k]; B=A-Q@(Q.T@A); norm=np.maximum(np.linalg.norm(B,axis=0),1e-30)
            cc=nnls(B/norm,obs,maxiter=2500)[0]/norm
            return cc,obs-B@cc
        def fun(z): return np.r_[calc(z)[1],s*pen*np.diff(z,n=2)]
        opt=least_squares(fun,z0,bounds=(np.log(.02),np.log(20)),max_nfev=65)
        cc,rr=calc(opt.x); return cc,np.exp(opt.x),np.linalg.norm(rr)**2/s**2
    return fit, z0

def scores(t,x,y,tt,xx,true,g):
    fit,z0=problem(t,x,y)
    fits={pen:fit(pen) for pen in (-1,0,3,6,10,18)}
    # Tighten spacing only when cost is within sampling noise for the extra shape constraints.
    fits['adaptive']=next((fits[a] for a in (18,10,6,3) if fits[a][2]-fits[0][2]<=max(1,len(z0)-2)),fits[0])
    rows={}
    for pen,(c,ts,sse) in fits.items():
        pred=design(tt,xx,ts)@c; slices=np.array_split(np.arange(len(tt)),4)
        rows[pen]=[abs(c[0]/g-1),np.linalg.norm(pred-true)/np.linalg.norm(true),max(np.linalg.norm((pred-true)[j])/np.linalg.norm(true[j]) for j in slices),c[0],sse-fits[0][2]]
    return len(z0),rows

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
    for (f,pen),a in res.items():
        b=np.asarray(a); print(f,pen,'max %',np.round(100*b.max(0),2),'median %',np.round(100*np.median(b,axis=0),2),'both pass',sum((b[:,0]<=.06)&(b[:,1]<=.022)))
