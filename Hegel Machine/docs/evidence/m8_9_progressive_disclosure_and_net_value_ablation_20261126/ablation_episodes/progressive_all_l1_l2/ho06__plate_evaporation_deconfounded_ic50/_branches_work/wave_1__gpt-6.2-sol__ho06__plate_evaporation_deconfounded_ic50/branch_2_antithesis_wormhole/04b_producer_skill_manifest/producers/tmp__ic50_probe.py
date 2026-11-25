import pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

LN10 = np.log(10.)

def fit(plates, mode='exp', loss='cauchy'):
    keys=sorted({w['compound_id'] for p in plates for w in p['wells'] if w['well_type']=='COMPOUND'})
    lookup={k:i for i,k in enumerate(keys)}
    n=len(plates); m=len(keys)
    a=np.array([(j,w['row'],w['col'],w['fluorescence'],lookup[w['compound_id']] if w['well_type']=='COMPOUND' else (-1 if w['well_type']=='DMSO' else -2),w['log10_conc_uM'] if w['well_type']=='COMPOUND' else 0.) for j,p in enumerate(plates) for w in p['wells']])
    p,r,c,y,k,dose=a.T; p=p.astype(int); k=k.astype(int)
    bg=np.array([np.median(y[(p==j)&(k==-2)]) for j in range(n)])
    scale=np.array([np.median(y[(p==j)&(k==-1)])-bg[j] for j in range(n)])
    obs=(y-bg[p])/scale[p]
    x=(c-11.5)/11.5; z=(r-7.5)/7.5
    dist=np.minimum.reduce((r,15-r,c,23-c)).astype(int)
    comp=k>=0; cc=k[comp]; dd=dose[comp]
    init=[]
    for j in range(m):
        ix=k==j; ds=np.unique(dose[ix]); ms=np.array([np.median(obs[ix&(dose==d)]) for d in ds]); init.append(ds[np.argmin(abs(ms-.5))])
    if mode=='free':
        basis=np.column_stack([(dist==d) for d in range(7)]+[x,z]); sp=np.r_[.4*np.exp(-np.arange(7)/2),-.02,.03]
    if mode=='flat':
        basis=np.empty((len(y),0)); sp=[]
    if mode.startswith('exp'):
        sp=[.4,2.,-.02,.03]+([1.] if mode=='exppow' else [])
    start=np.r_[np.zeros(n),np.full(n,.88 if mode!='flat' else 1.),init,np.ones(m),sp]
    p0=2*n+2*m
    J=np.zeros((len(y),len(start))); idx=np.arange(len(y)); idxc=idx[comp]
    def evaluate(t, jac=False):
        s=t[p0:]
        if mode.startswith('exp'):
            amp,tau,bx,bz=s[:4]; power=s[4] if mode=='exppow' else 1.
            q=dist/tau; u=q**power; e=np.exp(-u)
            field=1+amp*e+bx*x+bz*z
            if jac:
                db=[e,amp*e*u*power/tau,x,z]
                if mode=='exppow': db.append(-amp*e*u*np.log(np.maximum(q,1e-12)))
                db=np.column_stack(db)
        else:
            field=1+basis@s
            if jac: db=basis
        mu=t[(2*n+cc)]; hs=t[(2*n+m+cc)]
        v=(k==-1).astype(float); v[comp]=expit(LN10*hs*(mu-dd))
        g=t[n+p]; sig=g*field
        if not jac: return t[p]+sig*v-obs
        j=J.copy(); j[idx,p]=1.; j[idx,n+p]=field*v
        dv=sig[comp]*v[comp]*(1-v[comp])*LN10
        j[idxc,2*n+cc]=dv*hs; j[idxc,2*n+m+cc]=dv*(mu-dd)
        j[:,p0:]=(g*v)[:,None]*db
        return j
    lo=np.r_[np.full(n,-.2),np.full(n,.3),np.full(m,-7.),np.full(m,.15),np.full(len(sp),-.5)]
    hi=np.r_[np.full(n,.2),np.full(n,2.),np.full(m,7.),np.full(m,5.),np.full(len(sp),1.)]
    if mode.startswith('exp'): lo[p0:p0+2]=[0.,.15]; hi[p0:p0+2]=[1.5,12.]
    if mode=='exppow': lo[-1]=.35; hi[-1]=4.
    result=least_squares(evaluate,start,jac=lambda t:evaluate(t,True),bounds=(lo,hi),loss='soft_l1',f_scale=.025)
    result=least_squares(evaluate,result.x,jac=lambda t:evaluate(t,True),bounds=(lo,hi),loss=loss,f_scale=.012)
    t=result.x
    return np.column_stack([t[2*n:2*n+m],t[2*n+m:p0]]),keys,t[p0:],result.fun,(p,k,dist)

if __name__=='__main__':
    with open('/holdout_task/public_screen.pkl','rb') as f: data=pickle.load(f)
    plates=data['plates']
    for mode in ('flat','exp','exppow','free'):
        pred,keys,sp,res,(p,k,dist)=fit(plates,mode)
        gt=np.array([[data['gt'][cid][f] for f in ('log10_ic50_uM','hill_slope')] for cid in keys]); err=pred-gt
        loo=np.array([fit(plates[:i]+plates[i+1:],mode)[0] for i in range(len(plates))]); le=np.abs(loo-gt)
        print(mode,'shape',sp.round(4),'worst errors',abs(err).max(axis=0).round(4),'worst LOPO',le.max(axis=(0,1)).round(4),'worst LOPO ids',[keys[np.unravel_index(le[:,:,j].argmax(),le.shape[:2])[1]] for j in range(2)])
        print('residual medians by edge distance',[(d,round(float(np.median(res[(dist==d)&(k>=0)])),4)) for d in np.unique(dist) if np.any((dist==d)&(k>=0))]); print('compound errors',list(zip(keys,err.round(4).tolist())))
