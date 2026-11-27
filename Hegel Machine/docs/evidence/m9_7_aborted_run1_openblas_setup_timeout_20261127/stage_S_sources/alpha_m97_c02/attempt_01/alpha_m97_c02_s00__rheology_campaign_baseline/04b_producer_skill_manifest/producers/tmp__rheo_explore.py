import pickle, numpy as np
from scipy.optimize import least_squares
from itertools import combinations

def data(d, fill='balanced'):
    tr,te=d['train'],d['test']; keys=('lab_id','stratum','coupon_features_raw','ref_coupons','past_stress_cmd','stress_hold_cmd','strain_0')
    lab,st,x,r,p,u,s0=(np.concatenate((tr[k],te[k])) for k in keys)
    c=d['calibration_load_cells']; cal=[]
    for l in range(4):
        idx=c['lab_id']==l; cal.append(np.polyfit(c['cmd_stress'][idx],c['true_stress'][idx],1))
        idx=lab==l; x[idx]=(x[idx]-np.nanmedian(np.nanmean(r[idx],axis=0)))/np.nanmedian(np.nanstd(r[idx],axis=0))
    cells=[(lab==l)&(st==s) for l,s in np.unique(np.column_stack((lab,st)),axis=0)]
    means=np.array([np.nanmean(x[m],axis=0) for m in cells]); balanced=np.nanmean(means,axis=0)
    for m,mu in zip(cells,means): x[m]=np.where(np.isfinite(x[m]),x[m],np.where(np.isfinite(mu),mu,balanced) if fill=='local' else balanced)
    gain,bias=np.asarray(cal)[lab].T
    return gain[:,None]*p+bias[:,None],gain*u+bias,np.column_stack((np.ones(len(x)),x)),s0

def basis(lt,p,u,f,dt,t):
    tau=np.exp(lt); age=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; a=age[:,None]/tau; q=dt[:,None]/tau; r=t[:,None]/tau
    ea=np.exp(-a); eq=np.exp(-q); er=np.exp(-r); w=ea*-np.expm1(-q); dw=a*w-q*ea*eq
    g=u[:,None]-p@w; dg=-p@dw; e=-np.expm1(-r)
    curve=np.concatenate((g[:,None,:]*e,t[None,:,None]*f[:,None,:]),axis=2)
    rates=np.column_stack((g/tau,f)); dc=dg[:,None,:]*e-g[:,None,:]*r*er; dr=(dg-g)/tau
    return curve,rates,dc,dr

def fit(p,u,f,dt,t,y,rate,sc=.025,sr=None,starts=None):
    ns=len(t)
    def ev(b,jac):
        C,R,dc,dr=basis(b[:2],p,u,f,dt,t); w=b[2:]; A=C.reshape(-1,7); D=dc.reshape(-1,2)
        if jac:
            a=np.column_stack((D*w[:2],A))/sc
            return a if sr is None else np.vstack((a,np.column_stack((dr*w[:2],R))/sr))
        a=(A@w-y.ravel())/sc
        return a if sr is None else np.r_[a,(R@w-rate)/sr]
    if starts is None: starts=list(combinations([.035,.09,.20,.45,1.,2.4,6.,15.],2))
    out=[]
    for pair in starts:
        lt=np.log(pair); C,R,_,_=basis(lt,p,u,f,dt,t); A=C.reshape(-1,7)
        # Least-squares initialization with bounded response leverage, then Cauchy refinement.
        z=y.ravel(); wt=1/np.maximum(1,np.abs(z)/max(np.quantile(abs(z),.90),.01))
        h=np.linalg.lstsq(A*wt[:,None],z*wt,rcond=None)[0]
        sol=least_squares(lambda b:ev(b,False),np.r_[lt,h],jac=lambda b:ev(b,True),loss='cauchy',bounds=(np.r_[np.log([.01,.01]),np.full(7,-np.inf)],np.r_[np.log([100.,100.]),np.full(7,np.inf)]),max_nfev=130,x_scale='jac')
        b=sol.x; ts=np.exp(b[:2]); order=np.argsort(ts); out.append((sol.cost,ts[order],np.r_[b[2:4][order],b[4:]]))
    return sorted(out,key=lambda z:z[0])

if __name__=='__main__':
    with open('/holdout_task/public_alpha_m97_c02_s00.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
    tr=d['train']; rate=tr['instant_rate_obs']; n=len(rate); t=d['eval_times']; dt=d['past_dt']; y=tr['strain_curves_obs']-tr['strain_0'][:,None]
    for fill in ('balanced','local'):
        p,u,f,_=data(d,fill); p,u,f=p[:n],u[:n],f[:n]
        for sr in (None,.1,.025,.006):
            a=fit(p,u,f,dt,t,y,rate,sr=sr); cost,tau,h=a[0]; C,R,*_=basis(np.log(tau),p,u,f,dt,t); ce=C@h-y; re=R@h-rate
            print(fill,'rate scale',sr,'best',round(cost,3),tau,h,'basins',[(round(c,2),q.round(4).tolist()) for c,q,_ in a[::6]],flush=True)
            print('resid quantiles .5/.8/.95: curve',np.quantile(abs(ce),[.5,.8,.95]),'rate',np.quantile(abs(re),[.5,.8,.95]),'rate corr',np.corrcoef(R@h,rate)[0,1],flush=True)
            if sr is None: print('rate obs vs pred',np.c_[rate,R@h][:10].round(4),'curve horizon medians',np.median(abs(ce),axis=0).round(4),flush=True)
