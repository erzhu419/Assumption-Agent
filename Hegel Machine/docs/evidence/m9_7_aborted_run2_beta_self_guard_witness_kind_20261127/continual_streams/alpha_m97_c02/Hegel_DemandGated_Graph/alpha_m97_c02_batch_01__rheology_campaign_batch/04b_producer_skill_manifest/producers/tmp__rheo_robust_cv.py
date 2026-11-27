import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import alpha_m97_c02_batch_01_solver as m

def robust(dt,t,b,y,r,scale,rate_weight,log=True):
    u=np.r_[y.ravel(),r]
    fixed=np.r_[np.ones(y.size),np.full(r.size,rate_weight)]
    per_curve=np.tile(scale*(1+0.35*t),len(y))
    noise=np.r_[per_curve,np.full(r.size,scale*2)]
    def proj(x,full=False):
        c,a=m._basis(np.exp(x),dt,t,b,log)
        mat=np.concatenate((c.reshape(-1,9),a))
        w=np.linalg.lstsq(mat*wt[:,None],u*wt,rcond=None)[0]
        err=mat@w-u
        return (w,err) if full else err*wt
    wt=fixed
    starts=[np.log([.45,2.7]),np.log([.8,7.])]
    for iteration in range(12):
        opt=min((least_squares(proj,x,bounds=(np.log(.005),np.log(250.)),max_nfev=45) for x in starts),key=lambda f:f.cost)
        w,e=proj(opt.x,True)
        new=fixed/np.hypot(1,e/noise)
        if np.max(np.abs(new-wt))<1e-4: break
        wt=new
        starts=[opt.x]
    return np.exp(opt.x),w

if __name__=='__main__':
    d=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]['dataset']; dt,t,(b,_)=m._prepare(d); s=d['train']; y=s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs']; labs=s['lab_id']; active=b[-1]>0
    rm=lambda x: np.sqrt(np.mean(x*x))
    for scale,q,log in [(None,1,True),(.025,2,True),(.06,1,True),(.06,2,True),(.06,4,True),(.14,2,True),(.06,2,False)]:
        fit=(lambda bb,yy,rr:m._fit(dt,t,bb,yy,rr,q,log)) if scale is None else (lambda bb,yy,rr:robust(dt,t,bb,yy,rr,scale,q,log))
        ec=np.empty_like(y); er=np.empty_like(r); est=[]
        for lab in np.unique(labs):
            te=labs==lab; tr=~te
            tau,w=fit(tuple(x[tr] for x in b),y[tr],r[tr]); est.append(tau)
            c,a=m._basis(tau,dt,t,tuple(x[te] for x in b),log); ec[te]=c@w-y[te]; er[te]=a@w-r[te]
        wc=max(np.median(abs(ec[labs==lab,k])) for lab in np.unique(labs) for k in range(len(t)))
        wr=max(rm(er[(labs==lab)&(active==on)]) for lab in np.unique(labs) for on in (False,True))
        wp=max(np.quantile(abs(ec[(labs==lab)&(active==on)]),.90) for lab in np.unique(labs) for on in (False,True))
        wh=max(rm(ec[labs==lab,k]) for lab in np.unique(labs) for k in range(len(t)))
        print(scale,q,log,'worst horizon median/raw RMSE, lab-regime curve P90/rate RMSE',np.round([wc,wh,wp,wr],5),'taus',np.round(est,4).tolist(),flush=True)
        tau,w=fit(b,y,r); print('all',np.round(tau,5),np.round(w,5),flush=True)
