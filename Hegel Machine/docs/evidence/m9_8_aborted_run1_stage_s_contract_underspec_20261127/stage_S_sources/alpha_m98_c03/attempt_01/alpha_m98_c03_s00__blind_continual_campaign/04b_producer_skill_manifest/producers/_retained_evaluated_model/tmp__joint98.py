import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)

def joint(s, stress, dt, t, noise, design, initial):
    u,h=stress
    y=np.c_[s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']]
    z=np.r_[t,1.]
    scale=np.r_[np.full(len(t),noise),noise*.75]
    drift_basis=design[:,None,:]*(z/scale)[None,:,None]
    def fun(p):
        g=m._kernel(np.exp(p[:2]),dt,t,u,h)
        return ((g@p[2:4]+(design@p[4:])[:,None]*z-y)/scale).ravel()
    def jac(p):
        tau=np.exp(p[:2]); g=m._kernel(tau,dt,t,u,h)
        gp=(m._kernel(tau*np.exp(1e-5),dt,t,u,h)-g)*1e5
        return np.concatenate((gp*p[None,None,2:4]/scale[None,:,None],g/scale[None,:,None],drift_basis),axis=-1).reshape(-1,len(p))
    lo=np.r_[np.log([.050001]*2),0.,0.,np.full(5,-np.inf)]
    hi=np.r_[np.log([24.9999]*2),np.full(7,np.inf)]
    sol=least_squares(fun,initial,jac=jac,loss='cauchy',bounds=(lo,hi),max_nfev=150).x
    ix=np.argsort(sol[:2]); return np.exp(sol[:2][ix]),sol[2:4][ix],sol[4:]

if __name__=='__main__':
    d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; train,test,pilot=[d[k] for k in ('train','test','chamber_verification_trials')]
    dt,t=d['past_dt'],d['eval_times']; cal=m._calibrate(d['calibration_load_cells'])
    for label,source,noise in [('bench',train,.012),('pilot',pilot,.006)]:
        panel=m._panels(train,test) if label=='bench' else m._panels(pilot)
        x=m._standardize(source,panel); complete=np.isfinite(x).all(1)
        fill=m._impute(np.r_[x,m._standardize(test,panel)],train,test) if label=='bench' else m._impute(x,pilot)
        a=m._design(x,fill); scores={}
        for lab in [*np.unique(source['lab_id']),None]:
            keep=(source['lab_id']!=lab) if lab is not None else np.ones(len(x),bool)
            sub={k:v[keep] for k,v in source.items()}; stress=m._stress(sub,cal)
            tau,w,dr=m._fit_relaxation(sub,stress,dt,t,noise); coef=m._fit_drift(a[keep],dr,.5*noise)
            p=np.r_[np.log(tau),w,coef]
            cc=keep&complete
            opts={'profile':(tau,w,coef),'joint':joint(sub,stress,dt,t,noise,a[keep],p),'complete':joint({k:v[cc] for k,v in source.items()},m._stress({k:v[cc] for k,v in source.items()},cal),dt,t,noise,a[cc],p)}
            for method,(tau,w,coef) in opts.items():
                va=~keep if lab is not None else keep
                sample={k:v[va] for k,v in source.items()}; z=m._kernel(tau,dt,t,*m._stress(sample,cal))@w; rate=a[va]@coef
                e=abs(sample['strain_0'][:,None]+z[:,:-1]+rate[:,None]*t-sample['strain_curves_obs']); r=abs(z[:,-1]+rate-sample['instant_rate_obs']); miss=~complete[va]
                metrics=np.array([np.median(e),np.quantile(e,.9),r.mean(),e[:,-1].mean(),e[miss].mean() if miss.any() else 0]); print(label,lab,method,metrics.round(5),np.r_[tau,w,coef].round(4));
                if lab is not None:scores.setdefault(method,[]).append(metrics)
        print(label,'WORST', {k:np.max(v,axis=0).round(5).tolist() for k,v in scores.items()})
