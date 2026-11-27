import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('m','/holdout_task/alpha_m97_c02_batch_04_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def design(data,dt,times,tau,mode):
    x,z=m._design(*data,dt,times,tau)
    if mode=='log':
        inactive=data[-1]<=0
        x[inactive,:,2:7] *= (np.log1p(times)/times)[None,:,None]
    return x,z

def fit(data,dt,times,y,r,mode,curve_scale=.055,rate_scale=.13):
    response=np.r_[y.ravel(),r]; n=y.size
    scales=np.r_[np.full(n,curve_scale),np.full(len(r),rate_scale)]
    def matrix(p):
        c,z=design(data,dt,times,np.exp(p),mode)
        return np.vstack((c.reshape(-1,9),z))
    p=np.log([.6,3.5]); sw=np.ones(len(response)); bounds=(np.log(.035),np.log(60.))
    for step in range(25):
        b=sw*response
        def profile(p,ret=False):
            a=matrix(p)*sw[:,None]
            w=np.linalg.lstsq(a,b,rcond=None)[0]
            return w if ret else np.einsum('ij,j->i',a,w)-b
        p=least_squares(profile,p,bounds=bounds,max_nfev=45,ftol=1.e-8,xtol=1.e-8).x
        w=profile(p,True)
        e=matrix(p)@w-response
        s=scales*(3 if step<3 else 1)
        new=1/np.sqrt(s*s+e*e)
        if step>4 and np.max(abs(new/sw-1)) < 1.e-4: break
        sw=new
    return np.exp(p),w

if __name__=='__main__':
    with open('/holdout_task/public_alpha_m97_c02_batch_04.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
    t=d['train']; n=len(t['lab_id']); a=tuple(x[:n] for x in m._prepare(d)); dt,ts=d['past_dt'],d['eval_times']; y=t['strain_curves_obs']-t['strain_0'][:,None]; r=t['instant_rate_obs']; ok=np.isfinite(t['coupon_features_raw']).all(1)
    errs={}
    for mode in ('baseline','linear','log'):
        e=np.empty_like(y); er=np.empty_like(r)
        for c in (0,1,2,3,None):
            tr=ok if c is None else ok&(t['lab_id']!=c); va=np.ones(n,dtype=bool) if c is None else (t['lab_id']==c)
            data=tuple(x[tr] for x in a)
            if mode=='baseline': tau,w=m._fit(*data,dt,ts,y[tr],r[tr])
            else: tau,w=fit(data,dt,ts,y[tr],r[tr],mode)
            if c is None:
                print(mode,'full parameters',tau.round(5),w.round(5)); continue
            x,z=design(tuple(x[va] for x in a),dt,ts,tau,mode); e[va]=y[va]-x@w; er[va]=r[va]-z@w
        errs[mode]=(e,er)
        print(mode,'complete damage/time residual signed medians',[np.median(e[ok&((a[-1]>0)==p)],axis=0).round(3).tolist() for p in (True,False)])
    # Compare all models on identical observations, retaining an uncensored rate metric.
    common=abs(errs['linear'][0])<.6
    for mode,(e,er) in errs.items():
        scores=[]
        for c in range(4):
            for complete in (True,False):
                for active in (True,False):
                    row=(t['lab_id']==c)&(ok==complete)&((a[-1]>0)==active)
                    if not row.any(): continue
                    ec=e[row]; kept=common[row]
                    scores.append((round(np.sqrt(np.mean(ec[kept]**2)),4),round(np.sqrt(np.mean(er[row]**2)),4),c,complete,active,int(row.sum())))
        print(mode,'worst clean curve',max(scores),'worst rate',max(scores,key=lambda x:x[1]),'pooled median abs curve/rate',np.median(abs(e)),np.median(abs(er)))
        for active in (True,False):
            row=ok&((a[-1]>0)==active)
            print(mode,'complete active',active,'curve/rate clean rms',np.sqrt(np.mean(e[row][common[row]]**2)),np.sqrt(np.mean(er[row]**2)))
    print('common mask omitted points',int((~common).sum()),'of',common.size)
