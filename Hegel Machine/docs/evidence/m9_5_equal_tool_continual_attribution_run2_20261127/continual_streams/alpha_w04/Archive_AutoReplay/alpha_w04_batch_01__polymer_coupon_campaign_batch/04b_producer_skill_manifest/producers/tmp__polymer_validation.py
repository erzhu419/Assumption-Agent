import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w04_batch_01_solver.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
w=pickle.load(open('/holdout_task/public_alpha_w04_batch_01.pkl','rb'))[0];d=w['dataset']; a,b=d['train'],d['test']; q=w['gt']; dt=d['past_dt']; t=d['eval_times']; x,xt=m._align(a,b); cal=m._calibrate(d['calibration_load_cells']); h,s=m._stress(a,cal); ht,st=m._stress(b,cal); y=a['strain_curves_obs']-a['strain_0'][:,None]; r=a['instant_rate_obs']; z=a['initial_memory_states']; lab=a['lab_id']; ok=np.isfinite(a['coupon_features_raw']).all(1)

def fit(ids,kind):
    tau,theta=m._fit(dt,t,h[ids],s[ids],x[ids],y[ids],r[ids],z[ids])
    if not kind:return tau,theta,np.zeros(4)
    labs=np.eye(4)[lab[ids],:3]-np.eye(4)[lab[ids],3:]; tar=np.r_[y[ids].ravel(),r[ids]]; n=int(ids.sum())
    def res(p):
        v,u=m._design(p[:2],dt,t,h[ids],s[ids],x[ids],z[ids]); offset=labs@p[9:];
        return np.r_[(v@p[2:9]+offset[:,None]*t).ravel(),u@p[2:9]+offset]-tar
    p=np.r_[tau,theta,np.zeros(3)]; er=res(p); mad=lambda v:max(1e-6,1.4826*np.median(abs(v-np.median(v))));
    scale=np.r_[np.repeat(1.,n*len(t))*mad(er[:-n]),np.repeat(mad(er[-n:]),n)]*kind
    f=least_squares(lambda p:res(p)/scale,p,bounds=(np.r_[.2,.3,[-np.inf]*10],np.r_[8.,10.,[np.inf]*10]),loss='cauchy',max_nfev=70,x_scale='jac').x
    return f[:2],f[2:9],np.r_[f[9:],-f[9:].sum()]

rng=np.random.default_rng(42); fold=np.empty(len(y),int)
for g in range(4):
    ix=np.flatnonzero(lab==g); rng.shuffle(ix); fold[ix]=np.arange(len(ix))%5
for kind in [0,1.5,2.385,4.]:
    mm={f:np.empty_like(y) for f in [0.,.25,.5,1.]}; params=[]
    for i in range(5):
        te=fold==i; tau,theta,off=fit((~te)&ok,kind); params.append(tau); v,u=m._design(tau,dt,t,h[te],s[te],x[te],z[te]); base=v[:,:,:2]@theta[:2]; f=theta[2]+x[te]@theta[3:]+off[lab[te]]; delta=r[te]-u[:,:2]@theta[:2]-f
        for share in mm:
            drift=f+np.where(ok[te],share*np.clip(delta,-.035,.035),delta); mm[share][te]=base+drift[:,None]*t
    print('variant',kind,'OOF tau range',np.round(np.array(params).min(0),4),np.round(np.array(params).max(0),4))
    for share,pred in mm.items():
        e=abs(pred-y); print(' share',share,'worst lab terminal median',max(np.median(e[(lab==j)&ok,-1]) for j in range(4)),'all late median',np.median(e[ok,-3:]),'worst lab full clipped RMSE',max(np.mean(np.minimum(e[(lab==j)&ok],.075)**2)**.5 for j in range(4)))
    tau,theta,off=fit(ok,kind); print(' fitted',tau,off); v,u=m._design(tau,dt,t,ht,st,xt,b['initial_memory_states']); full=np.isfinite(b['coupon_features_raw']).all(1); f=theta[2]+xt@theta[3:]+off[b['lab_id']]; delta=b['instant_rate_obs']-u[:,:2]@theta[:2]-f
    for share in mm:
        p=b['strain_0'][:,None]+v[:,:,:2]@theta[:2]+(f+np.where(full,share*np.clip(delta,-.035,.035),delta))[:,None]*t
        ee=[p-q['te_clean'],(p[:,-1]-b['strain_0'])/np.maximum(abs(st),.5)-q['te_secant'],m._crossing(t,b['strain_0'],p,b['critical_strain'])-q['te_cross']]; print(' public',share,[round(max(np.mean(e[b['lab_id']==j]**2)**.5 for j in range(4)),5) for e in ee])
