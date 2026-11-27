import pickle, importlib.util, numpy as np
from scipy.optimize import least_squares
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=p['dataset']; q=p['campaign_diagnostics']; tr=ds['train']; te=ds['test']; t=ds['eval_times']; dt=ds['past_dt']; cal=ds['calibration_load_cells']
s=importlib.util.spec_from_file_location('arc','/tmp/hegel_continual_archive/alpha_m98_v2_c04_batch_06_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
def stack(*blocks): return {k:np.concatenate([b[k] for b in blocks]) for k in blocks[0] if all(k in b for b in blocks)}
X=m._features(tr,stack(te,q)); n=len(tr['lab_id']); x=X[-len(q['lab_id']):]; B=m._fit_bridge(cal); pg,h=m._loads(q,B); tau,th=m._fit(X[:n],*m._loads(tr,B),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); d=th[2]+x@th[3:]; lab=q['lab_id']; y=np.c_[q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']]; one=np.ones(len(h)); later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]
def z(tt): return pg@(-np.expm1(-dt/tt)*np.exp(-later/tt))
def fit(mode, take):
    def parts(par):
        a,b,c,e=np.exp(par)
        g,j=z(c),z(e); H=h-g; J=h-j
        if mode=='memory': u,v,w=np.c_[H],np.c_[J,d],np.c_[one,h,d]
        elif mode=='split': u,v,w=np.c_[h,g],np.c_[h,j,d],np.c_[one,h,d]
        elif mode=='two_drift': u,v,w=np.c_[h,g,d],np.c_[h,j,d],np.c_[one,h,d]
        elif mode=='both_memories': u,v,w=np.c_[h,g,j,d],np.c_[h,g,j,d],np.c_[one,h,d]
        elif mode=='drift_only': u,v,w=np.c_[h,g],np.c_[h,j],np.c_[one,h,d]
        elif mode=='saturating': u,v,w=np.c_[h,g],np.c_[h,j,d],np.c_[one,h,d,d*d]
        elif mode=='hinge': u,v,w=np.c_[h,g],np.c_[h,j,d],np.c_[one,h,d,abs(d)]
        elif mode=='features': u,v,w=np.c_[h,g],np.c_[h,j,d],np.c_[one,h,x]
        elif mode=='joint_features': u,v,w=np.c_[h,g],np.c_[h,j,x],np.c_[one,h,x]
        elif mode=='no_slow_memory': u,v,w=np.c_[h,g],np.c_[h,d],np.c_[one,h,d]
        ba=np.r_[-np.expm1(-t/a),1/a]; bb=np.r_[-np.expm1(-t/b),1/b]; bt=np.r_[t,1.]
        return np.concatenate([f[:,None,:]*b[None,:,None] for f,b in [(u,ba),(v,bb),(w,bt)]],axis=2), [u.shape[1],v.shape[1],w.shape[1]]
    def pred(p): return parts(p[:4])[0]@p[4:]
    fits=[]; scale=np.r_[np.full(len(t),.013),.013]
    for start in [(.47,1.5,.47,1.5),(.47,3,.47,4.4),(.47,1.5,.47,.8)]:
        pp=np.log(start); D,_=parts(pp); co=np.linalg.lstsq(D[take].reshape(-1,D.shape[-1]),y[take].ravel(),rcond=None)[0]; bounds=(np.r_[np.log([.25,.65,.15,.15]),np.full(len(co),-np.inf)],np.r_[np.log([.65,10,3,15]),np.full(len(co),np.inf)]); fits.append(least_squares(lambda p:((pred(p)[take]-y[take])/scale).ravel(),np.r_[pp,co],bounds=bounds,loss='soft_l1',max_nfev=220))
    f=min(fits,key=lambda f:f.cost); return pred(f.x)-y,np.exp(f.x[:4]),np.split(f.x[4:],np.cumsum(parts(f.x[:4])[1])[:-1])
if __name__=='__main__':
 for mode in ['memory','split','two_drift','both_memories','drift_only','saturating','hinge','features','joint_features','no_slow_memory']:
    err,pars,co=fit(mode,np.ones(len(y),bool)); cv=np.zeros_like(y)
    for l in np.unique(lab):
        e,*_=fit(mode,lab!=l); cv[lab==l]=e[lab==l]
    def report(e): return np.round([np.median(abs(e[:,:-1])),np.median(abs(e[:,-1])),max(np.median(abs(e[lab==l,:-1])) for l in np.unique(lab)),max(np.median(abs(e[lab==l,-1])) for l in np.unique(lab))],5)
    print(mode,'train',report(err),'cv',report(cv), 'times',np.round(pars,3),'co', [np.round(k,4) for k in co],flush=True)
