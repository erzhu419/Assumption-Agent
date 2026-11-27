import importlib.util, pickle
import numpy as np

spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: item=pickle.load(f)[0]
d=item['dataset']; c=item['campaign_diagnostics']; t=d['eval_times']; dt=d['past_dt']
out=m.solve_batch(d); tau=out['estimated_taus']; w=out['hereditary_weights']; assert .05<tau[0]<tau[1]<25 and all(np.isfinite(v).all() for v in out.values())
x=m._features(c,c)[0]; p,h=m._loads(c,m._bridge(d['calibration_load_cells'])); a,b=m._design(tau,x,p,h,dt,t)
r=w[2]+x@w[3:]; lab=c['lab_id']; complete=np.isfinite(c['coupon_features_raw']).all(axis=1)
u=-np.expm1(-t[:,None]/tau)*tau-t[:,None]; target=c['strain_curves_obs']-c['strain_0'][:,None]-a@w; rate=b@w-c['instant_rate_obs']; z=h[:,None]-b[:,:2]*tau
# Competing algebraic grammars share the onset invariant. Select grammar and ridge only on inner labs.
q=r[:,None]*np.ones((1,1)); grammars={'conserved':q,'memory contrasts':np.c_[q,r[:,None]*np.c_[h-z[:,0],h-z[:,1],z[:,0]-z[:,1],p[:,-1]-p[:,0]]],'cubic':np.c_[q,r*r,r**3,r*(z[:,0]-z[:,1])*(h-p[:,-1])],**{f'rational {k}':np.c_[q,r/(1+k*r*r),r/(1+k*(z[:,0]-z[:,1])**2)] for k in (.25,1.,4.)}}
A={k:(v[:,:,None,None]*u[None,None,:,:]).transpose(0,2,1,3).reshape(len(r),len(t),-1) for k,v in grammars.items()}
def predict(key,ridge,tr,te):
    f=A[key]; design=f[tr].reshape(-1,f.shape[-1]); rhs=target[tr].ravel(); n=design.shape[1]; scale=np.maximum(np.linalg.norm(design,axis=0),1e-12)
    reg=np.sqrt(ridge)*np.diag(scale); reg[:2,:2]=0
    coef=np.linalg.lstsq(np.vstack((design,reg)),np.r_[rhs,np.zeros(n)],rcond=None)[0]
    return f[te]@coef

def rmse(e): return np.sqrt(np.mean(e**2))
base=np.empty_like(target); alt=np.empty_like(target)
for k in np.unique(lab):
    tr=complete&(lab!=k); te=lab==k
    def inner(option):
        return np.mean([rmse(predict(*option,tr&(lab!=g),tr&(lab==g))-target[tr&(lab==g)])**2 for g in np.unique(lab[tr])])
    opts=[(key,reg) for key in A for reg in ((0.,) if key=='conserved' else (.01,.1,1.))]; best=min(opts,key=inner)
    base[te]=predict('conserved',0.,tr,te)-target[te]; alt[te]=predict(*best,tr,te)-target[te]
    print('outer lab',k,'selected',best,'strain RMSE baseline/selected/rate',*[round(rmse(v[te]),6) for v in (base,alt,rate)])
for name,mask in [('complete',complete),('missing',~complete)]:
    print(name,'strain baseline/selected, rate RMSE',*[round(rmse(v[mask]),6) for v in (base,alt,rate)])
