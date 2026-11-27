import pickle,numpy as np
from scipy.optimize import least_squares
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
t=p['eval_times']; dt=p['past_dt']; tr=p['train']; ch=p['chamber_verification_trials']; te=p['test']; c=p['calibration_load_cells']; b=np.linalg.lstsq(np.c_[c['bridge_raw'],np.ones(len(c['bridge_raw']))],c['true_channel_loads'],rcond=None)[0].sum(1)
def prep(q, ref=None):
    if ref is None: ref=q
    x=q['coupon_features_raw'].copy(); lab=q['lab_id']; s=q['stratum']; rl=ref['lab_id']
    for l in np.unique(lab):
        r=ref['ref_coupons'][rl==l]; r=r if len(r)>1 else q['ref_coupons'][lab==l]; r=r if len(r)>1 else ref['ref_coupons']; x[lab==l]=(x[lab==l]-np.median(r.mean(0)))/max(1e-8,np.median(r.std(0)))
    _,j,n=np.unique(np.c_[lab,s],axis=0,return_inverse=True,return_counts=True); w=(1./n[j])[:,None]; x=np.where(np.isfinite(x),x,np.nansum(w*x,0)/np.sum(w*np.isfinite(x),0)); past=q['past_channel_raw']@b[:4]+b[4]; hold=q['hold_channel_raw']@b[:4]+b[4]; return x,past,hold

def designs(q, tau, mode='z',ref=None):
    x,past,hold=prep(q,ref); before=np.exp(-np.r_[np.cumsum(dt[:0:-1])[::-1],0][:,None]/tau); m=(past@(before*-np.expm1(-dt[:,None]/tau))); ini=np.exp(-dt.sum()/tau)
    if mode=='z': a=hold[:,None]-m
    if mode=='first': a=hold[:,None]-m-past[:,0,None]*ini
    if mode=='rev': a=hold[:,None]-past[:,::-1]@(before*-np.expm1(-dt[:,None]/tau))
    if mode=='mem': a=m
    if mode=='memfirst': a=m+past[:,0,None]*ini
    if mode=='hold': a=np.broadcast_to(hold[:,None],m.shape)
    if mode=='jump': a=np.broadcast_to((hold-past[:,-1])[:,None],m.shape)
    if mode=='plus': a=m+hold[:,None]
    d=np.c_[np.ones(len(x)),x]; return np.concatenate([a[:,None,:]*(-np.expm1(-t[None,:,None]/tau)),t[None,:,None]*d[:,None,:]],axis=2),np.c_[a/tau,d]

def target(q): return q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']
def fit(q,mode,ref=None):
    y,r=target(q)
    def residual(par):
        tau=np.sort(np.exp(par[:2])); A,B=designs(q,tau,mode,ref); return np.r_[(A@par[2:]-y).ravel(),0.7*(B@par[2:]-r)]
    best=None
    for init in [(0.25,2),(.6,5),(1.5,12)]:
        A,B=designs(q,np.array(init),mode,ref); w=np.linalg.lstsq(np.r_[A.reshape(-1,7),0.7*B],np.r_[y.ravel(),0.7*r],rcond=None)[0]; o=least_squares(residual,np.r_[np.log(init),w],bounds=([np.log(.05001)]*2+[-np.inf]*7,[np.log(24.999)]*2+[np.inf]*7),loss='soft_l1',f_scale=.005,max_nfev=160)
        if best is None or o.cost<best.cost: best=o
    tau=np.exp(best.x[:2]); w=best.x[2:].copy(); idx=np.argsort(tau); return tau[idx],w

def err(q,tau,w,mode,ref=None):
    A,B=designs(q,tau,mode,ref); y,r=target(q); E=A@w-y; R=B@w-r; return np.round([np.median(abs(E)),np.quantile(abs(E),.90),np.median(abs(R)),np.quantile(abs(R),.9)],6)
if __name__=='__main__':
 for n,q in [('train',tr),('chamber',ch),('test',te)]:
    print(n, 'telemetry', np.unique(q.get('aux_chamber_telemetry',np.zeros((1,4))),axis=0)[:5], 'std', np.std(q.get('aux_chamber_telemetry',np.zeros((1,4))),axis=0)); print('lab references', [(l,int(sum(q['lab_id']==l)),round(np.median(q['ref_coupons'][q['lab_id']==l].mean(0)),3),round(np.median(q['ref_coupons'][q['lab_id']==l].std(0)),3)) for l in np.unique(q['lab_id'])]);
    if 'instant_rate_obs' in q:
        y,r=target(q); print('curve/rate percentiles',np.percentile(y,[0,10,50,90,100]),np.percentile(r,[0,10,50,90,100]))
 for mode in ('z','first','rev','mem','memfirst','hold','jump','plus'):
    for q,other in [(tr,ch),(ch,tr)]:
        tau,w=fit(q,mode); print(mode,'bench' if q is tr else 'chamb',np.round(tau,4),np.round(w,5),err(q,tau,w,mode),err(other,tau,w,mode),flush=True)
