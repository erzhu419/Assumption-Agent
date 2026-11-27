import pickle,numpy as np
import physical98 as m
from scipy.optimize import least_squares
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=i['dataset'];g=i['campaign_diagnostics'];c=d['calibration_load_cells'];x=c['true_channel_loads'];y=c['bridge_raw'];good=np.max(np.abs(y-np.median(y,axis=0)),axis=1)<8
pred=np.zeros_like(x); folds=np.array_split(np.random.default_rng(98).permutation(len(x)),5)
for f in folds:
    keep=~np.isin(np.arange(len(x)),f); ix=f[good[f]]; pred[ix]=m._loads(y[ix],m._calibration(dict(true_channel_loads=x[keep],bridge_raw=y[keep])))
print('calibration CV clean 50/90/max total abs',np.quantile(np.abs((pred-x)[good].sum(1)),[.5,.9,1]),'n clean',good.sum())
for name,s in [('train',d['train']),('test',d['test']),('diag',g)]:
    mo=m._calibration(c); raw=np.concatenate((s['past_channel_raw'].reshape(-1,4),s['hold_channel_raw'])); a=m._loads(raw,mo); o,L,q,b=mo; print(name,'forward max',np.max(abs(a@L+o+q*a*a[:,m._NEIGHBOR]+b*a**3-raw)),'inverted range',a.min(),a.max())
tr,te=m._prepare(d); t=d['eval_times']; dt=d['past_dt']; target=lambda p:np.c_[p['strain_curves_obs']-p['strain_0'][:,None],p['instant_rate_obs']]; yt=target(d['train']); tau,th=m._fit(tr,dt,t,yt); r=m._design(tr,dt,t,tau)@th-yt;print('tau/theta',tau,th,'train MAD',np.median(abs(r[:,:-1])),np.median(abs(r[:,-1])))
for norm in ['diagnostic','joint']:
    if norm=='diagnostic': f=m._features(g,g)[-len(g['lab_id']):]
    else:
        both={k:np.concatenate([d['train'][k],d['test'][k]]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}; f=m._features(both,g)[-len(g['lab_id']):]
    cp=m._coupon(g,f,mo); past,hold,F=cp; dr=F@th[2:]; ex=np.r_[-np.expm1(-t[:,None]/tau),[1/tau]]; lin=np.r_[t,1]; el=hold[:,None]*(ex@th[:2]); his=m._design(cp,dt,t,tau)[:,:,:2]@th[:2]-el; Y=target(g)-el
    scale=np.r_[np.repeat(max(.001,1.4826*np.median(abs(r[:,:-1]))),len(t)),max(.001,1.4826*np.median(abs(r[:,-1])))]; base=dr[:,None]*lin; em=lambda k:np.r_[-np.expm1(-t/k)*k,1.]; n=len(Y); ids=np.arange(n)
    models={'linear':base[...,None], 'linear+offset':np.stack([base,np.broadcast_to(lin,base.shape)],-1),'linear+memory':np.stack([base,his],-1),'unit':(dr[:,None]*em(1.))[...,None], 'linear+unit':np.stack([base,dr[:,None]*em(1.)],-1),'scaled hold+linear':np.stack([el,base],-1), 'branch':dr[:,None,None]*ex,'linear+branches':np.concatenate([base[...,None],dr[:,None,None]*ex],axis=2), 'free linear':F[:,None,:]*lin[None,:,None]}
    print(norm,'scale',scale[[0,-1]])
    for nm,A in models.items():
        def fit(ix):
            a=(A[ix]/scale[None,:,None]).reshape(-1,A.shape[-1]); b=(Y[ix]/scale).ravel(); p=np.linalg.lstsq(a,b,rcond=None)[0]; return least_squares(lambda p:a@p-b,p,loss='soft_l1').x
        err=np.array([A[j]@fit(ids[ids!=j])-Y[j] for j in ids]); print(nm,np.round(fit(ids),6),'LOO MAD q90 max',*[np.round(np.quantile(abs(v),[.5,.90,1]),6) for v in [err[:,:-1],err[:,-1]]])
    print('diagnostic initial/late remaining per coupon',np.round(np.c_[dr,Y[:,-1],Y[:,-2]/t[-1]],4))
