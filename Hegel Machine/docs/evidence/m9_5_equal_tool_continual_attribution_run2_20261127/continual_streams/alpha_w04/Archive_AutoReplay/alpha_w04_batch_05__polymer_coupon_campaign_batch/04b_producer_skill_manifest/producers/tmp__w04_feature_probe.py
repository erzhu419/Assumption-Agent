import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import StratifiedKFold
p='/holdout_task/alpha_w04_batch_05_solver.py'; spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
pub=pickle.load(open('/holdout_task/public_alpha_w04_batch_05.pkl','rb'))[0]; d=pub['dataset']; tr=d['train']; te=d['test']; t=d['eval_times']; late=t>=1.35

def trial(a,b):
    out=m.solve_alpha_w04_batch_05({**d,'train':a,'test':b}); w=np.r_[out['hereditary_weights'],out['viscoplastic_weights']]
    x,xt=m._align(a,b); cal=m._calibrate(d['calibration_load_cells'])
    A,B=m._design(out['estimated_taus'],d['past_dt'],t,*m._stress(a,cal),x,d['yield_stress'])
    train_rate_err=a['instant_rate_obs']-B@w
    strain_err=a['strain_curves_obs']-a['strain_0'][:,None]-A@w
    slopes=np.median(strain_err[:,late]/t[late],axis=1)
    te_innovation=b['instant_rate_obs']-out['initial_rate_pred']; corrections=np.zeros((5,len(te_innovation))); fits=[]
    for c in range(4):
        good=(a['lab_id']==c)&np.isfinite(a['coupon_features_raw']).all(axis=1)
        target=(b['lab_id']==c)&np.isfinite(b['coupon_features_raw']).all(axis=1)
        if good.sum()<8: continue
        xx,yy=train_rate_err[good],slopes[good]
        z=least_squares(lambda z: z[0]+z[1]*xx-yy,[0.,.1],bounds=([-np.inf,0],[np.inf,1]),loss='soft_l1',f_scale=.004).x
        fits.append((c,z.tolist()))
        r=np.clip(te_innovation[target],*np.quantile(xx,[.02,.98]))
        corrections[:2,target]=[z[1]*r,np.median(yy)*np.ones_like(r)]
        for j, shrink in enumerate((2., 6., 15.)):
            X=np.c_[np.ones(good.sum()), x[good], xx/.012]
            XT=np.c_[np.ones(target.sum()), xt[target], r/.012]
            def fun(q):
                return np.r_[(X@q-yy)/.003, np.sqrt(shrink)*q[1:5]/.003]
            fit=least_squares(fun,np.r_[z[0],np.zeros(4),z[1]*.012],bounds=(np.r_[np.full(5,-np.inf),0.],np.r_[np.full(5,np.inf),.012]),loss='soft_l1').x
            if j==1: fits.append((c,np.round(fit,6).tolist()))
            corrections[2+j,target]=XT@fit
    return out,corrections,fits

def report(label,split,s,r,cs):
    y=split['strain_curves_obs']; obs=split['instant_rate_obs']; ss=s[None]-y+cs[:,:,None]*t; rr=r[None]-obs+cs
    for c in [-1,0,1,2,3]:
        v=np.ones(len(r),bool) if c<0 else split['lab_id']==c
        es=ss[:,v]; er=rr[:,v]; slopes=np.median(es[:,:,late]/t[late],axis=2)
        print(label,c,'strain_RMSE',np.round(np.sqrt(np.mean(es**2,axis=(1,2))),5),'slope_MAE',np.round(np.mean(abs(slopes),axis=1),5),'rate_RMSE',np.round(np.sqrt(np.mean(er**2,axis=1)),5))

s=np.empty_like(tr['strain_curves_obs']); r=np.empty_like(tr['instant_rate_obs']); cs=np.zeros((6,len(r)))
for fit,val in StratifiedKFold(5,shuffle=True,random_state=94).split(tr['lab_id'],tr['lab_id']):
    out,c,_=trial({k:v[fit] for k,v in tr.items()},{k:v[val] for k,v in tr.items() if k!='strain_curves_obs'})
    s[val]=out['predicted_strain']; r[val]=out['initial_rate_pred']; cs[1:,val]=c
report('OOF',tr,s,r,cs)
out,cs,fits=trial(tr,te); print('lab fits intercept, innovation-weight and feature slopes',fits)
report('public clean',{**te,'strain_curves_obs':pub['gt']['te_clean'],'instant_rate_obs':pub['gt']['te_true_rate0']},out['predicted_strain'],out['initial_rate_pred'],np.vstack((np.zeros(len(te['lab_id'])),cs)))
