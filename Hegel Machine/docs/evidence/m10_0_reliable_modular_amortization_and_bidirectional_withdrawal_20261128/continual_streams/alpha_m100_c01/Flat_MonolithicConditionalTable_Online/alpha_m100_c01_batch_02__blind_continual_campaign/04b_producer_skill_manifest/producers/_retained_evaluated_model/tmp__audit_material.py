import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares, minimize_scalar
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr=d['train']; g=p['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; c=d['calibration_load_cells']; b=m._bridge(c['bridge_raw'],c['true_channel_loads']); x,_=m._features(tr,d['test']); xg,_=m._features(g,{k:v[:0] for k,v in g.items()}); P,H,F=m._inputs(tr,x,b); Pg,Hg,Fg=m._inputs(g,xg,b)
y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; yg=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]

def pred(a,w,P,H,F):
    C,R=m._design(a,P,H,F,dt,t); return np.c_[C@w,R@w]

def fit(i,loss,f,rate_weight):
    a,w=m._fit(P[i],H[i],F[i],dt,t,y[i,:-1],y[i,-1])
    if loss is None: return a,w
    def res(q):
        e=pred(np.exp(q[:2]),q[2:],P[i],H[i],F[i])-y[i]; e[:,-1]*=rate_weight; return e.ravel()
    q=least_squares(res,np.r_[np.log(a),w],loss=loss,f_scale=f,x_scale='jac').x
    return np.exp(q[:2]),q[2:]

labs=tr['lab_id']; miss=np.isnan(tr['coupon_features_raw']).any(1); dg=[g['lab_id'],g['stratum'],np.isnan(g['coupon_features_raw']).any(1)]
for setting in [(None,0,1),('soft_l1',.003,1),('soft_l1',.016,1),('soft_l1',.008,2),('soft_l1',.008,4),('cauchy',.016,2),('cauchy',.009,3),('linear',1,2)]:
    a,w=fit(np.ones(len(y),bool),*setting); e=np.empty_like(y); params=[]
    for lab in np.unique(labs):
        j=labs==lab; aa,ww=fit(~j,*setting); params.append(aa); e[j]=pred(aa,ww,P[j],H[j],F[j])-y[j]
    slices=[labs==v for v in np.unique(labs)]+[miss,~miss]
    worst=np.max([np.mean(np.abs(e[s]),axis=0) for s in slices],axis=0)
    base=pred(a,w,Pg,Hg,Fg); u=Fg@w[2:]; ee=[]
    for groups in dg:
        for v in np.unique(groups):
            j=groups==v
            def err(tau):
                correction=u[:,None]*(-tau*np.expm1(-t/tau)-t)
                return np.c_[base[:,:-1]+correction,base[:,-1]]-yg
            tau=minimize_scalar(lambda h: np.mean(err(h)[~j,:-1]**2),bounds=(.5,2),method='bounded').x
            ee.append(np.sqrt(np.mean(err(tau)[j]**2,axis=0)))
    worstg=np.max(ee,axis=0)
    print(setting,'tau',a.round(6),'LOLO tau range',np.min(params,axis=0).round(6),np.max(params,axis=0).round(6),'train worst horizon/rate MAE',np.round([worst[:-1].max(),worst[-1]],6),'diag worst horizon/rate RMSE',np.round([worstg[:-1].max(),worstg[-1]],6),flush=True)
