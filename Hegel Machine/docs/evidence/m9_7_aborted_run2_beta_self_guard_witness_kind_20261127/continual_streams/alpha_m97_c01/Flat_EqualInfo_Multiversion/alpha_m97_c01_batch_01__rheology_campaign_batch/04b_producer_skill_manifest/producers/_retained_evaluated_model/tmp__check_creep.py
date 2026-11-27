import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c01_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0]['dataset']; tr=d['train']; n=len(tr['lab_id']); t,dt=d['eval_times'],d['past_dt']
p,h,x,s=m._inputs(d); p,h,x,s=p[:n],h[:n],x[:n],s[:n]; a=np.tanh(np.maximum(abs(p)-d['crit_stress'],0)**2@dt); print('thresholds',d['crit_stress'],d['yield_stress'])
y=np.column_stack((tr['strain_curves_obs']-s[:,None],tr['instant_rate_obs']))
def model(q,kind):
    obj=m._Model(dt,t,p[q],h[q],x[q],a[q]); z=a[q]==0
    if kind=='log': obj.drift[z,:-1]=obj.xt[z,None,:]*np.log1p(t)[None,:,None]
    if kind=='zero': obj.drift[z,:-1]=0; obj.drift[z,-1]=0
    return obj
def residual(q,params,kind):
    l,w,r=params; mod=model(q,kind); return mod.design(l)[0]@(np.r_[w,r] if mod.regime else w)-y[q]
for kind in ('linear','log','zero'):
    r=np.empty_like(y)
    for lab in np.unique(tr['lab_id']):
        q=tr['lab_id']==lab; par=m._fit(model(~q,kind),y[~q]); r[q]=residual(q,par,kind)
        for phase in (False,True):
            v=q&((a>0)==phase); e=r[v]; print(kind,lab,phase,len(e),'curve median/p90; final signed median; rate MAE',np.round([np.median(abs(e[:,:-1])),np.quantile(abs(e[:,:-1]),.9),np.median(e[:,-2]),np.mean(abs(e[:,-1]))],5))
    par=m._fit(model(slice(None),kind),y); print(kind,'fit',np.exp(par[0]),par[1],par[2]); print(kind,'signed medians by phase',np.round([np.median(r[(a>0)==phase],axis=0) for phase in (False,True)],4))
